"""ChunkKV cache on the unmodified LitGPT model used by SemanticLogKV evaluation.

Each layer's ``CausalSelfAttention`` is wrapped, keeping its loaded weights,
projection, RoPE and SDPA call. The prompt is prefilled with ordinary causal
attention; its K/V are then compressed once with ChunkKV and later tokens are
appended uncompressed, as in the paper's prefill-then-decode setting.

The first forward of a request may include tokens after the prompt. LitGPT
callers mark where logits start with ``lm_head_start``: generation prefill
passes the last prompt token, loglikelihood scoring passes the last context
token. Tokens up to that position are the compressed prompt; the remaining
continuation is forwarded against the compressed cache. When the prompt fits
the budget nothing is compressed and the whole call is one dense forward.
"""

import torch
from torch import nn

from litgpt.model import CausalSelfAttention, apply_rope, apply_rope_interleave

from .core import ChunkKVSettings, keep_indices, observation_scores


def check_supported(config) -> None:
    unsupported = [name for name, used in (
        ("latent_attention", config.latent_attention is not None),
        ("sliding_window_size", config.sliding_window_size is not None),
        ("rope_indices", config.rope_indices is not None),
        ("attention_logit_softcapping", config.attention_logit_softcapping is not None),
        ("final_logit_softcapping", config.final_logit_softcapping is not None),
        ("norm_qk_type", config.norm_qk and config.norm_qk_type != "default"),
    ) if used]
    if unsupported:
        raise ValueError(f"ChunkKV wrapper does not support: {', '.join(unsupported)}")


class _Request:
    """Per-forward state shared by all layers: compression switch and reused indices."""

    def __init__(self, settings: ChunkKVSettings):
        self.settings = settings
        self.compress = False
        self.indices = None
        self.chunk_tokens = 0

    def begin(self, compress: bool) -> None:
        self.compress, self.indices, self.chunk_tokens = compress, None, 0


class ChunkKVAttention(nn.Module):
    """Wraps a loaded attention layer; stores this layer's kept K/V for one request."""

    growth = 256  # decode capacity added per reallocation

    def __init__(self, attn: CausalSelfAttention, request: _Request):
        super().__init__()
        self.inner = attn
        self.request = request
        self.layer_idx = attn.block_idx
        self.release()

    def release(self) -> None:
        self.keys = self.values = None
        self.length = self.chunk_tokens = 0

    def project(self, x, cos, sin):
        """Same Q/K/V, QK-norm and RoPE as ``CausalSelfAttention.forward``."""
        attn, config = self.inner, self.inner.config
        batch, length, _ = x.shape
        head_size, n_head, n_groups = config.head_size, config.n_head, config.n_query_groups
        q, k, v = attn.qkv(x).split((n_head * head_size, n_groups * head_size, n_groups * head_size), dim=-1)
        q = q.view(batch, length, n_head, head_size).transpose(1, 2)
        k = k.view(batch, length, n_groups, head_size).transpose(1, 2)
        v = v.view(batch, length, n_groups, head_size).transpose(1, 2)
        if config.norm_qk:
            q, k = attn.norm_q(q), attn.norm_k(k)
        rope = apply_rope_interleave if config.rope_interleave else apply_rope
        n_elem = config.rope_n_elem
        q = torch.cat((rope(q[..., :n_elem], cos, sin), q[..., n_elem:]), dim=-1)
        k = torch.cat((rope(k[..., :n_elem], cos, sin), k[..., n_elem:]), dim=-1)
        return q, k, v

    def attend(self, q, k, v, mask):
        config = self.inner.config
        if config.n_query_groups != config.n_head:
            repeats = config.n_head // config.n_query_groups
            k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        y = self.inner.scaled_dot_product_attention(q, k, v, mask)
        return self.inner.proj(y.reshape(q.size(0), q.size(2), -1))

    def forward(self, x, cos, sin, mask=None, input_pos=None, input_pos_maxp1=None):
        q, k, v = self.project(x, cos, sin)
        if self.keys is None:
            # Prompt prefill: the prompt itself always sees its full causal context.
            y = self.attend(q, k, v, None)
            if self.request.compress:
                k, v = self.compress(q, k, v)
            self.store(k, v)
            return y
        keys, values = self.append(k, v)
        cached = keys.size(2) - k.size(2)
        visible = torch.ones(k.size(2), keys.size(2), dtype=torch.bool, device=x.device).tril(diagonal=cached)
        return self.attend(q, keys, values, visible[None, None])

    def compress(self, q, k, v):
        request, settings = self.request, self.request.settings
        length = k.size(2)
        if not settings.compresses(length):
            return k, v
        if request.indices is None or settings.scores_layer(self.layer_idx):
            config = self.inner.config
            scale = (config.attention_scores_scalar or config.head_size) ** -0.5 * self.inner.mscale ** 2
            scores = observation_scores(q, k, settings.window_size, scale)
            request.indices, request.chunk_tokens = keep_indices(scores[0], length, settings)
        self.chunk_tokens = request.chunk_tokens
        return k.index_select(2, request.indices), v.index_select(2, request.indices)

    def store(self, k, v) -> None:
        length = k.size(2)
        shape = (*k.shape[:2], length + self.growth, k.size(3))
        self.keys = k.new_empty(shape)
        self.values = v.new_empty(shape)
        self.keys[:, :, :length] = k
        self.values[:, :, :length] = v
        self.length = length

    def append(self, k, v):
        end = self.length + k.size(2)
        if end > self.keys.size(2):
            shape = (*self.keys.shape[:2], end + self.growth, self.keys.size(3))
            keys, values = self.keys.new_empty(shape), self.values.new_empty(shape)
            keys[:, :, :self.length] = self.keys[:, :, :self.length]
            values[:, :, :self.length] = self.values[:, :, :self.length]
            self.keys, self.values = keys, values
        self.keys[:, :, self.length:end] = k
        self.values[:, :, self.length:end] = v
        self.length = end
        return self.keys[:, :, :end], self.values[:, :, :end]


class ChunkKVModel(nn.Module):
    """LitGPT ``GPT`` with a ChunkKV request cache and the same forward interface."""

    def __init__(self, gpt, settings: ChunkKVSettings):
        super().__init__()
        check_supported(gpt.config)
        self.gpt = gpt
        self.settings = settings
        self.request = _Request(settings)
        for block in gpt.transformer.h:
            if type(block.attn) is not CausalSelfAttention:
                raise TypeError("ChunkKV expects LitGPT CausalSelfAttention layers")
            block.attn = ChunkKVAttention(block.attn, self.request)
        self.processed = None

    @property
    def max_seq_length(self) -> int:
        return self.gpt.max_seq_length

    @property
    def attentions(self) -> list[ChunkKVAttention]:
        return [block.attn for block in self.gpt.transformer.h]

    def reset_cache(self) -> None:
        for attn in self.attentions:
            attn.release()
        self.processed = 0

    def release_cache(self) -> None:
        for attn in self.attentions:
            attn.release()
        self.processed = None

    def forward(self, idx, input_pos=None, input_pos_maxp1=None, lm_head_start=None):
        if self.processed is None:
            raise RuntimeError("Initialize the ChunkKV request cache before forwarding tokens")
        if idx.size(0) != 1:
            raise ValueError("ChunkKV evaluation uses batch size 1")
        length = idx.size(1)
        expected = torch.arange(self.processed, self.processed + length, device=idx.device)
        if input_pos is None or input_pos.shape != expected.shape or not torch.equal(input_pos, expected):
            raise ValueError("ChunkKV needs contiguous, append-only input positions")
        if self.processed + length > self.max_seq_length:
            raise ValueError("Request exceeds the model window")
        if self.processed:
            return self._run(idx, input_pos, lm_head_start, compress=False)
        prompt = length if lm_head_start is None else lm_head_start + 1
        if not 0 < prompt <= length:
            raise ValueError("lm_head_start must lie inside the first forward")
        if prompt < length and not self.settings.compresses(prompt):
            return self._run(idx, input_pos, lm_head_start, compress=False)
        logits = self._run(idx[:, :prompt], input_pos[:prompt], lm_head_start, compress=True)
        if prompt == length:
            return logits
        rest = self._run(idx[:, prompt:], input_pos[prompt:], 0, compress=False)
        return torch.cat((logits, rest), dim=1)

    def _run(self, idx, input_pos, lm_head_start, compress):
        gpt = self.gpt
        self.request.begin(compress)
        cos = gpt.cos.index_select(0, input_pos).unsqueeze(0)
        sin = gpt.sin.index_select(0, input_pos).unsqueeze(0)
        x = gpt.transformer.wte(idx)
        if gpt.config.scale_embeddings:
            x = x * torch.tensor(gpt.config.n_embd**0.5, dtype=x.dtype)
        for block in gpt.transformer.h:
            x = block(x, cos, sin, None, input_pos, None)
        self.processed += idx.size(1)
        if lm_head_start is not None:
            x = x[:, lm_head_start:]
        return gpt.lm_head(gpt.transformer.ln_f(x))

    def kv_compression_snapshot(self, processed_tokens: int) -> dict:
        """Counts in the shared metric's units; chunk tokens are ``exact``, the rest ``recent``."""
        if self.processed != processed_tokens:
            raise ValueError("ChunkKV cache and forwarded token counts differ")
        counts = dict(processed_tokens=processed_tokens, layer_group_count=0, recent_slots=0,
                      exact_slots=0, hierarchical_slots=0, retained_slots=0, dense_slots=0)
        groups = self.gpt.config.n_query_groups
        for attn in self.attentions:
            if not 0 <= attn.chunk_tokens <= attn.length <= processed_tokens:
                raise ValueError("Invalid ChunkKV retained KV count")
            counts["layer_group_count"] += groups
            counts["recent_slots"] += (attn.length - attn.chunk_tokens) * groups
            counts["exact_slots"] += attn.chunk_tokens * groups
            counts["retained_slots"] += attn.length * groups
            counts["dense_slots"] += processed_tokens * groups
        return counts
