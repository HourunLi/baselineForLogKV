"""Full/CSA Qwen attention with identical causal training and streaming semantics."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ._vendor.semanticlogkv.litgpt.model import CausalSelfAttention, apply_rope, apply_rope_interleave
from .core import GatedCompressor, LightningIndexer, indexer_kl
from .settings import HybridCSASettings


class TensorBuffer:
    """Append-only cache along axis 1, with bounded allocation slack."""

    def __init__(self, growth=128):
        self.data, self.length, self.growth = None, 0, growth

    def append(self, values):
        end = self.length + values.size(1)
        if self.data is None or end > self.data.size(1):
            capacity = math.ceil(end / self.growth) * self.growth
            replacement = values.new_empty(values.size(0), capacity, *values.shape[2:])
            if self.length:
                replacement[:, :self.length] = self.data[:, :self.length]
            self.data = replacement
        self.data[:, self.length:end] = values
        self.length = end
        return self.view()

    def view(self):
        return None if self.data is None else self.data[:, :self.length]


def rotate(tensor, cos, sin, config):
    # [B, T, H, D] -> LitGPT's [B, H, T, D].
    tensor = tensor.transpose(1, 2)
    rope = apply_rope_interleave if config.rope_interleave else apply_rope
    dim = config.rope_n_elem
    return torch.cat((rope(tensor[..., :dim], cos, sin), tensor[..., dim:]), dim=-1).transpose(1, 2)


class HybridAttention(nn.Module):
    def __init__(self, inner, settings, full):
        super().__init__()
        self.inner, self.settings, self.full = inner, settings, full
        self.stage = "sparse"
        c = inner.config
        if not full:
            device = inner.qkv.weight.device
            self.compressor = GatedCompressor(c.n_embd, c.n_query_groups, c.head_size, device)
            self.indexer = LightningIndexer(c.n_embd, settings, c.rope_base, device)
        self.release()
        self.streaming = False

    def release(self):
        self.key_buffer, self.value_buffer, self.index_buffer = TensorBuffer(), TensorBuffer(), TensorBuffer()
        self.local_k = self.local_v = None
        self.tail_x = self.tail_k = self.tail_v = None
        self.processed = 0
        self.index_loss = None

    def project(self, x):
        c = self.inner.config
        sizes = (c.n_head * c.head_size, c.n_query_groups * c.head_size, c.n_query_groups * c.head_size)
        q, k, v = self.inner.qkv(x).split(sizes, dim=-1)
        q = q.unflatten(-1, (c.n_head, c.head_size))
        k = k.unflatten(-1, (c.n_query_groups, c.head_size))
        v = v.unflatten(-1, (c.n_query_groups, c.head_size))
        if c.norm_qk:
            q, k = self.inner.norm_q(q), self.inner.norm_k(k)
        return q, k, v

    def forward(self, x, cos, sin, mask=None, input_pos=None, input_pos_maxp1=None):
        if mask is not None:
            raise ValueError("HybridCSA builds its own causal masks; external packed/padding masks are unsupported")
        c, s = self.inner.config, self.settings
        start = self.processed if self.streaming else 0
        end = start + x.size(1)
        positions = torch.arange(start, end, device=x.device)
        q, raw_k, raw_v = self.project(x)
        q = rotate(q, cos, sin, c)
        keys = rotate(raw_k, cos, sin, c)
        self.index_loss = x.new_zeros((), dtype=torch.float32)
        if self.full:
            if self.streaming:
                keys = self.key_buffer.append(keys)
                raw_v = self.value_buffer.append(raw_v)
            y = self.full_attention(q, keys, raw_v, positions)
        else:
            y = self.compressed_attention(x, q, raw_k, raw_v, keys, cos, sin, positions)
        if self.streaming:
            self.processed = end
        return self.inner.proj(y.flatten(-2))

    def full_attention(self, q, keys, values, positions):
        c = self.inner.config
        repetitions = c.n_head // c.n_query_groups
        # Full layers use SDPA's causal fast path for an entire training/prefill sequence.
        k = keys.transpose(1, 2).repeat_interleave(repetitions, dim=1)
        v = values.transpose(1, 2).repeat_interleave(repetitions, dim=1)
        scale = (c.attention_scores_scalar or c.head_size) ** -0.5 * self.inner.mscale ** 2
        causal = int(positions[0]) == 0 and len(positions) == keys.size(1)
        mask = None if causal else (torch.arange(keys.size(1), device=q.device)[None] <= positions[:, None])[None, None]
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k, v, attn_mask=mask,
                                         is_causal=causal, dropout_p=0.0, scale=scale)
        return y.transpose(1, 2)

    def compressed_attention(self, x, q, raw_k, raw_v, keys, cos, sin, positions):
        c, s = self.inner.config, self.settings
        start, end = int(positions[0]), int(positions[-1]) + 1
        source_start = max(0, (start // 4 - 1) * 4) if self.streaming else 0
        if self.streaming and self.tail_x is not None:
            source_x = torch.cat((self.tail_x, x), dim=1)
            source_k = torch.cat((self.tail_k, raw_k), dim=1)
            source_v = torch.cat((self.tail_v, raw_v), dim=1)
        else:
            source_x, source_k, source_v = x, raw_k, raw_v
        starts = torch.arange(start // 4 if self.streaming else 0, end // 4, device=x.device) * 4
        count = len(starts)
        if count:
            comp_k, comp_v = self.compressor(source_x, source_k, source_v, starts, source_start)
            if c.norm_qk:
                comp_k = self.inner.norm_k(comp_k)
            # RoPE anchor = block START; availability below uses block END.
            # A block can start before this prefill slice, so use the full model's tables.
            table_cos, table_sin = self.rope_tables
            comp_k = rotate(comp_k, table_cos[starts].unsqueeze(0), table_sin[starts].unsqueeze(0), c)
            index_keys = self.indexer.compress(source_x, starts, source_start).to(x.dtype) if self.stage != "compressor" else None
        else:
            comp_k = raw_k[:, :0]
            comp_v = raw_v[:, :0]
            index_keys = x.new_empty(x.size(0), 0, s.index_dim)
        if self.streaming:
            if count:
                self.key_buffer.append(comp_k)
                self.value_buffer.append(comp_v)
                if index_keys is not None:
                    self.index_buffer.append(index_keys)
            comp_k = self.key_buffer.view() if self.key_buffer.length else raw_k[:, :0]
            comp_v = self.value_buffer.view() if self.value_buffer.length else raw_v[:, :0]
            index_keys = self.index_buffer.view() if self.index_buffer.length else x.new_empty(x.size(0), 0, s.index_dim)
            local_keys = torch.cat((self.local_k, keys), dim=1) if self.local_k is not None else keys
            local_values = torch.cat((self.local_v, raw_v), dim=1) if self.local_v is not None else raw_v
            local_start = start - (0 if self.local_k is None else self.local_k.size(1))
        else:
            local_keys, local_values, local_start = keys, raw_v, 0
        outputs, losses, active = [], [], []
        for offset in range(0, x.size(1), s.query_chunk_size):
            stop = min(x.size(1), offset + s.query_chunk_size)
            query_positions = positions[offset:stop]
            # Bind the slice in the closure: checkpoint recomputation runs after the loop.
            def attend(query, query_x, ck, cv, ik, lk, lv, query_positions=query_positions):
                return self.attend(query, query_x, ck, cv, ik, lk, lv, query_positions, local_start)
            arguments = (q[:, offset:stop], x[:, offset:stop], comp_k, comp_v,
                         index_keys if index_keys is not None else x.new_empty(x.size(0), 0, s.index_dim), local_keys, local_values)
            if self.training and torch.is_grad_enabled():
                y, loss, n = checkpoint(attend, *arguments, use_reentrant=False)
            else:
                y, loss, n = attend(*arguments)
            outputs.append(y)
            losses.append(loss)
            active.append(n)
        denominator = torch.stack(active).sum().clamp_min(1)
        self.index_loss = torch.stack(losses).sum() / denominator
        if self.streaming:
            # clone is necessary: a view would keep the entire prefill allocation alive.
            self.local_k = local_keys[:, -s.window_size:].clone()
            self.local_v = local_values[:, -s.window_size:].clone()
            keep_start = max(0, (end // 4 - 1) * 4)
            offset = keep_start - source_start
            self.tail_x = source_x[:, offset:].clone()
            self.tail_k = source_k[:, offset:].clone()
            self.tail_v = source_v[:, offset:].clone()
        return torch.cat(outputs, dim=1)

    def attend(self, q, x, comp_k, comp_v, index_keys, local_k, local_v, positions, local_start):
        # .float() inputs alone do not stop autocast from lowering einsum logits.
        with torch.autocast(device_type=q.device.type, enabled=False):
            return self._attend(q, x, comp_k, comp_v, index_keys, local_k, local_v, positions, local_start)

    def _attend(self, q, x, comp_k, comp_v, index_keys, local_k, local_v, positions, local_start):
        c, s = self.inner.config, self.settings
        batch, queries = q.shape[:2]
        groups, repeats = c.n_query_groups, c.n_head // c.n_query_groups
        query = q.unflatten(2, (groups, repeats)).permute(0, 2, 3, 1, 4)
        # Each query gathers only its local window, never a Q x full-raw-context matrix.
        local_positions = positions[:, None] + torch.arange(1 - s.window_size, 1, device=q.device)[None]
        valid_local = local_positions >= max(0, local_start)
        indices = (local_positions - local_start).clamp(0, local_k.size(1) - 1)
        local_keys = local_k[:, indices.reshape(-1)].unflatten(1, (queries, s.window_size)).permute(0, 3, 1, 2, 4)
        local_values = local_v[:, indices.reshape(-1)].unflatten(1, (queries, s.window_size)).permute(0, 3, 1, 2, 4)
        scale = (c.attention_scores_scalar or c.head_size) ** -0.5 * self.inner.mscale ** 2
        local_logits = torch.einsum("bgrqd,bgqwd->bgrqw", query.float(), local_keys.float()) * scale
        local_logits = local_logits.masked_fill(~valid_local[None, None, None], float("-inf"))
        count = comp_k.size(1)
        loss = q.new_zeros((), dtype=torch.float32)
        active = q.new_zeros((), dtype=torch.float32)
        if count:
            visible = (torch.arange(count, device=q.device)[None] * 4 + 3 <= positions[:, None])
            valid = visible[None].expand(batch, -1, -1)
            index_scores = self.indexer.scores(x, positions, index_keys) if self.stage != "compressor" else None
            if self.stage == "sparse":
                # ReLU can produce exact ties. Stable ordering by block index keeps
                # selection independent of how many future (masked) blocks a slice contains.
                chosen = index_scores.masked_fill(~valid, float("-inf")).argsort(
                    dim=-1, descending=True, stable=True)[..., :min(s.index_topk, count)]
                valid = valid.gather(-1, chosen)
                index_scores = index_scores.gather(-1, chosen)
                gather_idx = chosen[:, None, :, :, None].expand(-1, groups, -1, -1, c.head_size)
                ck = comp_k.permute(0, 2, 1, 3)[:, :, None].expand(-1, -1, queries, -1, -1).gather(3, gather_idx)
                cv = comp_v.permute(0, 2, 1, 3)[:, :, None].expand(-1, -1, queries, -1, -1).gather(3, gather_idx)
                global_logits = torch.einsum("bgrqd,bgqnd->bgrqn", query.float(), ck.float()) * scale
            else:
                ck, cv = comp_k.permute(0, 2, 1, 3), comp_v.permute(0, 2, 1, 3)
                global_logits = torch.einsum("bgrqd,bgnd->bgrqn", query.float(), ck.float()) * scale
            global_logits = global_logits.masked_fill(~valid[:, None, None], float("-inf"))
            probabilities = torch.cat((local_logits, global_logits), dim=-1).softmax(-1)
            local_probability, global_probability = probabilities.split((s.window_size, global_logits.size(-1)), dim=-1)
            if self.training and index_scores is not None:
                loss, active = indexer_kl(index_scores, global_probability, valid)
            if self.stage == "sparse":
                y = torch.einsum("bgrqn,bgqnd->bgrqd", global_probability.to(cv.dtype), cv)
            else:
                y = torch.einsum("bgrqn,bgnd->bgrqd", global_probability.to(cv.dtype), cv)
        else:
            local_probability = local_logits.softmax(-1)
            y = torch.zeros_like(query)
        y = y + torch.einsum("bgrqw,bgqwd->bgrqd", local_probability.to(local_values.dtype), local_values)
        return y.permute(0, 3, 1, 2, 4).flatten(2, 3).to(q.dtype), loss, active


class HybridCSAModel(nn.Module):
    def __init__(self, gpt, settings: HybridCSASettings):
        super().__init__()
        c = gpt.config
        unsupported = (c.latent_attention is not None or c.sliding_window_size is not None or c.rope_indices is not None
                       or c.attention_logit_softcapping is not None or c.final_logit_softcapping is not None
                       or (c.norm_qk and c.norm_qk_type != "default"))
        if unsupported:
            raise ValueError("HybridCSA supports standard Qwen-style GQA with ordinary RoPE and QK normalization")
        self.gpt, self.settings = gpt, settings
        gpt.requires_grad_(False)
        full_layers = settings.full_layers(c.n_layer)
        for i, block in enumerate(gpt.transformer.h):
            if type(block.attn) is not CausalSelfAttention:
                raise TypeError("HybridCSA expects an unwrapped LitGPT Qwen model")
            block.attn = HybridAttention(block.attn, settings, i in full_layers)
        self.processed = None
        self.set_stage("sparse")

    @property
    def max_seq_length(self):
        return self.gpt.max_seq_length

    @property
    def attentions(self):
        return [block.attn for block in self.gpt.transformer.h]

    def adapter_named_parameters(self):
        for name, parameter in self.named_parameters():
            if ".compressor." in name or ".indexer." in name:
                yield name, parameter

    def adapter_state_dict(self):
        return {name: value.detach().cpu().clone() for name, value in self.adapter_named_parameters()}

    def load_adapter_state_dict(self, state):
        expected = dict(self.adapter_named_parameters())
        if expected.keys() != state.keys():
            raise ValueError(f"Adapter keys mismatch: missing={sorted(expected.keys() - state.keys())[:5]}, "
                             f"unexpected={sorted(state.keys() - expected.keys())[:5]}")
        for name, parameter in expected.items():
            if parameter.shape != state[name].shape:
                raise ValueError(f"Adapter shape mismatch: {name}")
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(state[name])

    def set_stage(self, stage):
        if stage not in ("compressor", "indexer", "sparse"):
            raise ValueError("stage must be compressor, indexer or sparse")
        if self.processed is not None:
            raise RuntimeError("Release the request cache before changing training stage")
        self.stage = stage
        for attn in self.attentions:
            attn.stage = stage
        for name, parameter in self.adapter_named_parameters():
            parameter.requires_grad_(stage == "sparse" or (stage == "indexer") == (".indexer." in name))

    def reset_cache(self):
        for attention in self.attentions:
            attention.release()
            attention.streaming = True
        self.processed = 0

    def release_cache(self):
        for attention in self.attentions:
            attention.release()
            attention.streaming = False
        self.processed = None

    def _features(self, idx, positions, activation_checkpointing=False):
        gpt = self.gpt
        cos, sin = gpt.cos[positions].unsqueeze(0), gpt.sin[positions].unsqueeze(0)
        x = gpt.transformer.wte(idx)
        if gpt.config.scale_embeddings:
            x = x * (gpt.config.n_embd ** 0.5)
        aux = x.new_zeros((), dtype=torch.float32)
        for block in gpt.transformer.h:
            block.attn.rope_tables = (gpt.cos, gpt.sin)
            def run(hidden, block=block):
                result = block(hidden, cos, sin)
                return result, block.attn.index_loss
            if activation_checkpointing and self.training and torch.is_grad_enabled():
                x, loss = checkpoint(run, x, use_reentrant=False)
            else:
                x, loss = run(x)
            aux = aux + loss
        compressed_layers = sum(not attn.full for attn in self.attentions)
        return gpt.transformer.ln_f(x), aux / max(1, compressed_layers)

    def forward_train(self, idx):
        self._check_training_input(idx)
        hidden, _ = self._features(idx, torch.arange(idx.size(1), device=idx.device))
        return self.gpt.lm_head(hidden)

    def _check_training_input(self, idx):
        if self.processed is not None:
            raise RuntimeError("Release evaluation caches before training")
        if idx.ndim != 2 or not 0 < idx.size(1) <= self.max_seq_length:
            raise ValueError("Training input must be nonempty and within the model window")

    def training_loss(self, input_ids, targets, loss_chunk_size=128, activation_checkpointing=True):
        self._check_training_input(input_ids)
        if targets.shape != input_ids.shape or loss_chunk_size < 1:
            raise ValueError("Targets must match input_ids; loss_chunk_size must be positive")
        hidden, auxiliary = self._features(input_ids, torch.arange(input_ids.size(1), device=input_ids.device),
                                          activation_checkpointing)
        flat_hidden, flat_targets = hidden.flatten(0, 1), targets.flatten()
        total = hidden.new_zeros((), dtype=torch.float32)
        def projected_loss(features, labels):
            return F.cross_entropy(self.gpt.lm_head(features).float(), labels, reduction="sum")
        for start in range(0, flat_targets.numel(), loss_chunk_size):
            features = flat_hidden[start:start + loss_chunk_size]
            labels = flat_targets[start:start + loss_chunk_size]
            total = total + checkpoint(projected_loss, features, labels, use_reentrant=False)
        return total / flat_targets.numel(), auxiliary

    def forward(self, idx, input_pos=None, input_pos_maxp1=None, lm_head_start=None):
        if self.processed is None:
            raise RuntimeError("Initialize the HybridCSA request cache before forwarding tokens")
        if idx.ndim != 2 or idx.size(0) != 1 or idx.size(1) == 0:
            raise ValueError("HybridCSA evaluation requires nonempty inputs with batch size 1")
        end = self.processed + idx.size(1)
        expected = torch.arange(self.processed, end, device=idx.device)
        if input_pos is None or not torch.equal(input_pos, expected):
            raise ValueError("HybridCSA needs contiguous append-only input positions")
        if end > self.max_seq_length:
            raise ValueError("Request exceeds the model window")
        first_logit = 0 if lm_head_start is None else lm_head_start
        if not 0 <= first_logit < idx.size(1):
            raise ValueError("lm_head_start must lie inside the forwarded tokens")
        outputs = []
        for start in range(0, idx.size(1), self.settings.prefill_chunk_size):
            stop = min(idx.size(1), start + self.settings.prefill_chunk_size)
            hidden, _ = self._features(idx[:, start:stop], expected[start:stop])
            if stop > first_logit:
                outputs.append(self.gpt.lm_head(hidden[:, max(0, first_logit - start):]))
            self.processed += stop - start
        return torch.cat(outputs, dim=1)

    def kv_compression_snapshot(self, processed_tokens):
        if self.processed != processed_tokens:
            raise ValueError("HybridCSA cache and forwarded token counts differ")
        return self.theoretical_budget(processed_tokens)

    def theoretical_budget(self, processed_tokens):
        from .budget import theoretical_budget
        c = self.gpt.config
        return theoretical_budget(processed_tokens, self.settings, n_layer=c.n_layer,
                                  n_query_groups=c.n_query_groups, head_size=c.head_size, n_embd=c.n_embd)
