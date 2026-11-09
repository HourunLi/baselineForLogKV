"""Inference-only online CaM attention for Hugging Face Qwen3's GQA layers."""

from types import MethodType

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, rotate_half

from .cache import CaMCache

QUERY_BLOCK = 256  # queries per attention block: bounds the float32 logits of a long prompt chunk


def install_cam_attention(hf_model, start_size, recent_size, merge_size=16, seed=0, merge_interval=1):
    """Bind CaM to Qwen3 attention, preserving pretrained projections and norms.

    Call the model with ``use_cache=False`` and reset every returned cache between independent requests.
    A forward's tokens are taken ``merge_interval`` at a time: each step appends its tokens, attends causally
    over the cache, adds the attention every entry received to its score and evicts back to
    ``start_size + recent_size`` (``CaMCache.update``). ``merge_interval=1`` is the official online path
    exactly, whatever the projection chunk; a larger interval (the 64K / 128K stages: one prompt chunk) evicts
    once per step, so queries inside a step see the cache as it stood before it. Cache-relative RoPE rotates
    the raw normalized K by its slot on every step, so positions never exceed the cache plus one step. A mask
    Transformers builds for the forward is not used: requests are single and unpadded, CaM is causal itself.
    """
    if merge_interval < 1:
        raise ValueError("merge_interval must be positive")
    backbone = hf_model.model
    if any(not isinstance(layer.self_attn, Qwen3Attention) for layer in backbone.layers):
        raise TypeError("CaM attention currently supports Hugging Face Qwen3Attention only")
    if any(getattr(layer.self_attn, "sliding_window", None) is not None for layer in backbone.layers):
        raise ValueError("CaM requires Qwen3 full-attention layers")
    rotary = backbone.rotary_emb  # Closure reference avoids registering duplicate modules.

    @torch.no_grad()
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        if self.training:
            raise RuntimeError("CaM attention is inference-only; call model.eval()")
        if past_key_values is not None or kwargs.get("past_key_value") is not None or kwargs.get("use_cache", False):
            raise ValueError("CaM owns its KV cache; pass use_cache=False and no external cache")
        batch_size, token_count, _ = hidden_states.shape
        hidden_shape = (batch_size, token_count, -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        keys = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        outputs = [step(self, hidden_states, query[:, :, begin:begin + merge_interval],
                        keys[:, :, begin:begin + merge_interval], values[:, :, begin:begin + merge_interval])
                   for begin in range(0, token_count, merge_interval)]
        return self.o_proj(torch.cat(outputs, dim=1)), None

    def step(self, hidden_states, query, keys, values):
        """Append, attend and evict for one step of ``token_count`` tokens; returns (B, tokens, heads x dim)."""
        cache = self.cam_cache
        batch_size, _, token_count, _ = query.shape
        kv_heads = keys.shape[1]
        groups = self.num_key_value_groups
        cache.append(keys, values)
        length = cache.keys.shape[2]
        positions = torch.arange(length, device=hidden_states.device).unsqueeze(0)
        cos, sin = rotary(hidden_states, positions)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        rotated_keys = cache.keys * cos + rotate_half(cache.keys) * sin
        query_cos, query_sin = cos[:, :, length - token_count:], sin[:, :, length - token_count:]
        query = query * query_cos + rotate_half(query) * query_sin
        grouped_query = query.reshape(batch_size, kv_heads, groups, token_count, self.head_dim)
        keys_t = rotated_keys.transpose(-1, -2).unsqueeze(2)
        outputs = []
        received = torch.zeros(batch_size, kv_heads, length, dtype=torch.float32, device=hidden_states.device)
        for begin in range(0, token_count, QUERY_BLOCK):
            end = min(begin + QUERY_BLOCK, token_count)
            weights = torch.matmul(grouped_query[:, :, :, begin:end], keys_t) * self.scaling
            if token_count > 1:  # query i sits at cache slot length - token_count + i
                hidden = torch.ones(end - begin, length, dtype=torch.bool, device=weights.device).tril(
                    diagonal=length - token_count + begin).logical_not()
                weights = weights.masked_fill(hidden, float("-inf"))
            probabilities = weights.float().softmax(dim=-1)
            outputs.append(torch.matmul(probabilities.to(cache.values.dtype), cache.values.unsqueeze(2)))
            # Shared KV heads: mean over their query heads, summed over this forward's queries.
            received += probabilities.mean(dim=2).sum(dim=2)
        output = torch.cat(outputs, dim=3).permute(0, 3, 1, 2, 4).reshape(batch_size, token_count, -1)
        # Merge after producing the outputs: this step's attention sees the victims.
        cache.update(received)
        return output

    caches = []
    for layer_index, layer in enumerate(backbone.layers):
        cache = CaMCache(start_size, recent_size, merge_size, seed + layer_index)
        layer.self_attn.cam_cache = cache
        layer.self_attn.forward = MethodType(forward, layer.self_attn)
        caches.append(cache)
    return caches
