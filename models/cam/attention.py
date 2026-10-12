"""Inference-only online CaM attention for Hugging Face Qwen3's GQA layers."""

from types import MethodType

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, rotate_half

from .cache import CaMCache


def install_cam_attention(hf_model, start_size, recent_size, merge_size=16, seed=0):
    """Bind CaM to Qwen3 attention, preserving pretrained projections and norms.

    Call the model with ``use_cache=False, attention_mask={"full_attention": None}``
    and reset every returned cache between independent requests. Cache-relative
    RoPE rotates the raw normalized K again after every eviction.
    """
    backbone = hf_model.model
    if any(not isinstance(layer.self_attn, Qwen3Attention) for layer in backbone.layers):
        raise TypeError("CaM attention currently supports Hugging Face Qwen3Attention only")
    if any(layer.self_attn.sliding_window is not None for layer in backbone.layers):
        raise ValueError("CaM requires Qwen3 full-attention layers")
    rotary = backbone.rotary_emb  # Closure reference avoids registering duplicate modules.

    @torch.no_grad()
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        if self.training:
            raise RuntimeError("CaM attention is inference-only; call model.eval()")
        if attention_mask is not None:
            raise ValueError("CaM handles unpadded causal requests; pass attention_mask={'full_attention': None}")
        if past_key_values is not None or kwargs.get("past_key_value") is not None or kwargs.get("use_cache", False):
            raise ValueError("CaM owns its KV cache; pass use_cache=False and no external cache")
        cache = self.cam_cache
        batch_size, token_count, _ = hidden_states.shape
        hidden_shape = (batch_size, token_count, -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        keys = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        kv_heads = keys.shape[1]
        groups = self.num_key_value_groups
        # Static Qwen3 RoPE can be shared by all tokens in this projection chunk.
        available = 0 if cache.keys is None else cache.keys.shape[2]
        rope_length = min(available + token_count, cache.cache_size + 1)
        positions = torch.arange(rope_length, device=hidden_states.device).unsqueeze(0)
        cos, sin = rotary(hidden_states, positions)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        outputs = []
        for index in range(token_count):
            cache.append(keys[:, :, index:index + 1], values[:, :, index:index + 1])
            length = cache.keys.shape[2]
            current_cos, current_sin = cos[:, :, :length], sin[:, :, :length]
            rotated_keys = cache.keys * current_cos + rotate_half(cache.keys) * current_sin
            current_query = query[:, :, index:index + 1]
            current_query = (current_query * current_cos[:, :, -1:]
                             + rotate_half(current_query) * current_sin[:, :, -1:])
            grouped_query = current_query.reshape(batch_size, kv_heads, groups, self.head_dim)
            weights = torch.matmul(grouped_query, rotated_keys.transpose(-1, -2)) * self.scaling
            probabilities = weights.float().softmax(dim=-1)
            output = torch.matmul(probabilities.to(cache.values.dtype), cache.values)
            outputs.append(output.reshape(batch_size, 1, -1))
            # Merge after producing the output: current attention sees the victim.
            cache.update(probabilities.mean(dim=2))
        return self.o_proj(torch.cat(outputs, dim=1)), None

    caches = []
    for layer_index, layer in enumerate(backbone.layers):
        cache = CaMCache(start_size, recent_size, merge_size, seed + layer_index)
        layer.self_attn.cam_cache = cache
        layer.self_attn.forward = MethodType(forward, layer.self_attn)
        caches.append(cache)
    return caches
