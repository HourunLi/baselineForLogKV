"""Inference-only online CaM attention for Hugging Face Qwen3's GQA layers."""

from types import MethodType

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, rotate_half

from .cache import CaMCache
from .grouped import attend_and_merge

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
        token_count = query.shape[2]
        cache.append(keys, values)
        length = cache.keys.shape[2]
        positions = torch.arange(length, device=hidden_states.device).unsqueeze(0)
        cos, sin = rotary(hidden_states, positions)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        rotated_keys = cache.keys * cos + rotate_half(cache.keys) * sin
        query_cos, query_sin = cos[:, :, length - token_count:], sin[:, :, length - token_count:]
        query = query * query_cos + rotate_half(query) * query_sin
        return attend_and_merge(query, rotated_keys, cache, self.scaling)

    caches = []
    for layer_index, layer in enumerate(backbone.layers):
        cache = CaMCache(start_size, recent_size, merge_size, seed + layer_index)
        layer.self_attn.cam_cache = cache
        layer.self_attn.forward = MethodType(forward, layer.self_attn)
        caches.append(cache)
    return caches
