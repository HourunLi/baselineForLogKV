"""Qwen3 with causal segment-local SDPA and an additive Infini memory per KV head."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from lla_bench.litgpt.chunked_loss import chunked_linear_cross_entropy


class SegmentCache:
    def __init__(self, cfg, batch, weight):
        shape = (batch, cfg.num_key_value_heads)
        d = cfg.head_dim
        self.k = weight.new_empty((*shape, cfg.infini_segment_length, d))
        self.v = torch.empty_like(self.k)
        self.m = weight.new_zeros((*shape, d, d), dtype=torch.float32)
        self.z = weight.new_zeros((*shape, d, 1), dtype=torch.float32)
        self.count = 0

    def reset(self):
        self.count = 0
        self.m.zero_()
        self.z.zero_()


class SegmentLayer(nn.Module):
    def __init__(self, original, cfg):
        super().__init__()
        self.self_attn = original.self_attn
        self.input_layernorm = original.input_layernorm
        self.post_attention_layernorm = original.post_attention_layernorm
        self.mlp = original.mlp
        self.segment = cfg.infini_segment_length
        self.groups = cfg.num_key_value_heads
        self.heads = cfg.num_attention_heads
        self.dim = cfg.head_dim
        # Learn a zero-origin offset so small BF16 updates are representable at initialization.
        self.memory_gate = nn.Parameter(original.self_attn.q_proj.weight.new_zeros(self.heads))
        self.cache = None

    def piece(self, x, cos, sin, m, z, offset, mlp_chunk):
        batch, length, _ = x.shape
        hidden = self.input_layernorm(x)
        attn = self.self_attn
        def project(proj, heads):
            return proj(hidden).reshape(batch, length, heads, self.dim).transpose(1, 2)
        q = attn.q_norm(project(attn.q_proj, self.heads))
        k = attn.k_norm(project(attn.k_proj, self.groups))
        v = project(attn.v_proj, self.groups)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        end = offset + length
        if self.cache is not None:
            self.cache.k[:, :, offset:end].copy_(k)
            self.cache.v[:, :, offset:end].copy_(v)
            k, v = self.cache.k[:, :, :end], self.cache.v[:, :, :end]
        mask = None
        if offset and length > 1:
            pos = torch.arange(end, device=x.device)
            mask = pos[None, :] <= pos[offset:, None]
        local = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=offset == 0 and length > 1,
            enable_gqa=self.heads != self.groups)
        with torch.autocast(device_type=x.device.type, enabled=False):
            # Flatten the four query heads in each group into the sequence dimension.
            query = (F.elu(q.float()) + 1).reshape(batch, self.groups, -1, self.dim)
            memory = ((query @ m) / (query @ z).clamp_min(1e-8)).reshape_as(q)
            if end == self.segment:
                key = F.elu(k.float()) + 1
                m = m + key.transpose(-1, -2) @ v.float()
                z = z + key.sum(dim=-2).unsqueeze(-1)
        gate = (self.memory_gate.float() - 2.0).sigmoid()[None, :, None, None]
        out = torch.lerp(local.float(), memory, gate).to(x.dtype)
        x = x + attn.o_proj(out.transpose(1, 2).reshape(batch, length, -1))
        pieces = [part + self.mlp(self.post_attention_layernorm(part))
                  for part in x.split(mlp_chunk or length, dim=1)]
        return torch.cat(pieces, dim=1), m, z

    def forward(self, x, cos, sin, mlp_chunk):
        cache = self.cache
        if cache is None:
            m = x.new_zeros((x.size(0), self.groups, self.dim, self.dim), dtype=torch.float32)
            z = x.new_zeros((x.size(0), self.groups, self.dim, 1), dtype=torch.float32)
            offset = 0
        else:
            m, z = cache.m, cache.z
            offset = cache.count % self.segment
        parts = []
        start = 0
        while start < x.size(1):
            end = min(x.size(1), start + self.segment - offset)
            args = (x[:, start:end], cos[:, start:end], sin[:, start:end], m, z, offset, mlp_chunk)
            if self.training:
                y, m, z = checkpoint(self.piece, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                y, m, z = self.piece(*args)
            parts.append(y)
            offset = (offset + end - start) % self.segment
            start = end
        if cache is not None:
            if m is not cache.m:
                cache.m.copy_(m)
                cache.z.copy_(z)
            cache.count += x.size(1)
        return torch.cat(parts, dim=1)


class InfiniLM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        if config.infini_segment_length < 1 or config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("invalid Infini segment length or GQA geometry")
        self.model.layers = nn.ModuleList(SegmentLayer(layer, config) for layer in self.model.layers)
        self.max_seq_length = config.max_position_embeddings
        self.mlp_chunk = 2048

    @classmethod
    def load_local(cls, folder, *, base=False, segment=2048, window=None, yarn=None, dtype=torch.bfloat16):
        cfg = Qwen3Config.from_pretrained(folder, local_files_only=True)
        raw = json.loads((Path(folder) / "config.json").read_text())
        if base == hasattr(cfg, "infini_segment_length") or raw.get("model_type") != "qwen3":
            raise ValueError(f"unexpected checkpoint type for {'base' if base else 'Infini'}: {folder}")
        if base:
            cfg.infini_segment_length = segment
        if yarn:
            if yarn["original_max_seq_len"] != cfg.max_position_embeddings:
                raise ValueError("YaRN original context must match the Qwen base")
            cfg.rope_scaling = {"rope_type": "yarn", "factor": float(yarn["factor"]),
                                "beta_fast": yarn["beta_fast"], "beta_slow": yarn["beta_slow"]}
        model, info = cls.from_pretrained(Path(folder), config=cfg, output_loading_info=True,
                                          torch_dtype=dtype, local_files_only=True, attn_implementation="sdpa")
        expected = {f"model.layers.{i}.memory_gate" for i in range(cfg.num_hidden_layers)} if base else set()
        if set(info["missing_keys"]) != expected or any(info[k] for k in ("unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"incomplete model transfer: {info}")
        if base:
            for layer in model.model.layers:
                nn.init.zeros_(layer.memory_gate)
        model.max_seq_length = window or cfg.max_position_embeddings
        return model

    def set_cache(self, batch=1):
        if self.training:
            raise ValueError("mutable caches are for inference only")
        for layer in self.model.layers:
            layer.cache = SegmentCache(self.config, batch, self.lm_head.weight)

    def reset_cache(self):
        for layer in self.model.layers:
            layer.cache.reset()

    def forward(self, idx, input_pos=None, input_pos_maxp1=None, lm_head_start=None, targets=None):
        if input_pos is not None:
            cache = self.model.layers[0].cache
            if cache is None or self.training:
                raise ValueError("streaming requires an evaluation cache")
            start = cache.count
            end = start + idx.size(1)
            if end > self.max_seq_length or not torch.equal(input_pos, torch.arange(start, end, device=idx.device)):
                raise ValueError("expected contiguous positions inside the configured window")
        else:
            if idx.size(1) > self.max_seq_length or getattr(self.model.layers[0], "cache", None) is not None:
                raise ValueError("invalid uncached input")
            input_pos = torch.arange(idx.size(1), device=idx.device)
        x = self.model.embed_tokens(idx)
        cos, sin = self.model.rotary_emb(x, input_pos.unsqueeze(0))
        for layer in self.model.layers:
            x = layer(x, cos, sin, self.mlp_chunk)
        if targets is not None:
            return chunked_linear_cross_entropy(self.model.norm(x), self.lm_head.weight, targets, chunk_size=1024)
        return self.lm_head(self.model.norm(x[:, lm_head_start or 0:]))
