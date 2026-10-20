"""Learned overlapping pooling and an independently trained lightweight indexer.

Algorithm reference: DeepSeek-V4, section 2.3.1. This Qwen adaptation retains
separate GQA K/V. Its payload transforms start as identity maps of the original
Qwen K/V; it does not import DeepSeek's shared-KV MQA or quantized kernels.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def overlap_windows(tensor, starts, source_start=0):
    """Gather previous/current groups of four; missing history has an explicit mask."""
    positions = starts[:, None] + torch.arange(-4, 4, device=tensor.device)[None]
    valid = positions >= 0
    indices = positions - source_start
    if valid.any() and (indices[valid].min() < 0 or indices[valid].max() >= tensor.size(1)):
        raise ValueError("Incomplete compressor history")
    indices = indices.clamp_min(0)
    gathered = tensor[:, indices.reshape(-1)].unflatten(1, (len(starts), 8))
    return gathered, valid


def select_roles(windows):
    # [B, blocks, 8, 2, ...]: previous block takes role 0, current takes role 1.
    return torch.cat((windows[:, :, :4, 0], windows[:, :, 4:, 1]), dim=2)


def gated_pool(payload, scores, valid):
    mask = valid.reshape(1, *valid.shape, *((1,) * (scores.ndim - 3)))
    weights = scores.float().masked_fill(~mask, float("-inf")).softmax(dim=2)
    return (payload.float() * weights).sum(dim=2)


class GatedCompressor(nn.Module):
    """Two role-specific, per-head payload maps and channel-wise learned gates."""

    def __init__(self, hidden_size, groups, head_dim, device=None):
        super().__init__()
        self.groups, self.head_dim = groups, head_dim
        width = 2 * groups * head_dim
        self.key_gate = nn.Linear(hidden_size, width, bias=False, device=device, dtype=torch.float32)
        self.value_gate = nn.Linear(hidden_size, width, bias=False, device=device, dtype=torch.float32)
        self.key_bias = nn.Parameter(torch.zeros(4, 2, groups, head_dim, device=device, dtype=torch.float32))
        self.value_bias = nn.Parameter(torch.zeros_like(self.key_bias))
        eye = torch.eye(head_dim, device=device, dtype=torch.float32).expand(2, groups, head_dim, head_dim).clone()
        self.key_payload = nn.Parameter(eye)
        self.value_payload = nn.Parameter(eye.clone())
        # Uniform pooling is only an initialization, never a reported trained baseline.
        nn.init.zeros_(self.key_gate.weight)
        nn.init.zeros_(self.value_gate.weight)

    def forward(self, x, keys, values, starts, source_start=0):
        # keys/values use [B, T, G, D], before any rotary embedding.
        outputs = []
        offsets = torch.arange(source_start, source_start + x.size(1), device=x.device) % 4
        with torch.autocast(device_type=x.device.type, enabled=False):
            for source, transform, gate, bias in (
                (keys, self.key_payload, self.key_gate, self.key_bias),
                (values, self.value_payload, self.value_gate, self.value_bias),
            ):
                payload = torch.einsum("btgd,rgde->btrge", source.float(), transform.float())
                scores = F.linear(x.float(), gate.weight.float()).unflatten(-1, (2, self.groups, self.head_dim))
                scores = scores + bias.float()[offsets]
                payload, valid = overlap_windows(payload, starts, source_start)
                scores, _ = overlap_windows(scores, starts, source_start)
                outputs.append(gated_pool(select_roles(payload), select_roles(scores), valid).to(source.dtype))
        return tuple(outputs)


def index_rope(tensor, positions, base):
    """Qwen's half-split RoPE for the indexer's own head dimension."""
    dim = tensor.size(-1)
    frequencies = base ** (-torch.arange(0, dim, 2, device=tensor.device, dtype=torch.float32) / dim)
    angles = positions.float()[:, None] * frequencies[None]
    angles = torch.cat((angles, angles), dim=-1)
    shape = (1, len(positions), *((1,) * (tensor.ndim - 3)), dim)
    left, right = tensor.float().chunk(2, dim=-1)
    return tensor.float() * angles.cos().reshape(shape) + torch.cat((-right, left), dim=-1) * angles.sin().reshape(shape)


class LightningIndexer(nn.Module):
    """Separate compressed-key stream; top-k is trained through attention KL."""

    def __init__(self, hidden_size, settings, rope_base, device=None):
        super().__init__()
        self.dim, self.heads, self.rope_base = settings.index_dim, settings.index_heads, rope_base
        args = dict(bias=False, device=device, dtype=torch.float32)
        self.key_proj = nn.Linear(hidden_size, 2 * self.dim, **args)
        self.key_gate = nn.Linear(hidden_size, 2 * self.dim, **args)
        self.key_bias = nn.Parameter(torch.zeros(4, 2, self.dim, device=device, dtype=torch.float32))
        self.query_proj = nn.Linear(hidden_size, self.heads * self.dim, **args)
        self.weight_proj = nn.Linear(hidden_size, self.heads, **args)
        nn.init.zeros_(self.key_gate.weight)

    def compress(self, x, starts, source_start=0):
        # Indexer supervision must not update the backbone or the main compressor.
        x = x.detach().float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            payload = F.linear(x, self.key_proj.weight.float()).unflatten(-1, (2, self.dim))
            offsets = torch.arange(source_start, source_start + x.size(1), device=x.device) % 4
            scores = F.linear(x, self.key_gate.weight.float()).unflatten(-1, (2, self.dim)) + self.key_bias[offsets]
            payload, valid = overlap_windows(payload, starts, source_start)
            scores, _ = overlap_windows(scores, starts, source_start)
            key = gated_pool(select_roles(payload), select_roles(scores), valid)
            key = key * torch.rsqrt(key.square().mean(-1, keepdim=True) + 1e-6)
            return index_rope(key, starts, self.rope_base)

    def scores(self, x, positions, keys):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.detach().float()
            query = F.linear(x, self.query_proj.weight.float()).unflatten(-1, (self.heads, self.dim))
            query = query * torch.rsqrt(query.square().mean(-1, keepdim=True) + 1e-6)
            query = index_rope(query, positions, self.rope_base)
            weights = F.linear(x, self.weight_proj.weight.float()) / math.sqrt(self.heads * self.dim)
            # Avoid the [B, Q, H_index, all_compressed_tokens] intermediate.
            score = 0
            for head in range(self.heads):
                products = torch.einsum("bqd,bnd->bqn", query[:, :, head], keys.float())
                score = score + products.relu() * weights[:, :, head, None]
            return score


def indexer_kl(scores, probabilities, valid):
    """KL from detached main-attention mass; empty-history rows contribute zero."""
    target = probabilities.detach().sum(dim=(1, 2))  # B, query, compressed
    mass = target.sum(-1, keepdim=True)
    active = valid.any(-1)
    target = target / mass.clamp_min(torch.finfo(target.dtype).tiny)
    masked = scores.masked_fill(~valid, float("-inf"))
    masked = torch.where(active[..., None], masked, torch.zeros_like(masked))
    log_probability = masked.log_softmax(-1)
    # Invalid slots have 0 mass and -inf log-probability; avoid 0 * inf.
    log_probability = log_probability.masked_fill(~valid, 0)
    terms = target * (target.clamp_min(1e-30).log() - log_probability)
    return terms.sum(), active.sum()
