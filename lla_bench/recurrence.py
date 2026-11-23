"""Base-two weak HAttention with a scalar decay (upstream HStruct.MAMBA2).

A prefix of length p has one completed, gated K^T V sum for each set bit of
p. At the next query, bit j is read with lambda[j+1], and the new token with
lambda[0]. Carry immediately after the read, equivalent to cascading before
the next insert in HanGuo97/log-linear-attention's HState.cascade_weak.
"""
import torch
from torch import nn


class MatrixCache(nn.Module):
    def __init__(self, batch, groups, key_dim, value_dim, window, device):
        super().__init__()
        self.window = window
        self.count = 0
        self.register_buffer("states", torch.empty(
            window.bit_length(), batch, groups, key_dim, value_dim,
            device=device, dtype=torch.float32), persistent=False)

    def reset(self):
        # Occupancy is completely determined by count. Each new live slot is overwritten.
        self.count = 0


def _tile_heads(tensor, heads):
    """Broadcast shared projections only for the current tile."""
    groups = tensor.size(1)
    if groups == heads:
        return tensor
    if groups == 1:
        return tensor.expand(-1, heads, -1, -1)
    return tensor.repeat_interleave(heads // groups, dim=1)


def weak_attention(q, k, v, log_decay, weights, cache=None, tile=128):
    """q/k [B,Q/KG,T,K], v [B,G,T,V], log_decay [B,G,T], positive weights [B,H,T,L].

    FP32 states and sums. Tiles are aligned powers of two so outside prefix
    blocks stay at the same read level throughout a tile. The only pairwise
    workspace is bounded by tile x tile. With cache=None this is differentiable.
    """
    batch, query_heads, length, key_dim = q.shape
    heads, groups, value_dim = weights.size(1), v.size(1), v.size(-1)
    start = 0 if cache is None else cache.count
    end = start + length
    if (length < 1 or tile < 1 or heads % groups or heads % query_heads or groups % k.size(1)
            or weights.size(-1) < (end - 1).bit_length() + 1):
        raise ValueError("invalid sequence length, GQA shape, tile size or number of read levels")
    if cache is not None and end > cache.window:
        raise ValueError("recurrent cache window exceeded")
    live = {j: cache.states[j] for j in range(start.bit_length()) if start & (1 << j)}
    outputs = []
    pos = start
    while pos < end:
        limit = min(tile, end - pos, (pos & -pos) if pos else end)
        j = limit.bit_length() - 1
        width = 1 << j
        span = slice(pos - start, pos - start + width)
        q_tile, k_tile = q[:, :, span].float(), k[:, :, span].float()
        query = _tile_heads(q_tile, heads).reshape(batch, groups, heads // groups, width, key_dim)
        key, value = _tile_heads(k_tile, groups), v[:, :, span].float()
        lam = weights[:, :, span].float().reshape(batch, groups, heads // groups, width, -1)
        decay = log_decay[:, :, span].float().cumsum(-1)

        # For a local causal pair (t,s), the most significant differing bit
        # selects the left sibling of t's path. Diagonal pairs read level 0.
        ids = torch.arange(width, device=q.device)
        xor = ids[:, None] ^ ids
        levels = torch.bucketize(xor.contiguous(), 2 ** torch.arange(j + 1, device=q.device), right=True)
        factor = lam.gather(-1, levels.expand(*lam.shape[:-2], width, width))
        delta = decay.unsqueeze(-1) - decay.unsqueeze(-2)
        gate = delta.masked_fill(ids[:, None] < ids, -torch.inf).exp()
        if query_heads == k.size(1):
            # Mamba-2 shares B and C within a projection group. QK is identical
            # for its heads, while decays, values and level weights remain head-specific.
            shared_scores = q_tile @ k_tile.transpose(-2, -1)
            scores = _tile_heads(shared_scores, heads).reshape(batch, groups, heads // groups, width, width)
        else:
            scores = query @ key.unsqueeze(2).transpose(-2, -1)
        y = (scores * factor * gate.unsqueeze(2)) @ value.unsqueeze(2)

        last = decay[..., -1:]
        for level, state in live.items():
            read = query @ state.unsqueeze(2)
            y = y + read * (decay.exp().unsqueeze(2) * lam[..., level + 1]).unsqueeze(-1)
            live[level] = state * last.exp().unsqueeze(-1)
        outputs.append(y.reshape(batch, heads, width, value_dim).to(v.dtype))

        # Advance all old states to the end of the tile and insert its summary.
        state = (key * (last - decay).exp().unsqueeze(-1)).transpose(-2, -1) @ value
        while j in live:
            state = live.pop(j) + state
            j += 1
        live[j] = state
        pos += width

    if cache is not None:
        if torch.is_grad_enabled():
            raise ValueError("a persistent inference cache requires no_grad")
        for level, state in live.items():
            cache.states[level].copy_(state)
        cache.count = end
    return torch.cat(outputs, dim=2)
