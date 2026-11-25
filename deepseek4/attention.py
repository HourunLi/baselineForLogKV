"""Torch implementation of V4-Flash shared KV attention. See docs/DEEPSEEK4.md."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Geometry:
    width: int = 4096
    heads: int = 64
    dim: int = 512
    rotary: int = 64
    q_rank: int = 1024
    groups: int = 8
    o_rank: int = 1024
    index_heads: int = 64
    index_dim: int = 128
    topk: int = 512
    window: int = 128


LAYOUT = (0, 0) + (4, 128) * 17


def rotate(x, positions, theta, factor, inverse=False):
    """Interleaved partial RoPE; the evaluation pass supplies its YaRN factor."""
    d = x.shape[-1]
    exponent = torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d
    freq = theta**-exponent
    if factor > 1:

        def cutoff(beta):
            return d * math.log(32768 / (2 * math.pi * beta)) / (2 * math.log(theta))

        low, high = max(0, math.floor(cutoff(32))), min(d - 1, math.ceil(cutoff(1)))
        ramp = ((exponent * d / 2 - low) / max(high - low, 0.001)).clamp(0, 1)
        freq = freq * (1 - ramp + ramp / factor)
    angle = positions.float()[:, None] * freq
    if inverse:
        angle = -angle
    shape = [1, positions.numel()] + [1] * (x.ndim - 3) + [d // 2]
    cos, sin = angle.cos().view(shape), angle.sin().view(shape)
    a, b = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2).to(x.dtype)


class Compressor(nn.Module):
    def __init__(self, g, ratio, dim):
        super().__init__()
        self.ratio, self.dim = ratio, dim
        channels = dim * (2 if ratio == 4 else 1)
        self.payload = nn.Linear(g.width, channels, bias=False)
        self.gate = nn.Linear(g.width, channels, bias=False)
        self.phase = nn.Parameter(torch.zeros(ratio, channels))
        self.norm = nn.RMSNorm(dim, eps=1e-6)

    def forward(self, blocks, previous):
        # blocks: [B, closed blocks, stride, width]. The previous block's
        # first channel bank overlaps the current block's second bank for CSA.
        def project(layer, bias=False):
            v = layer(blocks).float()
            if bias:
                v = v + self.phase
            if self.ratio != 4:
                return v
            left, right = v.split(self.dim, -1)
            if previous is None:
                first = torch.full_like(left[:, :1], -torch.inf if bias else 0)
            else:
                first = layer(previous[:, None]).float()[..., : self.dim]
                if bias:
                    first = first + self.phase[..., : self.dim]
            return torch.cat((torch.cat((first, left[:, :-1]), 1), right), 2)

        pooled = (project(self.payload) * project(self.gate, True).softmax(2)).sum(2)
        return self.norm(pooled.to(blocks.dtype))


class Cache(nn.Module):
    """Fixed storage; valid prefixes tracked by position, reset without reallocating."""

    def __init__(self, g, ratio, capacity, batch, device, dtype):
        super().__init__()

        def buffer(name, length, width):
            self.register_buffer(
                name, torch.empty(batch, length, width, device=device, dtype=dtype), persistent=False
            )

        buffer("local", min(g.window, capacity), g.dim)
        if ratio:
            buffer("pool", capacity // ratio, g.dim)
            buffer("tail", ratio * (2 if ratio == 4 else 1), g.width)
        if ratio == 4:
            buffer("keys", capacity // ratio, g.index_dim)
        self.pos = 0
        self.capacity = capacity


class Attention(nn.Module):
    def __init__(self, g, ratio, factor=1):
        super().__init__()
        self.g, self.ratio = g, ratio
        self.theta, self.factor = (160000, factor) if ratio else (10000, 1)
        self.qa = nn.Linear(g.width, g.q_rank, bias=False)
        self.q_norm = nn.RMSNorm(g.q_rank, eps=1e-6)
        self.qb = nn.Linear(g.q_rank, g.heads * g.dim, bias=False)
        self.kv = nn.Linear(g.width, g.dim, bias=False)
        self.kv_norm = nn.RMSNorm(g.dim, eps=1e-6)
        group_width = g.heads * g.dim // g.groups
        self.oa = nn.Parameter(torch.empty(g.groups, g.o_rank, group_width))
        nn.init.normal_(self.oa, std=group_width**-0.5)
        self.ob = nn.Linear(g.groups * g.o_rank, g.width, bias=False)
        self.sink = nn.Parameter(torch.zeros(g.heads))
        if ratio:
            self.compressor = Compressor(g, ratio, g.dim)
        if ratio == 4:
            self.index_compressor = Compressor(g, ratio, g.index_dim)
            self.iq = nn.Linear(g.q_rank, g.index_heads * g.index_dim, bias=False)
            self.iw = nn.Linear(g.width, g.index_heads, bias=False)
        self.kv_cache = None

    def rope(self, x, pos, inverse=False):
        d = self.g.rotary
        return torch.cat((x[..., :-d], rotate(x[..., -d:], pos, self.theta, self.factor, inverse)), -1)

    def compress(self, x, start, cache):
        ratio = self.ratio
        first, last = start // ratio, (start + x.size(1)) // ratio
        rem = start % ratio
        overlap = ratio if ratio == 4 and first else 0
        tail = x if cache is None else torch.cat((cache.tail[:, : overlap + rem], x), 1)
        complete = last - first
        pool = keys = None
        if complete:
            blocks = tail[:, overlap : overlap + complete * ratio].unflatten(1, (complete, ratio))
            previous = tail[:, :overlap] if overlap else None
            anchors = torch.arange(first, last, device=x.device) * ratio
            pool = self.rope(self.compressor(blocks, previous), anchors)
            if ratio == 4:
                keys = self.rope(self.index_compressor(blocks, previous), anchors)
        if cache is not None:
            keep = (start + x.size(1)) % ratio + (ratio if ratio == 4 and last else 0)
            if keep:
                cache.tail[:, :keep].copy_(tail[:, -keep:])
            if complete:
                cache.pool[:, first:last].copy_(pool)
                if ratio == 4:
                    cache.keys[:, first:last].copy_(keys)
            pool = cache.pool[:, :last]
            keys = cache.keys[:, :last] if ratio == 4 else None
        return pool, keys

    def index(self, qr, x, keys, visible):
        g = self.g
        pos, limits = visible
        iq = self.rope(self.iq(qr).unflatten(-1, (g.index_heads, g.index_dim)), pos)
        weights = self.iw(x) * (g.index_dim * g.index_heads) ** -0.5
        # Score every completed block, bounding the temporary heads x history tensor.
        scores = []
        for tile in keys.split(2048, 1):
            raw = torch.einsum("bqhd,bkd->bqhk", iq, tile)
            scores.append(torch.einsum("bqhk,bqh->bqk", raw.relu(), weights).float())
        score = torch.cat(scores, -1)
        valid = torch.arange(keys.size(1), device=x.device) < limits[:, None]
        score = score.masked_fill(~valid, -torch.inf)
        # ReLU gives exact ties (often zero). Prefer the earlier block so
        # chunked prefill and decode select the same history.
        ids = score.argsort(dim=-1, descending=True, stable=True)[..., : g.topk]
        return score, ids, valid

    def attend(self, x, qr, pos, local, local_start, pool, keys):
        g = self.g
        q = self.qb(qr).unflatten(-1, (g.heads, g.dim))
        q = self.rope(F.rms_norm(q, (g.dim,), eps=1e-6), pos)
        loc = pos[:, None] - torch.arange(min(g.window, local.size(1)), device=x.device).flip(0)
        values = local[:, (loc - local_start).clamp_min(0)]
        valid = (loc >= local_start)[None].expand(x.size(0), -1, -1)
        scores = index_ids = None
        if pool is not None and pool.size(1):
            limits = (pos + 1) // self.ratio
            if self.ratio == 4:
                scores, index_ids, _ = self.index(qr, x, keys, (pos, limits))
                batch = torch.arange(x.size(0), device=x.device)[:, None, None]
                history = pool[batch, index_ids]
                usable = index_ids < limits[None, :, None]
            else:
                history = pool[:, None].expand(-1, x.size(1), -1, -1)
                usable = (torch.arange(pool.size(1), device=x.device) < limits[:, None])[None].expand(
                    x.size(0), -1, -1
                )
            valid = torch.cat((valid, usable), -1)
            values = torch.cat((values, history), 2)
        logits = torch.einsum("bqhd,bqkd->bqhk", q, values).float() * g.dim**-0.5
        logits = logits.masked_fill(~valid[:, :, None], -torch.inf)
        sink = self.sink.float().expand(*logits.shape[:-2], -1)[..., None]
        probs = torch.cat((logits, sink), -1).softmax(-1)[..., :-1]
        loss = x.new_zeros((), dtype=torch.float32)
        if self.training and scores is not None:
            # Detached attention mass teaches the discrete indexer; invalid early
            # rows have zero target mass. Normalize over all eligible candidates.
            target = probs[..., -index_ids.size(-1) :].detach().sum(-2)
            target = target / target.sum(-1, keepdim=True).clamp_min(1e-10)
            logp = scores.nan_to_num(neginf=-1e9).log_softmax(-1).gather(-1, index_ids)
            loss = -(target * logp).sum(-1).sum()
        out = torch.einsum("bqhk,bqkd->bqhd", probs.to(values.dtype), values)
        out = self.rope(out, pos, inverse=True).reshape(*x.shape[:2], g.groups, -1)
        return self.ob(torch.einsum("bqgd,grd->bqgr", out, self.oa).flatten(2)), loss

    def forward(self, x, start=0, cached=False):
        g, cache = self.g, self.kv_cache if cached else None
        if cached and (cache is None or start != cache.pos or start + x.size(1) > cache.capacity):
            raise ValueError("KV writes must be contiguous and fit the configured window")
        pos = torch.arange(start, start + x.size(1), device=x.device)
        kv = self.rope(self.kv_norm(self.kv(x)), pos)
        recent = min(start, g.window)
        local = torch.cat((cache.local[:, :recent], kv), 1) if recent else kv
        if cache is not None:
            n = min(g.window, local.size(1))
            cache.local[:, :n].copy_(local[:, -n:])
        pool, keys = self.compress(x, start, cache) if self.ratio else (None, None)
        parts, losses = [], []
        for off in range(0, x.size(1), 16):
            part = x[:, off : off + 16]
            end = start + off + part.size(1)
            count = end // self.ratio if self.ratio else 0
            out, loss = self.attend(
                part,
                self.q_norm(self.qa(part)),
                pos[off : off + 16],
                local,
                start - recent,
                pool[:, :count] if count else None,
                keys[:, :count] if keys is not None else None,
            )
            parts.append(out)
            losses.append(loss)
        if cache is not None:
            cache.pos = start + x.size(1)
        return torch.cat(parts, 1), torch.stack(losses).sum() / (x.size(0) * x.size(1))
