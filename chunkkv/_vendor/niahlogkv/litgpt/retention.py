"""Learned retention heads: which archived spans stay exact (niahLogKV).

A query-agnostic incompressibility estimator. Each layer's head reads the
pre-RoPE keys and the values of the candidate tokens (the K/V the exact pool
would keep) and outputs a logit that the token belongs to a specific detail a
later request will need: the kind of content the semantic ladder cannot
average without losing it. A span's logit is the largest of its tokens'; its
value has the form of Gamma's cue and literal values,

    max(exp(min(z, RETENTION_CAP)) - exp(RETENTION_FLOOR), 0) / length,

added to the span's K/V score before the whole-span greedy. Spans the head
scores below the floor keep their K/V score alone.

Heads are trained apart from the LM (``train_retention.py``) on synthetic
retrieval documents and frozen afterwards. They are not part of the LM
checkpoint, hold no KV and run inside the selection's existing device pass,
so the selection still reads back one score per span.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn.functional as F
from torch import nn

RETENTION_FLOOR = 0.
RETENTION_CAP = 12.
RETENTION_HIDDEN = 64
# A fresh head scores every span below the floor, so selection starts as K/V-only.
RETENTION_INIT_BIAS = -4.

# Head training: called as tap(head, k, v, cut, token_batch, tokens, span_sizes)
# from every selection that scores with a head (see train_retention.py).
RETENTION_TAP = None


def _unit(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


class RetentionHead(nn.Module):
    """One layer's head: RMS-normalised [K | V] of a token -> GELU MLP -> logit."""

    def __init__(self, n_groups: int, k_dim: int, v_dim: int, hidden: int = RETENTION_HIDDEN) -> None:
        super().__init__()
        self.n_groups, self.k_dim, self.v_dim = n_groups, k_dim, v_dim
        self.inp = nn.Linear(n_groups * (k_dim + v_dim), hidden)
        self.out = nn.Linear(hidden, 1)
        nn.init.constant_(self.out.bias, RETENTION_INIT_BIAS)

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Token logits from k [n, G, Dk] and v [n, G, Dv] (any float dtype); float32 [n]."""
        x = torch.cat((_unit(k.flatten(1)), _unit(v.flatten(1))), dim=-1)
        h = F.gelu(F.linear(x, self.inp.weight.float(), self.inp.bias.float()))
        return F.linear(h, self.out.weight.float(), self.out.bias.float()).squeeze(-1)


class RetentionHeads(nn.Module):
    """A head per layer, saved and loaded as one file."""

    def __init__(self, n_layer: int, n_groups: int, k_dim: int, v_dim: int, hidden: int = RETENTION_HIDDEN) -> None:
        super().__init__()
        self.shape = dict(n_layer=n_layer, n_groups=n_groups, k_dim=k_dim, v_dim=v_dim, hidden=hidden)
        self.heads = nn.ModuleList(RetentionHead(n_groups, k_dim, v_dim, hidden) for _ in range(n_layer))

    def __len__(self) -> int:
        return len(self.heads)

    def __getitem__(self, layer: int) -> RetentionHead:
        return self.heads[layer]

    def tag(self) -> str:
        """Short digest of the weights: replay signatures and run metadata name the heads by it."""
        digest = hashlib.sha1()
        for name, tensor in sorted(self.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().float().cpu().numpy().tobytes())
        return digest.hexdigest()[:12]

    def blob(self, **meta) -> dict:
        """What ``save`` writes: shape, CPU weights and metadata (``load`` ignores any other keys)."""
        state = {name: tensor.detach().cpu() for name, tensor in self.state_dict().items()}
        return {"shape": self.shape, "state": state, "meta": meta}

    def save(self, path, **meta) -> None:
        torch.save(self.blob(**meta), path)

    @classmethod
    def load(cls, path, device=None) -> "RetentionHeads":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        heads = cls(**blob["shape"])
        heads.load_state_dict(blob["state"])
        heads.meta = blob.get("meta", {})
        heads.requires_grad_(False)
        return heads.to(device=device, dtype=torch.float32) if device is not None else heads.float()


def span_logits(head, k, v, token_batch, tokens, span_sizes):
    """Largest token logit per span.

    ``k``/``v`` are [B, G, T, D] (pool, then flush); ``token_batch``/``tokens``
    index each span's tokens in candidate order and ``span_sizes`` are the span
    lengths, all device tensors.
    """
    z = head(k[token_batch, :, tokens], v[token_batch, :, tokens])
    return torch.segment_reduce(z, "max", lengths=span_sizes, unsafe=True)


def span_values(head, k, v, cut, token_batch, tokens, span_sizes):
    """Device span values in candidate order (see the module docstring)."""
    if RETENTION_TAP is not None:
        RETENTION_TAP(head, k, v, cut, token_batch, tokens, span_sizes)
    with torch.no_grad():
        z = span_logits(head, k, v, token_batch, tokens, span_sizes)
        value = torch.exp(z.clamp(max=RETENTION_CAP)) - math.exp(RETENTION_FLOOR)
        return value.clamp_min_(0) / span_sizes
