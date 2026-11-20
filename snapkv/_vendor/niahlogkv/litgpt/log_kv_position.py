"""Position helpers for SemanticLogKV anchor readout.

The cache stores pre-RoPE key-space vectors plus integer position anchors.
Readout materializes up to three virtual slots per entry at real RoPE
coordinates and corrects the mass bias by the number of distinct anchors.
"""

from __future__ import annotations

import torch


def merge_anchors(
    lo1: torch.Tensor,
    hi1: torch.Tensor,
    sum_wp1: torch.Tensor,
    lo2: torch.Tensor,
    hi2: torch.Tensor,
    sum_wp2: torch.Tensor,
    w1: torch.Tensor | None = None,
    w2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge anchor metadata; exact and associative."""
    if w1 is None or w2 is None:
        return torch.minimum(lo1, lo2), torch.maximum(hi1, hi2), sum_wp1.long() + sum_wp2.long()
    real1 = w1 > 0
    real2 = w2 > 0
    lo = torch.where(real1 & real2, torch.minimum(lo1, lo2), torch.where(real1, lo1, lo2))
    hi = torch.where(real1 & real2, torch.maximum(hi1, hi2), torch.where(real1, hi1, hi2))
    return lo.long(), hi.long(), sum_wp1.long() + sum_wp2.long()


def mid_anchor(lo: torch.Tensor, hi: torch.Tensor, sum_wp: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Round-half-up weighted centroid, clamped into ``[lo, hi]``.

    Callers must filter ``w == 0`` pad/dead entries before using the returned
    position for RoPE lookup. ``dedup_anchors`` handles that sanitization.
    """
    real = w > 0
    safe_lo = torch.where(real, lo.long(), torch.zeros_like(lo.long()))
    safe_hi = torch.where(real, hi.long(), torch.zeros_like(hi.long()))
    ww = w.long().clamp_min(1)
    mid = torch.div(2 * sum_wp.long() + ww, 2 * ww, rounding_mode="floor")
    return torch.clamp(mid, safe_lo, safe_hi)


def dedup_anchors(
    lo: torch.Tensor,
    hi: torch.Tensor,
    mid: torch.Tensor,
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed ``[lo, mid, hi]`` anchors, validity mask, and clamped M.

    Invalid entries (``w <= 0``) get all anchors rewritten to safe index 0 and
    all three virtual slots marked invalid. Valid entries keep the first
    occurrence of each repeated anchor.
    """
    anchors = torch.stack((lo.long(), mid.long(), hi.long()), dim=-1)
    entry_valid = w > 0
    slot_valid = torch.stack(
        (
            entry_valid,
            entry_valid & (anchors[..., 1] != anchors[..., 0]),
            entry_valid & (anchors[..., 2] != anchors[..., 0]) & (anchors[..., 2] != anchors[..., 1]),
        ),
        dim=-1,
    )
    anchors = torch.where(entry_valid.unsqueeze(-1), anchors, torch.zeros_like(anchors))
    M = slot_valid.sum(dim=-1).clamp_min(1).long()
    return anchors, slot_valid, M


def _rotate_at_anchors(
    content: torch.Tensor,
    anchors: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    rope_n_elem: int,
) -> torch.Tensor:
    """Apply LitGPT's standard RoPE rotation to ``content`` at ``anchors``."""
    if cos_cache.shape != sin_cache.shape:
        raise ValueError(f"cos_cache/sin_cache shapes differ: {cos_cache.shape} vs {sin_cache.shape}")
    if cos_cache.dim() != 2:
        raise ValueError(f"cos_cache must be 2-D, got shape {cos_cache.shape}")
    if anchors.shape[:-1] != content.shape[:-1]:
        raise ValueError(f"anchors/content prefix shapes differ: {anchors.shape[:-1]} vs {content.shape[:-1]}")
    if rope_n_elem < 0 or rope_n_elem > content.size(-1):
        raise ValueError(f"rope_n_elem must be in [0, {content.size(-1)}], got {rope_n_elem}")

    x = content.unsqueeze(-2)[..., :rope_n_elem]
    if rope_n_elem == 0:
        return content.unsqueeze(-2).expand(*anchors.shape, content.size(-1))

    cos = cos_cache[anchors.long()]
    sin = sin_cache[anchors.long()]
    half = rope_n_elem // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    roped = (x * cos) + (rotated * sin)
    tail = content.unsqueeze(-2)[..., rope_n_elem:].expand(*roped.shape[:-1], content.size(-1) - rope_n_elem)
    return torch.cat((roped.to(content.dtype), tail), dim=-1)


def materialize_anchor_keys(
    k_raw: torch.Tensor,
    anchors: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    rope_n_elem: int,
) -> torch.Tensor:
    return _rotate_at_anchors(k_raw, anchors, cos_cache, sin_cache, rope_n_elem)


def materialize_anchor_directions(
    sigma_u_raw: torch.Tensor,
    gamma_a_raw: torch.Tensor,
    anchors: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    rope_n_elem: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _rotate_at_anchors(sigma_u_raw, anchors, cos_cache, sin_cache, rope_n_elem),
        _rotate_at_anchors(gamma_a_raw, anchors, cos_cache, sin_cache, rope_n_elem),
    )


def anchor_mass_bias(w: torch.Tensor, M_s: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Compute ``lam * log(w) - log(M_s)`` with ``-log(M_s)`` ungated."""
    bias = -M_s.float().clamp_min(1).log()
    if lam != 0.0:
        bias = bias + float(lam) * w.float().clamp_min(1).log()
    return bias
