# Vendored for the DMS baseline; local namespace changes are recorded in SOURCE.json.
"""Log-Structured KV Cache with merged-position slots — strict O(B·log N).

Architecture (uniform 2:1 compaction at every level):
- Buffer (sliding window): recent_size raw tokens, no compression (exact keys).
- Level 0 (write level):   B entries, each = 2 tokens merged. Partially fillable
                            (0 to B entries). Filled one entry at a time from buffer
                            flush. When full, carries to level 1.
- Level 1+ (carry levels):  B entries per level, each = 2 entries from prev level
                            merged. All-or-nothing (0 or B entries). Binary carry
                            promotes full blocks upward.

Position handling ("expected RoPE", merged into the compressed state):
- Keys enter the cache as FULL post-RoPE vectors
      k_j = [ R(p_j)·k_pos_j ; k_content_j ]
  (partial rotary: only the leading rope_n_elem dims are rotated; any remaining
  content channel is position-free; full rotary means the content channel is
  empty — both are supported).
- Merging a block mean-pools the WHOLE key. Mean commutes with concatenation, so
  the merged key is [ mean_j R(p_j)·k_pos_j ; mean_j k_content_j ]: the block's
  position embedding is the weighted mean of its tokens' rotated position keys —
  the exact expectation of the rotation over the span. Per RoPE frequency θ over
  a width-w span centred at c this equals R(c)·sin(wθ/2)/(w·sin(θ/2)): RoPE at
  the span centre, damped per-frequency by a Dirichlet factor. High frequencies
  fade as spans widen, so positional resolution decays ∝ span width ∝ distance
  (constant relative error). The merged key is deliberately NOT renormalized:
  the norm shrinkage encodes the span's positional uncertainty.
- Optional AlphaLogKV retains a fixed-size pool of exact spans outside recent.

Slot attention (``log_kv_slot_attention``), when rank-1 stats are supplied:
    score_s = scale·(q · k_s) + 0.5·scale²·sigma2_s·(q · sigma_u_s)² + λ·log(w_s)
    read_s  = v_s + scale·gamma_s·(q · gamma_a_s)·gamma_b_s
    out     = softmax(score) · read
The +λ·log(w_s) mass bias gives a w-token slot softmax mass ≈ w·exp(score),
while the Sigma/Gamma terms add the second-order score correction and first-order
value read-out correction. Exact recent / in-flight tokens carry zero stats, so
they reduce to ordinary token attention. Calling ``log_kv_slot_attention`` without
stats preserves the original first-order compatibility path.

Memory:  O(recent_size + B·log(N/2)) slots — no Θ(N) term.
Compute: O(recent_size + B·log(N/2)) per query — no Θ(N) term.
"""

from __future__ import annotations

import contextlib
import os
from copy import deepcopy
import math
import warnings
from array import array
from functools import lru_cache
from typing import Any, NamedTuple, NoReturn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.backends.cuda import SDPAParams, can_use_flash_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

from models._vendor.semanticlogkv.litgpt.log_kv_timing import HOST_STATS as LOGKV_HOST_STATS, logkv_take_host_stats, logkv_timed
from models._vendor.semanticlogkv.litgpt.log_kv_checkpoint import checkpoint_record_routes, checkpoint_route_replay

from models._vendor.semanticlogkv.litgpt.log_kv_position import (
    anchor_mass_bias,
    dedup_anchors,
    materialize_anchor_directions,
    materialize_anchor_keys,
    merge_anchors,
    mid_anchor,
)


_RANK1_EPS = 1e-12

# Squarings used to extract the dominant eigenvector of the tiny (r x r) cores
# below. Each round squares the eigenvalue gap, so k rounds separate the top
# eigenpair as well as 2^k power iterations would, at k batched GEMMs.
#
# 8 rounds because they are almost free and the accuracy is not: the cost of
# these helpers sits in the one (batch, r, D) projection, not in the r x r
# squarings, so 5 -> 8 rounds costs ~4% of the helper while cutting the
# worst-case truncation error ~8x (3.1e-3 -> 3.8e-4 excess relative Frobenius
# error over an exact eigh, measured on the near-degenerate case that converges
# slowest). TestRank1Approximation pins the resulting quality.
_RANK1_SQUARINGS = 8

LOG_KV_OP_NEW_CLUSTER = 0
LOG_KV_OP_NEW_SEGMENT = 1
LOG_KV_OP_JOIN = 2
LOG_KV_OP_WARD_MERGE = 3
LOG_KV_OP_PAD_INSERT = 4


class _SemanticReplayPlans:
    """CPU-only plans owned by one autograd/checkpoint invocation, never a cache.

    Freeze the host log so later caller mutations cannot invalidate a plan.
    Both replay passes share this object; it retains no tensors or cache refs.
    """

    def __init__(self, host_log):
        if host_log is None:
            raise RuntimeError("semantic replay plans require an existing CPU op-log")
        self.host_log = tuple(tuple(tuple(map(tuple, lane)) for lane in batch) for batch in host_log)
        self.plans = {}


class _SemanticReplayUpdates:
    """One training invocation's CPU schedule and detached device updates.

    Distinct from CPU-only _SemanticReplayPlans: these payloads intentionally
    reuse this forward's K/V, never another input or inference invocation.
    """

    def __init__(self, flushes=None, tensors=None):
        self.flushes = {} if flushes is None else flushes
        self.tensors = [] if tensors is None else tensors

    def wait(self, device):
        if device.type == "cuda":
            stream = torch.cuda.current_stream(device)
            for record in self.flushes.values():
                if record[-1] is not None:
                    stream.wait_event(record[-1])

    def save(self, tensor, owned=False):
        """Keep `tensor` for replay; copy it unless the caller owns it outright.

        Owned tensors are fresh temporaries that nothing writes afterwards, so
        holding the reference keeps the values without a device copy.
        """
        if tensor is None:
            return None
        index = len(self.tensors)
        self.tensors.append(tensor.detach() if owned else tensor.detach().clone())
        return index


class _SemanticTreeCluster(NamedTuple):
    # Unified production routing carries centroids in a separate packed tensor.
    centroid: torch.Tensor | None
    n_total: int
    p_hi: int
    existing: tuple[int, ...]
    tokens: tuple[int, ...]


class CacheAttentionState(NamedTuple):
    slot_k: torch.Tensor
    slot_v: torch.Tensor
    slot_w: torch.Tensor
    slot_valid: torch.Tensor | None = None
    M_s: torch.Tensor | None = None
    slot_sigma_u: torch.Tensor | None = None
    slot_sigma2: torch.Tensor | None = None
    slot_gamma_a: torch.Tensor | None = None
    slot_gamma_b: torch.Tensor | None = None
    slot_gamma: torch.Tensor | None = None


def _normalize(x: torch.Tensor, eps: float = _RANK1_EPS) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a unit direction and the original norm, computed in fp32."""
    xf = x.float()
    norm = xf.norm(dim=-1, keepdim=True)
    unit = torch.where(norm > eps, xf / norm.clamp_min(eps), torch.zeros_like(xf))
    return unit.to(x.dtype), norm.squeeze(-1).to(x.dtype)


def _dominant_eigvec_small(core: torch.Tensor) -> torch.Tensor:
    """Unit dominant right eigenvector of a batch of tiny ``(..., r, r)`` cores.

    ``core`` must have a real non-negative spectrum: either symmetric PSD, or a
    product of two PSD Gram matrices (similar to the PSD ``G^(1/2) H G^(1/2)``).

    Repeated squaring drives ``core^(2^k)`` towards the rank-1 dominant
    projector, so every column ends up parallel to the top eigenvector and the
    largest-norm column is the best-conditioned representative. Convergence is
    ``(lam2/lam1)^(2^k)``; the slowest case is a near-degenerate top pair, which
    is exactly the case where any vector of that eigenspace is an equally good
    rank-1 direction, so the resulting approximation error stays small either
    way.

    Why not ``torch.linalg.eigh``/``qr``/``svd``: those dispatch to cuSOLVER,
    whose batched small-matrix paths carry a large fixed cost (and, for tall-thin
    QR, degrade towards a per-matrix loop). LogKV calls this once per merged slot
    per compaction — thousands of matrices per call, thousands of calls per
    training step — where that fixed cost dominated everything else. This path is
    batched GEMM only.
    """
    r = core.size(-1)
    if r == 1:
        return torch.ones_like(core[..., 0])
    m = core
    for _ in range(_RANK1_SQUARINGS):
        # Renormalize by the trace before squaring: raw powers reach lam1^(2^k)
        # and overflow fp32 within a few rounds, and the scale is irrelevant to
        # the direction. trace > 0 for any matrix with non-negative spectrum.
        trace = m.diagonal(dim1=-2, dim2=-1).sum(-1).abs().clamp_min(_RANK1_EPS)
        m = m / trace[..., None, None]
        m = torch.matmul(m, m)
    # Columns of m^(2^k) are all parallel to the top eigenvector; pick the
    # longest one (zero everywhere iff the input was the zero matrix).
    col_norms = m.norm(dim=-2)                       # (..., r)
    pick = col_norms.argmax(dim=-1, keepdim=True)    # (..., 1)
    vec = torch.gather(m, -1, pick.unsqueeze(-2).expand(*m.shape[:-1], 1)).squeeze(-1)
    norm = vec.norm(dim=-1, keepdim=True)
    return torch.where(norm > _RANK1_EPS, vec / norm.clamp_min(_RANK1_EPS), torch.zeros_like(vec))


def _rank1_psd_from_factors(factors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Near-optimal rank-1 PSD approximation to ``sum_i f_i f_i^T``.

    ``factors`` is ``(..., r, D)`` with small ``r`` for merged slots (3 during
    carry; 2 for raw pairs). The non-zero spectrum of the D x D matrix lives in
    the r x r Gram matrix, so the truncation runs there, without ever
    materializing a full covariance matrix.

    NOT an exact top-eigen truncation: the direction comes from
    ``_dominant_eigvec_small``, which iterates rather than solves. The
    eigenvalue is then an exact Rayleigh quotient of that direction, so it is
    the best scale for whatever direction was found. ``TestRank1Approximation``
    pins the resulting reconstruction error against an ``eigh`` reference.
    """
    if factors.size(-2) == 0:
        return factors[..., :0, :].sum(dim=-2), factors.new_zeros(factors.shape[:-2])

    ff = factors.float()
    gram = torch.matmul(ff, ff.mT)
    coeff = _dominant_eigvec_small(gram)
    # raw = F^T c, so ||raw||^2 = c^T G c is the Rayleigh quotient — the top
    # eigenvalue, and second-order accurate in any eigenvector error. It also
    # comes for free from the projection we need anyway.
    raw = torch.matmul(coeff.unsqueeze(-2), ff).squeeze(-2)
    top = raw.square().sum(-1)
    direction = raw / top.sqrt().clamp_min(_RANK1_EPS).unsqueeze(-1)
    direction = torch.where(top.unsqueeze(-1) > _RANK1_EPS, direction, torch.zeros_like(direction))
    return direction.to(factors.dtype), top.to(factors.dtype)


def _rank1_cross_from_factors(
    left_factors: torch.Tensor,
    right_factors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Near-optimal rank-1 approximation to ``M = sum_i l_i r_i^T``.

    Like ``_rank1_psd_from_factors`` this is iterative, not an exact SVD
    truncation; ``TestRank1Approximation`` pins the error against a QR/SVD
    reference. The singular value is computed from the directions actually
    found, so the returned triplet is always self-consistent.

    Returns ``left_unit, right_unit, singular_value``. The full value-key cross
    covariance is never materialized, and neither are the thin bases of its
    factors. With ``L`` / ``R`` the ``(..., r, D)`` factor stacks, ``M = L^T R``
    and ``M M^T = L^T (R R^T) L``, so the top left singular vector lies in the
    row space of ``L``: writing it as ``L^T c`` reduces the eigenproblem to the
    r x r core ``(R R^T)(L L^T)``. From its dominant eigenvector ``c``,
    ``u ∝ L^T c`` and ``v ∝ M^T u = R^T (L L^T) c``, and the singular value is
    ``u^T M v = <L u, R v>``. Everything is r x r or a single (r, D) projection.
    """
    left = left_factors.float()    # (..., r, Dv)
    right = right_factors.float()  # (..., r, Dk)
    gram_left = torch.matmul(left, left.mT)     # (..., r, r)
    gram_right = torch.matmul(right, right.mT)  # (..., r, r)
    coeff = _dominant_eigvec_small(torch.matmul(gram_right, gram_left))  # (..., r)

    left_raw = torch.matmul(coeff.unsqueeze(-2), left).squeeze(-2)  # (..., Dv) = L^T c
    # R^T (L L^T) c — positively aligned with the true right vector for this u,
    # so the singular value below comes out non-negative without a sign fix.
    right_raw = torch.matmul(
        torch.matmul(coeff.unsqueeze(-2), gram_left), right
    ).squeeze(-2)  # (..., Dk)

    left_norm = left_raw.norm(dim=-1, keepdim=True)
    right_norm = right_raw.norm(dim=-1, keepdim=True)
    left_unit = torch.where(
        left_norm > _RANK1_EPS, left_raw / left_norm.clamp_min(_RANK1_EPS), torch.zeros_like(left_raw)
    )
    right_unit = torch.where(
        right_norm > _RANK1_EPS, right_raw / right_norm.clamp_min(_RANK1_EPS), torch.zeros_like(right_raw)
    )

    gamma = (
        torch.matmul(left, left_unit.unsqueeze(-1)) * torch.matmul(right, right_unit.unsqueeze(-1))
    ).sum(dim=(-2, -1)).clamp_min(0.0)
    left_unit = torch.where(gamma.unsqueeze(-1) > _RANK1_EPS, left_unit, torch.zeros_like(left_unit))
    right_unit = torch.where(gamma.unsqueeze(-1) > _RANK1_EPS, right_unit, torch.zeros_like(right_unit))
    return (
        left_unit.to(left_factors.dtype),
        right_unit.to(right_factors.dtype),
        gamma.to(left_factors.dtype),
    )


def _pair_rank1_stats(
    ka: torch.Tensor,
    kb: torch.Tensor,
    va: torch.Tensor,
    vb: torch.Tensor,
    frac_a: torch.Tensor | float = 0.5,
    frac_b: torch.Tensor | float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact rank-1 covariance / cross-covariance stats for a 2-token slot.

    ``frac_a``/``frac_b`` are the pooling weights used for the slot mean
    (default 0.5/0.5, i.e. an unweighted pair). For a 2-point set the
    weighted covariance ``p_a(ka-m)(ka-m)^T + p_b(kb-m)(kb-m)^T`` collapses
    to exactly ``p_a*p_b * dk dk^T`` (``m = p_a*ka + p_b*kb``, ``dk = ka-kb``)
    regardless of ``p_a``/``p_b`` — rank-1 with no truncation error, same as
    the unweighted case (which is this formula at ``p_a=p_b=0.5``).
    """
    dk = ka - kb
    dv = va - vb
    sigma_u, dk_norm = _normalize(dk)
    gamma_b, dv_norm = _normalize(dv)
    cross = frac_a * frac_b
    sigma2 = dk_norm.square() * cross
    gamma = dk_norm * dv_norm * cross
    gamma_a = sigma_u
    return sigma_u, sigma2, gamma_a, gamma_b, gamma


_EMPTY_INDEX = np.empty(0, dtype=np.int64)


# Kernel tests without a GPU: TRITON_INTERPRET=1 plus LOGKV_TRITON_CPU=1 runs the
# fused Triton paths on CPU tensors through the interpreter (slow; tests only).
_TRITON_CPU = os.environ.get("TRITON_INTERPRET") == "1" and os.environ.get("LOGKV_TRITON_CPU") == "1"


def _fused_on(x) -> bool:
    """Whether the fused Triton kernels serve this tensor or device."""
    device = x if isinstance(x, torch.device) else x.device
    return device.type == "cuda" or (_TRITON_CPU and device.type == "cpu")


@lru_cache(maxsize=1)
def _triton_updates():
    try:
        from models._vendor.semanticlogkv.litgpt import log_kv_updates_triton
        return log_kv_updates_triton
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _triton_route():
    try:
        from models._vendor.semanticlogkv.litgpt import log_kv_route_triton
        return log_kv_route_triton
    except ImportError:
        return None


def _spans_to_index_array(spans: list[tuple[int, int]]) -> np.ndarray:
    """Concatenate half-open ``[start, stop)`` ranges into one int64 array."""
    if len(spans) == 0:
        return _EMPTY_INDEX
    if len(spans) == 1:
        start, stop = spans[0]
        return np.arange(start, stop, dtype=np.int64) if stop > start else _EMPTY_INDEX
    # One repeat/arange instead of an arange per span: flushes build thousands.
    bounds = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    lengths = np.maximum(bounds[:, 1] - bounds[:, 0], 0)
    total = int(lengths.sum())
    if not total:
        return _EMPTY_INDEX
    return np.repeat(bounds[:, 0] - (np.cumsum(lengths) - lengths), lengths) + np.arange(total, dtype=np.int64)


def _ranges_to_index_array(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """`_spans_to_index_array` for array-valued ranges; empty ranges allowed."""
    lengths = np.maximum(np.asarray(lengths, dtype=np.int64), 0)
    total = int(lengths.sum())
    if not total:
        return _EMPTY_INDEX
    starts = np.asarray(starts, dtype=np.int64)
    return np.repeat(starts - (np.cumsum(lengths) - lengths), lengths) + np.arange(total, dtype=np.int64)


def _upload_indices(device: torch.device, arrays) -> list[torch.Tensor]:
    """Several int64 index arrays in one upload, returned as device slices."""
    sizes = [int(a.size) for a in arrays]
    if not sum(sizes):
        empty = torch.empty(0, dtype=torch.long, device=device)
        return [empty] * len(sizes)
    flat = _upload(np.concatenate(arrays).astype(np.int64, copy=False) if len(arrays) > 1
                   else np.asarray(arrays[0], dtype=np.int64), device)
    return list(flat.split(sizes))


def _upload(values, device) -> torch.Tensor:
    """Host metadata to `device` without a hidden stream synchronization.

    A blocking copy from pageable memory is `memcpy_and_sync`: the host waits
    for every queued kernel, so host scheduling cannot overlap device work.
    Pinned staging makes the copy asynchronous; the caching host allocator
    keeps the staging block until the copy's stream event completes, and the
    caller's array may be reused at once. CPU devices alias as before.
    """
    tensor = values if isinstance(values, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(values))
    device = torch.device(device)
    if device.type != "cuda" or not tensor.numel():
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


_PLAIN_SCALARS = (int, float, bool, tuple, type(None))
_NUMPY_DTYPES = {torch.bool: np.bool_, torch.int16: np.int16, torch.int32: np.int32, torch.int64: np.int64}


def _copy_host(value):
    """`deepcopy` for nested lists of plain scalars (the host mirrors).

    Replay records copy several such mirrors per flush; deepcopy's memo and
    dispatch dominated those copies. Anything else still uses deepcopy.
    """
    if type(value) is list:
        if value and type(value[0]) is list:
            return [_copy_host(x) for x in value]
        if all(type(x) in _PLAIN_SCALARS for x in value):
            return value[:]
    return deepcopy(value)


_UNIFIED_INF_KEY = 0x7F800000 << 32


def _unified_pack(cost: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """(cost bits << 32) | key; mirrors the Triton packing, including +0.0."""
    cost = torch.where(cost == 0, torch.zeros_like(cost), cost).contiguous()
    return (cost.view(torch.int32).to(torch.int64) << 32) | key


def _unified_pair_cost(d2, mi, mj, ri, rj, limit, valid):
    """Ward cost and validity with the Triton operation order."""
    denom = (mi + mj).clamp_min(1)
    valid = valid & (mi > 0) & (mj > 0)
    if limit is not None:
        distance = d2.sqrt()
        bound = torch.maximum(ri + (mj / denom) * distance, rj + (mi / denom) * distance)
        valid = valid & (bound <= limit)
    return (d2 * (mi * mj / denom)).masked_fill(~valid, float("inf")), valid


class _UnifiedReduceTorch:
    """Reference/fallback for incremental rounds; same state transitions as Triton.

    `dist` holds squared centroid distances. A merged row is updated with the
    Lance-Williams identity instead of recomputing the whole Gram matrix; only
    rows whose previous partner merged are rescanned, and the merged rows push
    their costs into the others. Pairwise costs of untouched clusters never
    change, so this reproduces a full recompute up to floating-point rounding
    (and XOR ties, whose indices are never compacted here).

    With passes > 1 a round also runs frozen-center sweeps: rows whose
    restricted partner was matched rescan unmatched columns only, then the
    unmatched rows pair again; every pair of the round is budgeted by cost.
    """

    def __init__(self, dist, mu, mass, count, limits, target, passes=1):
        lanes, size = mass.shape
        self.dist, self.mu, self.mass, self.limits, self.target = dist, mu, mass, limits, int(target)
        self.passes = max(1, int(passes))
        self.ids = torch.arange(size, device=mass.device)
        self.alive = self.ids[None, :] < count[:, None]
        self.radius = torch.zeros_like(mass)
        self.count = count.clone()
        self.stuck = torch.zeros(lanes, dtype=torch.bool, device=mass.device)
        self.traces = []
        self.packed = torch.full((lanes, size), _UNIFIED_INF_KEY, dtype=torch.int64, device=mass.device)
        lane, row = self.alive.nonzero(as_tuple=True)
        self.packed[lane, row] = self._scan(lane, row)

    def _scan(self, lanes, rows, excluded=None):
        """Packed row minima; `excluded` columns (matched this round) get no mass."""
        size = self.mass.size(1)
        out = torch.empty(rows.numel(), dtype=torch.int64, device=rows.device)
        chunk = max(1, (1 << 22) // max(size, 1))  # Bound [rows, M] temporaries.
        for start in range(0, rows.numel(), chunk):
            lane, row = lanes[start:start + chunk], rows[start:start + chunk]
            limited = self.limits is not None
            mass = self.mass[lane] if excluded is None else self.mass[lane].masked_fill(excluded[lane], 0)
            cost, _ = _unified_pair_cost(
                self.dist[lane, row], self.mass[lane, row][:, None], mass,
                self.radius[lane, row][:, None] if limited else None, self.radius[lane] if limited else None,
                self.limits[lane][:, None] if limited else None, self.ids[None, :] != row[:, None],
            )
            out[start:start + chunk] = _unified_pack(cost, row[:, None] ^ self.ids[None, :]).amin(1)
        return out

    def step(self, round_):
        size = self.mass.size(1)
        ids = self.ids
        best = ((self.packed & 0xFFFFFFFF) ^ ids).clamp_(0, size - 1)
        cost = (self.packed >> 32).to(torch.int32).view(torch.float32)
        valid = self.alive & (ids < best) & (best.gather(1, best) == ids) & (cost < float("inf"))
        nprop = valid.sum(1)
        rejected = torch.zeros_like(self.alive)
        bound = None
        if self.limits is not None:
            # Exact differences: GEMM cancellation must not widen a candidate.
            lane, row = valid.nonzero(as_tuple=True)
            other = best[lane, row]
            diff = self.mu[lane, row] - self.mu[lane, other]
            exact2 = (diff * diff).sum(-1)
            exact = exact2.sqrt()
            ma, mb = self.mass[lane, row], self.mass[lane, other]
            denom = (ma + mb).clamp_min(1)
            verified = torch.maximum(self.radius[lane, row] + (mb / denom) * exact,
                                     self.radius[lane, other] + (ma / denom) * exact)
            bad = verified > self.limits[lane]
            bl, br, bo = lane[bad], row[bad], other[bad]
            # The exact entry makes the rescan reproduce this rejection.
            self.dist[bl, br, bo] = exact2[bad]
            self.dist[bl, bo, br] = exact2[bad]
            rejected[bl, br] = True
            rejected[bl, bo] = True
            valid[bl, br] = False
            bound = torch.zeros_like(self.mass)
            bound[lane, row] = verified
        pair_cost = cost.masked_fill(~valid, float("inf"))
        mate = best
        if self.passes > 1:
            mate, pair_cost, extra = self._sweeps(valid, best, pair_cost)
            nprop = nprop + extra
        order = pair_cost.argsort(dim=1, stable=True)[:, :size // 2]
        ok = pair_cost.gather(1, order) < float("inf")
        ok &= ok.cumsum(1) <= (self.count - self.target)[:, None]
        lane, rank = ok.nonzero(as_tuple=True)
        a = order[lane, rank]
        b = mate[lane, a]
        keep = torch.zeros_like(self.alive)
        merged = torch.zeros_like(self.alive)
        if a.numel():
            self._merge(lane, a, b, bound, keep, merged)
        dirty = self.alive & ~keep & (merged.gather(1, best) | rejected)
        lane, row = dirty.nonzero(as_tuple=True)
        if row.numel():
            self.packed[lane, row] = self._scan(lane, row)
        self.count -= ok.sum(1)
        remaining = self.count > self.target
        self.stuck |= remaining & (nprop == 0)
        return remaining & (nprop > 0)

    def _sweeps(self, valid, best, pair_cost):
        """Frozen-center sweeps after the first mutual pairs (true minima untouched)."""
        size = self.mass.size(1)
        ids = self.ids
        matched = torch.zeros_like(self.alive)
        lane, row = valid.nonzero(as_tuple=True)
        matched[lane, row] = True
        matched[lane, best[lane, row]] = True
        restricted = self.packed.clone()
        mate, extra = best.clone(), torch.zeros_like(self.count)
        for _ in range(self.passes - 1):
            partner = ((restricted & 0xFFFFFFFF) ^ ids).clamp_(0, size - 1)
            lane, row = (self.alive & ~matched & matched.gather(1, partner)).nonzero(as_tuple=True)
            if row.numel():
                restricted[lane, row] = self._scan(lane, row, excluded=matched)
            partner = ((restricted & 0xFFFFFFFF) ^ ids).clamp_(0, size - 1)
            cost = (restricted >> 32).to(torch.int32).view(torch.float32)
            more = (self.alive & ~matched & (ids < partner) & (partner.gather(1, partner) == ids)
                    & (cost < float("inf")) & ~matched.gather(1, partner))
            extra += more.sum(1)
            lane, row = more.nonzero(as_tuple=True)
            matched[lane, row] = True
            matched[lane, partner[lane, row]] = True
            mate = torch.where(more, partner, mate)
            pair_cost = torch.where(more, cost, pair_cost)
        return mate, pair_cost, extra

    def _merge(self, lane, a, b, bound, keep, merged):
        size = self.mass.size(1)
        ids = self.ids
        ma, mb = self.mass[lane, a], self.mass[lane, b]
        dab = self.dist[lane, a, b]
        total = ma + mb
        alpha, beta = ma / total, mb / total
        shift = alpha * beta * dab
        keep[lane, a] = True
        merged[lane, a] = True
        merged[lane, b] = True
        partner = torch.zeros_like(self.packed)
        partner[lane, a] = b
        pma, pmb, pdab = torch.ones_like(self.mass), torch.zeros_like(self.mass), torch.zeros_like(self.mass)
        pma[lane, a], pmb[lane, a], pdab[lane, a] = ma, mb, dab
        self.mass[lane, a] = total
        self.mass[lane, b] = 0
        self.alive[lane, b] = False
        if bound is not None:
            self.radius[lane, a] = bound[lane, a]
        self.mu[lane, a] = (ma[:, None] * self.mu[lane, a] + mb[:, None] * self.mu[lane, b]) / total.clamp_min(1)[:, None]
        rows = lane[:, None]
        alpha, beta, shift = alpha[:, None], beta[:, None], shift[:, None]
        x_aa, x_ba = self.dist[lane, a], self.dist[lane, b]
        new = alpha * x_aa + beta * x_ba - shift
        is_keep = keep[lane]
        q = partner[lane]
        x_ab, x_bb = self.dist[rows, a[:, None], q], self.dist[rows, b[:, None], q]
        qa, qb, qd = pma[lane], pmb[lane], pdab[lane]
        qalpha, qbeta = qa / (qa + qb), qb / (qa + qb)
        qshift = qalpha * qbeta * qd
        # Both merged: lower keep first, so both owners write identical bits.
        own_first = qalpha * new + qbeta * (alpha * x_ab + beta * x_bb - shift) - qshift
        other_first = alpha * (qalpha * x_aa + qbeta * x_ab - qshift) + beta * (qalpha * x_ba + qbeta * x_bb - qshift) - shift
        both = torch.where(a[:, None] < ids[None, :], own_first, other_first)
        new = torch.where(is_keep, both, new).clamp_min_(0)
        new[torch.arange(a.numel(), device=a.device), a] = 0
        alive = self.alive[lane]
        pi, ci = (alive & ~is_keep).nonzero(as_tuple=True)
        self.dist[lane[pi], ci, a[pi]] = new[pi, ci]
        pi, ci = alive.nonzero(as_tuple=True)
        self.dist[lane[pi], a[pi], ci] = new[pi, ci]
        limited = self.limits is not None
        cost, valid = _unified_pair_cost(
            new, total[:, None], self.mass[lane], self.radius[lane, a][:, None] if limited else None,
            self.radius[lane] if limited else None, self.limits[lane][:, None] if limited else None,
            alive & (ids[None, :] != a[:, None]),
        )
        packed = _unified_pack(cost, ids[None, :] ^ a[:, None])
        pi, ci = (valid & ~is_keep).nonzero(as_tuple=True)
        self.packed.view(-1).scatter_reduce_(0, lane[pi] * size + ci, packed[pi, ci], reduce="amin")
        self.packed[lane, a] = packed.amin(1)
        self.traces.append(torch.stack((lane, a, b), 1))

    def finish(self):
        lanes = self.mass.size(0)
        rows = (torch.cat(self.traces) if self.traces else self.packed.new_zeros((0, 3))).cpu().numpy()
        # Traces were appended round by round, lane-major within a round.
        order = np.argsort(rows[:, 0], kind="stable")
        rows = rows[order]
        splits = np.searchsorted(rows[:, 0], np.arange(1, lanes))
        traces = [part[:, 1:] for part in np.split(rows, splits)]
        return traces, self.alive.cpu().numpy(), self.stuck.cpu().numpy()


class LogStructuredKVCache(nn.Module):
    """Log-Structured KV Cache storing full post-RoPE keys in merged slots.

    Args:
        k_shape: (batch_size, n_groups, max_seq_length, k_dim) — k_dim is the
            FULL post-RoPE key width (RoPE'd position channel + content channel).
            max_seq_length only sizes the level hierarchy and the append-only
            token counter; no O(N) buffer is allocated.
        v_shape: (batch_size, n_groups, max_seq_length, v_dim)
        B: slots per level (default 512)
        recent_size: sliding window size (default 0 = 2)
        device / dtype: torch device / dtype
        importance_pooling: if True, merges are weighted by a per-token
            importance mass (tracked separately from the token-count weight
            ``w``, which keeps driving the ``log(w)`` mass bias unchanged)
            instead of a uniform mean. See ``compact()``/``_compact_tokens()``.
        importance_pooling_lambda: only consulted when ``importance_pooling``
            is True. Linearly blends the importance-driven pooling share
            toward the uniform/count-driven share — ``1.0`` (default) is pure
            importance pooling, ``0.0`` is numerically equivalent to plain
            uniform pooling, values in between interpolate. Added after the
            first decisive eval (2026-08-14) showed importance pooling
            improves ACC/LongBench but regresses niah — the working
            hypothesis is that the pure key-norm heuristic over-weights
            high-norm non-needle tokens, and a partial blend may recover
            niah while keeping most of the LongBench gain (see CLAUDE.md 6.5).
        importance_pooling_temperature: only consulted when
            ``importance_pooling`` is True. Exponent applied to the raw
            per-token importance heuristic (key L2 norm) before it is
            normalized into pooling weights: ``imp = raw_imp ** temperature``.
            ``1.0`` (default) is the original heuristic, unchanged (bit-
            identical short-circuit). Values ``< 1.0`` compress the dynamic
            range, damping outlier tokens (e.g. attention-sink-style high-
            norm tokens) relative to the bulk, without fully discarding the
            salience signal the way blending the whole distribution toward
            uniform (``importance_pooling_lambda``) does; ``temperature -> 0``
            approaches uniform in the limit, ``> 1.0`` sharpens further.
            Orthogonal to ``importance_pooling_lambda`` — both may be set at
            once (temperature reshapes first, lambda blends the result).
    """

    def __init__(
        self,
        k_shape: tuple[int, int, int, int],
        v_shape: tuple[int, int, int, int],
        B: int = 512,
        recent_size: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        importance_pooling: bool = False,
        importance_pooling_lambda: float = 1.0,
        importance_pooling_temperature: float = 1.0,
        semantic_clusters: bool = False,
        cluster_k_max: int = 1,
        cluster_lambda_rel: float = 1.0,
        seg_eta: float = 1.0,
        seg_g0: float = 2048.0,
        seg_gap_max: float | None = None,
        seg_block_level: int = 0,
        seg_forget: float = 0.5,
        semantic_s_h: torch.Tensor | float | None = None,
        semantic_flush_granularity: int = 2,
        semantic_cluster_chunk_size: int = 0,
        semantic_capacity_beta: float = 0.0,
        semantic_capacity_hard_cap_mult: float = 0.0,
        semantic_legacy_route: bool = False,
        semantic_unified_route: bool = False,
        semantic_merge_passes: int = 1,
        semantic_anchor_mode: str = "multi",
        semantic_pack_backend: str = "auto",
        semantic_centroid_backend: str = "sequential",
        semantic_replay_updates: bool = False,
        cos_cache: torch.Tensor | None = None,
        sin_cache: torch.Tensor | None = None,
        rope_n_elem: int | None = None,
        allocate_second_order: bool = True,
        alpha_exact_tokens: int = 0,
        alpha_span_max_tokens: int = 64,
        beta_novelty: bool = False,
        beta_adaptive_merge: bool = False,
    ) -> None:
        super().__init__()

        batch_size, n_groups, max_seq_length, k_dim = k_shape
        _, _, _, v_dim = v_shape

        self.batch_size = batch_size
        self.n_groups = n_groups
        self.max_seq_length = max_seq_length
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.B = B
        self.allocate_second_order = bool(allocate_second_order)
        self.semantic_clusters = bool(semantic_clusters)
        self.alpha_exact_tokens = int(alpha_exact_tokens)
        self.alpha_span_max_tokens = int(alpha_span_max_tokens)
        self.beta_novelty = bool(beta_novelty)
        self.beta_adaptive_merge = bool(beta_adaptive_merge)
        self.alpha_count = 0
        if (self.beta_novelty or self.beta_adaptive_merge) and not self.alpha_exact_tokens:
            raise ValueError("BetaLogKV requires the Alpha exact-span cache (alpha_exact_tokens > 0)")
        if self.alpha_exact_tokens < 0:
            raise ValueError("alpha_exact_tokens must be >= 0")
        if self.alpha_exact_tokens:
            if not (semantic_clusters and semantic_anchor_mode == "mid"
                    and not allocate_second_order and cluster_k_max > 1):
                raise ValueError("AlphaLogKV requires semantic clusters, K > 1, mid anchors and no second-order allocation")
            if semantic_legacy_route or semantic_cluster_chunk_size:
                raise ValueError("AlphaLogKV archives through the attach or unified route; legacy/chunk-tree routing is not supported")
            if (seg_gap_max is not None and math.isfinite(seg_gap_max)) or seg_block_level:
                raise ValueError("AlphaLogKV requires disabled segment gaps/padding for delayed archival")
            if not 1 <= self.alpha_span_max_tokens <= self.alpha_exact_tokens:
                raise ValueError("alpha_span_max_tokens must be in [1, alpha_exact_tokens]")
        if semantic_anchor_mode not in ("mid", "multi"):
            raise ValueError("semantic_anchor_mode must be 'mid' or 'multi'")
        if semantic_pack_backend not in ("auto", "torch", "triton"):
            raise ValueError("semantic_pack_backend must be 'auto', 'torch', or 'triton'")
        self.semantic_anchor_mode = semantic_anchor_mode
        self.semantic_pack_backend = semantic_pack_backend
        self._mid_decode_state = None
        self.K_max = int(cluster_k_max) if self.semantic_clusters else 1
        if self.K_max < 1:
            raise ValueError(f"cluster_k_max must be >= 1, got {cluster_k_max}")
        if self.semantic_clusters and self.B < 2:
            raise ValueError(f"semantic LogKV requires B >= 2, got {self.B}")
        if semantic_centroid_backend not in ("sequential", "parallel"):
            raise ValueError("semantic_centroid_backend must be sequential or parallel")
        self.semantic_centroid_backend = semantic_centroid_backend
        self.semantic_replay_updates = bool(semantic_replay_updates)
        self.semantic_unified_route = bool(semantic_unified_route)
        self.semantic_merge_passes = int(semantic_merge_passes)
        if self.semantic_merge_passes < 1:
            raise ValueError("semantic_merge_passes must be >= 1")
        if self.semantic_unified_route:
            if not self.semantic_clusters or semantic_legacy_route or semantic_cluster_chunk_size:
                raise ValueError("unified routing requires semantic clusters and excludes legacy/chunk-tree routing")
            if semantic_capacity_beta != 0.0:
                raise ValueError("unified routing uses Ward costs; semantic_capacity_beta must be 0")
            if seg_gap_max is not None and math.isfinite(seg_gap_max) and seg_forget != 1.0:
                raise ValueError("unified Ward routing requires seg_forget=1 when segment boundaries are enabled")
        if self.semantic_replay_updates:
            if not self.semantic_clusters or self.K_max <= 1 or semantic_legacy_route or semantic_cluster_chunk_size:
                raise ValueError("update replay requires fast or unified semantic routing with K > 1")
        self._active_updates = None
        self._update_actions = None
        self._replaying_updates = False
        self.recent_size = recent_size if recent_size > 0 else 2
        # Explicit raise (not assert): must survive `python -O`.
        if self.recent_size < 2:
            raise ValueError(f"recent_size ({self.recent_size}) must be >= 2")
        if self.semantic_clusters and importance_pooling:
            raise ValueError("semantic LogKV rejects importance_pooling=True")
        self.importance_pooling = bool(importance_pooling)
        self.importance_pooling_lambda = float(importance_pooling_lambda)
        if not 0.0 <= self.importance_pooling_lambda <= 1.0:
            raise ValueError(
                f"importance_pooling_lambda must be in [0, 1], got {self.importance_pooling_lambda}"
            )
        self.importance_pooling_temperature = float(importance_pooling_temperature)
        if self.importance_pooling_temperature <= 0.0:
            raise ValueError(
                f"importance_pooling_temperature must be > 0, got {self.importance_pooling_temperature}"
            )

        if self.semantic_clusters:
            if cos_cache is None or sin_cache is None or rope_n_elem is None:
                raise ValueError("semantic LogKV requires cos_cache, sin_cache, and rope_n_elem")
            if cos_cache.dim() != 2 or sin_cache.dim() != 2 or cos_cache.shape != sin_cache.shape:
                raise ValueError(
                    f"semantic LogKV requires 2-D matching RoPE caches, got {cos_cache.shape} and {sin_cache.shape}"
                )
            self.cluster_lambda_rel = float(cluster_lambda_rel)
            if not (math.isfinite(self.cluster_lambda_rel) and self.cluster_lambda_rel > 0.0):
                raise ValueError(f"cluster_lambda_rel must be finite and > 0, got {cluster_lambda_rel}")
            self.seg_eta = float(seg_eta)
            self.seg_g0 = float(seg_g0)
            if not (math.isfinite(self.seg_g0) and self.seg_g0 > 0.0):
                raise ValueError(f"seg_g0 must be finite and > 0, got {seg_g0}")
            self.seg_gap_max = math.inf if seg_gap_max is None else float(seg_gap_max)
            if not (self.seg_gap_max == math.inf or (math.isfinite(self.seg_gap_max) and self.seg_gap_max >= 0.0)):
                raise ValueError(f"seg_gap_max must be finite >= 0 or inf, got {seg_gap_max}")
            self.seg_block_level = int(seg_block_level)
            if self.seg_block_level not in (0, 1, 2):
                raise ValueError(f"seg_block_level must be one of {{0,1,2}}, got {seg_block_level}")
            self.seg_forget = float(seg_forget)
            if not 0.0 <= self.seg_forget <= 1.0:
                raise ValueError(f"seg_forget must be in [0,1], got {seg_forget}")
            self.semantic_flush_granularity = int(semantic_flush_granularity)
            if not 1 <= self.semantic_flush_granularity <= self.recent_size:
                raise ValueError(
                    "semantic_flush_granularity must be in [1, recent_size], got "
                    f"{semantic_flush_granularity} for recent_size={self.recent_size}"
                )
            self.semantic_cluster_chunk_size = int(semantic_cluster_chunk_size)
            if not 0 <= self.semantic_cluster_chunk_size <= self.recent_size:
                raise ValueError(
                    "semantic_cluster_chunk_size must be 0 or in [1, recent_size], got "
                    f"{semantic_cluster_chunk_size} for recent_size={self.recent_size}"
                )
            self.semantic_capacity_beta = float(semantic_capacity_beta)
            if not (math.isfinite(self.semantic_capacity_beta) and self.semantic_capacity_beta >= 0.0):
                raise ValueError(f"semantic_capacity_beta must be finite >= 0, got {semantic_capacity_beta}")
            self.semantic_legacy_route = bool(semantic_legacy_route)
            self.semantic_capacity_hard_cap_mult = float(semantic_capacity_hard_cap_mult)
            if not (
                math.isfinite(self.semantic_capacity_hard_cap_mult)
                and self.semantic_capacity_hard_cap_mult >= 0.0
            ):
                raise ValueError(
                    "semantic_capacity_hard_cap_mult must be finite >= 0, got "
                    f"{semantic_capacity_hard_cap_mult}"
                )
            self.rope_n_elem = int(rope_n_elem)
            self.register_buffer("cos_cache", cos_cache.to(device=device), persistent=False)
            self.register_buffer("sin_cache", sin_cache.to(device=device), persistent=False)
            self.slot_k_dim = int(cos_cache.size(1)) + k_dim - self.rope_n_elem
            # §5.12: K_max changes ladder budget. +2 is the v1 safety margin.
            self.max_levels = max(2, math.ceil(math.log2(max_seq_length / max(self.K_max * B, 1) + 1.0)) + 2)
        else:
            self.cluster_lambda_rel = 1.0
            self.seg_eta = 0.0
            self.seg_g0 = 2048.0
            self.seg_gap_max = math.inf
            self.seg_block_level = 0
            self.seg_forget = 1.0
            self.semantic_flush_granularity = 2
            self.semantic_cluster_chunk_size = 0
            self.semantic_capacity_beta = 0.0
            self.semantic_capacity_hard_cap_mult = 0.0
            self.semantic_legacy_route = False
            self.rope_n_elem = None
            self.cos_cache = None
            self.sin_cache = None
            self.slot_k_dim = k_dim
            denom = B * 2
            # +1 for the write level (level 0). The formula gives the number of carry
            # levels needed; total levels = carry + 1 write level.
            self.max_levels = max(2, math.ceil(math.log2(max((max_seq_length + 1) / denom, 1))) + 1)
        self.L_alloc = self.max_levels
        if self.alpha_exact_tokens:
            # Pay for exact K/V and metadata from the ladder, never from the
            # recent window (which would shorten the training/prefill blocks).
            # Include the maximum reusable packed decode workspace as well.
            element = torch.empty((), dtype=dtype).element_size()
            packed_dim = ((max(k_dim + 1, v_dim) + 7) // 8) * 8
            entry_bytes = (k_dim + v_dim + 2 * packed_dim) * element + 41
            exact_bytes = self.alpha_exact_tokens * (n_groups * (k_dim + v_dim + 2 * packed_dim) * element + 9)
            baseline = n_groups * self.K_max * self.L_alloc * (B * entry_bytes + 2)
            self.alpha_baseline_B = B
            for candidate in range(B - 1, 1, -1):
                levels = max(2, math.ceil(math.log2(max_seq_length / (self.K_max * candidate) + 1.0)) + 2)
                needed = n_groups * self.K_max * levels * (candidate * entry_bytes + 2) + exact_bytes
                if needed <= baseline:
                    B = self.B = candidate
                    self.L_alloc = self.max_levels = levels
                    self.alpha_budget_bytes = (batch_size * baseline, batch_size * needed)
                    break
            else:
                raise ValueError("compressed ladder budget is too small for alpha_exact_tokens; increase B or reduce exact capacity")
        self.B_prime = B
        self.recent_capacity = self.recent_size * 2 if self.semantic_clusters else self.recent_size

        # Total tokens ever committed (scalar bookkeeping only — replaces the
        # former Θ(N) per-token position-key buffer).
        self.token_count: int = 0

        # ---- Sliding window: last < recent_size tokens (exact keys + values) ----
        self.register_buffer(
            "recent_k",
            torch.zeros(batch_size, n_groups, self.recent_capacity, self.slot_k_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "recent_v",
            torch.zeros(batch_size, n_groups, self.recent_capacity, v_dim, device=device, dtype=dtype),
            persistent=False,
        )
        if self.semantic_clusters:
            self.register_buffer(
                "recent_k_raw",
                torch.zeros(batch_size, n_groups, self.recent_capacity, k_dim, device=device, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                "recent_pos",
                torch.zeros(batch_size, self.recent_capacity, device=device, dtype=torch.int64),
                persistent=False,
            )
            self._recent_pos_host: list[list[int]] = [
                [0] * self.recent_capacity for _ in range(batch_size)
            ]
        self.recent_count: int = 0
        if self.alpha_exact_tokens:
            P = self.alpha_exact_tokens
            self.register_buffer("alpha_k_raw", torch.zeros(batch_size, n_groups, P, k_dim, device=device, dtype=dtype), persistent=False)
            self.register_buffer("alpha_v", torch.zeros(batch_size, n_groups, P, v_dim, device=device, dtype=dtype), persistent=False)
            self.register_buffer("alpha_pos", torch.zeros(batch_size, P, device=device, dtype=torch.int64), persistent=False)
            self.register_buffer("alpha_valid", torch.zeros(batch_size, P, device=device, dtype=torch.bool), persistent=False)
            self._alpha_spans = [[] for _ in range(batch_size)]
            self._alpha_positions = [[] for _ in range(batch_size)]
            self._recent_span_ends = [[False] * self.recent_capacity for _ in range(batch_size)]
            self._UPDATE_DEVICE_FIELDS = (*self._UPDATE_DEVICE_FIELDS, "alpha_k_raw", "alpha_v", "alpha_pos", "alpha_valid")
            self._UPDATE_HOST_FIELDS = (*self._UPDATE_HOST_FIELDS, "_alpha_spans", "_alpha_positions", "alpha_count")

        # ---- Hierarchical entries: (B, G, K_max, L_alloc, B', ·) ----
        # K_max=1 is the current single-cluster path; the indexed layout is the
        # Stage 1 storage contract that later semantic routing will populate.
        self.register_buffer(
            "level_k",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, k_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "level_v",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, v_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "level_w",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, device=device, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "level_imp",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, device=device, dtype=torch.float32),
            persistent=False,
        )
        # Rank-1 key covariance:
        #   Sigma_s ~= sigma2_s * sigma_u_s sigma_u_s^T
        self.register_buffer(
            "level_sigma_u",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, k_dim, device=device, dtype=dtype) if self.allocate_second_order else None,
            persistent=False,
        )
        self.register_buffer(
            "level_sigma2",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, device=device, dtype=dtype) if self.allocate_second_order else None,
            persistent=False,
        )
        # Rank-1 value-key cross covariance:
        #   Gamma_s ~= gamma_s * gamma_b_s gamma_a_s^T
        # where gamma_a lives in key/query space and gamma_b in value space.
        self.register_buffer(
            "level_gamma_a",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, k_dim, device=device, dtype=dtype) if self.allocate_second_order else None,
            persistent=False,
        )
        self.register_buffer(
            "level_gamma_b",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, v_dim, device=device, dtype=dtype) if self.allocate_second_order else None,
            persistent=False,
        )
        self.register_buffer(
            "level_gamma",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, device=device, dtype=dtype) if self.allocate_second_order else None,
            persistent=False,
        )
        self.register_buffer(
            "level_count",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, dtype=torch.int16, device=device),
            persistent=False,
        )
        self.register_buffer(
            "pad_mask",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, dtype=torch.bool, device=device),
            persistent=False,
        )
        if not self.semantic_clusters:
            # Host-side mirror for the aligned single-cluster path. Reading a CUDA
            # count tensor with .item() in the chunk loop serializes every layer.
            self._counts: list[int] = [0] * self.L_alloc
        else:
            # Per-(batch, group, cluster, level) host mirror. These counts drive
            # Python control flow; reading the CUDA buffer with .item() in that
            # path serializes the stream.
            self._semantic_counts: list[list[list[list[int]]]] = [
                [[[0] * self.L_alloc for _ in range(self.K_max)] for _ in range(n_groups)]
                for _ in range(batch_size)
            ]
            self._semantic_alive: list[list[list[bool]]] = [
                [[False] * self.K_max for _ in range(n_groups)]
                for _ in range(batch_size)
            ]
            self._semantic_p_hi_c: list[list[list[int]]] = [
                [[-1] * self.K_max for _ in range(n_groups)]
                for _ in range(batch_size)
            ]
            self._semantic_current_segment: list[list[list[int]]] = [
                [[0] * self.K_max for _ in range(n_groups)]
                for _ in range(batch_size)
            ]
            self._semantic_level0_phase: list[list[list[int]]] = [
                [[0] * self.K_max for _ in range(n_groups)]
                for _ in range(batch_size)
            ]
            self._semantic_n_total: list[list[list[int]]] = [
                [[0] * self.K_max for _ in range(n_groups)]
                for _ in range(batch_size)
            ]
            self._semantic_ward_dirty: list[list[bool]] = [
                [True] * n_groups for _ in range(batch_size)
            ]
        if self.semantic_clusters:
            i64max = torch.iinfo(torch.int64).max
            self.register_buffer(
                "level_p_lo",
                torch.full((batch_size, n_groups, self.K_max, self.L_alloc, B), i64max, dtype=torch.int64, device=device),
                persistent=False,
            )
            self.register_buffer(
                "level_p_hi",
                torch.full((batch_size, n_groups, self.K_max, self.L_alloc, B), -1, dtype=torch.int64, device=device),
                persistent=False,
            )
            self.register_buffer(
                "level_sum_wp",
                torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, dtype=torch.int64, device=device),
                persistent=False,
            )
            self.register_buffer(
                "level_order",
                torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, dtype=torch.int64, device=device),
                persistent=False,
            )
            self.register_buffer(
                "centroid",
                torch.zeros(batch_size, n_groups, self.K_max, k_dim, device=device, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "n_eff",
                torch.zeros(batch_size, n_groups, self.K_max, device=device, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "n_total",
                torch.zeros(batch_size, n_groups, self.K_max, device=device, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "p_hi_c",
                torch.full((batch_size, n_groups, self.K_max), -1, device=device, dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "current_segment",
                torch.zeros(batch_size, n_groups, self.K_max, device=device, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "level0_phase",
                torch.zeros(batch_size, n_groups, self.K_max, device=device, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "alive",
                torch.zeros(batch_size, n_groups, self.K_max, device=device, dtype=torch.bool),
                persistent=False,
            )
            self.register_buffer(
                "ward_cost",
                torch.full(
                    (batch_size, n_groups, self.K_max, self.K_max),
                    float("inf"),
                    device=device,
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            if semantic_s_h is None:
                s_h = torch.ones(batch_size, n_groups, device=device, dtype=torch.float32)
            elif isinstance(semantic_s_h, torch.Tensor):
                s_h = semantic_s_h.to(device=device, dtype=torch.float32)
                if s_h.shape == (n_groups,):
                    s_h = s_h.unsqueeze(0).expand(batch_size, -1).contiguous()
                elif s_h.shape != (batch_size, n_groups):
                    raise ValueError(f"semantic_s_h must have shape ({n_groups},) or ({batch_size},{n_groups}), got {tuple(s_h.shape)}")
            else:
                s_h = torch.full((batch_size, n_groups), float(semantic_s_h), device=device, dtype=torch.float32)
            if not bool(torch.isfinite(s_h).all().item()) or not bool((s_h > 0).all().item()):
                raise ValueError("semantic_s_h must be finite and > 0")
            self.register_buffer("s_h", s_h, persistent=False)
            self._semantic_s_h_host: list[list[float]] = s_h.detach().cpu().tolist()
            self.op_log: torch.Tensor | None = None
            self.op_log_len: torch.Tensor | None = None

        # When False, compaction skips the rank-1 second-order statistics
        # entirely (zero-filled if allocated, otherwise absent) and behaves like the first-order
        # cache. Driven by the second-order gate: a scale of 0 makes every
        # correction term vanish, so computing and merging the statistics that
        # feed it is pure overhead. See ``log_kv_chunk_attention``. Guarded by
        # the ``second_order`` property below — set the backing field directly
        # here since a fresh cache has nothing for the guard to protect.
        self._second_order: bool = self.allocate_second_order

    # ------------------------------------------------------------------
    # Second-order gate (guarded: cannot change regime on a live cache)
    # ------------------------------------------------------------------

    @property
    def second_order(self) -> bool:
        """Whether compaction builds and merges the rank-1 Sigma/Gamma stats.

        Guarded rather than a plain flag: flipping it while a level already
        holds compacted entries would leave the hierarchy straddling two
        regimes, silently.

        - True -> False mid-stream: the next binary carry takes the no-stats
          branch of ``compact()``, which ignores whatever stats the existing
          level already had and produces a merged entry with none — those
          real, previously-built stats are gone.
        - False -> True mid-stream: the next carry merges genuine new stats
          against a sibling that has structural zero stats (built while this
          was False) via Chan's formula. That reads as "this half of the slot
          had exactly zero variance", not "unknown" — an understated, wrong
          covariance for the merged slot, not a conservative one.

        Both keep producing finite numbers, so nothing downstream would flag
        it. Call ``reset_parameters()`` before switching regimes on a cache
        that has ever run ``add_recent()``; a fresh cache and same-value writes
        are always fine.
        """
        return self._second_order

    @second_order.setter
    def second_order(self, value: bool) -> None:
        value = bool(value)
        if value != self._second_order and self._has_compacted_levels():
            raise RuntimeError(
                f"LogStructuredKVCache: cannot change `second_order` "
                f"({self._second_order} -> {value}) while a compacted level "
                "already holds entries. Call reset_parameters() first -- see "
                "the `second_order` property docstring for why switching "
                "regimes in place is unsafe."
            )
        if value and not self.allocate_second_order:
            raise RuntimeError("Second-order storage is disabled; rebuild the cache with allocate_second_order=True")
        self._second_order = value

    # ------------------------------------------------------------------
    # Level accessors
    # ------------------------------------------------------------------

    def _get_level(self, ell: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.level_k[:, :, 0, ell],
            self.level_v[:, :, 0, ell],
            self.level_w[:, :, 0, ell],
        )

    def _get_level_stats(
        self, ell: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.level_sigma_u[:, :, 0, ell],
            self.level_sigma2[:, :, 0, ell],
            self.level_gamma_a[:, :, 0, ell],
            self.level_gamma_b[:, :, 0, ell],
            self.level_gamma[:, :, 0, ell],
        )

    def _get_level_imp(self, ell: int) -> torch.Tensor:
        return self.level_imp[:, :, 0, ell]

    def _level_count(self, ell: int) -> int:
        if hasattr(self, "_counts"):
            return self._counts[ell]
        return int(self.level_count[0, 0, 0, ell].item())

    def _has_compacted_levels(self) -> bool:
        if hasattr(self, "_counts"):
            return any(c > 0 for c in self._counts)
        if hasattr(self, "_semantic_counts"):
            return any(
                count > 0
                for b_counts in self._semantic_counts
                for g_counts in b_counts
                for c_counts in g_counts
                for count in c_counts
            )
        return bool((self.level_count > 0).any().item())

    def _semantic_level_count(self, b: int, g: int, c: int, ell: int) -> int:
        return self._semantic_counts[b][g][c][ell]

    # Host mirrors are authoritative during a route/replay call. Writing the
    # device twin per (lane, level) costs one kernel launch each -- hundreds per
    # flush -- so inside `_semantic_deferred_scalars()` the twins are marked
    # dirty and rebuilt with one H2D copy apiece on exit.
    _SEMANTIC_SCALAR_MIRRORS = (
        ("level_count", "_semantic_counts", torch.int16),
        ("alive", "_semantic_alive", torch.bool),
        ("p_hi_c", "_semantic_p_hi_c", torch.int64),
        ("current_segment", "_semantic_current_segment", torch.int32),
        ("level0_phase", "_semantic_level0_phase", torch.int32),
        ("n_total", "_semantic_n_total", torch.int32),
    )

    @contextlib.contextmanager
    def _semantic_deferred_scalars(self):
        outer = getattr(self, "_semantic_defer_scalars", False)
        self._semantic_defer_scalars = True
        try:
            yield
        finally:
            self._semantic_defer_scalars = outer
            if not outer:
                self._semantic_sync_device_scalars()
                self._flush_recorded_ops()

    def _semantic_sync_device_scalars(self) -> None:
        dirty = getattr(self, "_semantic_scalar_dirty", None)
        if not dirty:
            return
        for name, mirror, dtype in self._SEMANTIC_SCALAR_MIRRORS:
            if name not in dirty:
                continue
            buf = getattr(self, name)
            # NumPy converts the nested mirror lists far faster than torch.tensor,
            # straight to the buffer dtype; a pinned source keeps the copy async.
            host = torch.from_numpy(np.asarray(getattr(self, mirror), dtype=_NUMPY_DTYPES[dtype]))
            buf.copy_(host.pin_memory() if buf.is_cuda else host, non_blocking=buf.is_cuda)
        dirty.clear()

    def _semantic_mark_scalar_dirty(self, name: str) -> bool:
        """True when the device write should be skipped (deferred)."""
        if not getattr(self, "_semantic_defer_scalars", False):
            return False
        dirty = getattr(self, "_semantic_scalar_dirty", None)
        if dirty is None:
            dirty = self._semantic_scalar_dirty = set()
        dirty.add(name)
        return True

    def _set_semantic_level_count(self, b: int, g: int, c: int, ell: int, count: int) -> None:
        count = int(count)
        if not self._semantic_mark_scalar_dirty("level_count"):
            self.level_count[b, g, c, ell] = count
        self._semantic_counts[b][g][c][ell] = count

    def _set_semantic_level_counts(self, lanes, rows, ell: int, counts) -> None:
        """Batched `_set_semantic_level_count` for `lanes[rows]`; one dirty mark when deferred."""
        if not len(rows):
            return
        if self._semantic_mark_scalar_dirty("level_count"):
            mirror = self._semantic_counts
            for i, count in zip(rows.tolist(), counts.tolist()):
                b, g, c = lanes[i]
                mirror[b][g][c][ell] = count
            return
        for i, count in zip(rows.tolist(), counts.tolist()):
            self._set_semantic_level_count(*lanes[i], ell, count)

    def _clear_semantic_cluster_counts(self, b: int, g: int, c: int) -> None:
        if not self._semantic_mark_scalar_dirty("level_count"):
            self.level_count[b, g, c].zero_()
        self._semantic_counts[b][g][c][:] = [0] * self.L_alloc

    def _set_semantic_alive(self, b: int, g: int, c: int, alive: bool) -> None:
        alive = bool(alive)
        if not self._semantic_mark_scalar_dirty("alive"):
            self.alive[b, g, c] = alive
        self._semantic_alive[b][g][c] = alive

    _SEMANTIC_SCALAR_FIELDS = {
        "alive": ("_semantic_alive", bool),
        "p_hi_c": ("_semantic_p_hi_c", int),
        "current_segment": ("_semantic_current_segment", int),
        "level0_phase": ("_semantic_level0_phase", int),
        "n_total": ("_semantic_n_total", int),
    }

    def _set_semantic_scalars(self, name: str, lanes, values) -> None:
        """Bulk `_set_semantic_<name>` for many (b, g, c); one dirty mark when deferred."""
        mirror, cast = self._SEMANTIC_SCALAR_FIELDS[name]
        mirror = getattr(self, mirror)
        if self._semantic_mark_scalar_dirty(name):
            for (b, g, c), value in zip(lanes, values):
                mirror[b][g][c] = cast(value)
            return
        device = getattr(self, name)
        for (b, g, c), value in zip(lanes, values):
            value = cast(value)
            device[b, g, c] = value
            mirror[b][g][c] = value

    def _semantic_live_clusters(self, b: int, g: int) -> list[int]:
        return [c for c, alive in enumerate(self._semantic_alive[b][g]) if alive]

    def _semantic_free_clusters(self, b: int, g: int) -> list[int]:
        return [c for c, alive in enumerate(self._semantic_alive[b][g]) if not alive]

    def _set_semantic_p_hi(self, b: int, g: int, c: int, pos: int) -> None:
        pos = int(pos)
        if not self._semantic_mark_scalar_dirty("p_hi_c"):
            self.p_hi_c[b, g, c] = pos
        self._semantic_p_hi_c[b][g][c] = pos

    def _set_semantic_current_segment(self, b: int, g: int, c: int, segment: int) -> None:
        segment = int(segment)
        if not self._semantic_mark_scalar_dirty("current_segment"):
            self.current_segment[b, g, c] = segment
        self._semantic_current_segment[b][g][c] = segment

    def _set_semantic_level0_phase(self, b: int, g: int, c: int, phase: int) -> None:
        phase = int(phase)
        if not self._semantic_mark_scalar_dirty("level0_phase"):
            self.level0_phase[b, g, c] = phase
        self._semantic_level0_phase[b][g][c] = phase

    def _set_semantic_n_total(self, b: int, g: int, c: int, n_total: int) -> None:
        n_total = int(n_total)
        if not self._semantic_mark_scalar_dirty("n_total"):
            self.n_total[b, g, c] = n_total
        self._semantic_n_total[b][g][c] = n_total

    def _set_level(
        self,
        ell: int,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        sigma_u: torch.Tensor | None = None,
        sigma2: torch.Tensor | None = None,
        gamma_a: torch.Tensor | None = None,
        gamma_b: torch.Tensor | None = None,
        gamma: torch.Tensor | None = None,
        imp: torch.Tensor | None = None,
    ) -> None:
        self.level_k[:, :, 0, ell].copy_(k)
        self.level_v[:, :, 0, ell].copy_(v)
        self.level_w[:, :, 0, ell].copy_(w)
        if self.allocate_second_order:
            if sigma_u is None:
                self.level_sigma_u[:, :, 0, ell].zero_()
                self.level_sigma2[:, :, 0, ell].zero_()
                self.level_gamma_a[:, :, 0, ell].zero_()
                self.level_gamma_b[:, :, 0, ell].zero_()
                self.level_gamma[:, :, 0, ell].zero_()
            else:
                self.level_sigma_u[:, :, 0, ell].copy_(sigma_u)
                self.level_sigma2[:, :, 0, ell].copy_(sigma2)
                self.level_gamma_a[:, :, 0, ell].copy_(gamma_a)
                self.level_gamma_b[:, :, 0, ell].copy_(gamma_b)
                self.level_gamma[:, :, 0, ell].copy_(gamma)
        if imp is None:
            self.level_imp[:, :, 0, ell].zero_()
        else:
            self.level_imp[:, :, 0, ell].copy_(imp)
        self.level_count[:, :, 0, ell].fill_(self.B)
        if hasattr(self, "_counts"):
            self._counts[ell] = self.B
        self.pad_mask[:, :, 0, ell].zero_()

    def _clear_level(self, ell: int) -> None:
        self.level_k[:, :, 0, ell].zero_()
        self.level_v[:, :, 0, ell].zero_()
        self.level_w[:, :, 0, ell].zero_()
        self.level_imp[:, :, 0, ell].zero_()
        if self.allocate_second_order:
            self.level_sigma_u[:, :, 0, ell].zero_()
            self.level_sigma2[:, :, 0, ell].zero_()
            self.level_gamma_a[:, :, 0, ell].zero_()
            self.level_gamma_b[:, :, 0, ell].zero_()
            self.level_gamma[:, :, 0, ell].zero_()
        self.level_count[:, :, 0, ell].zero_()
        if hasattr(self, "_counts"):
            self._counts[ell] = 0
        self.pad_mask[:, :, 0, ell].zero_()

    def _semantic_clear_slot(self, b: int, g: int, c: int, ell: int, idx: int) -> None:
        self.level_k[b, g, c, ell, idx].zero_()
        self.level_v[b, g, c, ell, idx].zero_()
        self.level_w[b, g, c, ell, idx].zero_()
        self.level_imp[b, g, c, ell, idx].zero_()
        if self.allocate_second_order:
            self.level_sigma_u[b, g, c, ell, idx].zero_()
            self.level_sigma2[b, g, c, ell, idx].zero_()
            self.level_gamma_a[b, g, c, ell, idx].zero_()
            self.level_gamma_b[b, g, c, ell, idx].zero_()
            self.level_gamma[b, g, c, ell, idx].zero_()
        self.level_p_lo[b, g, c, ell, idx].zero_()
        self.level_p_hi[b, g, c, ell, idx].zero_()
        self.level_sum_wp[b, g, c, ell, idx].zero_()
        self.level_order[b, g, c, ell, idx].zero_()
        self.pad_mask[b, g, c, ell, idx] = False

    def _semantic_write_slot(
        self,
        b: int,
        g: int,
        c: int,
        ell: int,
        idx: int,
        entry: tuple[torch.Tensor, ...],
    ) -> None:
        k, v, w, su, s2, ga, gb, gm, p_lo, p_hi, sum_wp, order, is_pad = entry
        self.level_k[b, g, c, ell, idx].copy_(k)
        self.level_v[b, g, c, ell, idx].copy_(v)
        self.level_w[b, g, c, ell, idx].copy_(w.float())
        self.level_imp[b, g, c, ell, idx].zero_()
        if self.allocate_second_order:
            self.level_sigma_u[b, g, c, ell, idx].copy_(su)
            self.level_sigma2[b, g, c, ell, idx].copy_(s2)
            self.level_gamma_a[b, g, c, ell, idx].copy_(ga)
            self.level_gamma_b[b, g, c, ell, idx].copy_(gb)
            self.level_gamma[b, g, c, ell, idx].copy_(gm)
        self.level_p_lo[b, g, c, ell, idx].copy_(p_lo.long())
        self.level_p_hi[b, g, c, ell, idx].copy_(p_hi.long())
        self.level_sum_wp[b, g, c, ell, idx].copy_(sum_wp.long())
        self.level_order[b, g, c, ell, idx].copy_(order.long())
        self.pad_mask[b, g, c, ell, idx] = bool(is_pad)

    def _semantic_slot_entry(self, b: int, g: int, c: int, ell: int, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.level_k[b, g, c, ell, idx].clone(),
            self.level_v[b, g, c, ell, idx].clone(),
            self.level_w[b, g, c, ell, idx].clone(),
            self.level_sigma_u[b, g, c, ell, idx].clone() if self.allocate_second_order else None,
            self.level_sigma2[b, g, c, ell, idx].clone() if self.allocate_second_order else None,
            self.level_gamma_a[b, g, c, ell, idx].clone() if self.allocate_second_order else None,
            self.level_gamma_b[b, g, c, ell, idx].clone() if self.allocate_second_order else None,
            self.level_gamma[b, g, c, ell, idx].clone() if self.allocate_second_order else None,
            self.level_p_lo[b, g, c, ell, idx].clone(),
            self.level_p_hi[b, g, c, ell, idx].clone(),
            self.level_sum_wp[b, g, c, ell, idx].clone(),
            self.level_order[b, g, c, ell, idx].clone(),
            bool(self.pad_mask[b, g, c, ell, idx].item()),
        )

    @staticmethod
    def _semantic_slice_block(block: tuple[torch.Tensor, ...], start: int, end: int | None = None) -> tuple[torch.Tensor, ...]:
        return tuple(x[start:end] if x is not None else None for x in block)

    def _semantic_entry_from_block(self, block: tuple[torch.Tensor, ...], idx: int) -> tuple[torch.Tensor, ...]:
        return (
            block[0][idx], block[1][idx], block[2][idx],
            *(x[idx] if x is not None else None for x in block[3:8]),
            block[8][idx], block[9][idx], block[10][idx], block[11][idx],
            bool(block[12][idx].item()),
        )

    def _semantic_level_block(self, b: int, g: int, c: int, ell: int, count: int) -> tuple[torch.Tensor, ...]:
        src = slice(0, count)
        return (
            self.level_k[b, g, c, ell, src].clone(),
            self.level_v[b, g, c, ell, src].clone(),
            self.level_w[b, g, c, ell, src].clone(),
            self.level_sigma_u[b, g, c, ell, src].clone() if self.allocate_second_order else None,
            self.level_sigma2[b, g, c, ell, src].clone() if self.allocate_second_order else None,
            self.level_gamma_a[b, g, c, ell, src].clone() if self.allocate_second_order else None,
            self.level_gamma_b[b, g, c, ell, src].clone() if self.allocate_second_order else None,
            self.level_gamma[b, g, c, ell, src].clone() if self.allocate_second_order else None,
            self.level_p_lo[b, g, c, ell, src].clone(),
            self.level_p_hi[b, g, c, ell, src].clone(),
            self.level_sum_wp[b, g, c, ell, src].clone(),
            self.level_order[b, g, c, ell, src].clone(),
            self.pad_mask[b, g, c, ell, src].clone(),
        )

    def _semantic_write_block(
        self,
        b: int,
        g: int,
        c: int,
        ell: int,
        idx: int,
        block: tuple[torch.Tensor, ...],
    ) -> None:
        n = block[0].size(0)
        dst = slice(idx, idx + n)
        self.level_k[b, g, c, ell, dst].copy_(block[0])
        self.level_v[b, g, c, ell, dst].copy_(block[1])
        self.level_w[b, g, c, ell, dst].copy_(block[2].float())
        self.level_imp[b, g, c, ell, dst].zero_()
        if self.allocate_second_order:
            self.level_sigma_u[b, g, c, ell, dst].copy_(block[3])
            self.level_sigma2[b, g, c, ell, dst].copy_(block[4])
            self.level_gamma_a[b, g, c, ell, dst].copy_(block[5])
            self.level_gamma_b[b, g, c, ell, dst].copy_(block[6])
            self.level_gamma[b, g, c, ell, dst].copy_(block[7])
        self.level_p_lo[b, g, c, ell, dst].copy_(block[8].long())
        self.level_p_hi[b, g, c, ell, dst].copy_(block[9].long())
        self.level_sum_wp[b, g, c, ell, dst].copy_(block[10].long())
        self.level_order[b, g, c, ell, dst].copy_(block[11].long())
        self.pad_mask[b, g, c, ell, dst].copy_(block[12].bool())

    def _semantic_merge_block_pairs(self, block: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        n_pairs = block[0].size(0) // 2
        if n_pairs == 0:
            return self._semantic_slice_block(block, 0, 0)
        a = tuple(x[: 2 * n_pairs: 2] if x is not None else None for x in block)
        z = tuple(x[1: 2 * n_pairs: 2] if x is not None else None for x in block)
        base = (
            a[0].view(n_pairs, 1, 1, -1),
            a[1].view(n_pairs, 1, 1, -1),
            a[2].view(n_pairs, 1, 1),
            z[0].view(n_pairs, 1, 1, -1),
            z[1].view(n_pairs, 1, 1, -1),
            z[2].view(n_pairs, 1, 1),
        )
        anchors = {
            "p_lo1": a[8].view(n_pairs, 1, 1),
            "p_hi1": a[9].view(n_pairs, 1, 1),
            "sum_wp1": a[10].view(n_pairs, 1, 1),
            "p_lo2": z[8].view(n_pairs, 1, 1),
            "p_hi2": z[9].view(n_pairs, 1, 1),
            "sum_wp2": z[10].view(n_pairs, 1, 1),
        }
        if self.second_order:
            out = self.compact(
                *base,
                a[3].view(n_pairs, 1, 1, -1), a[4].view(n_pairs, 1, 1),
                a[5].view(n_pairs, 1, 1, -1), a[6].view(n_pairs, 1, 1, -1), a[7].view(n_pairs, 1, 1),
                z[3].view(n_pairs, 1, 1, -1), z[4].view(n_pairs, 1, 1),
                z[5].view(n_pairs, 1, 1, -1), z[6].view(n_pairs, 1, 1, -1), z[7].view(n_pairs, 1, 1),
                **anchors,
            )
            k, v, w, su, s2, ga, gb, gm, p_lo, p_hi, sum_wp = out
        else:
            k, v, w, p_lo, p_hi, sum_wp = self.compact(*base, **anchors)
            su, s2, ga, gb, gm = self._empty_stats(k, v, w)
        pair_order = block[11][: 2 * n_pairs].reshape(n_pairs, 2).min(dim=1).values
        pair_pad = block[12][: 2 * n_pairs].reshape(n_pairs, 2).all(dim=1)
        return (
            k[:, 0, 0], v[:, 0, 0], w[:, 0, 0],
            *(x[:, 0, 0] if x is not None else None for x in (su, s2, ga, gb, gm)),
            p_lo[:, 0, 0], p_hi[:, 0, 0], sum_wp[:, 0, 0], pair_order, pair_pad,
        )

    def _semantic_merge_entries(self, a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        k1, v1, w1, su1, s21, ga1, gb1, gm1, lo1, hi1, swp1, order1, pad1 = a
        k2, v2, w2, su2, s22, ga2, gb2, gm2, lo2, hi2, swp2, order2, pad2 = b
        base = (
            k1.view(1, 1, 1, -1),
            v1.view(1, 1, 1, -1),
            w1.view(1, 1, 1),
            k2.view(1, 1, 1, -1),
            v2.view(1, 1, 1, -1),
            w2.view(1, 1, 1),
        )
        anchors = {
            "p_lo1": lo1.view(1, 1, 1),
            "p_hi1": hi1.view(1, 1, 1),
            "sum_wp1": swp1.view(1, 1, 1),
            "p_lo2": lo2.view(1, 1, 1),
            "p_hi2": hi2.view(1, 1, 1),
            "sum_wp2": swp2.view(1, 1, 1),
        }
        if self.second_order:
            out = self.compact(
                *base,
                su1.view(1, 1, 1, -1), s21.view(1, 1, 1),
                ga1.view(1, 1, 1, -1), gb1.view(1, 1, 1, -1), gm1.view(1, 1, 1),
                su2.view(1, 1, 1, -1), s22.view(1, 1, 1),
                ga2.view(1, 1, 1, -1), gb2.view(1, 1, 1, -1), gm2.view(1, 1, 1),
                **anchors,
            )
            k, v, w, su, s2, ga, gb, gm, p_lo, p_hi, sum_wp = out
            return (
                k[0, 0, 0], v[0, 0, 0], w[0, 0, 0],
                su[0, 0, 0], s2[0, 0, 0], ga[0, 0, 0], gb[0, 0, 0], gm[0, 0, 0],
                p_lo[0, 0, 0], p_hi[0, 0, 0], sum_wp[0, 0, 0],
                torch.minimum(order1, order2), bool(pad1 and pad2),
            )
        k, v, w, p_lo, p_hi, sum_wp = self.compact(*base, **anchors)
        return (
            k[0, 0, 0], v[0, 0, 0], w[0, 0, 0],
            *self._empty_stats(k1, v1, w1),
            p_lo[0, 0, 0], p_hi[0, 0, 0], sum_wp[0, 0, 0],
            torch.minimum(order1, order2), bool(pad1 and pad2),
        )

    def _empty_stats(self, k, v, w=None):
        if not self.allocate_second_order:
            return (None,) * 5
        zeros_k = torch.zeros_like(k)
        zeros_w = k.new_zeros(k.shape[:-1]) if w is None else torch.zeros_like(w)
        # Staging statistics are read-only until copied into separate buffers.
        return zeros_k, zeros_w, zeros_k, torch.zeros_like(v), zeros_w

    def _semantic_flat_level_fields(self) -> tuple[torch.Tensor, ...]:
        """The 13 entry fields as flat (n_rows, ...) views of the level buffers.

        Row layout mirrors the buffers' own (b, g, c, ell, slot) contiguity:
        ``row = ((b * G + g) * K_max + c) * L_alloc * B + ell * B + slot``.
        Flat rows are what let one gather/scatter serve every (b, g, cluster)
        at once instead of one indexing chain per lane.
        """
        n = self.batch_size * self.n_groups * self.K_max * self.L_alloc * self.B
        return (
            self.level_k.view(n, self.k_dim),
            self.level_v.view(n, self.v_dim),
            self.level_w.view(n),
            self.level_sigma_u.view(n, self.k_dim) if self.allocate_second_order else None,
            self.level_sigma2.view(n) if self.allocate_second_order else None,
            self.level_gamma_a.view(n, self.k_dim) if self.allocate_second_order else None,
            self.level_gamma_b.view(n, self.v_dim) if self.allocate_second_order else None,
            self.level_gamma.view(n) if self.allocate_second_order else None,
            self.level_p_lo.view(n),
            self.level_p_hi.view(n),
            self.level_sum_wp.view(n),
            self.level_order.view(n),
            self.pad_mask.view(n),
        )

    def _semantic_lane_base(self, b: int, g: int, c: int) -> int:
        return ((b * self.n_groups + g) * self.K_max + c) * self.L_alloc * self.B

    def _semantic_append_top_level_overflow(
        self, b: int, g: int, c: int, incoming: tuple[torch.Tensor, ...]
    ) -> None:
        """Oldest-pair merge at the saturated top level, one entry at a time.

        Genuinely sequential (each merge consumes the pair the previous one
        produced), and only reachable once a cluster's whole ladder is full, so
        it stays on the scalar path.
        """
        ell = self.L_alloc - 1
        for i in range(incoming[0].size(0)):
            merged = self._semantic_merge_entries(
                self._semantic_slot_entry(b, g, c, ell, 0),
                self._semantic_slot_entry(b, g, c, ell, 1),
            )
            self._semantic_write_slot(b, g, c, ell, 0, merged)
            self._semantic_shift_after_oldest_pair_merge(
                b, g, c, ell, self._semantic_entry_from_block(incoming, i), top_level=True
            )
            self._set_semantic_level_count(b, g, c, ell, self.B)

    def _semantic_append_top_level_overflow_batched(self, jobs, stage) -> None:
        """Top-level overflow for many saturated ladders at once, without host syncs.

        `jobs` holds (b, g, c, start, m): m staged rows from `start` overflow a
        full top level. Per entry, the scalar path merges slots 0 and 1 into
        slot 0, shifts slots 2..B-1 left and appends the entry. With
        S = [top level | incoming], m such steps leave slot 0 = left fold of
        S[0..m] and slots 1..B-1 = S[m+1..m+B-1]. The fold runs one batched pair
        merge per step for every lane still folding, rounding the running entry
        to the cache dtypes after each step like the stored slot, so values
        match the scalar path.
        """
        fields = self._semantic_flat_level_fields()
        dev = fields[0].device
        B, top = self.B, self.L_alloc - 1
        # Longest folds first: the lanes still folding at step t are a prefix.
        jobs = sorted(jobs, key=lambda job: -job[4])
        n = len(jobs)
        m = np.asarray([job[4] for job in jobs], dtype=np.int64)
        rows = np.asarray([self._semantic_lane_base(b, g, c) + top * B for b, g, c, _, _ in jobs], dtype=np.int64)
        offsets = np.cumsum(m) - m
        level_rows = (rows[:, None] + np.arange(B)).reshape(-1)
        staged_rows = _ranges_to_index_array(np.asarray([job[3] for job in jobs], dtype=np.int64), m)

        def at(t):
            # Pool row of S[t] for the lanes with index t available.
            t = np.asarray(t, dtype=np.int64)
            lane = np.arange(t.shape[0]) if t.ndim == 1 else np.arange(t.shape[0])[:, None]
            return np.where(t < B, lane * B + t, n * B + offsets[lane] + t - B)

        steps = int(m.max())
        live = np.asarray([int((m >= t).sum()) for t in range(1, steps + 1)], dtype=np.int64)
        seed = at(np.zeros(n, dtype=np.int64))
        nxt = np.concatenate([at(np.full(k, t, dtype=np.int64)) for t, k in zip(range(1, steps + 1), live)])
        rest = at(m[:, None] + 1 + np.arange(B - 1)).reshape(-1)
        fs, fg, sd, nx, rs, head, tail = _upload_indices(dev, (
            level_rows, staged_rows, seed, nxt, rest, rows, (rows[:, None] + np.arange(1, B)).reshape(-1)))
        pool = tuple(
            torch.cat([f.index_select(0, fs), x.index_select(0, fg).to(f.dtype)], dim=0) if f is not None else None
            for f, x in zip(fields, stage)
        )
        acc = tuple(x.index_select(0, sd) if x is not None else None for x in pool)
        start = 0
        for k in live.tolist():
            later = tuple(x.index_select(0, nx[start:start + k]) if x is not None else None for x in pool)
            start += k
            pairs = tuple(torch.stack((a[:k], b), dim=1).flatten(0, 1) if a is not None else None
                          for a, b in zip(acc, later))
            merged = self._semantic_merge_block_pairs(pairs)
            acc = tuple(
                torch.cat((y.to(f.dtype), a[k:]), dim=0) if a is not None else None
                for f, a, y in zip(fields, acc, merged)
            )
        # Slot 0 of every lane, then slots 1..B-1: one scatter per field.
        for f, a, x in zip(fields, acc, pool):
            if f is not None:
                f.index_copy_(0, head, a.to(f.dtype))
                f.index_copy_(0, tail, x.index_select(0, rs))
        self.level_imp.view(-1).index_fill_(0, fs, 0)

    def _semantic_index_tensors(
        self, device: torch.device, *span_lists: list[tuple[int, int]], extra: np.ndarray | None = None,
    ) -> list[torch.Tensor]:
        """Build several range-derived index tensors with ONE host->device copy.

        ``torch.tensor(<python list>, device="cuda")`` is a blocking pageable
        transfer, and building the list costs more on the host than the kernels
        it feeds save. Every index here is a concatenation of contiguous ranges,
        so numpy builds them ~35x faster and they travel together.
        """
        arrays = [_spans_to_index_array(spans) for spans in span_lists]
        if extra is not None:
            arrays.append(extra.reshape(-1))
        sizes = [int(a.size) for a in arrays]
        total = sum(sizes)
        if total == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return [empty] * len(arrays)
        flat = _upload(np.concatenate(arrays) if len(arrays) > 1 else arrays[0], device)
        out: list[torch.Tensor] = []
        off = 0
        for n in sizes:
            out.append(flat[off:off + n])
            off += n
        return out

    def _semantic_append_entries_batched(
        self,
        lanes: list[tuple[int, int, int]],
        counts: list[int],
        block: tuple[torch.Tensor, ...],
        *,
        replay_cuts: dict[int, torch.Tensor] | None = None,
    ) -> None:
        """Fenwick-append `block` into many (b, g, cluster) ladders at once.

        `block` holds the incoming entries of every lane back to back, `counts[i]`
        rows for `lanes[i]`. CUDA first-order updates fuse field reads/merges
        and field writes; other layouts use the batched PyTorch reference.
        Work is scheduled per level, without a per-lane indexing chain.

        The per-level index arithmetic runs on the host against the
        `_semantic_counts` mirror, which is already the authority for these
        counts, so none of it costs a device sync. Indices are accumulated as
        ranges and moved once per level (see `_semantic_index_tensors`).
        """
        if self.beta_adaptive_merge and self._replaying_updates and replay_cuts is None:
            raise RuntimeError("Beta update replay requires recorded compaction cuts")
        cut_records = None
        if self._update_actions is not None:
            metadata = (tuple(lanes), tuple(counts))
            kind = "append"
            if self.beta_adaptive_merge:
                cut_records = []
                metadata = (*metadata, cut_records)
                kind = "append_beta"
            # Fresh staging rows (not views of cache buffers) are never written
            # again, so they are kept by reference instead of copied.
            shared = self._semantic_buffer_storages()
            self._update_actions.append((kind, metadata, tuple(
                self._active_updates.save(x, owned=x is not None and x.untyped_storage().data_ptr() not in shared)
                for x in block)))

        self._mid_decode_state = None
        lengths = np.asarray(counts, dtype=np.int64).reshape(-1)
        active = np.flatnonzero(lengths > 0)
        if not len(active):
            return
        fields = self._semantic_flat_level_fields()
        dev = fields[0].device
        fused = (_triton_updates() if _fused_on(dev) and not self.allocate_second_order
                 and fields[0].dtype in (torch.float16, torch.bfloat16, torch.float32)
                 and fields[1].dtype in (torch.float16, torch.bfloat16, torch.float32)
                 and fields[2].dtype == torch.float32
                 and all(x is None or x.is_contiguous() for x in block) else None)
        # Lanes are distinct, so each level is scheduled for all lanes at once
        # from a snapshot of the host counts, with one NumPy pass per level.
        B = self.B
        lane_ids = np.asarray(lanes, dtype=np.int64).reshape(-1, 3)
        bases = ((lane_ids[:, 0] * self.n_groups + lane_ids[:, 1]) * self.K_max + lane_ids[:, 2]) * self.L_alloc * B
        level_counts = np.array([self._semantic_counts[b][g][c] for b, g, c in lanes], dtype=np.int64)
        stage = block
        offsets = (np.cumsum(lengths) - lengths)[active]
        lengths = lengths[active]

        for ell in range(self.L_alloc):
            if not len(active):
                return
            top = ell == self.L_alloc - 1
            base = bases[active] + ell * B
            filled = level_counts[active, ell]
            take = np.minimum(B - filled, lengths)
            fill_src = np.stack((offsets, offsets + take), -1)
            fill_dst = np.stack((base + filled, base + filled + take), -1)
            filled = filled + take
            offsets = offsets + take
            lengths = lengths - take
            done = lengths == 0
            self._set_semantic_level_counts(lanes, active[done], ell, filled[done])
            over = ~done
            overflow_top = list(zip(active[over].tolist(), offsets[over].tolist(), lengths[over].tolist())) if top else []
            carry_mask = np.zeros_like(over) if top else over
            # The pool is [gathered storage rows] ++ [gathered staging rows], so
            # staging positions shift by the total storage row count.
            base, m = base[carry_mask], lengths[carry_mask]
            n_recs = len(m)
            carry_n = (m + 1) // 2
            n_pair_rows = 2 * carry_n
            n_surv = B + m - n_pair_rows
            store_off = np.arange(n_recs, dtype=np.int64) * B
            n_store = n_recs * B
            stage_off = n_store + np.cumsum(m) - m
            store_rows = np.stack((base, base + B), -1)
            stage_rows = np.stack((offsets[carry_mask], offsets[carry_mask] + m), -1)
            surv_dst = np.stack((base, base + n_surv), -1)
            clear_dst = np.stack((base + n_surv, base + B), -1)
            self._set_semantic_level_counts(lanes, active[carry_mask], ell, n_surv)
            ps = np.minimum(B, n_pair_rows)
            # Per record: storage pair rows, then staging pair rows; then the
            # storage survivors and staging survivors. Empty ranges vanish.
            pair_idx = np.stack((store_off, store_off + ps, stage_off, stage_off + n_pair_rows - ps), -1)
            surv_idx = np.stack((store_off + ps, store_off + B,
                                 stage_off + np.maximum(0, n_pair_rows - B), stage_off + m), -1)
            pair_idx, surv_idx = pair_idx.reshape(-1, 2), surv_idx.reshape(-1, 2)

            beta_lengths, beta_layout = None, None
            if self.beta_adaptive_merge and n_recs:
                beta_lengths = n_pair_rows.tolist()
                if fused is not None:
                    from models._vendor.semanticlogkv.litgpt.beta_log_kv import make_layout

                    beta_layout = make_layout(beta_lengths, "cpu").numpy()
            indices = self._semantic_index_tensors(
                dev, fill_src, fill_dst, store_rows, stage_rows,
                pair_idx, surv_idx, surv_dst, clear_dst, extra=beta_layout,
            )
            fs, fd, si, gi, pi, vi, sd, cd = indices[:8]
            if beta_layout is not None:
                beta_layout = indices[8].view(-1, 3)

            # Fill first: an overflowing lane's pooled rows include the slots this
            # very step just topped up.
            if fd.numel():
                if fused is not None:
                    fused.fill(fields, stage, fs, fd)
                else:
                    for f, x in zip(fields, stage):
                        if f is not None:
                            f.index_copy_(0, fd, x.index_select(0, fs).to(f.dtype))

            if overflow_top:
                # One batched left fold for every saturated lane (no host syncs).
                self._semantic_append_top_level_overflow_batched(
                    [(*lanes[i], start, n) for i, start, n in overflow_top], stage)
                self._set_semantic_level_counts(lanes, active[over], ell, np.full(int(over.sum()), B))

            if not n_recs:
                return

            cuts = None
            if self.beta_adaptive_merge:
                # Each lane supplies an even number of rows. Never group across
                # concatenated lane boundaries, including a two-row tail.
                if replay_cuts is not None:
                    if ell not in replay_cuts:
                        raise RuntimeError(f"missing Beta compaction cuts for level {ell}")
                    cuts = replay_cuts[ell]
            if fused is not None:
                if self.beta_adaptive_merge:
                    carry, cuts = fused.merge_scatter(fields, stage, si, gi, pi, vi, sd, cd,
                                                       beta_lengths=beta_lengths, beta_cuts=cuts,
                                                       beta_layout=beta_layout)
                else:
                    carry = fused.merge_scatter(fields, stage, si, gi, pi, vi, sd, cd)
            else:
                pool = tuple(
                    torch.cat([f.index_select(0, si), x.index_select(0, gi).to(f.dtype)], dim=0) if f is not None else None
                    for f, x in zip(fields, stage)
                )
                merge_block = tuple(x.index_select(0, pi) if x is not None else None for x in pool)
                if self.beta_adaptive_merge:
                    from models._vendor.semanticlogkv.litgpt.beta_log_kv import merge_blocks

                    carry, cuts = merge_blocks(merge_block, beta_lengths, cuts=cuts)
                else:
                    carry = self._semantic_merge_block_pairs(merge_block)
                for f, x in zip(fields, pool):
                    if f is not None:
                        f.index_copy_(0, sd, x.index_select(0, vi))
                if cd.numel():
                    for f in fields:
                        if f is not None:
                            f.index_fill_(0, cd, 0)

            if cut_records is not None:
                cut_records.append((ell, self._active_updates.save(cuts)))

            stage = carry
            active, lengths = active[carry_mask], carry_n
            offsets = np.cumsum(lengths) - lengths

    def _semantic_append_entry_block(self, b: int, g: int, c: int, block: tuple[torch.Tensor, ...]) -> None:
        self._semantic_append_entries_batched([(b, g, c)], [block[0].size(0)], block)

    def _semantic_shift_after_oldest_pair_merge(
        self,
        b: int,
        g: int,
        c: int,
        ell: int,
        incoming: tuple[torch.Tensor, ...],
        *,
        top_level: bool,
    ) -> None:
        # One block copy instead of cloning/writing every surviving slot.
        tail = self._semantic_slice_block(self._semantic_level_block(b, g, c, ell, self.B), 2)
        if top_level:
            self._semantic_write_block(b, g, c, ell, 1, tail)
            self._semantic_write_slot(b, g, c, ell, self.B - 1, incoming)
            return
        self._semantic_write_block(b, g, c, ell, 0, tail)
        self._semantic_write_slot(b, g, c, ell, self.B - 2, incoming)
        self._semantic_clear_slot(b, g, c, ell, self.B - 1)
        self._set_semantic_level_count(b, g, c, ell, self.B - 1)

    def _semantic_append_entry(self, b: int, g: int, c: int, entry: tuple[torch.Tensor, ...]) -> None:
        ell = 0
        incoming = entry
        while True:
            count = self._semantic_level_count(b, g, c, ell)
            if count < self.B:
                self._semantic_write_slot(b, g, c, ell, count, incoming)
                self._set_semantic_level_count(b, g, c, ell, count + 1)
                return

            merged = self._semantic_merge_entries(
                self._semantic_slot_entry(b, g, c, ell, 0),
                self._semantic_slot_entry(b, g, c, ell, 1),
            )
            if ell == self.L_alloc - 1:
                self._semantic_write_slot(b, g, c, ell, 0, merged)
                self._semantic_shift_after_oldest_pair_merge(b, g, c, ell, incoming, top_level=True)
                self._set_semantic_level_count(b, g, c, ell, self.B)
                return
            self._semantic_shift_after_oldest_pair_merge(b, g, c, ell, incoming, top_level=False)
            incoming = merged
            ell += 1

    def _semantic_clear_cluster(self, b: int, g: int, c: int) -> None:
        if self._update_actions is not None:
            self._update_actions.append(("clear", (b, g, c), ()))
        self._mid_decode_state = None
        self.level_k[b, g, c].zero_()
        self.level_v[b, g, c].zero_()
        self.level_w[b, g, c].zero_()
        self.level_imp[b, g, c].zero_()
        if self.allocate_second_order:
            self.level_sigma_u[b, g, c].zero_()
            self.level_sigma2[b, g, c].zero_()
            self.level_gamma_a[b, g, c].zero_()
            self.level_gamma_b[b, g, c].zero_()
            self.level_gamma[b, g, c].zero_()
        self.level_p_lo[b, g, c].zero_()
        self.level_p_hi[b, g, c].zero_()
        self.level_sum_wp[b, g, c].zero_()
        self.level_order[b, g, c].zero_()
        self._clear_semantic_cluster_counts(b, g, c)
        self.pad_mask[b, g, c].zero_()
        self._set_semantic_alive(b, g, c, False)
        self._set_semantic_n_total(b, g, c, 0)
        self._set_semantic_p_hi(b, g, c, -1)
        self._set_semantic_current_segment(b, g, c, 0)
        self._set_semantic_level0_phase(b, g, c, 0)
        self.ward_cost[b, g, c].fill_(float("inf"))
        self.ward_cost[b, g, :, c].fill_(float("inf"))

    def begin_op_log(self) -> None:
        """Bind a fresh per-(batch, group) route log for one training forward."""
        cap = max(1, self.max_seq_length * 3)
        # zeros, not empty: `take_op_log` returns a common `max_len` prefix, so
        # rows past a lane's own op_log_len would otherwise be uninitialized.
        self.op_log = torch.zeros(
            self.batch_size, self.n_groups, cap, 4,
            device=self.level_count.device,
            dtype=torch.int32,
        )
        self.op_log_len = torch.zeros(self.batch_size, self.n_groups, device=self.level_count.device, dtype=torch.int32)
        self._op_log_host: list[list[list[tuple[int, int, int, int]]]] = [
            [[] for _ in range(self.n_groups)] for _ in range(self.batch_size)
        ]
        self._op_log_len_host: list[list[int]] = [
            [0] * self.n_groups for _ in range(self.batch_size)
        ]
        self._op_replay_cursor = None
        self._op_replay_cursor_host = None
        self._pending_op_starts = {}
        self._pending_op_rows = {}

    def take_op_log(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if getattr(self, "op_log", None) is None or getattr(self, "op_log_len", None) is None:
            return None, None
        self._flush_recorded_ops()
        if hasattr(self, "_op_log_len_host"):
            max_len = max((max(row) for row in self._op_log_len_host), default=0)
            self._last_op_log_host = [[ops[:] for ops in batch] for batch in self._op_log_host]
        else:
            max_len = int(self.op_log_len.max().item())
            self._last_op_log_host = None
        return self.op_log[:, :, :max_len].clone(), self.op_log_len.clone()

    def _record_op(self, b: int, g: int, op: int, a: int, b_arg: int, c_arg: int) -> None:
        self._record_ops(b, g, [(op, a, b_arg, c_arg)])

    def _record_ops(self, b: int, g: int, rows: list[tuple[int, int, int, int]] | np.ndarray) -> None:
        """Append op rows; `rows` may be tuples or an int [n, 4] NumPy array."""
        if getattr(self, "op_log", None) is None or getattr(self, "op_log_len", None) is None:
            return
        if not len(rows):
            return
        idx = self._op_log_len_host[b][g] if hasattr(self, "_op_log_len_host") else int(self.op_log_len[b, g].item())
        end = idx + len(rows)
        if end > self.op_log.size(2):
            raise RuntimeError("semantic LogKV op_log overflow")
        array = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
        if getattr(self, "_semantic_defer_scalars", False) and hasattr(self, "_op_log_len_host"):
            self._pending_op_starts.setdefault((b, g), idx)
            # Keep the array for the deferred upload; no tuple round trip.
            pending = getattr(self, "_pending_op_rows", None)
            if pending is None:
                pending = self._pending_op_rows = {}
            pending.setdefault((b, g), []).append(array)
        else:
            self.op_log[b, g, idx:end] = _upload(array.astype(np.int32), self.op_log.device)
            self.op_log_len[b, g] = end
        if hasattr(self, "_op_log_len_host"):
            # Column lists give Python ints directly; zip builds the row tuples.
            self._op_log_host[b][g].extend(zip(*array.T.tolist()))
            self._op_log_len_host[b][g] = end

    def _flush_recorded_ops(self):
        """Upload all log rows appended in this flush together, in lane order."""
        starts = getattr(self, "_pending_op_starts", None)
        if not starts:
            return
        spans, rows = [], []
        capacity = self.op_log.size(2)
        pending = getattr(self, "_pending_op_rows", None) or {}
        for (b, g), start in starts.items():
            end = self._op_log_len_host[b][g]
            base = (b * self.n_groups + g) * capacity
            spans.append((base + start, base + end))
            chunks = pending.get((b, g))
            if chunks and sum(len(chunk) for chunk in chunks) == end - start:
                rows.append(np.concatenate(chunks).astype(np.int32))
            else:
                rows.append(np.asarray(self._op_log_host[b][g][start:end], dtype=np.int32).reshape(-1, 4))
        pending.clear()
        indices = _spans_to_index_array(spans)
        values = np.concatenate(rows)
        # One upload: indices, int32 rows widened to int64, then the lengths.
        packed = _upload(np.concatenate((indices, values.reshape(-1).astype(np.int64),
                                         np.asarray(self._op_log_len_host, dtype=np.int64).reshape(-1))),
                         self.op_log.device)
        n = len(indices)
        self.op_log.view(-1, 4).index_copy_(0, packed[:n], packed[n:5 * n].view(n, 4).to(torch.int32))
        self.op_log_len.copy_(packed[5 * n:].view(self.op_log_len.shape))
        starts.clear()

    def _record_semantic_join_run(
        self,
        b: int,
        g: int,
        c: int,
        segment: int,
        token_indices: list[int],
        *,
        new_segment: bool,
    ) -> None:
        if not token_indices:
            return
        rows: list[tuple[int, int, int, int]] = []
        start = 0
        if new_segment:
            rows.append((LOG_KV_OP_NEW_SEGMENT, c, int(segment), int(token_indices[0])))
            start = 1
        rows.extend((LOG_KV_OP_JOIN, c, int(segment), int(token_idx)) for token_idx in token_indices[start:])
        self._record_ops(b, g, rows)

    def _semantic_entry_from_token(
        self,
        b: int,
        g: int,
        c: int,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        pos: torch.Tensor,
        *,
        is_pad: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        order = self.level_order.new_tensor(self._semantic_level0_phase[b][g][c])
        if is_pad:
            return (
                torch.zeros_like(k_raw),
                torch.zeros_like(v),
                self.level_w.new_zeros(()),
                *self._empty_stats(k_raw, v),
                self.level_p_lo.new_zeros(()),
                self.level_p_hi.new_zeros(()),
                self.level_sum_wp.new_zeros(()),
                order,
                True,
            )
        p = pos.to(device=self.level_p_lo.device, dtype=torch.int64)
        return (
            k_raw.detach(),
            v.detach(),
            self.level_w.new_ones(()),
            *self._empty_stats(k_raw, v),
            p.clone(),
            p.clone(),
            p.clone(),
            order,
            False,
        )

    def _semantic_append_token_entries(
        self,
        b: int,
        g: int,
        c: int,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        pos: torch.Tensor,
    ) -> None:
        n = k_raw.size(0)
        if n <= 0:
            return
        order_start = self._semantic_level0_phase[b][g][c]
        order = torch.arange(order_start, order_start + n, device=self.level_order.device, dtype=torch.int64)
        p = pos.to(device=self.level_p_lo.device, dtype=torch.int64)
        block = (
            k_raw.detach(),
            v.detach(),
            self.level_w.new_ones(n),
            *self._empty_stats(k_raw, v),
            p,
            p,
            p,
            order,
            self.pad_mask.new_zeros(n),
        )
        self._semantic_append_entry_block(b, g, c, block)
        self._set_semantic_level0_phase(b, g, c, order_start + n)

    def _semantic_append_token_entry(
        self,
        b: int,
        g: int,
        c: int,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        pos: torch.Tensor,
        *,
        is_pad: bool = False,
    ) -> None:
        if not is_pad:
            self._semantic_append_token_entries(b, g, c, k_raw.unsqueeze(0), v.unsqueeze(0), pos.reshape(1))
            return
        self._semantic_append_entry(b, g, c, self._semantic_entry_from_token(b, g, c, k_raw, v, pos, is_pad=is_pad))
        self._set_semantic_level0_phase(b, g, c, self._semantic_level0_phase[b][g][c] + 1)

    def _semantic_insert_pad(self, b: int, g: int, c: int, count: int, *, record: bool) -> None:
        if count <= 0:
            return
        if record:
            self._record_op(b, g, LOG_KV_OP_PAD_INSERT, c, 0, count)
        z_k = self.level_k.new_zeros(self.k_dim)
        z_v = self.level_v.new_zeros(self.v_dim)
        z_pos = self.level_p_lo.new_zeros(())
        for _ in range(int(count)):
            self._semantic_append_token_entry(b, g, c, z_k, z_v, z_pos, is_pad=True)

    def _semantic_update_member_metadata(
        self,
        b: int,
        g: int,
        c: int,
        k_raw: torch.Tensor,
        pos: torch.Tensor,
        *,
        pos_idx: int,
        new_segment: bool = False,
        segment: int | None = None,
    ) -> None:
        if new_segment:
            self.n_eff[b, g, c].mul_(self.seg_forget)
            if segment is None:
                self._set_semantic_current_segment(b, g, c, self._semantic_current_segment[b][g][c] + 1)
            else:
                self._set_semantic_current_segment(b, g, c, segment)
        n_eff_pre = self.n_eff[b, g, c].clone()
        denom = n_eff_pre + 1.0
        self.centroid[b, g, c].copy_(
            torch.where(
                n_eff_pre > 0,
                (n_eff_pre * self.centroid[b, g, c] + k_raw.float()) / denom,
                k_raw.float(),
            )
        )
        self.n_eff[b, g, c].copy_(denom)
        self._set_semantic_n_total(b, g, c, self._semantic_n_total[b][g][c] + 1)
        self._set_semantic_p_hi(b, g, c, pos_idx)
        self._semantic_ward_dirty[b][g] = True

    def _semantic_new_cluster(
        self,
        b: int,
        g: int,
        c: int,
        token_idx: int,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        pos: torch.Tensor,
        *,
        record: bool,
    ) -> None:
        self._semantic_clear_cluster(b, g, c)
        self._set_semantic_alive(b, g, c, True)
        self.centroid[b, g, c].copy_(k_raw.float())
        self.n_eff[b, g, c] = 1.0
        self._set_semantic_n_total(b, g, c, 1)
        self._set_semantic_p_hi(b, g, c, token_idx)
        self._set_semantic_current_segment(b, g, c, 0)
        self._semantic_append_token_entry(b, g, c, k_raw, v, pos)
        self._semantic_ward_dirty[b][g] = True
        if record:
            self._record_op(b, g, LOG_KV_OP_NEW_CLUSTER, c, 0, int(token_idx))

    def _semantic_new_clusters(self, jobs, k_raw, v, positions, positions_host, *, record):
        """Initialize independent slots and append their first exact token together."""
        if not jobs:
            return
        lanes = [(b, g, c) for b, g, c, _ in jobs]
        self._semantic_clear_clusters(lanes)
        bi, gi, ci, ti = _upload(np.asarray(jobs, dtype=np.int64).reshape(-1, 4).T, k_raw.device).unbind(0)
        ids = (bi * self.n_groups + gi) * self.K_max + ci
        key, value, pos = k_raw.detach()[bi, gi, ti], v.detach()[bi, gi, ti], positions[bi, ti].long()
        self.centroid.flatten(0, 2).index_copy_(0, ids, key.float())
        self.n_eff.flatten().index_fill_(0, ids, 1.)
        n = len(jobs)
        block = (key, value, self.level_w.new_ones(n), *self._empty_stats(key, value),
                 pos, pos, pos, self.level_order.new_zeros(n), self.pad_mask.new_zeros(n))
        self._semantic_append_entries_batched(lanes, [1] * n, block)
        first = [positions_host[b][i] for b, _, _, i in jobs]
        self._set_semantic_scalars("alive", lanes, [True] * n)
        self._set_semantic_scalars("n_total", lanes, [1] * n)
        self._set_semantic_scalars("p_hi_c", lanes, first)
        self._set_semantic_scalars("level0_phase", lanes, [1] * n)
        for b, g, _ in lanes:
            self._semantic_ward_dirty[b][g] = True
        if record:
            # Consecutive jobs of one (b, g) share a log append, in job order.
            start = 0
            for stop in range(1, n + 1):
                if stop == n or jobs[stop][:2] != jobs[start][:2]:
                    b, g = jobs[start][:2]
                    self._record_ops(b, g, [(LOG_KV_OP_NEW_CLUSTER, c, 0, int(p))
                                            for (_, _, c, _), p in zip(jobs[start:stop], first[start:stop])])
                    start = stop

    def _semantic_join(
        self,
        b: int,
        g: int,
        c: int,
        segment: int,
        token_idx: int,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        pos: torch.Tensor,
        *,
        record: bool,
    ) -> None:
        self._semantic_append_token_entry(b, g, c, k_raw, v, pos)
        self._semantic_update_member_metadata(b, g, c, k_raw, pos, pos_idx=token_idx, segment=segment)
        if record:
            self._record_op(b, g, LOG_KV_OP_JOIN, c, int(segment), int(token_idx))

    def _semantic_new_segment(
        self,
        b: int,
        g: int,
        c: int,
        token_idx: int,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        pos: torch.Tensor,
        *,
        record: bool,
    ) -> None:
        align = 1 << self.seg_block_level
        pad_count = 0 if self.seg_block_level == 0 else (-self._semantic_level0_phase[b][g][c]) % align
        self._semantic_insert_pad(b, g, c, pad_count, record=record)
        new_seg = self._semantic_current_segment[b][g][c] + 1
        self._semantic_append_token_entry(b, g, c, k_raw, v, pos)
        self._semantic_update_member_metadata(
            b, g, c, k_raw, pos, pos_idx=token_idx, new_segment=True, segment=new_seg
        )
        if record:
            self._record_op(b, g, LOG_KV_OP_NEW_SEGMENT, c, new_seg, int(token_idx))

    def _semantic_collect_entries(self, b: int, g: int, c: int) -> list[tuple[torch.Tensor, ...]]:
        entries: list[tuple[torch.Tensor, ...]] = []
        for ell in range(self.L_alloc):
            count = self._semantic_level_count(b, g, c, ell)
            entries.extend(self._semantic_slot_entry(b, g, c, ell, idx) for idx in range(count))
        return sorted(entries, key=lambda entry: int(entry[11].item()))

    def _semantic_capacity_target(self) -> float:
        return max(float(self.max_seq_length) / max(self.K_max, 1), 1.0)

    def _semantic_hard_cap(self) -> float:
        if self.semantic_capacity_hard_cap_mult <= 0.0:
            return math.inf
        return self._semantic_capacity_target() * self.semantic_capacity_hard_cap_mult

    def _semantic_can_accept(self, b: int, g: int, c: int, extra: int = 1) -> bool:
        cap = self._semantic_hard_cap()
        return cap == math.inf or float(self._semantic_n_total[b][g][c] + extra) <= cap

    def _semantic_rebuild_ward_cost(self, b: int, g: int) -> None:
        self._semantic_sync_device_scalars()
        # Metadata updates only mark this table dirty; rebuild on demand at
        # Ward selection, never on direct joins or backward replay.
        live = self.alive[b, g]
        mu = self.centroid[b, g]
        n = self.n_total[b, g].float()
        diff = mu[:, None, :] - mu[None, :, :]
        dist2 = diff.square().sum(dim=-1)
        denom = (n[:, None] + n[None, :]).clamp_min(1.0)
        cost = (n[:, None] * n[None, :] / denom) * dist2
        cost.masked_fill_(~(live[:, None] & live[None, :]), float("inf"))
        cost.fill_diagonal_(float("inf"))
        self.ward_cost[b, g].copy_(cost)
        self._semantic_ward_dirty[b][g] = False

    def _semantic_apply_ward_merge_cost_update(
        self,
        b: int,
        g: int,
        keep: int,
        free: int,
        n_keep: torch.Tensor,
        n_free: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._semantic_ward_dirty[b][g]:
            return None
        live_idx = torch.tensor(
            [c for c in self._semantic_live_clusters(b, g) if c not in (keep, free)],
            device=self.alive.device, dtype=torch.long,
        )
        d_keep = self.ward_cost[b, g, keep, live_idx].clone()
        d_free = self.ward_cost[b, g, free, live_idx].clone()
        d_pair = self.ward_cost[b, g, keep, free].clone()
        if live_idx.numel() == 0:
            return live_idx, d_keep
        n_live = self.n_total[b, g, live_idx].float()
        denom = (n_keep + n_free + n_live).clamp_min(1.0)
        cost = ((n_keep + n_live) * d_keep + (n_free + n_live) * d_free - n_live * d_pair) / denom
        return live_idx, cost.clamp_min(0.0)

    def _semantic_ward_pair(self, b: int, g: int) -> tuple[int, int]:
        self._semantic_sync_device_scalars()
        live_clusters = self._semantic_live_clusters(b, g)
        if len(live_clusters) < 2:
            raise RuntimeError("semantic LogKV Ward merge needs at least two live clusters")
        if self._semantic_ward_dirty[b][g]:
            self._semantic_rebuild_ward_cost(b, g)
        idx = torch.tensor(live_clusters, device=self.alive.device, dtype=torch.long)
        cost = self.ward_cost[b, g].index_select(0, idx).index_select(1, idx)
        n = self.n_total[b, g, idx].float()
        cap = self._semantic_hard_cap()
        if cap != math.inf:
            capped = cost.masked_fill((n[:, None] + n[None, :]) > cap, float("inf"))
            if bool(torch.isfinite(capped).any().item()):
                cost = capped
        flat = int(cost.flatten().argmin().item())
        i = flat // idx.numel()
        j = flat % idx.numel()
        keep = min(live_clusters[i], live_clusters[j])
        free = max(live_clusters[i], live_clusters[j])
        return keep, free

    @staticmethod
    def _semantic_ward_entry_order(metadata, split):
        """Stable two-way merge, including padding's next-real-position rule."""
        left = sorted(range(split), key=lambda i: metadata[i][0])
        right = sorted(range(split, len(metadata)), key=lambda i: metadata[i][0])
        effective = [0] * len(metadata)
        for seq in (left, right):
            next_pos = None
            for i in reversed(seq):
                order, lo, has_mass = metadata[i]
                if has_mass:
                    next_pos = lo
                effective[i] = order if next_pos is None else next_pos
        merged = []
        i = j = 0
        while i < len(left) and j < len(right):
            if effective[left[i]] <= effective[right[j]]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    @staticmethod
    def _semantic_ward_entry_index(order, lo, has_mass, segment, segment_end):
        """Device form of `_semantic_ward_entry_order` for many merges at once.

        Segments 2j/2j+1 hold the keep/free entries of merge j, contiguously.
        Each segment is stably sorted by order; an entry's key is the p_lo of
        the next massive entry at or after it (else its own order). A
        two-pointer merge that prefers the keep side on ties equals a stable
        merge by each side's running maximum of that key, so one stable sort
        per stage replaces the host readback and Python merge.
        """
        shift = 32  # Orders and positions are below 2**32; segment ids are small.
        count = order.numel()
        if not count:
            return order.new_zeros(0)
        perm = torch.argsort((segment << shift) + order, stable=True)
        order, lo, has_mass = order[perm], lo[perm], has_mass[perm]
        ids = torch.arange(count, device=order.device)
        # Next massive entry at or after each slot; segments are contiguous, so
        # an index beyond the segment end means none exists inside it.
        following = torch.where(has_mass, ids, count).flip(0).cummin(0).values.flip(0)
        effective = torch.where(following < segment_end, lo[following.clamp_max(count - 1)], order)
        running = (segment << shift) + effective
        running = running.cummax(0).values - (segment << shift)
        merge = torch.argsort(((segment >> 1) << shift) + running, stable=True)
        return perm[merge]

    def _semantic_clear_clusters(self, lanes):
        """Clear independent clusters with one indexed fill per field."""
        if not lanes:
            return
        if self._update_actions is not None:
            self._update_actions.append(("clear_batch", tuple(lanes), ()))
        self._mid_decode_state = None
        ids = _upload(np.fromiter(((b * self.n_groups + g) * self.K_max + c for b, g, c in lanes),
                                  dtype=np.int64, count=len(lanes)), self.level_k.device)
        fields = ["level_k", "level_v", "level_w", "level_imp", "level_p_lo", "level_p_hi",
                  "level_sum_wp", "level_order", "pad_mask"]
        if self.allocate_second_order:
            fields += ["level_sigma_u", "level_sigma2", "level_gamma_a", "level_gamma_b", "level_gamma"]
        for name in fields:
            getattr(self, name).flatten(0, 2).index_fill_(0, ids, 0)
        ward = self.ward_cost.flatten(0, 1)
        ward[ids // self.K_max, ids % self.K_max, :] = float("inf")
        ward[ids // self.K_max, :, ids % self.K_max] = float("inf")
        if not self._semantic_mark_scalar_dirty("level_count"):
            for b, g, c in lanes:
                self._clear_semantic_cluster_counts(b, g, c)
                self._set_semantic_alive(b, g, c, False)
                self._set_semantic_n_total(b, g, c, 0)
                self._set_semantic_p_hi(b, g, c, -1)
                self._set_semantic_current_segment(b, g, c, 0)
                self._set_semantic_level0_phase(b, g, c, 0)
            return
        # Deferred: the same mirror writes, with one dirty mark per field.
        for name in ("alive", "n_total", "p_hi_c", "current_segment", "level0_phase"):
            self._semantic_mark_scalar_dirty(name)
        zeros = [0] * self.L_alloc
        counts, alive, n_total = self._semantic_counts, self._semantic_alive, self._semantic_n_total
        p_hi, segment, phase = self._semantic_p_hi_c, self._semantic_current_segment, self._semantic_level0_phase
        for b, g, c in lanes:
            counts[b][g][c][:] = zeros
            alive[b][g][c] = False
            n_total[b][g][c] = 0
            p_hi[b][g][c] = -1
            segment[b][g][c] = 0
            phase[b][g][c] = 0

    def _semantic_ward_merge_batch(self, jobs, *, record):
        """One ordered merge per independent group; batch readback and writes.

        Clean Ward tables need their Lance-Williams update, so that uncommon
        case uses the original implementation. Unified routing leaves them dirty.
        """
        if not jobs:
            return
        if any(not self._semantic_ward_dirty[b][g] or
               not any(self._semantic_counts[b][g][keep] + self._semantic_counts[b][g][free])
               for b, g, keep, free in jobs):
            for b, g, keep, free in jobs:
                self._semantic_ward_merge(b, g, keep, free, record=record)
            return
        fields = self._semantic_flat_level_fields()
        spans, counts, sides = [], [], []
        keep_ids, free_ids, masses = [], [], []
        for b, g, keep, free in jobs:
            count = 0
            for c in (keep, free):
                side = 0
                base = self._semantic_lane_base(b, g, c)
                for ell, n in enumerate(self._semantic_counts[b][g][c]):
                    if n:
                        spans.append((base + ell * self.B, base + ell * self.B + n))
                        side += n
                sides.append(side)
                count += side
                # Masses come from authoritative host mirrors; no scalar flush.
                masses.append(self._semantic_n_total[b][g][c])
            counts.append(count)
            keep_ids.append((b * self.n_groups + g) * self.K_max + keep)
            free_ids.append((b * self.n_groups + g) * self.K_max + free)
        source = _spans_to_index_array(spans)
        sides = np.asarray(sides, dtype=np.int64)
        segment = np.repeat(np.arange(len(sides), dtype=np.int64), sides)
        segment_end = np.repeat(np.cumsum(sides), sides)
        phases = np.arange(len(source), dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        # One upload for the entry ordering, phases, cluster ids and masses.
        packed = _upload(np.concatenate((
            source, segment, segment_end, phases, keep_ids, free_ids, masses,
        )).astype(np.int64), self.level_k.device)
        n, n_jobs = len(source), len(jobs)
        source_t, segment_t, end_t, phase_t = packed[:4 * n].view(4, n).unbind(0)
        ki, fi = packed[4 * n:4 * n + 2 * n_jobs].view(2, n_jobs).unbind(0)
        nk, nf = packed[4 * n + 2 * n_jobs:].view(n_jobs, 2).float().unbind(1)
        order = self._semantic_ward_entry_index(
            fields[11].index_select(0, source_t), fields[8].index_select(0, source_t),
            fields[2].index_select(0, source_t) > 0, segment_t, end_t,
        )
        index = source_t.index_select(0, order)
        block = tuple(field.index_select(0, index) if field is not None else None for field in fields)
        block = block[:11] + (phase_t, block[12])
        mu = self.centroid.flatten(0, 2)
        ne = self.n_eff.flatten()
        merged_mu = (nk[:, None] * mu[ki] + nf[:, None] * mu[fi]) / (nk + nf).clamp_min(1)[:, None]
        merged_ne = ne[ki] + ne[fi]
        host = [(self._semantic_n_total[b][g][k] + self._semantic_n_total[b][g][f],
                 max(self._semantic_p_hi_c[b][g][k], self._semantic_p_hi_c[b][g][f]),
                 max(self._semantic_current_segment[b][g][k], self._semantic_current_segment[b][g][f]))
                for b, g, k, f in jobs]
        for b, g, keep, free in jobs:
            if record:
                self._record_op(b, g, LOG_KV_OP_WARD_MERGE, keep, free, -1)
        self._semantic_clear_clusters([(b, g, c) for b, g, k, f in jobs for c in (k, f)])
        mu.index_copy_(0, ki, merged_mu)
        ne.index_copy_(0, ki, merged_ne)
        ne.index_fill_(0, fi, 0)
        lanes = []
        for (b, g, keep, _), (total, hi, segment), count in zip(jobs, host, counts):
            self._set_semantic_alive(b, g, keep, True)
            self._set_semantic_n_total(b, g, keep, total)
            self._set_semantic_p_hi(b, g, keep, hi)
            self._set_semantic_current_segment(b, g, keep, segment)
            self._set_semantic_level0_phase(b, g, keep, count)
            lanes.append((b, g, keep))
        self._semantic_append_entries_batched(lanes, counts, block)

    def _semantic_ward_merge(self, b: int, g: int, keep: int, free: int, *, record: bool) -> None:
        self._semantic_sync_device_scalars()
        if record:
            self._record_op(b, g, LOG_KV_OP_WARD_MERGE, keep, free, -1)
        # Gather whole levels, then transfer only ordering metadata once. The
        # old path synchronized and cloned 13 tensors for every single slot.
        blocks = []
        split = 0
        for c in (keep, free):
            for ell in range(self.L_alloc):
                count = self._semantic_level_count(b, g, c, ell)
                if count:
                    blocks.append(self._semantic_level_block(b, g, c, ell, count))
                    if c == keep:
                        split += count
        if not blocks:
            self._semantic_clear_cluster(b, g, free)
            self.n_eff[b, g, free].zero_()
            return
        block = tuple(torch.cat(fields, dim=0) if fields[0] is not None else None for fields in zip(*blocks))
        metadata = torch.stack((block[11], block[8], (block[2] > 0).long()), dim=-1).cpu().tolist()
        merged = self._semantic_ward_entry_order(metadata, split)
        idx = torch.tensor(merged, device=self.level_k.device, dtype=torch.long)
        block = tuple(field.index_select(0, idx) if field is not None else None for field in block)
        block = block[:11] + (torch.arange(len(merged), device=idx.device), block[12])
        n_keep_host = self._semantic_n_total[b][g][keep]
        n_free_host = self._semantic_n_total[b][g][free]
        n_keep = self.n_total[b, g, keep].float()
        n_free = self.n_total[b, g, free].float()
        ward_update = self._semantic_apply_ward_merge_cost_update(b, g, keep, free, n_keep, n_free)
        n_sum = (n_keep + n_free).clamp_min(1.0)
        centroid = (n_keep * self.centroid[b, g, keep] + n_free * self.centroid[b, g, free]) / n_sum
        n_eff = self.n_eff[b, g, keep] + self.n_eff[b, g, free]
        n_total_host = n_keep_host + n_free_host
        p_hi_host = max(self._semantic_p_hi_c[b][g][keep], self._semantic_p_hi_c[b][g][free])
        segment_host = max(self._semantic_current_segment[b][g][keep], self._semantic_current_segment[b][g][free])
        self._semantic_clear_cluster(b, g, keep)
        self._set_semantic_alive(b, g, keep, True)
        self.centroid[b, g, keep].copy_(centroid)
        self.n_eff[b, g, keep].copy_(n_eff)
        self._set_semantic_n_total(b, g, keep, n_total_host)
        self._set_semantic_p_hi(b, g, keep, p_hi_host)
        self._set_semantic_current_segment(b, g, keep, segment_host)
        self._semantic_append_entry_block(b, g, keep, block)
        self._set_semantic_level0_phase(b, g, keep, len(merged))
        self._semantic_clear_cluster(b, g, free)
        self.n_eff[b, g, free].zero_()
        if ward_update is not None:
            live_idx, cost = ward_update
            if live_idx.numel() > 0:
                self.ward_cost[b, g, keep, live_idx] = cost
                self.ward_cost[b, g, live_idx, keep] = cost
            self.ward_cost[b, g, keep, keep] = float("inf")

    def _semantic_existing_assignments(
        self,
        k_raw: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._semantic_sync_device_scalars()
        live = self.alive[: k_raw.size(0), : k_raw.size(1)]
        x = k_raw.float()
        mu = self.centroid[: x.size(0), : x.size(1)]
        x_norm2 = x.square().sum(dim=-1)
        mu_norm2 = mu.square().sum(dim=-1)
        cross = torch.matmul(x, mu.transpose(-1, -2))
        semantic = (x_norm2.unsqueeze(-1) + mu_norm2.unsqueeze(-2) - 2.0 * cross).clamp_min(0.0)
        gap = (positions[:, None, :, None].to(self.p_hi_c.dtype) - self.p_hi_c[: k_raw.size(0), : k_raw.size(1)].unsqueeze(2))
        gap = gap.float().clamp_min(0.0)
        temporal = self.seg_eta * (gap / (gap + self.seg_g0))
        cost = semantic + temporal
        if self.semantic_capacity_beta > 0.0:
            target = self._semantic_capacity_target()
            load = self.n_total[: k_raw.size(0), : k_raw.size(1)].float() / target
            over = (load - 1.0).clamp_min(0.0)
            penalty = self.semantic_capacity_beta * self.s_h[: k_raw.size(0), : k_raw.size(1)].unsqueeze(-1) * over.square()
            cost = cost + penalty.unsqueeze(2)
        cost = cost.masked_fill(~live.unsqueeze(2), float("inf"))
        winner = cost.argmin(dim=-1)
        s_winner = semantic.gather(-1, winner.unsqueeze(-1)).squeeze(-1)
        has_alive = live.any(dim=-1).unsqueeze(-1)
        direct = has_alive & (s_winner <= self.cluster_lambda_rel * self.s_h[: k_raw.size(0), : k_raw.size(1)].unsqueeze(-1))
        cap = self._semantic_hard_cap()
        if cap != math.inf:
            n_winner = self.n_total[: k_raw.size(0), : k_raw.size(1)].gather(-1, winner).float()
            direct = direct & ((n_winner + 1.0) <= cap)
        return winner, s_winner, direct

    def _semantic_join_or_segment(
        self,
        b: int,
        g: int,
        c: int,
        token_idx: int,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        pos: torch.Tensor,
        *,
        record: bool,
    ) -> None:
        if self.seg_gap_max == math.inf:
            self._semantic_join(b, g, c, 0, token_idx, k_raw, v, pos, record=record)
            return
        gap = int(token_idx) - self._semantic_p_hi_c[b][g][c]
        if gap > self.seg_gap_max:
            self._semantic_new_segment(b, g, c, token_idx, k_raw, v, pos, record=record)
            return
        self._semantic_join(
            b, g, c, self._semantic_current_segment[b][g][c], token_idx, k_raw, v, pos, record=record
        )

    def _semantic_route_k1_batch(
        self,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        positions_host: list[list[int]],
        *,
        record: bool,
    ) -> None:
        jobs: list[tuple[int, int, int, tuple[int, ...]]] = []
        for b in range(k_raw.size(0)):
            for g in range(k_raw.size(1)):
                start = 0
                if not self._semantic_alive[b][g][0]:
                    self._semantic_new_cluster(
                        b, g, 0, positions_host[b][0], k_raw[b, g, 0], v[b, g, 0], positions[b, 0], record=record
                    )
                    start = 1
                if start < k_raw.size(2):
                    jobs.append((b, g, 0, tuple(range(start, k_raw.size(2)))))
        self._semantic_commit_joins(jobs, k_raw, v, positions, positions_host, record=record)

    def _semantic_tree_threshold(self, b: int, g: int) -> float:
        return self.cluster_lambda_rel * self._semantic_s_h_host[b][g]

    @staticmethod
    def _semantic_tree_merge_node(a: _SemanticTreeCluster, b: _SemanticTreeCluster) -> _SemanticTreeCluster:
        n = a.n_total + b.n_total
        centroid = (a.centroid * float(a.n_total) + b.centroid * float(b.n_total)) / float(max(n, 1))
        return _SemanticTreeCluster(
            centroid,
            n,
            max(a.p_hi, b.p_hi),
            a.existing + b.existing,
            a.tokens + b.tokens,
        )

    def _semantic_tree_can_merge(self, a: _SemanticTreeCluster, b: _SemanticTreeCluster) -> bool:
        cap = self._semantic_hard_cap()
        return cap == math.inf or float(a.n_total + b.n_total) <= cap

    def _semantic_tree_best_candidate(
        self,
        b: int,
        g: int,
        pool: list[_SemanticTreeCluster],
        candidates: list[int],
        node: _SemanticTreeCluster,
    ) -> int | None:
        """Rank ``candidates`` (indices into ``pool``) the same way
        ``_semantic_existing_assignments`` ranks live clusters: semantic
        distance plus the temporal tie-break (and, if enabled, the capacity
        penalty) decide the winner, but accept/reject uses the winner's own
        semantic distance alone (algorithm-spec.md §5.3 -- eta only reorders
        candidates, novelty is judged on semantic distance).
        """
        mu = torch.stack([pool[i].centroid for i in candidates])
        semantic = (mu - node.centroid).square().sum(dim=-1)
        gap = torch.tensor(
            [max(node.p_hi - pool[i].p_hi, 0) for i in candidates],
            device=mu.device, dtype=torch.float32,
        )
        cost = semantic + self.seg_eta * (gap / (gap + self.seg_g0))
        if self.semantic_capacity_beta > 0.0:
            target = self._semantic_capacity_target()
            n = torch.tensor([pool[i].n_total for i in candidates], device=mu.device, dtype=torch.float32)
            over = (n / target - 1.0).clamp_min(0.0)
            cost = cost + self.semantic_capacity_beta * self._semantic_s_h_host[b][g] * over.square()
        best_idx = cost.argmin()
        best_f, best_semantic = torch.stack([best_idx.double(), semantic[best_idx].double()]).tolist()
        best = int(best_f)
        if best_semantic > self._semantic_tree_threshold(b, g):
            return None
        return candidates[best]

    def _semantic_tree_ward_cost(self, clusters: list[_SemanticTreeCluster]) -> torch.Tensor:
        if len(clusters) < 2:
            raise RuntimeError("semantic chunk tree Ward merge needs at least two clusters")
        mu = torch.stack([c.centroid for c in clusters])
        n = torch.tensor([c.n_total for c in clusters], device=mu.device, dtype=torch.float32)
        diff = mu[:, None, :] - mu[None, :, :]
        cost = (n[:, None] * n[None, :] / (n[:, None] + n[None, :]).clamp_min(1.0)) * diff.square().sum(dim=-1)
        cost.fill_diagonal_(float("inf"))
        cap = self._semantic_hard_cap()
        if cap != math.inf:
            capped = cost.masked_fill((n[:, None] + n[None, :]) > cap, float("inf"))
            if bool(torch.isfinite(capped).any().item()):
                cost = capped
        return cost

    def _semantic_tree_ward_pairs(
        self,
        clusters: list[_SemanticTreeCluster],
        max_pairs: int,
    ) -> list[tuple[int, int]]:
        cost = self._semantic_tree_ward_cost(clusters)
        n = len(clusters)
        cost = cost.masked_fill(~torch.triu(torch.ones(n, n, device=cost.device, dtype=torch.bool), diagonal=1), float("inf"))
        flat_cost = cost.flatten()
        order = flat_cost.argsort().detach().cpu().tolist()
        flat_cost_host = flat_cost.detach().cpu().tolist()
        pairs: list[tuple[int, int]] = []
        used: set[int] = set()
        for flat in order:
            if len(pairs) >= max_pairs or not math.isfinite(flat_cost_host[flat]):
                break
            i, j = flat // n, flat % n
            if i in used or j in used:
                continue
            pairs.append((i, j))
            used.add(i)
            used.add(j)
        return pairs

    def _semantic_tree_ward_pair(self, clusters: list[_SemanticTreeCluster]) -> tuple[int, int]:
        cost = self._semantic_tree_ward_cost(clusters)
        flat = int(cost.flatten().argmin().item())
        i, j = flat // len(clusters), flat % len(clusters)
        return (i, j) if i < j else (j, i)

    def _semantic_tree_reduce_to_budget(
        self,
        clusters: list[_SemanticTreeCluster],
    ) -> list[_SemanticTreeCluster]:
        clusters = list(clusters)
        while len(clusters) > self.K_max:
            pairs = self._semantic_tree_ward_pairs(clusters, len(clusters) - self.K_max)
            if not pairs:
                pairs = [self._semantic_tree_ward_pair(clusters)]
            merged = {i: self._semantic_tree_merge_node(clusters[i], clusters[j]) for i, j in pairs}
            dropped = {j for _i, j in pairs}
            clusters = [merged.get(i, node) for i, node in enumerate(clusters) if i not in dropped]
        return clusters

    def _semantic_tree_merge_sets(
        self,
        b: int,
        g: int,
        left: list[_SemanticTreeCluster],
        right: list[_SemanticTreeCluster],
    ) -> list[_SemanticTreeCluster]:
        out = list(left)
        for node in right:
            candidates = [i for i, current in enumerate(out) if self._semantic_tree_can_merge(current, node)]
            if candidates:
                best = self._semantic_tree_best_candidate(b, g, out, candidates, node)
                if best is not None:
                    out[best] = self._semantic_tree_merge_node(out[best], node)
                    continue
            out.append(node)
        return self._semantic_tree_reduce_to_budget(out)

    def _semantic_tree_existing_set(self, b: int, g: int) -> list[_SemanticTreeCluster]:
        return [
            _SemanticTreeCluster(
                self.centroid[b, g, c].float().clone(),
                max(self._semantic_n_total[b][g][c], 1),
                self._semantic_p_hi_c[b][g][c],
                (c,),
                (),
            )
            for c in self._semantic_live_clusters(b, g)
        ]

    def _semantic_tree_local_chunks(
        self,
        b: int,
        g: int,
        k_raw: torch.Tensor,
        start: int,
        end: int,
        positions_host: list[list[int]],
    ) -> list[_SemanticTreeCluster]:
        n = end - start
        if n <= 0:
            return []
        x = k_raw[b, g, start:end].float()
        norm = x.square().sum(dim=-1)
        dist = (norm[:, None] + norm[None, :] - 2.0 * (x @ x.T)).clamp_min(0.0)
        reach = dist <= self._semantic_tree_threshold(b, g)
        # ponytail: dense T_block^2 closure; switch to sparse/union-find only if large chunks make this hot.
        for _ in range(max(1, math.ceil(math.log2(max(n, 2))))):
            reach = (reach.float() @ reach.float()) > 0
        labels = torch.where(
            reach,
            torch.arange(n, device=x.device, dtype=torch.long).view(1, n),
            torch.full((n, n), n, device=x.device, dtype=torch.long),
        ).min(dim=1).values
        clusters: list[_SemanticTreeCluster] = []
        cap = self._semantic_hard_cap()
        max_local = n if cap == math.inf else max(1, int(math.floor(cap)))
        for label in labels.unique(sorted=True).detach().cpu().tolist():
            idx = torch.nonzero(labels == int(label), as_tuple=False).flatten()
            for part in idx.split(max_local):
                offsets = tuple(start + int(i) for i in part.detach().cpu().tolist())
                clusters.append(
                    _SemanticTreeCluster(
                        x.index_select(0, part).mean(dim=0).clone(),
                        len(offsets),
                        max(positions_host[b][i] for i in offsets),
                        (),
                        offsets,
                    )
                )
        return self._semantic_tree_reduce_to_budget(clusters)

    def _semantic_tree_local_chunks_batched(
        self,
        b: int,
        g: int,
        k_raw: torch.Tensor,
        chunk_size: int,
        positions_host: list[list[int]],
    ) -> list[list[_SemanticTreeCluster]]:
        """Batched equivalent of calling ``_semantic_tree_local_chunks`` once per
        ``chunk_size``-token slice of ``k_raw[b, g]``: the O(chunk_size^2) distance
        + reachability-closure math (1932-1938 in the per-chunk version) runs once
        across all chunks via ``bmm`` instead of once per chunk via a Python loop,
        and the label tensor is pulled to host in a single ``.tolist()`` instead of
        one sync per connected component per chunk. The final per-component
        centroid is still ``index_select(...).mean(dim=0)`` on the same underlying
        values in the same order as the per-chunk version, so it stays bit-identical
        -- only the (correctness-irrelevant-until-threshold-adjacent) distance
        computation changes kernel. Emission order (chunk asc, label asc, index
        asc, ``max_local``-sized splits) matches the per-chunk version exactly,
        since ``_semantic_tree_merge_sets`` consumes nodes greedily in that order.
        """
        total = k_raw.size(2)
        if total <= 0:
            return []
        n_chunks = math.ceil(total / chunk_size)
        pad = n_chunks * chunk_size - total
        x = k_raw[b, g].float()
        if pad > 0:
            x = torch.cat([x, x.new_zeros(pad, x.size(-1))], dim=0)
        xs = x.view(n_chunks, chunk_size, x.size(-1))

        norm = xs.square().sum(dim=-1)
        dist = (norm[:, :, None] + norm[:, None, :] - 2.0 * torch.bmm(xs, xs.transpose(1, 2))).clamp_min(0.0)
        reach = dist <= self._semantic_tree_threshold(b, g)
        if pad > 0:
            valid = torch.ones(n_chunks, chunk_size, dtype=torch.bool, device=x.device)
            valid[-1, chunk_size - pad:] = False
            reach = reach & (valid[:, :, None] & valid[:, None, :])
        rounds = max(1, math.ceil(math.log2(max(chunk_size, 2))))
        for _ in range(rounds):
            reach = torch.bmm(reach.float(), reach.float()) > 0
        idx_row = torch.arange(chunk_size, device=x.device, dtype=torch.long).view(1, 1, chunk_size)
        sentinel = torch.full((n_chunks, chunk_size, chunk_size), chunk_size, device=x.device, dtype=torch.long)
        labels = torch.where(reach, idx_row.expand_as(sentinel), sentinel).min(dim=2).values
        labels_host = labels.tolist()

        cap = self._semantic_hard_cap()
        max_local = chunk_size if cap == math.inf else max(1, int(math.floor(cap)))

        result: list[list[_SemanticTreeCluster]] = []
        for c in range(n_chunks):
            start = c * chunk_size
            real_n = min(chunk_size, total - start)
            groups: dict[int, list[int]] = {}
            for i in range(real_n):
                groups.setdefault(labels_host[c][i], []).append(i)
            clusters: list[_SemanticTreeCluster] = []
            for label in sorted(groups):
                members = groups[label]
                for lo in range(0, len(members), max_local):
                    part = members[lo:lo + max_local]
                    offsets = tuple(start + i for i in part)
                    part_idx = torch.tensor(part, device=x.device, dtype=torch.long)
                    clusters.append(
                        _SemanticTreeCluster(
                            xs[c].index_select(0, part_idx).mean(dim=0).clone(),
                            len(offsets),
                            max(positions_host[b][i] for i in offsets),
                            (),
                            offsets,
                        )
                    )
            result.append(self._semantic_tree_reduce_to_budget(clusters))
        return result

    @staticmethod
    def _semantic_unified_pairs(cost: torch.Tensor, max_pairs: int) -> torch.Tensor:
        """Mutual-nearest disjoint pairs, batched over independent KV groups.

        Batched results are padded with -1; this avoids a CUDA nonzero/host
        synchronization per group. The 2-D entry point keeps its compact result.
        """
        single = cost.ndim == 2
        if single:
            cost = cost.unsqueeze(0)
        size = cost.size(-1)
        ids = torch.arange(size, device=cost.device, dtype=torch.int32)
        best_cost = cost.amin(dim=-1, keepdim=True)
        tie = ids[:, None].bitwise_xor(ids[None, :])
        best = torch.where(cost == best_cost, tie, 2 * size).argmin(dim=-1)
        pairs = LogStructuredKVCache._semantic_unified_select_pairs(best, best_cost[..., 0], max_pairs)
        return pairs[0][pairs[0, :, 0] >= 0] if single else pairs

    @staticmethod
    def _semantic_unified_select_pairs(best: torch.Tensor, best_cost: torch.Tensor, max_pairs: int) -> torch.Tensor:
        """Select disjoint mutual pairs from row minima without an M x M cost tensor."""
        size = best.size(-1)
        ids = torch.arange(size, device=best.device)
        valid = (ids < best) & (best.gather(1, best) == ids) & torch.isfinite(best_cost)
        pair_cost = best_cost.masked_fill(~valid, float("inf"))
        left = pair_cost.argsort(dim=-1, stable=True)[:, :min(max_pairs, size // 2)]
        right = best.gather(1, left)
        pairs = torch.stack((left, right), dim=-1)
        pairs.masked_fill_(~valid.gather(1, left).unsqueeze(-1), -1)
        return pairs

    @staticmethod
    def _semantic_unified_distance2(mu: torch.Tensor) -> torch.Tensor:
        """FP32 squared distances via GEMM; direct differences for small sets.

        Subtract a shared origin before GEMM to avoid cancellation when keys
        have a large common offset. Close ties may differ from direct cdist
        because the floating-point summation order changes.
        """
        if mu.size(1) <= 32:
            return torch.cdist(mu, mu, compute_mode="donot_use_mm_for_euclid_dist").square()
        gram, norm = LogStructuredKVCache._semantic_unified_gram(mu)
        return (norm[:, :, None] + norm[:, None, :] - 2 * gram).clamp_min_(0)

    @staticmethod
    def _semantic_unified_gram(mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The CUDA row kernel consumes Gram/norm directly, without M x M intermediates."""
        precision = torch.get_float32_matmul_precision()
        try:
            # Training enables "high" globally. Routing distances need full
            # FP32 products even inside bf16 autocast; restore the caller's mode.
            if mu.is_cuda:
                torch.set_float32_matmul_precision("highest")
            with torch.autocast(device_type=mu.device.type, enabled=False):
                centered = mu.float() - mu[:, :1].float()
                norm = centered.square().sum(-1)
                gram = torch.bmm(centered, centered.transpose(1, 2))
        finally:
            if mu.is_cuda:
                torch.set_float32_matmul_precision(precision)
        return gram, norm

    @staticmethod
    def _semantic_unified_matching_sweeps(cost, passes):
        """Torch reference: freeze centers, peel disjoint mutual pairs, sort once."""
        size = cost.size(-1)
        ids = torch.arange(size, device=cost.device)
        tie = ids[:, None].bitwise_xor(ids[None, :])
        partner = ids.expand(cost.size(0), -1).clone()
        pair_cost = cost.new_full(partner.shape, float("inf"))
        remaining = cost.clone()
        for _ in range(passes):
            best_cost = remaining.amin(-1)
            best = torch.where(remaining == best_cost[..., None], tie, 2 * size).argmin(-1)
            accept = (ids != best) & (best.gather(1, best) == ids) & torch.isfinite(best_cost)
            partner = torch.where(accept, best, partner)
            pair_cost = torch.where(accept, best_cost, pair_cost)
            remaining.masked_fill_(accept[:, :, None] | accept[:, None, :], float("inf"))
        return partner, pair_cost

    def _semantic_unified_round_pairs(self, mu, mass, radius, limits, cap, allow_overflow):
        """Ward matching; optional frozen-center sweeps amortize global round overhead."""
        size = mu.size(1)
        # ponytail: approximate only uncapped global merges. Candidate radius
        # checks and finite-cap fallback retain the original matching rules.
        passes = self.semantic_merge_passes if limits is None and cap == math.inf else 1
        fused = _triton_route() if _fused_on(mu) else None
        if fused is not None:
            data, norm = (self._semantic_unified_gram(mu) if size > 32
                          else (self._semantic_unified_distance2(mu), None))
            if passes > 1:
                return self._semantic_unified_select_pairs(*fused.matching_sweeps(data, mass, norm, passes), size // 2)
            nearest, fallback = fused.nearest(data, mass, radius, limits, cap, norm=norm)
            pairs = self._semantic_unified_select_pairs(*nearest, size // 2)
            if allow_overflow and fallback is not None:
                fallback_pairs = self._semantic_unified_select_pairs(*fallback, size // 2)
                pairs = torch.where((pairs[:, :, 0] >= 0).any(1)[:, None, None], pairs, fallback_pairs)
        else:
            distance2 = self._semantic_unified_distance2(mu)
            total = mass[:, :, None] + mass[:, None, :]
            cost = distance2 * (mass[:, :, None] * mass[:, None, :] / total.clamp_min(1))
            live = mass > 0
            cost.masked_fill_(~(live[:, :, None] & live[:, None, :]), float("inf"))
            cost.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))
            if passes > 1:
                return self._semantic_unified_select_pairs(*self._semantic_unified_matching_sweeps(cost, passes), size // 2)
            if limits is not None:
                distance = distance2.sqrt_()
                bound = torch.maximum(
                    radius[:, :, None] + (mass[:, None, :] / total.clamp_min(1)) * distance,
                    radius[:, None, :] + (mass[:, :, None] / total.clamp_min(1)) * distance,
                )
                cost.masked_fill_(bound > limits[:, None, None], float("inf"))
            limited = cost.masked_fill(total > cap, float("inf")) if cap != math.inf else cost
            pairs = self._semantic_unified_pairs(limited, size // 2)
            if allow_overflow and cap != math.inf:
                fallback = self._semantic_unified_pairs(cost, size // 2)
                pairs = torch.where((pairs[:, :, 0] >= 0).any(1)[:, None, None], pairs, fallback)
        if limits is not None:
            # Verify selected radii with direct differences: GEMM cancellation
            # must not let a candidate exceed its original compactness bound.
            left, right = pairs.clamp_min(0).unbind(-1)
            lane_ids = torch.arange(mu.size(0), device=mu.device)[:, None]
            ml, mr = mass[lane_ids, left], mass[lane_ids, right]
            exact = (mu[lane_ids, left] - mu[lane_ids, right]).norm(dim=-1)
            bound = torch.maximum(
                radius[lane_ids, left] + mr / (ml + mr).clamp_min(1) * exact,
                radius[lane_ids, right] + ml / (ml + mr).clamp_min(1) * exact,
            )
            pairs.masked_fill_((bound > limits[:, None]).unsqueeze(-1), -1)
        return pairs

    @torch.no_grad()
    def _semantic_unified_reduce_batch(
        self, groups: list[list[_SemanticTreeCluster]], *,
        max_clusters: int | None = None, radius_limits: list[float] | None = None,
    ) -> list[tuple[list[_SemanticTreeCluster], list[tuple[int, int]]]]:
        """Node-based adapter for callers that need individual centroid tensors."""
        centers = [torch.stack([n.centroid for n in nodes]).float() if nodes else None for nodes in groups]
        packed = self._semantic_unified_reduce_packed(
            groups, centers, max_clusters=max_clusters, radius_limits=radius_limits,
        )
        return [(nodes if len(nodes) == len(original) else
                 [node._replace(centroid=center) for node, center in zip(nodes, mu.unbind(0))], merges)
                for original, (nodes, merges, mu) in zip(groups, packed)]

    @staticmethod
    def _semantic_unified_merge_pack(mu, mass, radius, indices, candidate_pass):
        """Merge and compact from immutable source rows; padding has source -1."""
        fused = _triton_route() if _fused_on(mu) and not candidate_pass else None
        if fused is not None:
            return fused.merge_pack(mu, mass, indices)
        src, partner = indices.unbind(0)
        live, merge = src >= 0, partner >= 0
        src, partner = src.clamp_min(0), partner.clamp_min(0)
        a, b = mu.flatten(0, 1)[src], mu.flatten(0, 1)[partner]
        ma, mb = mass.flatten()[src], mass.flatten()[partner]
        combined = ma + mb
        center = (ma[..., None] * a + mb[..., None] * b) / combined.clamp_min(1)[..., None]
        center = torch.where(merge[..., None], center, a).masked_fill_(~live[..., None], 0)
        weight = torch.where(merge, combined, ma).masked_fill_(~live, 0)
        if candidate_pass:
            ra, rb = radius.flatten()[src], radius.flatten()[partner]
            distance = (a - b).norm(dim=-1)
            bound = torch.maximum(ra + mb / combined.clamp_min(1) * distance,
                                  rb + ma / combined.clamp_min(1) * distance)
            out_radius = torch.where(merge, bound, ra).masked_fill_(~live, 0)
        else:
            out_radius = torch.zeros_like(weight)
        return center, weight, out_radius

    @torch.no_grad()
    def _semantic_unified_reduce_packed(
        self, groups: list[list[_SemanticTreeCluster]], centroids: list[torch.Tensor | None], *,
        max_clusters: int | None = None, radius_limits: list[float] | None = None,
    ) -> list[tuple[list[_SemanticTreeCluster], list[tuple[int, int]], torch.Tensor | None]]:
        """Compatibility adapter for diagnostic callers that need tree nodes."""
        weights = [np.fromiter((n.n_total for n in nodes), dtype=np.float32) for nodes in groups]
        reduced = self._semantic_unified_reduce_arrays(
            weights, centroids, max_clusters=max_clusters, radius_limits=radius_limits,
        )
        outputs = []
        for original, (surviving, traces, centers) in zip(groups, reduced):
            nodes = list(original)
            old_merges = []
            for trace in traces:
                for i, j in trace.tolist():
                    a, b = nodes[i], nodes[j]
                    if a.existing and b.existing:
                        old_merges.append(tuple(sorted((min(a.existing), min(b.existing)))))
                    nodes[i] = _SemanticTreeCluster(
                        a.centroid, a.n_total + b.n_total, max(a.p_hi, b.p_hi),
                        a.existing + b.existing, a.tokens + b.tokens,
                    )
            outputs.append(([nodes[i] for i in surviving], old_merges, centers))
        return outputs

    @staticmethod
    def _semantic_unified_pair_matrix(mu: torch.Tensor) -> torch.Tensor:
        """FP32 squared distances [L, M, M]; the persistent state of the rounds.

        Only one M x M buffer is allocated: with the fused kernels a plain GEMM
        followed by one norm/clamp/mirror pass over it, otherwise the GEMM
        accumulates into the norm sums in place.
        """
        if mu.size(1) <= 32:
            dist = torch.cdist(mu, mu, compute_mode="donot_use_mm_for_euclid_dist").square()
            return LogStructuredKVCache._semantic_unified_symmetrize_(dist)
        fused = _triton_route() if _fused_on(mu) and LogStructuredKVCache._semantic_fused_distances else None
        precision = torch.get_float32_matmul_precision()
        try:
            # Same FP32/centering rules as `_semantic_unified_gram`; the norm
            # expansion rides in the GEMM epilogue (beta=1) instead of extra passes.
            if mu.is_cuda:
                torch.set_float32_matmul_precision("highest")
            with torch.autocast(device_type=mu.device.type, enabled=False):
                centered = mu.float() - mu[:, :1].float()
                norm = centered.square().sum(-1)
                if fused is not None:
                    # One read of the upper triangle and one write replace the
                    # norm broadcast, the beta read, the clamp and the mirror.
                    gram = torch.bmm(centered, centered.transpose(1, 2))
                    try:
                        return fused.pair_distances(gram, norm)
                    except Exception as error:  # Compilation fails before launch.
                        LogStructuredKVCache._semantic_fused_distances = False
                        warnings.warn(f"fused pair distances unavailable ({type(error).__name__}: {error}); "
                                      "using the unfused expression")
                        del gram
                dist = norm[:, :, None] + norm[:, None, :]
                dist.baddbmm_(centered, centered.transpose(1, 2), alpha=-2)
        finally:
            if mu.is_cuda:
                torch.set_float32_matmul_precision(precision)
        return LogStructuredKVCache._semantic_unified_symmetrize_(dist.clamp_min_(0))

    @staticmethod
    def _semantic_unified_symmetrize_(dist: torch.Tensor) -> torch.Tensor:
        """Copy the upper triangle onto the lower one in place.

        The rounds keep D symmetric by construction; this removes any reliance
        on the GEMM producing bitwise-symmetric products. Exact copies only.
        """
        fused = _triton_route() if _fused_on(dist) else None
        if fused is not None and hasattr(fused, "mirror"):
            return fused.mirror(dist)
        size = dist.size(1)
        ids = torch.arange(size, device=dist.device)
        for start in range(0, size, 256):
            stop = min(size, start + 256)
            rows = dist[:, start:stop, :stop]
            # Sources (j, i) with j < i are never written, so chunks cannot race.
            mirrored = dist[:, :stop, start:stop].transpose(1, 2)
            rows.copy_(torch.where(ids[start:stop, None] > ids[None, :stop], mirrored, rows))
        return dist

    _SEMANTIC_ROUTE_FLAG_BUFFERS: dict = {}

    @torch.no_grad()
    def _semantic_unified_reduce_device(
        self, mu: torch.Tensor, counts: list[int], weights: np.ndarray, *,
        target: int, radius_limits: list[float] | None = None, strict: bool = False,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Mutual-nearest Ward rounds on a padded FP32 [L, M, D] buffer.

        `mu` is owned by the call and updated in place: surviving rows hold the
        final centers. Returns per-lane surviving row IDs (ascending) and merge
        pairs (left, right) in round order, then (cost, left) order within a
        round. Row IDs are never compacted, so they are the input node IDs.

        Pair rules are those of the full-recompute rounds: row minima with the
        XOR tie rule, disjoint mutual pairs, exact compactness verification and
        a per-lane budget of `count - target` pairs taken by increasing cost.
        Only the evaluation changed: distances are updated incrementally, and
        the host reads one active flag per lane after each four device rounds.
        A candidate pair rejected by the exact check is pinned to its exact
        distance, so its rows may pair elsewhere in later rounds.
        """
        lanes = len(counts)
        dev = mu.device
        active = [i for i, n in enumerate(counts) if n > target]
        survivors = [np.arange(n, dtype=np.int64) for n in counts]
        traces = [np.empty((0, 2), dtype=np.int64) for _ in counts]
        if not active:
            return survivors, traces
        buffer, lane_index = mu, None
        if len(active) < lanes:
            size = max(counts[i] for i in active)
            lane_index = _upload(np.asarray(active, dtype=np.int64), dev)
            mu = buffer.index_select(0, lane_index)[:, :size].contiguous()
            weights = weights[active, :size]
        else:
            size = mu.size(1)
            weights = weights[:, :size]
        active_counts = [counts[i] for i in active]
        active_limits = None if radius_limits is None else [radius_limits[i] for i in active]
        if self._semantic_hard_cap() != math.inf:
            # Capacity fallback keeps two minima per row; use full recompute.
            results = self._semantic_unified_reduce_fallback(
                mu, mu, range(len(active)), active_counts, weights,
                target=target, radius_limits=active_limits, strict=strict,
            )
        else:
            # A stuck lane is rerun from its inputs, so keep them for the budget pass.
            results = self._semantic_unified_reduce_incremental(
                mu, active_counts, weights, target=target, radius_limits=active_limits, strict=strict,
                original=mu.clone() if strict else None,
            )
        for (roots, trace), i in zip(results, active):
            survivors[i], traces[i] = roots, trace
        if lane_index is not None:
            buffer[:, :size].index_copy_(0, lane_index, mu)
        return survivors, traces

    def _semantic_unified_reduce_incremental(self, mu, counts, weights, *, target, radius_limits, strict, original):
        """Incremental rounds, checking CUDA completion once per four rounds."""
        dev = mu.device
        size = mu.size(1)
        count = _upload(np.asarray(counts, dtype=np.int64), dev)
        # A copy: the rounds update masses in place and CPU from_numpy would alias.
        mass = _upload(np.array(weights, dtype=np.float32), dev)
        limits = None if radius_limits is None else _upload(np.asarray(radius_limits, dtype=np.float32), dev)
        dist = self._semantic_unified_pair_matrix(mu)
        fused = _triton_route() if _fused_on(dev) else None
        reducer_type = getattr(fused, "UnifiedReduce", None) or _UnifiedReduceTorch
        # Frozen-center sweeps approximate only the uncapped global merge; the
        # candidate compactness pass keeps one sweep per round.
        passes = self.semantic_merge_passes if radius_limits is None else 1
        reducer = reducer_type(dist, mu, mass, count, limits, target, passes)
        asynchronous = reducer_type is not _UnifiedReduceTorch
        if asynchronous:
            # Reducers exposing (active, live count) rows also learn a live
            # bound at each poll, which narrows later launches.
            status = getattr(reducer, "status", None)
            key = (dev, len(counts)) if status is None else (dev, len(counts), 2)
            flags = self._SEMANTIC_ROUTE_FLAG_BUFFERS.get(key)
            if flags is None:
                shape = (len(counts),) if status is None else tuple(status.shape)
                flags = torch.zeros(shape, dtype=torch.int32, pin_memory=dev.type == "cuda")
                self._SEMANTIC_ROUTE_FLAG_BUFFERS[key] = flags
            host_flags = flags.numpy()  # Pinned memory viewed without a torch op per read.
            event = torch.cuda.Event() if dev.type == "cuda" else None
            stream = torch.cuda.current_stream(dev) if event is not None else None
        # Every round before the last merges or pins a pair; the cap is a guard.
        last_round = 2 * size + 7
        # Without radius limits a lane only stops at TARGET, and a round at
        # most halves it, so earlier polls cannot observe completion.
        bounded = asynchronous and status is not None and radius_limits is None
        next_poll = max(4, self._semantic_min_rounds(max(counts), target)) if bounded else 4
        for round_ in range(1, last_round + 1):
            flag = self._semantic_unified_round(reducer, round_)
            if not asynchronous:
                if not bool(flag.any()):
                    break
                continue
            # Round order and pair selection stay unchanged. Once inactive,
            # a lane cannot merge again: it reached TARGET or has no proposals
            # and no dirty rows to rescan. At most three extra no-op rounds
            # trade small launches for 4x fewer host waits and flag transfers.
            if round_ < next_poll and round_ != last_round:
                continue
            flags.copy_(flag if status is None else status, non_blocking=event is not None)
            if event is not None:
                event.record(stream)
                event.synchronize()
            next_poll = round_ + 4
            if status is not None:
                live = int(host_flags[1].max())
                reducer.limit_live(live)
                if bounded:
                    next_poll = round_ + max(4, self._semantic_min_rounds(live, target))
            if not (host_flags if status is None else host_flags[0]).any():
                break
        lane_traces, alive, stuck = reducer.finish()
        results = [(np.flatnonzero(alive[lane]), lane_traces[lane]) for lane in range(len(counts))]
        stuck_lanes = np.flatnonzero(stuck).tolist() if strict else []
        if stuck_lanes:
            # A symmetric D always has a mutual pair while any cost is finite,
            # so this should be unreachable. Rather than abort training, redo
            # those lanes with full recompute (which raises if nothing is finite).
            if not getattr(LogStructuredKVCache, "_semantic_warned_stuck", False):
                LogStructuredKVCache._semantic_warned_stuck = True
                warnings.warn("incremental unified routing found no mutual pair below the cluster "
                              "budget; recomputing those groups with full-recompute rounds")
            redo = self._semantic_unified_reduce_fallback(
                original, mu, stuck_lanes, counts, weights, target=target, radius_limits=radius_limits, strict=True,
            )
            for lane, result in zip(stuck_lanes, redo):
                results[lane] = result
        return results

    @staticmethod
    def _semantic_min_rounds(count: int, target: int) -> int:
        """Fewest disjoint-pair rounds that can take `count` clusters to `target`."""
        rounds = 0
        while count > target:
            count = max((count + 1) // 2, target)
            rounds += 1
        return rounds

    def _semantic_unified_reduce_fallback(self, source, out, lanes, counts, weights, *, target, radius_limits, strict):
        """Full-recompute rounds for selected lanes of a padded buffer.

        Reads inputs from `source`, writes final centers into `out` (they may
        alias) and returns (survivors, trace) per lane like the incremental path.
        """
        lanes = list(lanes)
        reduced = self._semantic_unified_reduce_rounds(
            [weights[lane, :counts[lane]] for lane in lanes],
            [source[lane, :counts[lane]] for lane in lanes],
            max_clusters=target if strict else None,
            radius_limits=None if radius_limits is None else [radius_limits[lane] for lane in lanes],
        )
        results = []
        for lane, (roots, trace, center) in zip(lanes, reduced):
            if trace:
                out[lane].index_copy_(0, torch.from_numpy(roots).to(out.device), center.to(out.dtype))
            results.append((roots, np.concatenate(trace) if trace else np.empty((0, 2), dtype=np.int64)))
        return results

    def _semantic_unified_round(self, reducer, round_):
        """One device round; separate method so profilers can label it."""
        return reducer.step(round_)

    @torch.no_grad()
    def _semantic_unified_reduce_arrays(
        self, groups: list[np.ndarray], centroids: list[torch.Tensor | None], *,
        max_clusters: int | None = None, radius_limits: list[float] | None = None,
    ) -> list[tuple[np.ndarray, list[np.ndarray], torch.Tensor | None]]:
        """Device centers plus numeric host roots; no per-token Python nodes."""
        target = 1 if max_clusters is None else max_clusters
        active = [i for i, nodes in enumerate(groups) if len(nodes) > target]
        active_set = set(active)
        outputs = [None] * len(groups)
        for i, nodes in enumerate(groups):
            if i not in active_set:
                outputs[i] = (np.arange(len(nodes), dtype=np.int64), [], centroids[i])
        if not active:
            return outputs
        counts = [len(groups[i]) for i in active]
        size = max(counts)
        mu = torch.stack([F.pad(centroids[i].float(), (0, 0, 0, size - n)) for i, n in zip(active, counts)])
        weights = np.zeros((len(active), size), dtype=np.float32)
        for lane, i in enumerate(active):
            weights[lane, :counts[lane]] = groups[i]
        survivors, traces = self._semantic_unified_reduce_device(
            mu, counts, weights, target=target, strict=max_clusters is not None,
            radius_limits=None if radius_limits is None else [radius_limits[i] for i in active],
        )
        for lane, i in enumerate(active):
            if len(traces[lane]):
                index = torch.from_numpy(survivors[lane]).to(mu.device)
                outputs[i] = (survivors[lane], [traces[lane]], mu[lane].index_select(0, index))
            else:
                outputs[i] = (survivors[lane], [], centroids[i])
        return outputs

    @torch.no_grad()
    def _semantic_unified_reduce_rounds(
        self, groups: list[np.ndarray], centroids: list[torch.Tensor | None], *,
        max_clusters: int | None = None, radius_limits: list[float] | None = None,
    ) -> list[tuple[np.ndarray, list[np.ndarray], torch.Tensor | None]]:
        """Full-recompute rounds; kept for the capacity-capped fallback."""
        target = 1 if max_clusters is None else max_clusters
        outputs = [None] * len(groups)
        active = [i for i, nodes in enumerate(groups) if len(nodes) > target]
        for i, nodes in enumerate(groups):
            if i not in active:
                outputs[i] = (np.arange(len(nodes), dtype=np.int64), centroids[i])
        if not active:
            return [(root, [], center) for root, center in outputs]
        counts = [len(groups[i]) for i in active]
        size = max(counts)
        dev = centroids[active[0]].device
        mu = torch.stack([F.pad(centroids[i].float(), (0, 0, 0, size - len(groups[i]))) for i in active])
        weights = np.zeros((len(active), size), dtype=np.float32)
        for lane, group in enumerate(active):
            weights[lane, :counts[lane]] = groups[group]
        mass = torch.from_numpy(weights).to(dev)
        radius = torch.zeros_like(mass)
        roots = [np.arange(n, dtype=np.int64) for n in counts]
        traces = [[] for _ in groups]
        cap = self._semantic_hard_cap()
        while active:
            limits = (None if radius_limits is None else
                      torch.tensor([radius_limits[i] for i in active], device=dev))
            pairs = self._semantic_unified_round_pairs(mu, mass, radius, limits, cap, max_clusters is not None)
            # NumPy views avoid materializing all padded pairs as Python objects.
            pair_rows = pairs.cpu().numpy()
            valid = pair_rows[:, :, 0] >= 0
            valid &= valid.cumsum(axis=1) <= (np.asarray(counts)[:, None] - target)
            stride = mu.size(1)
            pending = []
            for lane, group in enumerate(active):
                selected = pair_rows[lane, valid[lane]]
                if not len(selected):
                    if max_clusters is not None:
                        raise RuntimeError("unified routing has no finite merge cost to satisfy the cluster budget")
                    outputs[group] = (roots[lane], mu[lane, :counts[lane]] if traces[group] else centroids[group])
                    continue
                # Store original node IDs in round/cost order, then rebuild once.
                traces[group].append(roots[lane][selected])
                keep = np.ones(counts[lane], dtype=bool)
                keep[selected[:, 1]] = False
                surviving = np.flatnonzero(keep)
                partner = np.full(counts[lane], -1, dtype=np.int64)
                partner[selected[:, 0]] = lane * stride + selected[:, 1]
                pending.append((group, roots[lane][surviving], lane * stride + surviving, partner[surviving]))
            if not pending:
                break
            # Pack continuing groups first: a prefix view retires completed lanes
            # without three extra index_select kernels and another index upload.
            running = [entry for entry in pending if len(entry[1]) > target]
            finished = [entry for entry in pending if len(entry[1]) <= target]
            pending = running + finished
            size = max(len(entry[1]) for entry in pending)
            mapping = np.full((2, len(pending), size), -1, dtype=np.int64)
            for lane, (_, root, src, partner) in enumerate(pending):
                mapping[0, lane, :len(root)] = src
                mapping[1, lane, :len(root)] = partner
            # One array upload describes both merging and compaction. The device
            # reads old buffers and writes new ones, so disjoint pairs cannot race.
            indices = torch.from_numpy(mapping).to(dev)
            mu, mass, radius = self._semantic_unified_merge_pack(
                mu, mass, radius, indices, radius_limits is not None,
            )
            for lane, (group, root, _, _) in enumerate(finished, start=len(running)):
                outputs[group] = (root, mu[lane, :len(root)])
            active = [entry[0] for entry in running]
            roots = [entry[1] for entry in running]
            counts = [len(root) for root in roots]
            mu, mass, radius = (t[:len(running)] for t in (mu, mass, radius))
        return [(root, trace, center) for (root, center), trace in zip(outputs, traces)]

    def _semantic_unified_reduce(
        self, clusters: list[_SemanticTreeCluster], *,
        max_clusters: int | None = None, radius_limit: float | None = None,
    ) -> tuple[list[_SemanticTreeCluster], list[tuple[int, int]]]:
        return self._semantic_unified_reduce_batch(
            [clusters], max_clusters=max_clusters,
            radius_limits=None if radius_limit is None else [radius_limit],
        )[0]

    def _semantic_unified_candidates(
        self, b: int, g: int, k_raw: torch.Tensor, positions_host: list[list[int]]
    ) -> list[_SemanticTreeCluster]:
        nodes = [_SemanticTreeCluster(key, 1, positions_host[b][i], (), (i,))
                 for i, key in enumerate(k_raw[b, g].detach().float().unbind(0))]
        return self._semantic_unified_reduce(nodes, radius_limit=math.sqrt(self._semantic_tree_threshold(b, g)))[0]

    @staticmethod
    def _semantic_unified_labels(size, surviving, traces):
        """Map one lane's leaves to compact surviving rows; see the batched form."""
        trace = np.concatenate(traces) if traces else np.empty((0, 2), dtype=np.int64)
        return LogStructuredKVCache._semantic_unified_labels_batch([size], [surviving], [trace])[0]

    @staticmethod
    def _semantic_unified_labels_batch(sizes, survivors, traces):
        """Resolve merge forests in parallel, then map leaves to compact rows.

        Every right root is retired exactly once. Its parent is always a smaller
        original root, so pointer jumping terminates without a token-wise walk.
        Lanes never share nodes, so they are offset into one forest and
        resolved by a single pass.
        """
        sizes = np.asarray(sizes, dtype=np.int64)
        offsets = np.cumsum(sizes) - sizes
        parent = np.arange(int(sizes.sum()), dtype=np.int64)
        pairs = [trace + offset for trace, offset in zip(traces, offsets.tolist()) if len(trace)]
        if pairs:
            pairs = np.concatenate(pairs)
            parent[pairs[:, 1]] = pairs[:, 0]
            while True:
                grandparent = parent[parent]
                if np.array_equal(parent, grandparent):
                    break
                parent = grandparent
        compact = np.empty(len(parent), dtype=np.int64)
        compact[np.concatenate([np.asarray(rows, dtype=np.int64) + offset
                                for rows, offset in zip(survivors, offsets.tolist())])] = np.concatenate(
            [np.arange(len(rows), dtype=np.int64) for rows in survivors])
        return np.split(compact[parent], offsets[1:])

    # CUDA+Triton: 1 GiB holds the FP32 distances of 32 lanes x ~2.3K nodes
    # (batch 4 x 8 groups, flush plus evicted exact tokens), so the training
    # shape routes in one tile: one round sequence, poll and GEMM per stage.
    # LOGKV_ROUTE_TILE_MB lowers it when that peak does not fit next to the
    # activations. The Torch reference also materializes row temporaries.
    _SEMANTIC_ROUTE_TILE_BYTES = {True: int(os.environ.get("LOGKV_ROUTE_TILE_MB", "1024")) << 20, False: 64 << 20}
    _semantic_fused_distances = True

    def _semantic_unified_tile(self, lanes: int, tokens: int, device: torch.device) -> int:
        """Largest balanced lane tile whose FP32 distance matrix fits the budget.

        Every round costs a fixed number of launches per tile, so fewer, wider
        tiles are faster; balanced tiles avoid a nearly empty last tile.
        """
        fused = _fused_on(device) and _triton_route() is not None
        widest = max(1, self._SEMANTIC_ROUTE_TILE_BYTES[fused] // max(1, 4 * tokens * tokens))
        tiles = -(-lanes // widest)
        return max(1, -(-lanes // max(tiles, 1)))

    def _semantic_route_unified(
        self, k_raw: torch.Tensor, v: torch.Tensor, positions: torch.Tensor,
        positions_host: list[list[int]], *, record: bool, token_counts=None,
    ) -> None:
        jobs, routes, merge_jobs, new_jobs = [], [], [], []
        n_batch, n_groups, width_all, dim = k_raw.shape
        # AlphaLogKV archives a different number of tokens per batch row; rows
        # beyond a lane's count are padding with zero mass, and lanes of empty
        # rows cost no candidate pass, labels, global pass or host loops.
        token_counts = [width_all] * n_batch if token_counts is None else [int(n) for n in token_counts]
        lanes = [(b, g) for b in range(n_batch) for g in range(n_groups) if token_counts[b]]
        if not lanes:
            return
        tile_size = self._semantic_unified_tile(len(lanes), max(token_counts), k_raw.device)
        position_order = [np.argsort(row[:n], kind="stable") for row, n in zip(positions_host, token_counts)]
        keys = k_raw.detach().flatten(0, 1)
        centroids = self.centroid.flatten(0, 2)
        for start in range(0, len(lanes), tile_size):
            tile = lanes[start:start + tile_size]
            width = len(tile)
            candidate_counts = [token_counts[b] for b, _ in tile]
            tokens = max(candidate_counts)
            flat = [b * n_groups + g for b, g in tile]
            # A contiguous tile is a slice; otherwise gather its lanes once.
            source = (keys[flat[0]:flat[0] + width] if flat[-1] - flat[0] == width - 1
                      else keys.index_select(0, _upload(np.asarray(flat, dtype=np.int64), k_raw.device)))
            # Owned FP32 copy narrowed to this tile's widest lane: the reducer
            # writes centers in place, and padding columns cost M^2 work.
            mu = torch.empty((width, tokens, dim), dtype=torch.float32, device=k_raw.device)
            mu.copy_(source[:, :tokens])
            candidate_weights = (np.arange(tokens)[None, :] < np.asarray(candidate_counts)[:, None]).astype(np.float32)
            candidates, candidate_traces = self._semantic_unified_reduce_device(
                mu, candidate_counts, candidate_weights, target=1,
                radius_limits=[math.sqrt(self._semantic_tree_threshold(b, g)) for b, g in tile],
            )
            live = [np.asarray(self._semantic_live_clusters(b, g), dtype=np.int64) for b, g in tile]
            labels = self._semantic_unified_labels_batch(candidate_counts, candidates, candidate_traces)
            weights = []
            for (b, g), old, roots, label in zip(tile, live, candidates, labels):
                weights.append(np.concatenate((
                    np.asarray(self._semantic_n_total[b][g], dtype=np.float32)[old],
                    np.bincount(label, minlength=len(roots)).astype(np.float32),
                )))
            counts = [len(weight) for weight in weights]
            size = max(counts)
            # Old centroids then candidate centers, padded per lane: one upload,
            # two gathers, instead of a cat/index_select/pad chain per lane.
            src_old = np.concatenate([(b * n_groups + g) * self.K_max + old for (b, g), old in zip(tile, live)])
            dst_old = np.concatenate([lane * size + np.arange(len(old)) for lane, old in enumerate(live)])
            src_new = np.concatenate([lane * tokens + roots for lane, roots in enumerate(candidates)])
            dst_new = np.concatenate([lane * size + len(old) + np.arange(len(roots))
                                      for lane, (old, roots) in enumerate(zip(live, candidates))])
            index = _upload(np.concatenate((src_old, dst_old, src_new, dst_new)), k_raw.device)
            n_old, n_new = len(src_old), len(src_new)
            centers = mu.new_zeros((width * size, dim))
            if n_old:
                centers.index_copy_(0, index[n_old:2 * n_old], centroids.index_select(0, index[:n_old]).float())
            centers.index_copy_(0, index[2 * n_old + n_new:],
                                mu.view(-1, dim).index_select(0, index[2 * n_old:2 * n_old + n_new]))
            padded = np.zeros((width, size), dtype=np.float32)
            for lane, weight in enumerate(weights):
                padded[lane, :len(weight)] = weight
            plans = self._semantic_unified_reduce_device(
                centers.view(width, size, dim), counts, padded, target=self.K_max, strict=True,
            )
            merged_labels = self._semantic_unified_labels_batch(counts, *plans)
            for (b, g), old, candidate_labels, roots, trace, label in zip(tile, live, labels, *plans, merged_labels):
                # Old roots precede candidates; preserve their merge order.
                old_trace = trace[trace[:, 1] < len(old)]
                merge_jobs.append((b, g, [tuple(pair) for pair in old[old_trace].tolist()]))
                routes.append((b, g, old, label[len(old) + candidate_labels], roots))
        for step in range(max((len(pairs) for _, _, pairs in merge_jobs), default=0)):
            self._semantic_ward_merge_batch(
                [(b, g, *pairs[step]) for b, g, pairs in merge_jobs if step < len(pairs)], record=record,
            )
        for b, g, old, assignment, roots in routes:
            free_slots = iter(self._semantic_free_clusters(b, g))
            # Group tokens by cluster with one stable sort; each group keeps
            # increasing-position order, as the per-cluster boolean filter did.
            order = position_order[b]
            by_cluster = assignment[order]
            perm = np.argsort(by_cluster, kind="stable")
            grouped = order[perm]
            bounds = np.searchsorted(by_cluster[perm], np.arange(len(roots) + 1))
            for cluster, root in enumerate(roots):
                offsets = grouped[bounds[cluster]:bounds[cluster + 1]]
                if not len(offsets):
                    continue
                if root < len(old):
                    target = int(old[root])
                else:
                    target = next(free_slots)
                    first = int(offsets[0])
                    new_jobs.append((b, g, target, first))
                    offsets = offsets[1:]
                if len(offsets):
                    jobs.append((b, g, target, offsets))
        self._semantic_new_clusters(new_jobs, k_raw, v, positions, positions_host, record=record)
        commit = self._alpha_commit_joins if self.alpha_exact_tokens else self._semantic_commit_joins
        commit(sorted(jobs, key=lambda job: job[:3]), k_raw, v, positions, positions_host, record=record)

    def _alpha_commit_joins(self, jobs, k_raw, v, positions, positions_host, *, record):
        """Insert delayed exact evictions in position order, batching affected clusters.

        Most clusters receive some evicted exact token, so this path is common.
        Host work is whole-array NumPy, and the position order is one device
        stable sort keyed by (lane, position) instead of reading ladder
        positions back: old entries never lie past their cluster's p_hi.
        """
        if not jobs:
            return
        host_pos = np.asarray(positions_host, dtype=np.int64)
        n_jobs = len(jobs)
        lengths = np.fromiter((len(job[3]) for job in jobs), dtype=np.int64, count=n_jobs)
        offsets = np.concatenate([np.asarray(job[3], dtype=np.int64) for job in jobs])
        batch = np.fromiter((job[0] for job in jobs), dtype=np.int64, count=n_jobs)
        owner = np.repeat(np.arange(n_jobs), lengths)
        token_pos = host_pos[batch[owner], offsets]
        p_hi = np.fromiter((self._semantic_p_hi_c[b][g][c] for b, g, c, _ in jobs), dtype=np.int64, count=n_jobs)
        late = np.minimum.reduceat(token_pos, np.cumsum(lengths) - lengths) < p_hi
        flags = late.tolist()
        ordinary = [job for job, d in zip(jobs, flags) if not d]
        if not late.any():
            self._semantic_commit_joins(ordinary, k_raw, v, positions, positions_host, record=record)
            return
        delayed = [job for job, d in zip(jobs, flags) if d]
        nd, n_levels = len(delayed), self.L_alloc
        late_tokens = late[owner]
        new_len, new_off, new_pos = lengths[late], offsets[late_tokens], token_pos[late_tokens]
        lane_b = batch[late]
        lane_g = np.fromiter((g for _, g, _, _ in delayed), dtype=np.int64, count=nd)
        lane_c = np.fromiter((c for _, _, c, _ in delayed), dtype=np.int64, count=nd)
        lane_id = (lane_b * self.n_groups + lane_g) * self.K_max + lane_c
        level_counts = np.array([self._semantic_counts[b][g][c] for b, g, c, _ in delayed], dtype=np.int64)
        span_start = (lane_id * n_levels)[:, None] * self.B + np.arange(n_levels) * self.B
        src = _spans_to_index_array(np.stack((span_start, span_start + level_counts), -1).reshape(-1, 2))
        old_counts = level_counts.sum(1)
        counts = old_counts + new_len
        new_owner = np.repeat(np.arange(nd), new_len)
        # Per lane: its old entries (ladder order) then its new tokens, as in
        # the former per-lane concatenation; `rank` is also the new phase.
        segment = np.repeat(np.arange(nd), counts)
        rank = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        old_n = old_counts[segment]
        lane_major = np.where(rank < old_n, (np.cumsum(old_counts) - old_counts)[segment] + rank,
                              len(src) + (np.cumsum(new_len) - new_len)[segment] + rank - old_n)
        n_src, n_new, n_all = len(src), len(new_off), len(rank)
        # One upload for gathers, ordering, phases and centroid metadata.
        packed = _upload(np.concatenate((src, lane_b[new_owner], lane_g[new_owner], new_off, lane_id, new_len,
                                         lane_major, segment, rank)), k_raw.device)
        cut = np.cumsum([n_src, 3 * n_new, nd, nd, n_all, n_all])
        src_t = packed[:cut[0]]
        bi, gi, ti = packed[cut[0]:cut[1]].view(3, n_new).unbind(0)
        ids, lengths_t = packed[cut[1]:cut[2]], packed[cut[2]:cut[3]]
        lane_major_t, segment_t, phase_t = packed[cut[3]:cut[4]], packed[cut[4]:cut[5]], packed[cut[5]:]
        fields = self._semantic_flat_level_fields()
        nk, nv, np_ = k_raw[bi, gi, ti], v[bi, gi, ti], positions[bi, ti]
        new = (nk, nv, self.level_w.new_ones(n_new), *self._empty_stats(nk, nv),
               np_, np_, np_, self.level_order.new_zeros(n_new), self.pad_mask.new_zeros(n_new))
        combined = tuple(torch.cat((field[src_t], value), 0) if field is not None else None
                         for field, value in zip(fields, new))
        # Positions are nonnegative and below 2**40; the lane id leads the key.
        key = (segment_t << 40) | combined[8].index_select(0, lane_major_t)
        order = lane_major_t.index_select(0, torch.argsort(key, stable=True))
        # Delayed rows as (source, row index) pairs; the phases are final.
        tail = tuple(None if x is None else (x, order) for x in combined[:11]) + ((phase_t, None), (combined[12], order))
        new_start = np.cumsum(new_len) - new_len
        highs = np.maximum(p_hi[late], np.maximum.reduceat(new_pos, new_start)).tolist()
        sums = torch.segment_reduce(nk.float(), "sum", lengths=lengths_t, unsafe=True)
        mu, ne = self.centroid.flatten(0, 2), self.n_eff.flatten()
        pre = ne[ids]
        updated = (pre[:, None] * mu[ids] + sums) / (pre + lengths_t)[:, None]
        lanes = [job[:3] for job in delayed]
        totals = [self._semantic_n_total[b][g][c] + n for (b, g, c), n in zip(lanes, new_len.tolist())]
        self._semantic_clear_clusters(lanes)
        mu.index_copy_(0, ids, updated)
        ne.index_copy_(0, ids, pre + lengths_t)
        counts = counts.tolist()
        self._set_semantic_scalars("alive", lanes, [True] * nd)
        self._set_semantic_scalars("n_total", lanes, totals)
        self._set_semantic_scalars("p_hi_c", lanes, highs)
        self._set_semantic_scalars("level0_phase", lanes, counts)
        for b, g, _ in lanes:
            self._semantic_ward_dirty[b][g] = True
        # Ordinary and delayed lanes are disjoint and the delayed rows were
        # gathered before the clear, so one batched ladder append serves both.
        # Ordinary op rows still precede the delayed ones within each lane.
        if ordinary:
            self._semantic_commit_joins(ordinary, k_raw, v, positions, positions_host, record=record,
                                        tail=(lanes, counts, tail))
        else:
            self._semantic_append_entries_batched(lanes, counts, tuple(
                None if part is None else self._semantic_tail_rows(*part) for part in tail))
        if record:
            rows = np.empty((n_new, 4), dtype=np.int64)
            rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3] = LOG_KV_OP_JOIN, lane_c[new_owner], 0, new_pos
            lane_rows = (lane_b * self.n_groups + lane_g)[new_owner]
            bounds = np.concatenate(([0], np.flatnonzero(lane_rows[1:] != lane_rows[:-1]) + 1, [n_new]))
            for lo, hi in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
                self._record_ops(int(lane_b[new_owner[lo]]), int(lane_g[new_owner[lo]]), rows[lo:hi])

    @staticmethod
    def _semantic_tail_rows(source, index, out=None):
        """Rows `index` of `source` (all rows when None), optionally into `out`."""
        if index is None:
            return source if out is None else out.copy_(source)
        if out is None:
            return source.index_select(0, index)
        if out.dtype == source.dtype:
            return torch.index_select(source, 0, index, out=out)
        return out.copy_(source.index_select(0, index))

    @classmethod
    def _semantic_stack_rows(cls, head, part):
        """Head rows then tail rows in one staging tensor; dtypes promote exactly."""
        source, index = part
        rows = source.size(0) if index is None else index.numel()
        out = head.new_empty((head.size(0) + rows, *head.shape[1:]), dtype=torch.promote_types(head.dtype, source.dtype))
        out[:head.size(0)].copy_(head)
        cls._semantic_tail_rows(source, index, out[head.size(0):])
        return out

    # Prefetching only hides GPU latency; tests enable it on CPU to compare paths.
    _alpha_prefetch_on_cpu = False

    # Route decisions beside the attention on a side stream (CUDA, attach route).
    _route_overlap = os.environ.get("LOGKV_ROUTE_OVERLAP", "0") == "1"
    _ROUTE_STREAMS: dict = {}

    def _route_decision_stream(self, k_raw, v):
        """High-priority side stream ordered after the previous flush, or None.

        A flush's decisions read only its archived K/V and the cluster state
        the previous flush committed, so they need not wait for the attention
        queued since then: the host reads them back and plans the commit while
        that attention runs. Commits return to the main stream, after it.
        This holds only for the flush loop's own window, and only when every
        row it archives was already in the window when the event was recorded;
        rows appended after it (an unaligned chunk) keep the main stream.
        """
        device = k_raw.device
        ready = getattr(self, "_route_ready", None)
        if (not self._route_overlap or device.type != "cuda" or ready is None or self.semantic_unified_route
                or self.semantic_legacy_route or self._semantic_hard_cap() != math.inf or self.K_max == 1
                or not self._semantic_attach_one_transfer or getattr(self, "_semantic_scalar_dirty", None)
                or k_raw.data_ptr() != self.recent_k_raw.data_ptr() or v.data_ptr() != self.recent_v.data_ptr()
                or k_raw.size(2) > self._route_ready_rows):
            return None
        stream = self._ROUTE_STREAMS.get(device)
        if stream is None:
            stream = self._ROUTE_STREAMS[device] = torch.cuda.Stream(device, priority=-1)
        stream.wait_event(ready)
        return stream

    def _route_mark_ready(self, device) -> None:
        """Record where the next flush's decisions may start (end of this flush)."""
        if self._route_overlap and device.type == "cuda":
            if getattr(self, "_route_ready", None) is None:
                self._route_ready = torch.cuda.Event()
            self._route_ready.record(torch.cuda.current_stream(device))
            self._route_ready_rows = self.recent_count

    @torch.no_grad()
    def _alpha_prefetch_scores(self) -> None:
        """Score the next flush's candidate spans now, off the critical path.

        Right after a flush, the next flush's inputs are fixed: the exact pool,
        the recent block that will be flushed and, for Beta novelty, the cluster
        centers and alive flags (only flushes, resets and replay restores change
        them, and each replaces the pool lists). Scoring here and copying into
        pinned memory lets the next selection read the scores without waiting
        for the attention queued in between; the selection is the same.
        """
        from models._vendor.semanticlogkv.litgpt.alpha_log_kv import cut_spans, score_spans
        self._alpha_prefetch = None
        take = self.semantic_flush_granularity
        cuda = self.recent_k_raw.is_cuda
        if self.recent_count < take or not (cuda or self._alpha_prefetch_on_cpu):
            return
        positions_host = [row[:take] for row in self._recent_pos_host]
        span_ends = [row[:take] for row in self._recent_span_ends]
        width = self.alpha_count
        cut = cut_spans(self._alpha_spans, self._alpha_positions, positions_host, span_ends,
                        width, self.alpha_span_max_tokens)
        if cut is None:
            return
        keys = torch.cat((self.alpha_k_raw[:, :, :width], self.recent_k_raw[:, :, :take]), dim=2)
        values = torch.cat((self.alpha_v[:, :, :width], self.recent_v[:, :, :take]), dim=2)
        device_scores = score_spans(keys, values, cut, beta_novelty=self.beta_novelty,
                                    centroids=self.centroid, centroid_valid=self.alive)
        scores = torch.empty(device_scores.shape, dtype=torch.float32, pin_memory=cuda)
        scores.copy_(device_scores, non_blocking=cuda)
        ready = torch.cuda.Event() if cuda else None
        if cuda:
            ready.record(torch.cuda.current_stream(keys.device))
        # The pool lists are replaced on every flush, reset and replay restore,
        # so their identity is the generation of the state the scores belong to.
        self._alpha_prefetch = (self._alpha_spans, self._alpha_positions, width, positions_host, span_ends,
                                cut, scores, ready)

    def _alpha_take_prefetch(self, old_width, positions_host, span_ends):
        """Prefetched (cut, host scores) for exactly this flush, else None."""
        prefetch, self._alpha_prefetch = getattr(self, "_alpha_prefetch", None), None
        if prefetch is None:
            return None
        spans, pool_positions, width, host, ends, cut, scores, ready = prefetch
        if not (spans is self._alpha_spans and pool_positions is self._alpha_positions and width == old_width
                and host == positions_host and ends == span_ends):
            return None
        # The event has almost always passed: this read does not wait on attention.
        if ready is not None:
            ready.synchronize()
        return cut, scores.tolist()

    @torch.no_grad()
    def _alpha_route_flush(self, k, v, positions, positions_host, span_ends, *, record):
        from models._vendor.semanticlogkv.litgpt.alpha_log_kv import choose_spans, select_spans
        if span_ends is None:
            raise ValueError("AlphaLogKV requires natural span boundaries")
        old_width, old_positions = self.alpha_count, self._alpha_positions
        side = self._route_decision_stream(k, v)
        with torch.cuda.stream(side) if side is not None else contextlib.nullcontext():
            # Snapshots: the pool is rewritten below from these sources.
            keys = torch.cat((self.alpha_k_raw[:, :, :old_width], k), dim=2)
            values = torch.cat((self.alpha_v[:, :, :old_width], v), dim=2)
            pos = torch.cat((self.alpha_pos[:, :old_width], positions), dim=1)
            with logkv_timed("alpha_select"):
                prefetched = self._alpha_take_prefetch(old_width, positions_host, span_ends)
                if prefetched is not None:
                    selected = choose_spans(*prefetched, self.alpha_exact_tokens)
                else:
                    selected = select_spans(keys, values, self._alpha_spans, self._alpha_positions,
                                            positions_host, span_ends, old_width, self.alpha_exact_tokens,
                                            self.alpha_span_max_tokens, beta_novelty=self.beta_novelty,
                                            centroids=self.centroid, centroid_valid=self.alive)

        def gather(rows, width, out=None):
            offsets = np.zeros((self.batch_size, width), dtype=np.int64)
            valid = np.zeros_like(offsets, dtype=np.bool_)
            for b, row in enumerate(rows):
                offsets[b, :len(row)] = row
                valid[b, :len(row)] = True
            idx = _upload(offsets, k.device)
            mask = _upload(valid, k.device)
            ki = idx[:, None, :, None].expand(-1, self.n_groups, -1, self.k_dim)
            vi = idx[:, None, :, None].expand(-1, self.n_groups, -1, self.v_dim)
            result = (torch.gather(keys, 2, ki, out=None if out is None else out[0]),
                      torch.gather(values, 2, vi, out=None if out is None else out[1]),
                      torch.gather(pos, 1, idx, out=None if out is None else out[2]))
            for value in result[:2]:
                value.masked_fill_(~mask[:, None, :, None], 0)
            result[2].masked_fill_(~mask, 0)
            return (*result, mask)

        counts = list(map(len, selected.archive))
        width = max(counts, default=0)
        fused = _triton_updates() if _fused_on(keys) else None
        if side is not None:
            # Archive rows come from the snapshots on the side stream; the pool
            # itself is rewritten on the main stream once the queued attention,
            # which reads it, has run (after the route's decisions).
            with torch.cuda.stream(side):
                if width:
                    ak, av, ap, _ = gather(selected.archive, width)
        elif fused is not None:
            # One index upload and one copy kernel for both destinations,
            # including zero-filled ragged rows. The cat sources do not alias
            # the exact pool we overwrite, so no cross-CTA read/write race.
            offsets = np.full((self.batch_size, self.alpha_exact_tokens + width), -1, dtype=np.int64)
            for b, (keep, archive) in enumerate(zip(selected.keep, selected.archive)):
                offsets[b, :len(keep)] = keep
                offsets[b, self.alpha_exact_tokens:self.alpha_exact_tokens + len(archive)] = archive
            ak, av, ap = fused.alpha_partition(
                keys, values, pos, _upload(offsets, k.device),
                (self.alpha_k_raw, self.alpha_v, self.alpha_pos, self.alpha_valid),
            )
        else:
            *_, valid = gather(selected.keep, self.alpha_exact_tokens,
                               out=(self.alpha_k_raw, self.alpha_v, self.alpha_pos))
            self.alpha_valid.copy_(valid)
            if width:
                ak, av, ap, _ = gather(selected.archive, width)
        self.alpha_count = max(map(len, selected.keep), default=0)
        self._alpha_spans, self._alpha_positions = selected.spans, selected.positions
        if width:
            # [old pool | flush] host positions, gathered for the archive rows.
            host = np.zeros((self.batch_size, old_width + k.size(2)), dtype=np.int64)
            for b, old in enumerate(old_positions):
                host[b, :len(old)] = old
            host[:, old_width:] = positions_host
            rows = np.zeros((self.batch_size, width), dtype=np.int64)
            for b, row in enumerate(selected.archive):
                rows[b, :len(row)] = row
            live = np.arange(width) < np.asarray(counts)[:, None]
            archive_host = np.where(live, np.take_along_axis(host, rows, 1), 0).tolist()
            if self.semantic_unified_route:
                self._semantic_route_unified(ak, av, ap, archive_host, record=record, token_counts=counts)
            else:
                self._semantic_route_three_phase(ak, av, ap, archive_host, record=record, token_counts=counts,
                                                 stream=side)
        if side is not None:
            main = torch.cuda.current_stream(k.device)
            main.wait_stream(side)
            for tensor in (keys, values, pos):
                tensor.record_stream(main)
            keep = np.full((self.batch_size, self.alpha_exact_tokens), -1, dtype=np.int64)
            for b, row in enumerate(selected.keep):
                keep[b, :len(row)] = row
            if fused is not None:
                fused.alpha_partition(keys, values, pos, _upload(keep, k.device),
                                      (self.alpha_k_raw, self.alpha_v, self.alpha_pos, self.alpha_valid))
            else:
                *_, valid = gather(selected.keep, self.alpha_exact_tokens,
                                   out=(self.alpha_k_raw, self.alpha_v, self.alpha_pos))
                self.alpha_valid.copy_(valid)

    def _semantic_commit_joins(
        self,
        jobs: list[tuple[int, int, int, tuple[int, ...]]],
        k_raw: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        positions_host: list[list[int]],
        *,
        record: bool = False,
        tail=None,
    ) -> None:
        """Join many (b, g, cluster) -> token-offset assignments in one pass.

        `jobs` must already be in op-log order (b, then g, then cluster); each
        job's offsets are sorted by position here. Segment splitting, pad
        alignment and the hard cap keep the exact host-side rules the per-cluster
        path used -- the difference is that every lane's entries land in a single
        staging block, so one batched ladder append and one batched centroid
        update serve all of them. An optional `tail` of (lanes, counts, row
        sources) for other, already prepared lanes joins that ladder append.
        """
        if not jobs:
            return
        with self._semantic_deferred_scalars():
            self._semantic_commit_joins_inner(jobs, k_raw, v, positions, positions_host, record=record, tail=tail)

    def _semantic_commit_joins_inner(
        self,
        jobs: list[tuple[int, int, int, tuple[int, ...]]],
        k_raw: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        positions_host: list[list[int]],
        *,
        record: bool = False,
        tail=None,
    ) -> None:
        T, n_groups = k_raw.size(2), k_raw.size(1)
        host_pos = np.asarray(positions_host, dtype=np.int64)
        seg_on = self.seg_gap_max != math.inf
        if not seg_on:
            self._semantic_commit_joins_unsegmented(jobs, host_pos, k_raw, v, positions, record=record, tail=tail)
            return
        align = 1 << self.seg_block_level
        lanes, lane_counts, runs, pending_ops = [], [], [], []
        at_parts, src_parts, order_parts = [], [], []
        total = 0
        for b, g, c, offsets in jobs:
            if not len(offsets):
                continue
            ordered = np.asarray(offsets, dtype=np.int64)
            pos = host_pos[b, ordered]
            if np.any(pos[1:] < pos[:-1]):
                sort = np.argsort(pos, kind="stable")
                ordered, pos = ordered[sort], pos[sort]
            phase = self._semantic_level0_phase[b][g][c]
            seg_cur = self._semantic_current_segment[b][g][c]
            prev_hi = self._semantic_p_hi_c[b][g][c]
            kv_base = (b * n_groups + g) * T
            cluster_flat = (b * n_groups + g) * self.K_max + c
            lane_start = total
            # Only segment boundaries require host decisions. All token indexing
            # within a run is constructed by NumPy, including replay op rows.
            new_segment = np.zeros(len(ordered), dtype=bool)
            if seg_on:
                new_segment[0] = pos[0] - prev_hi > self.seg_gap_max
                new_segment[1:] = np.diff(pos) > self.seg_gap_max
            boundaries = np.concatenate(([0], np.flatnonzero(new_segment[1:]) + 1, [len(ordered)]))
            ops = []
            for ordinal, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                new = bool(new_segment[lo])
                if new:
                    pad_count = (-phase) % align
                    if pad_count:
                        order_parts.append(np.arange(phase, phase + pad_count, dtype=np.int64))
                        total += pad_count
                        phase += pad_count
                        if record:
                            ops.append(np.array([[LOG_KV_OP_PAD_INSERT, c, 0, pad_count]], dtype=np.int64))
                    seg_cur += 1
                segment = seg_cur if seg_on else 0
                count = int(hi - lo)
                at_parts.append(np.arange(total, total + count, dtype=np.int64))
                src_parts.append(kv_base + ordered[lo:hi])
                order_parts.append(np.arange(phase, phase + count, dtype=np.int64))
                runs.append([ordinal, cluster_flat, count, new])
                if record:
                    rows = np.empty((count, 4), dtype=np.int64)
                    rows[:, 0] = LOG_KV_OP_JOIN
                    if new:
                        rows[0, 0] = LOG_KV_OP_NEW_SEGMENT
                    rows[:, 1], rows[:, 2], rows[:, 3] = c, segment, pos[lo:hi]
                    ops.append(rows)
                total += count
                phase += count
            lanes.append((b, g, c))
            lane_counts.append(total - lane_start)
            self._set_semantic_level0_phase(b, g, c, phase)
            self._set_semantic_p_hi(b, g, c, int(pos[-1]))
            self._set_semantic_n_total(b, g, c, self._semantic_n_total[b][g][c] + len(ordered))
            self._set_semantic_current_segment(b, g, c, seg_cur)
            if ops:
                pending_ops.append((b, g, np.concatenate(ops)))
        if lanes:
            self._semantic_apply_join_plan(
                lanes, lane_counts, np.concatenate(at_parts), np.concatenate(src_parts), np.concatenate(order_parts),
                runs, pending_ops, total, k_raw, v, positions, tail=tail,
            )
        elif tail is not None:
            raise RuntimeError("a join tail needs at least one joined lane")

    def _semantic_commit_joins_unsegmented(self, jobs, host_pos, k_raw, v, positions, *, record, tail=None):
        """Unsegmented joins for every (b, g, cluster) with whole-array NumPy.

        Without segment gaps each job is one run with no pads, so staging rows
        are the identity and only per-job scalars need Python. Row order, run
        order, op-log rows and host counters equal the per-job loop.
        """
        jobs = [job for job in jobs if len(job[3])]
        if not jobs:
            if tail is not None:
                raise RuntimeError("a join tail needs at least one joined lane")
            return
        T, n_groups = k_raw.size(2), k_raw.size(1)
        lengths = np.fromiter((len(job[3]) for job in jobs), dtype=np.int64, count=len(jobs))
        starts = np.cumsum(lengths) - lengths
        owner = np.repeat(np.arange(len(jobs)), lengths)
        batch = np.fromiter((job[0] for job in jobs), dtype=np.int64, count=len(jobs))
        group = np.fromiter((job[1] for job in jobs), dtype=np.int64, count=len(jobs))
        cluster = np.fromiter((job[2] for job in jobs), dtype=np.int64, count=len(jobs))
        offsets = np.concatenate([np.asarray(job[3], dtype=np.int64) for job in jobs])
        pos = host_pos[batch[owner], offsets]
        if np.any((pos[1:] < pos[:-1]) & (owner[1:] == owner[:-1])):
            sort = np.lexsort((pos, owner))  # Stable: equal positions keep offset order.
            offsets, pos = offsets[sort], pos[sort]
        phase = np.fromiter((self._semantic_level0_phase[b][g][c] for b, g, c, _ in jobs),
                            dtype=np.int64, count=len(jobs))
        total = int(lengths.sum())
        rank = np.arange(total, dtype=np.int64) - starts[owner]
        lane_ids = batch * n_groups + group
        runs = [[0, int(flat), int(count), False]
                for flat, count in zip(lane_ids * self.K_max + cluster, lengths)]
        last = pos[starts + lengths - 1]
        # Lanes are unique here (the batched append requires it), so totals can
        # be read from the mirror up front.
        lanes = [job[:3] for job in jobs]
        totals = np.fromiter((self._semantic_n_total[b][g][c] for b, g, c in lanes),
                             dtype=np.int64, count=len(lanes)) + lengths
        self._set_semantic_scalars("level0_phase", lanes, (phase + lengths).tolist())
        self._set_semantic_scalars("p_hi_c", lanes, last.tolist())
        self._set_semantic_scalars("n_total", lanes, totals.tolist())
        # Segments are unchanged, but the per-job path rewrote them; keep the
        # same deferred device refresh.
        self._semantic_mark_scalar_dirty("current_segment")
        pending_ops = []
        if record:
            rows = np.empty((total, 4), dtype=np.int64)
            rows[:, 0] = LOG_KV_OP_JOIN
            rows[:, 1] = cluster[owner]
            rows[:, 2] = 0
            rows[:, 3] = pos
            # Jobs are sorted by (b, g, cluster): one contiguous block per lane.
            lane_of_row = lane_ids[owner]
            cuts = np.flatnonzero(lane_of_row[1:] != lane_of_row[:-1]) + 1
            for part in np.split(np.arange(total), cuts):
                job = jobs[owner[part[0]]]
                pending_ops.append((job[0], job[1], rows[part[0]:part[-1] + 1]))
        self._semantic_apply_join_plan(
            lanes, lengths.tolist(), np.arange(total, dtype=np.int64), (lane_ids * T)[owner] + offsets,
            phase[owner] + rank, runs, pending_ops, total, k_raw, v, positions, tail=tail,
        )


    def _semantic_commit_runs(
        self,
        records: list[tuple[int, int, int, int, bool, array]],
        k_raw: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        positions_host: list[list[int]],
    ) -> None:
        """Commit replayed runs -- explicit (cluster, segment) -- across lanes.

        Replay must apply the logged decisions verbatim rather than re-deriving
        segment boundaries, so this skips the gap/pad logic and only shares the
        plan applier. Each `records` entry is one recorded run and every lane
        appears at most once, which is what the batched append requires.
        """
        if not records:
            return
        T = k_raw.size(2)
        n_groups = k_raw.size(1)
        lanes: list[tuple[int, int, int]] = []
        lane_counts: list[int] = []
        blk_tok_at: list[np.ndarray] = []
        blk_tok_src: list[np.ndarray] = []
        order_vals: list[np.ndarray] = []
        runs: list[list] = []
        total = 0
        for b, g, c, segment, new_segment, offsets in records:
            phase = self._semantic_level0_phase[b][g][c]
            kv_base = (b * n_groups + g) * T
            runs.append([0, (b * n_groups + g) * self.K_max + c, len(offsets), new_segment])
            blk_tok_at.append(np.arange(total, total + len(offsets), dtype=np.int64))
            blk_tok_src.append(kv_base + np.asarray(offsets, dtype=np.int64))
            order_vals.append(np.arange(phase, phase + len(offsets), dtype=np.int64))
            total += len(offsets)
            phase += len(offsets)
            lanes.append((b, g, c))
            lane_counts.append(len(offsets))
            self._set_semantic_level0_phase(b, g, c, phase)
            self._set_semantic_p_hi(b, g, c, int(positions_host[b][offsets[-1]]))
            self._set_semantic_n_total(b, g, c, self._semantic_n_total[b][g][c] + len(offsets))
            if new_segment:
                self._set_semantic_current_segment(b, g, c, segment)
        self._semantic_apply_join_plan(
            lanes, lane_counts, np.concatenate(blk_tok_at), np.concatenate(blk_tok_src), np.concatenate(order_vals),
            runs, [], total, k_raw, v, positions,
        )

    def _semantic_apply_join_plan(
        self,
        lanes: list[tuple[int, int, int]],
        lane_counts: list[int],
        blk_tok_at: list[int] | np.ndarray,
        blk_tok_src: list[int] | np.ndarray,
        order_vals: list[int] | np.ndarray,
        runs: list[list],
        pending_ops: list[tuple[int, int, list[tuple[int, int, int, int]]]],
        total: int,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        *,
        tail=None,
    ) -> None:
        """Materialize one staging block for every lane and commit it.

        Shared by forward routing and op-log replay: the two differ only in how
        the plan is derived, never in how it lands. A `tail` of (lanes, counts,
        (source, row index) per field) is staged after these rows and shares
        the ladder append; centroid updates stay with the joined lanes.
        """
        dev = k_raw.device
        if not lanes:
            return
        for b, g, ops in pending_ops:
            self._record_ops(b, g, ops)

        # Convert logical rows to coordinates on the host and upload once.
        # Gathering directly from strided chunk views avoids a full K/V copy
        # that reshape(-1, D) would require before selecting the actual rows.
        n_tok = len(blk_tok_at)
        src = np.asarray(blk_tok_src, dtype=np.int64)
        lane, token = np.divmod(src, k_raw.size(2))
        batch, group = np.divmod(lane, k_raw.size(1))
        # Include centroid metadata in the staging upload. Runs targeting the
        # same cluster keep their ordinal order; no device-to-host inspection.
        nr = len(runs)
        starts = np.cumsum([0] + [r[2] for r in runs[:-1]], dtype=np.int64)
        run_order = sorted(range(nr), key=lambda j: runs[j][0])
        ranges = []
        for offset, j in enumerate(run_order):
            ordinal = runs[j][0]
            if not ranges or ranges[-1][0] != ordinal:
                ranges.append([ordinal, offset, 0])
            ranges[-1][2] += 1
        meta_offset = 4 * n_tok + len(order_vals) + nr
        packed = np.empty(meta_offset + 5 * nr, dtype=np.int64)
        packed[:n_tok] = blk_tok_at
        packed[n_tok:2 * n_tok] = batch
        packed[2 * n_tok:3 * n_tok] = group
        packed[3 * n_tok:4 * n_tok] = token
        packed[4 * n_tok:4 * n_tok + len(order_vals)] = order_vals
        packed[4 * n_tok + len(order_vals):meta_offset] = [r[2] for r in runs]
        meta = packed[meta_offset:].reshape(5, nr)
        meta[0] = [runs[j][1] for j in run_order]
        meta[1] = run_order
        meta[2] = starts[run_order]
        meta[3] = [runs[j][2] for j in run_order]
        meta[4] = np.asarray([self.seg_forget if runs[j][3] else 1.0 for j in run_order],
                             dtype=np.float32).view(np.int32)
        packed_t = _upload(packed, dev)
        at = packed_t[:n_tok]
        bi, gi, ti = packed_t[n_tok:4 * n_tok].view(3, n_tok).unbind(0)
        order_t = packed_t[4 * n_tok:4 * n_tok + len(order_vals)]
        run_lengths = packed_t[4 * n_tok + len(order_vals):meta_offset]
        metadata = packed_t[meta_offset:].view(5, nr)
        k_sel = k_raw.detach()[bi, gi, ti]
        v_sel = v.detach()[bi, gi, ti]
        p_sel = positions[bi, ti].to(torch.int64)

        # Both planners append token rows in increasing staging order. With
        # no pads, `at` is the identity: gathered payloads ARE the staging block.
        # All staging fields are read-only until copied to separate cache fields.
        if n_tok == total:
            block_k, block_v, block_p = k_sel, v_sel, p_sel
            block_w = self.level_w.new_ones(total)
            block_pad = self.pad_mask.new_zeros(total)
        else:
            block_k = self.level_k.new_zeros(total, self.k_dim).index_copy_(0, at, k_sel)
            block_v = self.level_v.new_zeros(total, self.v_dim).index_copy_(0, at, v_sel)
            block_p = self.level_p_lo.new_zeros(total).index_copy_(0, at, p_sel)
            block_w = self.level_w.new_zeros(total).index_fill_(0, at, 1.0)
            block_pad = self.pad_mask.new_ones(total).index_fill_(0, at, False)
        block = (
            block_k, block_v, block_w,
            *self._empty_stats(block_k, block_v),
            block_p, block_p, block_p,
            order_t,
            block_pad,
        )
        if tail is None:
            self._semantic_append_entries_batched(lanes, lane_counts, block)
        else:
            tail_lanes, tail_counts, tail_rows = tail
            self._semantic_append_entries_batched(
                list(lanes) + list(tail_lanes), list(lane_counts) + list(tail_counts),
                tuple(None if head is None else self._semantic_stack_rows(head, part)
                      for head, part in zip(block, tail_rows)))

        # Keep the increasing-token sum order: atomic/tree reductions can
        # perturb centroid routing. The CUDA loop fuses that sum with the update.
        centroid_flat = self.centroid.view(-1, self.k_dim)
        n_eff_flat = self.n_eff.view(-1)
        fused = (_triton_updates() if _fused_on(dev)
                 and k_sel.dtype in (torch.float16, torch.bfloat16, torch.float32)
                 and centroid_flat.dtype == n_eff_flat.dtype == torch.float32 else None)
        if fused is None:
            run_sums = torch.segment_reduce(k_sel.float(), "sum", lengths=run_lengths, unsafe=True)
            lengths = metadata[3].float()
            forgets = metadata[4].to(torch.int32).view(torch.float32)
        for _ordinal, start, count in ranges:
            if fused is not None:
                if self.semantic_centroid_backend == "parallel":
                    fused.centroid_parallel(k_sel, centroid_flat, n_eff_flat, metadata, start, count)
                else:
                    fused.centroid(k_sel, centroid_flat, n_eff_flat, metadata, start, count)
                continue
            end = start + count
            ci, sel_t = metadata[0, start:end], metadata[1, start:end]
            n, forget = lengths[start:end], forgets[start:end]
            sums = run_sums.index_select(0, sel_t)
            pre = n_eff_flat.index_select(0, ci) * forget
            denom = pre + n
            mu = centroid_flat.index_select(0, ci)
            centroid_flat.index_copy_(
                0, ci,
                torch.where(
                    (pre > 0).unsqueeze(-1),
                    (pre.unsqueeze(-1) * mu + sums) / denom.unsqueeze(-1),
                    sums / n.unsqueeze(-1),
                ),
            )
            n_eff_flat.index_copy_(0, ci, denom)
        for b, g, _c in lanes:
            self._semantic_ward_dirty[b][g] = True

    def _semantic_join_offsets_batch(
        self,
        b: int,
        g: int,
        c: int,
        offsets: tuple[int, ...],
        k_raw: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        positions_host: list[list[int]],
        *,
        record: bool = False,
    ) -> None:
        self._semantic_commit_joins(
            [(b, g, c, offsets)], k_raw, v, positions, positions_host, record=record
        )

    def _semantic_apply_tree_clusters(
        self,
        b: int,
        g: int,
        clusters: list[_SemanticTreeCluster],
        k_raw: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        positions_host: list[list[int]],
        *,
        record: bool = False,
    ) -> None:
        local_only: list[_SemanticTreeCluster] = []
        for node in clusters:
            existing = sorted(set(node.existing))
            if not existing:
                local_only.append(node)
                continue
            keep = existing[0]
            for free in existing[1:]:
                self._semantic_ward_merge(b, g, min(keep, free), max(keep, free), record=record)
                keep = min(keep, free)
            if node.tokens:
                self._semantic_join_offsets_batch(b, g, keep, node.tokens, k_raw, v, positions, positions_host, record=record)

        for node in local_only:
            tokens = tuple(sorted(node.tokens, key=lambda i: positions_host[b][i]))
            if not tokens:
                continue
            free_idx = self._semantic_free_clusters(b, g)
            if not free_idx:
                keep, free = self._semantic_ward_pair(b, g)
                self._semantic_ward_merge(b, g, keep, free, record=record)
                target = free
            else:
                target = free_idx[0]
            first, rest = tokens[0], tokens[1:]
            self._semantic_new_cluster(
                b, g, target, positions_host[b][first], k_raw[b, g, first], v[b, g, first], positions[b, first],
                record=record,
            )
            if rest:
                self._semantic_join_offsets_batch(b, g, target, rest, k_raw, v, positions, positions_host, record=record)

    def _semantic_route_chunk_tree(
        self,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        positions_host: list[list[int]],
        *,
        record: bool = False,
    ) -> None:
        chunk_size = self.semantic_cluster_chunk_size
        for b in range(k_raw.size(0)):
            for g in range(k_raw.size(1)):
                sets = [self._semantic_tree_existing_set(b, g)]
                sets.extend(self._semantic_tree_local_chunks_batched(b, g, k_raw, chunk_size, positions_host))
                while len(sets) > 1:
                    merged: list[list[_SemanticTreeCluster]] = []
                    for i in range(0, len(sets), 2):
                        if i + 1 == len(sets):
                            merged.append(sets[i])
                        else:
                            merged.append(self._semantic_tree_merge_sets(b, g, sets[i], sets[i + 1]))
                    sets = merged
                self._semantic_apply_tree_clusters(
                    b, g, sets[0] if sets else [], k_raw, v, positions, positions_host, record=record
                )

    def _semantic_route_three_phase(
        self,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        positions_host: list[list[int]],
        *,
        record: bool,
        token_counts: list[int] | None = None,
        stream=None,
    ) -> None:
        """Attach route: frozen nearest-cluster joins, then orphan seeding.

        `token_counts` marks ragged rows (Alpha archives): offsets at or past a
        row's count are padding and are neither assigned nor written.
        """
        if stream is not None and not (self.K_max > 1 and self._semantic_attach_one_transfer
                                       and not self.semantic_legacy_route and self._semantic_hard_cap() == math.inf):
            # Only the one-transfer planner splits decisions from commits; the
            # others write on the main stream, after the side stream's inputs.
            main = torch.cuda.current_stream(k_raw.device)
            main.wait_stream(stream)
            for tensor in (k_raw, v, positions):
                tensor.record_stream(main)
            stream = None
        if self.K_max == 1:
            if token_counts is not None:
                raise ValueError("ragged attach routing requires K > 1")
            self._semantic_route_k1_batch(k_raw, v, positions, positions_host, record=record)
            return
        if token_counts is not None and not any(token_counts):
            return
        if self._semantic_attach_one_transfer and not self.semantic_legacy_route and self._semantic_hard_cap() == math.inf:
            with self._semantic_deferred_scalars():
                self._semantic_route_attach(k_raw, v, positions, positions_host, record=record,
                                            token_counts=token_counts, stream=stream)
            return

        winner, s_winner, direct = self._semantic_existing_assignments(k_raw, positions)

        # Phase 1/3a: transfer all frozen decisions once and bucket them with
        # whole-array NumPy: per lane, one stable sort groups tokens by cluster
        # (ascending offsets), padding first, then unassigned tokens. No
        # per-token Python.
        n_batch, n_groups, tokens = k_raw.shape[:3]
        n_lanes, width = n_batch * n_groups, self.K_max + 2
        assign = winner.masked_fill(~direct, -1).cpu().numpy().reshape(n_batch, n_groups, tokens)
        if token_counts is not None:
            pad = np.arange(tokens)[None, :] >= np.asarray(token_counts, dtype=np.int64)[:, None]
            assign[np.broadcast_to(pad[:, None, :], assign.shape)] = -2
        assign = assign.reshape(n_lanes, tokens)
        order = np.argsort(assign, axis=1, kind="stable")
        counts = np.bincount((np.arange(n_lanes)[:, None] * width + assign + 2).ravel(),
                             minlength=n_lanes * width).reshape(n_lanes, width)
        starts = np.cumsum(counts, axis=1) - counts
        orphans = [[None] * n_groups for _ in range(n_batch)]
        cap = self._semantic_hard_cap()
        jobs: list[tuple[int, int, int, np.ndarray]] = []
        for lane in range(n_lanes):
            b, g = divmod(lane, n_groups)
            row = order[lane]
            lane_orphans = [row[starts[lane, 1]:starts[lane, 1] + counts[lane, 1]]]
            for c in np.flatnonzero(counts[lane, 2:]).tolist():
                offsets = row[starts[lane, c + 2]:starts[lane, c + 2] + counts[lane, c + 2]]
                if cap != math.inf:
                    room = max(0, int(cap) - self._semantic_n_total[b][g][c])
                    lane_orphans.append(offsets[room:])
                    offsets = offsets[:room]
                if len(offsets):
                    jobs.append((b, g, c, offsets))
            orphans[b][g] = np.concatenate(lane_orphans)
        if not self.semantic_legacy_route:
            # Plan both phases before writing: an early orphan can join the
            # same cluster as a later direct token in this flush. Host mirrors
            # stay authoritative; device twins refresh once at the end.
            with self._semantic_deferred_scalars():
                self._semantic_route_orphans_fast(
                    orphans, k_raw, v, positions, positions_host, s_winner, jobs, record=record
                )
            return

        self._semantic_commit_joins(jobs, k_raw, v, positions, positions_host, record=record)

        # Phase 2/3b (legacy): novelty orphans bind only to orphan-created
        # clusters, and every orphan past the last free slot forces a Ward merge.
        for b in range(k_raw.size(0)):
            for g in range(k_raw.size(1)):
                batch_clusters: list[int] = []
                for i in sorted(orphans[b][g]):
                    pos = positions[b, i]
                    token_idx = positions_host[b][i]
                    if batch_clusters:
                        accept_clusters = [c for c in batch_clusters if self._semantic_can_accept(b, g, c)]
                        cand = torch.tensor(accept_clusters, device=self.alive.device, dtype=torch.long)
                    else:
                        cand = self.alive.new_empty(0, dtype=torch.long)
                    if cand.numel() > 0:
                        dist = (self.centroid[b, g, cand] - k_raw[b, g, i].float()).square().sum(dim=-1)
                        best, nearest = dist.min(dim=0)
                        j = int(torch.where(
                            best <= self.cluster_lambda_rel * self._semantic_s_h_host[b][g], nearest, -1
                        ).item())
                        if j >= 0:
                            c = accept_clusters[j]
                            self._semantic_join_or_segment(
                                b, g, c, token_idx, k_raw[b, g, i], v[b, g, i], pos, record=record
                            )
                            continue

                    free_idx = self._semantic_free_clusters(b, g)
                    if not free_idx:
                        keep, free = self._semantic_ward_pair(b, g)
                        self._semantic_ward_merge(b, g, keep, free, record=record)
                        batch_clusters = [keep if c == free else c for c in batch_clusters if c != free]
                        c = free
                    else:
                        c = free_idx[0]
                    self._semantic_new_cluster(b, g, c, token_idx, k_raw[b, g, i], v[b, g, i], pos, record=record)
                    batch_clusters.append(c)

    # Uncapped attach planning on the device with one readback; False keeps the
    # two-phase host planner (the reference the tests compare against).
    _semantic_attach_one_transfer = True

    def _semantic_route_attach(self, k_raw, v, positions, positions_host, *, record, token_counts=None,
                               stream=None) -> None:
        """Uncapped attach route with one device-to-host transfer per flush.

        Same rules as the two-phase planner (`_semantic_route_orphans_fast`),
        evaluated for every lane on the device in position order: frozen
        nearest-cluster joins, farthest-point seeds for up to `free` new
        clusters per lane, and nearest live-or-seed cluster for the other
        orphans. Only the final cluster of each token (int8) is read back; the
        host groups tokens by (lane, cluster, position) with one stable sort.
        New clusters are the assigned clusters that are not alive yet.
        """
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            assign, order, counts = self._semantic_attach_decide(k_raw, positions, positions_host, token_counts)
        if stream is not None:
            # Commits write the ladder and centers on the main stream, after
            # the attention queued there; the side stream's inputs stay alive.
            main = torch.cuda.current_stream(k_raw.device)
            main.wait_stream(stream)
            for tensor in (k_raw, v, positions):
                tensor.record_stream(main)
        self._semantic_attach_commit(assign, order, counts, k_raw, v, positions, positions_host, record=record)

    def _semantic_attach_decide(self, k_raw, positions, positions_host, token_counts):
        """Final cluster per token in position order (host int8), the order and the counts."""
        n_batch, n_groups, tokens, dim = k_raw.shape
        dev = k_raw.device
        counts = (np.full(n_batch, tokens, dtype=np.int64) if token_counts is None
                  else np.asarray(token_counts, dtype=np.int64))
        host_pos = np.asarray(positions_host, dtype=np.int64)
        # Position order per row (stable); padding sorts last.
        valid_host = np.arange(tokens)[None, :] < counts[:, None]
        order = np.argsort(np.where(valid_host, host_pos, np.iinfo(np.int64).max), axis=1, kind="stable")
        free = [[self._semantic_free_clusters(b, g) for g in range(n_groups)] for b in range(n_batch)]
        n_seed = max(len(row) for lane in free for row in lane)
        slots = np.full((n_batch, n_groups, max(n_seed, 1)), self.K_max, dtype=np.int64)
        n_free = np.zeros((n_batch, n_groups), dtype=np.int64)
        for b, lane in enumerate(free):
            for g, row in enumerate(lane):
                slots[b, g, :len(row)] = row
                n_free[b, g] = len(row)
        metadata = _upload(np.concatenate((order.reshape(-1), counts, slots.reshape(-1), n_free.reshape(-1))), dev)
        cut = np.cumsum([order.size, n_batch, slots.size])
        order_t = metadata[:cut[0]].view(n_batch, tokens)
        counts_t, slots_t = metadata[cut[0]:cut[1]], metadata[cut[1]:cut[2]].view(slots.shape)
        n_free_t = metadata[cut[2]:].view(n_batch, n_groups)

        fused = (_triton_route() if _fused_on(dev) and not n_seed and self.semantic_capacity_beta == 0.0
                 and k_raw.dtype in (torch.float16, torch.bfloat16, torch.float32) else None)
        if fused is not None:
            # Every cluster is alive (no seeds): one kernel evaluates the frozen
            # joins and the orphans' nearest live cluster for all lanes.
            self._semantic_sync_device_scalars()
            assign = fused.attach_assign(
                k_raw.detach(), positions, order_t, counts_t, self.centroid[:n_batch, :n_groups],
                self.alive[:n_batch, :n_groups], self.p_hi_c[:n_batch, :n_groups], self.s_h[:n_batch, :n_groups],
                self.seg_eta, self.seg_g0, self.cluster_lambda_rel,
            ).cpu().numpy()
        else:
            assign = self._semantic_attach_plan_torch(k_raw, positions, order_t, counts_t, slots_t, n_free_t, n_seed)
        return assign, order, counts

    def _semantic_attach_plan_torch(self, k_raw, positions, order_t, counts_t, slots_t, n_free_t, n_seed):
        """Reference/seeding form of the attach decisions (int8 [B, G, T], position order)."""
        n_batch, n_groups, tokens, dim = k_raw.shape
        dev = k_raw.device
        winner, s_winner, direct = self._semantic_existing_assignments(k_raw, positions)
        perm = order_t[:, None, :].expand(-1, n_groups, -1)
        valid = (torch.arange(tokens, device=dev)[None, :] < counts_t[:, None])[:, None, :]
        direct = direct.gather(2, perm) & valid
        orphan = valid & ~direct
        assign = winner.gather(2, perm).masked_fill(~direct, -1)
        x = k_raw.detach().gather(2, perm[..., None].expand(-1, -1, -1, dim)).float()
        # One spare row per lane takes the writes of unused seed slots.
        mu = torch.cat((self.centroid[:n_batch, :n_groups].float(), x.new_zeros(n_batch, n_groups, 1, dim)), 2)
        candidates = torch.cat((self.alive[:n_batch, :n_groups], self.alive.new_zeros(n_batch, n_groups, 1)), 2)
        if n_seed:
            def distance_to(t):
                return (x - x.gather(2, t[..., None, None].expand(-1, -1, 1, dim))).square().sum(-1)

            nxt = s_winner.gather(2, perm).masked_fill(~orphan, -float("inf")).argmax(-1)
            picks = [nxt]
            dist = distance_to(nxt).masked_fill_(~orphan, -float("inf"))
            for _ in range(n_seed - 1):
                # Mask previous picks even for identical keys, so each
                # newly opened cluster reserves a distinct orphan token.
                dist.scatter_(2, nxt[..., None], -float("inf"))
                nxt = dist.argmax(-1)
                picks.append(nxt)
                dist = torch.minimum(dist, distance_to(nxt))
            picked = torch.stack(picks, -1)
            used = torch.arange(n_seed, device=dev) < torch.minimum(n_free_t, orphan.sum(-1))[..., None]
            target = torch.where(used, slots_t[..., :n_seed], self.K_max)
            mu.scatter_(2, target[..., None].expand(-1, -1, -1, dim),
                        x.gather(2, picked[..., None].expand(-1, -1, -1, dim)))
            candidates.scatter_(2, target, True)
        # Exact differences, as in the two-phase planner.
        distances = torch.cdist(x.flatten(0, 1), mu[:, :, :self.K_max].flatten(0, 1),
                                compute_mode="donot_use_mm_for_euclid_dist").view(n_batch, n_groups, tokens, -1)
        distances.masked_fill_(~candidates[:, :, None, :self.K_max], float("inf"))
        assign = torch.where(orphan, distances.argmin(-1), assign)
        if n_seed:
            # A seed opens its own cluster; unused picks land in a spare column.
            assign = torch.cat((assign, assign.new_zeros(n_batch, n_groups, 1)), 2)
            assign.scatter_(2, torch.where(used, picked, tokens), slots_t[..., :n_seed])
            assign = assign[..., :tokens]
        return assign.to(torch.int8).cpu().numpy()

    def _semantic_attach_commit(self, assign, order, counts, k_raw, v, positions, positions_host, *, record):
        """Commit attach decisions: one stable sort gives the (lane, cluster, position) plan."""
        n_batch, n_groups = k_raw.shape[:2]
        lane_ids = np.repeat(np.arange(n_batch * n_groups), np.repeat(counts, n_groups))
        offsets = np.concatenate([np.tile(order[b, :counts[b]], n_groups) for b in range(n_batch)])
        clusters = np.concatenate([assign[b, :, :counts[b]].reshape(-1) for b in range(n_batch)]).astype(np.int64)
        if not len(clusters):
            return
        sort = np.argsort(lane_ids * (self.K_max + 1) + clusters, kind="stable")
        lane_ids, clusters, offsets = lane_ids[sort], clusters[sort], offsets[sort]
        bounds = np.flatnonzero((lane_ids[1:] != lane_ids[:-1]) | (clusters[1:] != clusters[:-1])) + 1
        new_jobs, jobs = [], []
        for lo, hi in zip([0] + bounds.tolist(), bounds.tolist() + [len(clusters)]):
            b, g = divmod(int(lane_ids[lo]), n_groups)
            c = int(clusters[lo])
            if not self._semantic_alive[b][g][c]:
                new_jobs.append((b, g, c, int(offsets[lo])))
                lo += 1
            if hi > lo:
                jobs.append((b, g, c, tuple(offsets[lo:hi].tolist())))
        if new_jobs:
            self._semantic_new_clusters(new_jobs, k_raw, v, positions, positions_host, record=record)
        # A new cluster starts at its earliest token, so only Alpha's delayed
        # evictions into existing clusters can precede a cluster's p_hi.
        commit = self._alpha_commit_joins if self.alpha_exact_tokens else self._semantic_commit_joins
        commit(jobs, k_raw, v, positions, positions_host, record=record)

    def _semantic_route_orphans_fast(
        self,
        orphans: list[list[list[int] | np.ndarray]],
        k_raw: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        positions_host: list[list[int]],
        s_winner: torch.Tensor,
        direct_jobs: list[tuple[int, int, int, tuple[int, ...] | np.ndarray]],
        *,
        record: bool,
    ) -> None:
        """Plan orphan assignments, then commit each cluster in time order.

        Existing centroids stay frozen throughout this flush. Farthest-point
        seeds are temporary routing prototypes, not early ladder writes. At
        most n_free clusters are opened; other orphans join their nearest
        available cluster without per-token Ward merges.

        Host work is whole-array NumPy: orphan position sorts, the assignment
        of every non-seed orphan, and one lexsort that orders each cluster's
        tokens by position (direct tokens before orphans on ties, as the
        per-bucket stable sort did). New clusters open in one batched call.
        With Alpha, the commit inserts delayed exact evictions by position.
        """
        cap = self._semantic_hard_cap()
        n_groups = k_raw.size(1)
        host_pos = np.asarray(positions_host, dtype=np.int64)
        direct = {(b, g, c): np.asarray(offsets, dtype=np.int64) for b, g, c, offsets in direct_jobs}
        lanes = []
        for b in range(k_raw.size(0)):
            for g in range(n_groups):
                orph = np.asarray(orphans[b][g], dtype=np.int64)
                if len(orph):
                    orph = orph[np.argsort(host_pos[b, orph], kind="stable")]
                    free = self._semantic_free_clusters(b, g)[:len(orph)]
                    lanes.append((b, g, orph, free))

        # Orphan bucket parts: (lane, cluster, offset, rank). A seed ranks
        # first in its new cluster; other orphans rank by their orph order.
        parts: list[tuple[np.ndarray, ...]] = []
        new_clusters: set[tuple[int, int]] = set()
        if lanes:
            # Only orphan-bearing lanes participate. Padding is bounded by the
            # flush size; no [lanes, tokens, clusters, head_dim] distance tensor.
            width = max(len(orph) for _, _, orph, _ in lanes)
            n_seed = max(len(free) for _, _, _, free in lanes)
            offsets = np.full((len(lanes), width), -1, dtype=np.int64)
            candidates = np.zeros((len(lanes), self.K_max), dtype=np.bool_)
            seed_slots = []
            for row, (b, g, orph, free) in enumerate(lanes):
                offsets[row, :len(orph)] = orph
                candidates[row, self._semantic_live_clusters(b, g) + free] = True
                seed_slots.extend((row, c, j) for j, c in enumerate(free))
            dev = k_raw.device
            bi, gi = _upload(np.asarray([(b, g) for b, g, _, _ in lanes], dtype=np.int64), dev).unbind(-1)
            idx = _upload(offsets, dev)
            valid = idx >= 0
            idx = idx.clamp_min(0)
            x = k_raw[bi[:, None], gi[:, None], idx].float()
            mu = self.centroid[bi, gi]  # advanced indexing owns this temporary
            rows = torch.arange(len(lanes), device=dev)
            picked = torch.empty((len(lanes), 0), device=dev, dtype=torch.long)
            if n_seed:
                novelty = s_winner[bi[:, None], gi[:, None], idx].masked_fill(~valid, -float("inf"))
                nxt = novelty.argmax(-1)
                picks = [nxt]
                dist = (x - x[rows, nxt].unsqueeze(1)).square().sum(-1)
                dist.masked_fill_(~valid, -float("inf"))
                for _ in range(n_seed - 1):
                    # Mask previous picks even for identical keys, so each
                    # newly opened cluster reserves a distinct orphan token.
                    dist.scatter_(1, nxt[:, None], -float("inf"))
                    nxt = dist.argmax(-1)
                    picks.append(nxt)
                    dist = torch.minimum(dist, (x - x[rows, nxt].unsqueeze(1)).square().sum(-1))
                picked = torch.stack(picks, dim=1)
                sr, sc, sj = _upload(np.asarray(seed_slots, dtype=np.int64).reshape(-1, 3), dev).unbind(-1)
                mu[sr, sc] = x[sr, picked[sr, sj]]

            # Exact differences, not the GEMM expansion: the bits must not
            # depend on how many lanes or orphans share this call.
            distances = torch.cdist(x, mu, compute_mode="donot_use_mm_for_euclid_dist")
            distances.masked_fill_(~_upload(candidates, dev)[:, None, :], float("inf"))
            choices = distances.argmin(-1) if cap == math.inf else distances.argsort(dim=-1, stable=True).flatten(1)
            # One device-to-host transfer for ALL seeds and assignments. No
            # device scalar extraction or host round trip inside a lane loop.
            decisions = torch.cat((picked, choices), dim=1).cpu().numpy()
            for row, (b, g, orph, free) in enumerate(lanes):
                lane = b * n_groups + g
                result = decisions[row]
                seeds = result[:len(free)]
                seeded = np.zeros(len(orph), dtype=np.bool_)
                seeded[seeds] = True
                for c in free:
                    new_clusters.add((lane, c))
                    direct.pop((b, g, c), None)
                parts.append((np.full(len(free), lane), np.asarray(free, dtype=np.int64), orph[seeds],
                              np.full(len(free), -1)))
                rest = np.flatnonzero(~seeded)
                if cap == math.inf:
                    chosen = result[n_seed + rest]
                else:
                    sizes = {key[2]: len(value) for key, value in direct.items() if key[:2] == (b, g)}
                    room = {
                        c: max(0, int(cap) - self._semantic_n_total[b][g][c] - (1 if c in free else sizes.get(c, 0)))
                        for c in self._semantic_live_clusters(b, g) + free
                    }
                    chosen = np.empty(len(rest), dtype=np.int64)
                    for n, t in enumerate(rest.tolist()):
                        start = n_seed + t * self.K_max
                        order = result[start:start + self.K_max].tolist()
                        # ponytail: when all clusters are full, exceed the cap
                        # at the nearest one to preserve tokens; strict rejection
                        # would need a separate caller-visible overflow policy.
                        c = next((c for c in order if room.get(c, 0) > 0), order[0])
                        room[c] -= 1
                        chosen[n] = c
                parts.append((np.full(len(rest), lane), chosen.astype(np.int64), orph[rest], rest))

        # Direct tokens rank before orphans (rank < 0 - n); then one lexsort
        # orders every bucket by (lane, cluster, position, rank).
        for (b, g, c), offsets in direct.items():
            parts.append((np.full(len(offsets), b * n_groups + g), np.full(len(offsets), c), offsets,
                          np.arange(len(offsets)) - (len(offsets) + 1) - host_pos.shape[1]))
        new_jobs, jobs = [], []
        if parts:
            lane_a, c_a, off_a, rank_a = (np.concatenate(x).astype(np.int64) for x in zip(*parts))
            pos_a = host_pos[lane_a // n_groups, off_a]
            order = np.lexsort((rank_a, pos_a, c_a, lane_a))
            lane_a, c_a, off_a = lane_a[order], c_a[order], off_a[order]
            cut = np.flatnonzero((lane_a[1:] != lane_a[:-1]) | (c_a[1:] != c_a[:-1])) + 1
            for lo, hi in zip([0] + cut.tolist(), cut.tolist() + [len(off_a)]):
                lane, c = int(lane_a[lo]), int(c_a[lo])
                b, g = divmod(lane, n_groups)
                if (lane, c) in new_clusters:
                    new_jobs.append((b, g, c, int(off_a[lo])))
                    lo += 1
                if hi > lo:
                    jobs.append((b, g, c, tuple(off_a[lo:hi].tolist())))
        if new_jobs:
            self._semantic_new_clusters(new_jobs, k_raw, v, positions, positions_host, record=record)
        # A new cluster starts at its earliest token, so only Alpha's delayed
        # evictions into existing clusters can precede a cluster's p_hi.
        commit = self._alpha_commit_joins if self.alpha_exact_tokens else self._semantic_commit_joins
        commit(jobs, k_raw, v, positions, positions_host, record=record)

    @staticmethod
    def _semantic_build_replay_plan(host_log, cursors, positions_host, n_groups, n_tokens):
        """Parse one flush on the CPU; offsets and rounds contain no GPU state."""
        offset_by_token = []
        for b, row in enumerate(positions_host):
            offsets = {int(token): i for i, token in enumerate(row)}
            if len(offsets) != len(row):
                raise RuntimeError(f"semantic LogKV replay positions contain duplicate token ids for batch={b}: {row}")
            offset_by_token.append(offsets)
        lane_items = []
        end_cursors = []
        try:
            for b in range(len(cursors)):
                ends = []
                for g in range(n_groups):
                    consumed = 0
                    rows = host_log[b][g]
                    cursor = cursors[b][g]
                    items = []
                    while consumed < n_tokens and cursor < len(rows):
                        op, cluster, arg, token = rows[cursor]
                        cursor += 1
                        if op == LOG_KV_OP_PAD_INSERT:
                            items.append(("pad", cluster, token, None, None))
                        elif op == LOG_KV_OP_WARD_MERGE:
                            items.append(("ward", cluster, arg, None, None))
                        elif op == LOG_KV_OP_NEW_CLUSTER:
                            items.append(("new", cluster, token, offset_by_token[b][int(token)], None))
                            consumed += 1
                        elif op in (LOG_KV_OP_JOIN, LOG_KV_OP_NEW_SEGMENT):
                            offsets = array("q", [offset_by_token[b][int(token)]])
                            while cursor < len(rows) and consumed + len(offsets) < n_tokens:
                                next_op, next_c, next_seg, token = rows[cursor]
                                if next_op != LOG_KV_OP_JOIN or next_c != cluster or next_seg != arg:
                                    break
                                offsets.append(offset_by_token[b][int(token)])
                                cursor += 1
                            items.append(("join", cluster, arg, offsets, op == LOG_KV_OP_NEW_SEGMENT))
                            consumed += len(offsets)
                        else:
                            raise ValueError(f"unknown semantic LogKV op {op}")
                    if consumed != n_tokens:
                        raise RuntimeError(
                            f"semantic LogKV replay consumed {consumed}/{n_tokens} tokens for batch={b}, group={g}"
                        )
                    ends.append(cursor)
                    lane_items.append((b, g, items))
                end_cursors.append(tuple(ends))
        except KeyError as exc:
            raise RuntimeError(f"semantic LogKV replay token {exc.args[0]} not in current flush positions") from exc

        # Preserve the existing schedule: joins for distinct clusters share a
        # round; a structural operation is applied after that round's joins.
        ptr = [0] * len(lane_items)
        rounds = []
        while any(ptr[l] < len(lane_items[l][2]) for l in range(len(lane_items))):
            batch, specials = [], []
            for l, (b, g, items) in enumerate(lane_items):
                used = set()
                while ptr[l] < len(items):
                    kind, cluster, arg, offsets, new_segment = items[ptr[l]]
                    if kind != "join" or cluster in used:
                        break
                    used.add(cluster)
                    batch.append((b, g, cluster, arg, new_segment, offsets))
                    ptr[l] += 1
                if ptr[l] < len(items) and items[ptr[l]][0] != "join":
                    specials.append((b, g, items[ptr[l]]))
                    ptr[l] += 1
            rounds.append((tuple(batch), tuple(specials)))
        return tuple(end_cursors), tuple(rounds)

    def _semantic_buffer_storages(self) -> set[int]:
        """Storage addresses of this cache's buffers (constant while recording)."""
        return {buffer.untyped_storage().data_ptr() for buffer in self.buffers()}

    @contextlib.contextmanager
    def _semantic_update_context(self, updates, *, replay=False):
        previous = self._active_updates, self._replaying_updates
        self._active_updates, self._replaying_updates = updates, replay
        try:
            yield
        finally:
            self._active_updates, self._replaying_updates = previous

    _UPDATE_DEVICE_FIELDS = ("centroid", "n_eff", "n_total", "p_hi_c", "current_segment",
                             "level0_phase", "alive", "ward_cost")
    _UPDATE_HOST_FIELDS = ("_semantic_alive", "_semantic_p_hi_c", "_semantic_current_segment",
                           "_semantic_level0_phase", "_semantic_n_total", "_semantic_ward_dirty")

    def route_and_flush_batch(self, *args, **kwargs) -> None:
        self._mid_decode_state = None
        replay = any(kwargs.get(name) is not None for name in ("replay_op_log", "replay_op_log_host"))
        with logkv_timed("replay" if replay else "route"):
            updates = self._active_updates
            if updates is None:
                return self._route_and_flush_batch(*args, **kwargs)
            k, v = args[:2]
            host = kwargs.get("positions_host")
            if host is None:
                raise RuntimeError("update replay requires CPU flush positions; no device readback is allowed")
            if len(host) != k.size(0) or any(len(row) != k.size(2) for row in host):
                raise RuntimeError("update replay CPU positions do not match the flush shape")
            key = tuple(tuple(row) for row in host)
            signature = (k.shape, v.shape, k.dtype, v.dtype, k.device,
                         self.semantic_centroid_backend, self.second_order,
                         self.semantic_unified_route, self.semantic_merge_passes,
                         self.K_max, self.B, self.L_alloc, self.recent_size, self.semantic_flush_granularity,
                         self.alpha_exact_tokens, self.alpha_span_max_tokens,
                         self.beta_novelty, self.beta_adaptive_merge,
                         tuple(map(tuple, kwargs.get("span_ends") or ())))
            if self._replaying_updates:
                if not replay or key not in updates.flushes:
                    raise RuntimeError("missing semantic update record; rerouting is not allowed")
                expected, actions, state, host_state, starts, ends, ready = updates.flushes[key]
                cursor = getattr(self, "_op_replay_cursor_host", None)
                if cursor is None:
                    cursor = [[0] * self.n_groups for _ in range(self.batch_size)]
                if cursor != starts:
                    raise RuntimeError("semantic update replay flushes are out of order")
                if signature != expected:
                    raise RuntimeError("semantic update record does not match this flush/configuration")
                if ready is not None:
                    # Device-side wait only. No CPU synchronization or offload.
                    stream = torch.cuda.current_stream(k.device)
                    stream.wait_event(ready)
                with self._semantic_deferred_scalars():
                    for kind, metadata, indices in actions:
                        if kind == "clear":
                            self._semantic_clear_cluster(*metadata)
                        elif kind == "clear_batch":
                            self._semantic_clear_clusters(metadata)
                        else:
                            block = tuple(updates.tensors[i] if i is not None else None for i in indices)
                            if ready is not None:
                                for tensor in block:
                                    if tensor is not None:
                                        tensor.record_stream(stream)
                            if kind == "append_beta":
                                lanes, counts, cut_indices = metadata
                                cuts = {ell: updates.tensors[index] for ell, index in cut_indices}
                                if ready is not None:
                                    for tensor in cuts.values():
                                        tensor.record_stream(stream)
                                self._semantic_append_entries_batched(lanes, counts, block, replay_cuts=cuts)
                            else:
                                self._semantic_append_entries_batched(*metadata, block)
                for name, index in zip(self._UPDATE_DEVICE_FIELDS, state):
                    tensor = updates.tensors[index]
                    if ready is not None:
                        tensor.record_stream(stream)
                    getattr(self, name).copy_(tensor)
                for name, value in zip(self._UPDATE_HOST_FIELDS, host_state):
                    setattr(self, name, _copy_host(value))
                self._op_replay_cursor_host = _copy_host(ends)
                return
            if replay or key in updates.flushes:
                raise RuntimeError("duplicate or incorrectly bound semantic update recording")
            if kwargs.get("record_op_log") and self.op_log is None:
                self.begin_op_log()
            if not kwargs.get("record_op_log"):
                raise RuntimeError("semantic update recording requires record_op_log=True")
            starts = _copy_host(self._op_log_len_host)
            self._update_actions = []
            try:
                self._route_and_flush_batch(*args, **kwargs)
                state = tuple(updates.save(getattr(self, name)) for name in self._UPDATE_DEVICE_FIELDS)
                host_state = tuple(_copy_host(getattr(self, name)) for name in self._UPDATE_HOST_FIELDS)
                ready = None
                if k.is_cuda:
                    ready = torch.cuda.Event()
                    ready.record(torch.cuda.current_stream(k.device))
                updates.flushes[key] = (signature, tuple(self._update_actions), state, host_state, starts,
                                       _copy_host(self._op_log_len_host), ready)
            finally:
                self._update_actions = None

    def _route_and_flush_batch(
        self,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        *,
        positions_host: list[list[int]] | None = None,
        record_op_log: bool = False,
        replay_op_log: torch.Tensor | None = None,
        replay_op_log_len: torch.Tensor | None = None,
        replay_op_log_host: list[list[list[tuple[int, int, int, int]]]] | None = None,
        replay_plans: _SemanticReplayPlans | None = None,
        span_ends: list[list[bool]] | None = None,
    ) -> None:
        if not self.semantic_clusters:
            raise RuntimeError("route_and_flush_batch is only valid for semantic LogKV")
        if replay_plans is not None and replay_op_log is None:
            raise RuntimeError("semantic replay plans require the matching op-log; rerouting is not allowed")
        if record_op_log and self.op_log is None:
            self.begin_op_log()
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(k_raw.size(0), -1)
        if positions_host is None:
            if replay_plans is not None:
                raise RuntimeError("semantic replay plans require CPU positions; device readback is not allowed")
            positions_host = [[int(x) for x in row] for row in positions.detach().cpu().tolist()]
        if self.alpha_exact_tokens:
            if replay_op_log is not None:
                raise RuntimeError("AlphaLogKV replay requires semantic_replay_updates; reranking is not allowed")
            with self._semantic_deferred_scalars():
                self._alpha_route_flush(k_raw, v, positions, positions_host, span_ends, record=record_op_log)
            return
        if replay_op_log is not None:
            if replay_op_log_len is None:
                raise ValueError("replay_op_log_len is required with replay_op_log")
            if replay_plans is not None and replay_op_log_host is not replay_plans.host_log:
                raise RuntimeError("semantic replay plan does not match the CPU op-log")
            if replay_op_log_host is None:
                lengths = replay_op_log_len.cpu().tolist()
                replay_op_log_host = [
                    [replay_op_log[b, g, :lengths[b][g]].cpu().tolist() for g in range(k_raw.size(1))]
                    for b in range(k_raw.size(0))
                ]
            if getattr(self, "_op_replay_cursor_host", None) is None:
                self._op_replay_cursor_host = [[0] * self.n_groups for _ in range(self.batch_size)]
            B, G, T = k_raw.shape[:3]
            host_positions = positions_host[:B]
            current_positions = np.array(host_positions, dtype=np.int64, copy=True)
            if current_positions.shape != (B, T):
                raise RuntimeError("semantic replay positions do not match the current flush shape")
            if len(replay_op_log_host) != B or any(len(row) != G for row in replay_op_log_host):
                raise RuntimeError("semantic replay log does not match the current batch/groups")
            cursors = tuple(tuple(row[:G]) for row in self._op_replay_cursor_host[:B])
            plan = replay_plans.plans.get(cursors) if replay_plans is not None else None
            if plan is None:
                ends, rounds = self._semantic_build_replay_plan(
                    replay_op_log_host, cursors, host_positions, G, T
                )
                current_positions.setflags(write=False)
                plan = (current_positions, ends, rounds)
                if replay_plans is not None:
                    replay_plans.plans[cursors] = plan
            elif not np.array_equal(plan[0], current_positions):
                raise RuntimeError("semantic replay plan does not match the current flush positions")
            _, ends, rounds = plan
            for b, row in enumerate(ends):
                self._op_replay_cursor_host[b][:G] = row
            with self._semantic_deferred_scalars():
                for batch, specials in rounds:
                    self._semantic_commit_runs(batch, k_raw, v, positions, host_positions)
                    for b, g, (kind, cluster, arg, offset, _new) in specials:
                        if kind == "pad":
                            self._semantic_insert_pad(b, g, cluster, arg, record=False)
                        elif kind == "ward":
                            self._semantic_ward_merge(b, g, cluster, arg, record=False)
                        else:
                            self._semantic_new_cluster(
                                b, g, cluster, int(arg), k_raw[b, g, offset], v[b, g, offset],
                                positions[b, offset], record=False
                            )
            return

        if self.semantic_unified_route:
            with self._semantic_deferred_scalars():
                self._semantic_route_unified(k_raw, v, positions, positions_host, record=record_op_log)
            return

        if (
            self.K_max > 1
            and self.semantic_cluster_chunk_size > 0
        ):
            self._semantic_route_chunk_tree(k_raw, v, positions, positions_host, record=record_op_log)
            return

        self._semantic_route_three_phase(k_raw, v, positions, positions_host, record=record_op_log,
                                         stream=self._route_decision_stream(k_raw, v))


    # ------------------------------------------------------------------
    # Compact: compress tokens -> 1 entry via mean pooling (2:1 by default)
    # ------------------------------------------------------------------

    @staticmethod
    def _compact_tokens(
        k: torch.Tensor,  # (B, G, n, k_dim) full post-RoPE keys
        v: torch.Tensor,  # (B, G, n, v_dim)
        with_stats: bool = False,
        imp: torch.Tensor | None = None,  # (B, G, n) optional per-token importance weights
        imp_lambda: float = 1.0,
    ) -> tuple[torch.Tensor, ...]:
        """Compress n tokens into a single compact entry via mean pooling.

        Mean-pooling the full key merges content and position in one step:
        the position sub-channel becomes the expected rotation over the span
        (see module docstring). No renormalization — the per-frequency norm
        shrinkage encodes the span's positional uncertainty.
        Returns k_entry (B,G,1,k_dim), v_entry (B,G,1,v_dim), w_entry (B,G,1).
        With ``with_stats=True`` also returns rank-1 approximations to the
        within-slot key covariance and value-key cross covariance.

        ``imp``, when given, replaces the uniform 1/n pooling weight with a
        per-token importance-weighted mean (need not be pre-normalized); the
        raw sum of ``imp`` is returned as one extra trailing fp32 tensor (the
        slot's cumulative importance mass — independent of ``w_entry``, the
        token count, which is always uniform regardless of ``imp``). Omitting
        ``imp`` reproduces the exact prior uniform-mean behavior and return
        shape. ``imp_lambda`` (only consulted when ``imp`` is given) linearly
        blends the importance-normalized weights toward the uniform ``1/n``
        weights — ``1.0`` (default) is pure importance pooling (unchanged
        expression, bit-identical to the pre-``imp_lambda`` code), ``0.0`` is
        numerically equivalent to uniform pooling (same shares as ``imp=None``,
        via a different, imp-carrying code path so not bit-identical).
        """
        n = k.size(2)
        has_imp = imp is not None
        if has_imp:
            imp = imp.float()
            imp_entry = imp.sum(dim=2, keepdim=True)  # (B, G, 1), fp32
            if imp_lambda >= 1.0:
                p = (imp / imp_entry.clamp_min(_RANK1_EPS)).unsqueeze(-1)  # (B, G, n, 1)
            else:
                p_imp = imp / imp_entry.clamp_min(_RANK1_EPS)
                p = (imp_lambda * p_imp + (1.0 - imp_lambda) * (1.0 / n)).unsqueeze(-1)
            k_entry = (p * k.float()).sum(dim=2, keepdim=True).to(k.dtype)
            v_entry = (p * v.float()).sum(dim=2, keepdim=True).to(v.dtype)
        else:
            k_entry = k.mean(dim=2, keepdim=True)
            v_entry = v.mean(dim=2, keepdim=True)
        w_entry = torch.full(
            (k.size(0), k.size(1), 1),
            float(n),
            device=k.device,
            dtype=k.dtype,
        )
        if not with_stats:
            return (k_entry, v_entry, w_entry) if not has_imp else (k_entry, v_entry, w_entry, imp_entry)

        if has_imp:
            sqrt_p = p.sqrt()  # (B, G, n, 1), fp32
            k_centered = (k - k_entry).float() * sqrt_p
            v_centered = (v - v_entry).float() * sqrt_p
        else:
            inv_sqrt_n = float(n) ** -0.5
            k_centered = (k - k_entry).float() * inv_sqrt_n
            v_centered = (v - v_entry).float() * inv_sqrt_n
        sigma_u, sigma2 = _rank1_psd_from_factors(k_centered)
        gamma_b, gamma_a, gamma = _rank1_cross_from_factors(v_centered, k_centered)
        stats_out = (
            k_entry,
            v_entry,
            w_entry,
            sigma_u.unsqueeze(2).to(k.dtype),
            sigma2.unsqueeze(2).to(k.dtype),
            gamma_a.unsqueeze(2).to(k.dtype),
            gamma_b.unsqueeze(2).to(k.dtype),
            gamma.unsqueeze(2).to(k.dtype),
        )
        return stats_out if not has_imp else stats_out + (imp_entry,)

    # ------------------------------------------------------------------
    # Compact operation: merge two B-slot blocks -> one B-slot block (adjacent pairs)
    # ------------------------------------------------------------------

    @staticmethod
    def compact(
        k1: torch.Tensor, v1: torch.Tensor, w1: torch.Tensor,
        k2: torch.Tensor, v2: torch.Tensor, w2: torch.Tensor,
        sigma_u1: torch.Tensor | None = None,
        sigma2_1: torch.Tensor | None = None,
        gamma_a1: torch.Tensor | None = None,
        gamma_b1: torch.Tensor | None = None,
        gamma1: torch.Tensor | None = None,
        sigma_u2: torch.Tensor | None = None,
        sigma2_2: torch.Tensor | None = None,
        gamma_a2: torch.Tensor | None = None,
        gamma_b2: torch.Tensor | None = None,
        gamma2: torch.Tensor | None = None,
        imp1: torch.Tensor | None = None,
        imp2: torch.Tensor | None = None,
        imp_lambda: float = 1.0,
        p_lo1: torch.Tensor | None = None,
        p_hi1: torch.Tensor | None = None,
        sum_wp1: torch.Tensor | None = None,
        p_lo2: torch.Tensor | None = None,
        p_hi2: torch.Tensor | None = None,
        sum_wp2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Merge two B-slot blocks into one B-slot block.

        Concatenates the two blocks in time order (k1=older, k2=newer) and pairs
        ADJACENT slots: (slot 2i, slot 2i+1) -> slot i. Every merged slot covers
        a contiguous span, and the weighted mean keeps each slot's key equal to
        the true weighted mean over all tokens it covers (mean-merge is
        associative), so no error accumulates across levels.

        If the rank-1 stats are provided for both input blocks, they are merged
        with Chan's parallel covariance formula and truncated back to rank 1 via
        the iterative small-matrix routines above (near-optimal, not exact — see
        their docstrings). Without stats, this preserves the old three-tensor
        return for tests and diagnostic callers.

        ``imp1``/``imp2`` are optional per-slot importance-mass tensors (same
        shape as ``w1``/``w2``), independent of the token-count weights: when
        given, they (not ``w1``/``w2``) drive the pooling ``alpha`` and the
        Chan-merge fractions, and their sum is returned as one extra trailing
        tensor. ``w_total`` (token count, used only for the ``log(w)`` mass
        bias) is always computed from ``w1``/``w2`` regardless. Omitting them
        reproduces the exact prior count-weighted behavior and return shape.
        ``imp_lambda`` (only consulted when ``imp1``/``imp2`` are given) linearly
        blends the importance share toward the count share ``wa/w_total`` —
        ``1.0`` (default) is pure importance (unchanged expression, bit-identical
        to the pre-``imp_lambda`` code), ``0.0`` is numerically equivalent to the
        count-weighted path.

        ``p_lo*``/``p_hi*``/``sum_wp*`` are optional per-slot semantic-cluster
        anchor fields (see ``log_kv_position.merge_anchors``); when given for
        both inputs, the merged ``(p_lo, p_hi, sum_wp)`` is appended as a
        trailing 3-tuple, independent of the ``imp``/rank-1-stats branches.
        Omitting them reproduces the exact prior return shape.
        """
        k_cat = torch.cat([k1, k2], dim=-2)  # (B, G, 2B, D)
        v_cat = torch.cat([v1, v2], dim=-2)
        w_cat = torch.cat([w1, w2], dim=-1)  # (B, G, 2B)

        ka = k_cat[..., 0::2, :]  # even slots (older half of each adjacent pair)
        kb = k_cat[..., 1::2, :]  # odd slots  (newer half of each adjacent pair)
        va = v_cat[..., 0::2, :]
        vb = v_cat[..., 1::2, :]
        wa = w_cat[..., 0::2]
        wb = w_cat[..., 1::2]

        w_total = wa + wb  # (B, G, B) -- token count, always count-based

        has_imp = imp1 is not None
        if has_imp:
            imp_cat = torch.cat([imp1, imp2], dim=-1)
            impa, impb = imp_cat[..., 0::2], imp_cat[..., 1::2]
            imp_total = impa + impb
            if imp_lambda >= 1.0:
                alpha = (impa / imp_total.clamp(min=1e-8)).unsqueeze(-1)
            else:
                alpha_imp = impa / imp_total.clamp(min=1e-8)
                alpha_w = wa / w_total.clamp(min=1e-8)
                alpha = (imp_lambda * alpha_imp + (1.0 - imp_lambda) * alpha_w).unsqueeze(-1)
        else:
            alpha = (wa / w_total.clamp(min=1e-8)).unsqueeze(-1)  # (B, G, B, 1)

        k_out = alpha * ka + (1 - alpha) * kb
        v_out = alpha * va + (1 - alpha) * vb

        has_anchors = p_lo1 is not None
        if has_anchors:
            if any(x is None for x in (p_hi1, sum_wp1, p_lo2, p_hi2, sum_wp2)):
                raise ValueError("compact() requires either all anchor fields or none")
            anchors_out = merge_anchors(p_lo1, p_hi1, sum_wp1, p_lo2, p_hi2, sum_wp2, w1, w2)

        def _with_anchors(out: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            return out + anchors_out if has_anchors else out

        if sigma_u1 is None:
            return _with_anchors((k_out, v_out, w_total) if not has_imp else (k_out, v_out, w_total, imp_total))

        if any(x is None for x in (
            sigma2_1, gamma_a1, gamma_b1, gamma1,
            sigma_u2, sigma2_2, gamma_a2, gamma_b2, gamma2,
        )):
            raise ValueError("compact() requires either all rank-1 stats or none")

        su_cat = torch.cat([sigma_u1, sigma_u2], dim=-2)
        s2_cat = torch.cat([sigma2_1, sigma2_2], dim=-1)
        ga_cat = torch.cat([gamma_a1, gamma_a2], dim=-2)
        gb_cat = torch.cat([gamma_b1, gamma_b2], dim=-2)
        gm_cat = torch.cat([gamma1, gamma2], dim=-1)

        sua, sub = su_cat[..., 0::2, :], su_cat[..., 1::2, :]
        s2a, s2b = s2_cat[..., 0::2], s2_cat[..., 1::2]
        gaa, gab = ga_cat[..., 0::2, :], ga_cat[..., 1::2, :]
        gba, gbb = gb_cat[..., 0::2, :], gb_cat[..., 1::2, :]
        gma, gmb = gm_cat[..., 0::2], gm_cat[..., 1::2]

        if has_imp:
            frac_a_imp = impa.float() / imp_total.float().clamp_min(1e-8)
            frac_b_imp = impb.float() / imp_total.float().clamp_min(1e-8)
            if imp_lambda >= 1.0:
                frac_a, frac_b = frac_a_imp, frac_b_imp
            else:
                frac_a_w = wa.float() / w_total.float().clamp_min(1e-8)
                frac_b_w = wb.float() / w_total.float().clamp_min(1e-8)
                frac_a = imp_lambda * frac_a_imp + (1.0 - imp_lambda) * frac_a_w
                frac_b = imp_lambda * frac_b_imp + (1.0 - imp_lambda) * frac_b_w
        else:
            frac_a = wa.float() / w_total.float().clamp_min(1e-8)
            frac_b = wb.float() / w_total.float().clamp_min(1e-8)
        cross_frac = frac_a * frac_b

        dk = (ka - kb).float()
        dv = (va - vb).float()
        sigma_factors = torch.stack(
            [
                (frac_a * s2a.float()).clamp_min(0.0).sqrt().unsqueeze(-1) * sua.float(),
                (frac_b * s2b.float()).clamp_min(0.0).sqrt().unsqueeze(-1) * sub.float(),
                cross_frac.clamp_min(0.0).sqrt().unsqueeze(-1) * dk,
            ],
            dim=-2,
        )
        sigma_u, sigma2 = _rank1_psd_from_factors(sigma_factors)

        left_factors = torch.stack(
            [
                (frac_a * gma.float()).clamp_min(0.0).sqrt().unsqueeze(-1) * gba.float(),
                (frac_b * gmb.float()).clamp_min(0.0).sqrt().unsqueeze(-1) * gbb.float(),
                cross_frac.clamp_min(0.0).sqrt().unsqueeze(-1) * dv,
            ],
            dim=-2,
        )
        right_factors = torch.stack(
            [
                (frac_a * gma.float()).clamp_min(0.0).sqrt().unsqueeze(-1) * gaa.float(),
                (frac_b * gmb.float()).clamp_min(0.0).sqrt().unsqueeze(-1) * gab.float(),
                cross_frac.clamp_min(0.0).sqrt().unsqueeze(-1) * dk,
            ],
            dim=-2,
        )
        gamma_b, gamma_a, gamma = _rank1_cross_from_factors(left_factors, right_factors)
        stats_out = (
            k_out,
            v_out,
            w_total,
            sigma_u.to(k_out.dtype),
            sigma2.to(k_out.dtype),
            gamma_a.to(k_out.dtype),
            gamma_b.to(v_out.dtype),
            gamma.to(k_out.dtype),
        )
        return _with_anchors(stats_out if not has_imp else stats_out + (imp_total,))

    # ------------------------------------------------------------------
    # Add compact entry to level 0; carry to level 1+ when full
    # ------------------------------------------------------------------

    def _add_compact_entry(
        self,
        k_entry: torch.Tensor,
        v_entry: torch.Tensor,
        w_entry: torch.Tensor,
        sigma_u_entry: torch.Tensor | None = None,
        sigma2_entry: torch.Tensor | None = None,
        gamma_a_entry: torch.Tensor | None = None,
        gamma_b_entry: torch.Tensor | None = None,
        gamma_entry: torch.Tensor | None = None,
        imp_entry: torch.Tensor | None = None,
    ) -> None:
        """Add one compact entry to level 0. If level 0 is full, binary carry to levels 1+."""
        self._append_level0(
            k_entry.unsqueeze(2),
            v_entry.unsqueeze(2),
            w_entry.unsqueeze(2),
            None if sigma_u_entry is None else sigma_u_entry.unsqueeze(2),
            None if sigma2_entry is None else sigma2_entry.unsqueeze(2),
            None if gamma_a_entry is None else gamma_a_entry.unsqueeze(2),
            None if gamma_b_entry is None else gamma_b_entry.unsqueeze(2),
            None if gamma_entry is None else gamma_entry.unsqueeze(2),
            None if imp_entry is None else imp_entry.unsqueeze(2),
        )

    # ------------------------------------------------------------------
    # Binary carry: promote B entries through levels 1+
    # ------------------------------------------------------------------

    def _binary_carry(
        self,
        block_k: torch.Tensor,
        block_v: torch.Tensor,
        block_w: torch.Tensor,
        block_sigma_u: torch.Tensor | None = None,
        block_sigma2: torch.Tensor | None = None,
        block_gamma_a: torch.Tensor | None = None,
        block_gamma_b: torch.Tensor | None = None,
        block_gamma: torch.Tensor | None = None,
        block_imp: torch.Tensor | None = None,
    ) -> None:
        new_k, new_v, new_w = block_k, block_v, block_w
        new_su, new_s2 = block_sigma_u, block_sigma2
        new_ga, new_gb, new_gm = block_gamma_a, block_gamma_b, block_gamma
        new_imp = block_imp
        for ell in range(1, self.max_levels):
            ek, ev, ew = self._get_level(ell)
            if self._level_count(ell) == 0:
                self._set_level(ell, new_k, new_v, new_w, new_su, new_s2, new_ga, new_gb, new_gm, imp=new_imp)
                return

            eimp = self._get_level_imp(ell) if new_imp is not None else None
            if new_su is None:
                if new_imp is None:
                    new_k, new_v, new_w = self.compact(ek, ev, ew, new_k, new_v, new_w)
                else:
                    new_k, new_v, new_w, new_imp = self.compact(
                        ek, ev, ew, new_k, new_v, new_w,
                        imp1=eimp, imp2=new_imp, imp_lambda=self.importance_pooling_lambda,
                    )
            else:
                esu, es2, ega, egb, egm = self._get_level_stats(ell)
                if new_imp is None:
                    (
                        new_k,
                        new_v,
                        new_w,
                        new_su,
                        new_s2,
                        new_ga,
                        new_gb,
                        new_gm,
                    ) = self.compact(
                        ek, ev, ew,
                        new_k, new_v, new_w,
                        esu, es2, ega, egb, egm,
                        new_su, new_s2, new_ga, new_gb, new_gm,
                    )
                else:
                    (
                        new_k,
                        new_v,
                        new_w,
                        new_su,
                        new_s2,
                        new_ga,
                        new_gb,
                        new_gm,
                        new_imp,
                    ) = self.compact(
                        ek, ev, ew,
                        new_k, new_v, new_w,
                        esu, es2, ega, egb, egm,
                        new_su, new_s2, new_ga, new_gb, new_gm,
                        imp1=eimp, imp2=new_imp, imp_lambda=self.importance_pooling_lambda,
            )
            self._clear_level(ell)
        raise RuntimeError("LogStructuredKVCache: binary carry overflow; max_levels formula needs review.")

    # ------------------------------------------------------------------
    # Token accounting (append-only contract)
    # ------------------------------------------------------------------

    def _count_tokens(self, n: int) -> None:
        """Advance the committed-token counter, enforcing the sizing contract."""
        if self.token_count + n > self.max_seq_length:
            raise RuntimeError(
                f"LogStructuredKVCache: token overflow! "
                f"token_count={self.token_count}, n={n}, max_seq_length={self.max_seq_length}."
            )
        self.token_count += n

    # ------------------------------------------------------------------
    # Flush: compact oldest 2 tokens from buffer -> level 0
    # ------------------------------------------------------------------

    def _flush_recent(self) -> None:
        """Compact the oldest 2 tokens from the sliding window into level 0."""
        if self.recent_count < 2:
            return
        self._flush_pairs(2)

    def _flush_pairs(self, flush_len: int) -> None:
        """Batched equivalent of ``flush_len // 2`` sequential ``_flush_recent``
        calls: pairwise-merge the oldest ``flush_len`` window tokens into level-0
        entries in one shot, then shift the window once.

        The state trajectory is identical to the sequential version — the same
        (2i, 2i+1) pairs are merged with the same mean, entries reach level 0 in
        the same order, and carries fire at the same counts — but the O(window)
        clone+shift happens once instead of once per pair.
        """
        f = flush_len // 2
        rk = self.recent_k[:, :, :flush_len, :]
        rv = self.recent_v[:, :, :flush_len, :]
        B_, G_, _, kd = rk.shape
        vd = rv.size(-1)
        # Same arithmetic as _compact_tokens on each 2-token pair (mean over a
        # size-2 dim in the storage dtype), just for all f pairs at once.
        rk_pairs = rk.reshape(B_, G_, f, 2, kd)
        rv_pairs = rv.reshape(B_, G_, f, 2, vd)
        if self.importance_pooling:
            # Heuristic per-token importance: post-RoPE key L2 norm (no new
            # learnable params; a pure function of already-detached k, so no
            # backward-path change is needed -- see module docstring).
            imp_tok = rk.float().norm(dim=-1)  # (B, G, flush_len)
            temp = self.importance_pooling_temperature
            if temp != 1.0:
                imp_tok = imp_tok.clamp_min(_RANK1_EPS) ** temp
            imp_pairs = imp_tok.reshape(B_, G_, f, 2)
            imp_a, imp_b = imp_pairs[..., 0], imp_pairs[..., 1]
            pimp = imp_a + imp_b  # (B, G, f), raw (unclamped) cumulative mass
            frac_a_imp = imp_a / pimp.clamp_min(_RANK1_EPS)
            lam = self.importance_pooling_lambda
            # Raw tokens each carry w=1, so the uniform share of a pair is
            # exactly 0.5 -- blending toward it is `lam * frac_a_imp + (1-lam) * 0.5`.
            frac_a = (frac_a_imp if lam >= 1.0 else (lam * frac_a_imp + (1.0 - lam) * 0.5)).unsqueeze(-1)
            frac_b = 1.0 - frac_a
            pk = frac_a * rk_pairs[:, :, :, 0, :] + frac_b * rk_pairs[:, :, :, 1, :]
            pv = frac_a * rv_pairs[:, :, :, 0, :] + frac_b * rv_pairs[:, :, :, 1, :]
        else:
            pk = rk_pairs.mean(dim=3)
            pv = rv_pairs.mean(dim=3)
            pimp = None
            frac_a = frac_b = 0.5
        pw = torch.full((B_, G_, f), 2.0, device=rk.device, dtype=rk.dtype)
        if self.second_order:
            psu, ps2, pga, pgb, pgm = _pair_rank1_stats(
                rk_pairs[:, :, :, 0, :],
                rk_pairs[:, :, :, 1, :],
                rv_pairs[:, :, :, 0, :],
                rv_pairs[:, :, :, 1, :],
                frac_a.squeeze(-1) if self.importance_pooling else frac_a,
                frac_b.squeeze(-1) if self.importance_pooling else frac_b,
            )
            self._append_level0(pk, pv, pw, psu, ps2, pga, pgb, pgm, pimp=pimp)
        else:
            self._append_level0(pk, pv, pw, pimp=pimp)

        # Shift the survivors to the front. Source/destination overlap in the
        # same storage; PyTorch copy_ with overlapping src/dst is undefined (may
        # corrupt silently on CUDA), so the source must be cloned first.
        remaining = self.recent_count - flush_len
        if remaining > 0:
            self.recent_k[:, :, :remaining, :] = self.recent_k[:, :, flush_len:self.recent_count, :].clone()
            self.recent_v[:, :, :remaining, :] = self.recent_v[:, :, flush_len:self.recent_count, :].clone()
        self.recent_k[:, :, remaining:self.recent_count, :].zero_()
        self.recent_v[:, :, remaining:self.recent_count, :].zero_()
        self.recent_count = remaining

    def _semantic_input_positions(self, n: int, input_pos: torch.Tensor | None) -> torch.Tensor:
        if input_pos is None:
            return torch.arange(self.token_count, self.token_count + n, device=self.recent_pos.device).unsqueeze(0).expand(
                self.batch_size, -1
            )
        if input_pos.dim() == 1:
            if input_pos.numel() != n:
                raise ValueError(f"input_pos length {input_pos.numel()} != chunk size {n}")
            return input_pos.to(device=self.recent_pos.device, dtype=torch.int64).unsqueeze(0).expand(self.batch_size, -1)
        if input_pos.dim() == 2:
            if input_pos.shape != (self.batch_size, n):
                raise ValueError(f"input_pos shape {tuple(input_pos.shape)} != ({self.batch_size}, {n})")
            return input_pos.to(device=self.recent_pos.device, dtype=torch.int64)
        raise ValueError(f"input_pos must be 1-D or 2-D, got shape {tuple(input_pos.shape)}")

    def _semantic_input_positions_host(self, n: int, input_pos: torch.Tensor | None) -> list[list[int]]:
        if input_pos is None:
            row = list(range(self.token_count, self.token_count + n))
            return [row[:] for _ in range(self.batch_size)]
        if input_pos.dim() == 1:
            values = [int(x) for x in input_pos.detach().cpu().tolist()]
            return [values[:] for _ in range(self.batch_size)]
        if input_pos.dim() == 2:
            return [[int(x) for x in row] for row in input_pos.detach().cpu().tolist()]
        raise ValueError(f"input_pos must be 1-D or 2-D, got shape {tuple(input_pos.shape)}")

    def _semantic_flush_tokens(
        self,
        flush_len: int,
        *,
        record_op_log: bool = False,
        replay_op_log: torch.Tensor | None = None,
        replay_op_log_len: torch.Tensor | None = None,
        replay_op_log_host: list[list[list[tuple[int, int, int, int]]]] | None = None,
        replay_plans: _SemanticReplayPlans | None = None,
    ) -> None:
        if flush_len <= 0:
            return
        pending = int(flush_len)
        while pending > 0:
            take = min(self.semantic_flush_granularity, pending)
            if take > self.recent_count:
                raise RuntimeError(
                    f"semantic LogKV cannot flush {take} tokens from recent_count={self.recent_count}"
                )
            self.route_and_flush_batch(
                self.recent_k_raw[:, :, :take, :],
                self.recent_v[:, :, :take, :],
                self.recent_pos[:, :take],
                positions_host=[row[:take] for row in self._recent_pos_host],
                record_op_log=record_op_log,
                replay_op_log=replay_op_log,
                replay_op_log_len=replay_op_log_len,
                replay_op_log_host=replay_op_log_host,
                replay_plans=replay_plans,
                span_ends=[row[:take] for row in self._recent_span_ends] if self.alpha_exact_tokens else None,
            )
            remaining = self.recent_count - take
            if remaining > 0:
                # Disjoint ranges (remaining <= take, the training/prefill shape)
                # copy in place; only an overlapping shift needs a staging copy.
                stage = (lambda x: x) if remaining <= take else torch.Tensor.clone
                self.recent_k[:, :, :remaining, :].copy_(stage(self.recent_k[:, :, take:self.recent_count, :]))
                self.recent_k_raw[:, :, :remaining, :].copy_(stage(self.recent_k_raw[:, :, take:self.recent_count, :]))
                self.recent_v[:, :, :remaining, :].copy_(stage(self.recent_v[:, :, take:self.recent_count, :]))
                self.recent_pos[:, :remaining].copy_(stage(self.recent_pos[:, take:self.recent_count]))
                for b in range(self.batch_size):
                    self._recent_pos_host[b][:remaining] = self._recent_pos_host[b][take:self.recent_count]
                    if self.alpha_exact_tokens:
                        self._recent_span_ends[b][:remaining] = self._recent_span_ends[b][take:self.recent_count]
            for b in range(self.batch_size):
                self._recent_pos_host[b][remaining:self.recent_count] = [0] * (self.recent_count - remaining)
                if self.alpha_exact_tokens:
                    self._recent_span_ends[b][remaining:self.recent_count] = [False] * (self.recent_count - remaining)
            self.recent_k[:, :, remaining:self.recent_count, :].zero_()
            self.recent_k_raw[:, :, remaining:self.recent_count, :].zero_()
            self.recent_v[:, :, remaining:self.recent_count, :].zero_()
            self.recent_pos[:, remaining:self.recent_count].zero_()
            self.recent_count = remaining
            pending -= take
            self._route_mark_ready(self.recent_k.device)
        if self.alpha_exact_tokens and not self._replaying_updates:
            with logkv_timed("alpha_select"):
                self._alpha_prefetch_scores()

    def _semantic_add_recent(
        self,
        k_roped: torch.Tensor,
        v: torch.Tensor,
        *,
        k_raw: torch.Tensor,
        input_pos: torch.Tensor | None = None,
        record_op_log: bool = False,
        replay_op_log: torch.Tensor | None = None,
        replay_op_log_len: torch.Tensor | None = None,
        replay_op_log_host: list[list[list[tuple[int, int, int, int]]]] | None = None,
        replay_plans: _SemanticReplayPlans | None = None,
        span_ends: list[list[bool]] | None = None,
    ) -> None:
        n = k_roped.size(2)
        if n > self.recent_size:
            raise ValueError(f"chunk size {n} exceeds recent_size {self.recent_size}")
        if k_raw.shape[:3] != k_roped.shape[:3] or k_raw.size(-1) != self.k_dim:
            raise ValueError(f"k_raw shape {tuple(k_raw.shape)} does not match semantic cache shape")
        positions = self._semantic_input_positions(n, input_pos)
        positions_host = self._semantic_input_positions_host(n, input_pos)
        if self.alpha_exact_tokens:
            if span_ends is None or len(span_ends) != self.batch_size or any(len(row) != n for row in span_ends):
                raise ValueError("AlphaLogKV span boundaries must match the input batch and token count")
            for b in range(self.batch_size):
                self._recent_span_ends[b][self.recent_count:self.recent_count + n] = span_ends[b]
        self._count_tokens(n)
        if self.recent_count + n > self.recent_capacity:
            raise RuntimeError(
                f"semantic LogKV recent buffer overflow: recent_count={self.recent_count}, n={n}, "
                f"capacity={self.recent_capacity}"
            )
        dst = slice(self.recent_count, self.recent_count + n)
        self.recent_k[:, :, dst, :] = k_roped
        self.recent_k_raw[:, :, dst, :] = k_raw
        self.recent_v[:, :, dst, :] = v
        self.recent_pos[:, dst] = positions
        for b in range(self.batch_size):
            self._recent_pos_host[b][self.recent_count:self.recent_count + n] = positions_host[b]
        self.recent_count += n
        overflow = self.recent_count - self.recent_size
        if overflow > 0:
            flush_len = self.semantic_flush_granularity * ((overflow + self.semantic_flush_granularity - 1) // self.semantic_flush_granularity)
            self._semantic_flush_tokens(
                flush_len,
                record_op_log=record_op_log,
                replay_op_log=replay_op_log,
                replay_op_log_len=replay_op_log_len,
                replay_op_log_host=replay_op_log_host,
                replay_plans=replay_plans,
            )

    def _append_level0(
        self,
        pk: torch.Tensor,
        pv: torch.Tensor,
        pw: torch.Tensor,
        psu: torch.Tensor | None = None,
        ps2: torch.Tensor | None = None,
        pga: torch.Tensor | None = None,
        pgb: torch.Tensor | None = None,
        pgm: torch.Tensor | None = None,
        pimp: torch.Tensor | None = None,
    ) -> None:
        """Append f compact entries to level 0 in order, carrying when it fills.

        Trajectory-identical to f sequential ``_add_compact_entry`` calls: the
        binary carry fires exactly when the count reaches B, between the same
        two entries as in the sequential version.
        """
        f = pk.size(2)
        if f == 0:
            return
        off = 0
        while off < f:
            idx = self._level_count(0)
            take = min(self.B - idx, f - off)
            dst = slice(idx, idx + take)
            src = slice(off, off + take)
            self.level_k[:, :, 0, 0, dst, :].copy_(pk[:, :, src, :])
            self.level_v[:, :, 0, 0, dst, :].copy_(pv[:, :, src, :])
            self.level_w[:, :, 0, 0, dst].copy_(pw[:, :, src])
            if self.allocate_second_order:
                if psu is None:
                    self.level_sigma_u[:, :, 0, 0, dst, :].zero_()
                    self.level_sigma2[:, :, 0, 0, dst].zero_()
                    self.level_gamma_a[:, :, 0, 0, dst, :].zero_()
                    self.level_gamma_b[:, :, 0, 0, dst, :].zero_()
                    self.level_gamma[:, :, 0, 0, dst].zero_()
                else:
                    self.level_sigma_u[:, :, 0, 0, dst, :].copy_(psu[:, :, src, :])
                    self.level_sigma2[:, :, 0, 0, dst].copy_(ps2[:, :, src])
                    self.level_gamma_a[:, :, 0, 0, dst, :].copy_(pga[:, :, src, :])
                    self.level_gamma_b[:, :, 0, 0, dst, :].copy_(pgb[:, :, src, :])
                    self.level_gamma[:, :, 0, 0, dst].copy_(pgm[:, :, src])
            if pimp is None:
                self.level_imp[:, :, 0, 0, dst].zero_()
            else:
                self.level_imp[:, :, 0, 0, dst].copy_(pimp[:, :, src])
            self.pad_mask[:, :, 0, 0, dst].zero_()
            idx += take
            off += take
            self.level_count[:, :, 0, 0].fill_(idx)
            if hasattr(self, "_counts"):
                self._counts[0] = idx
            if idx == self.B:
                lk, lv, lw = self._get_level(0)
                limp = self._get_level_imp(0) if pimp is not None else None
                if self.second_order:
                    lsu, ls2, lga, lgb, lgm = self._get_level_stats(0)
                    self._binary_carry(
                        lk, lv, lw,
                        lsu, ls2, lga, lgb, lgm,
                        block_imp=limp,
                    )
                else:
                    self._binary_carry(lk, lv, lw, block_imp=limp)
                self._clear_level(0)

    # ------------------------------------------------------------------
    # Ingest chunk (testing): direct compact into level 0, bypass buffer
    # ------------------------------------------------------------------

    def ingest_chunk(
        self,
        k: torch.Tensor,   # (B, G, t_chunk, k_dim) full post-RoPE keys, detached
        v: torch.Tensor,   # (B, G, t_chunk, v_dim) detached
    ) -> None:
        """Compact a chunk of full keys + values straight into level 0.
        Bypasses the buffer. Intended for testing.

        When ``self.importance_pooling`` is set, uses the same heuristic
        per-token importance (post-RoPE key L2 norm) as the real streaming
        path (``_flush_pairs``) -- note this pools the whole chunk in one
        flat weighted average, not the real path's pairwise 2:1 hierarchy,
        so for ``t_chunk > 2`` this is a testing convenience, not a
        trajectory-identical stand-in (same caveat already applies to the
        unweighted default).
        """
        self._count_tokens(k.size(2))
        imp = k.float().norm(dim=-1) if self.importance_pooling else None
        if imp is not None and self.importance_pooling_temperature != 1.0:
            imp = imp.clamp_min(_RANK1_EPS) ** self.importance_pooling_temperature

        entries = self._compact_tokens(
            k, v, with_stats=True, imp=imp, imp_lambda=self.importance_pooling_lambda,
        )
        if self.importance_pooling:
            (
                k_entry, v_entry, w_entry,
                sigma_u_entry, sigma2_entry, gamma_a_entry, gamma_b_entry, gamma_entry,
                imp_entry,
            ) = entries
            imp_entry = imp_entry.squeeze(2)
        else:
            (
                k_entry, v_entry, w_entry,
                sigma_u_entry, sigma2_entry, gamma_a_entry, gamma_b_entry, gamma_entry,
            ) = entries
            imp_entry = None
        k_entry = k_entry.squeeze(2)
        v_entry = v_entry.squeeze(2)
        w_entry = w_entry.squeeze(2)
        sigma_u_entry = sigma_u_entry.squeeze(2)
        sigma2_entry = sigma2_entry.squeeze(2)
        gamma_a_entry = gamma_a_entry.squeeze(2)
        gamma_b_entry = gamma_b_entry.squeeze(2)
        gamma_entry = gamma_entry.squeeze(2)

        self._add_compact_entry(
            k_entry, v_entry, w_entry,
            sigma_u_entry, sigma2_entry, gamma_a_entry, gamma_b_entry, gamma_entry,
            imp_entry=imp_entry,
        )

    # ------------------------------------------------------------------
    # Add to recent (training): sliding window entry point
    # ------------------------------------------------------------------

    def add_recent(
        self,
        k: torch.Tensor,   # (B, G, n, k_dim) full post-RoPE keys, detached
        v: torch.Tensor,   # (B, G, n, v_dim) detached
        *,
        k_raw: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
        record_op_log: bool = False,
        replay_op_log: torch.Tensor | None = None,
        replay_op_log_len: torch.Tensor | None = None,
        replay_op_log_host: list[list[list[tuple[int, int, int, int]]]] | None = None,
        replay_plans: _SemanticReplayPlans | None = None,
        span_ends: list[list[bool]] | None = None,
    ) -> None:
        """Add tokens to the sliding window. When the window overflows, the
        oldest 2 tokens are flushed (compacted) into level 0.
        """
        if self.semantic_clusters:
            if k_raw is None:
                raise ValueError("semantic LogKV add_recent requires pre-RoPE k_raw")
            self._semantic_add_recent(
                k,
                v,
                k_raw=k_raw,
                input_pos=input_pos,
                record_op_log=record_op_log,
                replay_op_log=replay_op_log,
                replay_op_log_len=replay_op_log_len,
                replay_op_log_host=replay_op_log_host,
                replay_plans=replay_plans,
                span_ends=span_ends,
            )
            return
        n = k.size(2)
        # Explicit raise (not assert): must survive `python -O`.
        if n > self.recent_size:
            raise ValueError(f"chunk size {n} exceeds recent_size {self.recent_size}")

        self._count_tokens(n)

        # Batched flush: streaming lazily flushes the oldest pair each time the
        # window refills, so over this whole chunk it flushes ceil(overflow/2)
        # pairs — all taken from the CURRENT window front (appends only ever go
        # to the right). Flushing them up front in one batch reaches the exact
        # same final state with one shift instead of one per pair.
        overflow = self.recent_count + n - self.recent_size
        if overflow > 0:
            flush_len = 2 * ((overflow + 1) // 2)
            if flush_len <= self.recent_count:
                self._flush_pairs(flush_len)
            else:
                # Degenerate corner (odd recent_count with n == recent_size):
                # fall back to the lazy interleaved order.
                offset = 0
                while offset < n:
                    if self.recent_count == self.recent_size:
                        self._flush_recent()
                    capacity = self.recent_size - self.recent_count
                    if capacity <= 0:
                        raise RuntimeError(
                            "LogStructuredKVCache.add_recent() could not make room in the recent buffer; "
                            f"recent_count={self.recent_count}, recent_size={self.recent_size}."
                        )
                    take = min(capacity, n - offset)
                    self.recent_k[:, :, self.recent_count:self.recent_count + take, :] = k[:, :, offset:offset + take, :]
                    self.recent_v[:, :, self.recent_count:self.recent_count + take, :] = v[:, :, offset:offset + take, :]
                    self.recent_count += take
                    offset += take
                return

        self.recent_k[:, :, self.recent_count:self.recent_count + n, :] = k
        self.recent_v[:, :, self.recent_count:self.recent_count + n, :] = v
        self.recent_count += n

    # ------------------------------------------------------------------
    # nn.Module forward intentionally disabled
    # ------------------------------------------------------------------

    def forward(
        self,
        input_pos: torch.Tensor,
        k: torch.Tensor,   # (B, G, T, k_dim) post-RoPE
        v: torch.Tensor,   # (B, G, T, v_dim)
    ) -> NoReturn:
        """Disabled direct update API.

        LogKV attention must build a temporary state that includes the current
        token(s), compute slot attention from that state, and only then commit
        the token(s) to the cache. Calling the cache directly used to implement
        an incompatible raw-prefill / write-before-decode path, so it is
        intentionally disabled. ``input_pos`` is accepted only to keep the
        stale call site fail-fast and self-explanatory. Use
        ``GPT.set_log_kv_cache()`` and let ``CausalSelfAttention`` manage the
        update order, or call
        ``get_attention_state()`` and ``add_recent()`` explicitly.
        """
        raise RuntimeError(
            "LogStructuredKVCache.forward() is disabled/deprecated: direct cache calls cannot implement "
            "LogKV correctly because attention must include the current token(s) before committing them. "
            "Use GPT.set_log_kv_cache() so CausalSelfAttention owns the update order, or use the explicit "
            "get_attention_state()/add_recent() helpers in tests."
        )

    # ------------------------------------------------------------------
    # Build attention state: slot-granular, O(recent + B*log N) entries
    # ------------------------------------------------------------------

    @logkv_timed("plan")
    def _semantic_attention_plan(self):
        """Immutable slot indices and anchor metadata; no K/V payload is retained."""
        if self.semantic_anchor_mode == "mid":
            return self._semantic_mid_attention_plan()
        shape = (self.batch_size, self.n_groups, -1)
        entry_w = self.level_w.flip(3).reshape(shape)
        p_lo = self.level_p_lo.flip(3).reshape(shape)
        p_hi = self.level_p_hi.flip(3).reshape(shape)
        sum_wp = self.level_sum_wp.flip(3).reshape(shape)
        anchors, valid3, M = dedup_anchors(p_lo, p_hi, mid_anchor(p_lo, p_hi, sum_wp, entry_w), entry_w)
        flat_valid = valid3.reshape(shape)
        valid_count = flat_valid.sum(dim=-1)
        pooled_slots = int(valid_count.max().item())
        flat_src = torch.arange(flat_valid.size(-1), device=flat_valid.device).view(1, 1, -1).expand_as(flat_valid)
        gather_src = flat_src.masked_fill(~flat_valid, flat_valid.size(-1)).sort(dim=-1).values[..., :pooled_slots]
        gather_src = gather_src.clamp_max(flat_valid.size(-1) - 1)
        entry_idx = gather_src // 3
        ell = (entry_idx // self.B_prime) % self.L_alloc
        physical_idx = entry_idx + (self.L_alloc - 1 - 2 * ell) * self.B_prime
        anchor_sel = torch.gather(anchors.reshape(shape), 2, gather_src)
        M_s = torch.gather(M, 2, entry_idx)
        slot_valid = torch.arange(pooled_slots, device=flat_valid.device).view(1, 1, -1) < valid_count.unsqueeze(-1)
        return physical_idx, anchor_sel, M_s, slot_valid

    def _semantic_mid_attention_plan(self):
        """Enumerate occupied ranges from host counts; no sort, flip, or CUDA scalar read.

        Midpoints of positive-weight entries already lie inside [lo, hi], so
        this path never reads either endpoint. Segment-alignment pads remain
        masked inside occupied ranges; there are none in the fast configuration.
        """
        # Whole-array form of the per-lane (cluster, top level first) loop.
        counts = np.asarray(self._semantic_counts, dtype=np.int64)[..., ::-1]
        counts = counts.reshape(-1, self.K_max * self.L_alloc)
        levels = np.arange(self.L_alloc - 1, -1, -1, dtype=np.int64)
        starts = ((np.arange(self.K_max, dtype=np.int64)[:, None] * self.L_alloc + levels) * self.B_prime).reshape(-1)
        sizes = counts.sum(1)
        width = int(sizes.max(initial=0))
        indices = np.full((len(counts), width), -1, dtype=np.int64)
        flat = _spans_to_index_array(np.stack((np.broadcast_to(starts, counts.shape), starts + counts), -1).reshape(-1, 2))
        lane = np.repeat(np.arange(len(counts)), sizes)
        indices[lane, np.arange(len(flat)) - np.repeat(np.cumsum(sizes) - sizes, sizes)] = flat
        shape = (self.batch_size, self.n_groups, width)
        physical_idx = _upload(indices, self.level_w.device).view(shape)
        occupied = physical_idx >= 0
        physical_idx = physical_idx.clamp_min(0)
        weights = self.level_w.flatten(2).gather(2, physical_idx)
        sums = self.level_sum_wp.flatten(2).gather(2, physical_idx)
        valid = occupied & (weights > 0)
        ww = weights.long().clamp_min(1)
        anchors = torch.div(2 * sums + ww, 2 * ww, rounding_mode="floor")
        anchors = anchors.masked_fill(~valid, 0)
        return physical_idx, anchors, torch.ones_like(anchors), valid

    def _semantic_attention_state(self, with_stats: bool = False, plan=None) -> CacheAttentionState:
        assert self.rope_n_elem is not None
        physical_idx, anchor_sel, M_s, slot_valid = self._semantic_attention_plan() if plan is None else plan
        pooled_slots = physical_idx.size(-1)
        entry_w = self.level_w.reshape(self.batch_size, self.n_groups, -1)

        def gather_entry(x):
            x = x.reshape(self.batch_size, self.n_groups, -1, x.size(-1))
            return torch.gather(x, 2, physical_idx.unsqueeze(-1).expand(*physical_idx.shape, x.size(-1)))

        def gather_scalar(x):
            return torch.gather(x.reshape_as(entry_w), 2, physical_idx)

        if pooled_slots:
            slot_k = materialize_anchor_keys(
                gather_entry(self.level_k), anchor_sel.unsqueeze(-1), self.cos_cache, self.sin_cache, self.rope_n_elem
            ).squeeze(-2)
            slot_v = gather_entry(self.level_v)
            slot_w = gather_scalar(entry_w)
        else:
            slot_k = self.recent_k[:, :, :0, :]
            slot_v = self.recent_v[:, :, :0, :]
            slot_w = self.level_w[:, :, 0, 0, :0]

        k_parts: list[torch.Tensor] = [slot_k]
        v_parts: list[torch.Tensor] = [slot_v]
        w_parts: list[torch.Tensor] = [slot_w]
        if self.alpha_count:
            count = self.alpha_count
            valid = self.alpha_valid[:, None, :count].expand(-1, self.n_groups, -1)
            anchors = self.alpha_pos[:, None, :count, None].expand(-1, self.n_groups, -1, -1)
            keys = materialize_anchor_keys(self.alpha_k_raw[:, :, :count], anchors,
                                          self.cos_cache, self.sin_cache, self.rope_n_elem).squeeze(-2)
            k_parts.append(keys.masked_fill(~valid[..., None], 0))
            v_parts.append(self.alpha_v[:, :, :count])
            w_parts.append(valid.float())
            slot_valid = torch.cat((slot_valid, valid), dim=-1)
            M_s = torch.cat((M_s, torch.ones_like(valid, dtype=M_s.dtype)), dim=-1)
        sigma_u_parts: list[torch.Tensor] = []
        sigma2_parts: list[torch.Tensor] = []
        gamma_a_parts: list[torch.Tensor] = []
        gamma_b_parts: list[torch.Tensor] = []
        gamma_parts: list[torch.Tensor] = []
        if with_stats:
            entry_su = self.level_sigma_u.reshape(
                self.batch_size, self.n_groups, self.K_max * self.L_alloc * self.B_prime, self.k_dim
            )
            entry_ga = self.level_gamma_a.reshape_as(entry_su)
            if pooled_slots > 0:
                su, ga = materialize_anchor_directions(
                    gather_entry(entry_su), gather_entry(entry_ga), anchor_sel.unsqueeze(-1),
                    self.cos_cache, self.sin_cache, self.rope_n_elem
                )
                sigma_u_parts.append(su.squeeze(-2))
                gamma_a_parts.append(ga.squeeze(-2))
                sigma2_parts.append(gather_scalar(self.level_sigma2))
                gamma_b_parts.append(gather_entry(
                    self.level_gamma_b.reshape(
                        self.batch_size, self.n_groups, self.K_max * self.L_alloc * self.B_prime, self.v_dim
                    )
                ))
                gamma_parts.append(gather_scalar(self.level_gamma))
            else:
                sigma_u_parts.append(self.recent_k[:, :, :0, :])
                sigma2_parts.append(self.level_sigma2[:, :, 0, 0, :0])
                gamma_a_parts.append(self.recent_k[:, :, :0, :])
                gamma_b_parts.append(self.recent_v[:, :, :0, :])
                gamma_parts.append(self.level_gamma[:, :, 0, 0, :0])

        if self.recent_count > 0:
            k_parts.append(self.recent_k[:, :, :self.recent_count, :])
            v_parts.append(self.recent_v[:, :, :self.recent_count, :])
            w_parts.append(self.level_w.new_ones(self.batch_size, self.n_groups, self.recent_count))
            if with_stats:
                sigma_u_parts.append(self.recent_k[:, :, :self.recent_count, :].new_zeros(
                    self.batch_size, self.n_groups, self.recent_count, self.slot_k_dim
                ))
                sigma2_parts.append(self.level_w.new_zeros(self.batch_size, self.n_groups, self.recent_count))
                gamma_a_parts.append(self.recent_k[:, :, :self.recent_count, :].new_zeros(
                    self.batch_size, self.n_groups, self.recent_count, self.slot_k_dim
                ))
                gamma_b_parts.append(self.recent_v[:, :, :self.recent_count, :].new_zeros(
                    self.batch_size, self.n_groups, self.recent_count, self.v_dim
                ))
                gamma_parts.append(self.level_w.new_zeros(self.batch_size, self.n_groups, self.recent_count))

        if not with_stats:
            return CacheAttentionState(
                torch.cat(k_parts, dim=-2),
                torch.cat(v_parts, dim=-2),
                torch.cat(w_parts, dim=-1),
                slot_valid=slot_valid,
                M_s=M_s,
            )
        return CacheAttentionState(
            slot_k=torch.cat(k_parts, dim=-2),
            slot_v=torch.cat(v_parts, dim=-2),
            slot_w=torch.cat(w_parts, dim=-1),
            slot_valid=slot_valid,
            M_s=M_s,
            slot_sigma_u=torch.cat(sigma_u_parts, dim=-2),
            slot_sigma2=torch.cat(sigma2_parts, dim=-1),
            slot_gamma_a=torch.cat(gamma_a_parts, dim=-2),
            slot_gamma_b=torch.cat(gamma_b_parts, dim=-2),
            slot_gamma=torch.cat(gamma_parts, dim=-1),
        )

    def get_attention_state(self, with_stats: bool = False, *, plan=None) -> CacheAttentionState:
        if self.semantic_clusters and plan is None:
            plan = self._semantic_attention_plan()
        with logkv_timed("pack"):
            return self._get_attention_state(with_stats=with_stats and self.allocate_second_order, plan=plan)

    def _get_attention_state(self, with_stats: bool = False, *, plan=None) -> CacheAttentionState:
        """Assemble the cache state for ``log_kv_slot_attention``.

        Returns:
            slot_k: (B, G, n_slots, k_dim) — time-ordered slot keys: compact
                    levels oldest (highest level) first down to level 0, then
                    exact recent slots.
            slot_v: (B, G, n_slots, v_dim)
            slot_w: (B, G, n_slots) — token count per slot (1 for recent).
            Non-semantic caches concatenate only real slots and leave
            ``slot_valid=None``. Semantic caches use a fixed anchor layout and
            return ``slot_valid`` for empty anchors. If ``with_stats=True``,
            also fills:
            slot_sigma_u / slot_sigma2: rank-1 key covariance stats;
            slot_gamma_a / slot_gamma_b / slot_gamma: rank-1 value-key cross
                covariance stats. Exact recent tokens have zero stats.
        """
        if self.semantic_clusters:
            return self._semantic_attention_state(with_stats=with_stats, plan=plan)
        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        w_parts: list[torch.Tensor] = []
        sigma_u_parts: list[torch.Tensor] = []
        sigma2_parts: list[torch.Tensor] = []
        gamma_a_parts: list[torch.Tensor] = []
        gamma_b_parts: list[torch.Tensor] = []
        gamma_parts: list[torch.Tensor] = []

        # Compact levels: oldest first. Counts come from the host mirror; reading
        # the CUDA tensor here would sync once per level, per chunk, per layer.
        for ell in range(self.L_alloc - 1, -1, -1):
            count = self._level_count(ell)
            if count <= 0:
                continue
            lk, lv, lw = self._get_level(ell)
            k_parts.append(lk[:, :, :count, :])
            v_parts.append(lv[:, :, :count, :])
            w_parts.append(lw[:, :, :count])
            if with_stats:
                lsu, ls2, lga, lgb, lgm = self._get_level_stats(ell)
                sigma_u_parts.append(lsu[:, :, :count, :])
                sigma2_parts.append(ls2[:, :, :count])
                gamma_a_parts.append(lga[:, :, :count, :])
                gamma_b_parts.append(lgb[:, :, :count, :])
                gamma_parts.append(lgm[:, :, :count])

        # Recent window: exact tokens, weight 1 each
        if self.recent_count > 0:
            k_parts.append(self.recent_k[:, :, :self.recent_count, :])
            v_parts.append(self.recent_v[:, :, :self.recent_count, :])
            w_parts.append(
                self.level_w.new_ones(self.batch_size, self.n_groups, self.recent_count)
            )
            if with_stats:
                sigma_u_parts.append(self.recent_k[:, :, :self.recent_count, :].new_zeros(
                    self.batch_size, self.n_groups, self.recent_count, self.k_dim
                ))
                sigma2_parts.append(self.recent_k.new_zeros(self.batch_size, self.n_groups, self.recent_count))
                gamma_a_parts.append(self.recent_k[:, :, :self.recent_count, :].new_zeros(
                    self.batch_size, self.n_groups, self.recent_count, self.k_dim
                ))
                gamma_b_parts.append(self.recent_v[:, :, :self.recent_count, :].new_zeros(
                    self.batch_size, self.n_groups, self.recent_count, self.v_dim
                ))
                gamma_parts.append(self.recent_k.new_zeros(self.batch_size, self.n_groups, self.recent_count))

        if not w_parts:
            slot_k = self.recent_k[:, :, :0, :]
            slot_v = self.recent_v[:, :, :0, :]
            slot_w = self.level_w[:, :, 0, 0, :0]
            if not with_stats:
                return CacheAttentionState(slot_k, slot_v, slot_w)
            # slot_sigma2/slot_gamma must carry the activation dtype (matching
            # level_sigma2/level_gamma), not level_w's hardcoded float32 --
            # reusing slot_w here silently promotes downstream bf16 matmuls to
            # fp32 and then crashes on the dtype-mismatched matmul against
            # slot_gamma_b (recent_v's dtype).
            slot_stat0 = self.recent_k[:, :, :0, 0]
            return CacheAttentionState(
                slot_k=slot_k,
                slot_v=slot_v,
                slot_w=slot_w,
                slot_sigma_u=self.recent_k[:, :, :0, :],
                slot_sigma2=slot_stat0,
                slot_gamma_a=self.recent_k[:, :, :0, :],
                slot_gamma_b=self.recent_v[:, :, :0, :],
                slot_gamma=slot_stat0,
            )

        slot_k = torch.cat(k_parts, dim=-2)
        slot_v = torch.cat(v_parts, dim=-2)
        slot_w = torch.cat(w_parts, dim=-1)
        if not with_stats:
            return CacheAttentionState(slot_k, slot_v, slot_w)
        return CacheAttentionState(
            slot_k=slot_k,
            slot_v=slot_v,
            slot_w=slot_w,
            slot_sigma_u=torch.cat(sigma_u_parts, dim=-2),
            slot_sigma2=torch.cat(sigma2_parts, dim=-1),
            slot_gamma_a=torch.cat(gamma_a_parts, dim=-2),
            slot_gamma_b=torch.cat(gamma_b_parts, dim=-2),
            slot_gamma=torch.cat(gamma_parts, dim=-1),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _apply(self, fn, recurse=True):
        self._mid_decode_state = None
        self._mid_flash_support = None
        return super()._apply(fn, recurse=recurse)

    def _convert_dtype(self, dtype: torch.dtype) -> None:
        """Reconcile all data buffers with the activation dtype.

        Needed when the cache was allocated at a different (default) dtype than
        the running activations — e.g. bf16 inference where ``set_log_kv_cache``
        received no explicit ``dtype`` and the process default is fp32. Without
        this, ``get_attention_state`` / ``add_recent`` would try to ``cat`` /
        ``matmul`` fp32 buffers against bf16 activations and raise.

        Idempotent and cheap: a no-op once the buffers already match, so it is
        safe to call on every forward (mirrors ``KVCache.forward``'s reconcile).
        ``level_count``/``pad_mask`` stay integer/bool and are deliberately
        untouched. ``level_w``/``level_imp`` stay fp32 per the indexed storage
        contract: they feed scalar merge/bias math, not activation matmuls.
        """
        if self.recent_k.dtype == dtype:
            return
        self._mid_decode_state = None
        self.recent_k = self.recent_k.to(dtype)
        self.recent_v = self.recent_v.to(dtype)
        if self.semantic_clusters:
            self.recent_k_raw = self.recent_k_raw.to(dtype)
        if self.alpha_exact_tokens:
            self.alpha_k_raw = self.alpha_k_raw.to(dtype)
            self.alpha_v = self.alpha_v.to(dtype)
        self.level_k = self.level_k.to(dtype)
        self.level_v = self.level_v.to(dtype)
        if self.allocate_second_order:
            self.level_sigma_u = self.level_sigma_u.to(dtype)
            self.level_sigma2 = self.level_sigma2.to(dtype)
            self.level_gamma_a = self.level_gamma_a.to(dtype)
            self.level_gamma_b = self.level_gamma_b.to(dtype)
            self.level_gamma = self.level_gamma.to(dtype)

    def reset_parameters(self) -> None:
        """Reset all buffers to zero."""
        self._mid_decode_state = None
        self.token_count = 0
        self.recent_k.zero_()
        self.recent_v.zero_()
        if self.semantic_clusters:
            self.recent_k_raw.zero_()
            self.recent_pos.zero_()
            for row in self._recent_pos_host:
                row[:] = [0] * self.recent_capacity
        self.recent_count = 0
        self.alpha_count = 0
        if self.alpha_exact_tokens:
            for name in ("alpha_k_raw", "alpha_v", "alpha_pos", "alpha_valid"):
                getattr(self, name).zero_()
            self._alpha_spans = [[] for _ in range(self.batch_size)]
            self._alpha_positions = [[] for _ in range(self.batch_size)]
            self._recent_span_ends = [[False] * self.recent_capacity for _ in range(self.batch_size)]
            self._alpha_prefetch = None
        self._route_ready = None
        self.level_k.zero_()
        self.level_v.zero_()
        self.level_w.zero_()
        self.level_imp.zero_()
        if self.allocate_second_order:
            self.level_sigma_u.zero_()
            self.level_sigma2.zero_()
            self.level_gamma_a.zero_()
            self.level_gamma_b.zero_()
            self.level_gamma.zero_()
        self.level_count.zero_()
        if hasattr(self, "_counts"):
            self._counts[:] = [0] * self.L_alloc
        if hasattr(self, "_semantic_counts"):
            for b_counts in self._semantic_counts:
                for g_counts in b_counts:
                    for c_counts in g_counts:
                        c_counts[:] = [0] * self.L_alloc
        self.pad_mask.zero_()
        if self.semantic_clusters:
            self.level_p_lo.zero_()
            self.level_p_hi.zero_()
            self.level_sum_wp.zero_()
            self.level_order.zero_()
            self.centroid.zero_()
            self.n_eff.zero_()
            self.n_total.zero_()
            self.p_hi_c.fill_(-1)
            self.current_segment.zero_()
            self.level0_phase.zero_()
            self.alive.zero_()
            self.ward_cost.fill_(float("inf"))
            for b in range(self.batch_size):
                for g in range(self.n_groups):
                    self._semantic_alive[b][g][:] = [False] * self.K_max
                    self._semantic_p_hi_c[b][g][:] = [-1] * self.K_max
                    self._semantic_current_segment[b][g][:] = [0] * self.K_max
                    self._semantic_level0_phase[b][g][:] = [0] * self.K_max
                    self._semantic_n_total[b][g][:] = [0] * self.K_max
                    self._semantic_ward_dirty[b][g] = True
            self.op_log = None
            self.op_log_len = None
            self._op_replay_cursor = None
            self._op_replay_cursor_host = None

    @property
    def total_slots(self) -> int:
        if hasattr(self, "_counts"):
            return self.recent_count + sum(self._counts)
        if hasattr(self, "_semantic_counts"):
            return self.recent_count + self.alpha_count + sum(
                count
                for c_counts in self._semantic_counts[0][0]
                for count in c_counts
            )
        return self.recent_count + int(self.level_count[0, 0].sum().item())

    @property
    def total_tokens_covered(self) -> int:
        return self.token_count

    def semantic_cluster_stats(self) -> dict[str, Any]:
        if not self.semantic_clusters:
            return {"semantic_clusters": False}
        with torch.no_grad():
            alive = self.alive.detach()
            n = self.n_total.detach().to(torch.float32)
            target = self._semantic_capacity_target()
            live_counts = alive.sum(dim=-1).to(torch.float32)
            max_tokens = n.masked_fill(~alive, 0).max(dim=-1).values
            live_n = n[alive]
            ratios = max_tokens / target
            top_counts = self.level_count[..., -1].detach().to(torch.float32)
            top_live = top_counts.masked_fill(~alive, 0)
            return {
                "semantic_clusters": True,
                "alpha_exact_tokens": self.alpha_exact_tokens,
                "alpha_visible_slots": self.alpha_count,
                "effective_B": self.B,
                "target_tokens_per_cluster": target,
                "K_max_binding_rate": float((live_counts >= self.K_max).to(torch.float32).mean().item()),
                "live_clusters_mean": float(live_counts.mean().item()),
                "live_clusters_min": int(live_counts.min().item()),
                "live_clusters_max": int(live_counts.max().item()),
                "max_cluster_tokens_mean": float(max_tokens.mean().item()),
                "max_cluster_tokens_max": int(max_tokens.max().item()),
                "max_cluster_tokens_over_target_mean": float(ratios.mean().item()),
                "max_cluster_tokens_over_target_max": float(ratios.max().item()),
                "live_cluster_tokens_mean": float(live_n.mean().item()) if live_n.numel() else 0.0,
                "top_level_live_slots": int(top_live.sum().item()),
                "top_level_full_clusters": int(((top_counts >= self.B) & alive).sum().item()),
            }


# ======================================================================
# Slot attention: merged-position slots + log-multiplicity mass bias
# ======================================================================


@logkv_timed("pack")
def append_exact_tokens(
    state: CacheAttentionState,
    k_new: torch.Tensor,    # (B, G, n, k_dim) exact tokens, appended in time order
    v_new: torch.Tensor,    # (B, G, n, v_dim)
) -> CacheAttentionState:
    """Append exact per-token entries (w=1 slots) after the cached slots.

    Used by the attention layer to make the current chunk (and any pending
    token) visible to attention BEFORE it is committed to the cache. Gradient
    flows through ``k_new``/``v_new``; the ones-weights are constants.
    ``slot_valid``/``M_s`` cover only the pooled prefix and are passed through.
    If slot rank-1 stats are provided, appended exact tokens receive zero stats.
    """
    n_new = k_new.size(2)
    ones = state.slot_w.new_ones(state.slot_w.size(0), state.slot_w.size(1), n_new)
    k_all = torch.cat([state.slot_k, k_new], dim=2)
    v_all = torch.cat([state.slot_v, v_new], dim=2)
    w_all = torch.cat([state.slot_w, ones], dim=-1)
    if state.slot_sigma_u is None:
        if any(x is not None for x in (state.slot_sigma2, state.slot_gamma_a, state.slot_gamma_b, state.slot_gamma)):
            raise ValueError("append_exact_tokens() requires either all rank-1 stats or none")
        return state._replace(slot_k=k_all, slot_v=v_all, slot_w=w_all)

    if any(x is None for x in (state.slot_sigma2, state.slot_gamma_a, state.slot_gamma_b, state.slot_gamma)):
        raise ValueError("append_exact_tokens() requires either all rank-1 stats or none")
    return state._replace(
        slot_k=k_all,
        slot_v=v_all,
        slot_w=w_all,
        slot_sigma_u=torch.cat([state.slot_sigma_u, torch.zeros_like(k_new)], dim=2),
        slot_sigma2=torch.cat([state.slot_sigma2, state.slot_sigma2.new_zeros(ones.shape)], dim=-1),
        slot_gamma_a=torch.cat([state.slot_gamma_a, torch.zeros_like(k_new)], dim=2),
        slot_gamma_b=torch.cat([state.slot_gamma_b, torch.zeros_like(v_new)], dim=2),
        slot_gamma=torch.cat([state.slot_gamma, state.slot_gamma.new_zeros(ones.shape)], dim=-1),
    )


def _slot_mass_bias(slot_w: torch.Tensor, slot_M: torch.Tensor | None, lam: float) -> torch.Tensor | None:
    if slot_M is None:
        return lam * slot_w.to(torch.float32).log() if lam != 0.0 else None
    if slot_M.size(-1) > slot_w.size(-1):
        raise ValueError("slot_M cannot be wider than slot_w")
    bias = lam * slot_w.to(torch.float32).log() if lam != 0.0 else torch.zeros_like(slot_w, dtype=torch.float32)
    s_m = slot_M.size(-1)
    bias[..., :s_m] = anchor_mass_bias(slot_w[..., :s_m], slot_M, lam)
    return bias


def _slot_sdpa_inputs(q, slot_k, slot_v, slot_w, slot_M, slot_valid, scale, lam):
    """Encode the slot bias as one extra dot-product coordinate (no Q x S mask).

    scale * [q, 1/scale] @ [k, bias].T = scale*q@k.T + bias.
    Padding Q/K/V to the same multiple of eight keeps Flash SDPA eligible.
    """
    dim = ((max(q.size(-1) + 1, slot_v.size(-1)) + 7) // 8) * 8
    bias = _slot_mass_bias(slot_w, slot_M, lam)
    if bias is None:
        bias = torch.zeros_like(slot_w, dtype=torch.float32)
    k_aug = F.pad(slot_k, (0, dim - slot_k.size(-1)))
    v_aug = F.pad(slot_v, (0, dim - slot_v.size(-1)))
    if slot_valid is not None:
        valid = F.pad(slot_valid, (0, slot_k.size(2) - slot_valid.size(-1)), value=True)
        # Invalid pooled slots can contain arbitrary payload. Zero them before
        # the dot product, so their content cannot overcome the mask sentinel.
        k_aug.masked_fill_(~valid.unsqueeze(-1), 0)
        v_aug.masked_fill_(~valid.unsqueeze(-1), 0)
        # ponytail: finite masking avoids inf*0 in Flash backward's augmented
        # coordinate; -10000 underflows for normal model logits. Use a packed
        # variable-length kernel if arbitrary extreme logits must be supported.
        bias = bias.masked_fill(~valid, -10000.0)
    q_aug = F.pad(q, (0, dim - q.size(-1)))
    q_aug[..., q.size(-1)] = 1.0 / scale
    k_aug[..., q.size(-1)] = bias.to(q.dtype)
    return q_aug, k_aug, v_aug


def _slot_flash_attention(q, slot_k, slot_v, slot_w, scale, lam, causal_tail, slot_M, slot_valid, check_valid):
    """Return None when Flash cannot serve this call; never silently run SDPA math."""
    if (
        q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16)
        or slot_k.dtype != q.dtype or slot_v.dtype != q.dtype
        or not math.isfinite(scale) or not 1e-3 <= scale <= 1.0
        or q.size(-1) != slot_k.size(-1) or q.size(1) % slot_k.size(1)
        # Keep the no-grad forward eligible for backward on consumer GPUs too.
        or max(q.size(-1) + 1, slot_v.size(-1)) > 192
        or q.size(2) == 0 or causal_tail > slot_k.size(2)
    ):
        return None
    if slot_valid is not None:
        if slot_valid.size(-1) > slot_k.size(2) - causal_tail:
            return None  # The fast mask assumes the causal tail is all valid.
        if check_valid and slot_valid.size(-1) == slot_k.size(2) and not slot_valid.any(dim=-1).all():
            raise ValueError("log_kv_slot_attention(): a query row has no valid slot")
        if not check_valid and slot_valid.size(-1) == slot_k.size(2):
            return None  # No guaranteed visible exact suffix; preserve legacy all-masked behavior.
    q_aug, k_aug, v_aug = _slot_sdpa_inputs(q, slot_k, slot_v, slot_w, slot_M, slot_valid, scale, lam)
    gqa = q.size(1) != slot_k.size(1)
    # LOWER_RIGHT's dispatcher checks the same unmasked parameters internally.
    params = SDPAParams(q_aug, k_aug, v_aug, None, 0.0, False, gqa)
    if not can_use_flash_attention(params):
        return None
    causal_bias = causal_lower_right(q.size(2), slot_k.size(2)) if causal_tail else None
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(
            q_aug, k_aug, v_aug, attn_mask=causal_bias, dropout_p=0.0, scale=scale, enable_gqa=gqa,
        )
    return out[..., :slot_v.size(-1)]


@logkv_timed("attn_fwd")
def log_kv_slot_attention(
    q: torch.Tensor,        # (B, nh, T_q, k_dim) full post-RoPE queries
    slot_k: torch.Tensor,   # (B, G, S, k_dim) merged slot keys (position included)
    slot_v: torch.Tensor,   # (B, G, S, v_dim)
    slot_w: torch.Tensor,   # (B, G, S) token count per slot (>= 1)
    scale: float,
    mask: torch.Tensor | None = None,  # (T_q, S) bool, True = attend
    lam: float = 1.0,
    causal_tail: int = 0,
    slot_M: torch.Tensor | None = None,       # (B, G, S) distinct-anchor count per slot
    slot_valid: torch.Tensor | None = None,   # (B, G, S_pooled) bool; pooled-prefix only
    check_valid: bool = True,   # raise if a masked query row has no valid slot;
                                 # costs a host sync (.all() -> Python bool) per
                                 # call, so hot-path callers that can prove by
                                 # construction (via causal_tail) that the
                                 # appended exact suffix always keeps >= 1 valid
                                 # slot pass False. Masking itself always runs.
    slot_sigma_u: torch.Tensor | None = None,  # (B, G, S, k_dim)
    slot_sigma2: torch.Tensor | None = None,   # (B, G, S)
    slot_gamma_a: torch.Tensor | None = None,  # (B, G, S, k_dim)
    slot_gamma_b: torch.Tensor | None = None,  # (B, G, S, v_dim)
    slot_gamma: torch.Tensor | None = None,    # (B, G, S)
    second_order_scale: float = 1.0,
) -> torch.Tensor:
    """Slot-granular attention over merged-position entries.

        score_s = scale·(q · k_s) + 0.5·scale²·sigma2_s·(q · sigma_u_s)² + λ·log(w_s)
        read_s  = v_s + scale·gamma_s·(q · gamma_a_s)·gamma_b_s
        out     = softmax(score) · read

    One logit per SLOT — compute and memory are O(S) = O(recent + B·log N),
    never O(N). The position information lives inside ``slot_k`` (expected-RoPE
    sub-channel, see module docstring), so a single dot product covers both the
    content and the position score. The +λ·log(w_s) bias restores the softmax
    mass a w-token span would have contributed token-by-token; it is exact when
    the span's tokens are identical and first-order otherwise. When the optional
    rank-1 stats are present, the Sigma/Gamma terms add the cheap second-order
    slot corrections. λ=1 preserves mass ∝ token count; λ=0 yields a built-in
    ∝1/w long-range forgetting curve (ablation knob).

    Note the bias is added AFTER ``scale``: it is a multiplicity correction on
    the logits, not a similarity, so it must not be shrunk by 1/sqrt(d).

    Preconditions (enforced by construction in the callers):
      - ``slot_w`` entries are >= 1 (log is finite);
      - every ``mask`` row has at least one True (softmax row is finite) —
        chunk-causal masks always allow the diagonal;
      - if ``slot_valid`` is given, every row must keep >= 1 valid slot after
        all masking — checked at runtime (raises) since this one isn't
        guaranteed by construction the way ``mask``/``causal_tail`` are.

    Args:
        q: (B, nh, T_q, k_dim)
        slot_k / slot_v / slot_w: from ``get_attention_state()`` (+ optionally
            ``append_exact_tokens`` for the in-flight chunk)
        slot_sigma_u / slot_sigma2 / slot_gamma_a / slot_gamma_b / slot_gamma:
            optional rank-1 second-order slot statistics. When present, score
            receives ``0.5 * scale**2 * sigma2 * (q·sigma_u)**2`` and value
            read-out receives ``scale * gamma * (q·gamma_a) * gamma_b``. Both
            corrections are multiplied by ``second_order_scale`` as a single
            coupled gate, so CPT can warm them up together.
        scale: attention scale (applied to the dot product only)
        mask: optional (T_q, S) bool. True = allowed. General-purpose escape
            hatch (tests); the model uses ``causal_tail`` instead. None (and
            causal_tail == 0) = attend all.
        lam: weight of the log-multiplicity mass bias (default 1.0)
        slot_M: distinct-anchor count per slot (see ``log_kv_position.
            dedup_anchors``). When given, the mass bias becomes
            ``lam*log(w) - log(M)`` (the ``-log(M)`` term is unconditional,
            not gated by ``lam``) so anchor-expanded entries don't leak
            ``M``-fold extra softmax mass. None reproduces the exact prior
            ``lam*log(w)`` formula.
        slot_valid: (B, G, S_pooled) bool, True = real anchor. Covers only
            the pooled-entry prefix of the slot axis (S_pooled <= S); the
            exact/in-flight suffix is never invalid by construction. Masked
            slots get score -inf, same as ``mask``/``causal_tail`` but
            orthogonal to both (can combine with either).
        causal_tail: if > 0, the LAST ``causal_tail`` slots are the in-flight
            chunk appended behind the frozen cache state (see
            ``append_exact_tokens``); they are masked causally against the
            queries (query i sees appended entry j iff j <= i; requires
            ``causal_tail == T_q``), while everything before them stays fully
            visible. Memory: this replaces a (T_q, S) bool mask + a
            masked_fill copy of the full fp32 score tensor with one in-place
            fill on the (T_q, causal_tail) tail slice — per streaming chunk,
            per layer. Mutually exclusive with ``mask``.

    Returns:
        (B, nh, T_q, v_dim)
    """
    B, nh, T_q, k_dim = q.shape
    v_dim = slot_v.size(-1)
    S = slot_k.size(2)
    if S == 0:
        return torch.zeros(B, nh, T_q, v_dim, device=q.device, dtype=q.dtype)

    has_rank1_stats = slot_sigma_u is not None
    if has_rank1_stats and any(x is None for x in (slot_sigma2, slot_gamma_a, slot_gamma_b, slot_gamma)):
        raise ValueError("log_kv_slot_attention() requires either all rank-1 stats or none")
    use_rank1_stats = has_rank1_stats and second_order_scale != 0.0

    if causal_tail:
        # Explicit raises (not asserts): survive `python -O`.
        if mask is not None:
            raise ValueError("causal_tail and mask are mutually exclusive")
        if causal_tail != T_q:
            raise ValueError(
                f"causal_tail ({causal_tail}) must equal T_q ({T_q}): the tail entries "
                "are the appended in-flight chunk, aligned one-to-one with the queries"
            )

    if second_order_scale == 0.0 and mask is None:
        out = _slot_flash_attention(
            q, slot_k, slot_v, slot_w, scale, lam, causal_tail, slot_M, slot_valid, check_valid,
        )
        if out is not None:
            return out

    if causal_tail:
        # (T_q, T_q) bool, True above the diagonal = blocked. Tiny (chunk-sized,
        # not S-sized) and the only allocation masking costs on this path.
        tail_blocked = torch.ones(T_q, causal_tail, dtype=torch.bool, device=q.device).triu_(1)

    # Scores in fp32: the log-w bias shifts logits by up to ~log(N) and bf16
    # resolution degrades with magnitude; fp32 keeps cross-level logit
    # differences intact. S is O(log N), so the fp32 buffer is small.
    #
    # In-place ops (mul_/add_/masked_fill_) are deliberate: none of these
    # intermediates is a saved tensor for backward (matmul saves its operands,
    # softmax saves its output, the scalar/bias ops save nothing), so mutating
    # the score buffer is autograd-safe and avoids three full-size fp32
    # temporaries per call — once per streaming chunk, per layer.
    nkv = slot_k.size(1)
    if nh != nkv:
        # --- GQA without materializing an rf× copy of the slots ---
        # repeat_interleave(rf, dim=1) on slot_k/v/w would allocate a fresh
        # (B, nh, S, ·) copy AND, in training, save that expanded tensor for
        # backward — rf× the slot memory for every one of the O(T) streaming
        # chunks, which is what runs the GPU out of memory. Instead fold the rf
        # query heads that share a KV group into their own axis and let matmul
        # broadcast the (B, nkv, 1, …) group across them: the K/V operands stay
        # (B, nkv, S, ·), nothing is duplicated, and the tensor autograd saves is
        # rf× smaller. Mathematically identical to the repeat_interleave path
        # (query head h ↔ KV group h // rf, matching model.py's GQA convention);
        # any difference is kernel-level FP reduction order, ULP-scale.
        rf = nh // nkv
        qg = q.reshape(B, nkv, rf, T_q, k_dim)               # (B, nkv, rf, T_q, k_dim)
        scores = torch.matmul(qg, slot_k.unsqueeze(2).mT)    # (B, nkv, rf, T_q, S)
        scores = scores.to(torch.float32)
        scores.mul_(scale)
        if use_rank1_stats:
            sigma_dot = torch.matmul(qg, slot_sigma_u.unsqueeze(2).mT).to(torch.float32)
            scores.add_(
                second_order_scale * 0.5 * scale * scale
                * sigma_dot.square()
                * slot_sigma2.to(torch.float32)[:, :, None, None, :]
            )
        mass_bias = _slot_mass_bias(slot_w, slot_M, lam)
        if mass_bias is not None:
            scores.add_(mass_bias[:, :, None, None, :])
        if mask is not None:
            scores.masked_fill_(~mask.view(1, 1, 1, T_q, S), float("-inf"))
        elif causal_tail:
            scores[..., S - causal_tail:].masked_fill_(tail_blocked, float("-inf"))
        if slot_valid is not None:
            s_pooled = slot_valid.size(-1)
            scores[..., :s_pooled].masked_fill_(
                (~slot_valid)[:, :, None, None, :], float("-inf")
            )
            # Avoid materializing an S-sized bool tensor; masked rows have
            # max=-inf, valid rows have a finite max.
            if check_valid and not torch.isfinite(scores.amax(dim=-1)).all():
                raise ValueError("log_kv_slot_attention(): a query row has no valid slot")
        attn = torch.softmax(scores, dim=-1).to(q.dtype)     # (B, nkv, rf, T_q, S)
        out = torch.matmul(attn, slot_v.unsqueeze(2))        # (B, nkv, rf, T_q, v_dim)
        if use_rank1_stats:
            # Activation dtype, matching the ``attn @ slot_v`` matmul above —
            # see the MHA branch for why fp32 here bought nothing but cost the
            # tensor cores.
            gamma_dot = torch.matmul(qg, slot_gamma_a.unsqueeze(2).mT)
            gamma_weight = (attn * gamma_dot * slot_gamma[:, :, None, None, :]).to(slot_gamma_b.dtype)
            corr = torch.matmul(gamma_weight, slot_gamma_b.unsqueeze(2))
            out = out + corr * (second_order_scale * scale)
        return out.reshape(B, nh, T_q, v_dim)

    # MHA (nh == nkv): one logit per slot, no head expansion needed.
    scores = torch.matmul(q, slot_k.mT).to(torch.float32)  # (B, nh, T_q, S)
    scores.mul_(scale)
    if use_rank1_stats:
        sigma_dot = torch.matmul(q, slot_sigma_u.mT).to(torch.float32)
        scores.add_(
            second_order_scale * 0.5 * scale * scale
            * sigma_dot.square()
            * slot_sigma2.to(torch.float32).unsqueeze(-2)
        )
    mass_bias = _slot_mass_bias(slot_w, slot_M, lam)
    if mass_bias is not None:
        scores.add_(mass_bias.unsqueeze(-2))
    if mask is not None:
        scores.masked_fill_(~mask.view(1, 1, T_q, S), float("-inf"))
    elif causal_tail:
        scores[..., S - causal_tail:].masked_fill_(tail_blocked, float("-inf"))
    if slot_valid is not None:
        s_pooled = slot_valid.size(-1)
        scores[..., :s_pooled].masked_fill_((~slot_valid).unsqueeze(-2), float("-inf"))
        # Same check as isfinite(scores).any(dim=-1), without an S-sized bool tensor.
        if check_valid and not torch.isfinite(scores.amax(dim=-1)).all():
            raise ValueError("log_kv_slot_attention(): a query row has no valid slot")
    attn = torch.softmax(scores, dim=-1).to(q.dtype)  # (B, nh, T_q, S)
    out = torch.matmul(attn, slot_v)                  # (B, nh, T_q, v_dim)
    if use_rank1_stats:
        # Value read-out correction in the activation dtype, matching the
        # ``attn @ slot_v`` matmul right above it. Promoting this one to fp32
        # bought no accuracy — its largest factor, ``attn``, has already been
        # rounded to the activation dtype, and matmul accumulates in fp32
        # regardless — while running a same-shaped matmul off the tensor cores.
        gamma_dot = torch.matmul(q, slot_gamma_a.mT)
        gamma_weight = (attn * gamma_dot * slot_gamma.unsqueeze(-2)).to(slot_gamma_b.dtype)
        corr = torch.matmul(gamma_weight, slot_gamma_b)
        out = out + corr * (second_order_scale * scale)
    return out


# ======================================================================
# Low-memory training autograd: stream without a graph, replay in backward
# ======================================================================


def _mid_flash_supported(cache, q, k, v, scale):
    if (not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16)
            or k.dtype != q.dtype or v.dtype != q.dtype
            or not math.isfinite(scale) or not 1e-3 <= scale <= 1.0
            or q.size(-1) != k.size(-1) or q.size(1) % k.size(1)
            or max(q.size(-1) + 1, v.size(-1)) > 192
            or cache.slot_k_dim != cache.k_dim or cache.cos_cache.size(1) != cache.rope_n_elem
            or cache.cos_cache.dtype not in (torch.float32, q.dtype)
            or cache.sin_cache.dtype != cache.cos_cache.dtype):
        return False
    dim = ((max(q.size(-1) + 1, v.size(-1)) + 7) // 8) * 8
    key = (q.device, q.dtype, q.size(0), q.size(1), k.size(1), dim)
    saved = getattr(cache, "_mid_flash_support", None)
    if saved is None or saved[0] != key:
        # Probe backend eligibility once per layout, before materializing a prefix.
        qa = q.new_empty(q.size(0), q.size(1), 1, dim)
        ka = k.new_empty(k.size(0), k.size(1), 1, dim)
        supported = can_use_flash_attention(SDPAParams(qa, ka, ka, None, 0.0, False, q.size(1) != k.size(1)))
        cache._mid_flash_support = (key, supported)
    return cache._mid_flash_support[1]


@logkv_timed("attn_fwd")
def _packed_flash_attention(q, k, v, scale, causal_tail, v_dim):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=causal_lower_right(q.size(2), k.size(2)) if causal_tail else None,
            dropout_p=0.0, scale=scale, enable_gqa=q.size(1) != k.size(1),
        )
    return out[..., :v_dim]


def _rope_cache_key(tensor):
    # Inference-mode tensors have no version counter; replacement still changes id.
    try:
        version = tensor._version
    except RuntimeError:
        version = None
    return id(tensor), version


def _mid_chunk_attention(cache, q, k, v, scale, causal_tail, plan, reuse_prefix):
    if not _mid_flash_supported(cache, q, k, v, scale):
        return None
    from models._vendor.semanticlogkv.litgpt.log_kv_pack import pack_mid_kv

    dim = ((max(q.size(-1) + 1, v.size(-1)) + 7) // 8) * 8
    buffers, skip_pooled, recent_start = None, False, 0
    # A mutable workspace is safe only for immediate inference consumption.
    reuse = reuse_prefix and not torch.is_grad_enabled() and q.size(2) <= 2 and not k.requires_grad and not v.requires_grad
    if reuse:
        key = (k.device, k.dtype, dim, cache.semantic_pack_backend,
               _rope_cache_key(cache.cos_cache), _rope_cache_key(cache.sin_cache))
        saved = cache._mid_decode_state
        if saved is not None and saved[0] == key:
            _, plan, buffers, recent_start = saved
            skip_pooled = True
        else:
            plan = cache._semantic_attention_plan() if plan is None else plan
    else:
        plan = cache._semantic_attention_plan() if plan is None else plan
    with logkv_timed("pack"):
        if reuse and buffers is None:
            capacity = plan[0].size(-1) + cache.alpha_count + cache.recent_size + max(2, k.size(2))
            buffers = (k.new_empty(*k.shape[:2], capacity, dim), v.new_empty(*v.shape[:2], capacity, dim))
        ka, va = pack_mid_kv(cache, plan, k, v, dim, buffers=buffers, skip_pooled=skip_pooled, recent_start=recent_start)
        if reuse:
            cache._mid_decode_state = (key, plan, buffers, cache.recent_count)
        qa = F.pad(q, (0, dim - q.size(-1)))
        qa[..., q.size(-1)] = 1.0 / scale
    return _packed_flash_attention(qa, ka, va, scale, causal_tail, v.size(-1))


def log_kv_chunk_attention(
    cache: LogStructuredKVCache,
    q_b: torch.Tensor,   # (B, nh, t, k_dim) current-chunk queries, post-RoPE
    k_b: torch.Tensor,   # (B, G, t, k_dim) current-chunk keys, post-RoPE
    v_b: torch.Tensor,   # (B, G, t, v_dim)
    scale: float,
    second_order_scale: float = 1.0,
    attention_plan=None,
    *,
    causal_tail: int | None = None,
    reuse_prefix: bool = False,
) -> torch.Tensor:
    """One streaming attention step, WITHOUT committing the chunk.

    [frozen cache state (fully visible)] + [current chunk (causal)]. This is
    the shared building block of ``LogKVStreamTrainingAttention``: its forward
    stream and its backward replay must both compute chunks through this exact
    function so the recomputed graphs match the streamed outputs.

    ``second_order_scale == 0`` skips the rank-1 statistics end to end — they are
    not gathered, not concatenated, and not multiplied into the scores — so a
    zero gate costs exactly the first-order path rather than the full
    second-order path multiplied by zero.
    """
    causal_tail = q_b.size(2) if causal_tail is None else causal_tail
    if causal_tail and causal_tail != q_b.size(2):
        raise ValueError("causal_tail must equal query length or be zero")
    if second_order_scale == 0.0 and cache.semantic_clusters and cache.semantic_anchor_mode == "mid":
        out = _mid_chunk_attention(cache, q_b, k_b, v_b, scale, causal_tail, attention_plan, reuse_prefix)
        if out is not None:
            return out
    if second_order_scale == 0.0:
        state = append_exact_tokens(cache.get_attention_state(with_stats=False, plan=attention_plan), k_b, v_b)
        return log_kv_slot_attention(
            q_b, state.slot_k, state.slot_v, state.slot_w,
            scale=scale,
            causal_tail=causal_tail,
            slot_M=state.M_s,
            slot_valid=state.slot_valid,
            # causal_tail=q_b.size(2) > 0 appends the real current chunk as an
            # exact w=1 suffix outside slot_valid's pooled-prefix range, so
            # every query row keeps its own (diagonal) token as a valid slot
            # by construction -- the row-has-no-valid-slot case this guards
            # against cannot happen here. Skips a per-call host sync that
            # otherwise fires on every chunk, every layer, every step.
            check_valid=False,
            second_order_scale=0.0,
        )
    state = append_exact_tokens(cache.get_attention_state(with_stats=True, plan=attention_plan), k_b, v_b)
    return log_kv_slot_attention(
        q_b, state.slot_k, state.slot_v, state.slot_w,
        scale=scale,
        causal_tail=causal_tail,
        slot_M=state.M_s,
        slot_valid=state.slot_valid,
        check_valid=False,  # see check_valid=False comment above
        slot_sigma_u=state.slot_sigma_u,
        slot_sigma2=state.slot_sigma2,
        slot_gamma_a=state.slot_gamma_a,
        slot_gamma_b=state.slot_gamma_b,
        slot_gamma=state.slot_gamma,
        second_order_scale=second_order_scale,
    )


class LogKVStreamTrainingAttention(torch.autograd.Function):
    """Constant-in-T-times-S memory autograd for logKV streaming training.

    Problem: the naive training graph saves, for every 2-token chunk, the
    concatenated [detached prefix + chunk] keys/values (O(S) each) that its
    two matmuls need for backward. Per layer that is O(T/2 * S) saved tensors
    — hundreds of GB for a 32K stream at B=512/recent=1024 — the long-context
    training OOM (the failing allocation is merely whichever op runs next
    once the accumulated graph has eaten the device).

    Forward here streams the whole sequence with NO recorded graph and returns
    the attention output. Backward resets the cache and REPLAYS the same stream:
    each block's attention is recomputed with a throwaway local graph that
    ``torch.autograd.grad`` consumes immediately, so at most one block graph is
    alive at any time. Peak memory: O(T + train_block*S) instead of O(T*S).
    Semantic replay additionally saves O(T/train_block*S) compact integer/bool
    anchor metadata, avoiding repeated sorting and device-to-host counts. It
    releases plans after backward. Update replay additionally retains detached
    write payloads and state until this graph's backward completes.
    When a Block checkpoint has LogKV routing contexts installed, its recompute
    reuses the original op-log instead of calculating routing decisions again.
    Both replay passes share CPU-only parsed schedules owned by this graph;
    neither old K/V nor device addresses are retained in these schedules.
    Cost: one extra streaming pass plus the per-block backwards.

    Correctness:
      - For a fixed train_block, gradients are EXACT for that block-streaming
        objective: in the naive graph, gradient reaches q/k/v only through each
        token's own block (the cache commits detached copies), so per-block
        grads are complete and disjoint, and the replayed per-block
        ``autograd.grad`` reproduces them one-to-one.
      - Routing and exact-span choices are recorded. Beta also records its
        content-dependent compaction cuts; replay applies these same weighted
        merges without reselecting, rebuilding the forward prefix states.
      - Backward does not depend on any cache state forward left behind: it
        resets the buffers AND re-derives ``cache.second_order`` from its own
        ``second_order_scale`` before replaying, so neither interleaved
        forward/backward across micro-batches, nor activation-checkpoint
        recompute ordering, nor another layer running at a different gate can
        make the replay rebuild a cache that forward never attended over.

    ``train_block=2`` is the strict 2-token streaming reference. Larger blocks
    freeze the prefix state at block start and use causal exact attention within
    the block, matching the inference prefill speed/memory tradeoff while
    preserving the exact final cache state. First-order autograd only
    (``once_differentiable``) — CPT never needs grad-of-grad.
    """

    @staticmethod
    def forward(ctx, q, k, v, cache, scale, train_block, second_order_scale, k_raw=None, span_ends=None):
        commit_k_raw = k if k_raw is None else k_raw
        ctx.input_count = len(ctx.needs_input_grad)
        ctx.span_ends = None if span_ends is None else tuple(map(tuple, span_ends))
        if cache.alpha_exact_tokens and (not cache.semantic_replay_updates or second_order_scale != 0.):
            raise ValueError("AlphaLogKV training requires semantic_replay_updates and second_order_scale=0")

        T = q.size(2)
        train_block = int(train_block)
        if train_block < 2:
            raise ValueError(f"logKV train_block must be >= 2, got {train_block}")
        if train_block > cache.recent_size:
            raise ValueError(
                f"logKV train_block ({train_block}) must be <= recent_size ({cache.recent_size})"
            )
        signature = (
            q.shape, k.shape, v.shape, q.dtype, k.dtype, v.dtype, q.device,
            scale, train_block, second_order_scale, k_raw is not None,
            cache.semantic_anchor_mode, cache.K_max, cache.B, cache.recent_size,
            cache.semantic_centroid_backend, cache.semantic_replay_updates,
            cache.semantic_unified_route, cache.semantic_merge_passes,
            cache.alpha_exact_tokens, cache.alpha_span_max_tokens, ctx.span_ends,
            cache.beta_novelty, cache.beta_adaptive_merge,
        )
        checkpoint_log = checkpoint_route_replay(cache, signature)
        updates = (_SemanticReplayUpdates() if checkpoint_log is None else checkpoint_log[4]) if cache.semantic_replay_updates else None
        outputs: list[torch.Tensor] = []
        attention_plans = []
        # Explicit no_grad: the memory guarantee of this whole scheme rests on
        # this pass recording nothing (Function.forward already runs detached;
        # this makes the invariant local and future-proof).
        with torch.no_grad():
            if updates is not None:
                updates.wait(q.device)
            cache.reset_parameters()
            needs_op_log = cache.semantic_clusters and cache.K_max > 1
            if needs_op_log and checkpoint_log is None:
                cache.begin_op_log()
            # Own the flag rather than trusting the caller, and set it AFTER
            # the reset above so the cache is always empty when this runs (the
            # guarded setter would otherwise raise on a cache left non-empty by
            # a previous call at a different gate). It decides whether
            # add_recent() builds the second-order statistics, so leaving it to
            # ambient state breaks this Function two ways: a caller that never
            # sets it runs the attention math against statistics that were
            # never built (silently first-order), and anything that flips it
            # between forward and backward makes the replay rebuild a
            # different cache than the one forward attended over, so the
            # gradients belong to a different function than the output.
            # Deriving it here from the gate keeps the pair consistent no
            # matter who calls.
            cache.second_order = second_order_scale != 0.0
            start = 0
            while start < T:  # mirrored in backward() — keep in sync
                end = min(start + train_block, T)
                plan = None
                if cache.semantic_clusters:
                    plan = cache._semantic_attention_plan()
                attention_plans.append(plan)
                outputs.append(
                    log_kv_chunk_attention(
                        cache,
                        q[:, :, start:end], k[:, :, start:end], v[:, :, start:end],
                        scale,
                        second_order_scale,
                        attention_plan=plan,
                    )
                )
                with cache._semantic_update_context(updates, replay=checkpoint_log is not None):
                    cache.add_recent(
                        k[:, :, start:end],
                        v[:, :, start:end],
                        k_raw=commit_k_raw[:, :, start:end] if cache.semantic_clusters else None,
                        record_op_log=needs_op_log and checkpoint_log is None,
                        replay_op_log=checkpoint_log[0] if checkpoint_log is not None else None,
                        replay_op_log_len=checkpoint_log[1] if checkpoint_log is not None else None,
                        replay_op_log_host=checkpoint_log[2] if checkpoint_log is not None else None,
                        replay_plans=checkpoint_log[3] if checkpoint_log is not None else None,
                        span_ends=[row[start:end] for row in ctx.span_ends] if ctx.span_ends is not None else None,
                    )
                start = end
        if checkpoint_log is None:
            op_log, op_log_len = cache.take_op_log() if needs_op_log else (None, None)
            op_log_host = getattr(cache, "_last_op_log_host", None) if needs_op_log else None
            replay_plans = _SemanticReplayPlans(op_log_host) if needs_op_log else None
            if replay_plans is not None:
                op_log_host = replay_plans.host_log
            record = (op_log, op_log_len, op_log_host, replay_plans)
            checkpoint_record_routes(cache, signature, record + ((updates,) if updates is not None else ()))
        else:
            op_log, op_log_len, op_log_host, replay_plans = checkpoint_log[:4]
        # Use saved tensors so autograd releases plan storage after backward
        # (and still supports retain_graph), just like the saved Q/K/V.
        plan_tensors = [tensor for plan in attention_plans if plan is not None for tensor in plan]
        ctx.update_tensor_offset = 3 + int(k_raw is not None) + len(plan_tensors)
        ctx.update_flushes = updates.flushes if updates is not None else None
        ctx.update_config = (cache.semantic_centroid_backend,
                             cache.semantic_replay_updates, cache.semantic_unified_route, cache.semantic_merge_passes,
                             cache.alpha_exact_tokens, cache.alpha_span_max_tokens,
                             cache.beta_novelty, cache.beta_adaptive_merge)
        ctx.save_for_backward(q, k, v, *([k_raw] if k_raw is not None else []), *plan_tensors,
                              *(updates.tensors if updates is not None else ()))
        ctx.has_attention_plans = cache.semantic_clusters
        ctx.cache = cache
        ctx.scale = scale
        ctx.train_block = train_block
        ctx.second_order_scale = second_order_scale
        ctx.semantic_anchor_mode = cache.semantic_anchor_mode
        ctx.has_k_raw = k_raw is not None
        ctx.needs_op_log = needs_op_log
        ctx.op_log = op_log
        ctx.op_log_len = op_log_len
        ctx.op_log_host = op_log_host
        ctx.replay_plans = replay_plans
        return torch.cat(outputs, dim=2)  # (B, nh, T, v_dim)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y):
        saved = ctx.saved_tensors
        if ctx.has_k_raw:
            q, k, v, k_raw = saved[:4]
        else:
            q, k, v = saved[:3]
            k_raw = k
        cache = ctx.cache
        if ctx.update_config != (cache.semantic_centroid_backend,
                                 cache.semantic_replay_updates, cache.semantic_unified_route, cache.semantic_merge_passes,
                                 cache.alpha_exact_tokens, cache.alpha_span_max_tokens,
                                 cache.beta_novelty, cache.beta_adaptive_merge):
            raise RuntimeError("semantic update configuration changed between forward and backward")
        updates = (_SemanticReplayUpdates(ctx.update_flushes, saved[ctx.update_tensor_offset:])
                   if ctx.update_flushes is not None else None)
        if cache.semantic_anchor_mode != ctx.semantic_anchor_mode:
            raise RuntimeError("semantic_anchor_mode changed between forward and backward")
        scale = ctx.scale
        train_block = ctx.train_block
        second_order_scale = ctx.second_order_scale
        needs_op_log = ctx.needs_op_log
        T = q.size(2)
        # Blocks partition [0, T) and each position's grad comes from exactly
        # its own block, so the empty buffers are fully overwritten.
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        if updates is not None:
            updates.wait(q.device)
        cache.reset_parameters()
        # Re-derive AFTER the reset, same reasoning and ordering as forward():
        # the replay must rebuild the cache exactly as forward built it, the
        # flag is mutable state anyone could have changed in between, and the
        # guarded setter needs the cache empty to accept either value.
        cache.second_order = second_order_scale != 0.0
        start = 0
        while start < T:  # mirrors forward() — keep in sync
            end = min(start + train_block, T)
            q_b = q[:, :, start:end].detach().requires_grad_(True)
            k_b = k[:, :, start:end].detach().requires_grad_(True)
            v_b = v[:, :, start:end].detach().requires_grad_(True)
            plan_offset = (4 if ctx.has_k_raw else 3) + 4 * (start // train_block)
            with torch.enable_grad():
                y_b = log_kv_chunk_attention(
                    cache, q_b, k_b, v_b, scale, second_order_scale,
                    attention_plan=saved[plan_offset:plan_offset + 4] if ctx.has_attention_plans else None,
                )
            with logkv_timed("attn_bwd"):
                g_q, g_k, g_v = torch.autograd.grad(y_b, (q_b, k_b, v_b), grad_y[:, :, start:end])
            dq[:, :, start:end] = g_q
            dk[:, :, start:end] = g_k
            dv[:, :, start:end] = g_v
            with torch.no_grad(), cache._semantic_update_context(updates, replay=True):
                cache.add_recent(
                    k[:, :, start:end],
                    v[:, :, start:end],
                    k_raw=k_raw[:, :, start:end] if cache.semantic_clusters else None,
                    replay_op_log=ctx.op_log if needs_op_log else None,
                    replay_op_log_len=ctx.op_log_len if needs_op_log else None,
                    replay_op_log_host=ctx.op_log_host if needs_op_log else None,
                    replay_plans=ctx.replay_plans if needs_op_log else None,
                    span_ends=[row[start:end] for row in ctx.span_ends] if ctx.span_ends is not None else None,
                )
            start = end
        return (dq, dk, dv) + (None,) * (ctx.input_count - 3)
