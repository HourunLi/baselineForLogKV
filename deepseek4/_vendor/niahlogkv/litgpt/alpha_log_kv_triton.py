"""Fused Alpha span scores: one program per candidate span, every KV group.

A program reads its span's K and V rows once per group and returns the score
`alpha_log_kv.score_spans` defines, including Beta novelty against the
pre-flush centers. Sums run in float32 in token-tile order, so scores equal the
PyTorch reference up to rounding, not bitwise.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _moments(X, base, tok, live, d, DIM: tl.constexpr):
    """Sum of a token tile's rows and the sum of their mean squares."""
    x = tl.load(X + base + tok[:, None] + d[None, :], live[:, None] & (d < DIM)[None, :], 0.0).to(tl.float32)
    return tl.sum(x, axis=0), tl.sum(tl.div_rn(tl.sum(x * x, axis=1), DIM), axis=0)


@triton.jit
def _variance_ratio(total, energy, n, DIM: tl.constexpr):
    """Mean vector and the part of a span's energy its mean does not explain."""
    mean = tl.div_rn(total, n)
    energy = tl.div_rn(energy, n)
    mean_square = tl.div_rn(tl.sum(mean * mean, axis=0), DIM)
    return mean, mean_square, energy, tl.div_rn(tl.maximum(energy - mean_square, 0.0), tl.maximum(energy, 1e-12))


@triton.jit(do_not_specialize=["S", "G"])
def _span_scores(K, V, C, VALID, TOK, META, OUT, S, G,
                 SKB, SKG, SKT, SVB, SVG, SVT, SCB, SCG, SCK, SAB, SAG,
                 NC: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                 BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr, BT: tl.constexpr, BETA: tl.constexpr):
    s = tl.program_id(0).to(tl.int64)
    start = tl.load(META + s).to(tl.int64)
    n = tl.load(META + S + s).to(tl.int64)
    b = tl.load(META + 2 * S + s).to(tl.int64)
    count = n.to(tl.float32)
    dk = tl.arange(0, BK)
    dv = tl.arange(0, BV)
    top1 = tl.full([], -float("inf"), tl.float32)
    top2 = tl.full([], -float("inf"), tl.float32)
    total_k = tl.zeros([], tl.float32)
    total_v = tl.zeros([], tl.float32)
    for g in range(G):
        sum_k = tl.zeros([BK], tl.float32)
        sum_v = tl.zeros([BV], tl.float32)
        energy_k = tl.zeros([], tl.float32)
        energy_v = tl.zeros([], tl.float32)
        for t0 in range(0, n, BT):
            t = t0 + tl.arange(0, BT)
            live = t < n
            tok = tl.load(TOK + start + t, live, 0).to(tl.int64)
            a, e = _moments(K, b * SKB + g * SKG, tok * SKT, live, dk, DK)
            sum_k += a
            energy_k += e
            a, e = _moments(V, b * SVB + g * SVG, tok * SVT, live, dv, DV)
            sum_v += a
            energy_v += e
        mean_k, square_k, energy_k, part_k = _variance_ratio(sum_k, energy_k, count, DK)
        _, _, _, part_v = _variance_ratio(sum_v, energy_v, count, DV)
        if BETA:
            c = tl.arange(0, BC)
            centers = tl.load(C + b * SCB + g * SCG + c[:, None] * SCK + dk[None, :],
                              (c < NC)[:, None] & (dk < DK)[None, :], 0.0)
            valid = (tl.load(VALID + b * SAB + g * SAG + c, c < NC, 0) != 0) & (c < NC)
            dots = tl.div_rn(tl.sum(centers * mean_k[None, :], axis=1), DK)
            distance = tl.maximum(square_k + tl.div_rn(tl.sum(centers * centers, axis=1), DK) - 2.0 * dots, 0.0)
            nearest = tl.min(tl.where(valid, distance, float("inf")), axis=0)
            nearest = tl.where(tl.sum(valid.to(tl.int32), axis=0) > 0, nearest, 0.0)
            novelty = tl.div_rn(nearest, tl.maximum(energy_k, 1e-12))
            score = (part_k + part_v) * 0.5 + tl.div_rn(novelty, 1.0 + novelty)
            top2 = tl.maximum(top2, tl.minimum(top1, score))
            top1 = tl.maximum(top1, score)
        else:
            total_k += part_k
            total_v += part_v
    if BETA:
        out = tl.where(G > 1, (top1 + top2) * 0.5, top1)
    else:
        out = tl.div_rn(total_k, G.to(tl.float32)) + tl.div_rn(total_v, G.to(tl.float32))
    tl.store(OUT + s, out)


def span_scores(k, v, tokens, meta, centroids=None, valid=None):
    """Scores (float32 [S]) of S spans over [rows, groups, tokens, dim] K/V.

    `tokens` lists the spans' token indices back to back; `meta` [3, S] holds
    each span's start in `tokens`, its length (>= 1) and its row. With
    `centroids` [rows, groups, K, dim] and `valid` [rows, groups, K], the score
    is Beta's: the top-two group mean of half the variance ratio plus
    novelty / (1 + novelty).
    """
    spans = meta.size(1)
    out = torch.empty(spans, dtype=torch.float32, device=k.device)
    if not spans:
        return out
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    beta = centroids is not None
    if beta:
        centroids = centroids.detach().float().contiguous()
        valid = valid.contiguous()
    else:
        centroids, valid = k, tokens  # Unused pointers.
    n_clusters = centroids.size(2) if beta else 1
    _span_scores[(spans,)](
        k.detach(), v.detach(), centroids, valid, tokens, meta.contiguous(), out, spans, k.size(1),
        k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
        centroids.stride(0) if beta else 0, centroids.stride(1) if beta else 0, centroids.stride(2) if beta else 0,
        valid.stride(0) if beta else 0, valid.stride(1) if beta else 0,
        n_clusters, k.size(-1), v.size(-1),
        triton.next_power_of_2(k.size(-1)), triton.next_power_of_2(v.size(-1)),
        triton.next_power_of_2(n_clusters), 16, beta,
        num_warps=4, enable_fp_fusion=False,
    )
    return out
