# Vendored for HybridCSA; local namespace changes are recorded in SOURCE.json.
"""First-order ladder updates and ordered centroid updates on the current stream.

Merge reads finish before survivor writes launch: an in-place cross-CTA merge
would race with readers of the old slots. No atomics or host readbacks are used.
"""

import torch
import triton
import triton.language as tl


_FIELDS = (0, 1, 2, 8, 9, 10, 11, 12)


@triton.jit
def _read(F, S, SI, GI, index, stored_rows, d, mask, DIM: tl.constexpr):
    stored = index < stored_rows
    fi = tl.load(SI + index, mask & stored, 0).to(tl.int64)
    si = tl.load(GI + index - stored_rows, mask & ~stored, 0).to(tl.int64)
    a = tl.load(F + fi[:, None] * DIM + d[None, :], mask[:, None] & stored[:, None] & (d < DIM)[None, :], 0)
    b = tl.load(S + si[:, None] * DIM + d[None, :], mask[:, None] & ~stored[:, None] & (d < DIM)[None, :], 0)
    # Reference pool construction casts staging rows to the storage dtype.
    return tl.where(stored[:, None], a, b.to(F.dtype.element_ty))


@triton.jit(do_not_specialize=["NS", "NP", "NV"])
def _merge(F, S, O, SI, GI, PI, VI, NS, NP, NV,
           DK: tl.constexpr, DV: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    d = tl.arange(0, BD).to(tl.int64)
    live, pair = r < NP + NV, r < NP
    a_pair = tl.load(PI + 2 * r, pair, 0).to(tl.int64)
    a_copy = tl.load(VI + r - NP, live & ~pair, 0).to(tl.int64)
    ai = tl.where(pair, a_pair, a_copy)
    bi = tl.load(PI + 2 * r + 1, pair, 0).to(tl.int64)
    zero = tl.arange(0, 1).to(tl.int64)
    wa = _read(F[2], S[2], SI, GI, ai, NS, zero, live, 1).to(tl.float32)
    wb = _read(F[2], S[2], SI, GI, bi, NS, zero, pair, 1).to(tl.float32)
    alpha = tl.div_rn(wa, tl.maximum(wa + wb, 1.e-8))
    for f in tl.static_range(8):
        dim = DK if f == 0 else DV if f == 1 else 1
        a = _read(F[f], S[f], SI, GI, ai, NS, d, live, dim)
        b = _read(F[f], S[f], SI, GI, bi, NS, d, pair, dim)
        if f < 2:
            value = alpha * a.to(tl.float32) + (1.0 - alpha) * b.to(tl.float32)
        elif f == 2 or f == 5:
            value = a + b
        elif f == 3:
            value = tl.where((wa > 0) & (wb > 0), tl.minimum(a, b), tl.where(wa > 0, a, b))
        elif f == 4:
            value = tl.where((wa > 0) & (wb > 0), tl.maximum(a, b), tl.where(wa > 0, a, b))
        elif f == 6:
            value = tl.minimum(a, b)
        else:
            value = a & b
        value = tl.where(pair[:, None], value, a)
        tl.store(O[f] + r[:, None] * dim + d[None, :], value, live[:, None] & (d < dim)[None, :])


@triton.jit(do_not_specialize=["N", "NC", "OFFSET"])
def _scatter(F, S, SRC, DST, CLEAR, N, NC, OFFSET, DIRECT: tl.constexpr,
             DK: tl.constexpr, DV: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    d = tl.arange(0, BD).to(tl.int64)
    copy = r < N
    clear = (r >= N) & (r < N + NC)
    if DIRECT:
        src = OFFSET + r
    else:
        src = tl.load(SRC + r, copy, 0).to(tl.int64)
    dst = tl.where(copy, tl.load(DST + r, copy, 0), tl.load(CLEAR + r - N, clear, 0)).to(tl.int64)
    for f in tl.static_range(8):
        dim = DK if f == 0 else DV if f == 1 else 1
        value = tl.load(S[f] + src[:, None] * dim + d[None, :], copy[:, None] & (d < dim)[None, :], 0)
        tl.store(F[f] + dst[:, None] * dim + d[None, :], value,
                 (copy | clear)[:, None] & (d < dim)[None, :])


def fill(fields, stage, src, dst):
    if dst.numel():
        f, s = tuple(fields[i] for i in _FIELDS), tuple(stage[i] for i in _FIELDS)
        _scatter[(triton.cdiv(dst.numel(), 8),)](
            f, s, src, dst, dst, dst.numel(), 0, 0, False,
            fields[0].size(1), fields[1].size(1), 8,
            triton.next_power_of_2(max(fields[0].size(1), fields[1].size(1))),
            num_warps=4, enable_fp_fusion=False,
        )


@triton.jit(do_not_specialize=["NS", "NP", "NV", "NG"])
def _beta_merge(F, S, O, SI, GI, PI, VI, LAYOUT, CUTS, NS, NP, NV, NG,
                DK: tl.constexpr, DV: tl.constexpr, BD: tl.constexpr, REPLAY: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BD).to(tl.int64)
    zero = tl.arange(0, 1).to(tl.int64)
    if group < NG:
        start = tl.load(LAYOUT + 3 * group)
        width = tl.load(LAYOUT + 3 * group + 1)
        dest = tl.load(LAYOUT + 3 * group + 2)
        row = tl.arange(0, 4).to(tl.int64)
        valid = row < width
        index = tl.load(PI + start + row, valid, 0).to(tl.int64)
        k = _read(F[0], S[0], SI, GI, index, NS, d, valid, DK).to(tl.float32)
        v = _read(F[1], S[1], SI, GI, index, NS, d, valid, DV).to(tl.float32)
        w = tl.reshape(_read(F[2], S[2], SI, GI, index, NS, zero, valid, 1), (4,)).to(tl.float32)
        if REPLAY:
            cut = tl.load(CUTS + group).to(tl.int32)
        else:
            cut = 1
            if width == 4:
                total = tl.maximum(tl.sum(w, 0), 1.e-12)
                ek = tl.maximum(tl.sum(tl.sum(k * k, 1) * w, 0) / total, 1.e-12)
                ev = tl.maximum(tl.sum(tl.sum(v * v, 1) * w, 0) / total, 1.e-12)
                best = float("inf")
                for option in tl.static_range(3):
                    candidate = 2 if option == 0 else 1 if option == 1 else 3
                    cost = 0.0
                    for side in tl.static_range(2):
                        member = (row < candidate) if side == 0 else (row >= candidate)
                        ww = tl.where(member & valid, w, 0.0)
                        mass = tl.maximum(tl.sum(ww, 0), 1.e-12)
                        km = tl.sum(k * ww[:, None], 0) / mass
                        vm = tl.sum(v * ww[:, None], 0) / mass
                        dk, dv = k - km[None, :], v - vm[None, :]
                        cost += tl.sum(tl.sum(dk * dk, 1) * ww, 0) / ek
                        cost += tl.sum(tl.sum(dv * dv, 1) * ww, 0) / ev
                    better = cost < best
                    cut = tl.where(better, candidate, cut)
                    best = tl.minimum(best, cost)
            tl.store(CUTS + group, cut)
        left = valid & ((width == 2) | (row < cut))
        right = valid & (width == 4) & (row >= cut)
        wl, wr = tl.where(left, w, 0.0), tl.where(right, w, 0.0)
        ml, mr = tl.sum(wl, 0), tl.sum(wr, 0)
        for side in tl.static_range(2):
            ww = wl if side == 0 else wr
            mass = ml if side == 0 else mr
            write = (side == 0) | (width == 4)
            ko = tl.sum(k * ww[:, None], 0) / tl.maximum(mass, 1.e-12)
            vo = tl.sum(v * ww[:, None], 0) / tl.maximum(mass, 1.e-12)
            tl.store(O[0] + (dest + side) * DK + d, ko, write & (d < DK))
            tl.store(O[1] + (dest + side) * DV + d, vo, write & (d < DV))
            tl.store(O[2] + dest + side, mass, write)
        # Read each metadata field once and reduce both output segments from
        # registers; preserve the same four-row reduction order as K/V.
        for f in tl.static_range(3, 8):
            x = tl.reshape(_read(F[f], S[f], SI, GI, index, NS, zero, valid, 1), (4,))
            for side in tl.static_range(2):
                member = left if side == 0 else right
                mass = ml if side == 0 else mr
                write = (side == 0) | (width == 4)
                if f == 3:
                    value = tl.min(tl.where(member & (w > 0), x, 9223372036854775807), 0)
                    value = tl.where(mass > 0, value, 0)
                elif f == 4:
                    value = tl.max(tl.where(member & (w > 0), x, -9223372036854775807), 0)
                    value = tl.where(mass > 0, value, 0)
                elif f == 5:
                    value = tl.sum(tl.where(member, x, 0), 0)
                elif f == 6:
                    value = tl.min(tl.where(member, x, 9223372036854775807), 0)
                else:
                    value = tl.min(tl.where(member, x.to(tl.int32), 1), 0)
                tl.store(O[f] + dest + side, value, write)
    else:
        copy_row = (group - NG) * 8 + tl.arange(0, 8).to(tl.int64)
        copy_live = copy_row < NV
        copy_index = tl.load(VI + copy_row, copy_live, 0).to(tl.int64)
        for field in tl.static_range(8):
            copy_dim = DK if field == 0 else DV if field == 1 else 1
            copy_value = _read(F[field], S[field], SI, GI, copy_index, NS, d, copy_live, copy_dim)
            tl.store(O[field] + (NP + copy_row[:, None]) * copy_dim + d[None, :], copy_value,
                     copy_live[:, None] & (d < copy_dim)[None, :])


def merge_scatter(fields, stage, si, gi, pi, vi, dst, clear, *, beta_lengths=None, beta_cuts=None,
                  beta_layout=None):
    f, s = tuple(fields[i] for i in _FIELDS), tuple(stage[i] for i in _FIELDS)
    pairs, survivors = pi.numel() // 2, vi.numel()
    n = pairs + survivors
    # Carry K/V remain fp32, as in compact(); each next-level load rounds to
    # storage dtype. Survivor scratch prevents writes racing with merge reads.
    out = tuple(torch.empty((n, *x.shape[1:]), device=x.device,
                            dtype=torch.float32 if i < 2 else x.dtype) for i, x in enumerate(f))
    dk, dv = fields[0].size(1), fields[1].size(1)
    bd = triton.next_power_of_2(max(dk, dv))
    if beta_lengths is None:
        _merge[(triton.cdiv(n, 8),)](f, s, out, si, gi, pi, vi, si.numel(), pairs, survivors,
                                   dk, dv, 8, bd, num_warps=4, enable_fp_fusion=False)
    else:
        from hybrid_csa._vendor.semanticlogkv.litgpt.beta_log_kv import make_layout

        if sum(beta_lengths) != pi.numel():
            raise ValueError("Beta lane lengths must cover the compaction input")
        # The cache can include this host-known layout in its existing batched
        # index upload instead of paying another pageable transfer per level.
        layout = make_layout(beta_lengths, pi.device) if beta_layout is None else beta_layout
        if (layout.ndim != 2 or layout.shape[1] != 3 or layout.dtype != torch.int64
                or layout.device != pi.device or not layout.is_contiguous()
                or len(layout) != sum((n + 3) // 4 for n in beta_lengths)):
            raise ValueError("Beta layout does not match the compaction lanes")
        replay = beta_cuts is not None
        if not replay:
            beta_cuts = torch.empty(len(layout), dtype=torch.uint8, device=pi.device)
        elif beta_cuts.shape != (len(layout),) or beta_cuts.dtype != torch.uint8 or beta_cuts.device != pi.device:
            raise ValueError("recorded Beta cuts do not match the compaction layout")
        if len(layout) or survivors:
            _beta_merge[(len(layout) + triton.cdiv(survivors, 8),)](
                f, s, out, si, gi, pi, vi, layout, beta_cuts, si.numel(), pairs, survivors, len(layout),
                dk, dv, bd, replay, num_warps=4, enable_fp_fusion=False)
    if survivors + clear.numel():
        _scatter[(triton.cdiv(survivors + clear.numel(), 8),)](
            f, out, vi, dst, clear, survivors, clear.numel(), pairs, True,
            dk, dv, 8, bd, num_warps=4, enable_fp_fusion=False,
        )
    carry = [None] * 13
    for i, x in zip(_FIELDS, out):
        carry[i] = x[:pairs]
    return tuple(carry) if beta_lengths is None else (tuple(carry), beta_cuts)


@triton.jit(do_not_specialize=["NR", "START"])
def _centroid(K, MU, NE, OUT_MU, OUT_NE, META, NR, START, D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr = 1):
    row = tl.program_id(0).to(tl.int64)
    r = START + row
    d = tl.arange(0, BD).to(tl.int64)
    cluster = tl.load(META + r).to(tl.int64)
    begin = tl.load(META + 2 * NR + r).to(tl.int64)
    length = tl.load(META + 3 * NR + r)
    forget = tl.load(META + 4 * NR + r).to(tl.int32).to(tl.float32, bitcast=True)
    acc = tl.full((BD,), 0., tl.float32)
    # BT=1 is the bit-exact reference. BT=32 parallelizes tokens as well as
    # features; fixed tiles avoid atomics and preserve reproducible ordering.
    if BT == 1:
        for i in range(length):
            acc = acc + tl.load(K + (begin + i) * D + d, d < D, 0).to(tl.float32)
    else:
        t = tl.arange(0, BT).to(tl.int64)
        for i in range(0, length, BT):
            values = tl.load(K + (begin + i + t[:, None]) * D + d[None, :],
                             (i + t[:, None] < length) & (d[None, :] < D), 0).to(tl.float32)
            acc = acc + tl.sum(values, axis=0)
    pre = tl.load(NE + cluster) * forget
    n = length.to(tl.float32)
    denom = pre + n
    mu = tl.load(MU + cluster * D + d, d < D, 0)
    value = tl.where(pre > 0, tl.div_rn(pre * mu + acc, denom), tl.div_rn(acc, n))
    # Input state is read-only throughout this launch. A separate launch
    # commits outputs, so no warp/CTA can observe a partially updated count.
    tl.store(OUT_MU + row * D + d, value, d < D)
    tl.store(OUT_NE + row, denom)


@triton.jit(do_not_specialize=["START", "COUNT"])
def _centroid_commit(MU, NE, OUT_MU, OUT_NE, META, START, COUNT,
                     D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    d = tl.arange(0, BD).to(tl.int64)
    live = row < COUNT
    cluster = tl.load(META + START + row, live, 0).to(tl.int64)
    mask = live[:, None] & (d < D)[None, :]
    value = tl.load(OUT_MU + row[:, None] * D + d[None, :], mask, 0)
    denom = tl.load(OUT_NE + row, live, 0)
    tl.store(MU + cluster[:, None] * D + d[None, :], value, mask)
    tl.store(NE + cluster, denom, live)


def centroid(k, mu, n_eff, metadata, start, count, *, token_tile=1):
    # Temporary storage is O(updated clusters * D), independent of token count.
    out_mu = mu.new_empty((count, k.size(1)))
    out_ne = n_eff.new_empty(count)
    bd = triton.next_power_of_2(k.size(1))
    _centroid[(count,)](k, mu, n_eff, out_mu, out_ne, metadata, metadata.size(1), start, k.size(1),
                        bd, token_tile, num_warps=4, enable_fp_fusion=False)
    _centroid_commit[(triton.cdiv(count, 8),)](
        mu, n_eff, out_mu, out_ne, metadata, start, count, k.size(1), bd, 8,
        num_warps=4, enable_fp_fusion=False,
    )


def centroid_parallel(k, mu, n_eff, metadata, start, count):
    centroid(k, mu, n_eff, metadata, start, count, token_tile=32)


@triton.jit(do_not_specialize=["N", "P", "A"])
def _alpha_partition(K, V, POS, INDEX, EK, EV, EP, VALID, AK, AV, AP, N, P, A,
                     G: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                     BT: tl.constexpr, BD: tl.constexpr):
    """Copy exact and archived rows together; masked rows are real zero writes."""
    b, g = tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    row = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    d = tl.arange(0, BD)
    live = row < P + A
    src = tl.load(INDEX + b * (P + A) + row, live, -1).to(tl.int64)
    token = live & (src >= 0)
    base = (b * G + g) * N + src
    k = tl.load(K + base[:, None] * DK + d[None, :], token[:, None] & (d < DK)[None, :], 0)
    v = tl.load(V + base[:, None] * DV + d[None, :], token[:, None] & (d < DV)[None, :], 0)
    keep, archive = live & (row < P), live & (row >= P)
    exact_row = (b * G + g) * P + row
    archive_row = (b * G + g) * A + row - P
    tl.store(EK + exact_row[:, None] * DK + d[None, :], k, keep[:, None] & (d < DK)[None, :])
    tl.store(EV + exact_row[:, None] * DV + d[None, :], v, keep[:, None] & (d < DV)[None, :])
    tl.store(AK + archive_row[:, None] * DK + d[None, :], k, archive[:, None] & (d < DK)[None, :])
    tl.store(AV + archive_row[:, None] * DV + d[None, :], v, archive[:, None] & (d < DV)[None, :])
    if g == 0:
        pos = tl.load(POS + b * N + src, token, 0)
        tl.store(EP + b * P + row, pos, keep)
        tl.store(VALID + b * P + row, token, keep)
        tl.store(AP + b * A + row - P, pos, archive)


def alpha_partition(k, v, pos, indices, exact):
    """Contiguous cat inputs are independent of the persistent exact outputs.

    Indices are [batch, exact capacity + archive width], padded with -1.
    No reduction or scoring occurs here: every retained value is copied exactly.
    """
    ek, ev, ep, valid = exact
    b, g, n, dk = k.shape
    p, a, dv = ek.size(2), indices.size(1) - ek.size(2), v.size(-1)
    ak, av = k.new_empty((b, g, a, dk)), v.new_empty((b, g, a, dv))
    ap = pos.new_empty((b, a))
    _alpha_partition[(triton.cdiv(p + a, 16), b, g)](
        k, v, pos, indices, ek, ev, ep, valid, ak, av, ap, n, p, a,
        g, dk, dv, 16, triton.next_power_of_2(max(dk, dv)), num_warps=4,
    )
    return ak, av, ap
