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


@triton.jit(do_not_specialize=["N", "NL", "START"])
def _stage(F, O, K, V, POS, TB, TG, TT, LROW, PERM, ORDER, N, NL, START,
           SKB, SKG, SKT, SVB, SVG, SVT, SPB,
           TAIL: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    """Staging row START + r: token r, or with TAIL, entry PERM[r] of [ladder rows | tokens]."""
    r = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    live = r < N
    if TAIL:
        j = tl.load(PERM + r, live, 0).to(tl.int64)
        ladder = live & (j < NL)
        row = tl.load(LROW + j, ladder, 0).to(tl.int64)
        t = j - NL
    else:
        ladder = r < 0
        row = r
        t = r
    token = live & ~ladder
    b = tl.load(TB + t, token, 0).to(tl.int64)
    g = tl.load(TG + t, token, 0).to(tl.int64)
    s = tl.load(TT + t, token, 0).to(tl.int64)
    pos = tl.load(POS + b * SPB + s, token, 0)
    order = tl.load(ORDER + r, live, 0)
    out = START + r
    d = tl.arange(0, BD).to(tl.int64)
    for f in tl.static_range(8):
        dim = DK if f == 0 else DV if f == 1 else 1
        mask = (d < dim)[None, :]
        stored = tl.load(F[f] + row[:, None] * dim + d[None, :], ladder[:, None] & mask, 0)
        # A token enters with unit weight at its own position, unpadded.
        if f == 0:
            fresh = tl.load(K + (b * SKB + g * SKG + s * SKT)[:, None] + d[None, :], token[:, None] & mask, 0)
        elif f == 1:
            fresh = tl.load(V + (b * SVB + g * SVG + s * SVT)[:, None] + d[None, :], token[:, None] & mask, 0)
        elif f == 2:
            fresh = tl.full([BT, 1], 1.0, tl.float32)
        elif f < 6:
            fresh = pos[:, None]
        else:
            fresh = tl.zeros([BT, 1], tl.int32)
        kind = O[f].dtype.element_ty
        value = tl.where(ladder[:, None], stored.to(kind), fresh.to(kind))
        if f == 6:
            value = order[:, None].to(kind) + tl.zeros([BT, BD], kind)
        tl.store(O[f] + out[:, None] * dim + d[None, :], value, live[:, None] & mask)


def stage(out, fields, k, v, positions, tokens, order, *, start=0, ladder_rows=None, perm=None):
    """Write first-order staging rows [start, start + len(order)) of `out`.

    Row r is token (tokens[0][r], tokens[1][r], tokens[2][r]) with unit weight
    at its own position, or with `perm`, entry perm[r] of [ladder rows
    `ladder_rows` of the flat `fields` | those tokens]. Its order is order[r].
    Values convert exactly: every staging dtype contains its sources'.
    """
    n = order.numel()
    if not n:
        return
    if k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("staged K/V rows must have a contiguous last dimension")
    if positions.stride(-1) != 1:
        positions = positions.contiguous()
    o, f = tuple(out[i] for i in _FIELDS), tuple(fields[i] for i in _FIELDS)
    tb, tg, tt = tokens
    tail = perm is not None
    dk, dv = fields[0].size(1), fields[1].size(1)
    _stage[(triton.cdiv(n, 8),)](
        f, o, k, v, positions, tb, tg, tt, ladder_rows if tail else tb, perm if tail else tb, order,
        n, ladder_rows.numel() if tail else 0, start,
        k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2), positions.stride(0),
        tail, dk, dv, 8, triton.next_power_of_2(max(dk, dv)), num_warps=4, enable_fp_fusion=False,
    )


@triton.jit(do_not_specialize=["N"])
def _clear(F, IMP, ROWS, N, W: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BT: tl.constexpr,
           BD: tl.constexpr):
    """Zero W consecutive slots from each ROWS start in the 8 first-order fields and importance."""
    r = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    live = r < N * W
    row = tl.load(ROWS + r // W, live, 0).to(tl.int64) + r % W
    d = tl.arange(0, BD).to(tl.int64)
    for f in tl.static_range(8):
        dim = DK if f == 0 else DV if f == 1 else 1
        mask = live[:, None] & (d < dim)[None, :]
        tl.store(F[f] + row[:, None] * dim + d[None, :], tl.zeros([BT, BD], F[f].dtype.element_ty), mask)
    tl.store(IMP + row, tl.zeros([BT], IMP.dtype.element_ty), live)


def clear(fields, importance, starts, width):
    """`index_fill_(0)` of `width` slots from each flat row in `starts`, one launch for all fields."""
    if starts.numel():
        f = tuple(fields[i] for i in _FIELDS)
        dk, dv = fields[0].size(1), fields[1].size(1)
        _clear[(triton.cdiv(starts.numel() * width, 16),)](
            f, importance, starts, starts.numel(), width, dk, dv, 16, triton.next_power_of_2(max(dk, dv)),
            num_warps=4,
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
        from litgpt.beta_log_kv import make_layout

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


@triton.jit(do_not_specialize=["N"])
def _top_lightest(F, S, P, IDS, WS, IMP, META, N, B: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                  BB: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    """One saturated top level per program: m lightest-pair merges, each followed by an append.

    The lane's pool P holds its B level rows, then its m staged rows. Slots name
    pool rows, so a merge rewrites one row (the older of the pair) and shifts B
    ids instead of B rows; slot order stays pool order, as in the reference.
    Slot ids and masses live in registers; each step passes them through IDS/WS
    once to read the left-shifted copies. Steps are sequential, so one warp runs
    a lane and reductions stay in-warp.
    """
    lane = tl.program_id(0).to(tl.int64)
    top = tl.load(META + lane)
    start = tl.load(META + N + lane)
    m = tl.load(META + 2 * N + lane).to(tl.int32)
    pool = tl.load(META + 3 * N + lane)
    d = tl.arange(0, BD).to(tl.int64)
    r = tl.arange(0, BT).to(tl.int64)
    # Staged rows enter in the storage dtypes, like the reference's `.to(f.dtype)`.
    for r0 in range(0, B + m, BT):
        row = r0 + r
        level = row < B
        staged = (row >= B) & (row < B + m)
        for f in tl.static_range(8):
            dim = DK if f == 0 else DV if f == 1 else 1
            mask = (d < dim)[None, :]
            a = tl.load(F[f] + (top + row)[:, None] * dim + d[None, :], level[:, None] & mask, 0)
            s = tl.load(S[f] + (start + row - B)[:, None] * dim + d[None, :], staged[:, None] & mask, 0)
            value = tl.where(level[:, None], a, s.to(F[f].dtype.element_ty))
            tl.store(P[f] + (pool + row)[:, None] * dim + d[None, :], value, (level | staged)[:, None] & mask)
    tl.debug_barrier()
    j = tl.arange(0, BB)
    first = j < B
    ids = j
    w = tl.load(P[2] + pool + j, first, 0.0)
    scratch = lane * BB + j
    for t in range(m):
        tl.store(IDS + scratch, ids, first)
        tl.store(WS + scratch, w, first)
        tl.debug_barrier()  # Also orders the previous step's merged row before its reads.
        later = j < B - 1
        nxt = tl.load(IDS + scratch + 1, later, 0)
        wn = tl.load(WS + scratch + 1, later, 0.0)
        # Masses are >= 0, so float bits order like values: one min over
        # (pair mass bits, slot) finds the first minimal pair, as `argmin`.
        key = ((w + wn).to(tl.int32, bitcast=True).to(tl.int64) << 32) | j.to(tl.int64)
        i = (tl.min(tl.where(later, key, 9223372036854775807), 0) & 0xFFFFFFFF).to(tl.int32)
        pick = j == i
        ia = pool + tl.sum(tl.where(pick, ids, 0), 0).to(tl.int64)
        ib = pool + tl.sum(tl.where(pick, nxt, 0), 0).to(tl.int64)
        wa = tl.sum(tl.where(pick, w, 0.0), 0)
        wb = tl.sum(tl.where(pick, wn, 0.0), 0)
        # The pair merges into the older row: read both rows, then write.
        alpha = tl.div_rn(wa, tl.maximum(wa + wb, 1.e-8))
        ka = tl.load(P[0] + ia * DK + d, d < DK, 0).to(tl.float32)
        kb = tl.load(P[0] + ib * DK + d, d < DK, 0).to(tl.float32)
        va = tl.load(P[1] + ia * DV + d, d < DV, 0).to(tl.float32)
        vb = tl.load(P[1] + ib * DV + d, d < DV, 0).to(tl.float32)
        both = (wa > 0) & (wb > 0)
        lo_a, lo_b = tl.load(P[3] + ia), tl.load(P[3] + ib)
        hi_a, hi_b = tl.load(P[4] + ia), tl.load(P[4] + ib)
        lo = tl.where(both, tl.minimum(lo_a, lo_b), tl.where(wa > 0, lo_a, lo_b))
        hi = tl.where(both, tl.maximum(hi_a, hi_b), tl.where(wa > 0, hi_a, hi_b))
        sum_wp = tl.load(P[5] + ia) + tl.load(P[5] + ib)
        order = tl.minimum(tl.load(P[6] + ia), tl.load(P[6] + ib))
        pad = tl.load(P[7] + ia) & tl.load(P[7] + ib)
        tl.debug_barrier()
        tl.store(P[0] + ia * DK + d, (alpha * ka + (1.0 - alpha) * kb).to(P[0].dtype.element_ty), d < DK)
        tl.store(P[1] + ia * DV + d, (alpha * va + (1.0 - alpha) * vb).to(P[1].dtype.element_ty), d < DV)
        tl.store(P[2] + ia, wa + wb)
        tl.store(P[3] + ia, lo)
        tl.store(P[4] + ia, hi)
        tl.store(P[5] + ia, sum_wp)
        tl.store(P[6] + ia, order)
        tl.store(P[7] + ia, pad)
        # Slots before the pair keep their rows, the pair's slot names the merged
        # row, later slots shift left and the staged entry takes the last slot.
        incoming = tl.load(P[2] + pool + B + t)
        ids = tl.where(j < i, ids, tl.where(pick, (ia - pool).to(tl.int32), tl.where(later, nxt, B + t)))
        w = tl.where(j < i, w, tl.where(pick, wa + wb, tl.where(later, wn, incoming)))
        tl.debug_barrier()  # Every thread has read this step's shifted copies.
    tl.store(IDS + scratch, ids, first)
    tl.debug_barrier()
    for r0 in range(0, B, BT):
        row = r0 + r
        live = row < B
        src = pool + tl.load(IDS + lane * BB + row, live, 0).to(tl.int64)
        for f in tl.static_range(8):
            dim = DK if f == 0 else DV if f == 1 else 1
            mask = live[:, None] & (d < dim)[None, :]
            value = tl.load(P[f] + src[:, None] * dim + d[None, :], mask, 0)
            tl.store(F[f] + (top + row)[:, None] * dim + d[None, :], value, mask)
        tl.store(IMP + top + row, tl.zeros([BT], IMP.dtype.element_ty), live)


def top_lightest(fields, importance, stage, meta, pool_rows, width):
    """Gamma's lightest-pair overflow of saturated top levels, one program per level.

    `meta` [4, n] (int64) holds each level's flat row of slot 0, its first
    staged row in `stage`, its staged count m >= 1 and its offset in a pool of
    `pool_rows` = sum(width + m) rows. Per staged row, the adjacent pair with the
    smallest combined mass merges (ties: the oldest), later slots shift left
    and the row takes the last slot; merged K/V round to the storage dtype each
    step. Importance of the written slots is cleared.
    """
    n = meta.size(1)
    if not n:
        return
    f, s = tuple(fields[i] for i in _FIELDS), tuple(stage[i] for i in _FIELDS)
    pool = tuple(x.new_empty((pool_rows, *x.shape[1:])) for x in f)
    bb = triton.next_power_of_2(width)
    ids = torch.empty((n, bb), dtype=torch.int32, device=meta.device)
    masses = torch.empty((n, bb), dtype=torch.float32, device=meta.device)
    dk, dv = fields[0].size(1), fields[1].size(1)
    _top_lightest[(n,)](
        f, s, pool, ids, masses, importance, meta.contiguous(), n, width, dk, dv, bb, 8,
        triton.next_power_of_2(max(dk, dv)), num_warps=1, enable_fp_fusion=False,
    )


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
