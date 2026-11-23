"""Local first-order compaction: four ordered entries become two contiguous groups."""

import numpy as np
import torch


def make_layout(lengths, device):
    """Host-known lane boundaries; columns are input start, width, output start."""
    lengths = np.asarray(lengths, dtype=np.int64)
    if np.any((lengths < 0) | (lengths % 2 != 0)):
        raise ValueError("Beta compaction requires even nonnegative lane lengths")
    groups = (lengths + 3) // 4
    offsets = 4 * (np.arange(groups.sum()) - np.repeat(groups.cumsum() - groups, groups))
    starts = np.repeat(lengths.cumsum() - lengths, groups) + offsets
    widths = np.minimum(4, np.repeat(lengths, groups) - offsets)
    # Vectorize the host schedule and transfer it once, without per-group
    # Python objects or device-to-host reads of the chosen cuts.
    return torch.from_numpy(np.column_stack((starts, widths, starts // 2))).to(device)


def _group_inputs(block, layout):
    offsets = torch.arange(4, device=layout.device)
    valid = offsets[None, :] < layout[:, 1, None]
    indices = (layout[:, :1] + offsets).clamp_max(block[0].size(0) - 1)
    w = block[2][indices].float() * valid
    return indices, valid, w


def select_cuts(block, layout):
    """One byte per group. Ties prefer 2|2; a two-entry tail has sentinel 1."""
    indices, valid, w = _group_inputs(block, layout)
    offsets = torch.arange(4, device=w.device)
    total = w.sum(-1).clamp_min(1e-12)
    costs = w.new_zeros((len(layout), 3))
    for values in block[:2]:
        x = values[indices].float()
        energy = ((x.square().sum(-1) * w).sum(-1) / total).clamp_min(1e-12)
        for j, cut in enumerate((2, 1, 3)):
            left = valid & (offsets < cut)
            for mask in (left, valid & ~left):
                weights = w * mask
                mean = (x * weights[..., None]).sum(1) / weights.sum(-1).clamp_min(1e-12)[:, None]
                loss = ((x - mean[:, None]).square().sum(-1) * weights).sum(-1)
                costs[:, j] += loss / energy
    choices = torch.tensor((2, 1, 3), dtype=torch.uint8, device=w.device)
    return torch.where(layout[:, 1] == 2, 1, choices[costs.argmin(-1)]).to(torch.uint8)


def apply_cuts(block, layout, cuts):
    """Apply a recorded decision without evaluating candidate costs."""
    indices, valid, w = _group_inputs(block, layout)
    offsets = torch.arange(4, device=w.device)
    n = block[0].size(0) // 2
    out = tuple(None if x is None else torch.empty((n, *x.shape[1:]), device=x.device,
                dtype=torch.float32 if i < 2 else x.dtype) for i, x in enumerate(block))
    sides = []
    for side in (0, 1):
        mask = valid & ((offsets < cuts[:, None]) if side == 0 else (offsets >= cuts[:, None]))
        mask = torch.where((layout[:, 1] == 2)[:, None], valid if side == 0 else False, mask)
        weights = w * mask
        mass = weights.sum(-1)
        active = torch.ones_like(cuts, dtype=torch.bool) if side == 0 else layout[:, 1] == 4
        dst = layout[active, 2] + side
        sides.append((mask, weights, mass, active, dst))
    for f, x in enumerate(block):
        if x is None:
            continue
        # Reuse one gather for both output groups, as in the fused CUDA path.
        rows = None if f == 2 else x[indices]
        if f < 2:
            rows = rows.float()
        for mask, weights, mass, active, dst in sides:
            if f < 2:
                value = (rows * weights[..., None]).sum(1) / mass.clamp_min(1e-12)[:, None]
            elif f == 2:
                value = mass
            elif f in (8, 9):
                live = mask & (w > 0)
                limit = torch.iinfo(x.dtype).max if f == 8 else torch.iinfo(x.dtype).min
                masked = rows.masked_fill(~live, limit)
                value = masked.amin(-1) if f == 8 else masked.amax(-1)
                value = torch.where(mass > 0, value, 0)
            elif f == 10:
                value = (rows * mask).sum(-1)
            elif f == 11:
                value = rows.masked_fill(~mask, torch.iinfo(x.dtype).max).amin(-1)
            elif f == 12:
                value = (rows | ~mask).all(-1)
            else:
                raise ValueError("Beta compaction only supports first-order entries")
            out[f][dst] = value[active]
    return out


def merge_blocks(block, lengths, cuts=None):
    if sum(lengths) != block[0].size(0):
        raise ValueError("Beta lane lengths must cover the compaction input")
    layout = make_layout(lengths, block[0].device)
    if cuts is None:
        cuts = select_cuts(block, layout) if len(layout) else torch.empty(0, dtype=torch.uint8, device=layout.device)
    elif cuts.shape != (len(layout),) or cuts.dtype != torch.uint8 or cuts.device != layout.device:
        raise ValueError("recorded Beta cuts do not match the compaction layout")
    if not len(layout):
        return tuple(x[:0] if x is not None else None for x in block), cuts
    return apply_cuts(block, layout, cuts), cuts
