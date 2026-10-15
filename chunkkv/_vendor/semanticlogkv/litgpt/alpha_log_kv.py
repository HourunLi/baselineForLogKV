# Vendored for ChunkKV; local namespace changes are recorded in SOURCE.json.
"""Bounded whole-span selection. Scores are compression-risk proxies, not labels."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpanSelection:
    keep: list[list[int]]
    archive: list[list[int]]
    spans: list[list[tuple[int, bool]]]
    positions: list[list[int]]


def _new_runs(new_pos, ends, pending, pending_pos, max_span):
    """Whole-array form of the per-token span loop over one flush row.

    Returns (pending_closed, starts, stops, closed). A run [start, stop) of new
    tokens closes after a boundary or the length cap, or before a position
    jump. The first run continues the old unfinished span unless a jump
    closes that span first; the cap counts its old tokens too.
    """
    count = len(new_pos)
    empty = np.empty(0, dtype=np.int64)
    if not count:
        return False, empty, empty, np.empty(0, dtype=bool)
    pos = np.asarray(new_pos, dtype=np.int64)
    end = np.asarray(ends, dtype=bool)
    pending_closed = bool(pending) and int(pos[0]) != pending_pos + 1
    prefix = 0 if pending_closed else pending
    jump = np.zeros(count, dtype=bool)
    jump[1:] = pos[1:] != pos[:-1] + 1
    # Hard segments restart at jumps and after boundaries; the cap splits
    # each segment into max_span chunks, counted from its first token.
    first = np.ones(count, dtype=bool)
    first[1:] = jump[1:] | end[:-1]
    segment = np.cumsum(first) - 1
    index = np.arange(count, dtype=np.int64)
    filled = index - np.flatnonzero(first)[segment] + np.where(segment == 0, prefix, 0) + 1
    cap = filled % max_span == 0
    if prefix >= max_span:
        # An over-long old span never equals the cap again in the loop.
        cap &= segment != 0
    close = end | cap
    stop = close.copy()
    stop[:-1] |= jump[1:]
    stop[-1] = True
    stops = np.flatnonzero(stop) + 1
    starts = np.concatenate(([0], stops[:-1]))
    closed = close[stops - 1]
    closed[:-1] |= jump[stops[:-1]]
    return pending_closed, starts, stops, closed


@dataclass
class SpanCut:
    """Candidate whole spans of one flush; host data only."""
    candidates: list  # Per row: (spans as (token ranges, old, closed), positions of [pool | flush]).
    token_rows: list  # Per row: flat token indices of the spans, in span order.
    length_rows: list  # Per row: span lengths.


def _empty_selection(rows):
    return SpanSelection([[] for _ in range(rows)], [[] for _ in range(rows)],
                         [[] for _ in range(rows)], [[] for _ in range(rows)])


def select_spans(k, v, old_spans, old_positions, new_positions, ends, old_width, budget, max_span,
                 *, beta_novelty=False, centroids=None, centroid_valid=None):
    """Indices refer to [padded old pool | current flush], shared across KV groups.

    A final unfinished span is mandatory until a boundary or the length cap.
    Scoring makes one batched GPU pass and one small score transfer per flush.
    Beta scores novelty against the old centroids captured before this flush.
    """
    cut = cut_spans(old_spans, old_positions, new_positions, ends, old_width, max_span)
    if cut is None:
        return _empty_selection(len(ends))
    scores = score_spans(k, v, cut, beta_novelty=beta_novelty, centroids=centroids, centroid_valid=centroid_valid)
    return choose_spans(cut, scores.cpu().tolist(), budget)


def cut_spans(old_spans, old_positions, new_positions, ends, old_width, max_span):
    """Whole candidate spans per row (host only); None when there are none."""
    from chunkkv._vendor.semanticlogkv.litgpt.log_kv_cache import _spans_to_index_array

    candidates, token_rows, length_rows = [], [], []
    for spans, old_pos, new_pos, boundaries in zip(old_spans, old_positions, new_positions, ends):
        # A span is (token ranges, old, closed); ranges are [start, stop) pairs.
        row, offset, pending = [], 0, None
        for length, closed in spans:
            if closed:
                row.append(([(offset, offset + length)], True, True))
            else:
                pending = (offset, offset + length)
            offset += length
        all_pos = np.zeros(old_width + len(new_pos), dtype=np.int64)
        all_pos[:len(old_pos)] = old_pos
        all_pos[old_width:] = new_pos
        pending_len = 0 if pending is None else pending[1] - pending[0]
        pending_closed, starts, stops, closed = _new_runs(
            new_pos, boundaries, pending_len, int(all_pos[pending[1] - 1]) if pending_len else 0, max_span)
        if pending_closed:
            row.append(([pending], False, True))
        bounds = (starts + old_width).tolist(), (stops + old_width).tolist(), closed.tolist()
        for i, (start, stop, done) in enumerate(zip(*bounds)):
            ranges = [(start, stop)]
            if not i and pending_len and not pending_closed:
                ranges.insert(0, pending)
            row.append((ranges, False, done))
        if pending_len and not pending_closed and not len(starts):
            row.append(([pending], False, False))
        ranges = [r for span, _, _ in row for r in span]
        tokens = _spans_to_index_array(ranges)
        lengths = [sum(stop - start for start, stop in span) for span, _, _ in row]
        candidates.append((row, all_pos))
        token_rows.append(tokens)
        length_rows.append(lengths)
    if not any(length_rows):
        return None
    return SpanCut(candidates, token_rows, length_rows)


def score_spans(k, v, cut, *, beta_novelty=False, centroids=None, centroid_valid=None):
    """Device tensor of span scores in candidate order; nothing is read back."""
    from chunkkv._vendor.semanticlogkv.litgpt.log_kv_cache import _upload

    if beta_novelty and (centroids is None or centroid_valid is None):
        raise ValueError("Beta span scoring requires centroids and centroid_valid from before the flush")
    token_rows, length_rows = cut.token_rows, cut.length_rows
    sizes = [len(lengths) for lengths in length_rows]
    tokens = np.concatenate(token_rows)
    lengths = np.concatenate([np.asarray(row, dtype=np.int64) for row in length_rows])
    token_batch = np.repeat(np.arange(len(token_rows)), [len(row) for row in token_rows])
    parts = [token_batch, tokens, lengths]
    if beta_novelty:
        parts += [np.repeat(np.arange(len(sizes)), sizes),
                  np.concatenate([np.arange(n, dtype=np.int64) for n in sizes])]
    # One asynchronous metadata upload for gathers, segments and span slots.
    metadata = _upload(np.concatenate(parts), k.device)
    index = metadata[:2 * len(tokens)].view(2, -1)
    span_sizes = metadata[2 * len(tokens):2 * len(tokens) + len(lengths)]
    scores = None
    for component, values in enumerate((k, v)):
        x = values[index[0], :, index[1]].detach().float()
        means = torch.segment_reduce(x, "mean", lengths=span_sizes, unsafe=True)
        energy = torch.segment_reduce(x.square().mean(-1), "mean", lengths=span_sizes, unsafe=True)
        mean_square = means.square().mean(-1)
        residual = (energy - mean_square).clamp_min(0)
        part = residual / energy.clamp_min(1e-12)
        if beta_novelty and component == 0:
            key_means, key_energy, key_mean_square = means, energy, mean_square
        if not beta_novelty:
            part = part.mean(-1)
        scores = part if scores is None else scores + part
    if beta_novelty:
        span_index = metadata[2 * len(tokens) + len(lengths):].view(2, -1)
        # Batch the ragged spans instead of repeating all 12 D-dimensional
        # centers per span; the distance workspace is only span x cluster.
        padded = key_means.new_zeros(k.size(0), max(sizes), k.size(1), k.size(-1))
        padded[span_index[0], span_index[1]] = key_means
        centers = centroids.detach().float()
        dots = (padded.transpose(1, 2) @ centers.transpose(-1, -2)).transpose(1, 2)
        dots = dots[span_index[0], span_index[1]] / k.size(-1)
        distances = (key_mean_square[..., None]
                     + centers.square().mean(-1)[span_index[0]] - 2 * dots).clamp_min_(0)
        valid = centroid_valid[span_index[0]]
        nearest = distances.masked_fill(~valid, torch.inf).amin(-1)
        nearest = torch.where(centroid_valid.any(-1)[span_index[0]], nearest, 0.)
        novelty = nearest / key_energy.clamp_min(1e-12)
        scores = scores * .5 + novelty / (1 + novelty)
        scores = scores.topk(min(2, k.size(1)), dim=-1).values.mean(-1)
    return scores


def choose_spans(cut, host_scores, budget):
    """Bounded whole-span greedy from host scores in candidate order."""
    keep, archive, output_spans, output_positions = [], [], [], []
    offset = 0
    for (row, positions), tokens, size in zip(cut.candidates, cut.token_rows, cut.length_rows):
        mandatory = {i for i, (_, _, closed) in enumerate(row) if not closed}
        room = budget - sum(size[i] for i in mandatory)
        density = [host_scores[offset + i] * (1.1 if old else 1.) for i, (_, old, _) in enumerate(row)]
        chosen = mandatory | {i for i, (_, old, _) in enumerate(row) if old}
        room -= sum(size[i] for i in chosen - mandatory)
        victims = lambda: sorted(chosen - mandatory, key=lambda i: (density[i], -i))
        # Make room for the bounded unfinished tail first.
        for i in victims():
            if room >= 0:
                break
            chosen.remove(i)
            room += size[i]
        # ponytail: bounded whole-span swaps, not an exact knapsack solver.
        # A short new span must pay for EVERY token in the whole spans evicted.
        ranked = sorted((i for i in range(len(row)) if i not in chosen), key=lambda i: (-density[i], i))
        victim_order = None
        for i in ranked:
            length = size[i]
            available, loss, remove = room, 0., []
            if available < length:
                # Failed replacements do not change the ordering. Sorting it
                # again for every rejected candidate dominated short spans.
                if victim_order is None:
                    victim_order = victims()
                for j in victim_order:
                    remove.append(j)
                    available += size[j]
                    loss += density[j] * size[j]
                    if available >= length:
                        break
            if available >= length and (not remove or density[i] * length > loss):
                chosen.difference_update(remove)
                chosen.add(i)
                room = available - length
                victim_order = None
        selected = sorted(chosen)
        kept = np.zeros(len(row), dtype=bool)
        kept[selected] = True
        kept = np.repeat(kept, size)
        keep.append(tokens[kept].tolist())
        archive.append(tokens[~kept].tolist())
        output_spans.append([(size[i], row[i][2]) for i in selected])
        output_positions.append(positions[tokens[kept]].tolist())
        offset += len(row)
    return SpanSelection(keep, archive, output_spans, output_positions)
