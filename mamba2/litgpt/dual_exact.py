"""Two fixed-quota exact pools. Physical duplicates vote twice in ordinary attention."""
from dataclasses import dataclass

import numpy as np

from litgpt.alpha_log_kv import SpanCut, SpanSelection, add_cue_values, choose_spans, cut_spans


@dataclass
class DualSpanCut:
    word: SpanCut
    query: SpanCut
    merged: SpanCut


def cut_dual_spans(pool_spans, positions, new_positions, ends, old_width, max_span):
    """Build independent candidate sets, mapped to one shared [old physical pools | flush] source."""
    rows = len(positions)
    word_sizes = [sum(length for length, _ in spans) for spans in pool_spans[0]]
    cuts = []
    for pool in range(2):
        starts = word_sizes if pool else [0] * rows
        old_positions = []
        for b, spans in enumerate(pool_spans[pool]):
            n = sum(length for length, _ in spans)
            old_positions.append(positions[b][starts[b]:starts[b] + n])
        cut = cut_spans(pool_spans[pool], old_positions, new_positions, ends, old_width, max_span)
        if cut is None:
            return None
        for b in range(rows):
            tokens = cut.token_rows[b]
            cut.token_rows[b] = np.where(tokens < old_width, tokens + starts[b], tokens)
            all_pos = np.zeros(old_width + len(new_positions[b]), dtype=np.int64)
            all_pos[:len(positions[b])] = positions[b]
            all_pos[old_width:] = new_positions[b]
            cut.positions[b] = all_pos
        cuts.append(cut)
    word, query = cuts
    merged = SpanCut(
        word.positions,
        [np.concatenate((a, b)) for a, b in zip(word.token_rows, query.token_rows)],
        [np.concatenate((a, b)) for a, b in zip(word.length_rows, query.length_rows)],
        [np.concatenate((a, b)) for a, b in zip(word.old_rows, query.old_rows)],
        [np.concatenate((a, b)) for a, b in zip(word.closed_rows, query.closed_rows)],
    )
    return DualSpanCut(word, query, merged)


def choose_dual_spans(cut, base_scores, total_budget, query_budget, word_stream, query_stream, stops):
    """Reuse one GPU score transfer, but never mix word/query scores or borrow quotas."""
    bases = [[], []]
    offset = 0
    for a, b in zip(cut.word.length_rows, cut.query.length_rows):
        na, nb = len(a), len(b)
        bases[0].extend(base_scores[offset:offset + na])
        bases[1].extend(base_scores[offset + na:offset + na + nb])
        offset += na + nb
    choices = []
    for candidate, scores, budget, stream in (
            (cut.word, bases[0], total_budget - query_budget, word_stream),
            (cut.query, bases[1], query_budget, query_stream)):
        if stream is not None:
            scores = add_cue_values(candidate, scores, stream, stops)
        choices.append(choose_spans(candidate, scores, budget))
    word, query = choices
    keeps, archives, spans, positions, archive_positions = [], [], [], [], []
    for b in range(len(word.keep)):
        keep = word.keep[b] + query.keep[b]  # Deliberate physical duplicates; original RoPE positions.
        kept_positions = set(word.positions[b]) | set(query.positions[b])
        # A true token is archived only when neither pool keeps it, and exactly once.
        sources = {}
        for token in cut.merged.token_rows[b]:
            position = int(cut.merged.positions[b][token])
            if position not in kept_positions:
                sources.setdefault(position, int(token))
        archive_pos = sorted(sources)
        keeps.append(keep)
        archives.append([sources[p] for p in archive_pos])
        spans.append(word.spans[b] + query.spans[b])
        positions.append(word.positions[b] + query.positions[b])
        archive_positions.append(np.asarray(archive_pos, dtype=np.int64))
    selected = SpanSelection(keeps, archives, spans, positions, archive_positions)
    return selected, [word.spans, query.spans]
