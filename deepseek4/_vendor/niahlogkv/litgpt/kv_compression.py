"""Logical KV entries at request completion, relative to an uncompressed cache.

An entry is one stored K/V pair in one batch item, layer and KV group. Counts
include recent/deferred tokens, Alpha exact spans (including unfinished spans), and live
hierarchical entries. They exclude padding, multi-anchor attention expansion,
preallocated capacity and packing workspaces; these are not GPU-memory metrics.
"""

_TOTAL_FIELDS = (
    "processed_tokens",
    "layer_group_count",
    "recent_slots",
    "exact_slots",
    "hierarchical_slots",
    "retained_slots",
    "dense_slots",
)


def _with_ratios(counts: dict) -> dict:
    retained, dense = counts["retained_slots"], counts["dense_slots"]
    layer_groups = counts["layer_group_count"]
    return {
        **counts,
        "retained_slots_per_layer_group": retained / layer_groups if layer_groups else None,
        "kv_retention_ratio": retained / dense if dense else None,
        "kv_saving_ratio": 1 - retained / dense if dense else None,
        "kv_compression_factor": dense / retained if dense and retained else None,
    }


def cache_compression_snapshot(model, processed_tokens: int) -> dict:
    """Count every layer and batch/group lane before the request cache is reset.

    LogKV's committed tokens plus its deferred tail are authoritative. ``processed_tokens``
    supplies that count for native caches, which do not track it themselves. The caller
    must count tokens actually forwarded, excluding an unforwarded final output.
    ``layer_group_count`` includes the batch dimension.
    """
    counts = dict.fromkeys(_TOTAL_FIELDS, 0)
    token_count = None
    for block in model.transformer.h:
        cache = block.attn.kv_cache
        if cache is None:
            raise ValueError("Cannot measure KV compression without an initialized cache in every layer")
        current_tokens = cache.pos
        if current_tokens != processed_tokens:
            raise ValueError(f"V4 cache cursor {current_tokens} differs from forwarded count {processed_tokens}")
        token_count = current_tokens
        batch_groups = cache.local.size(0)  # V4 has one shared KV group
        recent = min(cache.local.size(1), current_tokens) * batch_groups
        exact = 0
        ratio = block.attn.ratio
        hierarchical = current_tokens // ratio * batch_groups if ratio else 0
        counts["layer_group_count"] += batch_groups
        counts["recent_slots"] += recent
        counts["exact_slots"] += exact
        counts["hierarchical_slots"] += hierarchical
        counts["dense_slots"] += current_tokens * batch_groups
    if token_count is None:
        raise ValueError("Cannot measure KV compression for a model without attention layers")
    counts["processed_tokens"] = token_count
    counts["retained_slots"] = counts["recent_slots"] + counts["exact_slots"] + counts["hierarchical_slots"]
    return _with_ratios(counts)


def summarize_compression(records: list[dict]) -> dict:
    """Aggregate request snapshots by total entries, rather than averaging ratios."""
    counts = {field: sum(record[field] for record in records) for field in _TOTAL_FIELDS}
    return {"request_count": len(records), **_with_ratios(counts)}
