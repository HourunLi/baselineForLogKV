"""KV-entry and inference-speed summaries with SemanticLogKV's field definitions.

An entry is one stored K/V pair in one batch item, layer and stored head. The
dense denominator counts forwarded tokens for every layer and KV group, so the
ratios are directly comparable with SemanticLogKV's ``kv_compression`` block.
With the legacy ``query_head`` layout each KV group stores several heads and
the retention ratio can exceed 1; that is the real entry count, not a bug.
Counts are logical entries, not allocator bytes.
"""

from __future__ import annotations

TOTAL_FIELDS = (
    "processed_tokens",
    "layer_group_count",
    "recent_slots",
    "exact_slots",
    "hierarchical_slots",
    "retained_slots",
    "dense_slots",
)


def with_ratios(counts: dict) -> dict:
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
    """Count all layers before the request cache is released.

    ``processed_tokens`` is the number of tokens actually forwarded (an unforwarded
    final output token excluded); every layer must agree with it.
    """
    counts = dict.fromkeys(TOTAL_FIELDS, 0)
    retained_bytes = 0
    caches = model.caches
    if not caches or any(cache is None for cache in caches):
        raise ValueError("Cannot measure KV compression without an initialized cache in every layer")
    for cache in caches:
        row = cache.snapshot()
        if row["processed_tokens"] != processed_tokens:
            raise ValueError(f"Layer {cache.layer_idx} forwarded {row['processed_tokens']} tokens, "
                             f"expected {processed_tokens}")
        for field in TOTAL_FIELDS[1:]:
            counts[field] += row[field]
        retained_bytes += row["retained_slots"] * row["entry_bytes"]
    counts["processed_tokens"] = processed_tokens
    if counts["retained_slots"] != counts["recent_slots"] + counts["exact_slots"] + counts["hierarchical_slots"]:
        raise ValueError("Retained KV entries do not add up")
    return {**with_ratios(counts), "retained_kv_bytes": retained_bytes}


def summarize_compression(records: list[dict]) -> dict:
    """Pool entries over requests (ΣS / ΣD); never average per-request ratios."""
    counts = {field: sum(record[field] for record in records) for field in TOTAL_FIELDS}
    return {"request_count": len(records), **with_ratios(counts),
            "retained_kv_bytes": sum(record.get("retained_kv_bytes") or 0 for record in records)}


def summarize_inference_metrics(gen_metrics: list[dict], ppl_metrics: list[dict]) -> dict:
    """Pool token counts and stage times across requests, never average their rates."""
    def rate(tokens, seconds):
        return tokens / seconds if tokens > 0 and seconds > 0 else None

    def peak_memory(rows):
        return {key: max((row[key] for row in rows if row[key] is not None), default=None)
                for key in ("peak_memory_allocated_bytes", "peak_memory_reserved_bytes")}

    prompt_tokens = sum(row["prompt_len"] for row in gen_metrics)
    generated_tokens = sum(row["generated_tokens"] for row in gen_metrics)
    decode_tokens = sum(row["decode_tokens"] for row in gen_metrics)
    total_time = sum(row["total_time_s"] for row in gen_metrics)
    prefill_time = sum(row["prefill_time_s"] for row in gen_metrics)
    decode_time = sum(row["decode_time_s"] for row in gen_metrics)
    forward_tokens = sum(row["total_seq_len"] for row in ppl_metrics)
    forward_time = sum(row["forward_time_s"] for row in ppl_metrics)
    return {
        "throughput_scope": "Effective single-device throughput: sum(tokens) / sum(request stage seconds) across all ranks; not aggregate cluster throughput.",
        "timing_scope": "Model calls only, excluding tokenization, cache setup/reset, result gathering, and file output. Prefill includes prompt compression and first-token sampling; decode excludes the first generated token. Generation counts include EOS; no warmup requests are discarded.",
        "memory_scope": "Maximum per-request CUDA allocator peak across ranks, including resident model, cache and compression scoring workspace; not KV-only memory or total device memory. CPU values are null.",
        "generation": {
            "requests": len(gen_metrics),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "decode_tokens": decode_tokens,
            "total_time_s": total_time,
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "avg_ttft_ms": 1000 * prefill_time / len(gen_metrics) if gen_metrics else None,
            "prefill_tokens_per_sec": rate(prompt_tokens, prefill_time),
            "decode_tokens_per_sec": rate(decode_tokens, decode_time),
            "gen_tokens_per_sec": rate(generated_tokens, total_time),
            "decode_ms_per_token": 1000 * decode_time / decode_tokens if decode_tokens else None,
            **peak_memory(gen_metrics),
        },
        "scoring": {
            "requests": len(ppl_metrics),
            "input_tokens": forward_tokens,
            "forward_time_s": forward_time,
            "forward_tokens_per_sec": rate(forward_tokens, forward_time),
            **peak_memory(ppl_metrics),
        },
    }
