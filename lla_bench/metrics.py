"""Physical state bytes; matrix states have no count of retained token KV pairs."""

# Dense BF16 K and V for Qwen3-1.7B-Base. Both lm-eval adapters process batch 1.
QWEN_DENSE_BYTES_PER_TOKEN = 28 * 8 * 128 * 2 * 2


def allocated_byte_ratios(budget, summary):
    """Pool one persistent allocation per request against the same dense byte denominator."""
    allocated = budget.get("persistent_cache_bytes")
    dense = summary["processed_tokens"] * QWEN_DENSE_BYTES_PER_TOKEN
    total = allocated * summary["request_count"] if allocated is not None else None
    ratio = total / dense if total is not None and dense else None
    return dict(dense_reference_bytes=dense, allocated_byte_ratio=ratio,
                allocated_byte_saving_ratio=1 - ratio if ratio is not None else None,
                allocated_byte_compression_factor=dense / total if total and dense else None)


def with_ratios(counts):
    dense = counts["dense_kv_bytes"]
    live = counts["live_state_bytes"]
    return {**counts, "live_state_byte_ratio": live / dense if dense else None,
            "live_state_byte_saving_ratio": 1 - live / dense if dense else None,
            "live_state_byte_compression_factor": dense / live if live else None,
            "retained_slots": None, "kv_retention_ratio": None,
            "kv_saving_ratio": None, "kv_compression_factor": None}


def cache_compression_snapshot(model, processed_tokens):
    # Common baseline denominator: Qwen3-1.7B-Base dense BF16, 28 layers,
    # 8 KV groups and width 128, at this model's actual forwarded token count.
    live_bytes = live_matrices = conv_bytes = 0
    for block in model.backbone.layers:
        cache = block.mixer.cache
        if cache.count != processed_tokens:
            raise ValueError(f"matrix cache processed {cache.count} tokens, harness counted {processed_tokens}")
        _, batch, heads, kdim, vdim = cache.states.shape
        count = cache.count.bit_count() * batch * heads
        live_bytes += count * kdim * vdim * cache.states.element_size()
        live_matrices += count
        conv = cache.conv
        conv_bytes += conv.shape[0] * conv.shape[1] * min(processed_tokens, conv.shape[2]) * conv.element_size()
    dense_slots = processed_tokens * batch * 28 * 8
    read_levels = (processed_tokens - 1).bit_length() + 1 if processed_tokens else 0
    return with_ratios(dict(processed_tokens=processed_tokens, live_matrices=live_matrices,
                            extrapolated_read_levels=max(0, read_levels - model.config.trained_levels),
                            live_state_bytes=live_bytes + conv_bytes, live_conv_bytes=conv_bytes,
                            dense_kv_bytes=processed_tokens * batch * QWEN_DENSE_BYTES_PER_TOKEN, dense_slots=dense_slots))


def summarize_compression(records):
    fields = ("processed_tokens", "live_matrices", "live_conv_bytes", "live_state_bytes", "dense_kv_bytes", "dense_slots")
    return with_ratios({"request_count": len(records),
                        "requests_past_trained_levels": sum(r["extrapolated_read_levels"] > 0 for r in records),
                        "max_extrapolated_read_levels": max((r["extrapolated_read_levels"] for r in records), default=0),
                        **{f: sum(r[f] for r in records) for f in fields}})
