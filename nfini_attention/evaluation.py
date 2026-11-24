from dataclasses import dataclass
from pathlib import Path

from lla_bench.metrics import with_ratios
from lla_bench.student import checkpoint_report
from .model import InfiniLM


@dataclass
class EvalConfig:
    block_size: int
    segment_length: int
    dense_geometry: tuple
    rope_adjustments: dict | None
    inference_mlp_chunk_size: int = 2048


def load_model(folder, device, options):
    if set(options) - {"block_size", "inference_mlp_chunk_size", "rope_adjustments"}:
        raise ValueError(f"unsupported Infini model options: {options}")
    manifest = checkpoint_report(Path(folder))
    window = int(options.get("block_size", 32768))
    yarn = options.get("rope_adjustments")
    model = InfiniLM.load_local(folder, window=window, yarn=yarn).to(device).eval()
    c = model.config
    geometry = (c.num_hidden_layers, c.num_key_value_heads, c.head_dim)
    return model, EvalConfig(window, c.infini_segment_length, geometry, yarn), manifest


def cache_budget(model):
    caches = [layer.cache for layer in model.model.layers]
    def size(names):
        return sum(getattr(c, name).nbytes for c in caches for name in names)
    local, memory = size(("k", "v")), size(("m", "z"))
    return dict(persistent_cache_bytes=local + memory, local_kv_bytes=local,
                memory_bytes=memory, state_dtype="float32", local_dtype=str(caches[0].k.dtype),
                window=model.max_seq_length)


def snapshot(model, processed):
    c = model.config
    caches = [layer.cache for layer in model.model.layers]
    if any(cache.count != processed for cache in caches):
        raise ValueError("cache position disagrees with actual forwarded tokens")
    groups = caches[0].k.shape[0] * c.num_hidden_layers * c.num_key_value_heads
    dense = processed * groups
    slots = (processed % c.infini_segment_length) * groups
    local = slots * 2 * c.head_dim * caches[0].k.element_size()
    memory = sum(cache.m.nbytes + cache.z.nbytes for cache in caches) if processed >= c.infini_segment_length else 0
    return with_ratios(dict(processed_tokens=processed, dense_slots=dense,
                            dense_kv_bytes=dense * 2 * c.head_dim * 2, local_kv_slots=slots,
                            live_local_kv_bytes=local, live_memory_bytes=memory, live_state_bytes=local + memory))
