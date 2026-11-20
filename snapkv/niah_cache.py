"""Streaming SnapKV at the reference LM's measured persistent-cache budget."""
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from .compressor import StreamingSnapKV
from .kv_cache import CompressedKVCache
from ._vendor.niahlogkv.litgpt.kv_compression import _with_ratios


class ReferenceCache(CompressedKVCache):
    @staticmethod
    def attention(q, k, v):
        t, length = q.size(2), k.size(2)
        mask = causal_lower_right(t, length) if 1 < t < length else None
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=t == length,
                                              enable_gqa=q.size(1) != k.size(1))


def snapshot(model, processed_tokens):
    counts = dict.fromkeys(("recent_slots", "exact_slots", "dense_slots", "hierarchical_slots",
                            "layer_group_count", "retained_slots"), 0)
    for block in model.transformer.h:
        row = block.attn.snap_cache.snapshot()
        if row["processed_tokens"] != processed_tokens:
            raise ValueError("SnapKV cache disagrees with evaluator token count")
        for key in counts:
            counts[key] += row[key]
    return _with_ratios(dict(counts, processed_tokens=processed_tokens))


def install_backend(evaluator, backend, provenance, settings=None):
    class RecordedLM(evaluator.LogKVLM):
        baseline_provenance = dict(provenance, model_backend=backend)

    if backend != "snapkv":
        evaluator.LogKVLM = RecordedLM
        return

    selection = dict(settings or {})
    chunk_size = selection.pop("prefill_chunk_size", None)

    class PyramidLM(RecordedLM):
        def _prefill_chunk(self):
            return self.log_kv_prefill_block if chunk_size is None else chunk_size

        def _set_eval_cache(self):
            if self._eval_cache_ready:
                self._reset_eval_cache()
                return
            # Use the actual preset, dtype, device and model window for sizing.
            super()._set_eval_cache()
            target = self.cache_budget["persistent_cache_bytes"]
            dense = self.cache_budget["dense_cache_bytes"]
            self.model.clear_kv_cache()
            if torch.device(self._device).type == "cuda":
                # Release unused allocator blocks from the sizing run before measuring requests.
                with torch.cuda.device(self._device):
                    torch.cuda.empty_cache()
            slot_bytes = dense // (self.config.n_layer * self.model.max_seq_length)
            # One total capacity for every layer, as in SnapKV.
            all_layer_slot_bytes = slot_bytes * self.config.n_layer
            capacity = min(self.model.max_seq_length, target // all_layer_slot_bytes)
            method = StreamingSnapKV(self.config.n_layer, capacity, **selection)
            allocated = 0
            for layer, block in enumerate(self.model.transformer.h):
                cache = ReferenceCache(layer, method)
                weight = block.attn.qkv.weight
                template = weight.new_empty(1, self.config.n_query_groups, 0, self.config.head_size)
                cache.allocate(template, capacity)
                block.attn.snap_cache = cache
                allocated += sum(t.untyped_storage().nbytes() for t in (cache._k, cache._v))
            if not 0 <= target - allocated < all_layer_slot_bytes:
                raise RuntimeError(f"SnapKV allocated {allocated} bytes for a {target}-byte target")
            self.cache_budget = dict(method="SnapKV", entries_per_layer=[capacity] * self.config.n_layer,
                                     selection=method.config(), prefill_chunk=self._prefill_chunk(),
                                     persistent_cache_bytes=allocated, reference_logkv_bytes=target,
                                     dense_cache_bytes=dense, allocated_fraction_of_dense=allocated / dense,
                                     reference_fraction_of_dense=target / dense, unused_reference_bytes=target - allocated)

        def _reset_eval_cache(self):
            for block in self.model.transformer.h:
                block.attn.snap_cache.rewind()

    evaluator.LogKVLM = PyramidLM
    evaluator.cache_compression_snapshot = snapshot
