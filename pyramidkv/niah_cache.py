"""Streaming PyramidKV at the reference LM's measured persistent-cache budget."""
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from .compressor import PyramidKVCompressor
from .kv_cache import CompressedKVCache
from ._vendor.niahlogkv.litgpt.kv_compression import _with_ratios


class MatchedPyramid(PyramidKVCompressor):
    def __init__(self, layers, window, slots, recent=64):
        # One integer slot holds all KV heads in one layer. Keep the pyramid
        # symmetric when its widest layer would exceed the physical window.
        if not layers * recent <= slots <= layers * window:
            raise ValueError(f"Cannot distribute {slots} slots over {layers} layers of [{recent}, {window}]")
        mean, remainder = divmod(slots, layers)
        super().__init__(layers, max(mean, recent + 1), window_size=recent, streaming=True)
        older = mean - recent
        spread = min(older - older // self.beta, window - mean)
        # Round each slope toward zero, so opposite layers cancel exactly.
        self.capacities = []
        for layer in range(layers):
            numerator = spread * (layers - 1 - 2 * layer)
            offset = abs(numerator) // max(1, layers - 1)
            self.capacities.append(mean + (offset if numerator >= 0 else -offset))
        for layer, capacity in enumerate(self.capacities):
            if remainder and capacity < window:
                self.capacities[layer] += 1
                remainder -= 1

    def stream_capacity(self, layer_idx):
        return self.capacities[layer_idx] - self.window_size


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
        row = block.attn.pyramid_cache.snapshot()
        if row["processed_tokens"] != processed_tokens:
            raise ValueError("PyramidKV cache disagrees with evaluator token count")
        for key in counts:
            counts[key] += row[key]
    return _with_ratios(dict(counts, processed_tokens=processed_tokens))


def install_backend(evaluator, backend, provenance):
    class RecordedLM(evaluator.LogKVLM):
        baseline_provenance = dict(provenance, model_backend=backend)

    if backend != "pyramidkv":
        evaluator.LogKVLM = RecordedLM
        return

    class PyramidLM(RecordedLM):
        def _prefill_chunk(self):
            return self.log_kv_prefill_block

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
            method = MatchedPyramid(self.config.n_layer, self.model.max_seq_length, target // slot_bytes)
            allocated = 0
            for layer, block in enumerate(self.model.transformer.h):
                cache = ReferenceCache(layer, method)
                weight = block.attn.qkv.weight
                template = weight.new_empty(1, self.config.n_query_groups, 0, self.config.head_size)
                cache.allocate(template, method.capacities[layer])
                block.attn.pyramid_cache = cache
                allocated += sum(t.untyped_storage().nbytes() for t in (cache._k, cache._v))
            if not 0 <= target - allocated < slot_bytes:
                raise RuntimeError(f"PyramidKV allocated {allocated} bytes for a {target}-byte target")
            self.cache_budget = dict(method="PyramidKV", entries_per_layer=method.capacities,
                                     selection=method.config(), prefill_chunk=self._prefill_chunk(),
                                     persistent_cache_bytes=allocated, reference_logkv_bytes=target,
                                     dense_cache_bytes=dense, allocated_fraction_of_dense=allocated / dense,
                                     reference_fraction_of_dense=target / dense, unused_reference_bytes=target - allocated)

        def _reset_eval_cache(self):
            for block in self.model.transformer.h:
                block.attn.pyramid_cache.rewind()

    evaluator.LogKVLM = PyramidLM
    evaluator.cache_compression_snapshot = snapshot
