"""CaM on the already-loaded niahLogKV LitGPT Qwen3 weights and YaRN tables."""
from types import MethodType

import torch
from torch import nn

from models._vendor.niahlogkv.litgpt.model import CausalSelfAttention, apply_rope
from models._vendor.niahlogkv.litgpt.kv_compression import _with_ratios
from .cache import CaMCache
from .grouped import attend_and_merge


class NiahCaM(nn.Module):
    def __init__(self, gpt, capacity, merge_interval, *, sinks=32, merge=16, seed=42):
        super().__init__()
        self.gpt = gpt
        self.config = cfg = gpt.config
        if (cfg.rope_n_elem != cfg.head_size or cfg.rope_interleave or cfg.rope_indices is not None
                or cfg.sliding_window_size or not cfg.norm_qk or cfg.norm_qk_type != "default"
                or cfg.attention_logit_softcapping or merge_interval < 1):
            raise ValueError("CaM requires full-attention Qwen3 with standard QK norm and full-width RoPE")
        self.max_seq_length = gpt.max_seq_length
        self.caches = []

        def forward(attn, x, *args, **kwargs):
            batch, tokens, _ = x.shape
            dim, groups = cfg.head_size, cfg.n_query_groups
            widths = (dim * cfg.n_head, dim * groups, dim * groups)
            q, k, v = (t.reshape(batch, tokens, -1, dim).transpose(1, 2)
                       for t in attn.qkv(x).split(widths, dim=-1))
            q, k = attn.norm_q(q), attn.norm_k(k)
            scale = (cfg.attention_scores_scalar or dim) ** -0.5 * attn.mscale ** 2
            cache = attn.cam_cache
            outputs = []
            begin = 0
            while begin < tokens:
                # Before the cache fills no query can observe an eviction. Evaluate those rows together.
                free = capacity - min(cache.processed_tokens, capacity)
                if free == 0 and merge_interval == 1 and tokens - begin > 1:
                    cos, sin = gpt.cos[None, :capacity + 1], gpt.sin[None, :capacity + 1]
                    queries = apply_rope(q[:, :, begin:], cos[:, -1:], sin[:, -1:])
                    for i in cache.strict_windows(k[:, :, begin:], v[:, :, begin:]):
                        keys = apply_rope(cache.keys, cos, sin)
                        outputs.append(attend_and_merge(queries[:, :, i:i + 1], keys, cache, scale,
                                                        sliding=True))
                    break
                end = min(tokens, begin + (free if free > 0 else merge_interval))
                cache.append(k[:, :, begin:end], v[:, :, begin:end])
                length = cache.keys.shape[2]
                cos, sin = gpt.cos[None, :length], gpt.sin[None, :length]
                first = length - (end - begin)
                query = apply_rope(q[:, :, begin:end], cos[:, first:], sin[:, first:])
                keys = apply_rope(cache.keys, cos, sin)
                outputs.append(attend_and_merge(query, keys, cache, scale))
                begin = end
            return attn.proj(torch.cat(outputs, dim=1))

        for index, block in enumerate(gpt.transformer.h):
            if type(block.attn) is not CausalSelfAttention:
                raise TypeError("Expected LitGPT Qwen3 attention")
            cache = CaMCache(sinks, capacity - sinks, merge, seed + index)
            cache.allocate(block.attn.qkv.weight, cfg.n_query_groups, cfg.head_size)
            self.caches.append(cache)
            block.attn.cam_cache = cache
            block.attn.forward = MethodType(forward, block.attn)
        self.reset_cache()

    def reset_cache(self):
        for cache in self.caches:
            cache.reset()
        self.processed = 0

    @torch.no_grad()
    def forward(self, idx, input_pos, input_pos_maxp1=None, lm_head_start=None):
        end = self.processed + idx.size(1)
        if idx.size(0) != 1 or end > self.max_seq_length or not torch.equal(
                input_pos, torch.arange(self.processed, end, device=idx.device)):
            raise ValueError("CaM expects one contiguous, unpadded request within the pass window")
        # Attention uses its own cache slots. Keep the original embedding, residual, MLP and output paths.
        logits = self.gpt(idx, lm_head_start=lm_head_start)
        self.processed = end
        return logits

    def storage_bytes(self):
        return sum(t.untyped_storage().nbytes() for cache in self.caches for t in cache.storage)

    def snapshot(self, tokens):
        if tokens != self.processed or any(c.processed_tokens != tokens for c in self.caches):
            raise ValueError("Forwarded token count differs between request and layer caches")
        groups = self.config.n_layer * self.config.n_query_groups
        retained = self.caches[0].keys.size(2) * groups
        sinks = min(tokens, self.caches[0].start_size) * groups
        return _with_ratios(dict(processed_tokens=tokens, layer_group_count=groups,
                                 recent_slots=retained - sinks, exact_slots=sinks, hierarchical_slots=0,
                                 retained_slots=retained, dense_slots=tokens * groups))


def install_adapter(evaluator, nominal, budget_mode, merge_interval, prefill_chunk):
    """Retain the reference tokenizer, checkpoint load, request loop, metrics and task registry."""
    from .niah_budget import cache_budget

    class CaMLM(evaluator.LogKVLM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Use the same architecture, YaRN window, dtype and LogKV settings as the control.
            super()._set_eval_cache()
            reference_bytes = self.cache_budget["persistent_cache_bytes"]
            self.model.clear_kv_cache()
            if next(self.model.parameters()).is_cuda:
                # Release unused sizing buffers before the benchmark samples allocator-reserved peaks.
                torch.cuda.empty_cache()
            cfg = self.config
            self.cache_budget = cache_budget(budget_mode, nominal, self.model.max_seq_length, cfg.n_layer,
                                            cfg.n_query_groups, cfg.head_size,
                                            next(self.model.parameters()).element_size(), reference_bytes)
            self.model = NiahCaM(self.model, self.cache_budget["capacity_tokens"], merge_interval).eval()
            allocated = self.model.storage_bytes()
            if allocated != self.cache_budget["persistent_cache_bytes"]:
                raise RuntimeError(f"Allocated CaM storage {allocated} disagrees with budget {self.cache_budget}")
            self.cache_budget.update(merge_interval=merge_interval, prefill_chunk=prefill_chunk,
                                     sink_tokens=32, merge_size=16, seed=42)
            self._stream_score_logits = True
            if evaluator._is_main():
                print(f"CaM persistent cache budget: {self.cache_budget}")

        def _prefill_chunk(self):
            return prefill_chunk

        def _set_eval_cache(self):
            self.model.reset_cache()

        _reset_eval_cache = _set_eval_cache

    evaluator.LogKVLM = CaMLM
    evaluator.cache_compression_snapshot = lambda model, tokens: model.snapshot(tokens)
