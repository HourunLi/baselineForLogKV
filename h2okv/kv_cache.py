"""Per-request layer cache with KVCache-Factory's compression lifecycle.

The first forward of a request is the prefill: it attends over the complete,
uncompressed prompt, then the compressor keeps a subset of the post-RoPE K/V
entries (a real drop: the rest is freed). Later forwards (decode tokens or a
scored continuation) append to the compressed cache and attend to it. This is
the KVCache-Factory ``update_kv`` behaviour, reimplemented for Qwen3 without
monkey-patching Transformers.

With H2O decode eviction enabled the cache also keeps each entry's accumulated
attention score; every later forward computes attention explicitly, adds the
new probabilities to those scores and evicts back to the budget.

Storage granularity:
* ``kv_head`` (default): one entry per KV head; per-query-head scores are
  reduced over each GQA group before selection, so memory matches a dense
  GQA cache entry for entry.
* ``query_head``: KVCache-Factory's legacy layout; K/V are expanded to every
  query head before selection and stored at that width.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .compressor import attention_with_scores
from .qwen3 import repeat_kv

GRANULARITIES = ("kv_head", "query_head")
# How a scored continuation meets the cache: after the compressed context, or inside one prefill.
LOGLIKELIHOOD_MODES = ("compress_context", "single_forward")


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention of the last ``T`` positions over ``L`` cached entries (the last ``T`` are the queries' own)."""
    T, L = q.size(2), k.size(2)
    n_rep = q.size(1) // k.size(1)
    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    if T == 1:
        return F.scaled_dot_product_attention(q, k, v)
    if T == L:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    mask = torch.ones(T, L, dtype=torch.bool, device=q.device).tril(diagonal=L - T)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class CompressedKVCache:
    """One layer, one request. ``compressor=None`` keeps the full cache (FullKV reference)."""

    def __init__(self, layer_idx: int, compressor=None, granularity: str = "kv_head",
                 growth: int = 256):
        if granularity not in GRANULARITIES:
            raise ValueError(f"kv_cache_granularity must be one of {GRANULARITIES}")
        self.layer_idx = layer_idx
        self.compressor = compressor
        self.granularity = granularity
        self.growth = growth
        self.reset()

    def reset(self) -> None:
        self._k = self._v = None
        self.length = 0  # valid entries per stored head
        self.processed_tokens = 0  # tokens forwarded through this layer for the request
        self.num_recent = 0  # protected recent window + entries appended after prefill
        self.num_selected = 0  # older entries kept by the selection policy
        self.prefill_tokens = 0
        self.compressed = False
        self.n_kv_heads = None
        self.scores = None  # (B, stored heads, length) fp32, only for decode-time eviction
        self.evicted = 0

    @property
    def k(self) -> torch.Tensor | None:
        return None if self._k is None else self._k[:, :, :self.length]

    @property
    def v(self) -> torch.Tensor | None:
        return None if self._v is None else self._v[:, :, :self.length]

    @property
    def stored_heads(self) -> int:
        return 0 if self._k is None else self._k.size(1)

    def _store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        B, H, L, D = k.shape
        if self._k is None or self._k.size(2) < L:  # decode-time eviction reuses the buffer
            self._k = k.new_empty(B, H, L + self.growth, D)
            self._v = v.new_empty(B, H, L + self.growth, D)
        self._k[:, :, :L] = k
        self._v[:, :, :L] = v
        self.length = L

    def _append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        T = k.size(2)
        if self.length + T > self._k.size(2):
            capacity = self.length + T + max(self.growth, self.length // 4)
            for name in ("_k", "_v"):
                old = getattr(self, name)
                new = old.new_empty(*old.shape[:2], capacity, old.size(3))
                new[:, :, :self.length] = old[:, :, :self.length]
                setattr(self, name, new)
        self._k[:, :, self.length:self.length + T] = k
        self._v[:, :, self.length:self.length + T] = v
        self.length += T

    def _layout(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if self.granularity == "query_head":
            n_rep = q.size(1) // k.size(1)
            return repeat_kv(k, n_rep), repeat_kv(v, n_rep)
        return k, v

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.n_kv_heads is None:
            self.n_kv_heads = k.size(1)
        T = q.size(2)
        k, v = self._layout(q, k, v)
        if self._k is None:
            # Prefill: exact attention over the whole prompt, then compress what is kept.
            y = causal_attention(q, k, v)
            self.prefill_tokens = T
            self.processed_tokens = T
            if self.compressor is None:
                # Same accounting as the reference dense cache: every entry is "recent".
                kept_k, kept_v, selected, recent, scores = k, v, 0, T, None
            else:
                kept_k, kept_v, selected, recent, scores = self.compressor.compress(self.layer_idx, q, k, v)
            self.compressed = kept_k.size(2) < T
            self.num_selected, self.num_recent = selected, recent
            self.scores = scores
            self._store(kept_k, kept_v)
            return y
        self._append(k, v)
        self.processed_tokens += T
        self.num_recent += T
        if self.scores is None:
            return causal_attention(q, self.k, self.v)
        y, probs = attention_with_scores(q, self.k, self.v, self.compressor.gqa_score_agg)
        self.scores = torch.cat((self.scores, self.scores.new_zeros(*self.scores.shape[:2], T)), dim=-1) + probs
        kept = self.compressor.evict(self.scores, self.k, self.v)
        if kept is not None:
            keys, values, self.scores = kept
            self.evicted += self.length - keys.size(2)
            self._store(keys, values)
            self.num_recent = self.compressor.window_size
            self.num_selected = self.length - self.num_recent
        return y

    def snapshot(self) -> dict:
        """Logical entries at the KV-group level used by the shared metric definition."""
        heads, groups = self.stored_heads, self.n_kv_heads
        return dict(
            processed_tokens=self.processed_tokens,
            layer_group_count=groups,
            recent_slots=self.num_recent * heads,
            exact_slots=self.num_selected * heads,
            hierarchical_slots=0,
            retained_slots=self.length * heads,
            dense_slots=self.processed_tokens * groups,
            stored_heads=heads,
            entry_bytes=0 if self._k is None else 2 * self._k.size(3) * self._k.element_size(),
        )
