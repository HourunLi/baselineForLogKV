"""H2O (Heavy-Hitter Oracle) KV eviction for Qwen3, ported from KVCache-Factory.

Prompt compression follows https://github.com/Zefan-Cai/KVCache-Factory (MIT),
``pyramidkv/pyramidkv_utils.py`` ``H2OKVCluster.update_kv`` / ``_update_kv_kv_head``:
each older key is scored by the attention it receives, summed over every prompt
query (causal softmax); the top ``max_capacity_prompt - window_size`` older
entries plus the last ``window_size`` tokens are kept. KVCache-Factory's runner
uses ``window_size=8`` for H2O.

Two implementation choices for 32K prompts on Qwen3:
* the column sums are computed in query chunks, so the full L x L attention
  matrix is never materialised (KVCache-Factory builds it at once);
* the sums accumulate in fp32 rather than in the model dtype.

``decode_evict=True`` adds the decode-time policy of the original H2O real-drop
cache (FMInference/H2O): accumulated scores keep growing with each new query and,
once the cache exceeds the budget, the lowest-scoring entries outside the recent
window are evicted so the cache stays at ``max_capacity_prompt``. KVCache-Factory
(the default here) compresses only once, after prefill.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .options import GQA_SCORE_AGGS, H2OSettings


def reduce_group_scores(scores: torch.Tensor, groups: int, agg: str) -> torch.Tensor:
    """(B, H_kv, groups, ...) -> (B, H_kv, ...); query head h belongs to KV head h // groups."""
    if groups == 1:
        return scores.squeeze(2)
    if agg == "mean":
        return scores.mean(dim=2)
    if agg == "max":
        return scores.amax(dim=2)
    if agg == "sum":
        return scores.sum(dim=2)
    raise ValueError(f"gqa_score_agg must be one of {GQA_SCORE_AGGS}, got {agg!r}")


def accumulated_attention(query_states: torch.Tensor, key_states: torch.Tensor, agg: str,
                          chunk_size: int = 512) -> torch.Tensor:
    """Σ over causal prompt queries of softmax(q·k / sqrt(d)), per stored head: (B, H_store, L) in fp32.

    Grouped queries are matched to their KV head by reshaping (equal to a transient ``repeat_kv``).
    """
    B, H, L, D = query_states.shape
    H_kv = key_states.size(1)
    groups = H // H_kv
    keys_t = key_states.transpose(-1, -2).unsqueeze(2)  # (B, H_kv, 1, D, L)
    total = torch.zeros(B, H_kv, groups, L, dtype=torch.float32, device=query_states.device)
    positions = torch.arange(L, device=query_states.device)
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        queries = query_states[:, :, start:end].reshape(B, H_kv, groups, end - start, D)
        scores = torch.matmul(queries, keys_t[..., :end]) / math.sqrt(D)
        future = positions[:end].view(1, end) > positions[start:end].view(end - start, 1)
        scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
        total[..., :end] += F.softmax(scores, dim=-1, dtype=torch.float32).sum(dim=-2)
    return reduce_group_scores(total, groups, agg)


def attention_with_scores(query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor,
                          agg: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit attention of the last ``T`` positions over the cache, also returning Σ_queries probabilities."""
    B, H, T, D = query_states.shape
    H_kv, L = key_states.size(1), key_states.size(2)
    groups = H // H_kv
    queries = query_states.reshape(B, H_kv, groups, T, D)
    scores = torch.matmul(queries, key_states.transpose(-1, -2).unsqueeze(2)) / math.sqrt(D)
    if T > 1:
        future = torch.ones(T, L, dtype=torch.bool, device=scores.device).tril(diagonal=L - T).logical_not()
        scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32)
    output = torch.matmul(probs.to(value_states.dtype), value_states.unsqueeze(2)).reshape(B, H, T, D)
    return output, reduce_group_scores(probs.sum(dim=-2), groups, agg)


def keep_heavy_hitters(scores: torch.Tensor, capacity: int, window_size: int, *tensors: torch.Tensor):
    """Keep the top-``capacity`` older entries per head (position order) plus the last ``window_size``."""
    length = scores.size(-1)
    older = scores[..., :length - window_size]
    indices = older.topk(capacity, dim=-1).indices.sort(dim=-1).values
    recent = torch.arange(length - window_size, length, device=scores.device).expand(*indices.shape[:-1], -1)
    keep = torch.cat((indices, recent), dim=-1)
    out = []
    for tensor in tensors:
        index = keep if tensor.dim() == 3 else keep.unsqueeze(-1).expand(-1, -1, -1, tensor.size(-1))
        out.append(tensor.gather(dim=2, index=index))
    return out


class H2OCompressor(H2OSettings):
    """H2O settings (validated in :class:`H2OSettings`) plus prompt compression and decode eviction."""

    def compress(self, layer_idx: int, query_states: torch.Tensor, key_states: torch.Tensor,
                 value_states: torch.Tensor):
        """Return ``(keys, values, num_selected, num_recent, scores)``; ``scores`` only with decode eviction."""
        q_len = key_states.size(2)
        if q_len < self.max_capacity_prompt:
            recent = min(q_len, self.window_size)
            scores = (accumulated_attention(query_states, key_states, self.gqa_score_agg, self.score_chunk_size)
                      if self.decode_evict else None)
            return key_states, value_states, q_len - recent, recent, scores
        scores = accumulated_attention(query_states, key_states, self.gqa_score_agg, self.score_chunk_size)
        capacity = self.max_capacity_prompt - self.window_size
        keys, values, kept_scores = keep_heavy_hitters(scores, capacity, self.window_size,
                                                       key_states, value_states, scores)
        return keys, values, capacity, self.window_size, kept_scores if self.decode_evict else None

    def evict(self, scores: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        """Decode-time H2O: shrink back to the budget, keeping the recent window and the heaviest older entries."""
        if scores.size(-1) <= self.max_capacity_prompt:
            return None
        return keep_heavy_hitters(scores, self.max_capacity_prompt - self.window_size, self.window_size,
                                  keys, values, scores)
