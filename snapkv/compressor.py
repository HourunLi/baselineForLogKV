"""SnapKV prompt-cache selection for Qwen3, ported from KVCache-Factory.

Sources:
* https://github.com/Zefan-Cai/KVCache-Factory (MIT) @ 68cd9551a63fedf362ddb0edb008badd04c24996,
  ``pyramidkv/pyramidkv_utils.py``: ``SnapKVCluster.update_kv`` / ``_update_kv_kv_head`` with the helpers
  ``_grouped_window_attn_cache``, ``_reduce_group_scores`` and ``_select_topk_kv``; runner defaults of
  ``run_longbench.py`` (``window_size=8``, ``kernel_size=7``, ``pooling='maxpool'``).
* The algorithm comes from https://github.com/FasterDecoding/SnapKV (Apache-2.0) @
  e216ddc84c5bd210378cbdbbba12ba02102aa640, ``snapkv/monkeypatch/snapkv_utils.py`` ``SnapKVCluster``, whose
  ``update_kv`` the KVCache-Factory ``query_head`` path keeps line for line.

Per layer, after the prefill has attended over the complete prompt, the last ``window_size`` prompt queries
(the observation window) score every older key: causal softmax inside the window, summed over the window
queries, reduced over each GQA group at ``kv_head`` granularity, then 1-D pooled. Every layer keeps the same
number of entries: the top ``max_capacity_prompt - window_size`` older entries plus the window. Prompts
shorter than ``max_capacity_prompt`` are kept whole.

Selected entries are stored in position order (the upstream gathers them in score order); attention over a
cache does not depend on the order of its entries, so the kept set and the outputs are the same.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .options import GQA_SCORE_AGGS, POOLINGS, SnapKVSettings


def reduce_group_scores(scores: torch.Tensor, groups: int, agg: str) -> torch.Tensor:
    """(B, H_q, ...) -> (B, H_kv, ...); query head h belongs to KV head h // groups."""
    if groups == 1:
        return scores
    grouped = scores.reshape(scores.size(0), scores.size(1) // groups, groups, *scores.shape[2:])
    if agg == "mean":
        return grouped.mean(dim=2)
    if agg == "max":
        return grouped.amax(dim=2)
    if agg == "sum":
        return grouped.sum(dim=2)
    raise ValueError(f"gqa_score_agg must be one of {GQA_SCORE_AGGS}, got {agg!r}")


def observation_window_scores(query_states: torch.Tensor, key_states: torch.Tensor, window_size: int,
                              kernel_size: int, pooling: str, agg: str) -> torch.Tensor:
    """Scores for the ``L - window_size`` older keys at the cache's head granularity.

    Grouped queries are matched to their KV head by reshaping, which equals
    KVCache-Factory's transient ``repeat_kv`` without materialising it.
    """
    B, H, _, D = query_states.shape
    H_kv, L = key_states.size(1), key_states.size(2)
    groups = H // H_kv
    window = query_states[:, :, -window_size:].reshape(B, H_kv, groups, window_size, D)
    attn = torch.matmul(window, key_states.unsqueeze(2).transpose(-1, -2)) / math.sqrt(D)
    attn = attn.reshape(B, H, window_size, L)
    mask = torch.full((window_size, window_size), torch.finfo(attn.dtype).min, device=attn.device)
    cond = torch.arange(window_size, device=attn.device)
    mask.masked_fill_(cond < (cond + 1).view(window_size, 1), 0)
    attn[:, :, :, -window_size:] += mask
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
    scores = reduce_group_scores(attn[:, :, :, :-window_size].sum(dim=-2), groups, agg)
    if pooling == "avgpool":
        return F.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    if pooling == "maxpool":
        return F.max_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")


def keep_topk(key_states: torch.Tensor, value_states: torch.Tensor, scores: torch.Tensor,
              capacity: int, window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the top-``capacity`` older entries per head (position order) plus the window."""
    indices = scores.topk(capacity, dim=-1).indices.sort(dim=-1).values
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, key_states.size(-1))
    past_k = key_states[:, :, :-window_size].gather(dim=2, index=indices)
    past_v = value_states[:, :, :-window_size].gather(dim=2, index=indices)
    return (torch.cat((past_k, key_states[:, :, -window_size:]), dim=2),
            torch.cat((past_v, value_states[:, :, -window_size:]), dim=2))


class SnapKVCompressor(SnapKVSettings):
    """SnapKV settings and the uniform layer budget (see :class:`SnapKVSettings`) plus prompt compression."""

    def compress(self, layer_idx: int, query_states: torch.Tensor, key_states: torch.Tensor,
                 value_states: torch.Tensor):
        """Return ``(keys, values, num_selected, num_recent)`` for the prefilled prompt."""
        q_len = key_states.size(2)
        capacity = self.layer_capacity(layer_idx, q_len)
        if capacity is None:
            recent = min(q_len, self.window_size)
            return key_states, value_states, q_len - recent, recent
        scores = observation_window_scores(query_states, key_states, self.window_size,
                                           self.kernel_size, self.pooling, self.gqa_score_agg)
        keys, values = keep_topk(key_states, value_states, scores, capacity, self.window_size)
        return keys, values, capacity, self.window_size
