"""PyramidKV prompt-cache selection for Qwen3, ported from KVCache-Factory.

Source: https://github.com/Zefan-Cai/KVCache-Factory (MIT), ``pyramidkv/pyramidkv_utils.py``
``PyramidKVCluster.update_kv`` / ``_update_kv_kv_head`` and the runner defaults
(``window_size=8``, ``kernel_size=7``, ``pooling='maxpool'``, ``beta=20``).

Per layer, the last ``window_size`` prompt queries score every older key
(SnapKV observation window: causal softmax, summed over the window, 1-D
pooling); the top-``capacity`` older entries plus the window are kept. The
capacity follows a pyramid: lower layers keep more, upper layers less, with the
average close to ``max_capacity_prompt`` entries per layer (window included).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .options import GQA_SCORE_AGGS, POOLINGS, PyramidKVSettings


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


class PyramidKVCompressor(PyramidKVSettings):
    """PyramidKV settings and layer budgets (see :class:`PyramidKVSettings`) plus prompt compression."""

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
