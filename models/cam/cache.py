"""Online CaM cache, following generate_task/streaming_llm/kv_cache_cam.py, one forward at a time."""

import torch


class CaMCache:
    """One layer's unrotated normalized K, merged V, and cumulative attention.

    The caller appends a forward's tokens, computes their attention output, then calls ``update`` with the
    attention every entry received from those queries (GQA callers average the probabilities over the Q heads
    that share each KV head, then sum over queries). ``update`` evicts the entries beyond ``cache_size``.

    One token per forward is the official online path exactly: append, attend, add the scores, evict index
    ``start_size`` after merging its V into the next ``merge_size`` entries with probability
    ``clip(A[e] / mean(A[e+1:e+1+M]), 0, 1)`` (one Bernoulli draw per KV head). A forward of ``T`` tokens (a
    prompt chunk) evicts its ``e`` excess entries the same way, oldest first, with the scores the whole forward
    produced; each victim's V already carries the merges of the victims before it, which is solved exactly as
    one unit lower-triangular system instead of ``e`` sequential steps.
    """

    def __init__(self, start_size, recent_size, merge_size=16, seed=0):
        if start_size < 0 or not 1 <= merge_size <= recent_size:
            raise ValueError("CaM requires start_size >= 0 and recent_size >= merge_size >= 1")
        self.start_size = start_size
        self.recent_size = recent_size
        self.merge_size = merge_size
        self.cache_size = start_size + recent_size
        self.seed = seed
        self.storage = None
        self.reset()

    def reset(self, seed=None):
        """Start a request; replaying a seed replays this layer's private RNG."""
        if seed is not None:
            self.seed = seed
        self.keys = None
        self.values = None
        self.scores = None
        self.processed_tokens = 0
        self._generator = None

    def allocate(self, weight, groups, head_size):
        """Reserve the whole batch-one budget once; reset only releases active views."""
        shape = (1, groups, self.cache_size, head_size)
        self.storage = (weight.new_empty(shape), weight.new_empty(shape),
                        weight.new_empty(shape[:3], dtype=torch.float32))
        self.reset()

    def append(self, keys, values):
        if keys.ndim != 4 or keys.shape != values.shape or keys.shape[2] < 1:
            raise ValueError("CaM appends K/V tokens with shape [batch, kv_heads, tokens, head_dim]")
        used = 0 if self.keys is None else self.keys.shape[2]
        end = used + keys.shape[2]
        if used > self.cache_size:
            raise RuntimeError("CaM.update must follow each forward's attention output")
        if self._generator is None:
            self._generator = torch.Generator(device=keys.device).manual_seed(self.seed)
        if self.storage is not None and end <= self.cache_size:
            raw_k, raw_v, scores = self.storage
            raw_k[:, :, used:end].copy_(keys)
            raw_v[:, :, used:end].copy_(values)
            scores[:, :, used:end].zero_()
            self.keys, self.values, self.scores = (t[:, :, :end] for t in self.storage)
        elif not used:
            self.keys = keys
            # An oversized first step may merge in-place before compaction.
            self.values = values.clone() if end > self.cache_size else values
            self.scores = keys.new_zeros(keys.shape[:3], dtype=torch.float32)
        else:
            self.keys = torch.cat((self.keys, keys), dim=2)
            self.values = torch.cat((self.values, values), dim=2)
            self.scores = torch.cat((self.scores, self.scores.new_zeros(keys.shape[:3])), dim=2)
        self.processed_tokens += keys.shape[2]

    def update(self, attention_scores):
        """Accumulate this forward's attention, then merge and remove the oldest recent entries over budget."""
        if self.scores is None or attention_scores.shape != self.scores.shape:
            raise ValueError("attention_scores must match [batch, kv_heads, cached_tokens]")
        self.scores.add_(attention_scores.float())
        excess = self.keys.shape[2] - self.cache_size
        if excess <= 0:
            return

        start, merge = self.start_size, self.merge_size
        span = slice(start, start + excess + merge)  # the victims and every entry they merge into
        scores = self.scores[..., span]
        # Victim k (cache index start + k) merges into the next `merge` entries, start + k + 1 ... + merge.
        targets = scores[..., 1:].unfold(-1, merge, 1)[..., :excess, :]
        probability = scores[..., :excess] / targets.mean(dim=-1)
        # The upstream maps 0/0 to zero, and clamps positive infinity to one.
        probability = probability.nan_to_num(nan=0.0).clamp_(0, 1)
        selected = torch.bernoulli(probability, generator=self._generator)
        if excess == 1:
            # The streaming paper path has a single victim. No triangular system is needed.
            contribution = self.values[:, :, start:start + 1].float() * selected[..., None] / merge
            self.values[:, :, start + 1:start + 1 + merge].add_(contribution)
        else:
            # Propagate the banded recurrence in small tiles. The fp32 carry contains the
            # contributions to the next tile, including victims that will themselves merge.
            carry = self.values[:, :, span].float().clone()
            for begin in range(0, excess, 64):
                n = min(64, excess - begin)
                offsets = torch.arange(n + merge, device=scores.device)
                gap = offsets[:, None] - offsets[:n]
                band = ((gap > 0) & (gap <= merge)) * (selected[..., None, begin:begin + n] / merge)
                system = torch.eye(n, device=scores.device) - band[..., :n, :]
                victims = torch.linalg.solve_triangular(system, carry[:, :, begin:begin + n],
                                                        upper=False, unitriangular=True)
                carry[:, :, begin + n:begin + n + merge].add_(band[..., n:, :] @ victims)
            self.values[:, :, start + excess:start + excess + merge] = carry[:, :, excess:]

        def compact(tensor, out):
            if out is None:
                return torch.cat((tensor[:, :, :start], tensor[:, :, start + excess:]), dim=2)
            out[:, :, :start].copy_(tensor[:, :, :start])
            out[:, :, start:].copy_(tensor[:, :, start + excess:])
            return out

        buffers = self.storage if self.storage is not None else (None, None, None)
        self.keys, self.values, self.scores = (
            compact(t, out) for t, out in zip((self.keys, self.values, self.scores), buffers)
        )
