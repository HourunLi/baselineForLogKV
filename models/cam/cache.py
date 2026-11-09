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

    def append(self, keys, values):
        if keys.ndim != 4 or keys.shape != values.shape or keys.shape[2] < 1:
            raise ValueError("CaM appends K/V tokens with shape [batch, kv_heads, tokens, head_dim]")
        if self.keys is None:
            self.keys, self.values = keys, values
            self.scores = keys.new_zeros(keys.shape[:3], dtype=torch.float32)
            self._generator = torch.Generator(device=keys.device).manual_seed(self.seed)
        else:
            if self.keys.shape[2] > self.cache_size:
                raise RuntimeError("CaM.update must follow each forward's attention output")
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
        # V' = V + L V' with L[j, k] = selected_k / merge for 1 <= j - k <= merge (k a victim): CaM's chain of
        # merges, every victim passing on what earlier victims gave it. Solved in float32.
        size = excess + merge
        offset = torch.arange(size, device=scores.device)
        gap = offset[:, None] - offset[None, :]
        band = (gap >= 1) & (gap <= merge) & (offset[None, :] < excess)
        weights = torch.zeros(*selected.shape[:2], size, size, device=scores.device, dtype=torch.float32)
        weights[..., :, :excess] = selected[..., None, :] / merge
        system = torch.eye(size, device=scores.device) - weights * band
        merged = torch.linalg.solve_triangular(system, self.values[:, :, span].float(), upper=False,
                                               unitriangular=True)
        # CaM transfers V, but does not transfer the evicted cumulative scores.
        keep = torch.cat((torch.arange(start, device=scores.device),
                          torch.arange(start + excess, self.keys.shape[2], device=scores.device)))
        self.keys = self.keys.index_select(2, keep)
        self.values = self.values.index_select(2, keep)
        self.scores = self.scores.index_select(2, keep)
        # The entries right after the victims (now at start ... start + merge) received their merges.
        self.values[:, :, start:start + merge] = merged[:, :, excess:].to(self.values.dtype)
