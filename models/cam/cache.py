"""Online CaM cache, following generate_task/streaming_llm/kv_cache_cam.py."""

import torch


class CaMCache:
    """One layer's unrotated normalized K, merged V, and cumulative attention.

    The caller appends one token, computes its attention output, then calls
    ``update``. GQA callers average probabilities over the Q heads that share
    each KV head before updating; each KV head gets one Bernoulli draw.
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
        if keys.ndim != 4 or keys.shape != values.shape or keys.shape[2] != 1:
            raise ValueError("CaM appends one K/V token with shape [batch, kv_heads, 1, head_dim]")
        if self.keys is None:
            self.keys, self.values = keys, values
            self.scores = keys.new_zeros(keys.shape[:3], dtype=torch.float32)
            self._generator = torch.Generator(device=keys.device).manual_seed(self.seed)
        else:
            if self.keys.shape[2] > self.cache_size:
                raise RuntimeError("CaM.update must follow each token's attention output")
            self.keys = torch.cat((self.keys, keys), dim=2)
            self.values = torch.cat((self.values, values), dim=2)
            self.scores = torch.cat((self.scores, self.scores.new_zeros(keys.shape[:3])), dim=2)
        self.processed_tokens += 1

    def update(self, attention_scores):
        """Accumulate current attention, then merge and remove the oldest recent KV."""
        if self.scores is None or attention_scores.shape != self.scores.shape:
            raise ValueError("attention_scores must match [batch, kv_heads, cached_tokens]")
        self.scores.add_(attention_scores.float())
        if self.keys.shape[2] <= self.cache_size:
            return

        victim = self.start_size
        targets = slice(victim + 1, victim + 1 + self.merge_size)
        probability = self.scores[..., victim] / self.scores[..., targets].mean(dim=-1)
        # The upstream maps 0/0 to zero, and clamps positive infinity to one.
        probability = probability.nan_to_num(nan=0.0).clamp_(0, 1)
        selected = torch.bernoulli(probability, generator=self._generator)
        contribution = self.values[:, :, victim] * selected[..., None] / self.merge_size
        self.values[:, :, targets].add_(contribution.to(self.values.dtype).unsqueeze(2))
        self.keys = torch.cat((self.keys[:, :, :victim], self.keys[:, :, victim + 1:]), dim=2)
        self.values = torch.cat((self.values[:, :, :victim], self.values[:, :, victim + 1:]), dim=2)
        # CaM transfers V, but does not transfer the evicted cumulative score.
        self.scores = torch.cat((self.scores[..., :victim], self.scores[..., victim + 1:]), dim=-1)
