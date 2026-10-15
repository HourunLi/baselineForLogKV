"""ChunkKV settings and validation, without model or tensor imports."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ChunkKVSettings:
    chunk_size: int = 10
    window_size: int = 64
    # Prompt KV entries per layer and KV head, window included (the paper's L_max).
    budget: int | None = None
    # Alternative to ``budget``: ceil(keep_ratio * prompt length) entries.
    keep_ratio: float | None = None
    reuse_layers: int = 1

    def __post_init__(self):
        if self.chunk_size < 1 or self.window_size < 1 or self.reuse_layers < 1:
            raise ValueError("chunk_size, window_size and reuse_layers must be positive")
        if (self.budget is None) == (self.keep_ratio is None):
            raise ValueError("Set exactly one of budget and keep_ratio")
        if self.budget is not None and self.budget < self.window_size + self.chunk_size:
            raise ValueError("budget must hold the observation window and at least one chunk")
        if self.keep_ratio is not None and not 0 < self.keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")

    def prompt_budget(self, prompt_len: int) -> int:
        if self.budget is not None:
            return self.budget
        return math.ceil(self.keep_ratio * prompt_len)

    def compresses(self, prompt_len: int) -> bool:
        return prompt_len > self.prompt_budget(prompt_len)

    def scores_layer(self, layer_idx: int) -> bool:
        """Algorithm 2: only the first layer of each reuse group selects chunks."""
        return layer_idx % self.reuse_layers == 0

    def as_dict(self) -> dict:
        return asdict(self)

