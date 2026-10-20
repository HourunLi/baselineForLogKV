"""Fixed-stride CSA adaptation settings; deliberately independent of torch."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HybridCSASettings:
    full_layer_interval: int = 3
    compress_ratio: int = 4
    window_size: int = 128
    index_topk: int = 512
    index_dim: int = 64
    index_heads: int = 4
    query_chunk_size: int = 64
    prefill_chunk_size: int = 512

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.compress_ratio != 4:
            raise ValueError("This CSA baseline fixes compress_ratio=4")
        if self.index_dim % 2:
            raise ValueError("index_dim must be even for RoPE")

    def full_layers(self, n_layers: int) -> tuple[int, ...]:
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        return tuple(sorted(set(range(0, n_layers, self.full_layer_interval)) | {n_layers - 1}))

    def as_dict(self) -> dict:
        return asdict(self)
