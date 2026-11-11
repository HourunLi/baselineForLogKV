"""Method choices and SnapKV settings, without torch.

The CLI resolves and validates every method option here, so ``--print-config``
shows the final settings and the per-layer budget without importing torch;
:class:`snapkv.compressor.SnapKVCompressor` adds the tensor operations on top of
:class:`SnapKVSettings`.
"""

from __future__ import annotations

METHODS = ("snapkv", "fullkv")
POOLINGS = ("avgpool", "maxpool")
GQA_SCORE_AGGS = ("mean", "max", "sum")
GRANULARITIES = ("kv_head", "query_head")
# How a scored continuation meets the cache: after the compressed context, or inside one prefill.
LOGLIKELIHOOD_MODES = ("compress_context", "single_forward")


class SnapKVSettings:
    """SnapKV keeps the same budget in every layer (``SnapKVCluster`` with one ``max_capacity_prompt``)."""

    name = "snapkv"

    def __init__(self, num_hidden_layers: int, max_capacity_prompt: int = 2048, window_size: int = 8,
                 kernel_size: int = 7, pooling: str = "maxpool", gqa_score_agg: str = "mean"):
        if window_size < 1 or max_capacity_prompt - window_size <= 0:
            raise ValueError("SnapKV requires 0 < window_size < max_capacity_prompt")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number so pooling keeps the sequence length")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        if gqa_score_agg not in GQA_SCORE_AGGS:
            raise ValueError(f"gqa_score_agg must be one of {GQA_SCORE_AGGS}")
        self.num_hidden_layers = num_hidden_layers
        self.max_capacity_prompt = max_capacity_prompt
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.gqa_score_agg = gqa_score_agg

    def config(self) -> dict:
        return dict(method=self.name, max_capacity_prompt=self.max_capacity_prompt, window_size=self.window_size,
                    kernel_size=self.kernel_size, pooling=self.pooling, gqa_score_agg=self.gqa_score_agg)

    def layer_capacity(self, layer_idx: int, q_len: int) -> int | None:
        """Older entries kept in ``layer_idx`` for a ``q_len``-token prompt; ``None`` keeps everything.

        ``SnapKVCluster.update_kv``: a prompt shorter than ``max_capacity_prompt`` is kept whole; otherwise
        every layer keeps the top ``max_capacity_prompt - window_size`` older entries plus the window.
        """
        if q_len < self.max_capacity_prompt:
            return None
        return self.max_capacity_prompt - self.window_size

    def budget_schedule(self, q_len: int) -> list[int]:
        """Total entries per layer (selected + window) for a prompt of ``q_len`` tokens."""
        out = []
        for layer in range(self.num_hidden_layers):
            capacity = self.layer_capacity(layer, q_len)
            out.append(q_len if capacity is None else capacity + self.window_size)
        return out
