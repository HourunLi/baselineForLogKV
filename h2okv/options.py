"""Method choices and H2O settings, without torch.

The CLI resolves and validates every method option here, so ``--print-config``
shows the final settings without importing torch; :class:`h2okv.compressor.H2OCompressor`
adds the tensor operations on top of :class:`H2OSettings`.
"""

from __future__ import annotations

METHODS = ("h2o", "fullkv")
GQA_SCORE_AGGS = ("mean", "max", "sum")
GRANULARITIES = ("kv_head", "query_head")
# How a scored continuation meets the cache: after the compressed context, or inside one prefill.
LOGLIKELIHOOD_MODES = ("compress_context", "single_forward")


class H2OSettings:
    name = "h2o"

    def __init__(self, max_capacity_prompt: int = 2048, window_size: int = 8, gqa_score_agg: str = "mean",
                 decode_evict: bool = False, score_chunk_size: int = 512, streaming: bool = False):
        if window_size < 1 or max_capacity_prompt - window_size <= 0:
            raise ValueError("H2O requires 0 < window_size < max_capacity_prompt")
        if gqa_score_agg not in GQA_SCORE_AGGS:
            raise ValueError(f"gqa_score_agg must be one of {GQA_SCORE_AGGS}")
        if score_chunk_size < 1:
            raise ValueError("score_chunk_size must be positive")
        self.max_capacity_prompt = max_capacity_prompt
        self.window_size = window_size
        self.gqa_score_agg = gqa_score_agg
        self.decode_evict = decode_evict
        self.score_chunk_size = score_chunk_size
        # Streaming H2O evicts after every forward, prompt chunks included, so the cache never exceeds the budget.
        self.streaming = bool(streaming)

    def config(self) -> dict:
        return dict(method=self.name, max_capacity_prompt=self.max_capacity_prompt, window_size=self.window_size,
                    gqa_score_agg=self.gqa_score_agg, decode_evict=self.decode_evict or self.streaming,
                    score_chunk_size=self.score_chunk_size, streaming=self.streaming)

    def entry_bytes(self, head_dim: int, element_size: int) -> int:
        """Persistent bytes of one kept entry in one stored head: K, V and (when kept) the fp32 score."""
        return 2 * head_dim * element_size + (4 if self.decode_evict or self.streaming else 0)

    def budget_schedule(self, q_len: int, num_hidden_layers: int) -> list[int]:
        """H2O uses the same budget in every layer."""
        return [min(q_len, self.max_capacity_prompt)] * num_hidden_layers
