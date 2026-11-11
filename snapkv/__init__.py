"""SnapKV baseline for Qwen3 under SemanticLogKV's evaluation contract.

Self-contained: model, tokenizer, compressed cache and evaluator live here.
The SnapKV selection rule is ported from KVCache-Factory (MIT), which keeps the
original SnapKV implementation (FasterDecoding/SnapKV, Apache-2.0).
"""

__version__ = "0.1.0"
