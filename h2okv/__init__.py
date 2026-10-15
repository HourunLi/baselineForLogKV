"""H2O baseline for Qwen3 under SemanticLogKV's evaluation contract.

Self-contained: model, tokenizer, compressed cache and evaluator live here.
The H2O eviction rule is ported from KVCache-Factory (MIT).
"""

__version__ = "0.1.0"
