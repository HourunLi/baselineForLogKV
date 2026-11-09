"""Method switch for this baseline: ``pyramidkv`` or the ``fullkv`` reference.

KVCache-Factory selects methods with ``--method`` and sets the budget with
``--max_capacity_prompts``; the same knobs are exposed here with that
project's runner defaults. ``fullkv`` runs the identical model, cache code and
evaluator without dropping entries, which checks the pipeline itself.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from .options import GQA_SCORE_AGGS, GRANULARITIES, METHODS, POOLINGS, PyramidKVSettings

DEFAULT_METHOD = "pyramidkv"


def add_method_args(parser) -> None:
    group = parser.add_argument_group("PyramidKV (KVCache-Factory names and runner defaults)")
    group.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
    group.add_argument("--max-capacity-prompt", type=int, default=2048,
                       help="Average KV entries kept per layer and head after prefill, window included")
    group.add_argument("--budget-from-result", type=Path, default=None,
                       help="Set --max-capacity-prompt to the generate_until retained entries per "
                            "layer-group of another result JSON (e.g. SemanticLogKV)")
    group.add_argument("--window-size", type=int, default=8, help="Observation window / always-kept recent tokens")
    group.add_argument("--kernel-size", type=int, default=7, help="1-D pooling kernel over window scores (odd)")
    group.add_argument("--pooling", choices=POOLINGS, default="maxpool")
    group.add_argument("--beta", type=int, default=20, help="Pyramid steepness: top/bottom layer budget ratio control")
    group.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=False,
                       help="Select after every forward, prompt chunks included (needs --prefill-chunk-size): "
                            "every layer stays within its pyramid budget plus one chunk")
    group.add_argument("--budget-bytes", type=int, default=None,
                       help="Set --max-capacity-prompt to the largest average budget whose streaming layer budgets "
                            "hold their K/V within this many bytes, e.g. the LogKV cache's allocated bytes")
    group.add_argument("--kv-cache-granularity", choices=GRANULARITIES, default="kv_head")
    group.add_argument("--gqa-score-agg", choices=GQA_SCORE_AGGS, default="mean")


def budget_from_result(path: Path) -> int:
    """Match the retained-entry count of a reference run at request end (entries, not bytes)."""
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    report = result.get("kv_compression") or {}
    summary = (report.get("by_request_type") or {}).get("generate_until") or report.get("summary")
    value = (summary or {}).get("retained_slots_per_layer_group")
    if not value:
        raise ValueError(f"{path} has no kv_compression retained_slots_per_layer_group")
    return int(round(value))


def resolve_method_args(args) -> None:
    if args.budget_from_result is not None:
        args.max_capacity_prompt = budget_from_result(args.budget_from_result)
    if args.method != "fullkv":
        # Validate eagerly, before any weights or task data are loaded.
        pyramidkv_settings(args, num_hidden_layers=2)


def pyramidkv_settings(args, num_hidden_layers: int, settings_class=PyramidKVSettings):
    return settings_class(
        num_hidden_layers=num_hidden_layers, max_capacity_prompt=args.max_capacity_prompt,
        window_size=args.window_size, kernel_size=args.kernel_size, pooling=args.pooling,
        beta=args.beta, gqa_score_agg=args.gqa_score_agg, streaming=bool(args.streaming),
    )


def _stored_heads(args, config) -> int:
    return config.num_attention_heads if args.kv_cache_granularity == "query_head" else config.num_key_value_heads


def persistent_bytes(args, config, element_size: int) -> int:
    """K/V bytes of a full streaming cache: every layer at its pyramid budget, every stored head."""
    entries = sum(pyramidkv_settings(args, config.num_hidden_layers).stream_schedule())
    return entries * _stored_heads(args, config) * 2 * config.head_dim * element_size


def budget_from_bytes(args, config, element_size: int) -> int:
    """The largest average budget whose streaming pyramid holds its K/V within ``args.budget_bytes``."""
    per_entry = _stored_heads(args, config) * 2 * config.head_dim * element_size
    budget = int(args.budget_bytes) // (config.num_hidden_layers * per_entry)
    while budget > args.window_size + 1:
        trial = argparse.Namespace(**{**vars(args), "max_capacity_prompt": budget, "streaming": True})
        if persistent_bytes(trial, config, element_size) <= args.budget_bytes:
            return budget
        budget -= 1
    raise ValueError(f"--budget-bytes {args.budget_bytes} cannot hold the window of {args.window_size} entries")


def method_config(args, num_hidden_layers: int | None = None, max_seq_length: int | None = None) -> dict:
    if args.method == "fullkv":
        return {"method": "fullkv", "kv_cache_granularity": args.kv_cache_granularity}
    settings = pyramidkv_settings(args, num_hidden_layers or 2)
    config = {**settings.config(), "kv_cache_granularity": args.kv_cache_granularity,
              "budget_from_result": None if args.budget_from_result is None else str(args.budget_from_result),
              "budget_bytes": args.budget_bytes}
    if num_hidden_layers and max_seq_length:
        config["entries_per_layer_for_full_window_prompt"] = settings.budget_schedule(max_seq_length)
    return config


def build_cache_factory(args, num_hidden_layers: int):
    # Imported here so that resolving and printing the configuration does not import torch.
    from .compressor import PyramidKVCompressor
    from .kv_cache import CompressedKVCache

    compressor = (None if args.method == "fullkv"
                  else pyramidkv_settings(args, num_hidden_layers, PyramidKVCompressor))
    return partial(CompressedKVCache, compressor=compressor, granularity=args.kv_cache_granularity)
