"""Method switch for this baseline: ``h2o`` or the ``fullkv`` reference.

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

from .options import GQA_SCORE_AGGS, GRANULARITIES, METHODS, H2OSettings

DEFAULT_METHOD = "h2o"


def add_method_args(parser) -> None:
    group = parser.add_argument_group("H2O (KVCache-Factory names and runner defaults)")
    group.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
    group.add_argument("--max-capacity-prompt", type=int, default=2048,
                       help="KV entries kept per layer and head after prefill (heavy hitters + recent window)")
    group.add_argument("--budget-from-result", type=Path, default=None,
                       help="Set --max-capacity-prompt to the generate_until retained entries per "
                            "layer-group of another result JSON (e.g. SemanticLogKV)")
    group.add_argument("--window-size", type=int, default=8,
                       help="Always-kept recent tokens (KVCache-Factory: 8; original H2O splits the budget in half)")
    group.add_argument("--decode-evict", action=argparse.BooleanOptionalAction, default=False,
                       help="Also evict during decoding to hold the budget (original H2O); default compresses once")
    group.add_argument("--score-chunk-size", type=int, default=512,
                       help="Prompt queries per chunk when accumulating attention scores (memory/speed only)")
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
        h2o_settings(args)  # validate before any weights or task data are loaded


def h2o_settings(args, settings_class=H2OSettings):
    return settings_class(max_capacity_prompt=args.max_capacity_prompt, window_size=args.window_size,
                          gqa_score_agg=args.gqa_score_agg, decode_evict=bool(args.decode_evict),
                          score_chunk_size=args.score_chunk_size)


def method_config(args, num_hidden_layers: int | None = None, max_seq_length: int | None = None) -> dict:
    if args.method == "fullkv":
        return {"method": "fullkv", "kv_cache_granularity": args.kv_cache_granularity}
    settings = h2o_settings(args)
    config = {**settings.config(), "kv_cache_granularity": args.kv_cache_granularity,
              "budget_from_result": None if args.budget_from_result is None else str(args.budget_from_result)}
    if num_hidden_layers and max_seq_length:
        config["entries_per_layer_for_full_window_prompt"] = settings.budget_schedule(max_seq_length,
                                                                                       num_hidden_layers)
    return config


def build_cache_factory(args, num_hidden_layers: int):
    # Imported here so that resolving and printing the configuration does not import torch.
    from .compressor import H2OCompressor
    from .kv_cache import CompressedKVCache

    compressor = None if args.method == "fullkv" else h2o_settings(args, H2OCompressor)
    return partial(CompressedKVCache, compressor=compressor, granularity=args.kv_cache_granularity)
