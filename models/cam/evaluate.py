"""Evaluate CaM, identical-weight Full KV, and SemanticLogKV through one request pipeline.

A config with ``context: 64k|128k`` runs SemanticLogKV's long-context stages with CaM: the window and RULER
length from the evaluation config's ``contexts``, the 'long' suite (BABILong and long-document perplexity, task
files under ``tasks/``) and the 'ruler' suite (13 RULER tasks), 100 samples per task.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path

import yaml

from models.qwen3.config import BASELINE_ROOT, RUNTIME_ROOT, require_project_file
from models.qwen3.evaluate import (
    LONG_SUITES, build_eval_kwargs, comparison_contract, configure_environment, evaluated_model, load_context,
    load_eval_suite, load_evaluator, run_evaluation, runtime_environment,
)


ROOT = BASELINE_ROOT
DEFAULT_CONFIG = ROOT / "configs/cam_qwen3_1.7b.yaml"
METHOD_FIELDS = ("start_size", "recent_size", "merge_size", "seed", "prefill_chunk_size", "merge_interval")
BACKENDS = ("cam", "dense", "logkv")
SUITES = ("main", "longbench", "niah", "long", "ruler")
CONTEXTS = ("32k", "64k", "128k")
# Project copies of SemanticLogKV's long-context task files (BABILong by length, long-document perplexity).
LONG_TASKS_DIR = BASELINE_ROOT / "tasks"
UPSTREAM_COMMIT = "fc80e64f460b343be96c87596f2494fc9d319c45"
COMPATIBILITY_FILES = (
    ROOT / "models/qwen3/evaluate.py", ROOT / "models/qwen3/config.py", ROOT / "models/qwen3/provenance.py",
    *(ROOT / "models/cam" / name for name in ("evaluate.py", "semanticlogkv_eval.py", "model.py",
                                               "attention.py", "cache.py")),
)


def _project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def resolve_run(args):
    # HF libraries capture cache variables at first import; set them before anything loads them.
    configure_environment()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    experiment = require_project_file(_project_path(config["experiment_config"]))
    evaluation = require_project_file(_project_path(config["evaluation_config"]))
    model = args.model or Path(config["model"])
    if args.backend == "logkv" and args.model is None:
        raise ValueError("--backend logkv requires --model pointing to the SemanticLogKV checkpoint")
    model = _project_path(model).resolve()
    if (args.suite is None) == (args.tasks is None):
        raise ValueError("Choose --suite main|niah|longbench|long|ruler (reference task lists) or --tasks (diagnostics)")
    context = str(config.get("context") or "32k")
    if context not in CONTEXTS:
        raise ValueError(f"context must be one of {CONTEXTS}, got {context!r}")
    long_context = load_context(evaluation, context)
    if long_context is not None:
        if args.backend != "cam":
            raise ValueError("64k/128k run CaM here; evaluate SemanticLogKV and Full KV with its own eval_long configs")
        if args.limit is None and args.suite in LONG_SUITES:
            args.limit = long_context["limit"]
        # Data locations of the long-context task files; the environment wins.
        for name, key in (("LONG_PPL_DATA", "long_ppl_data"), ("BABILONG_DATASET", "babilong_dataset")):
            if long_context.get(key) and not os.environ.get(name):
                os.environ[name] = str(long_context[key])
    lengths = None
    if args.suite:
        if (args.suite in LONG_SUITES) != (long_context is not None):
            raise ValueError(f"--suite {args.suite} runs at {'64k/128k' if args.suite in LONG_SUITES else '32k'}, "
                             f"not at the config's context {context}")
        tasks, lengths = load_eval_suite(evaluation, args.suite, context)
    else:
        tasks = args.tasks
        if any(not task.strip() for task in tasks.split(",")):
            raise ValueError("Select at least one nonempty benchmark task")
    options = {key: getattr(args, key) if getattr(args, key) is not None else config.get(key, 1 if key == "merge_interval"
               else None) for key in METHOD_FIELDS}
    budget_bytes = config.get("budget_bytes")
    if budget_bytes is not None and args.recent_size is None:
        # Entries whose raw K + V and fp32 score fit the bytes, in every layer and KV head; the sinks first.
        options["recent_size"] = cache_entries(model, budget_bytes) - options["start_size"]
    if options["merge_interval"] < 1:
        raise ValueError("Require positive merge_interval")
    if options["start_size"] < 0 or not 1 <= options["merge_size"] <= options["recent_size"]:
        raise ValueError("Require start_size >= 0 and recent_size >= merge_size >= 1")
    if options["prefill_chunk_size"] < 1 or options["seed"] < 0:
        raise ValueError("Require positive prefill_chunk_size and nonnegative seed")
    output = (args.output or ROOT / "outputs" / f"{args.backend}_qwen3_1.7b_eval").expanduser().resolve()
    if args.suite:
        output = output / args.suite
    if output.is_relative_to(RUNTIME_ROOT):
        raise ValueError("Keep evaluation outputs outside the bundled runtime directory")
    return experiment, model, output, tasks, lengths, options, context, long_context, budget_bytes


def cache_entries(model_dir, budget_bytes, element_size=2):
    """CaM entries per layer and KV head whose persistent bytes (K, V and an fp32 score) fit ``budget_bytes``."""
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise ValueError(f"budget_bytes needs the prepared Base's config.json to size entries: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    per_entry = 2 * head_dim * element_size + 4
    return int(budget_bytes) // (config["num_hidden_layers"] * config["num_key_value_heads"] * per_entry)


def method_metadata(backend, options):
    return {
        "method": "CaM-online-Qwen3-GQA" if backend == "cam" else backend,
        "training": "none; original prepared Base" if backend in ("cam", "dense") else "external checkpoint",
        "cam": {
            **options, "upstream_commit": UPSTREAM_COMMIT,
            "reference_path": "generate_task/streaming_llm",
            "gqa_scores": "mean query-head probabilities within each shared KV group",
            "score_dtype": "float32", "position_policy": "cache-relative RoPE on raw normalized K",
            "rng": "independent generator per layer, reset to the same seed for each request",
            "implementation": ("chunked projections; one merge step per token (the official online path)"
                               if options.get("merge_interval", 1) == 1 else
                               f"chunked projections; one merge step per {options['merge_interval']} tokens: their "
                               "queries attend to the cache as it stood before the step, then the step's victims "
                               "merge in order (exact chain, triangular solve)"),
        } if backend == "cam" else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--backend", choices=BACKENDS, default="cam")
    parser.add_argument("--model", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--suite", choices=SUITES,
                           help="Reference task group; outputs go into a same-named subdirectory")
    selection.add_argument("--tasks", help="Same lm-eval task CSV for all compared backends (diagnostics)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    for key in METHOD_FIELDS:
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        experiment, model, output, tasks, lengths, options, context, long_context, budget_bytes = resolve_run(args)
    except ValueError as error:
        parser.error(str(error))
    evaluator = None if args.print_config else load_evaluator()
    kwargs = build_eval_kwargs(evaluator, experiment, model, args.backend, tasks, args.limit, output,
                               max_seq_lengths=lengths, long_context=long_context)
    metadata = method_metadata(args.backend, options)
    if long_context is not None:
        metadata["cam"].update(context=context, budget_bytes=budget_bytes,
                               capacity_entries=options["start_size"] + options["recent_size"])
    resolved = dict(backend=args.backend, suite=args.suite, context=context, experiment_config=str(experiment),
                    evaluation=kwargs, method_metadata=metadata,
                    long_ppl_data=os.environ.get("LONG_PPL_DATA") if long_context else None)
    if args.print_config:
        print(json.dumps(resolved, ensure_ascii=False, indent=2))
        return
    if args.backend in ("cam", "dense"):
        from .semanticlogkv_eval import install_adapter
        install_adapter(evaluator, options, dense=args.backend == "dense")
    contract = None
    if int(os.environ.get("RANK", "0")) == 0:
        contract = comparison_contract(RUNTIME_ROOT, experiment, kwargs, COMPATIBILITY_FILES)
        model_record = evaluated_model(evaluator, kwargs["checkpoint_dir"], args.backend)
        output.mkdir(parents=True, exist_ok=True)
        launch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved = {**resolved, "launch_id": launch_id, "evaluated_model": model_record,
                    "environment": runtime_environment(), "comparison_contract": contract}
        # One file per launch: a later run in the same directory never replaces this record.
        resolved_path, suffix = output / f"resolved_eval_{launch_id}.yaml", 1
        while resolved_path.exists():
            resolved_path, suffix = output / f"resolved_eval_{launch_id}_{suffix}.yaml", suffix + 1
        resolved_path.write_text(yaml.safe_dump(resolved, allow_unicode=True))
        contract = {**contract, "evaluated_model": model_record}
    run_evaluation(evaluator, kwargs, args.backend, contract, resolved, result_metadata=metadata,
                   include_path=LONG_TASKS_DIR if long_context is not None else None)


if __name__ == "__main__":
    main()
