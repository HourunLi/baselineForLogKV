"""Evaluate CaM, identical-weight Full KV, and SemanticLogKV through one request pipeline."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path

import yaml

from models.qwen3.config import BASELINE_ROOT, RUNTIME_ROOT, require_project_file
from models.qwen3.evaluate import (
    build_eval_kwargs, comparison_contract, configure_environment, evaluated_model, load_eval_suite,
    load_evaluator, run_evaluation, runtime_environment,
)


ROOT = BASELINE_ROOT
DEFAULT_CONFIG = ROOT / "configs/cam_qwen3_1.7b.yaml"
METHOD_FIELDS = ("start_size", "recent_size", "merge_size", "seed", "prefill_chunk_size")
BACKENDS = ("cam", "dense", "logkv")
SUITES = ("main", "longbench", "niah")
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
        raise ValueError("Choose --suite main|niah|longbench (reference task lists) or --tasks (diagnostics)")
    lengths = None
    if args.suite:
        tasks, lengths = load_eval_suite(evaluation, args.suite)
    else:
        tasks = args.tasks
        if any(not task.strip() for task in tasks.split(",")):
            raise ValueError("Select at least one nonempty benchmark task")
    options = {key: getattr(args, key) if getattr(args, key) is not None else config[key] for key in METHOD_FIELDS}
    if options["start_size"] < 0 or not 1 <= options["merge_size"] <= options["recent_size"]:
        raise ValueError("Require start_size >= 0 and recent_size >= merge_size >= 1")
    if options["prefill_chunk_size"] < 1 or options["seed"] < 0:
        raise ValueError("Require positive prefill_chunk_size and nonnegative seed")
    output = (args.output or ROOT / "outputs" / f"{args.backend}_qwen3_1.7b_eval").expanduser().resolve()
    if args.suite:
        output = output / args.suite
    if output.is_relative_to(RUNTIME_ROOT):
        raise ValueError("Keep evaluation outputs outside the bundled runtime directory")
    return experiment, model, output, tasks, lengths, options


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
            "implementation": "chunked projections, sequential causal attention; no optimized CaM CUDA kernel",
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
        experiment, model, output, tasks, lengths, options = resolve_run(args)
    except ValueError as error:
        parser.error(str(error))
    evaluator = None if args.print_config else load_evaluator()
    kwargs = build_eval_kwargs(evaluator, experiment, model, args.backend, tasks, args.limit, output,
                               max_seq_lengths=lengths)
    metadata = method_metadata(args.backend, options)
    resolved = dict(backend=args.backend, suite=args.suite, experiment_config=str(experiment),
                    evaluation=kwargs, method_metadata=metadata)
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
    run_evaluation(evaluator, kwargs, args.backend, contract, resolved, result_metadata=metadata)


if __name__ == "__main__":
    main()
