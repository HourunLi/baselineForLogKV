"""Evaluate CaM, identical-weight Full KV, and SemanticLogKV through one request pipeline."""

import argparse
import json
import os
from pathlib import Path

import yaml

from models.qwen3.evaluate import (
    build_eval_kwargs, comparison_contract, configure_environment, load_evaluator, run_evaluation,
)
from models.qwen3.semanticlogkv_eval import file_sha256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/cam_qwen3_1.7b.yaml"
METHOD_FIELDS = ("start_size", "recent_size", "merge_size", "seed", "prefill_chunk_size")
UPSTREAM_COMMIT = "fc80e64f460b343be96c87596f2494fc9d319c45"


def resolve_run(args):
    config = yaml.safe_load(args.config.read_text())
    root = Path(args.litgpt_root or config["litgpt_root"]).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    root = root.resolve(strict=True)
    configure_environment(root)
    from models.qwen3.prepare_base import load_semantic_config
    semantic = (root / (args.semantic_config or config["semantic_config"])).resolve(strict=True)
    model = args.model or Path(config["model"])
    if args.backend == "logkv" and args.model is None:
        raise ValueError("--backend logkv requires --model pointing to the SemanticLogKV checkpoint")
    model = model.expanduser()
    if not model.is_absolute():
        model = ROOT / model
    model = model.resolve()
    tasks = args.tasks or config["tasks"]
    if isinstance(tasks, list):
        tasks = ",".join(tasks)
    if not isinstance(tasks, str) or not tasks or any(not task.strip() for task in tasks.split(",")):
        raise ValueError("Select at least one nonempty benchmark task")
    lengths = [int(x) for x in args.ruler_lengths.split(",")] if args.ruler_lengths else config["ruler_lengths"]
    source = load_semantic_config(semantic)
    context = int(source["context_length"])
    if not lengths or any(not isinstance(n, int) or n < 1 or n > context for n in lengths):
        raise ValueError("RULER lengths must be positive and within the comparison context window")
    options = {key: getattr(args, key) if getattr(args, key) is not None else config[key] for key in METHOD_FIELDS}
    if options["start_size"] < 0 or not 1 <= options["merge_size"] <= options["recent_size"]:
        raise ValueError("Require start_size >= 0 and recent_size >= merge_size >= 1")
    if options["prefill_chunk_size"] < 1 or options["seed"] < 0:
        raise ValueError("Require positive prefill_chunk_size and nonnegative seed")
    output = (args.output or ROOT / "outputs" / f"{args.backend}_qwen3_1.7b_eval").expanduser().resolve()
    if output.is_relative_to(root):
        raise ValueError("Keep baseline outputs outside the reference litgpt checkout")
    return root, semantic, model, output, tasks, lengths, options


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
    parser.add_argument("--backend", choices=("cam", "dense", "logkv"), default="cam")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--litgpt-root", type=Path)
    parser.add_argument("--semantic-config", type=Path)
    parser.add_argument("--tasks", help="Same lm-eval task CSV for all compared backends")
    parser.add_argument("--ruler-lengths", help="RULER length CSV, e.g. 32768 for a single-length comparison")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    for key in METHOD_FIELDS:
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    root, semantic, model, output, tasks, lengths, options = resolve_run(args)
    evaluator = load_evaluator(root)
    kwargs = build_eval_kwargs(evaluator, semantic, model, args.backend, tasks, args.limit, output)
    kwargs["metadata"]["max_seq_lengths"] = lengths
    metadata = method_metadata(args.backend, options)
    resolved = dict(backend=args.backend, evaluation=kwargs, method_metadata=metadata)
    if args.print_config:
        print(json.dumps(resolved, ensure_ascii=False, indent=2))
        return
    if args.backend in ("cam", "dense"):
        from .semanticlogkv_eval import install_adapter
        install_adapter(evaluator, options, dense=args.backend == "dense")
    contract = None
    if int(os.environ.get("RANK", "0")) == 0:
        contract = comparison_contract(root, semantic, kwargs)
        contract["compatibility_code"].update({
            "cam/" + name: file_sha256(Path(__file__).parent / name)
            for name in ("evaluate.py", "semanticlogkv_eval.py", "model.py", "attention.py", "cache.py")
        })
        contract["evaluation_code"]["eval.sh"] = file_sha256(root / "eval.sh")
        output.mkdir(parents=True, exist_ok=True)
        (output / "resolved_eval.yaml").write_text(yaml.safe_dump(
            {**resolved, "comparison_contract": contract}, allow_unicode=True,
        ))
    run_evaluation(evaluator, kwargs, args.backend, contract, result_metadata=metadata)


if __name__ == "__main__":
    main()
