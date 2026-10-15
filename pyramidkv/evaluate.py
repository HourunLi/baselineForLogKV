"""Evaluate this KV-compression baseline under SemanticLogKV's evaluation contract.

Same tasks (lm-eval 0.4.9; default RULER niah_single_2/3 at 32K), same offline
HF/RULER caches, same prompt/scoring/generation rules, same KV-entry and speed
metrics and the same result-file layout, so results line up with SemanticLogKV
runs. Single GPU or one model per GPU with torchrun:

    torchrun --standalone --nproc_per_node 1 -m pyramidkv.evaluate \\
        --config configs/qwen3_1.7b_32k_pyramidkv.yaml --limit 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from . import dist_utils as du
from . import method
from .data_env import DEFAULT_HF_CACHE_BASE, configure_hf_environment
from .kv_cache import LOGLIKELIHOOD_MODES
from .qwen3 import detect_weights_format, load_config

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_BASE = "/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "merges.txt", "generation_config.json")


def _limit(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("--limit must be positive")
    return int(number) if number >= 1 and number.is_integer() else number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="YAML of option names (dashes or underscores); CLI wins")
    group = parser.add_argument_group("model, tasks and evaluation contract")
    group.add_argument("--model", default=DEFAULT_BASE,
                       help="Base checkpoint dir: SemanticLogKV ckpt_dir with lit_model.pth, or a HF Qwen3 dir")
    group.add_argument("--tokenizer-dir", default=None, help="Defaults to --model")
    group.add_argument("--weights-format", choices=("auto", "litgpt", "hf"), default="auto")
    group.add_argument("--tasks", default="niah_single_2,niah_single_3", help="Comma-separated lm-eval tasks")
    group.add_argument("--include-path", default=None, help="Extra directory of lm-eval task YAMLs")
    group.add_argument("--limit", type=_limit, default=None, help="First N docs per task (same for every method)")
    group.add_argument("--max-seq-length", type=int, default=32768, help="Model window shared with SemanticLogKV")
    group.add_argument("--ruler-lengths", default=None, help="metadata.max_seq_lengths (default: --max-seq-length)")
    group.add_argument("--ruler-pretrained", default=None,
                       help="metadata.pretrained string for RULER (default: tokenizer dir exactly as given)")
    group.add_argument("--hf-cache-base", default=DEFAULT_HF_CACHE_BASE)
    group.add_argument("--hf-online", action="store_true", help="Allow downloads instead of the offline cache")
    group.add_argument("--output", default=None, help="Result directory (default: outputs/<method>-<budget>)")
    group.add_argument("--device", default=None, help="Default: cuda:LOCAL_RANK, or cpu without CUDA")
    group.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    group.add_argument("--loglikelihood-mode", choices=LOGLIKELIHOOD_MODES, default="compress_context",
                       help="compress_context: score the continuation against the compressed context; "
                            "single_forward: one prefill (scores then equal FullKV)")
    group.add_argument("--no-weight-hash", action="store_true", help="Skip hashing weights for provenance")
    group.add_argument("--print-config", action="store_true", help="Resolve settings without loading weights")
    method.add_method_args(parser)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    known, _ = parser.parse_known_args(argv)
    if known.config is not None:
        config = yaml.safe_load(Path(known.config).read_text(encoding="utf-8")) or {}
        config = {str(key).replace("-", "_"): value for key, value in config.items()}
        unknown = sorted(set(config) - {action.dest for action in parser._actions} - {"config"})
        if unknown:
            parser.error(f"Unknown keys in {known.config}: {', '.join(unknown)}")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if isinstance(args.limit, str):
        args.limit = _limit(args.limit)
    args.tokenizer_dir = args.tokenizer_dir or args.model
    args.ruler_pretrained = args.ruler_pretrained or str(args.tokenizer_dir)
    lengths = args.ruler_lengths
    if lengths is None:
        lengths = [args.max_seq_length]
    elif isinstance(lengths, str):
        lengths = [int(item) for item in lengths.split(",") if item.strip()]
    args.ruler_lengths = [int(item) for item in lengths]
    if max(args.ruler_lengths) > args.max_seq_length:
        parser.error("--ruler-lengths cannot exceed --max-seq-length")
    method.resolve_method_args(args)
    if args.output is None:
        budget = "full" if args.method == "fullkv" else f"b{args.max_capacity_prompt}"
        args.output = str(REPO_ROOT / "outputs" / f"{args.method}-{budget}")
    return args


def evaluation_settings(args, num_hidden_layers: int | None = None) -> dict:
    return {
        "model_backend": args.method,
        "checkpoint_dir": str(args.model),
        "tokenizer_dir": str(args.tokenizer_dir),
        "weights_format": args.weights_format,
        "benchmark": args.tasks,
        "include_path": args.include_path,
        "limit": args.limit,
        "max_seq_length": args.max_seq_length,
        "metadata": {"pretrained": args.ruler_pretrained, "max_seq_lengths": args.ruler_lengths},
        "loglikelihood_mode": args.loglikelihood_mode,
        "dtype": args.dtype,
        "hf_cache_base": args.hf_cache_base,
        "hf_offline": not args.hf_online,
        "output": str(args.output),
        "method_config": method.method_config(args, num_hidden_layers, args.max_seq_length),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict:
    out = {}
    for name in ("torch", "lm-eval", "datasets", "transformers", "tokenizers", "numpy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _git_state() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(["git", "-C", str(REPO_ROOT), *cmd], capture_output=True, text=True,
                                  timeout=10, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = run("status", "--porcelain", "--", PACKAGE_DIR.name, "configs", "scripts")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def provenance(args, model_info: dict, hash_weights: bool) -> dict:
    model_dir, tokenizer_dir = Path(args.model).expanduser(), Path(args.tokenizer_dir).expanduser()
    weights = {}
    if hash_weights:
        if model_info.get("weights_format") == "litgpt":
            names = ["lit_model.pth"]
        else:
            names = sorted(p.name for p in model_dir.iterdir()
                           if p.suffix in (".safetensors", ".bin") and p.is_file())
        weights = {name: file_sha256(model_dir / name) for name in names}
    return {
        "weights_sha256": weights,
        "model_config_files": {name: file_sha256(model_dir / name) for name in ("config.json", "model_config.yaml")
                               if (model_dir / name).is_file()},
        "tokenizer_files": {name: file_sha256(tokenizer_dir / name) for name in TOKENIZER_FILES
                            if (tokenizer_dir / name).is_file()},
        "code_sha256": {path.name: file_sha256(path) for path in sorted(PACKAGE_DIR.glob("*.py"))},
        "git": _git_state(),
        "versions": _versions(),
        "cache": {key: os.environ.get(key) for key in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE",
                                                       "RULER_CACHE_DIR", "HF_DATASETS_OFFLINE")},
        "model": model_info,
    }


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
        except ImportError:
            pass
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            if isinstance(obj, torch.dtype):
                return str(obj)
        except ImportError:
            pass
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if isinstance(obj, Path):
            return str(obj)
        if callable(obj):
            return f"<function: {getattr(obj, '__name__', str(obj))}>"
        if hasattr(obj, "__dict__"):
            return f"<{obj.__class__.__name__} object>"
        return super().default(obj)


def task_metric_rows(results: dict) -> list[dict]:
    rows = []
    for task, metrics in (results.get("results") or {}).items():
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and "_stderr" not in key:
                rows.append({"dataset": task, "metric": key, "value": value})
    return rows


def compression_report(samples: list[dict]) -> dict:
    from .metrics import summarize_compression

    return {
        "measurement_point": "request_end_before_cache_reset",
        "slot_definition": "valid stored KV entries across all layers, batches and stored heads; dense "
                           "denominator = forwarded tokens x layers x KV groups; the prompt is compressed "
                           "once after prefill, later tokens are appended",
        "summary": summarize_compression(samples),
        "by_request_type": {kind: summarize_compression([row for row in samples if row["request_type"] == kind])
                            for kind in sorted({row["request_type"] for row in samples})},
        "samples": samples,
    }


def write_outputs(output_dir: Path, settings: dict, results: dict, compression_samples: list[dict],
                  speed_samples: list[dict], cache_budget: dict, prov: dict, world_size: int) -> Path:
    from .metrics import summarize_inference_metrics

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_metrics = [row for rank in speed_samples for row in rank["generation"]]
    ppl_metrics = [row for rank in speed_samples for row in rank["scoring"]]
    inference_summary = summarize_inference_metrics(gen_metrics, ppl_metrics)
    for kind in ("generation", "scoring"):
        if inference_summary[kind]["requests"]:
            print(f"⏱ inference speed ({kind}, all ranks): " + json.dumps(inference_summary[kind]), flush=True)
    report = compression_report(compression_samples)
    summary = report["summary"]
    if summary["request_count"]:
        print(f"🧮 KV entries ({summary['request_count']} calls, request-end snapshot): retention "
              f"{summary['kv_retention_ratio']:.2%}, saving {summary['kv_saving_ratio']:.2%}, "
              f"compression {summary['kv_compression_factor']:.2f}x", flush=True)
    try:
        from lm_eval.utils import make_table
        print(make_table(results))
    except Exception as error:  # table printing is cosmetic
        print(f"⚠️ could not print the results table: {error}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"eval_results_{ts}.json"
    payload = {
        "timestamp": ts,
        "benchmark": settings["benchmark"],
        "checkpoint_dir": settings["checkpoint_dir"],
        "model_backend": settings["model_backend"],
        "method_config": settings["method_config"],
        "max_seq_length": settings["max_seq_length"],
        "tokenizer_dir": settings["tokenizer_dir"],
        "loglikelihood_mode": settings["loglikelihood_mode"],
        "cache_budget": cache_budget,
        "kv_compression": report,
        "inference_metrics": {**inference_summary, "world_size": world_size,
                              "generate_until": gen_metrics, "loglikelihood": ppl_metrics},
        "metadata": settings["metadata"],
        "evaluation_settings": settings,
        "provenance": prov,
        "results": results,
    }
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False, cls=SafeJSONEncoder) + "\n",
                           encoding="utf-8")
    print(f"✅ results saved to {output_file}")

    if compression_samples:
        columns = list(dict.fromkeys(key for row in compression_samples for key in row))
        with output_file.with_suffix(".kv_compression.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(compression_samples)
    with output_file.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "metric", "value"])
        writer.writeheader()
        writer.writerows(task_metric_rows(results))
    try:
        import pandas as pd

        with pd.ExcelWriter(output_file.with_suffix(".inference_metrics.xlsx")) as writer:
            pd.DataFrame([{"request_type": kind, **inference_summary[kind]} for kind in ("generation", "scoring")]
                         ).to_excel(writer, sheet_name="speed_summary", index=False)
            if gen_metrics:
                gen_df = pd.DataFrame(gen_metrics)
                gen_df.to_excel(writer, sheet_name="generate_until", index=False)
                gen_df["prompt_bucket"] = pd.cut(gen_df["prompt_len"], bins=[0, 1024, 4096, 8192, 16384, 32768, 999999],
                                                 labels=["0-1K", "1K-4K", "4K-8K", "8K-16K", "16K-32K", "32K+"])
                pd.DataFrame([{"prompt_bucket": str(bucket),
                               **summarize_inference_metrics(rows.to_dict("records"), [])["generation"]}
                              for bucket, rows in gen_df.groupby("prompt_bucket", observed=True)]
                             ).to_excel(writer, sheet_name="gen_by_bucket", index=False)
            if ppl_metrics:
                pd.DataFrame(ppl_metrics).to_excel(writer, sheet_name="loglikelihood", index=False)
    except Exception as error:  # optional export, the JSON keeps every row
        print(f"⚠️ inference xlsx export skipped: {error}")
    return output_file


def main(argv=None) -> None:
    args = parse_args(argv)
    weights_format = (detect_weights_format(Path(args.model).expanduser())
                      if args.weights_format == "auto" else args.weights_format)
    if args.print_config:
        try:
            layers = load_config(Path(args.model).expanduser(), weights_format).num_hidden_layers
        except (FileNotFoundError, OSError):
            layers = None
        print(json.dumps(evaluation_settings(args, layers), indent=2, ensure_ascii=False))
        return

    # HF libraries read the cache variables at import time.
    configure_hf_environment(args.hf_cache_base, offline=not args.hf_online)
    import torch
    import torch.distributed as dist
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager

    from .data_env import install_dataset_alias_fallback, install_ruler_cache
    from .lm import CompressedKVLM

    install_ruler_cache()
    install_dataset_alias_fallback()
    config = load_config(Path(args.model).expanduser(), weights_format)
    settings = evaluation_settings(args, config.num_hidden_layers)
    output_dir = Path(args.output).expanduser().resolve()
    world = du.world_size()
    pg_owned, group, done = False, None, False
    try:
        if world > 1 and not dist.is_initialized():
            dist.init_process_group(backend="gloo", timeout=timedelta(hours=12))
            pg_owned, group = True, dist.group.WORLD
        device = args.device or (f"cuda:{du.local_rank()}" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
        if du.is_main():
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "resolved_eval.yaml").write_text(
                yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding="utf-8")
            print(f"🚀 {args.method} | tasks: {args.tasks} | window: {args.max_seq_length} | "
                  f"{json.dumps(settings['method_config'], ensure_ascii=False)}", flush=True)
        lm = CompressedKVLM(
            args.model, method.build_cache_factory(args, config.num_hidden_layers),
            tokenizer_dir=args.tokenizer_dir, max_seq_length=args.max_seq_length, device=device,
            dtype=getattr(torch, args.dtype), weights_format=weights_format,
            loglikelihood_mode=args.loglikelihood_mode, process_group=group,
        )
        results = evaluator.simple_evaluate(
            model=lm, tasks=[task.strip() for task in args.tasks.split(",") if task.strip()],
            confirm_run_unsafe_code=True, batch_size=1, metadata=settings["metadata"],
            limit=args.limit, log_samples=True,
            # lm-eval only forwards ``metadata`` to tasks through the TaskManager it builds itself.
            task_manager=(TaskManager(include_path=args.include_path, metadata=settings["metadata"])
                          if args.include_path else None),
        )
        compression = lm.all_gather_results(lm.compression_metrics, tag="KV compression")
        speed = lm.all_gather_results([{"generation": lm.gen_metrics, "scoring": lm.ppl_metrics}],
                                      tag="inference speed")
        if du.is_main():
            cache_budget = {"allocation": "dynamic", "persistent_cache_bytes": None,
                            **settings["method_config"]}
            prov = provenance(args, lm.model_info, hash_weights=not args.no_weight_hash)
            prov.update(world_size=world, device=torch.cuda.get_device_name(device)
                        if device.startswith("cuda") else device)
            write_outputs(output_dir, settings, results, compression, speed, cache_budget, prov, world)
        done = True
    finally:
        if du.dist_ready() and done:
            du.wait_all_ranks("final sync")
        if pg_owned and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
