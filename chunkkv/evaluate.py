"""ChunkKV, SemanticLogKV and Full-KV through the project-owned evaluation runtime."""

import argparse
import ast
from datetime import datetime
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import traceback

import yaml

from .config import BASELINE_ROOT, RUNTIME_ROOT, DEFAULT_EXPERIMENT_CONFIG, load_experiment_config, require_project_file

DEFAULT_ROOT = RUNTIME_ROOT
DEFAULT_CONFIG = DEFAULT_EXPERIMENT_CONFIG
DEFAULT_CHUNKKV_CONFIG = BASELINE_ROOT / "configs/chunkkv/qwen3_1.7b_32k.yaml"
DEFAULT_EVALUATION_CONFIG = BASELINE_ROOT / "configs/chunkkv/eval_semantic_default.yaml"
CHUNKKV_FIELDS = ("chunk_size", "window_size", "budget", "keep_ratio", "reuse_layers")
BACKENDS = ("chunkkv", "logkv", "dense")
# Requirements of the bundled protocol code and the shared offline cache, not a target environment:
# evaluation adapts to SemanticLogKV's environment (torch218: lm-eval 0.4.11, datasets 4.8.4 at the
# time of writing). lm-eval 0.4.9 is the oldest release the protocol was checked against; the cache's
# IFEval entry uses the Json feature type, which datasets reads from 4.7 on. Results record the exact
# versions and compare_results refuses to compare runs whose versions differ.
EVAL_LM_EVAL_MIN, EVAL_DATASETS_MIN = (0, 4, 9), (4, 7)


def configure_environment():
    # HF libraries capture these values at first import. This path is data, not source code.
    base = os.environ.get("CHUNKKV_HF_CACHE", "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache")
    os.environ.update(
        HF_HOME=base, HF_DATASETS_CACHE=f"{base}/hf_cache", HF_EVALUATE_CACHE=f"{base}/evaluate",
        HF_MODULES_CACHE=f"{base}/modules", HF_HUB_CACHE=f"{base}/hub",
        HUGGINGFACE_HUB_CACHE=f"{base}/hub", RULER_CACHE_DIR=f"{base}/ruler_cache",
        NLTK_DATA=f"{base}/nltk_data", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        HF_DATASETS_IN_MEMORY_MAX_SIZE="0", HF_DATASETS_TRUST_REMOTE_CODE="1",
        TOKENIZERS_PARALLELISM="false", HF_ALLOW_CODE_EVAL="1",
    )


def check_environment():
    """Fail early when the installed packages cannot run the protocol or read the shared cache."""
    for name, minimum, needed_by in (("lm-eval", EVAL_LM_EVAL_MIN, "The bundled evaluation protocol"),
                                     ("datasets", EVAL_DATASETS_MIN, "The shared offline cache")):
        found = importlib.metadata.version(name)
        if tuple(int(part) for part in re.findall(r"\d+", found)[:len(minimum)]) < minimum:
            raise RuntimeError(f"{needed_by} needs {name}>={'.'.join(map(str, minimum))}, found {found}; "
                               "evaluate in SemanticLogKV's environment without changing its packages")


def load_evaluator(root=RUNTIME_ROOT):
    """Private evaluator instance with imports confined to the bundled runtime."""
    # Checked before the evaluator imports them; the exact versions are recorded in every result.
    check_environment()
    configure_environment()
    spec = importlib.util.spec_from_file_location(
        "chunkkv._vendor.semanticlogkv._evaluation", root / "eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def load_chunkkv_options(path, overrides):
    """Method settings from YAML, then explicit CLI values; validated before any model load."""
    from dataclasses import fields

    from .settings import ChunkKVSettings

    options = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = sorted(set(options) - set(CHUNKKV_FIELDS))
    if unknown:
        raise ValueError(f"Unknown ChunkKV settings in {path}: {unknown}")
    options = {key: value for key, value in options.items() if value is not None}
    given = {key: value for key, value in overrides.items() if value is not None}
    # A budget given on the command line replaces the YAML's, in either form.
    if "keep_ratio" in given:
        options.pop("budget", None)
    if "budget" in given:
        options.pop("keep_ratio", None)
    options = {**{field.name: field.default for field in fields(ChunkKVSettings)}, **options, **given}
    if isinstance(options["budget"], str) and options["budget"] != "logkv":
        options["budget"] = int(options["budget"])
    byte_matched = options["budget"] == "logkv"
    if byte_matched and options["keep_ratio"] is not None:
        raise ValueError("Set exactly one of budget and keep_ratio")
    # The byte-matched budget needs the loaded model; validate every other field now.
    ChunkKVSettings(**{**options, **({"budget": None, "keep_ratio": 1.0} if byte_matched else {})})
    return options


def slot_matched_budget(path):
    """Prompt budget = SemanticLogKV's mean retained KV entries per layer and KV head on generation."""
    from .train import sha256

    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if result.get("model_backend") != "logkv":
        raise ValueError("Expected a result of `chunkkv.evaluate --backend logkv`")
    stats = result["kv_compression"]["by_request_type"].get("generate_until") or {}
    if not stats.get("request_count"):
        raise ValueError("The SemanticLogKV result has no generate_until KV snapshots")
    entries = stats["retained_slots_per_layer_group"]
    return int(entries), {"result": str(Path(path).resolve()), "sha256": sha256(path),
                          "retained_slots_per_layer_group": entries, "requests": stats["request_count"]}


def load_eval_suite(path, suite):
    """Read a project-owned snapshot of the reference shell scripts' task groups."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tasks = config["niah_tasks" if suite == "niah" else "main_tasks"]
    if suite == "longbench":
        tasks = [task for task in tasks if task.startswith("longbench_")]
    if not tasks:
        raise ValueError(f"Evaluation suite {suite} has no tasks in {path}")
    lengths = config["niah_max_seq_lengths"] if suite == "niah" else None
    return ",".join(tasks), lengths


def build_eval_kwargs(evaluator, source_config, model, backend, tasks, limit, output, options=None,
                      max_seq_lengths=None):
    source = load_experiment_config(source_config)
    # Preserve the configured spelling (including trailing slash) used by RULER metadata/cache keys.
    base = source["ckpt_dir"]
    if model is None:
        if backend == "logkv":
            raise ValueError("--model must name the SemanticLogKV checkpoint for the logkv backend")
        model = base  # ChunkKV is training-free: the original Base by default
    context_length = int(source["context_length"])
    if context_length <= 0:
        raise ValueError("Experiment context_length must be positive")
    metadata = None
    if any(task.startswith(("niah_", "ruler_")) for task in tasks.split(",")):
        lengths = [context_length] if max_seq_lengths is None else max_seq_lengths
        if not lengths or any(not isinstance(n, int) or n <= 0 or n > context_length for n in lengths):
            raise ValueError("NIAH lengths must be positive and within the experiment context_length")
        metadata = {"pretrained": str(base), "max_seq_lengths": lengths}
    # LogKV method parameters cross over only for LogKV itself or to size a byte-matched budget.
    needs_logkv = backend == "logkv" or (backend == "chunkkv" and (options or {}).get("budget") == "logkv")
    if evaluator is None:
        # --print-config must work before installing evaluation/model dependencies.
        tree = ast.parse((RUNTIME_ROOT / "eval.py").read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        supported = {arg.arg for arg in main.args.args + main.args.kwonlyargs}
    else:
        supported = inspect.signature(evaluator.main).parameters
    kwargs = {k: v for k, v in source.items()
              if needs_logkv and k.startswith("log_kv_")
              and not k.startswith("log_kv_diag_") and k in supported}
    if backend == "dense":
        kwargs["log_kv_dense_mode"] = True
    kwargs.update(
        checkpoint_dir=str(model), tokenizer_dir=str(base),
        config_overrides={"block_size": context_length}, benchmark=tasks, limit=limit,
        metadata=metadata,
        output_path=str(output),
    )
    return kwargs


def comparison_contract(root, source_config, kwargs):
    from .semanticlogkv_eval import file_sha256

    base = Path(kwargs["tokenizer_dir"])
    tokenizer_names = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                       "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
                       "generation_config.json")
    source_files = sorted(root.rglob("*.py")) + [root / "SOURCE.json"]
    return {
        "source_config": str(source_config),
        "source_checkpoint_dir": str(base),
        "source_weight_sha256": file_sha256(base / "lit_model.pth"),
        "source_config_files": {name: file_sha256(base / name)
                                for name in ("config.json", "model_config.yaml")},
        "tokenizer_files": {name: file_sha256(base / name) for name in tokenizer_names
                            if (base / name).is_file()},
        "evaluation_code": {str(path.relative_to(root)): file_sha256(path) for path in source_files},
        "compatibility_code": {name: file_sha256(Path(__file__).parent / name)
                               for name in ("evaluate.py", "semanticlogkv_eval.py", "config.py")},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "datasets", "pyarrow", "lm-eval", "tokenizers")},
        "cache": {key: os.environ[key] for key in
                  ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "RULER_CACHE_DIR")},
        "benchmark": kwargs["benchmark"], "metadata": kwargs["metadata"],
        "limit": kwargs["limit"], "max_seq_length": kwargs["config_overrides"]["block_size"],
    }


def runtime_environment():
    """Interpreter, CUDA/driver and launch topology; recorded, not part of the protocol check."""
    import torch

    driver = None
    if torch.cuda.is_available():
        try:
            driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=30, check=True).stdout.split()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            driver = None
    return {
        "python": sys.version.split()[0], "python_executable": sys.executable, "platform": platform.platform(),
        "host": socket.gethostname(), "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "nvidia_driver": driver,
        # torchrun: GROUP_WORLD_SIZE nodes x LOCAL_WORLD_SIZE processes = WORLD_SIZE ranks.
        "topology": {key: os.environ.get(key) for key in
                     ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                      "LITGPT_EXPECTED_WORLD_SIZE")},
    }


def evaluated_model(evaluator, checkpoint_dir):
    """Identity of the weights actually evaluated, and the dense-CPT manifest when present."""
    from .semanticlogkv_eval import file_sha256

    # The reference main replaces a directory with its newest step_* checkpoint; hash that one.
    path = Path(evaluator._resolve_checkpoint_dir(str(checkpoint_dir)))
    record = {"checkpoint_dir": str(path), "lit_model_sha256": file_sha256(path / "lit_model.pth")}
    for directory in (path, path.parent):  # step_* checkpoints sit inside the run directory
        manifest = directory / "chunkkv_train_manifest.json"
        if manifest.is_file():
            record["dense_cpt_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
            break
    return record


def install_chunkkv_adapter(evaluator, options):
    """Bind ChunkKV only inside this private evaluator instance, not the litgpt package."""
    from .semanticlogkv_eval import compression_snapshot, make_chunkkv_lm

    evaluator.LogKVLM = make_chunkkv_lm(evaluator, options)
    evaluator.cache_compression_snapshot = compression_snapshot

    # The reference main labels every non-dense/SWA model as LogKV.
    def print_backend(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("🧩 logKV "):
            args = (f"🧩 ChunkKV {options} | SemanticLogKV-compatible request and metric interfaces", *args[1:])
        print(*args, **kwargs)
    evaluator.print = print_backend


def run_evaluation(evaluator, kwargs, backend, contract, options=None, budget_reference=None, resolved=None):
    """Keep reference main's lifecycle/output, then attach baseline-owned provenance."""
    output = Path(kwargs["output_path"]).resolve()
    kwargs = {**kwargs, "output_path": str(output)}
    root = Path(evaluator.__file__).parent.resolve()
    if output.is_relative_to(root):
        raise ValueError("Evaluation outputs must be outside the bundled runtime directory")
    output.mkdir(parents=True, exist_ok=True)
    is_main = int(os.environ.get("RANK", "0")) == 0
    before = set(output.glob("eval_results_[0-9]*.json")) if is_main else set()
    if backend == "chunkkv":
        install_chunkkv_adapter(evaluator, options)
    import datasets

    # map/filter caches otherwise share a filename across ranks on the mounted cache.
    # This keeps load_dataset's offline source cache, but gives transforms private temp files.
    caching_enabled = datasets.is_caching_enabled()
    datasets.disable_caching()
    if is_main:
        print("[hf-cache] Dataset transforms use per-process temporary files; offline source cache is unchanged.",
              flush=True)
    # The unchanged main writes eval_results_cache.json to its working directory.
    previous_dir = Path.cwd()
    try:
        os.chdir(output)
        evaluator.main(**kwargs)
    except ValueError as exc:
        # Raised while task data loads from the offline cache, before any model call.
        in_cache_lookup = any("packaged_modules/cache" in frame.filename.replace(os.sep, "/")
                              for frame in traceback.extract_tb(exc.__traceback__))
        if isinstance(exc, json.JSONDecodeError) and in_cache_lookup:
            raise RuntimeError(
                f"A dataset_info.json in the offline cache under {os.environ.get('HF_DATASETS_CACHE')} is not "
                "valid JSON (datasets parses it for every entry of the config it loads). Find it with "
                "`python -m chunkkv.check_dataset_cache`; see docs/CHUNKKV.md."
            ) from exc
        if not re.match(r"Feature type '\w+' not found", str(exc)):
            raise
        raise RuntimeError(
            f"An offline cache entry under {os.environ.get('HF_DATASETS_CACHE')} was written by a newer "
            f"`datasets` than the installed {importlib.metadata.version('datasets')}. List such entries with "
            "`python -m chunkkv.check_dataset_cache`; see docs/CHUNKKV.md."
        ) from exc
    finally:
        os.chdir(previous_dir)
        if caching_enabled:
            datasets.enable_caching()
    if is_main:
        created = set(output.glob("eval_results_[0-9]*.json")) - before
        if len(created) != 1:
            raise RuntimeError("Expected one new evaluation result; refusing to annotate missing or old results")
        path, = created
        result = json.loads(path.read_text())
        if not isinstance(result.get("results"), dict):
            raise RuntimeError("Reference evaluator did not save structured task results")
        if backend != "logkv":
            # LogKV method fields describe a different cache; ChunkKV's budget is in cache_budget.
            result = {k: v for k, v in result.items() if not k.startswith("log_kv_") and k != "swa_window_size"}
        result.update(model_backend=backend, comparison_contract=contract,
                      tokenizer_dir=kwargs["tokenizer_dir"],
                      max_seq_length=kwargs["config_overrides"]["block_size"])
        if backend == "dense":
            result["log_kv_dense_mode"] = True
        if budget_reference:
            result["chunkkv_budget_reference"] = budget_reference
        if resolved is not None:
            result["resolved_eval"] = resolved
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"Annotated {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="chunkkv")
    parser.add_argument("--model", type=Path,
                        help="LitGPT checkpoint dir; default for chunkkv/dense is the experiment's original Base")
    parser.add_argument("--experiment-config", "--semantic-config", type=Path, default=DEFAULT_CONFIG,
                        help="Project-owned experiment YAML (relative CLI paths use the current directory)")
    parser.add_argument("--chunkkv-config", type=Path, default=DEFAULT_CHUNKKV_CONFIG)
    parser.add_argument("--budget", help='prompt KV entries per layer/KV head, or "logkv" for byte matching')
    parser.add_argument("--keep-ratio", type=float, help="fraction of prompt tokens kept instead of --budget")
    parser.add_argument("--budget-from-result", type=Path,
                        help="SemanticLogKV result of this launcher; its mean retained KV entries per layer "
                             "and KV head on generate_until become the prompt budget")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--reuse-layers", type=int, help="layer-wise index reuse depth; 1 disables reuse")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tasks", help="Comma-separated task names for a diagnostic check (not a formal run)")
    selection.add_argument("--suite", choices=("main", "longbench", "niah"),
                           help="Reference task group; outputs go into a same-named subdirectory")
    parser.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG,
                        help="Project-owned task lists and NIAH lengths")
    parser.add_argument("--limit", type=int, help="Same first N samples per task for every method")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-config", action="store_true", help="Resolve shared settings without loading weights")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    lengths = None
    if args.suite:
        args.tasks, lengths = load_eval_suite(args.evaluation_config, args.suite)
    elif args.tasks is None:
        parser.error("Choose --suite main|niah|longbench (reference task lists) or --tasks (diagnostics)")
    options = budget_reference = None
    if args.budget_from_result:
        if args.backend != "chunkkv" or args.budget or args.keep_ratio:
            parser.error("--budget-from-result sets the ChunkKV budget; do not combine it with another")
        budget, budget_reference = slot_matched_budget(args.budget_from_result)
        args.budget = str(budget)
    if args.backend == "chunkkv":
        options = load_chunkkv_options(args.chunkkv_config, {
            "budget": args.budget, "keep_ratio": args.keep_ratio, "chunk_size": args.chunk_size,
            "window_size": args.window_size, "reuse_layers": args.reuse_layers,
        })
    root = RUNTIME_ROOT
    try:
        source_config = require_project_file(args.experiment_config)
    except ValueError as error:
        parser.error(str(error))
    model = args.model.resolve() if args.model else None
    output = (args.output or BASELINE_ROOT / "outputs/eval" / args.backend).resolve()
    if args.suite:
        output = output / args.suite
    if output.is_relative_to(root):
        parser.error("Keep evaluation outputs outside the bundled runtime directory")
    evaluator = None if args.print_config else load_evaluator()
    kwargs = build_eval_kwargs(evaluator, source_config, model, args.backend,
                               args.tasks, args.limit, output, options, max_seq_lengths=lengths)
    if args.print_config:
        print(json.dumps({"experiment_config": str(source_config), "runtime_root": str(root),
                          "suite": args.suite,
                          "backend": args.backend, "chunkkv": options, **kwargs}, ensure_ascii=False, indent=2))
        return
    contract = resolved = None
    if int(os.environ.get("RANK", "0")) == 0:
        contract = comparison_contract(root, source_config, kwargs)
        model_record = evaluated_model(evaluator, kwargs["checkpoint_dir"])
        output.mkdir(parents=True, exist_ok=True)
        launch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved = {"launch_id": launch_id, "suite": args.suite, "backend": args.backend, "chunkkv": options,
                    "budget_reference": budget_reference, "evaluation": kwargs, "evaluated_model": model_record,
                    "environment": runtime_environment(), "comparison_contract": contract}
        # One file per launch: a later run in the same directory never replaces this record.
        resolved_path = output / f"resolved_eval_{launch_id}.yaml"
        suffix = 1
        while resolved_path.exists():
            resolved_path, suffix = output / f"resolved_eval_{launch_id}_{suffix}.yaml", suffix + 1
        resolved_path.write_text(yaml.safe_dump(resolved, allow_unicode=True))
        contract = {**contract, "evaluated_model": model_record}
    run_evaluation(evaluator, kwargs, args.backend, contract, options, budget_reference, resolved)


if __name__ == "__main__":
    main()
