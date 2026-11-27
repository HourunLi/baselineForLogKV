"""HybridCSA, SemanticLogKV and Full-KV through the project-owned evaluation runtime."""

import argparse
import ast
from datetime import datetime
import importlib.metadata
import importlib.util
import inspect
import hashlib
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

from .config import (BASELINE_ROOT, RUNTIME_ROOT, DEFAULT_EXPERIMENT_CONFIG, environment_path,
                     load_experiment_config, require_project_file)

DEFAULT_ROOT = RUNTIME_ROOT
DEFAULT_CONFIG = DEFAULT_EXPERIMENT_CONFIG
DEFAULT_HYBRID_CSA_CONFIG = BASELINE_ROOT / "configs/hybrid_csa/qwen3_1.7b_32k.yaml"
DEFAULT_EVALUATION_CONFIG = BASELINE_ROOT / "configs/hybrid_csa/eval_semantic_default.yaml"
HYBRID_CSA_FIELDS = ("full_layer_interval", "compress_ratio", "window_size", "index_topk", "index_dim",
                     "index_heads", "query_chunk_size", "prefill_chunk_size")
BACKENDS = ("hybrid_csa", "logkv", "dense")
# The current reference uses 0.4.11 task definitions and native List/Json data.
# Exact versions are recorded and must match on both sides of a comparison.
EVAL_LM_EVAL_MIN, EVAL_DATASETS_MIN = (0, 4, 11), (4, 8, 4)


def configure_environment():
    # HF libraries capture these values at first import. This path is data, not source code.
    base = environment_path("HYBRID_CSA_HF_CACHE", "BASELINE_HF_CACHE", "HF_HOME") or \
        "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache"
    os.environ["HF_HOME"] = base
    hub = environment_path("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE") or f"{base}/hub"
    for key, value in dict(
        HF_DATASETS_CACHE=f"{base}/hf_cache", HF_EVALUATE_CACHE=f"{base}/evaluate",
        HF_MODULES_CACHE=f"{base}/modules", HF_HUB_CACHE=hub,
        HUGGINGFACE_HUB_CACHE=hub, RULER_CACHE_DIR=f"{base}/ruler_cache",
        NLTK_DATA=f"{base}/nltk_data",
    ).items():
        environment_path(key)
        os.environ.setdefault(key, value)
    os.environ.update(
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        HF_DATASETS_IN_MEMORY_MAX_SIZE="0", HF_DATASETS_TRUST_REMOTE_CODE="1",
        TOKENIZERS_PARALLELISM="false", HF_ALLOW_CODE_EVAL="1",
    )


def check_environment():
    """Fail early when the installed packages cannot run the protocol or read the shared cache."""
    for name, minimum, needed_by in (("lm-eval", EVAL_LM_EVAL_MIN, "The bundled evaluation protocol"),
                                     ("datasets", EVAL_DATASETS_MIN, "The shared offline cache"),
                                     ("pyarrow", (21,), "Native Json extension storage")):
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
        "hybrid_csa._vendor.semanticlogkv._evaluation", root / "eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def load_hybrid_csa_options(path, overrides=None):
    """Method settings from project YAML, then explicit CLI values; no model imports."""
    from .settings import HybridCSASettings

    path = require_project_file(path)
    options = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(options) - set(HYBRID_CSA_FIELDS))
    if unknown:
        raise ValueError(f"Unknown HybridCSA settings in {path}: {unknown}")
    options.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return HybridCSASettings(**options).as_dict()


def load_eval_suite(path, suite):
    """Read a project-owned snapshot of the reference shell scripts' task groups."""
    config = yaml.safe_load(require_project_file(path).read_text(encoding="utf-8"))
    if config["niah_max_seq_lengths"] != [32768]:
        raise ValueError("Formal NIAH lengths must be exactly [32768]")
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
        model = base  # The separately loaded trained adapter is bound to this original Base.
    context_length = int(source["context_length"])
    if context_length != 32768:
        raise ValueError("Formal evaluation context_length must be exactly 32768")
    metadata = None
    if any(task.startswith(("niah_", "ruler_")) for task in tasks.split(",")):
        lengths = [context_length] if max_seq_lengths is None else max_seq_lengths
        if lengths != [32768]:
            raise ValueError("NIAH lengths must be exactly [32768]")
        metadata = {"pretrained": str(base), "max_seq_lengths": lengths}
    # LogKV method parameters describe only that backend.
    needs_logkv = backend == "logkv"
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


def normalized_protocol_code(root):
    """Hash all vendored Python after only known namespace/cache-env relocation.

    No algorithm, prompt, score or other source differences are normalized away.
    Raw hashes are retained separately so the relocation remains auditable.
    """
    hashes = {}
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for old, new in (("hybrid_csa", "baseline"), ("chunkkv", "baseline"),
                         ("HYBRID_CSA", "BASELINE"), ("CHUNKKV", "BASELINE"),
                         ("HybridCSA", "Baseline"), ("ChunkKV", "Baseline")):
            source = source.replace(old, new)
        hashes[str(path.relative_to(root))] = hashlib.sha256(source.encode()).hexdigest()
    return hashes


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
        "protocol_code": normalized_protocol_code(root),
        "compatibility_code": {name: file_sha256(Path(__file__).parent / name)
                               for name in ("evaluate.py", "semanticlogkv_eval.py", "config.py", "model.py", "settings.py",
                                            "checkpoint.py", "core.py", "budget.py") if (Path(__file__).parent / name).is_file()},
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
    return record


def install_hybrid_csa_adapter(evaluator, options, adapter_path):
    """Bind HybridCSA only inside this private evaluator instance, not the litgpt package."""
    from .semanticlogkv_eval import compression_snapshot, make_hybrid_csa_lm, summarize_compression

    evaluator.hybrid_csa_adapter_record = None
    evaluator.LogKVLM = make_hybrid_csa_lm(evaluator, options, adapter_path)
    evaluator.cache_compression_snapshot = compression_snapshot
    evaluator.summarize_compression = summarize_compression
    # This experiment reports theoretical cache counts; keep request/scoring code intact.
    evaluator._reset_inference_peak_memory = lambda device: None
    evaluator._inference_peak_memory = lambda device: {
        "peak_memory_allocated_bytes": None, "peak_memory_reserved_bytes": None}
    summarize_inference = evaluator.summarize_inference_metrics
    def summarize_without_memory(*args):
        result = summarize_inference(*args)
        result["memory_scope"] = "Not measured in this baseline; cache budgets use theoretical element counts."
        return result
    evaluator.summarize_inference_metrics = summarize_without_memory

    # The reference main labels every non-dense/SWA model as LogKV.
    def print_backend(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("🧩 logKV "):
            args = (f"🧩 HybridCSA {options} | SemanticLogKV-compatible request and metric interfaces", *args[1:])
        print(*args, **kwargs)
    evaluator.print = print_backend


def run_evaluation(evaluator, kwargs, backend, contract, options=None, adapter_path=None, resolved=None):
    """Keep reference main's lifecycle/output, then attach baseline-owned provenance."""
    output = Path(kwargs["output_path"]).resolve()
    kwargs = {**kwargs, "output_path": str(output)}
    for name in ("checkpoint_dir", "tokenizer_dir"):
        if kwargs.get(name) and not Path(kwargs[name]).is_absolute():
            kwargs[name] = str(Path(kwargs[name]).resolve())
    adapter_path = Path(adapter_path).resolve() if adapter_path else None
    root = Path(evaluator.__file__).parent.resolve()
    if output.is_relative_to(root):
        raise ValueError("Evaluation outputs must be outside the bundled runtime directory")
    validate_output(output, kwargs)
    output.mkdir(parents=True, exist_ok=True)
    is_main = int(os.environ.get("RANK", "0")) == 0
    before = set(output.glob("eval_results_[0-9]*.json")) if is_main else set()
    if backend == "hybrid_csa":
        install_hybrid_csa_adapter(evaluator, options, adapter_path)
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
                "`python -m hybrid_csa.check_dataset_cache`; see docs/HYBRID_CSA.md."
            ) from exc
        if not re.match(r"Feature type '\w+' not found", str(exc)):
            raise
        raise RuntimeError(
            f"An offline cache entry under {os.environ.get('HF_DATASETS_CACHE')} was written by a newer "
            f"`datasets` than the installed {importlib.metadata.version('datasets')}. List such entries with "
            "`python -m hybrid_csa.check_dataset_cache`; see docs/HYBRID_CSA.md."
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
            # LogKV method fields describe a different cache; HybridCSA's budget is in cache_budget.
            result = {k: v for k, v in result.items() if not k.startswith("log_kv_") and k != "swa_window_size"}
        result.update(model_backend=backend, comparison_contract=contract,
                      tokenizer_dir=kwargs["tokenizer_dir"],
                      max_seq_length=kwargs["config_overrides"]["block_size"])
        if backend == "dense":
            result["log_kv_dense_mode"] = True
        if backend == "hybrid_csa":
            adapter = getattr(evaluator, "hybrid_csa_adapter_record", None)
            if not adapter:
                raise RuntimeError("No validated trained adapter identity was recorded")
            result["evaluated_adapter"] = adapter
            result["comparison_contract"]["evaluated_adapter"] = adapter
            if resolved is not None:
                resolved["evaluated_adapter"] = adapter
        if resolved is not None:
            result["resolved_eval"] = resolved
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"Annotated {path}")


def validate_output(output, kwargs):
    protected = [BASELINE_ROOT / "hybrid_csa", Path("/data/lihourun/semanticLogKV")]
    protected += [Path(kwargs[k]).resolve() for k in ("checkpoint_dir", "tokenizer_dir") if kwargs.get(k)]
    protected += [Path(os.environ[k]).resolve() for k in ("HF_HOME", "HF_DATASETS_CACHE", "RULER_CACHE_DIR")
                  if os.environ.get(k)]
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError(f"Output overlaps original assets or runtime: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="hybrid_csa")
    parser.add_argument("--model", type=Path,
                        help="LitGPT checkpoint dir; default for hybrid_csa/dense is the experiment's original Base")
    parser.add_argument("--experiment-config", "--semantic-config", type=Path, default=DEFAULT_CONFIG,
                        help="Project-owned experiment YAML (relative CLI paths use the current directory)")
    parser.add_argument("--hybrid-csa-config", type=Path, default=DEFAULT_HYBRID_CSA_CONFIG)
    parser.add_argument("--adapter", type=Path, help="Trained sparse-stage adapter checkpoint; required for HybridCSA")
    for field in HYBRID_CSA_FIELDS:
        parser.add_argument("--" + field.replace("_", "-"), type=int)
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
    if args.backend == "hybrid_csa" and not args.adapter and not args.print_config:
        parser.error("--adapter is required: untrained HybridCSA must not be used for evaluation")
    if args.backend == "hybrid_csa" and not args.print_config and not args.adapter.is_file():
        parser.error(f"Trained adapter file does not exist: {args.adapter}")
    if args.backend != "hybrid_csa" and args.adapter:
        parser.error("--adapter only applies to --backend hybrid_csa")
    options = None
    if args.backend == "hybrid_csa":
        options = load_hybrid_csa_options(args.hybrid_csa_config,
                                           {field: getattr(args, field) for field in HYBRID_CSA_FIELDS})
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
    validate_output(output, kwargs)
    if args.print_config:
        print(json.dumps({"experiment_config": str(source_config), "runtime_root": str(root),
                          "suite": args.suite,
                          "backend": args.backend, "hybrid_csa": options,
                          "adapter": str(args.adapter.resolve()) if args.adapter else None, **kwargs}, ensure_ascii=False, indent=2))
        return
    contract = resolved = None
    if int(os.environ.get("RANK", "0")) == 0:
        contract = comparison_contract(root, source_config, kwargs)
        model_record = evaluated_model(evaluator, kwargs["checkpoint_dir"])
        output.mkdir(parents=True, exist_ok=True)
        launch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved = {"launch_id": launch_id, "suite": args.suite, "backend": args.backend, "hybrid_csa": options,
                    "adapter": str(args.adapter.resolve()) if args.adapter else None,
                    "evaluation": kwargs, "evaluated_model": model_record,
                    "environment": runtime_environment(), "comparison_contract": contract}
        # One file per launch: a later run in the same directory never replaces this record.
        resolved_path = output / f"resolved_eval_{launch_id}.yaml"
        suffix = 1
        while resolved_path.exists():
            resolved_path, suffix = output / f"resolved_eval_{launch_id}_{suffix}.yaml", suffix + 1
        resolved_path.write_text(yaml.safe_dump(resolved, allow_unicode=True))
        contract = {**contract, "evaluated_model": model_record}
    run_evaluation(evaluator, kwargs, args.backend, contract, options,
                   args.adapter.resolve() if args.adapter else None, resolved)


if __name__ == "__main__":
    main()
