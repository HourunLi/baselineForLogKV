"""Shared helpers for running the project-owned SemanticLogKV evaluation runtime.

The evaluator (``eval.py``: tasks, prompts, truncation, generation, scoring,
KV/speed metrics and result files) is a private snapshot under
``models/_vendor/semanticlogkv``; no sibling checkout is imported or read.
The CaM command line is ``models.cam.evaluate``.
"""

import ast
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

import yaml

from .config import RUNTIME_ROOT, load_experiment_config

DEFAULT_HF_CACHE = "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache"
TOKENIZER_NAMES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
                   "generation_config.json")


def configure_environment():
    # HF libraries capture these values at first import. This path is data, not source code.
    base = os.environ.get("CAM_HF_CACHE", DEFAULT_HF_CACHE)
    os.environ.update(
        HF_HOME=base, HF_DATASETS_CACHE=f"{base}/hf_cache", HF_EVALUATE_CACHE=f"{base}/evaluate",
        HF_MODULES_CACHE=f"{base}/modules", HF_HUB_CACHE=f"{base}/hub",
        HUGGINGFACE_HUB_CACHE=f"{base}/hub", RULER_CACHE_DIR=f"{base}/ruler_cache",
        NLTK_DATA=f"{base}/nltk_data", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        HF_DATASETS_IN_MEMORY_MAX_SIZE="0", HF_DATASETS_TRUST_REMOTE_CODE="1",
        TOKENIZERS_PARALLELISM="false", HF_ALLOW_CODE_EVAL="1",
    )


def load_evaluator(root=RUNTIME_ROOT):
    """Private evaluator instance with imports confined to the bundled runtime."""
    if importlib.metadata.version("lm-eval") != "0.4.9":
        raise RuntimeError("Shared evaluation requires lm-eval==0.4.9")
    if int(importlib.metadata.version("datasets").split(".")[0]) >= 4:
        raise RuntimeError("Use the shared evaluation environment (datasets<4)")
    configure_environment()
    spec = importlib.util.spec_from_file_location("models._vendor.semanticlogkv._evaluation", root / "eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def load_eval_suite(path, suite):
    """Read the project-owned snapshot of the reference shell scripts' task groups."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tasks = config["niah_tasks" if suite == "niah" else "main_tasks"]
    if suite == "longbench":
        tasks = [task for task in tasks if task.startswith("longbench_")]
    if not tasks:
        raise ValueError(f"Evaluation suite {suite} has no tasks in {path}")
    lengths = config["niah_max_seq_lengths"] if suite == "niah" else None
    return ",".join(tasks), lengths


def build_eval_kwargs(evaluator, source_config, model, backend, tasks, limit, output, max_seq_lengths=None):
    source = load_experiment_config(source_config)
    # Preserve the configured spelling (including trailing slash) used by RULER metadata/cache keys.
    base = source["ckpt_dir"]
    context_length = int(source["context_length"])
    if context_length <= 0:
        raise ValueError("Experiment context_length must be positive")
    metadata = None
    if any(task.startswith(("niah_", "ruler_")) for task in tasks.split(",")):
        lengths = [context_length] if max_seq_lengths is None else max_seq_lengths
        if not lengths or any(not isinstance(n, int) or n <= 0 or n > context_length for n in lengths):
            raise ValueError("NIAH lengths must be positive and within the experiment context_length")
        metadata = {"pretrained": str(base), "max_seq_lengths": lengths}
    if evaluator is None:
        # --print-config must work before installing evaluation/model dependencies.
        tree = ast.parse((RUNTIME_ROOT / "eval.py").read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        supported = {arg.arg for arg in main.args.args + main.args.kwonlyargs}
    else:
        supported = inspect.signature(evaluator.main).parameters
    # Only the SemanticLogKV control reads its method parameters; the task contract is explicit.
    kwargs = {k: v for k, v in source.items()
              if backend == "logkv" and k.startswith("log_kv_")
              and not k.startswith("log_kv_diag_") and k in supported}
    kwargs.update(
        checkpoint_dir=str(model), tokenizer_dir=str(base),
        config_overrides={"block_size": context_length}, benchmark=tasks, limit=limit,
        metadata=metadata, output_path=str(output),
    )
    return kwargs


def comparison_contract(root, source_config, kwargs, compatibility_files):
    from .provenance import file_sha256

    base = Path(kwargs["tokenizer_dir"])
    source_files = sorted(root.rglob("*.py")) + [root / "SOURCE.json"]
    project = Path(__file__).resolve().parents[2]
    return {
        "source_config": str(source_config),
        "source_checkpoint_dir": str(base),
        "source_weight_sha256": file_sha256(base / "lit_model.pth"),
        "source_config_files": {name: file_sha256(base / name)
                                for name in ("config.json", "model_config.yaml")},
        "tokenizer_files": {name: file_sha256(base / name) for name in TOKENIZER_NAMES
                            if (base / name).is_file()},
        "evaluation_code": {str(path.relative_to(root)): file_sha256(path) for path in source_files},
        "compatibility_code": {str(Path(path).resolve().relative_to(project)): file_sha256(path)
                               for path in compatibility_files},
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


def evaluated_model(evaluator, checkpoint_dir, backend):
    """Identity of the weights actually evaluated (after the reference step_* resolution)."""
    from .provenance import file_sha256

    path = Path(evaluator._resolve_checkpoint_dir(str(checkpoint_dir)))
    if backend == "logkv":
        return {"checkpoint_dir": str(path), "lit_model_sha256": file_sha256(path / "lit_model.pth")}
    weights = sorted(item for item in path.iterdir()
                     if item.is_file() and (item.suffix in (".safetensors", ".bin") or item.name.endswith(".index.json")))
    if not weights:
        raise FileNotFoundError(f"No HF weight files in {path}")
    manifest = path / "semanticlogkv_base_manifest.json"
    return {
        "checkpoint_dir": str(path),
        "weight_files": {item.name: file_sha256(item) for item in weights},
        "config_sha256": file_sha256(path / "config.json"),
        "prepared_base_manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
    }


def run_evaluation(evaluator, kwargs, backend, contract, resolved=None, result_metadata=None):
    """Keep reference main's lifecycle/output, then attach baseline-owned provenance."""
    output = Path(kwargs["output_path"]).resolve()
    kwargs = {**kwargs, "output_path": str(output)}
    root = Path(evaluator.__file__).parent.resolve()
    if output.is_relative_to(root):
        raise ValueError("Evaluation outputs must be outside the bundled runtime directory")
    output.mkdir(parents=True, exist_ok=True)
    is_main = int(os.environ.get("RANK", "0")) == 0
    before = set(output.glob("eval_results_[0-9]*.json")) if is_main else set()
    import datasets

    # map/filter caches otherwise share a filename across ranks on the mounted cache.
    # This keeps load_dataset's offline source cache, but gives transforms private temp files.
    caching_enabled = datasets.is_caching_enabled()
    datasets.disable_caching()
    # The unchanged main writes eval_results_cache.json to its working directory.
    previous_dir = Path.cwd()
    try:
        os.chdir(output)
        evaluator.main(**kwargs)
    except ValueError as exc:
        # Raised while task data loads from the offline cache, before any model call.
        if not re.match(r"Feature type '\w+' not found", str(exc)):
            raise
        raise RuntimeError(
            f"An offline cache entry under {os.environ.get('HF_DATASETS_CACHE')} was written by a newer "
            f"`datasets` than the installed {importlib.metadata.version('datasets')}. The shared protocol "
            "reads entries readable by datasets<4: repair or restore that entry instead of upgrading "
            "datasets or switching data."
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
            # LogKV method fields describe a different cache; other backends record their own budget.
            result = {k: v for k, v in result.items() if not k.startswith("log_kv_") and k != "swa_window_size"}
        result.update(model_backend=backend, comparison_contract=contract,
                      tokenizer_dir=kwargs["tokenizer_dir"],
                      max_seq_length=kwargs["config_overrides"]["block_size"])
        result.update(result_metadata or {})
        if resolved is not None:
            result["resolved_eval"] = resolved
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"Annotated {path}")
