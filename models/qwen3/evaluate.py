"""Baseline-owned compatibility layer for an unmodified SemanticLogKV checkout."""

import argparse
import ast
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys


DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "litgpt"
DEFAULT_CONFIG = "exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml"


def configure_environment(root):
    # Read only the source cache root; HF libraries capture these values at first import.
    tree = ast.parse((root / "eval.py").read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "BASE" for target in node.targets))
    base = ast.literal_eval(assignment.value)
    os.environ.update(
        HF_HOME=base, HF_DATASETS_CACHE=f"{base}/hf_cache", HF_EVALUATE_CACHE=f"{base}/evaluate",
        HF_MODULES_CACHE=f"{base}/modules", HF_HUB_CACHE=f"{base}/hub",
        HUGGINGFACE_HUB_CACHE=f"{base}/hub", RULER_CACHE_DIR=f"{base}/ruler_cache",
        NLTK_DATA=f"{base}/nltk_data", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        HF_DATASETS_IN_MEMORY_MAX_SIZE="0", HF_DATASETS_TRUST_REMOTE_CODE="1",
        TOKENIZERS_PARALLELISM="false", HF_ALLOW_CODE_EVAL="1",
    )


def load_evaluator(root):
    configure_environment(root)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("_baseline_semanticlogkv_evaluation", root / "eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    import litgpt
    if Path(litgpt.__file__).resolve().parent != root / "litgpt":
        raise RuntimeError(f"Wrong LitGPT checkout imported: {litgpt.__file__}")
    if Path(sys.modules["utils"].__file__).resolve() != root / "utils.py":
        raise RuntimeError("An unrelated utils module was imported instead of SemanticLogKV utils.py")
    if importlib.metadata.version("lm-eval") != "0.4.9":
        raise RuntimeError("Shared evaluation requires lm-eval==0.4.9")
    if int(importlib.metadata.version("datasets").split(".")[0]) >= 4:
        raise RuntimeError("Use the evaluation environment (datasets<4), not the DMS training environment")
    return module


def install_dms_adapter(evaluator):
    """Bind this baseline only inside its private evaluator instance, not the litgpt package."""
    from .semanticlogkv_eval import compression_snapshot, make_dms_lm

    evaluator.LogKVLM = make_dms_lm(evaluator)
    evaluator.cache_compression_snapshot = compression_snapshot
    # The reference main labels every non-dense/SWA model as LogKV.
    def print_backend(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("🧩 logKV "):
            args = ("🧩 DMS adapters | SemanticLogKV-compatible request and metric interfaces", *args[1:])
        print(*args, **kwargs)
    evaluator.print = print_backend


def build_eval_kwargs(evaluator, source_config, model, backend, tasks, limit, output):
    source = evaluator.expand_env_vars(evaluator._load_yaml_config(str(source_config), str(model)))
    # Preserve the configured spelling (including trailing slash) used by RULER metadata/cache keys.
    base = os.path.expanduser(str(source["ckpt_dir"]))
    if not Path(base).is_absolute():
        base = str((Path(evaluator.__file__).parent / base).resolve())
    context_length = int(source["context_length"])
    if context_length <= 0:
        raise ValueError("SemanticLogKV context_length must be positive")
    # Only method parameters cross over from the training YAML; the task contract is explicit.
    supported = inspect.signature(evaluator.main).parameters
    kwargs = {k: v for k, v in source.items()
              if backend == "logkv" and k.startswith("log_kv_")
              and not k.startswith("log_kv_diag_") and k in supported}
    kwargs.update(
        checkpoint_dir=str(model), tokenizer_dir=str(base),
        config_overrides={"block_size": context_length}, benchmark=tasks, limit=limit,
        metadata={"pretrained": str(base), "max_seq_lengths": [context_length]},
        output_path=str(output),
    )
    calibration = kwargs.get("log_kv_semantic_s_h_path")
    if calibration and not Path(calibration).is_absolute():
        kwargs["log_kv_semantic_s_h_path"] = str((Path(evaluator.__file__).parent / calibration).resolve())
    return kwargs


def comparison_contract(root, source_config, kwargs):
    from .prepare_base import TOKENIZER_NAMES
    from .semanticlogkv_eval import file_sha256

    base = Path(kwargs["tokenizer_dir"])
    source_files = ("eval.py", "utils.py", "litgpt/tokenizer.py", "litgpt/generate/base.py",
                    "litgpt/kv_compression.py", "litgpt/ruler_patch.py")
    return {
        "source_config": str(source_config),
        "source_checkpoint_dir": str(base),
        "source_weight_sha256": file_sha256(base / "lit_model.pth"),
        "source_config_files": {name: file_sha256(base / name)
                                for name in ("config.json", "model_config.yaml")},
        "tokenizer_files": {name: file_sha256(base / name) for name in TOKENIZER_NAMES
                            if (base / name).is_file()},
        "evaluation_code": {name: file_sha256(root / name) for name in source_files},
        "compatibility_code": {name: file_sha256(Path(__file__).parent / name)
                               for name in ("evaluate.py", "semanticlogkv_eval.py")},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "datasets", "lm-eval", "tokenizers")},
        "cache": {key: os.environ[key] for key in
                  ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "RULER_CACHE_DIR")},
        "benchmark": kwargs["benchmark"], "metadata": kwargs["metadata"],
        "limit": kwargs["limit"], "max_seq_length": kwargs["config_overrides"]["block_size"],
    }


def run_evaluation(evaluator, kwargs, backend, contract):
    """Keep reference main's lifecycle/output, then attach baseline-owned provenance."""
    output = Path(kwargs["output_path"]).resolve()
    kwargs = {**kwargs, "output_path": str(output)}
    root = Path(evaluator.__file__).parent.resolve()
    if output.resolve().is_relative_to(root):
        raise ValueError("Baseline evaluation outputs must be outside the reference litgpt checkout")
    output.mkdir(parents=True, exist_ok=True)
    is_main = int(os.environ.get("RANK", "0")) == 0
    before = set(output.glob("eval_results_[0-9]*.json")) if is_main else set()
    if backend == "dms":
        install_dms_adapter(evaluator)
    # The unchanged main writes eval_results_cache.json to its working directory.
    previous_dir = Path.cwd()
    try:
        os.chdir(output)
        evaluator.main(**kwargs)
    finally:
        os.chdir(previous_dir)
    if is_main:
        created = set(output.glob("eval_results_[0-9]*.json")) - before
        if len(created) != 1:
            raise RuntimeError("Expected one new evaluation result; refusing to annotate missing or old results")
        path, = created
        result = json.loads(path.read_text())
        if not isinstance(result.get("results"), dict):
            raise RuntimeError("Reference evaluator did not save structured task results")
        if backend == "dms":
            result = {k: v for k, v in result.items() if not k.startswith("log_kv_") and k != "swa_window_size"}
        result.update(model_backend=backend, comparison_contract=contract,
                      tokenizer_dir=kwargs["tokenizer_dir"],
                      max_seq_length=kwargs["config_overrides"]["block_size"])
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--backend", choices=("dms", "logkv"), default="dms")
    parser.add_argument("--litgpt-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--semantic-config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--tasks", default="niah_single_2,niah_single_3")
    parser.add_argument("--limit", type=int, help="Same first N samples per task for both methods")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-config", action="store_true", help="Resolve shared settings without loading weights")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    root = args.litgpt_root.resolve(strict=True)
    source_config = (root / args.semantic_config).resolve(strict=True)
    model = args.model.resolve()
    output = (args.output or model / "eval_aligned").resolve()
    evaluator = load_evaluator(root)
    kwargs = build_eval_kwargs(evaluator, source_config, model, args.backend,
                               args.tasks, args.limit, output)
    if args.print_config:
        print(json.dumps({"backend": args.backend, **kwargs}, ensure_ascii=False, indent=2))
        return
    if output.is_relative_to(root):
        parser.error("Keep baseline outputs outside the reference litgpt checkout")
    contract = None
    if int(os.environ.get("RANK", "0")) == 0:
        contract = comparison_contract(root, source_config, kwargs)
        output.mkdir(parents=True, exist_ok=True)
        import yaml
        (output / "resolved_eval.yaml").write_text(yaml.safe_dump(
            {"backend": args.backend, "evaluation": kwargs, "comparison_contract": contract}, allow_unicode=True,
        ))
    run_evaluation(evaluator, kwargs, args.backend, contract)


if __name__ == "__main__":
    main()
