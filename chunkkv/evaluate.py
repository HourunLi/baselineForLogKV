"""ChunkKV, SemanticLogKV and Full-KV through one unmodified SemanticLogKV ``eval.py``."""

import argparse
import ast
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = BASELINE_ROOT.parent / "litgpt"
DEFAULT_CONFIG = "exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml"
DEFAULT_CHUNKKV_CONFIG = BASELINE_ROOT / "configs/chunkkv/qwen3_1.7b_32k.yaml"
CHUNKKV_FIELDS = ("chunk_size", "window_size", "budget", "keep_ratio", "reuse_layers")
BACKENDS = ("chunkkv", "logkv", "dense")


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
    """Private module instance of the reference ``eval.py``; the litgpt checkout stays untouched."""
    configure_environment(root)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("_chunkkv_semanticlogkv_evaluation", root / "eval.py")
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
        raise RuntimeError("Use the evaluation environment (datasets<4)")
    return module


def load_chunkkv_options(path, overrides):
    """Method settings from YAML, then explicit CLI values; validated before any model load."""
    from dataclasses import fields

    from .core import ChunkKVSettings

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
    from .semanticlogkv_eval import file_sha256

    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if result.get("model_backend") != "logkv":
        raise ValueError("Expected a result of `chunkkv.evaluate --backend logkv`")
    stats = result["kv_compression"]["by_request_type"].get("generate_until") or {}
    if not stats.get("request_count"):
        raise ValueError("The SemanticLogKV result has no generate_until KV snapshots")
    entries = stats["retained_slots_per_layer_group"]
    return int(entries), {"result": str(Path(path).resolve()), "sha256": file_sha256(path),
                          "retained_slots_per_layer_group": entries, "requests": stats["request_count"]}


def build_eval_kwargs(evaluator, source_config, model, backend, tasks, limit, output, options=None):
    source = evaluator.expand_env_vars(evaluator._load_yaml_config(str(source_config), str(model or "")))
    # Preserve the configured spelling (including trailing slash) used by RULER metadata/cache keys.
    base = os.path.expanduser(str(source["ckpt_dir"]))
    if not Path(base).is_absolute():
        base = str((Path(evaluator.__file__).parent / base).resolve())
    if model is None:
        if backend == "logkv":
            raise ValueError("--model must name the SemanticLogKV checkpoint for the logkv backend")
        model = base  # ChunkKV is training-free: the original Base by default
    context_length = int(source["context_length"])
    if context_length <= 0:
        raise ValueError("SemanticLogKV context_length must be positive")
    # LogKV method parameters cross over only for LogKV itself or to size a byte-matched budget.
    needs_logkv = backend == "logkv" or (backend == "chunkkv" and (options or {}).get("budget") == "logkv")
    supported = inspect.signature(evaluator.main).parameters
    kwargs = {k: v for k, v in source.items()
              if needs_logkv and k.startswith("log_kv_")
              and not k.startswith("log_kv_diag_") and k in supported}
    if backend == "dense":
        kwargs["log_kv_dense_mode"] = True
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
    from .semanticlogkv_eval import file_sha256

    base = Path(kwargs["tokenizer_dir"])
    tokenizer_names = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                       "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
                       "generation_config.json")
    source_files = ("eval.py", "utils.py", "litgpt/tokenizer.py", "litgpt/generate/base.py",
                    "litgpt/kv_compression.py", "litgpt/ruler_patch.py", "litgpt/model.py")
    return {
        "source_config": str(source_config),
        "source_checkpoint_dir": str(base),
        "source_weight_sha256": file_sha256(base / "lit_model.pth"),
        "source_config_files": {name: file_sha256(base / name)
                                for name in ("config.json", "model_config.yaml")},
        "tokenizer_files": {name: file_sha256(base / name) for name in tokenizer_names
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


def run_evaluation(evaluator, kwargs, backend, contract, options=None, budget_reference=None):
    """Keep reference main's lifecycle/output, then attach baseline-owned provenance."""
    output = Path(kwargs["output_path"]).resolve()
    kwargs = {**kwargs, "output_path": str(output)}
    root = Path(evaluator.__file__).parent.resolve()
    if output.is_relative_to(root):
        raise ValueError("Baseline evaluation outputs must be outside the reference litgpt checkout")
    output.mkdir(parents=True, exist_ok=True)
    is_main = int(os.environ.get("RANK", "0")) == 0
    before = set(output.glob("eval_results_[0-9]*.json")) if is_main else set()
    if backend == "chunkkv":
        install_chunkkv_adapter(evaluator, options)
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
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"Annotated {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="chunkkv")
    parser.add_argument("--model", type=Path,
                        help="LitGPT checkpoint dir; default for chunkkv/dense is the experiment's original Base")
    parser.add_argument("--litgpt-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--semantic-config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--chunkkv-config", type=Path, default=DEFAULT_CHUNKKV_CONFIG)
    parser.add_argument("--budget", help='prompt KV entries per layer/KV head, or "logkv" for byte matching')
    parser.add_argument("--keep-ratio", type=float, help="fraction of prompt tokens kept instead of --budget")
    parser.add_argument("--budget-from-result", type=Path,
                        help="SemanticLogKV result of this launcher; its mean retained KV entries per layer "
                             "and KV head on generate_until become the prompt budget")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--reuse-layers", type=int, help="layer-wise index reuse depth; 1 disables reuse")
    parser.add_argument("--tasks", default="niah_single_2,niah_single_3")
    parser.add_argument("--limit", type=int, help="Same first N samples per task for every method")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-config", action="store_true", help="Resolve shared settings without loading weights")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
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
    root = args.litgpt_root.resolve(strict=True)
    source_config = (root / args.semantic_config).resolve(strict=True)
    model = args.model.resolve() if args.model else None
    output = (args.output or BASELINE_ROOT / "outputs/eval" / args.backend).resolve()
    if output.is_relative_to(root):
        parser.error("Keep baseline outputs outside the reference litgpt checkout")
    evaluator = load_evaluator(root)
    kwargs = build_eval_kwargs(evaluator, source_config, model, args.backend,
                               args.tasks, args.limit, output, options)
    if args.print_config:
        print(json.dumps({"backend": args.backend, "chunkkv": options, **kwargs}, ensure_ascii=False, indent=2))
        return
    contract = None
    if int(os.environ.get("RANK", "0")) == 0:
        contract = comparison_contract(root, source_config, kwargs)
        model_record = evaluated_model(evaluator, kwargs["checkpoint_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "resolved_eval.yaml").write_text(yaml.safe_dump(
            {"backend": args.backend, "chunkkv": options, "budget_reference": budget_reference,
             "evaluation": kwargs, "evaluated_model": model_record, "comparison_contract": contract},
            allow_unicode=True,
        ))
        contract = {**contract, "evaluated_model": model_record}
    run_evaluation(evaluator, kwargs, args.backend, contract, options, budget_reference)


if __name__ == "__main__":
    main()
