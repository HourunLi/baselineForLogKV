"""Evaluate Base ChunkKV and CPT LogKV/dense on the same reference length passes."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from ._vendor.niahlogkv import long_eval_plan as reference
from .config import BASELINE_ROOT
from .evaluate import BACKENDS, check_environment, configure_environment, run_evaluation, runtime_environment

ROOT = Path(__file__).parent / "_vendor/niahlogkv"
CONFIGS = BASELINE_ROOT / "configs/chunkkv/niah"
MAIN_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]


def stages(args):
    lists = yaml.safe_load((CONFIGS / "tasks.yaml").read_text())
    native_main, native_niah = lists["main"], lists["niah"]
    selected = args.lengths.lower()
    ultra_only = {name for name in selected.replace(",", " ").split()} <= {"512k", "1m"}
    include_main = args.main if args.main is not None else not ultra_only
    suites = args.suites or ("ruler,babilong,ppl,books,turns,longbench2" if ultra_only else "")
    rows = []

    def add(folder, tasks, overrides, lengths, limit, optional=()):
        if tasks:
            rows.append(dict(folder=folder, benchmark=",".join(tasks), config_overrides=overrides,
                             lengths=lengths, limit=limit, optional=list(optional)))

    if include_main:
        if not args.niah_only:
            add("main", native_main, {"block_size": 32768}, MAIN_LENGTHS, args.limit,
                ("niah_turns_32k", "long_ppl_books_32k"))
        add("niah", native_niah, {"block_size": 32768}, MAIN_LENGTHS, args.limit)
    if selected != "none":
        skip = reference.measured(",".join([] if args.niah_only else native_main),
                                  ",".join(native_niah), MAIN_LENGTHS) if include_main else []
        for p in reference.plan(selected, suites, args.yarn_mode, args.niah_only, skip=skip):
            for group in ("main", "niah"):
                add(f"{p['output_dir']}/{group}", p[group], p["config_overrides"], p["ruler_lengths"],
                    args.limit or args.long_limit, p["optional"] if group == "main" else [])
    if not rows:
        raise ValueError("No evaluation stages selected")
    return rows


def run_stage(args, row, config, model, tokenizer, output):
    check_environment()
    if os.getenv("HF_HOME"):
        os.environ.setdefault("CHUNKKV_HF_CACHE", os.environ["HF_HOME"])
    configure_environment()
    from ._vendor.niahlogkv import eval as evaluator
    from .semanticlogkv_eval import file_sha256

    os.environ["EVAL_OPTIONAL_TASKS"] = ",".join(row["optional"])
    options = dict(args.chunkkv_options)
    if "kv_capacity_per_layer" in row:
        options["budget"] = row["kv_capacity_per_layer"]
    needs_metadata = any(task.startswith(("niah_", "ruler_", "long_ppl", "infinitebench"))
                         for task in row["benchmark"].split(","))
    kwargs = dict(checkpoint_dir=model, tokenizer_dir=tokenizer, arch_name=config["arch_name"],
                  output_path=str(output / row["folder"]), benchmark=row["benchmark"], limit=row["limit"],
                  config_overrides=row["config_overrides"],
                  metadata={"pretrained": tokenizer, "max_seq_lengths": row["lengths"]} if needs_metadata else None)
    if args.backend == "dense":
        kwargs["log_kv_dense_mode"] = True
    else:
        kwargs.update({k: v for k, v in config.items() if k.startswith("log_kv_")})
    contract = resolved = None
    if int(os.environ.get("RANK", "0")) == 0:
        checkpoint = Path(evaluator._resolve_checkpoint_dir(model))
        kwargs["checkpoint_dir"] = str(checkpoint)  # rank zero broadcasts this step inside eval.main
        binary = evaluator._checkpoint_file_for_loading(checkpoint / "lit_model.pth")
        digest = file_sha256(binary)
        # Pin the selected step for this launch even if another checkpoint arrives later.
        record = dict(checkpoint_dir=str(checkpoint), binary=str(binary), lit_model_sha256=digest,
                      weight_role=args.weight_role)
        architecture = evaluator._config_for_evaluation(str(checkpoint), row["config_overrides"], config["arch_name"])
        evaluator._configure_inference_memory(architecture, row["config_overrides"])
        tokenizer_files = {p.name: file_sha256(p) for p in Path(tokenizer).iterdir()
                           if p.is_file() and p.suffix in (".json", ".model", ".txt", ".jinja")}
        contract = dict(protocol="niahLogKV", source_weight_sha256=digest, evaluated_model=record,
                        effective_model_config=asdict(architecture),
                        source_config_files={p.name: file_sha256(p) for p in [checkpoint/"model_config.yaml"]
                                             if p.is_file()}, tokenizer_files=tokenizer_files,
                        evaluation_code={str(p.relative_to(ROOT)): file_sha256(p) for p in ROOT.rglob("*")
                                         if p.is_file() and "__pycache__" not in p.parts},
                        compatibility_code={p: file_sha256(Path(__file__).with_name(p)) for p in
                                            ("niah.py", "evaluate.py", "semanticlogkv_eval.py")},
                        versions={p: importlib.metadata.version(p) for p in
                                  ("torch", "lm-eval", "datasets", "transformers", "tokenizers", "pyarrow")},
                        cache={k: os.getenv(k) for k in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE",
                              "RULER_CACHE_DIR", "RULER_DATASET_CACHE_DIR", "RULER_LONG_SAMPLES", "LONG_PPL_DATA",
                              "LONG_PPL_BOOKS", "LONG_PPL_DOCS", "BABILONG_DIR", "BABILONG_DATASET", "LONGBENCH2_DIR", "INFINITEBENCH_DIR")},
                        benchmark=kwargs["benchmark"], metadata=kwargs["metadata"], limit=kwargs["limit"],
                        max_seq_length=row["config_overrides"]["block_size"],
                        rope_adjustments=row["config_overrides"].get("rope_adjustments"))
        resolved = dict(stage=row, reference_config=config, environment=runtime_environment(), evaluated_model=record)
        if args.backend == "chunkkv":
            resolved["chunkkv_code_sha256"] = {name: file_sha256(Path(__file__).with_name(name))
                                               for name in ("model.py", "core.py", "settings.py")}
    run_evaluation(evaluator, kwargs, args.backend, contract, options, resolved=resolved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="chunkkv")
    parser.add_argument("--reference", choices=("baseline", "word_first", "word_first_query", "word_first_query_dual"),
                        default="baseline", help="reference eval_full_<method> preset for LogKV and budget sizing")
    parser.add_argument("--model-size", choices=("1.7b", "8b"), default="1.7b")
    parser.add_argument("--model", type=Path, help="Override the backend default checkpoint (ChunkKV: Base, LogKV/dense: gamma CPT)")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--lengths", default=os.getenv("EVAL_LENGTHS") or ",".join(reference.LENGTHS))
    parser.add_argument("--suites", default=os.getenv("EVAL_SUITES"), help="comma-separated long-context suites")
    parser.add_argument("--main", action=argparse.BooleanOptionalAction,
                        default=None if "EVAL_MAIN" not in os.environ else os.environ["EVAL_MAIN"] == "1")
    parser.add_argument("--niah-only", action="store_true", default=os.getenv("NIAH_ONLY") == "1")
    parser.add_argument("--yarn-mode", choices=("per_length", "fixed"), default=os.getenv("YARN_MODE","per_length"))
    parser.add_argument("--limit", type=int, default=os.getenv("LIMIT"))
    parser.add_argument("--long-limit", type=int, default=os.getenv("LONG_EVAL_LIMIT") or 100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--stage", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--job-config", type=Path)
    original = sys.argv[1:]
    job_parser = argparse.ArgumentParser(add_help=False)
    job_parser.add_argument("--job-config", type=Path)
    job_path = job_parser.parse_known_args(original)[0].job_config
    job = {}
    defaults = []
    if job_path:
        job = yaml.safe_load(job_path.read_text())
        if not isinstance(job, dict) or job.get("protocol") != "niahLogKV":
            parser.error("expected protocol: niahLogKV in job YAML")
        cli_fields = ("backend", "model_size", "reference", "lengths", "suites", "limit", "long_limit", "output")
        unknown = set(job) - set(cli_fields) - {"protocol", "base_checkpoint", "cpt_checkpoint", "chunkkv", "main", "budget_by_length"}
        if unknown:
            parser.error(f"unknown job fields: {sorted(unknown)}")
        for field in cli_fields:
            value = job.get(field)
            if value is not None:
                if isinstance(value, list):
                    value = ",".join(map(str, value))
                defaults.extend(["--" + field.replace("_", "-"), str(value)])
        if "main" in job:
            if not isinstance(job["main"], bool):
                parser.error("main must be a YAML boolean")
            defaults.append("--main" if job["main"] else "--no-main")
    args = parser.parse_args(defaults + original)
    if args.long_limit < 1 or (args.limit is not None and args.limit < 1):
        parser.error("sample limits must be positive")
    try:
        rows = stages(args)
    except ValueError as error:
        parser.error(str(error))
    config = yaml.safe_load((CONFIGS/f"qwen{args.model_size}.yaml").read_text())
    for field, key in (("base_checkpoint", "ckpt_dir"), ("cpt_checkpoint", "save_path")):
        if field in job:
            config[key] = job[field]
    tuning = job.get("chunkkv", {})
    if not isinstance(tuning, dict) or set(tuning) - {"chunk_size", "window_size", "reuse_layers", "prefill_chunk_size"}:
        parser.error("invalid chunkkv settings in job YAML")
    if any(type(value) is not int or value < 1 for value in tuning.values()):
        parser.error("chunkkv sizes must be positive integers")
    args.chunkkv_options = dict(chunk_size=10, window_size=64, reuse_layers=1, streaming=True,
                                prefill_chunk_size=config["log_kv_prefill_block"], budget="logkv")
    args.chunkkv_options.update(tuning)
    budgets = job.get("budget_by_length")
    if budgets is not None and args.backend == "chunkkv":
        if not isinstance(budgets, dict):
            parser.error("budget_by_length must map nominal context lengths to KV capacities")
        for row in rows:
            length = max(row["lengths"])
            capacity = budgets.get(length)
            if type(capacity) is not int or not args.chunkkv_options["window_size"] < capacity <= length:
                parser.error(f"invalid or missing KV capacity for length {length}")
            row["kv_capacity_per_layer"] = capacity
    # These eval_full configurations inherit the same gamma checkpoint and change only admission.
    if args.reference != "baseline":
        config["log_kv_gamma_surprisal"] = "word_first"
    if args.reference in ("word_first_query", "word_first_query_dual"):
        config.update(log_kv_gamma_query_cue=True, log_kv_gamma_cue_guard=True)
    if args.reference == "word_first_query_dual":
        config["log_kv_gamma_query_pool_tokens"] = 512
    default_base = args.backend == "chunkkv"
    default_checkpoint = config["ckpt_dir" if default_base else "save_path"]
    args.weight_role = "explicit" if args.model else ("base" if default_base else "cpt")
    model = str((args.model or Path(default_checkpoint)).resolve())
    tokenizer = str((args.tokenizer or Path(config["ckpt_dir"])).resolve())
    output = (args.output or BASELINE_ROOT/"outputs/niahlogkv"/args.model_size/args.reference/args.backend).resolve()
    if args.print_config:
        print(json.dumps(dict(checkpoint=model, weight_role=args.weight_role, tokenizer=tokenizer, backend=args.backend,
                              reference_config=config, chunkkv=args.chunkkv_options, output=str(output), stages=rows), indent=2))
    elif args.stage is not None:
        run_stage(args, rows[args.stage], config, model, tokenizer, output)
    else:
        nodes = int(os.getenv("MA_NUM_HOSTS") or os.getenv("GROUP_WORLD_SIZE") or 1)
        gpus = os.getenv("MA_NUM_GPUS") or os.getenv("NPROC_PER_NODE") or "1"
        rendezvous = ["--standalone"]
        if nodes > 1:
            rank = os.getenv("VC_TASK_INDEX") or os.environ["NODE_RANK"]
            rendezvous = ["--nnodes", str(nodes), "--node_rank", rank, "--master_addr", os.environ["MASTER_ADDR"]]
            base_port = int(os.getenv("EVAL_MASTER_PORT") or os.getenv("MASTER_PORT") or 6001)
            if not 1 <= base_port <= 65536 - len(rows):
                parser.error("the rendezvous port range for the selected stages exceeds 1..65535")
        for i, row in enumerate(rows):
            print(f"Running {row['folder']} in {output}", flush=True)
            stage_port = ["--master_port", str(base_port + i)] if nodes > 1 else []
            subprocess.run([sys.executable, "-m", "torch.distributed.run", *rendezvous, *stage_port, "--nproc_per_node", gpus,
                            "-m", "chunkkv.niah", *sys.argv[1:], "--stage", str(i)], check=True)


if __name__ == "__main__":
    main()
