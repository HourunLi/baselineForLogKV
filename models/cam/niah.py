"""Evaluate CaM, dense and LogKV on the same checkpoint and niahLogKV length passes."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from models._vendor.niahlogkv import long_eval_plan as reference
from models.qwen3.config import BASELINE_ROOT
from models.qwen3.evaluate import check_environment, configure_environment, run_evaluation, runtime_environment

ROOT = BASELINE_ROOT / "models/_vendor/niahlogkv"
CONFIGS = BASELINE_ROOT / "configs/niah"
MAIN_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]


def stages(args):
    lists = yaml.safe_load((CONFIGS / "tasks.yaml").read_text())
    native_main, native_niah = lists["main"], lists["niah"]
    selected = args.lengths.lower()
    ultra_only = set(selected.replace(",", " ").split()) <= {"512k", "1m"}
    include_main = args.main if args.main is not None else not ultra_only
    suites = args.suites or ("ruler,babilong,books,longbench2" if ultra_only else "")
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
        os.environ.setdefault("CAM_HF_CACHE", os.environ["HF_HOME"])
    configure_environment()
    from models._vendor.niahlogkv import eval as evaluator
    from models.qwen3.provenance import file_sha256
    for name, folder in (("BABILONG_DIR", "babilong"), ("LONGBENCH2_DIR", "longbench2")):
        os.environ.setdefault(name, str(Path(os.environ["HF_HOME"]) / folder))

    os.environ["EVAL_OPTIONAL_TASKS"] = ",".join(row["optional"])
    needs_metadata = any(task.startswith(("niah_", "ruler_", "long_ppl", "infinitebench"))
                         for task in row["benchmark"].split(","))
    kwargs = dict(checkpoint_dir=model, tokenizer_dir=tokenizer, arch_name=config["arch_name"],
                  output_path=str(output / row["folder"]), benchmark=row["benchmark"], limit=row["limit"],
                  config_overrides=row["config_overrides"],
                  metadata={"pretrained": args.metadata_tokenizer, "max_seq_lengths": row["lengths"]} if needs_metadata else None)
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
        if needs_metadata:
            generator_assets = Path(args.metadata_tokenizer)
            if not (generator_assets / "tokenizer.json").is_file():
                raise ValueError(f"Missing RULER tokenizer.json in {generator_assets}. Set --metadata-tokenizer to the reference assets.")
            for name in ("tokenizer.json", "tokenizer_config.json"):
                if name in tokenizer_files and file_sha256(generator_assets / name) != tokenizer_files[name]:
                    raise ValueError(f"Model and task generator tokenizer files differ: {name}")
        contract = dict(protocol="niahLogKV", source_weight_sha256=digest, evaluated_model=record,
                        effective_model_config=asdict(architecture),
                        source_config_files={p.name: file_sha256(p) for p in [checkpoint/"model_config.yaml"]
                                             if p.is_file()}, tokenizer_files=tokenizer_files,
                        evaluation_code={str(p.relative_to(ROOT)): file_sha256(p) for p in ROOT.rglob("*")
                                         if p.is_file() and "__pycache__" not in p.parts},
                        launcher_sha256=file_sha256(__file__),
                        versions={p: importlib.metadata.version(p) for p in
                                  ("torch", "lm-eval", "datasets", "transformers", "tokenizers", "pyarrow")},
                        cache={k: os.getenv(k) for k in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE",
                              "RULER_CACHE_DIR", "RULER_DATASET_CACHE_DIR", "RULER_LONG_SAMPLES", "LONG_PPL_DATA",
                              "LONG_PPL_BOOKS", "LONG_PPL_DOCS", "BABILONG_DIR", "BABILONG_DATASET", "LONGBENCH2_DIR", "INFINITEBENCH_DIR")},
                        benchmark=kwargs["benchmark"], metadata=kwargs["metadata"], limit=kwargs["limit"],
                        max_seq_length=row["config_overrides"]["block_size"],
                        rope_adjustments=row["config_overrides"].get("rope_adjustments"))
        resolved = dict(stage=row, reference_config=config, environment=runtime_environment(), evaluated_model=record)
        if args.backend == "cam":
            resolved["cam_code_sha256"] = {name: file_sha256(Path(__file__).with_name(name))
                                           for name in ("cache.py", "grouped.py", "niah_model.py", "niah_budget.py")}
    if args.backend == "cam":
        from .niah_model import install_adapter
        install_adapter(evaluator, max(row["lengths"]), args.budget, args.merge_interval, args.prefill_chunk)
    run_evaluation(evaluator, kwargs, args.backend, contract, resolved=resolved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cam", "dense", "logkv"), default="cam")
    parser.add_argument("--model-size", choices=("1.7b", "8b"), default="1.7b")
    parser.add_argument("--model", type=Path, help="LitGPT checkpoint; all backends default to the same Gamma CPT")
    parser.add_argument("--tokenizer", type=Path, help="Model tokenizer directory (defaults to original Base)")
    parser.add_argument("--metadata-tokenizer", type=Path, help="Task generator tokenizer (reference eval.sh: SAVE_DIR)")
    parser.add_argument("--budget", choices=("logkv", "target-bytes", "target-slots"), default="logkv")
    parser.add_argument("--merge-interval", type=int, default=1,
                        help="1 = online CaM; larger values defer eviction within each prefill block")
    parser.add_argument("--prefill-chunk", type=int, default=256)
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
    args = parser.parse_args()
    if not 1 <= args.merge_interval <= args.prefill_chunk:
        parser.error("require 1 <= merge-interval <= prefill-chunk")
    if args.long_limit < 1 or (args.limit is not None and args.limit < 1):
        parser.error("sample limits must be positive")
    try:
        rows = stages(args)
    except ValueError as error:
        parser.error(str(error))
    config = yaml.safe_load((CONFIGS/f"qwen{args.model_size}.yaml").read_text())
    args.weight_role = "explicit" if args.model else "gamma_cpt"
    model = str((args.model or Path(config["save_path"])).resolve())
    tokenizer = str((args.tokenizer or Path(config["ckpt_dir"])).resolve())
    args.metadata_tokenizer = str((args.metadata_tokenizer or Path(model)).resolve())
    tag = f"cam-{args.budget}-step{args.merge_interval}" if args.backend == "cam" else args.backend
    output = (args.output or BASELINE_ROOT / "outputs/niahlogkv" / args.model_size / tag).resolve()
    if args.print_config:
        print(json.dumps(dict(checkpoint=model, weight_role=args.weight_role, tokenizer=tokenizer,
                              metadata_tokenizer=args.metadata_tokenizer, backend=args.backend,
                              reference_config=config, budget=args.budget, merge_interval=args.merge_interval,
                              prefill_chunk=args.prefill_chunk, output=str(output), stages=rows), indent=2))
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
                            "-m", "models.cam.niah", *sys.argv[1:], "--stage", str(i)], check=True)


if __name__ == "__main__":
    main()
