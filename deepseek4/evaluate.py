"""Project-local launcher for the pinned niahLogKV protocol."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from ._vendor.niahlogkv import long_eval_plan as reference
from .train import DEFAULT, ROOT, config
from .attention import LAYOUT
from .data import sha256

MAIN_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]


def stages(lengths, suites, main, limit, long_limit, yarn, niah_only=False):
    tasks = yaml.safe_load((ROOT / "configs/deepseek4/tasks.yaml").read_text())
    rows = []

    def add(folder, names, overrides, lens, cap, optional=()):
        if names:
            rows.append(
                dict(
                    folder=folder,
                    benchmark=",".join(names),
                    overrides=overrides,
                    lengths=lens,
                    limit=cap,
                    optional=list(optional),
                )
            )

    skip = []
    if main:
        for group, names in tasks.items():
            if niah_only and group == "main":
                continue
            optional = ("niah_turns_32k", "long_ppl_books_32k") if group == "main" else ()
            add(group, names, {"block_size": 32768}, MAIN_LENGTHS, limit, optional)
        skip = reference.measured(
            "" if niah_only else ",".join(tasks["main"]), ",".join(tasks["niah"]), MAIN_LENGTHS
        )
    if lengths != "none":
        for item in reference.plan(lengths, suites, yarn, niah_only=niah_only, skip=skip):
            for group in ("main", "niah"):
                add(
                    item["output_dir"] + "/" + group,
                    item[group],
                    item["config_overrides"],
                    item["ruler_lengths"],
                    limit or long_limit,
                    item["optional"] if group == "main" else (),
                )
    if not rows:
        raise ValueError("Empty task selection")
    return rows


def launch_args(nproc):
    nodes = int(os.environ.get("MA_NUM_HOSTS", os.environ.get("NNODES", 1)))
    if nodes == 1:
        return ["--standalone", "--nproc_per_node", str(nproc)]
    rank = os.environ.get("VC_TASK_INDEX") or os.environ["NODE_RANK"]
    return [
        "--nnodes",
        str(nodes),
        "--node_rank",
        rank,
        "--master_addr",
        os.environ["MASTER_ADDR"],
        "--master_port",
        os.environ.get("MASTER_PORT", "6001"),
        "--nproc_per_node",
        str(nproc),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--lengths", default=os.getenv("EVAL_LENGTHS"))
    p.add_argument("--suites", default=os.getenv("EVAL_SUITES", ""))
    p.add_argument("--main", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--yarn-mode", choices=("per_length", "fixed"), default=os.getenv("YARN_MODE", "per_length"))
    p.add_argument("--niah-only", action="store_true", default=os.getenv("NIAH_ONLY") == "1")
    p.add_argument("--long-limit", type=int, default=os.getenv("LONG_EVAL_LIMIT"))
    p.add_argument("--limit", type=int, default=os.getenv("LIMIT"))
    p.add_argument("--stage", type=int, help=argparse.SUPPRESS)
    p.add_argument("--print-config", action="store_true")
    args = p.parse_args()
    cfg = config(args.config)
    lengths = args.lengths or ",".join(cfg["eval_lengths"])
    ultra = set(lengths.lower().split(",")) <= {"512k", "1m"}
    main_stage = args.main if args.main is not None else os.getenv("EVAL_MAIN", str(int(not ultra))) == "1"
    suites = args.suites or ("ruler,babilong,books,longbench2" if ultra else "")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    long_limit = cfg["long_limit"] if args.long_limit is None else args.long_limit
    if long_limit < 1:
        p.error("--long-limit must be positive")
    rows = stages(lengths, suites, main_stage, args.limit, long_limit, args.yarn_mode, args.niah_only)
    model = (args.checkpoint or Path(cfg["output_dir"])).resolve()
    tokenizer = Path(cfg["base_checkpoint_dir"])
    output = (args.output or model / "evaluation").resolve()
    if args.print_config:
        print(
            json.dumps(
                dict(checkpoint=str(model), tokenizer=str(tokenizer), layout=LAYOUT, stages=rows), indent=2
            )
        )
        return
    if args.stage is None:
        # Verify the artifact once per job on each node before starting any stage.
        record = json.loads((model / "training.json").read_text())
        if record["method"] != "qwen3-v4flash-36" or record["steps"] < 1 or record["layout"] != list(LAYOUT):
            raise ValueError("Expected a trained Qwen3-V4-Flash checkpoint")
        if sha256(model / "lit_model.pth") != record["weight_sha256"]:
            raise ValueError("Trained checkpoint fingerprint changed")
        for name, digest in record["base"].items():
            if name != "lit_model.pth" and sha256(tokenizer / name) != digest:
                raise ValueError(f"Incompatible Base/tokenizer file: {name}")
        if sha256(model / "model_config.yaml") != record["base"]["model_config.yaml"]:
            raise ValueError("Checkpoint architecture file changed")
        import torch

        nproc = int(os.getenv("NPROC_PER_NODE") or os.getenv("MA_NUM_GPUS") or torch.cuda.device_count())
        if nproc < 1:
            raise RuntimeError("CUDA evaluation requires at least one device")
        for i in range(len(rows)):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    *launch_args(nproc),
                    "-m",
                    "deepseek4.evaluate",
                    *sys.argv[1:],
                    "--stage",
                    str(i),
                ],
                check=True,
            )
        return
    versions = {
        p: importlib.metadata.version(p)
        for p in ("torch", "lm-eval", "datasets", "transformers", "tokenizers", "pyarrow")
    }
    if (versions["lm-eval"], versions["datasets"]) != ("0.4.11", "4.8.4"):
        raise RuntimeError(f"Use the reference torch218 evaluation environment: {versions}")
    row = rows[args.stage]
    folder = output / row["folder"]
    os.environ["EVAL_OPTIONAL_TASKS"] = ",".join(row["optional"])
    from ._vendor.niahlogkv import eval as evaluator

    if int(os.getenv("RANK", 0)) == 0:
        folder.mkdir(parents=True, exist_ok=False)
        record = json.loads((model / "training.json").read_text())
        contract = dict(
            protocol="niahLogKV@0f4396b",
            method=record["method"],
            weight_sha256=record["weight_sha256"],
            base=record["base"],
            tokenizer=str(tokenizer),
            model=str(model),
            stage=row,
            versions=versions,
            code={
                str(f.relative_to(ROOT)): sha256(f)
                for f in (ROOT / "deepseek4").rglob("*")
                if f.suffix in {".py", ".yaml", ".json"}
            },
            data={
                k: os.getenv(k)
                for k in (
                    "HF_HOME",
                    "RULER_CACHE_DIR",
                    "BABILONG_DIR",
                    "LONG_PPL_DATA",
                    "LONG_PPL_BOOKS",
                    "LONG_PPL_DOCS",
                    "LONGBENCH2_DIR",
                    "INFINITEBENCH_DIR",
                )
            },
        )
        (folder / "protocol.json").write_text(json.dumps(contract, indent=2))
    evaluator.main(
        checkpoint_dir=str(model),
        arch_name="Qwen/Qwen3-8B-Base",
        tokenizer_dir=str(tokenizer),
        benchmark=row["benchmark"],
        config_overrides=row["overrides"],
        output_path=str(folder),
        metadata={"pretrained": str(tokenizer), "max_seq_lengths": row["lengths"]},
        limit=row["limit"],
    )


if __name__ == "__main__":
    main()
