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


def full_stages(cfg):
    rows = stages(
        ",".join(reference.LENGTHS), ",".join(reference.SUITES), True, None, cfg["long_limit"], "per_length"
    )
    for row in rows:
        row["optional"] = []
    return rows


def eval_versions():
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "lm-eval", "datasets", "transformers", "tokenizers", "pyarrow")
    }
    if (versions["lm-eval"], versions["datasets"]) != ("0.4.11", "4.8.4"):
        raise RuntimeError(f"Use the reference torch218 evaluation environment: {versions}")
    return versions


def gpu_count():
    import torch

    torch.cuda.init()
    visible = torch.cuda.device_count()
    available = min(visible, int(os.getenv("MA_NUM_GPUS") or visible))
    selected = int(os.getenv("NPROC_PER_NODE") or os.getenv("GPUS_PER_NODE") or available)
    if not 1 <= selected <= available:
        raise ValueError(f"Requested {selected} GPUs, available {available} (visible {visible})")
    return selected


def node_count():
    return int(os.getenv("MA_NUM_HOSTS") or os.getenv("NNODES") or os.getenv("GROUP_WORLD_SIZE") or 1)


def launch_args(nproc, port_offset=0):
    nodes = node_count()
    if nodes == 1:
        return ["--standalone", "--nproc_per_node", str(nproc)]
    rank = int(os.getenv("VC_TASK_INDEX") or os.getenv("NODE_RANK") or os.environ["GROUP_RANK"])
    addr = os.getenv("MASTER_ADDR")
    if os.getenv("MA_VJ_NAME") and os.getenv("MA_TASK_NAME"):
        job = os.environ["MA_VJ_NAME"]
        addr = f"{job}-{os.environ['MA_TASK_NAME']}-{os.getenv('MA_MASTER_INDEX') or '0'}.{job}"
    port = int(os.getenv("MASTER_PORT") or 6001) + port_offset
    if (
        not 0 <= rank < nodes
        or not 1 <= port <= 65535
        or not addr
        or addr in ("localhost", "::1")
        or addr.startswith("127.")
    ):
        raise ValueError(
            f"Invalid multi-node rendezvous: nodes={nodes}, rank={rank}, address={addr}, port={port}"
        )
    return [
        "--nnodes",
        str(nodes),
        "--node_rank",
        str(rank),
        "--master_addr",
        addr,
        "--master_port",
        str(port),
        "--nproc_per_node",
        str(nproc),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT)
    p.add_argument(
        "--full",
        action="store_true",
        help="Require the native main line and every supported suite across all six lengths",
    )
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
    rows = (
        full_stages(cfg)
        if args.full
        else stages(lengths, suites, main_stage, args.limit, long_limit, args.yarn_mode, args.niah_only)
    )
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
        nproc = gpu_count()
        worker_env = dict(os.environ, LITGPT_EXPECTED_WORLD_SIZE=str(nproc * node_count()))
        for i in range(len(rows)):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    *launch_args(nproc, port_offset=3 + i),
                    "-m",
                    "deepseek4.evaluate",
                    *sys.argv[1:],
                    "--stage",
                    str(i),
                ],
                env=worker_env,
                check=True,
            )
        return
    versions = eval_versions()
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
            training={
                k: record[k]
                for k in (
                    "steps",
                    "tokens",
                    "gpu_hours",
                    "objective",
                    "trainable",
                    "precision",
                    "world_size",
                    "accumulation_steps",
                    "data",
                    "config",
                )
                if k in record
            },
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
                    "HF_DATASETS_CACHE",
                    "SOCIAL_IQA_CACHE_DIR",
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
