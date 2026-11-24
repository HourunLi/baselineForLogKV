"""managed_cluster entry point for a budget-matched four-node CPT and evaluation."""
import argparse
import os
from pathlib import Path

import yaml

from lla_bench.pipeline import node_environment, run_stages


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_mutually_exclusive_group(required=True)
    modes.add_argument("--pipeline", action="store_true")
    modes.add_argument("--check", action="store_true")
    args = cli.parse_args()
    nodes, rank, env = node_environment()
    if nodes != 4 or not any(os.getenv(k) for k in ("VC_TASK_INDEX", "NODE_RANK", "GROUP_RANK")):
        cli.error("Expected 4 managed_cluster nodes with MA_NUM_HOSTS=4 and a node rank per host.")
    if any(int(env.get(k) or 8) != 8 for k in ("GPUS_PER_NODE", "MA_NUM_GPUS")):
        cli.error("This job requires eight GPUs per node.")
    cfg = yaml.safe_load(Path("configs/nfini_cpt.yaml").read_text())
    root = Path(env.get("NFINI_OUTPUT_ROOT") or
                Path("/workspace/ckpt-logKV/nfini-attention") /
                env.get("MA_VJ_NAME", "qwen8b-32k")).resolve()
    env.update(NFINI_OUTPUT_ROOT=str(root), NFINI_CHECKPOINT_DIR=str(root / "cpt"),
               GPUS_PER_NODE="8", NFINI_QWEN_BASE=env.get("NFINI_QWEN_BASE") or cfg["base"])
    env.setdefault("NFINI_OUTPUT_DIR", str(root / "evaluate"))
    print(f"CPT output: {root}/cpt\nEvaluation: {env['NFINI_OUTPUT_DIR']}", flush=True)
    train = ["bash", "scripts/train_nfini.sh"]
    stages = [
        ("GPU, NCCL, base and shared storage", train + ["check"], False),
        ("offline evaluation environment", ["bash", "eval.sh", "--check-runtime"], False),
        ("offline evaluation assets", ["bash", "eval.sh", "--check-data"], True),
    ]
    if args.pipeline:
        stages.extend([("prepare ProLong", train + ["prepare"], True),
                       ("CPT", train + ["cpt"], False),
                       ("evaluation", ["bash", "eval.sh"], False)])
    run_stages(nodes, rank, env, stages)


if __name__ == "__main__":
    main()
