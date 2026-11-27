"""ModelArts entry point for a budget-matched distributed CPT and evaluation."""
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
    cli.add_argument("--steps", type=int, help="Run a training smoke test in a separate output directory")
    args = cli.parse_args()
    if args.steps is not None and not args.pipeline:
        cli.error("--steps requires --pipeline")
    nodes, rank, env = node_environment()
    if nodes > 1 and not any(os.getenv(k) for k in ("VC_TASK_INDEX", "NODE_RANK", "GROUP_RANK")):
        cli.error("Multi-node ModelArts jobs require a node rank on each host.")
    import torch
    allocated = int(env.get("MA_NUM_GPUS") or torch.cuda.device_count())
    gpus = int(env.get("GPUS_PER_NODE") or allocated)
    if gpus > allocated:
        cli.error(f"Requested {gpus} GPUs per node but only {allocated} allocated")
    cfg = yaml.safe_load(Path("configs/infini_cpt.yaml").read_text())
    world = nodes * gpus
    if gpus < 1 or cfg["global_batch_size"] % world:
        cli.error(f"Global batch {cfg['global_batch_size']} must be divisible by {nodes} nodes x {gpus} GPUs")
    print(f"Topology: {nodes} nodes x {gpus} GPUs, accumulation={cfg['global_batch_size'] // world}", flush=True)
    root = Path(env.get("INFINI_OUTPUT_ROOT") or
                Path("/home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/Infini-Attention") /
                env.get("MA_VJ_NAME", "qwen8b-32k")).resolve()
    checkpoint = "cpt"
    step_args = []
    if args.steps is not None:
        if not 1 <= args.steps < cfg["steps"]:
            cli.error(f"Smoke test steps must be between 1 and {cfg['steps'] - 1}")
        root = root / "smoke" / f"steps-{args.steps}"
        checkpoint = f"trial-{args.steps}"
        step_args = ["--steps", str(args.steps)]
    env.update(INFINI_OUTPUT_ROOT=str(root), INFINI_CHECKPOINT_DIR=str(root / checkpoint),
               GPUS_PER_NODE=str(gpus), INFINI_QWEN_BASE=env.get("INFINI_QWEN_BASE") or cfg["base"])
    env.setdefault("INFINI_OUTPUT_DIR", str(root / "evaluate"))
    print(f"CPT output: {root / checkpoint}\nEvaluation: {env['INFINI_OUTPUT_DIR']}", flush=True)
    train = ["bash", "scripts/train_infini.sh"]
    stages = [
        ("GPU, NCCL, base and shared storage", train + ["check"], False),
        ("offline evaluation environment", ["bash", "eval.sh", "--check-runtime"], False),
        ("offline evaluation assets", ["bash", "eval.sh", "--check-data"], True),
    ]
    if args.pipeline:
        stages.extend([("prepare ProLong", train + ["prepare"], True),
                       ("CPT", train + ["cpt"] + step_args, False),
                       ("evaluation", ["bash", "eval.sh"], False)])
    run_stages(nodes, rank, env, stages)


if __name__ == "__main__":
    main()
