"""One managed_cluster process per node coordinates the existing training and evaluation launchers."""
import os
import subprocess
from datetime import timedelta
from pathlib import Path

import torch
import yaml
import torch.distributed as dist


def remaining_stages(root):
    for stage, remaining in (("cpt", ("eval",)), ("distill", ("prepare", "cpt", "eval"))):
        if not (root / stage).exists():
            print(f"[resume] Missing checkpoint directory: {root / stage}", flush=True)
        else:
            from .train_student import reuse_checkpoint

            config = yaml.safe_load(Path("configs/qwen8b_hattention_train.yaml").read_text())
            expected = dict(stage=stage, recipe=config, actual_steps=config[stage]["steps"])
            if reuse_checkpoint(root / stage, config, expected):
                return remaining
    return ("convert", "prepare", "probe", "distill", "cpt", "eval")


def main():
    nodes = int(os.environ.get("MA_NUM_HOSTS", os.environ.get("GROUP_WORLD_SIZE", "1")))
    rank = int(os.environ.get("VC_TASK_INDEX", os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", "0"))))
    if not 0 <= rank < nodes:
        raise ValueError(f"Invalid node rank {rank} for {nodes} nodes")
    env = dict(os.environ, MA_NUM_HOSTS=str(nodes), NODE_RANK=str(rank))
    configured_root = env.get("LLA_STUDENT_ROOT")
    if not configured_root:
        configured_root = yaml.safe_load(Path("configs/qwen8b_hattention_train.yaml").read_text())["output_root"]
    root = Path(configured_root).resolve()
    if rank == 0:
        print(f"[resume] Student checkpoint root: {root}", flush=True)
    env.update(LLA_STUDENT_ROOT=str(root), LLA_CHECKPOINT_DIR=str(root / "cpt"))
    if nodes > 1:
        addr = env.get("MASTER_ADDR")
        if env.get("MA_VJ_NAME"):
            job = env["MA_VJ_NAME"]
            addr = f"{job}-{env['MA_TASK_NAME']}-{env.get('MA_MASTER_INDEX', '0')}.{job}"
        if not addr or addr == "localhost":
            raise ValueError("Multi-node pipeline requires a reachable master address")
        # Training and evaluation use MASTER_PORT and MASTER_PORT + 1.
        port = int(env.get("MASTER_PORT", "6000")) + 2
        dist.init_process_group("gloo", init_method=f"tcp://{addr}:{port}", rank=rank,
                                world_size=nodes, timeout=timedelta(days=7))
    try:
        decision = [None, None]
        if rank == 0:
            try:
                decision[0] = ("eval",) if env.get("LLA_PIPELINE_EVAL_ONLY") == "1" else remaining_stages(root)
            except Exception as exc:
                decision[1] = str(exc)
        if nodes > 1:
            dist.broadcast_object_list(decision, src=0)
        stages, error = decision
        if error is not None:
            raise SystemExit(f"Cannot resume pipeline: {error}")
        print(f"[pipeline node {rank}] remaining stages: {', '.join(stages)}", flush=True)
        for stage in stages:
            failed = False
            if rank == 0 or stage in {"distill", "cpt", "eval"}:
                print(f"[pipeline node {rank}] {stage}", flush=True)
                command = (["bash", "eval.sh", "configs/qwen8b_hattention_eval.yaml"] if stage == "eval"
                           else ["bash", "scripts/train_hattention.sh", stage])
                if stage == "eval" and nodes > 1:
                    check = subprocess.run([*command[:2], "--preflight", *command[2:]], env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    failures = [None] * nodes
                    dist.all_gather_object(failures, check.stdout[-3000:] if check.returncode else None)
                    if any(message is not None for message in failures):
                        for node, message in enumerate(failures):
                            if rank == 0 and message is not None:
                                print(f"[eval preflight node {node}] failed\n{message}", flush=True)
                        raise SystemExit("Evaluation preflight failed")
                failed = subprocess.run(command, env=env).returncode != 0
            if nodes > 1:
                status = torch.tensor(int(failed))
                dist.all_reduce(status, op=dist.ReduceOp.MAX)
                failed = bool(status.item())
            if failed:
                raise SystemExit(f"[pipeline node {rank}] {stage} failed")
    finally:
        if nodes > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
