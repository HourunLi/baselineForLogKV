"""One managed_cluster process per node coordinates the existing training and evaluation launchers."""
import os
import subprocess
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    nodes = int(os.environ.get("MA_NUM_HOSTS", os.environ.get("GROUP_WORLD_SIZE", "1")))
    rank = int(os.environ.get("VC_TASK_INDEX", os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", "0"))))
    if not 0 <= rank < nodes:
        raise ValueError(f"Invalid node rank {rank} for {nodes} nodes")
    env = dict(os.environ, MA_NUM_HOSTS=str(nodes), NODE_RANK=str(rank))
    root = Path(env.get("LLA_STUDENT_ROOT") or "checkpoints/qwen8b-hattention").resolve()
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
        for stage in ("convert", "prepare", "probe", "distill", "cpt", "eval"):
            failed = False
            if rank == 0 or stage in {"distill", "cpt", "eval"}:
                print(f"[pipeline node {rank}] {stage}", flush=True)
                command = (["bash", "eval.sh", "configs/qwen8b_hattention_eval.yaml"] if stage == "eval"
                           else ["bash", "scripts/train_hattention.sh", stage])
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
