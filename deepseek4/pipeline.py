"""ModelArts orchestration. Run once per node with the same arguments."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from .evaluate import full_stages, gpu_count, launch_args
from .train import DEFAULT, config


def run(mode, path):
    cfg = config(path)
    nproc = 1 if mode == "prepare" else gpu_count()
    if mode in ("all", "check"):
        # niahLogKV's 8B recipe inherits both held-out PPL sources from the
        # reference job environment. Its documentation examples are not defaults.
        missing = [name for name in ("LONG_PPL_DATA", "LONG_PPL_BOOKS") if not os.getenv(name)]
        if missing:
            raise ValueError(f"Full evaluation needs the reference job's exported {', '.join(missing)}")
        launch_args(nproc, 2 + len(full_stages(cfg)))  # validate the last multi-node port up front

    def workers(module, count, offset, *flags):
        print(f"Starting {module} {' '.join(flags)} ({count} processes/node)", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                *launch_args(count, offset),
                "-m",
                module,
                "--config",
                str(path),
                *flags,
            ],
            check=True,
        )

    if mode in ("all", "check"):
        workers("deepseek4.preflight", 1, 0)
    if mode in ("all", "prepare"):
        workers("deepseek4.train", 1, 1, "--prepare")
    if mode in ("all", "train"):
        workers("deepseek4.train", nproc, 2)
    if mode == "all":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "deepseek4.evaluate",
                "--config",
                str(path),
                "--full",
                "--checkpoint",
                cfg["output_dir"],
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("all", "check", "prepare", "train"))
    parser.add_argument("--config", type=Path, default=DEFAULT)
    args = parser.parse_args()
    run(args.mode, args.config)
