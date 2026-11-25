"""Pre-generate the 512K/1M synthetic RULER datasets on CPU for all evaluation ranks."""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from long_eval_plan import LENGTHS, parse_list, ruler_tasks  # noqa: E402


def prepare(job):
    from lm_eval.tasks import TaskManager

    from litgpt.ruler_patch import apply_patch

    length, task, tokenizer = job
    apply_patch()
    manager = TaskManager(include_path=str(ROOT / "eval_tasks"),
                          metadata={"pretrained": tokenizer, "max_seq_lengths": [length]})
    result = manager.load_task_or_group(task)[task]
    return len(result.dataset["test"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path, help="Local tokenizer directory used by eval")
    parser.add_argument("--lengths", default="512k,1m")
    parser.add_argument("--tasks", help="Comma-separated RULER tasks or turns. Defaults to all supported tasks.")
    parser.add_argument("--samples", type=int, default=int(os.environ.get("RULER_LONG_SAMPLES", "100")))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, help="Ordinary shared directory read by evaluation")
    args = parser.parse_args()
    if not args.tokenizer.is_dir() or args.samples < 1 or args.workers < 1:
        parser.error("Provide a local tokenizer directory and positive --samples / --workers.")
    os.environ["RULER_LONG_SAMPLES"] = str(args.samples)
    os.environ["RULER_DATASET_PREPARE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.cache_dir:
        os.environ["RULER_DATASET_CACHE_DIR"] = str(args.cache_dir.resolve())

    tasks = (*ruler_tasks(LENGTHS["512k"]), "turns")
    try:
        lengths = parse_list(args.lengths, ("512k", "1m"), "--lengths")
        selected = parse_list(args.tasks, tasks, "--tasks") if args.tasks else tasks
    except ValueError as error:
        parser.error(str(error))
    jobs = [(LENGTHS[n], f"niah_turns_{n}" if t == "turns" else t, str(args.tokenizer.resolve()))
            for n in lengths for t in selected]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)), mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(prepare, job): job for job in jobs}
        for future in as_completed(futures):
            length, task, _ = futures[future]
            print(f"[RULER CPU prepare] {task} {length}: {future.result()} rows ready", flush=True)


if __name__ == "__main__":
    main()
