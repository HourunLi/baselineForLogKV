"""Check the full offline task plan before allocating optimizer state."""

import argparse
from datetime import timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

from .evaluate import eval_versions, full_stages
from .train import config, DEFAULT


def check(cfg, rank):
    from litdata import StreamingDataset, TokensLoader  # noqa: F401
    from torch.distributed.fsdp import FullyShardedDataParallel  # noqa: F401
    from ._vendor.niahlogkv import eval as evaluator
    from ._vendor.niahlogkv.eval_tasks.long_ppl import long_ppl_utils as ppl
    from ._vendor.niahlogkv.eval_tasks.babilong_long import babilong_long_utils as babi
    from ._vendor.niahlogkv.litgpt import Tokenizer, ruler_patch
    from .data import DATA_MANIFEST, validate_prepared_data, corpus_files
    import nltk

    versions = eval_versions()
    versions.update({n: importlib.metadata.version(n) for n in ("lightning", "litdata", "jsonargparse")})
    print(
        json.dumps(
            dict(
                python=sys.version,
                executable=sys.executable,
                versions=versions,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0),
                data={
                    k: os.getenv(k)
                    for k in (
                        "HF_HOME",
                        "HF_DATASETS_CACHE",
                        "BABILONG_DIR",
                        "LONGBENCH2_DIR",
                        "INFINITEBENCH_DIR",
                        "LONG_PPL_DATA",
                        "LONG_PPL_BOOKS",
                        "LONG_PPL_DOCS",
                        "RULER_CACHE_DIR",
                        "NLTK_DATA",
                    )
                },
            ),
            default=str,
        ),
        flush=True,
    )
    if not torch.cuda.is_bf16_supported() or not dist.is_nccl_available():
        raise RuntimeError("Training requires BF16 CUDA and NCCL")
    base = Path(cfg["base_checkpoint_dir"])
    for name in ("lit_model.pth", "model_config.yaml"):
        if not (base / name).is_file():
            raise FileNotFoundError(base / name)
    Tokenizer(base)
    if (Path(cfg["tokenized_dir"]) / DATA_MANIFEST).is_file():
        validate_prepared_data(cfg, verify_source=True)
    elif not corpus_files(Path(cfg["data_dir"]))[0]:
        raise FileNotFoundError(f"No ProLong training documents in {cfg['data_dir']}")

    # Exercise the installed LongBench metrics without editing shared packages.
    from importlib import import_module

    for module in ("jieba", "fuzzywuzzy.fuzz", "rouge"):
        import_module(module)

    # Check punkt first: IFEval otherwise tries nltk.download at import time.
    nltk.data.find("tokenizers/punkt_tab/english/")
    nltk.sent_tokenize("First sentence. Second sentence.")
    rows = full_stages(cfg)
    manager = evaluator._task_manager({"pretrained": str(base), "max_seq_lengths": [32768]})
    names = {name for row in rows for name in row["benchmark"].split(",")}
    missing = names.difference(manager.all_tasks)
    if missing:
        raise ValueError(f"Unregistered evaluation tasks: {sorted(missing)}")
    print(f"Registered {len(names)} distinct entries across {len(rows)} stages", flush=True)

    for env in ("LONG_PPL_DATA", "LONG_PPL_BOOKS"):
        try:
            ppl.validate_inputs(os.environ[env])
        except Exception as exc:
            raise RuntimeError(f"{env}={os.environ.get(env)}: {exc}") from exc

    # Exercise the harness's own dataset/metric imports on the native tasks.
    # Synthetic long inputs are checked from their assets, without generating
    # every 1M prompt a second time.
    for name in rows[0]["benchmark"].split(","):
        if name == "niah_turns_32k" or name.startswith("long_ppl_books_"):
            continue
        print(f"Checking native task {name}", flush=True)
        manager.load_task_or_group(name)

    for filename in ruler_patch._URL_TO_FILENAME.values():
        if not ruler_patch._load_json_local(filename):
            raise FileNotFoundError(Path(ruler_patch._get_ruler_cache_dir()) / filename)
    ruler_patch._read_cwe_wordlist(ruler_patch._cwe_wordlist_file())
    essays = ruler_patch._essay_file()
    if not essays.is_file() or not essays.stat().st_size:
        raise FileNotFoundError(essays)
    evaluator._check_task_data(["longbench2_1m", "infinitebench_256k"])
    from ._vendor.niahlogkv.eval_tasks.infinitebench import infinitebench_utils as ib

    for task in ib.TASKS:
        path = ib.data_file(task)
        with open(path, encoding="utf-8") as f:
            first = next((line for line in f if line.strip()), None)
        if first is None:
            raise ValueError(f"Empty InfiniteBench data: {path}")
        example = json.loads(first)
        ib.prompt_of(task, example)
        ib.answer_of(task, example)
    from ._vendor.niahlogkv.litgpt import longbench2_patch as lb2

    for subset in lb2.subsets():
        if not len(lb2.load_subset(subset, split="train")):
            raise ValueError(f"Empty LongBench v2 subset: {subset}")
    for length in ("32k", "64k", "128k", "256k", "512k", "1m"):
        for i in range(1, 6):
            split = f"qa{i}"
            options = {"length": length, "qa_split": split}
            if length == "256k":
                options["dataset"] = "RMT-team/babilong"
            if not len(babi.load_dataset(**options)[split]):
                raise ValueError(f"Empty BABILong {length}/{split}")

    # Selection uses the exact evaluator tokenizer and cache key; the result
    # can be reused in evaluation. Only one node populates the shared cache.
    if rank == 0:
        for env, minima in (
            ("LONG_PPL_DATA", (ppl.LONGEST, ppl.LONGEST_256K, ppl.LONGEST_1M)),
            ("LONG_PPL_BOOKS", (ppl.LONGEST_256K, ppl.LONGEST_1M)),
        ):
            for minimum in minima:
                docs = ppl._documents({"pretrained": str(base)}, minimum, env=env)
                if not docs or min(map(len, docs)) < minimum:
                    raise ValueError(f"{env} does not supply documents at {minimum} tokens")
                print(f"{env}: {len(docs)} documents at {minimum} tokens", flush=True)
                del docs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT)
    args = parser.parse_args()
    multi = int(os.getenv("WORLD_SIZE", 1)) > 1
    if multi:
        dist.init_process_group("gloo", timeout=timedelta(hours=12))
    try:
        check(config(args.config), int(os.getenv("RANK", 0)))
        if multi:
            dist.barrier()
        print("Full evaluation preflight passed on this node", flush=True)
    finally:
        if multi:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
