"""Check, offline and read-only, that the long-context evaluation data is complete (what fetch_long_eval_data.py
put into eval.sh's caches). Run it the way eval.sh runs, with diag.sh's offline environment:

    bash diag.sh unused/check_long_eval_data.py [--tokenizer <Qwen3-1.7B-Base dir>]

with LONG_PPL_DATA / LONG_PPL_BOOKS exported as for eval.sh. Each line says OK or MISSING / BAD:
    BABILong      RMT-team/babilong-1k-samples 32k / 64k / 128k and RMT-team/babilong 256k, qa1 ... qa5 (rows)
    LongBench v2  the hub snapshot's files are whole parquet (through flattened links to their blobs, as eval.py
                  reads them), and every subset lm-eval's longbench2 reads loads (503 questions in all)
    RULER QA      SQuAD v2 dev and HotpotQA dev in RULER_CACHE_DIR
    InfiniteBench the ten tasks' jsonl files (INFINITEBENCH_DIR, or the hub cache) are whole, and every sample
                  makes a prompt and an answer
    long_ppl      LONG_PPL_DATA: documents of >= 131072 and >= 262144 tokens
    books         LONG_PPL_BOOKS: books of >= 262144 tokens (want LONG_PPL_DOCS, 100)
The ppl lines need --tokenizer (they read the cached selection, or select again when it is missing).
--storage-only checks environment paths and up to three raw files per PPL source on this host, without
loading a tokenizer, selecting documents, downloading files, or modifying storage. Run on each job node.
Exits 1 when anything is missing.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BABILONG = [("RMT-team/babilong-1k-samples", n) for n in ("32k", "64k", "128k")] + [("RMT-team/babilong", "256k")]
RULER_QA = ("dev-v2.0.json", "hotpot_dev_distractor_v1.json")


def whole_parquet(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4)
            f.seek(-4, os.SEEK_END)
            return path.stat().st_size >= 12 and head == b"PAR1" and f.read(4) == b"PAR1"
    except OSError:
        return False


def check_storage(report):
    """Distinguish missing job mounts/paths from PPL token-cache failures with stdlib-only reads."""
    import itertools
    import socket

    from eval_tasks.long_ppl.input_files import describe, file_state, resolve_input

    print(f"[storage] host={socket.gethostname()} rank={os.environ.get('RANK', 'unset')} "
          f"uid={os.geteuid()} cwd={Path.cwd()}", flush=True)
    print(f"[storage] HF_HOME={os.environ.get('HF_HOME', '<unset>')}", flush=True)
    for env in ("LONG_PPL_BOOKS", "LONG_PPL_DATA"):
        source = os.environ.get(env)
        print(f"[storage] {env}={source or '<unset>'}", flush=True)
        if not source:
            report(False, f"{env}: not exported in this job process")
            continue
        path = Path(os.path.abspath(os.path.expanduser(source)))
        nearest = path
        try:
            while not nearest.exists() and not nearest.is_symlink() and nearest != nearest.parent:
                nearest = nearest.parent
            print(f"[storage] nearest existing path: {nearest}", flush=True)
            mount, filesystem = Path("/"), "unknown"
            try:
                for line in Path("/proc/self/mountinfo").read_text().splitlines():
                    before, after = line.split(" - ", 1)
                    candidate = Path(before.split()[4].replace("\\040", " ").replace("\\134", "\\"))
                    if (nearest == candidate or candidate in nearest.parents) and len(candidate.parts) >= len(mount.parts):
                        mount, filesystem = candidate, after.split()[0]
                print(f"[storage] mount={mount} filesystem={filesystem}", flush=True)
            except (OSError, ValueError, IndexError) as error:
                print(f"[storage] mount info unavailable: {error}", flush=True)
            path.stat()  # preserve PermissionError instead of reporting every failure as a missing folder
            if path.is_file():
                files = [path]
            elif path.is_dir():
                files = list(itertools.islice((p for p in path.rglob("*")
                                              if p.suffix in (".parquet", ".jsonl")), 3))
            else:
                raise ValueError("not a regular file or directory")
            if not files:
                report(False, f"{env}: directory exists but contains no visible parquet/jsonl files: {path}")
                continue
            for filename in files:
                actual = resolve_input(filename, report=lambda message: print(message, flush=True))
                state = file_state(actual)
                if state[0] == 0:
                    raise ValueError(f"empty input file: {actual}")
                if filename.suffix == ".parquet":
                    size, head, footer = state
                    if not (size >= 12 and head[:4] == footer == b"PAR1"):
                        raise ValueError(f"invalid Parquet markers: {describe(actual, state)}")
                report(True, f"{env}: readable input={filename}; {describe(actual, state)}")
        except (OSError, ValueError, RuntimeError) as error:
            report(False, f"{env}: {type(error).__name__}: {error}")
            if nearest.is_dir():
                try:
                    children = [p.name for p in itertools.islice(nearest.iterdir(), 10)]
                    print(f"[storage] nearest directory entries (up to 10): {children}", flush=True)
                except OSError as listing_error:
                    print(f"[storage] directory listing failed: {listing_error}", flush=True)
    print("[storage] This checks visibility on this host; platform mount/data-preload errors need platform logs. "
          "It does not verify document counts or warm the tokenizer cache.", flush=True)


def check_babilong(report, lengths=None) -> None:
    import datasets

    for path, length in BABILONG:
        if lengths and length not in lengths:
            continue
        for split in (f"qa{i}" for i in range(1, 6)):
            try:
                rows = len(datasets.load_dataset(path, name=length, split=split))
                report(rows > 0, f"BABILong {path} {length} {split}: {rows} rows")
            except Exception as error:  # noqa: BLE001
                report(False, f"BABILong {path} {length} {split}: {type(error).__name__}: {str(error)[:200]}")
    if lengths:
        from unused.long_eval_download import babi_utils
        utils = babi_utils()
        for length in (n for n in lengths if n in ("512k", "1m")):
            for split in (f"qa{i}" for i in range(1, 6)):
                try:
                    data = utils.load_plain(length, split)
                    report(len(data) == 100, f"BABILong {length} {split}: {len(data)} ordinary-file rows")
                except Exception as error:
                    report(False, f"BABILong {length} {split}: {error}")


def check_longbench2(report) -> None:
    import datasets

    from litgpt.longbench2_patch import REPO, apply_patch, snapshot, subsets

    try:
        root, files = snapshot()
    except Exception as error:  # noqa: BLE001
        report(False, f"LongBench v2: no snapshot in the hub cache ({type(error).__name__}: {str(error)[:200]})")
        return
    bad = [f for f in files if f.endswith(".parquet")
           and ((Path(root) / f).is_symlink() or not whole_parquet(Path(root) / f))]
    report(bool(files) and not bad, f"LongBench v2 plain directory {root}: {len(files)} files"
           + (f", not ordinary whole parquet: {bad}" if bad else ", all ordinary whole parquet"))
    apply_patch()  # as eval.py
    names = subsets()
    total = 0
    for name in names:
        try:
            rows = len(datasets.load_dataset(REPO, name, split="train"))
            total += rows
            report(rows > 0, f"LongBench v2 {name}: {rows} questions")
        except Exception as error:  # noqa: BLE001
            report(False, f"LongBench v2 {name}: {type(error).__name__}: {str(error)[:200]}")
    report(total == 503, f"LongBench v2: {total} questions in {len(names)} subsets (want 503)")


def check_ruler_qa(report, extended=False) -> None:
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    folder = Path(os.environ.get("RULER_CACHE_DIR", os.path.join(home, "ruler_cache")))
    for name in RULER_QA:
        path = folder / name
        report(path.is_file() and path.stat().st_size > 1e6,
               f"RULER QA {path}: " + (f"{path.stat().st_size / 1e6:.1f} MB" if path.is_file() else "missing"))
    if extended:
        from litgpt.ruler_patch import _essay_file
        path = _essay_file()
        report(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
               f"RULER ordinary essay background: {path}")
        nltk = Path(os.environ.get("NLTK_DATA", str(folder.parent / "nltk_data")).split(os.pathsep)[0])
        path = nltk / "tokenizers/punkt_tab/english/collocations.tab"
        report(path.is_file() and not path.is_symlink(), f"RULER ordinary NLTK data: {path}")


def check_ruler_cwe(report) -> None:
    from litgpt.ruler_patch import _cwe_wordlist_file, _read_cwe_wordlist

    path = _cwe_wordlist_file()
    try:
        _read_cwe_wordlist(path)
    except (OSError, ValueError) as error:
        report(False, str(error))
    else:
        report(True, f"RULER CWE NVIDIA word list, SHA256 verified: {path}")


def check_infinitebench(report) -> None:
    import json

    spec = importlib.util.spec_from_file_location(
        "infinitebench_utils", ROOT / "eval_tasks" / "infinitebench" / "infinitebench_utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    for task in utils.TASKS:
        try:
            path = utils.data_file(task)
            rows = 0
            with open(path, encoding="utf8") as f:
                for line in f:
                    if line.strip():
                        eg = json.loads(line)
                        utils.prompt_of(task, eg), utils.answer_of(task, eg)
                        rows += 1
            report(rows > 0, f"InfiniteBench {task}: {rows} samples in {path}")
        except Exception as error:  # noqa: BLE001
            report(False, f"InfiniteBench {task}: {type(error).__name__}: {str(error)[:800]}")


def check_ppl(report, tokenizer: str | None, lengths=None, suites=None) -> None:
    spec = importlib.util.spec_from_file_location("long_ppl_utils", ROOT / "eval_tasks" / "long_ppl" / "long_ppl_utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    want = int(os.environ.get("LONG_PPL_DOCS", "100"))
    checks = []
    lengths = lengths or ("32k", "64k", "128k", "256k")
    suites = suites or ("ppl", "books")
    if "ppl" in suites:
        if any(n in ("32k", "64k", "128k") for n in lengths):
            checks.append(("LONG_PPL_DATA", utils.LONGEST, "long_ppl 32k-128k", 1))
        if "256k" in lengths:
            checks.append(("LONG_PPL_DATA", utils.LONGEST_256K, "long_ppl_256k", 1))
        if any(n in ("512k", "1m") for n in lengths):
            checks.append(("LONG_PPL_DATA", utils.LONGEST_1M, "long_ppl 512k/1m shared documents", 1))
    if "books" in suites:
        if any(n in ("32k", "64k", "128k", "256k") for n in lengths):
            checks.append(("LONG_PPL_BOOKS", utils.LONGEST_256K, "long_ppl_books <=256k", want))
        if any(n in ("512k", "1m") for n in lengths):
            checks.append(("LONG_PPL_BOOKS", utils.LONGEST_1M, "long_ppl_books 512k/1m shared documents", want))
    for env, longest, label, need in checks:
        if not os.environ.get(env):
            report(False, f"{label}: {env} is not set")
            continue
        if not tokenizer:
            report(False, f"{label}: give --tokenizer to count the documents in {os.environ[env]}")
            continue
        try:
            utils.validate_inputs(os.environ[env])
            docs = utils._documents({"pretrained": tokenizer}, longest, env=env)
            report(len(docs) >= need, f"{label}: {len(docs)} documents of >= {longest} tokens in {os.environ[env]}"
                   + ("" if len(docs) >= want else f" (want {want})"))
        except Exception as error:  # noqa: BLE001
            report(False, f"{label}: {type(error).__name__}: {str(error)[:1500]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", default=None, help="tokenizer dir, for the long_ppl / books document counts")
    p.add_argument("--lengths", default="32k,64k,128k,256k")
    p.add_argument("--suites", default="ruler,babilong,longbench2,infinitebench,ppl,books")
    p.add_argument("--storage-only", action="store_true", help="inspect PPL paths/files on this host, without a tokenizer")
    args = p.parse_args()
    failures = []

    def report(ok, text):
        print(("OK       " if ok else "MISSING  ") + text, flush=True)
        if not ok:
            failures.append(text)
    if args.storage_only:
        check_storage(report)
        print(f"\n{len(failures)} storage problem(s)", flush=True)
        raise SystemExit(1 if failures else 0)
    from long_eval_plan import LENGTHS, SUITES, parse_list
    lengths = parse_list(args.lengths, LENGTHS, "--lengths")
    suites = parse_list(args.suites, SUITES, "--suites")
    if os.environ.get("HF_DATASETS_OFFLINE") != "1":
        print("note: HF_DATASETS_OFFLINE is not 1; run through diag.sh to check what eval.sh will see")
    if "babilong" in suites:
        check_babilong(report, lengths)
    if "longbench2" in suites:
        check_longbench2(report)
    extended = any(n in ("512k", "1m") for n in lengths)
    if "ruler" in suites or "turns" in suites:
        check_ruler_qa(report, extended)
    if "ruler" in suites and extended:
        check_ruler_cwe(report)
    if "infinitebench" in suites:
        check_infinitebench(report)
    check_ppl(report, args.tokenizer, lengths, suites)
    print(f"\n{len(failures)} problem(s)" + ("" if failures else ": the long-context data is complete"))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
