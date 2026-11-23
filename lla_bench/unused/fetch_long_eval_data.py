"""Fetch the data of the long-context evaluations into the caches eval.sh reads (run once, online).

eval.sh runs offline (HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1) with HF_HOME, HF_DATASETS_CACHE and
RULER_CACHE_DIR from its environment header; diag.sh sets the same paths. This script turns the offline
switches off for itself only, then:

--lengths 512k,1m prepares the official BABILong JSON plus ordinary memory-mappable Arrow copies,
RULER's ordinary essay background/NLTK data, and shared >=1M-token PPL selections when a PPL source is given.
The default lengths stay 32k,64k,128k,256k. --ruler-essays and --ruler-cwe prepare individual RULER assets.

    BABILong      RMT-team/babilong-1k-samples, configs 32k / 64k / 128k, and RMT-team/babilong (100
                  samples per length), config 256k; splits qa1 ... qa5
    RULER QA      SQuAD v2 dev and HotpotQA dev (distractor) JSON, into RULER_CACHE_DIR, where
                  litgpt/ruler_patch.py looks first
    LongBench v2  recursal/longbench-v2, every subset lm-eval's longbench2 tasks load (--longbench2)
                  plus byte-verified ordinary parquet copies in <HF_HOME>/longbench2 (or LONGBENCH2_DIR),
                  which evaluation reads before the hub/datasets caches; 20 subsets, 503 questions
    InfiniteBench xinrongzhang2022/InfiniteBench, the jsonl files of the ten tasks eval_tasks/infinitebench reads
                  (--infinitebench; into the hub cache, and copied as plain files to <HF_HOME>/infinitebench, where
                  they are read without INFINITEBENCH_DIR)
    long_ppl      --ppl-data: finds the documents of at least 131072 tokens (32K-128K) and of at least
                  262144 (256K) with --tokenizer and caches both selections the long_ppl tasks use, so
                  eval ranks do not tokenize them again (no 256K document only stops long_ppl_256k);
                  --ppl-hub <dataset repo> (e.g. emozilla/pg19-test, PG19's 100 test books) downloads
                  the repository first into ordinary files under <HF_HOME>/long_ppl_data/<owner--name>
                  and uses that folder as --ppl-data (export the printed path as LONG_PPL_DATA for eval.sh)
    long_ppl_books --ppl-books-hub <dataset repo> (e.g. emozilla/pg19): downloads its train parquet files
                  one at a time, in name order, until LONG_PPL_DOCS (100) books of at least 262144 tokens are
                  found (4-6% of PG19's 28,602 train books are that long), and caches the selection the
                  long_ppl_books_<L> tasks read. All lengths use ordinary files; verified Hub-cache bytes
                  are copied, otherwise files are downloaded directly (export the printed folder as LONG_PPL_BOOKS);
                  --ppl-books-data <file or folder> selects from local data instead

and finally loads every dataset again in an offline subprocess, as eval.py will (skipping the
<hash>.incomplete folders an interrupted download leaves, litgpt/hf_cache_patch.py; the summary names any
that remain). A summary goes to <--out>.summary.txt beside --out (default fetch_long_eval_data.json in the
current directory).

The platform's proxy re-signs HTTPS with a self-signed CA, so certificate checks are off unless --verify-ssl
is given (as litgpt/ruler_patch.py does for RULER); downloads then stay on requests, without hf_transfer or
hf_xet, which check certificates themselves. hf-mirror answers 429 when the Hub quota it shares across its users
runs out; every request waits and retries then.

    bash diag.sh unused/fetch_long_eval_data.py --longbench2 \\
        --ppl-data <held-out ProLong parquet file or directory> --tokenizer <Qwen3-1.7B-Base dir>
"""
from __future__ import annotations

import os

# Before any Hugging Face import: these are read when the libraries load.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
# The mirror can pause mid-file for longer than the Hub's 10 s default read timeout.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

from lla_bench.unused.fetch_progress import ProgressLines, run_stage  # noqa: E402

BABILONG = "RMT-team/babilong-1k-samples"
BABILONG_LENGTHS = ("32k", "64k", "128k")
BABILONG_LONG = "RMT-team/babilong"  # the 1000-sample set stops at 128k
BABILONG_LONG_LENGTHS = ("256k",)
QA_SPLITS = tuple(f"qa{i}" for i in range(1, 6))
LONGBENCH2 = "recursal/longbench-v2"
RULER_QA = {
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json": "dev-v2.0.json",
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json": "hotpot_dev_distractor_v1.json",
}


def _progress(lines):
    # Plain-list callers retain their summary order; only the live CLI requests per-document logs.
    return lines.append if isinstance(lines, ProgressLines) else None


def patch_requests(verify_ssl: bool, retries: int = 10) -> None:
    """Every requests call (RULER QA, and the Hub API calls and files of datasets) retries a 429 after its
    Retry-After (60 s without one, at most 300 s), and skips certificate checks unless verify_ssl. Call before
    any Hugging Face import: without the checks, hf_transfer and hf_xet are switched off when the Hub loads."""
    import time

    import requests
    import urllib3

    if not verify_ssl:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    request = requests.Session.request

    def patched_request(self, *args, **kwargs):
        if not verify_ssl:
            kwargs["verify"] = False
        for attempt in range(retries + 1):
            response = request(self, *args, **kwargs)
            if response.status_code != 429 or attempt == retries:
                return response
            wait = response.headers.get("Retry-After", "")
            wait = min(int(wait), 300) if wait.isdigit() else 60
            print(f"429 rate limit on {response.url}; retry {attempt + 1}/{retries} in {wait} s",
                  file=sys.stderr, flush=True)
            response.close()
            time.sleep(wait)

    requests.Session.request = patched_request


def longbench2_subsets() -> list[str]:
    from lla_bench.litgpt.longbench2_patch import subsets

    return subsets()


def ruler_cache_dir() -> Path:
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return Path(os.environ.get("RULER_CACHE_DIR", os.path.join(home, "ruler_cache")))


def fetch_ruler_qa(lines: list[str]) -> None:
    import requests

    folder = ruler_cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for url, name in RULER_QA.items():
        path = folder / name
        if not path.is_file():
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            path.write_text(json.dumps(response.json(), ensure_ascii=False), encoding="utf-8")
        lines.append(f"RULER QA {name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def fetch_datasets(babilong: bool, longbench2: bool, lines: list[str], lengths=None) -> tuple[list[tuple[str, str, str]], bool]:
    """Each dataset group in turn; a group that fails is reported and the others still run."""
    if not (babilong or longbench2):
        return [], True
    import datasets

    from lla_bench.litgpt.hf_cache_patch import leftovers
    from lla_bench.litgpt.longbench2_patch import apply_patch, snapshot

    apply_patch()  # as eval.py: LongBench v2 subsets from the snapshot's files when the card has no configs
    groups = []
    lengths = tuple(lengths or (*BABILONG_LENGTHS, *BABILONG_LONG_LENGTHS))
    if babilong:
        groups.append(("BABILong", [(BABILONG, n, s) for n in BABILONG_LENGTHS if n in lengths for s in QA_SPLITS]))
        groups.append(("BABILong 256k", [(BABILONG_LONG, n, s) for n in BABILONG_LONG_LENGTHS if n in lengths for s in QA_SPLITS]))
    if longbench2:
        groups.append(("LongBench v2", [(LONGBENCH2, name, "train") for name in longbench2_subsets()]))
    loads, ok = [], True
    if babilong:
        from lla_bench.unused.long_eval_download import prepare_babilong
        for length in (n for n in lengths if n in ("512k", "1m")):
            try:
                run_stage(lines, f"BABILong {length}", prepare_babilong, length, lines)
            except Exception as error:
                ok = False
                lines.append(f"BABILong {length}: FAILED {type(error).__name__}: {error}")
    for label, group in groups:
        try:
            if label == "LongBench v2":
                root, files = run_stage(lines, "LongBench v2 snapshot", snapshot, prefer_plain=False)
                lines.append(f"LongBench v2 snapshot {root}: {len(files)} data files, e.g. {files[:4]}")
                if not repair_parquet(root, files, lines):
                    raise ValueError("some LongBench v2 files are still not parquet after a fresh download (above)")
                materialize_longbench2(root, files, lines)
            for path, name, split in group:
                rows = len(run_stage(lines, f"load {path}/{name}/{split}", datasets.load_dataset,
                                     path, name=name, split=split))
                lines.append(f"{path} {name} {split}: {rows} rows")
            loads += group
            for path in sorted({path for path, _, _ in group}):
                left = leftovers(path)
                if left:
                    lines.append(f"{path}: unfinished folders of earlier downloads, skipped offline and safe to "
                                 f"delete: {left}")
        except Exception as error:  # noqa: BLE001 - report and go on with the next group
            ok = False
            lines.append(f"{label}: FAILED {type(error).__name__}: {str(error)[:1500]}")
    return loads, ok


def is_parquet(path: Path) -> bool:
    """A parquet file starts and ends with the magic bytes PAR1."""
    if not path.is_file() or path.stat().st_size < 12:
        return False
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(-4, os.SEEK_END)
        return head == b"PAR1" and f.read(4) == b"PAR1"


def describe(path: Path) -> str:
    """Size and first bytes of a file that should be parquet (a Git LFS pointer, an error page, a cut file)."""
    if not path.exists():
        return "missing"
    with open(path, "rb") as f:
        head = f.read(120)
    return f"{path.stat().st_size} bytes, starts {head!r}"


def flattened_link(path: Path) -> str | None:
    """The target of a hub-cache link that a copy or sync turned into a small regular file holding the link's text
    (``../../../../blobs/<sha256>``), or None."""
    from lla_bench.litgpt.longbench2_patch import flattened_link as read_link

    return read_link(path)


def refetch(repo: str, name: str, root: str, whole=None) -> tuple[Path, str]:
    """Make the snapshot file ``name`` whole again; returns its path and what was done. hf_hub_download leaves a
    snapshot file that exists as it is, even with force_download (it only refreshes the blob), so a bad one is
    removed first. A flattened link whose blob is whole (``whole``, default a whole parquet file) is always
    replaced with ordinary copied bytes: the training filesystem does not support symlinks."""
    whole = whole or is_parquet
    from huggingface_hub import hf_hub_download

    pointer = Path(root) / name
    text = flattened_link(pointer)
    if text is not None:
        blob = Path(os.path.normpath(pointer.parent / text))
        pointer.unlink()
        if whole(blob):
            shutil.copyfile(blob, pointer)
            return pointer, f"a link to blobs/{blob.name} flattened into a file; the blob copied in its place"
    elif pointer.is_symlink() or pointer.exists():
        pointer.unlink()
    # The snapshot's own revision, so the file lands in this snapshot folder.
    fresh = hf_hub_download(repo, name, repo_type="dataset", revision=Path(root).name, force_download=True)
    fresh = Path(fresh)
    text = flattened_link(fresh)
    if fresh.is_symlink():
        blob = fresh.resolve()
    elif text is not None:
        blob = Path(os.path.normpath(fresh.parent / text))
    else:
        blob = fresh
    if blob != fresh and whole(blob):
        fresh.unlink()
        shutil.copyfile(blob, fresh)
    return fresh, "downloaded again as an ordinary file"


def repair_parquet(root: str, files: list[str], lines: list[str], repo: str | None = None) -> bool:
    """Make every snapshot parquet file that is not a whole parquet file whole again (refetch)."""
    from lla_bench.litgpt.longbench2_patch import REPO

    repo = repo or REPO
    bad = [f for f in files if f.endswith(".parquet") and not is_parquet(Path(root) / f)]
    ok = True
    for f in bad:
        before = describe(Path(root) / f)
        fresh, done = refetch(repo, f, root)
        good = is_parquet(fresh)
        ok = ok and good
        lines.append(f"{repo} {f}: was {before}; {done}: " + ("ok" if good else describe(fresh)))
    if not bad:
        lines.append(f"{repo} files: all whole parquet")
    return ok


def materialize_longbench2(root: str, files: list[str], lines: list[str]) -> Path:
    """Copy every subset as ordinary bytes on the shared mount, then verify rows and byte identity.

    A valid snapshot on the download node can be unreadable on evaluation nodes when its links or blobs
    do not survive the cache sync. These copies never point into blobs/ and need no rename support.
    """
    import pyarrow.parquet as pq

    from lla_bench.litgpt.longbench2_patch import plain_folder, real_file, subset_files, subsets

    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        return value.digest()

    target_root = plain_folder()
    if target_root.resolve() == Path(root).resolve():
        raise ValueError("LONGBENCH2_DIR must differ from the hub snapshot directory")
    total = 0
    names = subsets()
    for subset in names:
        selected = subset_files(files, subset)
        if not selected or any(not name.endswith(".parquet") for name in selected):
            raise ValueError(f"LongBench v2 subset {subset!r} has no complete parquet source in {root}")
        rows = 0
        for name in selected:
            source = Path(real_file(Path(root) / name))
            expected = pq.ParquetFile(source).metadata.num_rows
            expected_hash = digest(source)
            target = target_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            same = (target.is_file() and not target.is_symlink() and target.stat().st_size == source.stat().st_size
                    and digest(target) == expected_hash)
            if not same:
                if target.is_symlink():
                    target.unlink()
                shutil.copyfile(source, target)
            if (target.is_symlink() or not is_parquet(target) or digest(target) != expected_hash
                    or pq.ParquetFile(target).metadata.num_rows != expected):
                raise ValueError(f"LongBench v2 plain copy did not verify on this node: {target}")
            rows += expected
        if not rows:
            raise ValueError(f"LongBench v2 subset {subset!r} is empty in {target_root}")
        total += rows
        lines.append(f"LongBench v2 plain {subset}: {rows} rows, SHA256 verified")
    if total != 503:
        raise ValueError(f"LongBench v2 plain data has {total} questions in {len(names)} subsets; expected 503")
    lines.append(f"LongBench v2 plain files: {target_root} ({len(names)} subsets, {total} questions; no cache links)")
    return target_root


def fetch_ppl_hub(repo: str, lines: list[str]) -> str:
    """Every PPL length uses ordinary files, rather than publishing a Hub snapshot path."""
    from huggingface_hub import HfApi
    from lla_bench.unused.long_eval_download import download_plain

    root = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "long_ppl_data" / repo.replace("/", "--")
    names = sorted(n for n in run_stage(lines, f"list PPL files {repo}", HfApi().list_repo_files,
                                       repo, repo_type="dataset") if n.endswith((".parquet", ".jsonl")))
    if not names:
        raise ValueError(f"No parquet/jsonl in {repo}")
    for name in names:
        download_plain(repo, name, root, lines)
    lines.append(f"  for eval.sh: export LONG_PPL_DATA={root}")
    return str(root)


def jsonl_rows(path: Path) -> int | None:
    """The rows of a whole jsonl file (its first and last lines parse), or None."""
    try:
        first, last, rows = None, None, 0
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    first, last, rows = first or line, line, rows + 1
        json.loads(first)
        json.loads(last)
        return rows
    except (OSError, TypeError, ValueError):
        return None


def copy_plain(source: Path, target: Path, lines: list[str]) -> bool:
    """Copy ``source`` to ``target`` as a plain file, unless a whole copy is there. Written in place: an object-store
    mount (the bucket under HF_HOME) refuses renames, so no temporary name; the copy is checked afterwards (its size,
    and that its first and last lines parse), and a cut copy is written again on the next run."""
    size = source.stat().st_size
    if target.is_file() and not target.is_symlink() and target.stat().st_size == size and jsonl_rows(target):
        lines.append(f"  plain copy {target}: already there")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    shutil.copyfile(source, target)
    leftover = target.with_name(target.name + ".partial")  # from an earlier version, which renamed
    if leftover.exists():
        try:
            leftover.unlink()
        except OSError:
            lines.append(f"  could not remove {leftover}; delete it by hand")
    whole = target.stat().st_size == size and jsonl_rows(target) is not None
    lines.append(f"  plain copy {target}: " + ("ok" if whole else f"NOT whole ({describe(target)})"))
    return whole


def fetch_infinitebench(lines: list[str]) -> bool:
    """InfiniteBench's ten jsonl files into the hub cache, and copied as plain files to <HF_HOME>/infinitebench,
    where eval_tasks/infinitebench reads them first (the cache's snapshot links may not survive a copy or sync of
    the cache to the machines that evaluate); a file that is not whole jsonl (a cut download, a flattened link) is
    fetched again."""
    from huggingface_hub import hf_hub_download

    spec = importlib.util.spec_from_file_location(
        "infinitebench_utils", ROOT / "eval_tasks" / "infinitebench" / "infinitebench_utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    ok = True
    for task in utils.TASKS:
        name = f"{task}.jsonl"
        try:
            path = Path(hf_hub_download(utils.REPO, name, repo_type="dataset"))
            if jsonl_rows(path) is None:
                before = describe(path)
                path, done = refetch(utils.REPO, name, str(path.parent), whole=lambda p: jsonl_rows(p) is not None)
                lines.append(f"InfiniteBench {name}: was {before}; {done}")
            rows = jsonl_rows(path)
            ok = ok and rows is not None
            lines.append(f"InfiniteBench {name}: " + (f"{rows} samples, {path.stat().st_size / 1e6:.0f} MB"
                                                      if rows else f"NOT whole jsonl ({describe(path)})"))
            if rows:  # a plain copy, which survives a copy or sync of the cache that drops links
                plain = Path(utils.plain_folder()) / name
                ok = copy_plain(path, plain, lines) and ok
        except Exception as error:  # noqa: BLE001
            ok = False
            lines.append(f"InfiniteBench {name}: FAILED {type(error).__name__}: {str(error)[:1500]}")
    return ok


def check_offline(loads, lines: list[str]) -> bool:
    """Load every dataset again with the offline switches eval.sh sets (and eval.py's LongBench v2 and
    unfinished-folder patches), in a fresh process."""
    script = ("import datasets, json, sys\n"
              "from lla_bench.litgpt.longbench2_patch import apply_patch\n"
              "apply_patch()\n"
              "from lla_bench.litgpt.hf_cache_patch import apply_patch\n"
              "apply_patch()\n"
              "for path, name, split in json.loads(sys.argv[1]):\n"
              "    datasets.load_dataset(path, name=name, split=split)\n"
              "print('ok')\n")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    done = subprocess.run([sys.executable, "-c", script, json.dumps(loads)], env=env, capture_output=True, text=True)
    ok = done.returncode == 0 and done.stdout.strip().endswith("ok")
    lines.append("offline reload: " + ("ok" if ok else "FAILED\n" + done.stderr[-2000:]))
    return ok


def cache_ppl(data: str, tokenizer: str, lines: list[str], lengths=None) -> None:
    """Select the long_ppl documents now (the tasks read the same caches): those of at least LONGEST tokens
    (32K-128K) and those of at least LONGEST_256K (256K). Data without the latter is reported, not an error."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("long_ppl_utils", ROOT / "eval_tasks" / "long_ppl" / "long_ppl_utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    os.environ["LONG_PPL_DATA"] = data
    want = int(os.environ.get("LONG_PPL_DOCS", "100"))
    if lengths and any(n in ("512k", "1m") for n in lengths):
        docs = utils._documents({"pretrained": tokenizer}, utils.LONGEST_1M, progress=_progress(lines))
        lines.append(f"long_ppl_512k/1m: {len(docs)} shared documents of >= {utils.LONGEST_1M} tokens "
                     f"from {data} (want {want}); cache {utils.cache_dir()}")
        if all(n in ("512k", "1m") for n in lengths):
            return
    try:
        docs = utils._documents({"pretrained": tokenizer}, progress=_progress(lines))
    except ValueError:
        ppl_data_stats(data, tokenizer, utils, lines)
        raise
    lines.append(f"long_ppl: {len(docs)} documents of >= {utils.LONGEST} tokens from {data} (want {want}); "
                 f"cache {utils.cache_dir()}")
    try:
        found = len(utils._documents({"pretrained": tokenizer}, utils.LONGEST_256K, progress=_progress(lines)))
    except ValueError:
        found = 0
    lines.append(f"long_ppl_256k: {found} documents of >= {utils.LONGEST_256K} tokens from {data} (want {want})"
                 + ("" if found else "; long_ppl_256k cannot run on this data"))
    if min(len(docs), found) < want:
        ppl_data_stats(data, tokenizer, utils, lines)
        lines.append("  fewer than wanted: point --ppl-data at more shards, or if the rows are pieces of longer "
                     "texts, rerun with LONG_PPL_CONCAT=1 (and run eval.sh with it too)")


def _long_ppl_utils():
    import importlib.util

    spec = importlib.util.spec_from_file_location("long_ppl_utils", ROOT / "eval_tasks" / "long_ppl" / "long_ppl_utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return utils


def fetch_ppl_books(repo: str, tokenizer: str, lines: list[str], pattern: str | None = None,
                    longest: int | None = None) -> str:
    """Download a dataset repository's train files one at a time, in name order, until enough long books are
    in them; returns the ordinary folder that holds them (the LONG_PPL_BOOKS of eval.sh)."""
    from fnmatch import fnmatch

    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from lla_bench.eval_tasks.long_ppl.input_files import is_hub_snapshot
    from lla_bench.unused.long_eval_download import download_plain

    utils = _long_ppl_utils()
    longest = longest or utils.LONGEST_256K
    want = int(os.environ.get("LONG_PPL_DOCS", "100"))
    lines.append(f"PPL books: need {want} documents of >= {longest} tokens; "
                 "scan shards in name order until enough qualify (LONG_PPL_DOCS controls the target)")
    files = sorted(f for f in run_stage(lines, f"list PPL shards {repo}", HfApi().list_repo_files,
                                       repo, repo_type="dataset")
                   if f.endswith((".parquet", ".jsonl")) and (fnmatch(f, pattern) if pattern else "train" in f))
    if not files:
        raise ValueError(f"no {'file matching ' + pattern if pattern else 'train parquet / jsonl file'} in {repo}")
    tok = run_stage(lines, f"load PPL tokenizer {tokenizer}", AutoTokenizer.from_pretrained,
                    tokenizer, trust_remote_code=True)
    subdir = "long_ppl_books_1m" if longest >= utils.LONGEST_1M else "long_ppl_books"
    folder = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / subdir / repo.replace("/", "--")
    configured = os.environ.get("LONG_PPL_BOOKS")
    if configured and is_hub_snapshot(configured):
        lines.append(f"LONG_PPL_BOOKS points to a Hub snapshot ({configured}); materializing ordinary files in {folder}")
        configured = None
    root = str(Path(configured).expanduser() if configured else folder)
    found = 0
    for shard, name in enumerate(files, 1):
        lines.append(f"PPL shard {shard}/{len(files)}: {name}; selected {found}/{want} books so far")
        path = download_plain(repo, name, root, lines)
        if name.endswith(".parquet") and not is_parquet(path):
            raise ValueError(f"PPL download is not whole Parquet: {path}; {describe(path)}")
        found += len(run_stage(lines, f"PPL scan/tokenize {path}", utils.select_documents,
                               str(path), tok, want - found, longest=longest, progress=_progress(lines)))
        lines.append(f"long_ppl_books {repo} {name}: {found} books of >= {longest} tokens so far")
        if found >= want:
            break
    # The tasks read every parquet / jsonl file under the folder, in name order: other files there change the books.
    read = set(files[:files.index(name) + 1])
    others = sorted(str(f.relative_to(root)) for f in Path(root).rglob("*")
                    if f.suffix in (".parquet", ".jsonl") and str(f.relative_to(root)) not in read)
    if others:
        lines.append(f"  WARNING: the folder also holds {others[:5]} ({len(others)} files) from earlier downloads; "
                     "the tasks would read them too")
    lines.append(f"  for eval.sh: export LONG_PPL_BOOKS={root}")
    if not found:
        raise ValueError(f"No books of >= {longest} tokens in {repo}; select another held-out source")
    if found < want:
        lines.append(f"  fewer than wanted: only {found}/{want} books; all methods must use this same selection")
    return root


def cache_ppl_books(data: str, tokenizer: str, lines: list[str], longest: int | None = None) -> None:
    """Select the long_ppl_books documents now (every long_ppl_books_<L>_<point> task reads this cache)."""
    utils = _long_ppl_utils()
    longest = longest or utils.LONGEST_256K
    os.environ["LONG_PPL_BOOKS"] = data
    want = int(os.environ.get("LONG_PPL_DOCS", "100"))
    docs = utils._documents({"pretrained": tokenizer}, longest, env="LONG_PPL_BOOKS", progress=_progress(lines))
    lines.append(f"long_ppl_books: {len(docs)} books of >= {longest} tokens from {data} (want {want}); "
                 f"cache {utils.cache_dir()}" + ("" if len(docs) >= want else "; fewer than wanted: give more data"))


def ppl_data_stats(data: str, tokenizer: str, utils, lines: list[str], files: int = 2, show: int = 5) -> None:
    """How long the rows are: characters over every row of the first files, tokens of the first and the
    longest rows (rows of one length throughout are packed sequences, not documents)."""
    import heapq

    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    paths = utils._files(data)[:files]
    lengths, first, longest = [], [], []
    for path in paths:
        for i, text in enumerate(utils._texts(path)):
            lengths.append(len(text))
            if len(first) < show:
                first.append(text)
            heapq.heappush(longest, (len(text), len(lengths), text))
            if len(longest) > show:
                heapq.heappop(longest)
    if not lengths:
        lines.append(f"  long_ppl data: no text rows in {[str(p) for p in paths]}")
        return
    chars = np.asarray(lengths)

    def tokens(texts):
        return [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    lines.append(f"  long_ppl data: {len(chars)} rows in {len(paths)} file(s); characters p50 {int(np.median(chars))}, "
                 f"p90 {int(np.percentile(chars, 90))}, max {int(chars.max())}")
    lines.append(f"  tokens of the first {len(first)} rows: {tokens(first)}")
    lines.append(f"  tokens of the {len(longest)} longest rows: {sorted(tokens([t for _, _, t in longest]), reverse=True)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--longbench2", action="store_true", help="also fetch LongBench v2 (every subset)")
    p.add_argument("--infinitebench", action="store_true", help="also fetch InfiniteBench (ten tasks, jsonl)")
    p.add_argument("--no-babilong", action="store_true")
    p.add_argument("--no-ruler-qa", action="store_true")
    p.add_argument("--lengths", default="32k,64k,128k,256k", help="data lengths, including 512k,1m; 1m=1048576 tokens")
    p.add_argument("--ruler-essays", action="store_true", help="prepare ordinary RULER essays and NLTK files")
    p.add_argument("--ruler-cwe", action="store_true", help="prepare the pinned NVIDIA word list for 512K/1M")
    p.add_argument("--ppl-data", default=None, help="held-out ProLong parquet / jsonl file or directory")
    p.add_argument("--ppl-hub", default=None,
                   help="dataset repository to download and use as --ppl-data (e.g. emozilla/pg19-test)")
    p.add_argument("--ppl-books-hub", default=None,
                   help="dataset repository whose train files hold long books (e.g. emozilla/pg19), for long_ppl_books")
    p.add_argument("--ppl-books-files", default=None,
                   help="glob of the --ppl-books-hub files to read (default: names containing 'train')")
    p.add_argument("--ppl-books-data", default=None, help="local file or folder of long books, for long_ppl_books")
    p.add_argument("--tokenizer", default=None, help="tokenizer dir for --ppl-data / --ppl-books-* (e.g. the "
                   "Qwen3-1.7B-Base dir)")
    p.add_argument("--out", default="fetch_long_eval_data.json")
    p.add_argument("--verify-ssl", action="store_true", help="keep HTTPS certificate checks (off by default)")
    args = p.parse_args()
    from lla_bench.long_eval_plan import LENGTHS, parse_list
    lengths = parse_list(args.lengths, LENGTHS, "--lengths")
    ultra = any(n in ("512k", "1m") for n in lengths)
    patch_requests(args.verify_ssl)
    # Hub 0.36 falls back to ordinary files when symlinks are unavailable.
    # Select that same path on the preparation host as well as the bucket mount.
    from huggingface_hub import file_download
    file_download.are_symlinks_supported = lambda *args, **kwargs: False
    out = Path(args.out).expanduser()
    summary = out.with_name(out.stem + ".summary.txt")
    with ProgressLines(summary) as lines:
        lines.append(f"Live progress: {summary}; a WAIT heartbeat is emitted every 20 seconds")
        for line in [f"HF_HOME={os.environ.get('HF_HOME')}",
                     f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}",
                     f"RULER_CACHE_DIR={ruler_cache_dir()}",
                     f"HTTPS certificate checks: {'on' if args.verify_ssl else 'off (--verify-ssl turns them on)'}"]:
            lines.append(line)
        ok = True
        if args.ruler_cwe or (ultra and not args.no_ruler_qa):
            try:
                from lla_bench.unused.long_eval_download import prepare_ruler_cwe
                run_stage(lines, "RULER CWE word list", prepare_ruler_cwe, lines)
            except Exception as error:
                ok = False
                lines.append(f"RULER CWE: FAILED {type(error).__name__}: {error}")
        if args.ruler_essays or (ultra and not args.no_ruler_qa):
            try:
                from lla_bench.unused.long_eval_download import prepare_ruler_essays
                run_stage(lines, "RULER essays/NLTK", prepare_ruler_essays, lines)
            except Exception as error:
                ok = False
                lines.append(f"RULER essays: FAILED {type(error).__name__}: {error}")
        if (args.ppl_data or args.ppl_hub or args.ppl_books_hub or args.ppl_books_data) and not args.tokenizer:
            raise SystemExit("--ppl-data / --ppl-hub / --ppl-books-* need --tokenizer (the dir with tokenizer.json)")
        if not args.no_ruler_qa:
            try:
                run_stage(lines, "RULER QA", fetch_ruler_qa, lines)
            except Exception as error:  # noqa: BLE001
                ok = False
                lines.append(f"RULER QA: FAILED {type(error).__name__}: {error}")
        loads, fetched = run_stage(lines, "BABILong/LongBench v2 preparation", fetch_datasets,
                                  not args.no_babilong, args.longbench2, lines, lengths=lengths)
        ok = ok and fetched
        if loads:
            ok = run_stage(lines, "offline reload check", check_offline, loads, lines) and ok
        if args.infinitebench:
            ok = run_stage(lines, "InfiniteBench", fetch_infinitebench, lines) and ok
        if args.ppl_hub:
            try:
                args.ppl_data = run_stage(lines, "PPL source download", fetch_ppl_hub,
                                          args.ppl_hub, lines)
            except Exception as error:  # noqa: BLE001
                ok = False
                lines.append(f"long_ppl data {args.ppl_hub}: FAILED {type(error).__name__}: {str(error)[:1500]}")
        if args.ppl_data:
            try:
                run_stage(lines, "PPL selection cache", cache_ppl, args.ppl_data, args.tokenizer, lines, lengths=lengths)
            except Exception as error:  # noqa: BLE001
                ok = False
                lines.append(f"long_ppl: FAILED {type(error).__name__}: {error}")
        if args.ppl_books_hub:
            try:
                args.ppl_books_data = run_stage(lines, "PPL books download and selection", fetch_ppl_books,
                                                args.ppl_books_hub, args.tokenizer, lines, args.ppl_books_files,
                                                longest=1048576 if ultra else None)
            except Exception as error:  # noqa: BLE001
                ok = False
                lines.append(f"long_ppl_books {args.ppl_books_hub}: FAILED {type(error).__name__}: {str(error)[:1500]}")
        if args.ppl_books_data:
            try:
                run_stage(lines, "PPL books selection cache", cache_ppl_books,
                          args.ppl_books_data, args.tokenizer, lines, longest=1048576 if ultra else None)
            except Exception as error:  # noqa: BLE001
                ok = False
                lines.append(f"long_ppl_books: FAILED {type(error).__name__}: {error}")
        lines.append(f"Preparation {'complete' if ok else 'FAILED'}; summary written to {summary}")
        out.write_text(json.dumps(dict(lines=lines, ok=ok), indent=1))
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
