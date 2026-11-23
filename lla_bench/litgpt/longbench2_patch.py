"""LongBench v2 from ordinary local parquet files, using the same subset names as lm-eval.

Requests for recursal/longbench-v2 load the byte-verified copies in LONGBENCH2_DIR or <HF_HOME>/longbench2.
Evaluation never falls back to hub snapshots or an older datasets cache, including when the copy directory
is missing. The download script alone opens hub snapshots and resolves flattened links (with optional NUL
terminators) to make ordinary copies. A file belongs to a subset when its stem or folder is the subset's
name, or its stem starts with that name plus '-' or '.'. No symlink creation is needed.
Parquet is read with PyArrow into in-memory datasets, bypassing the datasets builder's cache-directory rename.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

import yaml

REPO = "recursal/longbench-v2"
BUILDERS = {".parquet": "parquet", ".jsonl": "json", ".json": "json", ".csv": "csv", ".arrow": "arrow"}
TASKS_DIR = Path(__file__).resolve().parents[1] / "eval_tasks" / "longbench2"


def subsets() -> list[str]:
    """The subsets evaluated by our pinned task definitions, including on lm-eval 0.4.9."""
    names = {(yaml.safe_load(path.read_text()) or {}).get("dataset_name") for path in TASKS_DIR.glob("*.yaml")}
    names.discard(None)
    if not names:
        raise FileNotFoundError(f"LongBench v2 task definitions missing from {TASKS_DIR}")
    return sorted(names)


def subset_files(files, name: str) -> list[str]:
    """The data files (paths relative to the snapshot) of subset ``name``."""
    chosen = []
    for f in files:
        path = Path(f)
        if path.suffix not in BUILDERS:
            continue
        stem = path.name[:-len(path.suffix)]
        if stem == name or path.parent.name == name or stem.startswith((name + "-", name + ".")):
            chosen.append(f)
    return sorted(chosen)


def flattened_link(path) -> str | None:
    """Read a mount-flattened hub link, including the terminating NUL used by the training filesystem."""
    p = Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size > 512:
        return None
    try:
        text = p.read_bytes().decode().strip(" \t\r\n\0")
    except UnicodeDecodeError:
        return None
    return text if "/blobs/" in text and "\n" not in text and "\0" not in text else None


def real_file(path) -> str:
    """Resolve a flattened source-cache link for the downloader; evaluation reads ordinary files directly."""
    p = Path(path)
    text = flattened_link(p)
    if text is not None:
        blob = os.path.normpath(p.parent / text)
        if os.path.isfile(blob):
            return blob
    return str(p)


def _cached_snapshot() -> str | None:
    """A snapshot folder of the hub cache when refs/main is lost (the newest one)."""
    from huggingface_hub.constants import HF_HUB_CACHE

    folder = Path(os.environ.get("HF_HUB_CACHE") or HF_HUB_CACHE) / ("datasets--" + REPO.replace("/", "--")) / "snapshots"
    shots = sorted((p for p in folder.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime) if folder.is_dir() else []
    return str(shots[-1]) if shots else None


def plain_folder() -> Path:
    """Shared ordinary files survive cache-link flattening and missing node-local blobs."""
    home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return Path(os.environ.get("LONGBENCH2_DIR") or home / "longbench2").expanduser()


def _data_files(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*")
                  if path.is_file() and path.suffix in BUILDERS)


def plain_snapshot() -> tuple[str, list[str]]:
    root = plain_folder()
    files = _data_files(root) if root.is_dir() else []
    return str(root), files


def snapshot(prefer_plain: bool = True) -> tuple[str, list[str]]:
    """Evaluation uses ordinary files exclusively, even when the selected directory is missing/empty.
    Only the downloader passes False to inspect/repair the hub source; it resolves flattened source links."""
    if prefer_plain:
        return plain_snapshot()
    from huggingface_hub import snapshot_download

    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    try:
        root = snapshot_download(REPO, repo_type="dataset", local_files_only=offline)
    except Exception:
        root = _cached_snapshot() if offline else None
        if root is None:
            raise
    return root, _data_files(Path(root))


def file_state(path) -> str:
    """What this node actually reads, rather than assuming the downloader's cache view is shared."""
    path = Path(path)
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            head = stream.read(64)
            stream.seek(max(0, size - 4))
            tail = stream.read(4)
        return (f"{path}: {size} bytes, symlink={path.is_symlink()}, target={path.resolve()}, "
                f"starts={head!r}, ends={tail!r}")
    except OSError as error:
        return f"{path}: {type(error).__name__}: {error}"


def data_problem() -> str | None:
    """Why the offline data is incomplete; use the same subset list for fetching and evaluation."""
    repair = ("Run bash diag.sh unused/fetch_long_eval_data.py --longbench2 --no-babilong --no-ruler-qa "
              "with the evaluation job's HF_HOME, then retry.")
    try:
        root, files = snapshot()
    except Exception as error:
        return f"cannot read LongBench v2 plain directory {plain_folder()} ({type(error).__name__}: {error}). {repair}"
    missing = [name for name in subsets() if not subset_files(files, name)]
    if missing:
        return (f"LongBench v2 plain directory {root} on host {socket.gethostname()} is missing subsets: {missing}. "
                f"Evaluation does not fall back to hub snapshots. {repair}")
    bad = []
    for name in files:
        if not name.endswith(".parquet"):
            continue
        path = Path(root) / name
        try:
            with path.open("rb") as stream:
                head = stream.read(4)
                stream.seek(-4, os.SEEK_END)
                whole = not path.is_symlink() and path.stat().st_size >= 12 and head == b"PAR1" and stream.read(4) == b"PAR1"
        except OSError:
            whole = False
        if not whole:
            bad.append(name)
    if bad:
        observed = "; ".join(file_state(Path(root) / name) for name in bad[:3])
        return (f"LongBench v2 files in {root} are not whole parquet: {bad}. "
                f"Observed on host {socket.gethostname()}: {observed}. {repair}")
    return None


def load_subset(name: str, split=None):
    """Read the small evaluation dataset into memory without creating a datasets builder/cache directory.

    load_dataset('parquet', ...) finalizes an Arrow cache by renaming its .incomplete directory. The shared
    training mount forbids that operation, including with keep_in_memory=True on the builder. Constructing
    Dataset from the Arrow table avoids cache writes, locks and rename races between evaluation ranks.
    """
    from datasets import Dataset, DatasetDict
    import pyarrow.parquet as pq

    root, files = snapshot()
    chosen = subset_files(files, name)
    if not chosen:
        raise FileNotFoundError(f"LongBench v2 plain directory {root} has no file for subset {name!r}")
    if any(Path(path).suffix != ".parquet" for path in chosen):
        raise ValueError(f"LongBench v2 subset {name!r} must consist of ordinary parquet files in {root}")
    table = pq.read_table([os.path.join(root, path) for path in chosen], partitioning=None)
    data = Dataset(table, split="train")
    return DatasetDict(train=data)[split] if split else DatasetDict(train=data)


def apply_patch() -> None:
    import datasets

    original = datasets.load_dataset
    if getattr(original, "_longbench2_patched", False):
        return

    def load_dataset(path=None, name=None, *args, **kwargs):
        if path != REPO or not name:
            return original(path, name, *args, **kwargs)
        # Use the same ordinary files that preflight checked, even if an older datasets cache exists.
        return load_subset(name, kwargs.get("split"))

    load_dataset._longbench2_patched = True
    datasets.load_dataset = load_dataset
