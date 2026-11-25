"""Ordinary-file preparation for long evaluations; no symlinks or rename-based publishing."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import time
from functools import cache
from pathlib import Path

from unused.fetch_progress import run_stage, stage
from eval_tasks.long_ppl.input_files import resolve_input, whole_parquet

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


@cache
def hub_info(repo):
    from huggingface_hub import HfApi
    return HfApi().dataset_info(repo, files_metadata=True, timeout=60)


def download_plain(repo, name, folder, lines, *, model_info=None):
    """Fetch a pinned resolve URL directly; verify Hub size and SHA256 before using the file."""
    import requests
    from huggingface_hub import hf_hub_url

    info = model_info or run_stage(lines, f"Hub metadata {repo}", hub_info, repo)
    repo_type = "model" if model_info else "dataset"
    item = next((entry for entry in info.siblings if entry.rfilename == name), None)
    if item is None:
        raise FileNotFoundError(f"{repo}@{info.sha} has no {name}")
    lfs = getattr(item, "lfs", None)
    expected_hash = (lfs.get("sha256") or lfs.get("oid")) if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    size = getattr(item, "size", None)
    target = Path(folder) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()

    digest = None

    def verified(path):
        nonlocal digest
        digest = None
        if not path.is_file() or (size is not None and path.stat().st_size != size):
            return False
        if name.endswith(".parquet") and not whole_parquet(path):
            return False
        digest = run_stage(lines, f"SHA256 {path}", sha256, path)
        return not expected_hash or digest == expected_hash

    complete = verified(target)
    if not complete and expected_hash:
        home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
        hub = Path(os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or home / "hub")
        cache = hub / (repo_type + "s--" + repo.replace("/", "--"))
        candidates = [target, cache / "snapshots" / info.sha / name]
        if expected_hash and len(expected_hash) == 64 and all(c in "0123456789abcdef" for c in expected_hash):
            candidates.append(cache / "blobs" / expected_hash)
        for candidate in candidates:
            if not candidate.exists() and not candidate.is_symlink():
                continue
            try:
                source = resolve_input(candidate)
                if source == target or not verified(source):
                    continue
                with stage(lines, f"copy verified cached file {source} -> {target}"):
                    shutil.copyfile(source, target)
                if not verified(target):
                    raise ValueError(f"Ordinary-file copy failed size/SHA256/Parquet check: {target}")
                lines.append(f"reused verified Hub cache bytes as ordinary file: {target}")
                complete = True
                break
            except (OSError, ValueError) as error:
                lines.append(f"cached source unusable {candidate}: {error}")
    if not complete:
        url = hf_hub_url(repo, name, repo_type=repo_type, revision=info.sha)
        total = f"{size / 1e6:.1f} MB" if size is not None else "unknown size"
        with stage(lines, f"download {repo}/{name} ({total}) -> {target}"):
            received, started = 0, time.monotonic()
            last_report = started
            with requests.get(url, stream=True, timeout=(60, 300)) as response:
                response.raise_for_status()
                with target.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            received += len(chunk)
                            now = time.monotonic()
                            if now - last_report >= 5:
                                lines.append(f"download {name}: {received / 1e6:.1f} MB / {total}, "
                                             f"{received / 1e6 / max(now - started, .001):.2f} MB/s")
                                last_report = now
            lines.append(f"download {name}: received {received} bytes")
        if not verified(target):
            raise ValueError(f"Incomplete download or SHA256 mismatch: {target}")
    lines.append(f"ordinary file {repo}@{info.sha}/{name}: {target.stat().st_size} bytes, SHA256={digest}")
    return target


def babi_utils():
    spec = importlib.util.spec_from_file_location("babilong_plain_utils", ROOT / "eval_tasks/babilong_long/babilong_long_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_babilong(length, lines):
    """Keep official JSON plus memory-mappable Arrow streams so eval never builds a shared cache."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    utils = babi_utils()
    for split in (f"qa{i}" for i in range(1, 6)):
        arrow, target = utils.plain_paths(length, split)
        name = str(target.relative_to(utils.plain_folder()))
        download_plain("RMT-team/babilong", name, utils.plain_folder(), lines)
        source_hash = run_stage(lines, f"BABILong {length} {split}: hash source", sha256, target)
        manifest = arrow.with_suffix(".manifest.json")
        valid = False
        try:
            record = json.loads(manifest.read_text())
            with stage(lines, f"BABILong {length} {split}: verify existing Arrow"):
                valid = (not arrow.is_symlink() and record["source_sha256"] == source_hash
                         and record["arrow_sha256"] == sha256(arrow) and len(utils.load_plain(length, split)) == 100)
        except (OSError, ValueError, KeyError):
            pass
        if not valid:
            with stage(lines, f"BABILong {length} {split}: read JSON {target.stat().st_size / 1e6:.1f} MB"):
                with target.open(encoding="utf-8") as stream:
                    rows = json.load(stream)
            if (not isinstance(rows, list) or len(rows) != 100
                    or any(not isinstance(row, dict) or not {"input", "question", "target"} <= row.keys() for row in rows)):
                raise ValueError(f"BABILong {target} must contain 100 input/question/target records")
            with stage(lines, f"BABILong {length} {split}: write 100 rows as ordinary Arrow {arrow}"):
                table = pa.Table.from_pylist(rows)
                if arrow.is_symlink():
                    arrow.unlink()
                with pa.OSFile(str(arrow), "wb") as sink:
                    with ipc.new_stream(sink, table.schema) as writer:
                        writer.write_table(table)
                manifest.write_text(json.dumps(dict(source_sha256=source_hash, arrow_sha256=sha256(arrow), rows=100)))
        data = run_stage(lines, f"BABILong {length} {split}: offline reload", utils.load_plain, length, split)
        if len(data) != 100:
            raise ValueError(f"BABILong {arrow}: expected 100 rows, found {len(data)}")
        lines.append(f"BABILong {length} {split}: {len(data)} rows, ordinary Arrow, offline reload: ok ({arrow})")


def prepare_ruler_cwe(lines):
    import requests

    from litgpt.ruler_patch import RULER_CWE_WORDS_URL, _cwe_wordlist_file, _read_cwe_wordlist, _verify_cwe_wordlist

    target = _cwe_wordlist_file()
    try:
        _read_cwe_wordlist(target)
    except (FileNotFoundError, ValueError):
        if target.is_symlink():
            raise ValueError(f"RULER CWE word list must be an ordinary file: {target}") from None
        response = run_stage(lines, "download NVIDIA CWE word list", requests.get, RULER_CWE_WORDS_URL, timeout=120)
        response.raise_for_status()
        _verify_cwe_wordlist(response.content, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        _read_cwe_wordlist(target)
    lines.append(f"RULER CWE NVIDIA word list: {target}, SHA256 verified")
    return target


def prepare_ruler_essays(lines):
    import pyarrow.parquet as pq
    from litgpt.ruler_patch import _essay_file

    target = _essay_file()
    source = download_plain("baber/paul_graham_essays", "essays.parquet", target.parent / "source", lines)
    table = pq.read_table(source, columns=["text"], partitioning=None)
    text = " ".join(str(value) for value in table["text"].to_pylist() if value)
    if not text.strip():
        raise ValueError(f"Empty RULER essay source: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    target.write_text(text, encoding="utf-8")
    lines.append(f"RULER ordinary essays: {target}, {len(text)} characters, SHA256={sha256(target)}; "
                 "512K/1M generation cycles the background and inserts each needle once")
    prepare_nltk(lines)


def prepare_nltk(lines):
    """Use the same requests SSL policy as the downloader; no urllib certificate dependency."""
    import io
    import zipfile
    import requests

    home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    root = Path(os.environ.get("NLTK_DATA", str(home / "nltk_data")).split(os.pathsep)[0]).expanduser()
    if not (root / "tokenizers/punkt_tab/english/collocations.tab").is_file():
        url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
        response = run_stage(lines, "download NLTK punkt_tab.zip", requests.get, url, timeout=300)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                path = root / "tokenizers" / item.filename
                if Path(item.filename).is_absolute() or ".." in Path(item.filename).parts:
                    raise ValueError(f"Invalid NLTK archive entry: {item.filename}")
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.is_symlink():
                    path.unlink()
                with archive.open(item) as source, path.open("wb") as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
    lines.append(f"RULER NLTK punkt_tab ordinary files: {root / 'tokenizers/punkt_tab'}")
