"""Memory-mapped 512K/1M RULER datasets, shared by ranks on a host or prepared on CPU."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache, wraps
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from filelock import FileLock

_ROOT = Path(__file__).resolve().parents[1]
_BUILDERS = dict(get_cw_dataset="ruler_cwe", fwe_download="ruler_fwe",
                 get_vt_dataset="ruler_vt", get_hotpotqa="ruler_qa_hotpot")
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                    "vocab.json", "merges.txt", "tokenizer.model", "spiece.model", "config.json")
_ESSAY_TASKS = {"niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multivalue", "niah_multiquery"}
_EVAL_LIMIT = ContextVar("ruler_eval_limit", default=None)


@contextmanager
def evaluation_sample_limit(limit):
    count = int(limit) if limit is not None and limit >= 1 else None
    token = _EVAL_LIMIT.set(count)
    try:
        yield
    finally:
        _EVAL_LIMIT.reset(token)


def shared_cache_dir():
    from litgpt.ruler_patch import _get_ruler_cache_dir

    return Path(os.environ.get("RULER_DATASET_CACHE_DIR") or Path(_get_ruler_cache_dir()) / "generated")


@cache
def _file_digest(path, size, mtime_ns):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def _digest(path):
    stat = path.stat()
    return _file_digest(path, stat.st_size, stat.st_mtime_ns)


@cache
def _generator_version():
    import lm_eval

    upstream = Path(lm_eval.__file__).parent / "tasks/ruler"
    sources = [Path(__file__), Path(__file__).with_name("ruler_patch.py"),
               _ROOT / "eval_tasks/niah_turns/niah_turns_utils.py", *sorted(upstream.glob("*.py"))]
    packages = ("lm-eval", "transformers", "tokenizers", "numpy", "nltk", "scipy")
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return [list(sys.version_info[:2]), versions, [_digest(p) for p in sources if p.is_file()]]


def _entry_stem(task, length, count, kwargs):
    from litgpt.ruler_patch import _cwe_wordlist_file, _essay_file, _get_ruler_cache_dir

    name = kwargs.get("tokenizer") or kwargs.get("pretrained")
    if not isinstance(name, (str, os.PathLike)):
        return None
    folder = Path(name).expanduser()
    if not any((folder / file).is_file() for file in ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json")):
        return None
    corpus = None
    if task in _ESSAY_TASKS or task.startswith("niah_turns_"):
        corpus = _essay_file()
    elif task == "ruler_cwe":
        corpus = _cwe_wordlist_file()
    elif task == "ruler_qa_hotpot":
        corpus = Path(_get_ruler_cache_dir()) / "hotpot_dev_distractor_v1.json"
    if corpus is not None and (not corpus.is_file() or corpus.is_symlink()):
        return None
    tokenizer = [(file, _digest(folder / file)) for file in _TOKENIZER_FILES if (folder / file).is_file()]
    corpus_hash = _digest(corpus) if corpus is not None else None
    if task in _ESSAY_TASKS or task.startswith("niah_turns_"):
        import nltk

        try:
            directory = Path(str(nltk.data.find("tokenizers/punkt_tab/english/")))
        except LookupError:
            return None
        if not directory.is_dir():
            return None
        corpus_hash = [corpus_hash, [(p.name, _digest(p)) for p in sorted(directory.iterdir()) if p.is_file()]]
    payload = [task, length, count, _generator_version(), tokenizer, corpus_hash]
    fingerprint = hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:32]
    return f"{task}-{length}-{count}-{fingerprint}"


def _read(path, expected):
    import datasets
    import pyarrow as pa

    if not path.is_file() or path.is_symlink():
        return None
    try:
        # IPC files have a footer, so a reader cannot use a partial shared copy.
        with pa.memory_map(str(path)) as source:
            table = pa.ipc.open_file(source).read_all()
        if len(table) == expected and {"input", "outputs", "max_length"} <= set(table.column_names):
            fingerprint = hashlib.sha256(path.stem.encode()).hexdigest()[:32]
            return datasets.Dataset(table, split=datasets.Split.TEST, fingerprint=fingerprint)
    except (OSError, pa.ArrowException):
        pass
    return None


def _write(path, dataset):
    import pyarrow as pa

    with pa.OSFile(str(path), "wb") as stream, pa.ipc.new_file(stream, dataset.data.schema) as writer:
        writer.write_table(dataset.data.table)


def _report(message):
    if os.environ.get("RANK", "0") == "0":
        print(f"[RULER datasets] {message}", flush=True)


def _build(builder, kwargs, count):
    import numpy as np

    state, numpy_state = random.getstate(), np.random.get_state()
    saved = os.environ.get("RULER_LONG_SAMPLES")
    try:
        # VT's short few-shot example still draws from the upstream global RNGs.
        random.seed(0)
        np.random.seed(1234)
        os.environ["RULER_LONG_SAMPLES"] = str(count)
        return builder(**kwargs)
    finally:
        random.setstate(state)
        np.random.set_state(numpy_state)
        if saved is None:
            os.environ.pop("RULER_LONG_SAMPLES", None)
        else:
            os.environ["RULER_LONG_SAMPLES"] = saved


def cached_builder(builder, filename):
    """Wrap an lm-eval custom dataset entry point, leaving short runs alone."""
    task = _BUILDERS.get(builder.__name__)
    if filename.endswith("/eval_tasks/niah_turns/niah_turns_utils.py"):
        task = builder.__name__ if builder.__name__ in {"niah_turns_512k", "niah_turns_1m"} else None
    elif filename.endswith("/lm_eval/tasks/ruler/niah_utils.py") and builder.__name__.startswith("niah_"):
        task = builder.__name__
    elif "/lm_eval/tasks/ruler/" not in filename:
        task = None
    if not task:
        return builder

    @wraps(builder)
    def load(**kwargs):
        limit = _EVAL_LIMIT.get()
        lengths = kwargs.get("max_seq_lengths") or ()
        if task.startswith("niah_turns_"):
            lengths = (524288 if task.endswith("_512k") else 1048576,)
        if len(lengths) != 1 or lengths[0] < 524288:
            return builder(**kwargs)
        cap = int(os.environ.get("RULER_LONG_SAMPLES", "100"))
        if cap < 1:
            raise ValueError("RULER_LONG_SAMPLES must be positive")
        cap = min(cap, 100 if task.startswith("niah_turns_") else 500)
        count = min(cap, limit) if limit is not None else cap
        length = lengths[0]
        prepare = os.environ.get("RULER_DATASET_PREPARE") == "1"
        require_shared = bool(os.environ.get("RULER_DATASET_CACHE_DIR")) and not prepare
        stem = _entry_stem(task, length, count, kwargs)
        if stem is None:
            if prepare or require_shared:
                raise ValueError(f"Cannot identify cached inputs for {task}. Provide a local tokenizer and prepared RULER files.")
            return _build(builder, kwargs, count)
        shared = shared_cache_dir()
        # Full-size prebuilt entries can serve a smaller explicit LIMIT.
        for size in dict.fromkeys((cap, count)):
            candidate = stem if size == count else _entry_stem(task, length, size, kwargs)
            for path in sorted(shared.glob(f"{candidate}-*.arrow")):
                data = _read(path, size)
                if data is not None:
                    _report(f"hit {task} {length} ({count}/{size} samples) {path}")
                    return {"test": data if size == count else data.select(range(count))}
        if require_shared:
            raise FileNotFoundError(f"No matching RULER dataset for {task} {length} ({count} samples) in {shared}. "
                                    "Run unused/prepare_ruler_datasets.py with the evaluation tokenizer and environment.")

        local = Path(tempfile.gettempdir()) / f"logkv-ruler-datasets-{os.getuid()}"
        local.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = local / f"{stem}.arrow"
        with FileLock(str(path.with_suffix(".lock"))):
            data = _read(path, count)
            if data is None:
                _report(f"building {task} {length} ({count} samples)")
                start = time.monotonic()
                generated = _build(builder, kwargs, count)["test"]
                if len(generated) != count:
                    raise ValueError(f"{task} returned {len(generated)} rows, expected {count}")
                with tempfile.TemporaryDirectory(dir=local) as scratch:
                    staged = Path(scratch) / path.name
                    _write(staged, generated)
                    os.replace(staged, path)
                del generated
                data = _read(path, count)
                _report(f"built {task} {length} in {time.monotonic() - start:.1f}s")
            if data is None:
                raise ValueError(f"Unreadable local RULER dataset: {path}")
        if prepare:
            shared.mkdir(parents=True, exist_ok=True)
            # Unique ordinary filenames allow publishing on bucket mounts without rename or shared locks.
            output = shared / f"{stem}-{uuid.uuid4().hex}.arrow"
            shutil.copyfile(path, output)
            if _read(output, count) is None:
                raise ValueError(f"Incomplete published RULER dataset: {output}")
            _report(f"published {output}")
        return {"test": data}

    return load
