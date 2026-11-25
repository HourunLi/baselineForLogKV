"""Long-document perplexity at 32K through 1M, including curves on a shared long-document selection.

Each document is cut to L tokens; the last TAIL tokens are scored given the L - TAIL before them, so the
number says how well the model predicts text once most of its context is far behind it (in LogKV,
compressed). 32K / 64K / 128K share their documents: only documents of at least LONGEST (131072) tokens
are taken, in file order, so the three differ only in how much context precedes the same kind of tail.
256K takes its own documents, of at least 262144 tokens (LONGEST_256K), so that the shorter lengths keep
the documents (and caches) they had; its numbers compare methods at 256K, not lengths.
512K and 1M both select documents of at least 1048576 tokens into the same independent cache.
Their book curves also share that selection, using min_length in the task YAML; older curves keep 256K books.

The group long_ppl_books_<L> reads another source, LONG_PPL_BOOKS (books of at least 262144 tokens, e.g.
PG19's train set, whose test set holds only 4 such books): its tasks long_ppl_books_<L>_8k, _16k, ... _<L> take
the same first LONG_PPL_DOCS books and score the last TAIL tokens of each book cut to 8K, 16K, ... up to L. A
point is the loss at that position with everything before it in context, so the curve shows how prediction
changes as more of the context lies far behind (in LogKV, compressed); each point costs one forward of its
length. The group's length names the configuration (its YaRN factor), so each configuration's curve keeps
its own columns, and one task per point keeps lm-eval's --limit counting books.

Data: LONG_PPL_DATA, a .parquet / .jsonl file or a directory of them (text in a "text", "content" or
"code" column), e.g. held-out ProLong shards. LONG_PPL_DOCS caps the documents (default 100);
LONG_PPL_CONCAT=1 joins consecutive rows of a file into one document when the rows are pieces of
longer texts. The selection is cached under LONG_PPL_CACHE (default $HF_HOME/logkv_long_ppl, shared by
every node when HF_HOME is, else ~/.cache/logkv_long_ppl).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import datasets

LONGEST = 131072
LONGEST_256K = 262144
LONGEST_1M = 1048576
TAIL = 4096
COLUMNS = ("text", "content", "code")


def _missing(path: Path) -> str:
    """What this node sees of a LONG_PPL_DATA that is not there: a dangling link, or its nearest folder."""
    if path.is_symlink():
        return f"a symlink to {os.readlink(path)!r}, which is missing"
    parent = path.parent
    while not parent.is_dir() and parent != parent.parent:
        parent = parent.parent
    try:
        names = sorted(p.name for p in parent.iterdir())
    except OSError as error:
        names = [f"<unreadable: {error}>"]
    return (f"the nearest existing folder is {str(parent)!r}, holding {names[:10]}"
            + (f" and {len(names) - 10} more" if len(names) > 10 else "")
            + "; a bucket path written from another machine may need syncing (warming up) to this one")


def _files(source: str) -> list[Path]:
    path = Path(source).expanduser()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"LONG_PPL_DATA={source!r} is neither a file nor a directory: {_missing(path)}")
    return sorted([*path.rglob("*.parquet"), *path.rglob("*.jsonl")])


def _texts(path: Path):
    if path.suffix == ".parquet":
        from eval_tasks.long_ppl.input_files import open_parquet

        with open_parquet(path) as pf:
            column = next((c for c in COLUMNS if c in pf.schema_arrow.names), None)
            if column is None:
                return
            for group in range(pf.num_row_groups):
                for value in pf.read_row_group(group, columns=[column]).column(column).to_pylist():
                    if value:
                        yield str(value)
    else:
        from eval_tasks.long_ppl.input_files import resolve_input

        with open(resolve_input(path), encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = next((row.get(c) for c in COLUMNS if row.get(c)), None) if isinstance(row, dict) else None
                if value:
                    yield str(value)


def validate_inputs(source):
    """Check the raw files as well as the selection cache during an offline preflight."""
    from eval_tasks.long_ppl.input_files import open_parquet

    files = _files(source)
    if not files:
        raise ValueError(f"No parquet/jsonl PPL input files in {source!r}")
    for path in files:
        if path.suffix == ".parquet":
            with open_parquet(path):
                pass


def select_documents(source: str, tokenizer, count: int, concat: bool = False,
                     longest: int | None = None, progress=None) -> list[list[int]]:
    """The first ``count`` documents of at least ``longest`` (default LONGEST) tokens, cut to it."""
    longest = longest or LONGEST
    found: list[list[int]] = []
    report = progress if progress is not None else lambda message: None
    for path in _files(source):
        report(f"PPL scan {path}: need {count} documents of >= {longest} tokens")
        pending: list[int] = []
        scanned = 0
        for scanned, text in enumerate(_texts(path), 1):
            if not concat and len(text) < longest:  # fewer characters than tokens: too short
                if scanned % 100 == 0:
                    report(f"PPL scan {path.name}: scanned {scanned} rows, selected {len(found)}/{count}")
                continue
            report(f"PPL tokenize {path.name} row {scanned}: {len(text)} characters, "
                   f"selected {len(found)}/{count}")
            started = time.monotonic()
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            report(f"PPL tokenized {path.name} row {scanned}: {len(ids)} tokens "
                   f"in {time.monotonic() - started:.1f}s (need >= {longest})")
            if concat:
                pending += ids
                if len(pending) < longest:
                    continue
                ids, pending = pending, []
            if len(ids) >= longest:
                found.append(ids[:longest])
                report(f"PPL selected {len(found)}/{count} documents")
                if len(found) >= count:
                    return found
        report(f"PPL finished {path.name}: scanned {scanned} rows, selected {len(found)}/{count}")
    return found


def cache_dir() -> Path:
    default = os.path.join(os.environ["HF_HOME"], "logkv_long_ppl") if os.environ.get("HF_HOME") else "~/.cache/logkv_long_ppl"
    return Path(os.environ.get("LONG_PPL_CACHE") or default).expanduser()


def cache_key(source: str, tokenizer_name: str, count: int, concat: bool, longest: int | None = None) -> str:
    """The documents depend on the data, the tokenizer's files (not the path it was named by: eval.sh names
    the checkpoint, a warm-up may name the base model), the count, the concat flag and the minimum length
    (LONGEST by default, which keeps the keys of the caches written before 256K)."""
    digest = hashlib.sha256()
    folder = Path(tokenizer_name).expanduser()
    files = [folder / n for n in ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt")]
    files = [f for f in files if f.is_file()]
    for f in files:
        digest.update(f.read_bytes())
    if not files:
        digest.update(tokenizer_name.encode())
    digest.update(json.dumps([os.path.abspath(os.path.expanduser(source)), count, concat,
                              longest or LONGEST]).encode())
    return digest.hexdigest()[:16]


def _tokenizer(kwargs):
    from transformers import AutoTokenizer

    name = kwargs.get("tokenizer") or kwargs.get("pretrained")
    if not name:
        raise ValueError("long_ppl needs the tokenizer: metadata 'tokenizer' or 'pretrained' (eval.sh passes it)")
    return name, AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def _write_selection_cache(cache: Path, docs: list[list[int]]) -> None:
    """Stage outside the bucket; optional publication never uses shared PID-only temp names."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="logkv-ppl-", suffix=".json") as tmp:
            json.dump(docs, tmp)
            tmp.flush()
            # mkdir success need not make a bucket directory immediately visible to open() on this node.
            for attempt in range(3):
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(tmp.name, cache)  # ordinary files; no bucket rename or symlink required
                    return
                except FileNotFoundError:
                    if attempt == 2:
                        raise
                    time.sleep(0.2 * (attempt + 1))
    except OSError as error:
        warnings.warn(
            f"PPL selection cache could not be saved to {cache}: {error}. "
            "Continuing with the selected documents; set LONG_PPL_CACHE to a writable node-local directory "
            "to reuse them across curve points.", RuntimeWarning, stacklevel=2,
        )


def _documents(kwargs, longest: int | None = None, env: str = "LONG_PPL_DATA", progress=None) -> list[list[int]]:
    source = os.environ.get(env)
    if not source:
        raise ValueError(f"long_ppl needs {env} (a .parquet / .jsonl file or a directory, e.g. held-out "
                         f"ProLong shards; eval.sh exports the YAML's {env.lower()})")
    count = int(os.environ.get("LONG_PPL_DOCS", "100"))
    concat = os.environ.get("LONG_PPL_CONCAT", "0") == "1"
    report = progress if progress is not None else lambda message: None
    report(f"PPL load tokenizer: {kwargs.get('tokenizer') or kwargs.get('pretrained')}")
    name, tokenizer = _tokenizer(kwargs)
    longest = longest or LONGEST
    key = cache_key(source, name, count, concat, longest)
    cache = cache_dir() / f"{key}.json"
    try:
        report(f"PPL read selection cache: {cache}")
        with cache.open(encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as error:  # unavailable cache or a concurrent incomplete copy
        report(f"PPL selection cache unavailable ({cache}): {error}; selecting from source")
    docs = select_documents(source, tokenizer, count, concat, longest, progress=progress)
    if not docs:
        raise ValueError(f"no document of {longest} tokens or more in {source!r}; if its rows are pieces of longer "
                         "texts, set LONG_PPL_CONCAT=1")
    report(f"PPL write selection cache: {len(docs)} documents -> {cache}")
    _write_selection_cache(cache, docs)
    return docs


def build(length: int, kwargs, longest: int | None = None, env: str = "LONG_PPL_DATA") -> dict:
    _, tokenizer = _tokenizer(kwargs)
    rows = []
    for ids in _documents(kwargs, longest, env):
        context, tail = ids[:length - TAIL], ids[length - TAIL:length]
        continuation = tokenizer.decode(tail)
        # eval.py encodes the continuation on its own: count the tokens it scores.
        scored = len(tokenizer(continuation, add_special_tokens=False)["input_ids"])
        rows.append(dict(context=tokenizer.decode(context), continuation=continuation, length=length,
                         tail_tokens=scored, tail_bytes=len(continuation.encode("utf-8"))))
    return {"test": datasets.Dataset.from_list(rows)}


def long_ppl_32k(**kwargs):
    return build(32768, kwargs)


def long_ppl_64k(**kwargs):
    return build(65536, kwargs)


def long_ppl_128k(**kwargs):
    return build(131072, kwargs)


def long_ppl_256k(**kwargs):
    return build(262144, kwargs, longest=LONGEST_256K)


def long_ppl_512k(**kwargs):
    return build(524288, kwargs, longest=LONGEST_1M)


def long_ppl_1m(**kwargs):
    return build(1048576, kwargs, longest=LONGEST_1M)


def book_points(length: int) -> list[int]:
    """8K, 16K, ... up to ``length``: the points of long_ppl_books_<length>'s curve."""
    points, n = [], 8192
    while n <= length:
        points.append(n)
        n *= 2
    return points


def long_ppl_books(**kwargs):
    """long_ppl_books_<L>_<point>: the books cut to ``point`` (dataset_kwargs), their last TAIL tokens scored."""
    return build(int(kwargs["point"]), kwargs, longest=int(kwargs.get("min_length", LONGEST_256K)),
                 env="LONG_PPL_BOOKS")


def process_results(doc, results):
    loglikelihood = results[0][0] if isinstance(results[0], (tuple, list)) else results[0]
    return {"ppl": (loglikelihood, doc["tail_tokens"]), "bits_per_byte": (loglikelihood, doc["tail_bytes"])}


def perplexity(items) -> float:
    """exp of the mean negative log-likelihood per tail token, over all documents."""
    return math.exp(-sum(ll for ll, _ in items) / max(sum(n for _, n in items), 1))


def bits_per_byte(items) -> float:
    return -sum(ll for ll, _ in items) / max(sum(n for _, n in items), 1) / math.log(2)
