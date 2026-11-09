"""Long-document perplexity at a fixed context length (64K, 128K, and 32K for reference).

Each document is cut to L tokens; the last TAIL tokens are scored given the L - TAIL before them, so the
number says how well the model predicts text once most of its context is far behind it (in LogKV,
compressed). The same documents serve every length: only documents of at least the longest length
(131072 tokens) are taken, in file order, so 32K / 64K / 128K differ only in how much context precedes
the same kind of tail.

Data: LONG_PPL_DATA, a .parquet / .jsonl file or a directory of them (text in a "text", "content" or
"code" column), e.g. held-out ProLong shards. LONG_PPL_DOCS caps the documents (default 100);
LONG_PPL_CONCAT=1 joins consecutive rows of a file into one document when the rows are pieces of
longer texts. The selection is cached under LONG_PPL_CACHE (default $HF_HOME/logkv_long_ppl, shared by
every node when HF_HOME is, else ~/.cache/logkv_long_ppl).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import datasets

LONGEST = 131072
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
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        column = next((c for c in COLUMNS if c in pf.schema_arrow.names), None)
        if column is None:
            return
        for group in range(pf.num_row_groups):
            for value in pf.read_row_group(group, columns=[column]).column(column).to_pylist():
                if value:
                    yield str(value)
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = next((row.get(c) for c in COLUMNS if row.get(c)), None) if isinstance(row, dict) else None
                if value:
                    yield str(value)


def select_documents(source: str, tokenizer, count: int, concat: bool = False) -> list[list[int]]:
    """The first ``count`` documents of at least LONGEST tokens (token ids, cut to LONGEST)."""
    found: list[list[int]] = []
    for path in _files(source):
        pending: list[int] = []
        for text in _texts(path):
            if not concat and len(text) < LONGEST:  # fewer characters than tokens: too short
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if concat:
                pending += ids
                if len(pending) < LONGEST:
                    continue
                ids, pending = pending, []
            if len(ids) >= LONGEST:
                found.append(ids[:LONGEST])
                if len(found) >= count:
                    return found
    return found


def cache_dir() -> Path:
    default = os.path.join(os.environ["HF_HOME"], "logkv_long_ppl") if os.environ.get("HF_HOME") else "~/.cache/logkv_long_ppl"
    return Path(os.environ.get("LONG_PPL_CACHE") or default).expanduser()


def cache_key(source: str, tokenizer_name: str, count: int, concat: bool) -> str:
    """The documents depend on the data, the tokenizer's files (not the path it was named by: eval.sh names
    the checkpoint, a warm-up may name the base model), the count and the concat flag."""
    digest = hashlib.sha256()
    folder = Path(tokenizer_name).expanduser()
    files = [folder / n for n in ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt")]
    files = [f for f in files if f.is_file()]
    for f in files:
        digest.update(f.read_bytes())
    if not files:
        digest.update(tokenizer_name.encode())
    digest.update(json.dumps([os.path.abspath(os.path.expanduser(source)), count, concat, LONGEST]).encode())
    return digest.hexdigest()[:16]


def _tokenizer(kwargs):
    from transformers import AutoTokenizer

    name = kwargs.get("tokenizer") or kwargs.get("pretrained")
    if not name:
        raise ValueError("long_ppl needs the tokenizer: metadata 'tokenizer' or 'pretrained' (eval.sh passes it)")
    return name, AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def _documents(kwargs) -> list[list[int]]:
    source = os.environ.get("LONG_PPL_DATA")
    if not source:
        raise ValueError("long_ppl needs LONG_PPL_DATA (a .parquet / .jsonl file or a directory, e.g. held-out "
                         "ProLong shards; eval.sh exports the YAML's long_ppl_data)")
    count = int(os.environ.get("LONG_PPL_DOCS", "100"))
    concat = os.environ.get("LONG_PPL_CONCAT", "0") == "1"
    name, tokenizer = _tokenizer(kwargs)
    key = cache_key(source, name, count, concat)
    cache = cache_dir() / f"{key}.json"
    if cache.is_file():
        try:
            return json.loads(cache.read_text())
        except ValueError:  # another rank is still copying it (see below): build it here too
            pass
    docs = select_documents(source, tokenizer, count, concat)
    if not docs:
        raise ValueError(f"no document of {LONGEST} tokens or more in {source!r}; if its rows are pieces of longer "
                         "texts, set LONG_PPL_CONCAT=1")
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(docs))
    try:
        os.replace(tmp, cache)  # several ranks may build it at once
    except OSError:  # a bucket mount (OBS) refuses the rename: copy the complete file over, then drop it
        shutil.copyfile(tmp, cache)
        with contextlib.suppress(OSError):
            tmp.unlink()
    return docs


def build(length: int, kwargs) -> dict:
    _, tokenizer = _tokenizer(kwargs)
    rows = []
    for ids in _documents(kwargs):
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


def process_results(doc, results):
    loglikelihood = results[0][0] if isinstance(results[0], (tuple, list)) else results[0]
    return {"ppl": (loglikelihood, doc["tail_tokens"]), "bits_per_byte": (loglikelihood, doc["tail_bytes"])}


def perplexity(items) -> float:
    """exp of the mean negative log-likelihood per tail token, over all documents."""
    return math.exp(-sum(ll for ll, _ in items) / max(sum(n for _, n in items), 1))


def bits_per_byte(items) -> float:
    return -sum(ll for ll, _ in items) / max(sum(n for _, n in items), 1) / math.log(2)
