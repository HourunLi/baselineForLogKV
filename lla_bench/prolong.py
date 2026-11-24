"""Read compatible 32K ProLong LitData chunks; optionally prepare a private copy from raw files."""
import hashlib
import json
import math
import os
import tempfile
from functools import partial
from pathlib import Path


def cache_name(tokenizer_dir, length):
    if not (tokenizer_dir / "tokenizer.json").is_file():
        raise FileNotFoundError(f"missing Qwen tokenizer in {tokenizer_dir}")
    h = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
        path = tokenizer_dir / name
        if path.exists():
            h.update(name.encode())
            h.update(path.read_bytes())
    return f"litdata_{h.hexdigest()[:16]}_ctx{length}"


def tokens(path, tokenizer):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        column = "text"
        if "tulu" in str(path).lower():
            column = "messages"
        elif "textbookchapters" in str(path).lower():
            column = next((s for s in ("text", "chapter", "content") if s in parquet.schema_arrow.names), "text")
        if column not in parquet.schema_arrow.names:
            raise ValueError(f"{path}: missing {column}")
        for batch in parquet.iter_batches(columns=[column]):
            for raw in batch.column(0).to_pylist():
                if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                    continue
                if column == "messages":
                    raw = "\n\n".join(f"### {m.get('role', 'user')}\n{str(m['content']).strip()}"
                                      for m in (raw or []) if m.get("content") is not None and str(m["content"]).strip())
                if str(raw).strip():
                    yield tokenizer.encode(str(raw).strip(), bos=False, eos=True)
    else:
        with path.open() as f:
            first = next((s for s in f if s.strip()), "")
            f.seek(0)
            rows = json.load(f) if first.lstrip().startswith("[") else (json.loads(s) for s in f if s.strip())
            for i, row in enumerate(rows):
                text = row.get("content") or row.get("text") or row.get("code")
                if text is None:
                    raise ValueError(f"{path} record {i}: missing content/text/code")
                if str(text).strip():
                    yield tokenizer.encode(str(text).strip(), bos=False, eos=True)


def raw_files(raw):
    return (p for p in raw.rglob("*") if p.suffix in {".json", ".jsonl", ".parquet"}
            and not any(s.startswith("litdata_") for s in p.relative_to(raw).parts[:-1]))


def prepare(raw, destination, base, length, workers):
    from litdata import optimize
    from litdata.streaming import TokensLoader
    from .litgpt.tokenizer import Tokenizer

    files = sorted(raw_files(raw))
    if not files:
        raise FileNotFoundError(f"no ProLong source files in {raw}")
    if destination.exists() or destination.is_relative_to(raw):
        raise ValueError(f"choose a fresh private cache outside the raw data: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp, tempfile.TemporaryDirectory() as scratch:
        built = Path(tmp) / "chunks"
        # LitData removes its scratch directories at startup. Give each preparation its own pair.
        scratch_dirs = {"DATA_OPTIMIZER_CACHE_FOLDER": f"{scratch}/chunks",
                        "DATA_OPTIMIZER_DATA_CACHE_FOLDER": f"{scratch}/data"}
        old = {key: os.environ.get(key) for key in scratch_dirs}
        try:
            os.environ.update(scratch_dirs)
            optimize(fn=partial(tokens, tokenizer=Tokenizer(base)), inputs=[str(p) for p in files],
                     output_dir=str(built), num_workers=min(workers, len(files)), chunk_bytes="200MB",
                     item_loader=TokensLoader(block_size=length + 1))
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if not (built / "index.json").is_file():
            raise RuntimeError("LitData did not produce a complete index")
        built.rename(destination)


def check_chunks(cache):
    index = json.loads((cache / "index.json").read_text())
    if index["config"].get("compression"):
        raise ValueError(f"expected the uncompressed reference ProLong cache in {cache}")
    if not index["chunks"]:
        raise ValueError(f"empty cache {cache}")
    for chunk in index["chunks"]:
        path = cache / chunk["filename"]
        if not path.is_file() or path.stat().st_size != chunk["chunk_bytes"]:
            raise ValueError(f"incomplete ProLong chunk: {path}")


def loader(cache, length, seed, workers):
    from litdata import StreamingDataset, StreamingDataLoader, TokensLoader

    check_chunks(cache)
    dataset = StreamingDataset(input_dir=str(cache), item_loader=TokensLoader(block_size=length + 1),
                               shuffle=True, seed=seed)
    return StreamingDataLoader(dataset, batch_size=1, num_workers=workers, drop_last=True)
