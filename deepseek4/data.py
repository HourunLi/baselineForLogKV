"""Offline ProLong preparation. Adapted from hybrid_csa at 9af3958 (Apache-2.0)."""

from functools import partial
import hashlib
import json
import os
from pathlib import Path
import tempfile

DATA_MANIFEST = "deepseek4_data.json"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def base_identity(checkpoint_dir):
    path = Path(checkpoint_dir).resolve(strict=True)
    names = (
        "lit_model.pth",
        "model_config.yaml",
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "generation_config.json",
    )
    if not (path / "lit_model.pth").is_file() or not (path / "model_config.yaml").is_file():
        raise ValueError("The original Base needs lit_model.pth and model_config.yaml")
    return {name: sha256(path / name) for name in names if (path / name).is_file()}


def tokenizer_identity(directory):
    names = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "generation_config.json",
    )
    files = {name: sha256(Path(directory) / name) for name in names if (Path(directory) / name).is_file()}
    if not {"tokenizer.json", "tokenizer.model"} & files.keys():
        raise FileNotFoundError(f"No tokenizer in {directory}")
    return files


TEXT_FIELDS = ("text", "chapter", "content", "code")


def is_empty_document(record):
    """A record whose supported fields are all present-but-empty (None, blank text, no messages).

    Such documents carry no text; they are skipped and counted (SemanticLogKV's reader of the same corpus skips
    them too). A record without any supported field, or with a field of another type, is malformed instead.
    """
    if not isinstance(record, dict):
        return False
    present = [record[name] for name in TEXT_FIELDS + ("messages",) if name in record]
    return bool(present) and all(
        value is None
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, list) and not value)
        for value in present
    )


def text_from_record(record, source):
    """Explicit local corpus schema: never skip a malformed/empty document silently."""
    if not isinstance(record, dict):
        raise ValueError(f"Expected a document object in {source}")
    fields = [
        name
        for name in ("text", "chapter", "content", "code")
        if isinstance(record.get(name), str) and record[name].strip()
    ]
    if fields:
        return record[fields[0]].strip()
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        parts = []
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError(f"Invalid messages document in {source}")
            parts.append(f"### {message.get('role', 'user')}\n{message['content']}")
        return "\n\n".join(parts)
    raise ValueError(f"No nonempty text/chapter/content/code/messages in {source}")


def is_litdata_output(directory):
    """A directory LitData's optimize wrote: its index.json lists chunks and the writer config."""
    index = Path(directory) / "index.json"
    try:
        record = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(record, dict)
        and isinstance(record.get("chunks"), list)
        and isinstance(record.get("config"), dict)
    )


def corpus_files(source):
    """Raw documents under ``source`` and the LitData output directories skipped there.

    A shared corpus can hold another run's prepared chunks (SemanticLogKV writes litdata_<key>_ctx<N>/ into its
    data_dir); their index.json is not a document. Symlinked directories are not followed.
    """
    files, skipped = [], []
    for directory, subdirectories, names in os.walk(source):
        directory = Path(directory)
        if "index.json" in names and is_litdata_output(directory):
            skipped.append(directory)
            subdirectories.clear()
            continue
        subdirectories.sort()
        files += [directory / name for name in names if Path(name).suffix in {".parquet", ".jsonl", ".json"}]
    return sorted(files), sorted(skipped)


def tokenize_file(path, tokenizer_dir, empty_log=None):
    """Token ids per document; empty documents are skipped and their count written to ``empty_log``."""
    from ._vendor.niahlogkv.litgpt.tokenizer import Tokenizer

    tokenizer = Tokenizer(tokenizer_dir)
    path = Path(path)
    empty = 0
    for record, source in read_records(path):
        if is_empty_document(record):
            empty += 1
            continue
        yield tokenizer.encode(text_from_record(record, source), bos=False, eos=True)
    if empty_log is not None:
        # One small file per input: LitData runs this in worker processes on the preparing node.
        name = hashlib.sha256(str(path).encode()).hexdigest() + ".json"
        (Path(empty_log) / name).write_text(json.dumps({"path": str(path), "empty_documents": empty}))


def read_records(path):
    """``(record, source)`` for every document of a parquet, JSON-array or JSON-lines file."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        columns = [name for name in TEXT_FIELDS + ("messages",) if name in parquet.schema_arrow.names]
        if not columns:
            raise ValueError(f"Missing supported document column in {path}")
        number = 0
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                number += 1
                yield row, f"{path} row {number}"
    else:
        with path.open(encoding="utf-8") as source:
            first = next((line for line in source if line.strip()), "")
            source.seek(0)
            if first.lstrip().startswith("["):
                for number, row in enumerate(json.load(source), 1):
                    yield row, f"{path} item {number}"
            else:
                for number, line in enumerate(source, 1):
                    if line.strip():
                        yield json.loads(line), f"{path}:{number}"


def prepare_data(cfg):
    """Single-process explicit preparation into a new project-owned directory."""
    from litdata import optimize
    from litdata.streaming import TokensLoader

    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "1")) != 1 and not (dist.is_initialized() and dist.get_rank() == 0):
        raise ValueError("Run --prepare-data once, outside torchrun")
    source = Path(cfg["data_dir"])
    if not source.is_dir():
        raise FileNotFoundError(f"Local corpus does not exist: {source}")
    files, skipped = corpus_files(source)
    for directory in skipped:
        print(f"Skipping prepared LitData chunks, not raw documents: {directory}", flush=True)
    if not files:
        raise FileNotFoundError(
            f"No local parquet/jsonl/json documents in {source}"
            + (f" (skipped LitData outputs: {[str(d) for d in skipped]})" if skipped else "")
        )
    destination = Path(cfg["tokenized_dir"])
    if (destination / DATA_MANIFEST).exists():
        raise FileExistsError(f"Refusing to overwrite tokenized data: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Choose a fresh tokenized_dir; incomplete directory: {destination}")
    record = {
        "schema": 1,
        "context_length": cfg["context_length"],
        "tokenizer": tokenizer_identity(cfg["base_checkpoint_dir"]),
        "documents": {
            str(p.relative_to(source)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files
        },
        "data_dir": str(source),
        "skipped_litdata_outputs": [str(d.relative_to(source)) for d in skipped],
        "text_fields_in_priority_order": ["text", "chapter", "content", "code", "messages"],
        "bos": False,
        "eos": True,
        "preparer_sha256": sha256(__file__),
    }
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="deepseek4_empty_") as empty_log:
        optimize(
            fn=partial(tokenize_file, tokenizer_dir=cfg["base_checkpoint_dir"], empty_log=empty_log),
            inputs=[str(p) for p in files],
            output_dir=str(destination),
            num_workers=min(cfg["prepare_workers"], len(files)),
            chunk_bytes="200MB",
            item_loader=TokensLoader(block_size=cfg["context_length"] + 1),
        )
        counts = [json.loads(entry.read_text()) for entry in Path(empty_log).glob("*.json")]
    if len(counts) != len(files):
        raise RuntimeError(f"Preparation reported {len(counts)} of {len(files)} input files")
    record["skipped_empty_documents"] = {
        str(Path(entry["path"]).relative_to(source)): entry["empty_documents"]
        for entry in sorted(counts, key=lambda e: e["path"])
        if entry["empty_documents"]
    }
    total = sum(record["skipped_empty_documents"].values())
    print(
        f"Skipped {total} empty documents (no text) in {len(record['skipped_empty_documents'])} files",
        flush=True,
    )
    record["index_sha256"] = sha256(destination / "index.json")
    (destination / DATA_MANIFEST).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(f"Prepared offline token data: {destination}")


def reference_chunks(cfg):
    # Same cache key as niahLogKV@a126a1b data.py, independent of checkout paths.
    base = Path(cfg["base_checkpoint_dir"])
    if not any((base / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise FileNotFoundError(f"No Qwen tokenizer in {base}")
    digest = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
        path = base / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    leaf = f"litdata_{digest.hexdigest()[:16]}_ctx{cfg['context_length']}"
    directory = Path(cfg["tokenized_dir"] or Path(cfg["data_dir"]) / leaf)
    if directory.name != leaf:
        raise ValueError(f"Expected the reference tokenizer/length cache {leaf}, got {directory}")
    return directory


def reference_data(cfg):
    directory = reference_chunks(cfg)
    index = directory / "index.json"
    chunks = json.loads(index.read_text())["chunks"]
    if not chunks:
        raise ValueError(f"Empty ProLong index: {index}")
    for chunk in chunks:
        path = directory / chunk["filename"]
        if path.stat().st_size != chunk["chunk_bytes"]:
            raise ValueError(f"ProLong chunk size differs from index: {path}")
    return {"path": str(directory), "index_sha256": sha256(index), "chunks": len(chunks)}


def validate_prepared_data(cfg, verify_source=False):
    directory = Path(cfg["tokenized_dir"])
    manifest = directory / DATA_MANIFEST
    if not manifest.is_file() or not (directory / "index.json").is_file():
        raise FileNotFoundError(f"Missing prepared data at {directory}; run bash majob.sh prepare first")
    record = json.loads(manifest.read_text())
    if record["tokenizer"] != tokenizer_identity(cfg["base_checkpoint_dir"]):
        raise ValueError("Prepared data tokenizer differs from original Base tokenizer")
    if record["context_length"] != cfg["context_length"] or record["index_sha256"] != sha256(
        directory / "index.json"
    ):
        raise ValueError("Prepared data length/index changed; prepare a new directory")
    if Path(record["data_dir"]).resolve() != Path(cfg["data_dir"]).resolve():
        raise ValueError("Prepared data came from a different raw corpus; choose a new tokenized_dir")
    if verify_source:
        # The same files with the same sizes. Their SHA-256 were recorded once at preparation; re-reading a
        # 100+ GB shared corpus before every reuse would cost more than the run itself.
        source = Path(cfg["data_dir"])
        files, _ = corpus_files(source)
        actual = {str(p.relative_to(source)): p.stat().st_size for p in files}
        if actual != {name: entry["bytes"] for name, entry in record["documents"].items()}:
            raise ValueError("Raw corpus files or sizes changed since preparation; choose a new tokenized_dir")
    return record


def prepare_distributed(cfg):
    """One process per node; rank zero prepares, all nodes receive success or failure."""
    from datetime import timedelta
    import torch.distributed as dist

    if cfg.get("training_data") == "niahlogkv_chunks":
        print(f"Using reference ProLong data: {reference_data(cfg)}", flush=True)
        return

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("gloo", timeout=timedelta(hours=24))
    try:
        if world > 1:
            configurations = [None] * world
            dist.all_gather_object(configurations, cfg)
            if any(item != cfg for item in configurations):
                raise ValueError("Nodes resolved different training configs or asset/output paths")
        result = [None]
        if world == 1 or dist.get_rank() == 0:
            try:
                directory = Path(cfg["tokenized_dir"])
                if (directory / DATA_MANIFEST).is_file():
                    validate_prepared_data(cfg, verify_source=True)
                    print(f"Reusing verified prepared data: {directory}", flush=True)
                else:
                    prepare_data(cfg)
                result[0] = {"ok": True}
            except Exception as error:
                result[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        if world > 1:
            dist.broadcast_object_list(result, src=0)
        if not result[0]["ok"]:
            raise RuntimeError(f"Shared data preparation failed: {result[0]['error']}")
        # Every node must see the completed shared files before it can train.
        error_message = None
        try:
            validate_prepared_data(cfg)
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
        if world > 1:
            errors = [None] * world
            dist.all_gather_object(errors, error_message)
        else:
            errors = [error_message]
        if any(errors):
            raise RuntimeError(f"Prepared files are not readable on every node: {errors}")
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def make_loader(cfg):
    from litdata import StreamingDataLoader, StreamingDataset, TokensLoader

    if cfg.get("training_data") == "niahlogkv_chunks":
        record = reference_data(cfg)
        directory = Path(record["path"])
    else:
        record = validate_prepared_data(cfg)
        directory = Path(cfg["tokenized_dir"])
    # LitData detects the initialized torch distributed world and shards across ranks
    # and loader workers. A second DistributedSampler would incorrectly shard twice.
    dataset = StreamingDataset(
        input_dir=str(directory),
        item_loader=TokensLoader(block_size=cfg["context_length"] + 1),
        shuffle=True,
        seed=cfg["seed"],
    )
    loader = StreamingDataLoader(
        dataset,
        batch_size=cfg["micro_batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    return loader, record
