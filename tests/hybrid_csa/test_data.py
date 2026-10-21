"""Tiny local documents exercise real tokenization, packing, and LitData loading."""

import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from hybrid_csa.train import (DATA_MANIFEST, corpus_files, make_loader, prepare_data, tokenize_file,
                              validate_prepared_data)


def test_offline_preparation_and_loader_bind_tokenizer_and_length(tmp_path, monkeypatch):
    pytest.importorskip("litdata")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    base, source = tmp_path / "base", tmp_path / "raw"
    base.mkdir()
    source.mkdir()
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(base / "tokenizer.json"))
    (base / "tokenizer_config.json").write_text(json.dumps({"eos_token": "[EOS]", "tokenizer_class": "PreTrainedTokenizerFast"}))
    docs = source / "documents.jsonl"
    docs.write_text("\n".join(json.dumps({"text": "hello world " * 8}) for _ in range(8)))
    tokens = list(tokenize_file(docs, base))
    assert len(tokens) == 8 and all(t.tolist() == [2, 3] * 8 + [1] for t in tokens)
    cfg = dict(base_checkpoint_dir=str(base), data_dir=str(source), tokenized_dir=str(tmp_path / "packed"),
               context_length=7, prepare_workers=1, seed=42, micro_batch_size=1, num_workers=0)
    prepare_data(cfg)
    loader, manifest = make_loader(cfg)
    batch = next(iter(loader))
    assert batch.shape == (1, 8) and batch.dtype == tokens[0].dtype
    assert set(batch.flatten().tolist()) <= {1, 2, 3}
    assert manifest["eos"] and not manifest["bos"]
    with pytest.raises(FileExistsError):
        prepare_data(cfg)
    with pytest.raises(ValueError, match="length/index"):
        make_loader({**cfg, "context_length": 8})
    (base / "generation_config.json").write_text('{"eos_token_id": 0}')
    with pytest.raises(ValueError, match="tokenizer"):
        make_loader(cfg)
    assert (tmp_path / "packed" / DATA_MANIFEST).is_file()


def test_prepared_litdata_chunks_inside_the_corpus_are_not_documents(tmp_path):
    """SemanticLogKV writes litdata_<key>_ctx<N>/ (index.json + chunk files) into the shared data_dir."""
    source = tmp_path / "raw"
    (source / "books").mkdir(parents=True)
    (source / "books/a.jsonl").write_text(json.dumps({"text": "hello"}) + "\n")
    (source / "b.parquet").write_bytes(b"")
    chunks = source / "litdata_719749c807061589_ctx32768"
    (chunks / "nested").mkdir(parents=True)
    (chunks / "index.json").write_text(json.dumps({"chunks": [{"filename": "chunk-0-0.bin"}],
                                                   "config": {"data_format": ["no_header_tensor:16"]}}))
    (chunks / "chunk-0-0.bin").write_bytes(b"\0" * 8)
    (chunks / "nested/stray.json").write_text("{}")
    (source / "notes").mkdir()
    (source / "notes/index.json").write_text(json.dumps({"text": "a document named index.json"}))
    files, skipped = corpus_files(source)
    assert [str(p.relative_to(source)) for p in files] == ["b.parquet", "books/a.jsonl", "notes/index.json"]
    assert skipped == [chunks]


def test_preparation_skips_and_records_litdata_chunks_and_verifies_the_same_files(tmp_path, monkeypatch):
    pytest.importorskip("litdata")
    base, source = tmp_path / "base", tmp_path / "raw"
    base.mkdir()
    source.mkdir()
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(base / "tokenizer.json"))
    (base / "tokenizer_config.json").write_text(json.dumps({"eos_token": "[EOS]", "tokenizer_class": "PreTrainedTokenizerFast"}))
    (source / "documents.jsonl").write_text("\n".join(json.dumps({"text": "hello world " * 8}) for _ in range(8)))
    chunks = source / "litdata_key_ctx32768"
    chunks.mkdir()
    (chunks / "index.json").write_text(json.dumps({"chunks": [], "config": {}}))
    cfg = dict(base_checkpoint_dir=str(base), data_dir=str(source), tokenized_dir=str(tmp_path / "packed"),
               context_length=7, prepare_workers=1, seed=42, micro_batch_size=1, num_workers=0)
    (tmp_path / "packed").mkdir()
    (tmp_path / "packed/chunk-9-9.bin").write_bytes(b"left by a preparation that failed")  # no manifest
    prepare_data(cfg)
    assert not (tmp_path / "packed/chunk-9-9.bin").exists()
    record = validate_prepared_data(cfg, verify_source=True)
    assert list(record["documents"]) == ["documents.jsonl"]
    assert record["skipped_litdata_outputs"] == ["litdata_key_ctx32768"]
    assert len(record["documents"]["documents.jsonl"]["sha256"]) == 64  # full hashes, recorded once
    # Reuse checks the file list and sizes without re-reading the corpus.
    (source / "extra.jsonl").write_text(json.dumps({"text": "hello"}) + "\n")
    with pytest.raises(ValueError, match="files or sizes changed"):
        validate_prepared_data(cfg, verify_source=True)
    (source / "extra.jsonl").unlink()
    with (source / "documents.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="files or sizes changed"):
        validate_prepared_data(cfg, verify_source=True)


def test_empty_documents_are_skipped_counted_and_recorded(tmp_path):
    pytest.importorskip("litdata")
    import pyarrow as pa
    import pyarrow.parquet as pq

    base, source = tmp_path / "base", tmp_path / "raw"
    base.mkdir()
    source.mkdir()
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(base / "tokenizer.json"))
    (base / "tokenizer_config.json").write_text(json.dumps({"eos_token": "[EOS]", "tokenizer_class": "PreTrainedTokenizerFast"}))
    texts = ["hello world " * 8, None, "hello " * 8, "   ", "world " * 8]  # two empty documents
    pq.write_table(pa.table({"text": texts}), source / "train-00000-of-00002.parquet")
    (source / "train-00001-of-00002.jsonl").write_text(
        "\n".join(json.dumps({"text": t}) for t in ["hello world " * 8, ""]) + "\n")
    assert len(list(tokenize_file(source / "train-00000-of-00002.parquet", base))) == 3
    cfg = dict(base_checkpoint_dir=str(base), data_dir=str(source), tokenized_dir=str(tmp_path / "packed"),
               context_length=7, prepare_workers=2, seed=42, micro_batch_size=1, num_workers=0)
    prepare_data(cfg)
    record = validate_prepared_data(cfg, verify_source=True)
    assert record["skipped_empty_documents"] == {"train-00000-of-00002.parquet": 2, "train-00001-of-00002.jsonl": 1}
    # A malformed record still stops preparation, naming the file and row.
    pq.write_table(pa.table({"text": ["hello", None, "world"], "chapter": [None, 5, None]}),
                   source / "train-00000-of-00002.parquet")
    with pytest.raises(Exception, match="row 2"):
        list(tokenize_file(source / "train-00000-of-00002.parquet", base))
