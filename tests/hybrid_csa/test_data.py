"""Tiny local documents exercise real tokenization, packing, and LitData loading."""

import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from hybrid_csa.train import DATA_MANIFEST, make_loader, prepare_data, tokenize_file


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
