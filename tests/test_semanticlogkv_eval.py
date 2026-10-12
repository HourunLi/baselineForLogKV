"""Exercise DMS through SemanticLogKV's real request and generation code on CPU.

The fake HF model replaces only CUDA attention. Tokenization calls, truncation,
logit slicing, sampling, cache reset and metric aggregation use the shared code.
"""

import ast
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import litgpt
from litgpt.generate.base import generate
from models.qwen3.semanticlogkv_eval import (
    DMSModel, compression_snapshot, file_sha256, make_dms_lm, verify_base_source,
)


class FakeHF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=32, num_hidden_layers=2)
        self.calls = []

    def get_cache(self):
        return SimpleNamespace(layers=[])

    def forward(self, **kwargs):
        self.calls.append({**kwargs, "input_ids": kwargs["input_ids"].clone(),
                           "cache_position": kwargs["cache_position"].clone()})
        processed = int(kwargs["cache_position"][-1]) + 1
        kwargs["past_key_values"].layers = [make_layer(processed, [processed] * 2, None)
                                           for _ in range(self.config.num_hidden_layers)]
        logits = torch.zeros(1, kwargs["logits_to_keep"], 32)
        logits[..., 0] = 2.0
        return SimpleNamespace(logits=logits)


def make_layer(processed, paged, contiguous, window=3):
    def part(lengths):
        return SimpleNamespace(cache_seq_lengths=None if lengths is None else torch.tensor(lengths))
    return SimpleNamespace(paged_cache=part(paged), cont_cache=part(contiguous),
                           get_seq_length=lambda: processed, dms_window_size=window)


def test_forward_keeps_absolute_positions_and_only_requested_logits():
    hf = FakeHF()
    model = DMSModel(hf, max_seq_length=16)
    with pytest.raises(RuntimeError, match="request cache"):
        model(torch.tensor([[2, 3]]))
    model.reset_kv_cache()
    first_cache = model.cache
    result = model(torch.tensor([[2, 3, 4, 5]]), input_pos=torch.arange(7, 11),
                   input_pos_maxp1=11, lm_head_start=2)
    call, = hf.calls
    assert call["cache_position"].tolist() == [7, 8, 9, 10]
    assert call["logits_to_keep"] == 2 and result.shape == (1, 2, 32)
    assert call["past_key_values"] is first_cache and call["use_cache"] is True
    model.reset_kv_cache()
    assert model.cache is not first_cache and model.cache.layers == []
    with pytest.raises(ValueError, match="position window"):
        DMSModel(hf, max_seq_length=33)


@pytest.mark.parametrize("paged,contiguous", [([2, 3], [3, 3]), ([5, 6], None)])
def test_prefill_and_decode_count_both_storage_regions(paged, contiguous):
    model = DMSModel(FakeHF(), max_seq_length=16)
    model.cache = SimpleNamespace(layers=[make_layer(10, paged, contiguous) for _ in range(2)])
    counts = compression_snapshot(model, processed_tokens=10)
    assert counts["layer_group_count"] == 4
    assert counts["recent_slots"] == 12
    assert counts["exact_slots"] == 10
    assert counts["hierarchical_slots"] == 0
    assert counts["retained_slots"] == 22 and counts["dense_slots"] == 40
    assert counts["kv_retention_ratio"] == pytest.approx(22 / 40)
    assert counts["kv_compression_factor"] == pytest.approx(40 / 22)


def test_short_prefill_and_invalid_cache_are_not_misreported():
    model = DMSModel(FakeHF(), max_seq_length=16)
    model.cache = SimpleNamespace(layers=[make_layer(2, None, [2, 2]) for _ in range(2)])
    assert compression_snapshot(model, 2)["kv_retention_ratio"] == 1
    with pytest.raises(ValueError, match="forwarded token counts"):
        compression_snapshot(model, 3)
    model.cache.layers.pop()
    with pytest.raises(ValueError, match="incomplete"):
        compression_snapshot(model, 2)
    model.cache.layers = [make_layer(10, [1, 1], None) for _ in range(2)]
    with pytest.raises(ValueError, match="retained KV count"):
        compression_snapshot(model, 10)


def test_base_identity_rejects_missing_provenance_and_changed_inputs(tmp_path):
    (tmp_path / "lit_model.pth").write_bytes(b"the shared original Base")
    (tmp_path / "tokenizer.json").write_text('{"vocab": "same"}')
    provenance = {
        "source_weight_sha256": file_sha256(tmp_path / "lit_model.pth"),
        "tokenizer_files": {"tokenizer.json": file_sha256(tmp_path / "tokenizer.json")},
    }
    verify_base_source(provenance, tmp_path)
    with pytest.raises(ValueError, match="Missing"):
        verify_base_source(None, tmp_path)
    (tmp_path / "tokenizer.json").write_text('{"vocab": "different"}')
    with pytest.raises(ValueError, match="tokenizer differs"):
        verify_base_source(provenance, tmp_path)
    (tmp_path / "lit_model.pth").write_bytes(b"another Base version")
    with pytest.raises(ValueError, match="Base weights differ"):
        verify_base_source(provenance, tmp_path)


class Tokenizer:
    bos_id = 1
    eos_id = None

    def __init__(self):
        self.calls = []

    def encode(self, text, device="cpu", **kwargs):
        self.calls.append((text, kwargs))
        return torch.tensor([int(token) for token in text.split()], device=device)

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens.tolist())


@pytest.fixture(scope="module")
def wrapper_class():
    # Import real methods without eval.py's distributed/CUDA/HF initialization.
    source = Path(litgpt.__file__).resolve().parents[1] / "eval.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LogKVLM")
    names = {"_score_tokens", "generate_until", "loglikelihood", "loglikelihood_rolling",
             "_set_eval_cache", "_reset_eval_cache"}
    cls.bases, cls.decorator_list = [], []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(
        torch=torch, time=time, litgpt_generate=generate,
        cache_compression_snapshot=compression_snapshot,
        tqdm=SimpleNamespace(tqdm=lambda items, **kwargs: items),
        _global_rank=lambda: 0, _world_size=lambda: 1, _local_rank=lambda: 0,
        _rank_label=lambda: "CPU", _tqdm_position=lambda: 0,
        _request_sample_id=lambda req, **kwargs: kwargs["global_request_index"],
    )
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {
        "_synchronize_inference_device", "_reset_inference_peak_memory", "_inference_peak_memory",
    }]
    exec(compile(ast.Module(body=helpers + [cls], type_ignores=[]), str(source), "exec"), scope)
    return make_dms_lm(SimpleNamespace(LogKVLM=scope["LogKVLM"]))


def make_lm(wrapper_class, max_seq_length=8):
    # The request tests replace model loading only; cache lifecycle comes from
    # the baseline subclass, while all request methods remain inherited.
    lm = wrapper_class.__new__(wrapper_class)
    lm.model = DMSModel(FakeHF(), max_seq_length=max_seq_length)
    lm._device, lm.tokenizer = "cpu", Tokenizer()
    lm.all_gather_results = lambda items, tag: items
    lm.compression_metrics, lm.ppl_metrics, lm.gen_metrics = [], [], []
    lm.max_gen_toks = 3
    return lm


def test_shared_scoring_preserves_truncation_and_bos_policy(wrapper_class):
    lm = make_lm(wrapper_class)
    score, greedy = lm.loglikelihood([SimpleNamespace(args=("2 3 4 5 6 7 8", "9 10 11"))])[0]
    call, = lm.model.hf_model.calls
    assert call["input_ids"].tolist() == [[4, 5, 6, 7, 8, 9, 10, 11]]
    assert call["cache_position"].tolist() == list(range(8))
    assert call["logits_to_keep"] == 4
    assert lm.tokenizer.calls == [("2 3 4 5 6 7 8", {}), ("9 10 11", {"bos": False})]
    expected = -3 * math.log(math.exp(2) + 31)
    assert score == pytest.approx(expected, abs=1e-5) and greedy is False
    assert lm.compression_metrics[0]["processed_tokens"] == 8
    assert lm.compression_metrics[0]["kv_retention_ratio"] == 1
    assert lm.model.cache is None


@pytest.mark.parametrize("eos_id,generated", [(None, 3), (0, 1)])
def test_shared_generation_truncates_and_counts_only_forwarded_tokens(wrapper_class, eos_id, generated):
    lm = make_lm(wrapper_class)
    lm.tokenizer.eos_id = eos_id
    request = SimpleNamespace(args=("2 3 4 5 6 7 8 9 10 11", {
        "max_gen_toks": 3, "do_sample": False, "temperature": 2.0, "top_p": 1.0,
        "until": ["0"],  # Current shared evaluator stops on EOS only, not this string.
    }))
    result = lm.generate_until([request])
    assert result == [" ".join(["0"] * generated)]
    first, *decode = lm.model.hf_model.calls
    assert first["input_ids"].tolist() == [[7, 8, 9, 10, 11]]
    assert first["cache_position"].tolist() == list(range(5))
    assert first["logits_to_keep"] == 1
    assert [call["cache_position"].item() for call in decode] == list(range(5, 5 + generated - 1))
    record, = lm.compression_metrics
    assert record["prompt_len"] == 5 and record["generated_tokens"] == generated
    assert record["processed_tokens"] == 5 + generated - 1
    assert record["retained_slots"] == record["dense_slots"] == 4 * (5 + generated - 1)
    speed, = lm.gen_metrics
    assert speed["decode_tokens"] == generated - 1
    assert speed["total_time_s"] == pytest.approx(speed["prefill_time_s"] + speed["decode_time_s"])
    assert lm.model.cache is None


def test_shared_rolling_windows_keep_previous_token_context(wrapper_class):
    lm = make_lm(wrapper_class, max_seq_length=4)
    score, = lm.loglikelihood_rolling([SimpleNamespace(args=("2 3 4 5 6 7 8",))])
    assert math.isfinite(score)
    assert [call["input_ids"].tolist()[0] for call in lm.model.hf_model.calls] == [
        [1, 2, 3, 4], [4, 5, 6, 7], [7, 8],
    ]
    assert [row["processed_tokens"] for row in lm.compression_metrics] == [4, 4, 2]
    assert all(row["request_type"] == "loglikelihood_rolling" for row in lm.compression_metrics)
