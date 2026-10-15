"""Request handling follows SemanticLogKV's contract with a real model and cache on CPU."""

import math
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from h2okv.compressor import H2OCompressor
from h2okv.kv_cache import CompressedKVCache
from h2okv.lm import CompressedKVLM
from h2okv.qwen3 import Qwen3
from tests.helpers import TINY, tiny_model


class Tokenizer:
    """Space-separated integer ids; BOS=1, configurable EOS."""
    bos_id = 1

    def __init__(self, eos_id=None):
        self.eos_id = eos_id
        self.calls = []

    def encode(self, text, device=None, bos=None, **kwargs):
        self.calls.append((text, {"bos": bos} if bos is not None else {}))
        return torch.tensor([int(token) for token in text.split()], dtype=torch.int, device=device)

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens.tolist())


class FavourZero(Qwen3):
    """Real model and cache; logits nudged so greedy decoding emits token 0."""

    def forward(self, *args, **kwargs):
        logits = super().forward(*args, **kwargs)
        logits[..., 0] += 1000.0
        return logits


def make_lm(max_seq_length=8, compressor=None, mode="compress_context", eos_id=None, favour_zero=False):
    model = tiny_model(max_seq_length=max_seq_length)
    if favour_zero:
        biased = FavourZero(TINY, max_seq_length=max_seq_length)
        biased.load_state_dict(model.state_dict())
        model = biased.eval()
    factory = partial(CompressedKVCache, compressor=compressor)
    return CompressedKVLM("unused", factory, model=model, tokenizer=Tokenizer(eos_id), device="cpu",
                          loglikelihood_mode=mode)


def manual_score(model, ctx, cont):
    with torch.no_grad():
        logits = model(torch.tensor([ctx + cont]))[0, len(ctx) - 1:-1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[torch.arange(len(cont)), torch.tensor(cont)].sum().item()


def test_scoring_truncates_left_and_encodes_continuation_without_bos():
    lm = make_lm()
    score, greedy = lm.loglikelihood([SimpleNamespace(args=("2 3 4 5 6 7 8", "9 10 11"))])[0]
    assert lm.tokenizer.calls == [("2 3 4 5 6 7 8", {}), ("9 10 11", {"bos": False})]
    assert score == pytest.approx(manual_score(lm.model, [4, 5, 6, 7, 8], [9, 10, 11]), abs=1e-4)
    assert isinstance(greedy, bool)
    record, = lm.compression_metrics
    assert record["processed_tokens"] == 8 and record["context_len"] == 5 and record["kv_retention_ratio"] == 1
    assert lm.ppl_metrics[0]["total_seq_len"] == 8
    assert all(cache is None for cache in lm.model.caches)  # released after the request


def test_long_continuation_drops_context_and_uses_bos():
    lm = make_lm(max_seq_length=4)
    lm.loglikelihood([SimpleNamespace(args=("2 3", "4 5 6 7 8"))])
    record, = lm.compression_metrics
    assert record["context_len"] == 1 and record["cont_len"] == 3 and record["processed_tokens"] == 4


def test_compressed_context_scoring_differs_from_single_forward_which_equals_fullkv():
    ctx = " ".join(str(2 + i % 50) for i in range(40))
    request = [SimpleNamespace(args=(ctx, "7 8 9 10"))]
    make = partial(H2OCompressor, max_capacity_prompt=12, window_size=4)
    full = make_lm(max_seq_length=64).loglikelihood(request)[0][0]
    single = make_lm(max_seq_length=64, compressor=make(), mode="single_forward")
    compressed = make_lm(max_seq_length=64, compressor=make())
    assert single.loglikelihood(request)[0][0] == pytest.approx(full, abs=1e-4)
    assert abs(compressed.loglikelihood(request)[0][0] - full) > 1e-3
    record, = compressed.compression_metrics
    assert record["processed_tokens"] == 44 and record["kv_retention_ratio"] < 1


@pytest.mark.parametrize("eos_id,generated", [(None, 3), (0, 1)])
def test_generation_reserves_output_space_and_counts_forwarded_tokens(eos_id, generated):
    lm = make_lm(eos_id=eos_id, favour_zero=True)
    request = SimpleNamespace(args=("2 3 4 5 6 7 8 9 10 11", {
        "max_gen_toks": 3, "do_sample": False, "temperature": 2.0, "top_p": 1.0,
        "until": ["0"],  # not applied: generation stops on EOS or length, like the reference
    }))
    assert lm.generate_until([request]) == [" ".join(["0"] * generated)]
    record, = lm.compression_metrics
    assert record["prompt_len"] == 5 and record["generated_tokens"] == generated
    assert record["processed_tokens"] == 5 + generated - 1
    groups = TINY.num_hidden_layers * TINY.num_key_value_heads
    assert record["retained_slots"] == record["dense_slots"] == groups * (5 + generated - 1)
    speed, = lm.gen_metrics
    assert speed["decode_tokens"] == generated - 1 and speed["prompt_len"] == 5
    assert speed["total_time_s"] == pytest.approx(speed["prefill_time_s"] + speed["decode_time_s"])


@pytest.mark.parametrize("decode_evict,kept", [(False, 12 + 4), (True, 12)])
def test_generation_with_h2o_keeps_budget_plus_generated_tokens(decode_evict, kept):
    compressor = H2OCompressor(max_capacity_prompt=12, window_size=4, decode_evict=decode_evict)
    lm = make_lm(max_seq_length=64, compressor=compressor, favour_zero=True)
    prompt = " ".join(str(2 + i % 50) for i in range(40))
    lm.generate_until([SimpleNamespace(args=(prompt, {"max_gen_toks": 5, "do_sample": False}))])
    record, = lm.compression_metrics
    assert record["processed_tokens"] == 44
    assert record["retained_slots"] == kept * TINY.num_hidden_layers * TINY.num_key_value_heads


def test_rolling_windows_condition_on_previous_token():
    lm = make_lm(max_seq_length=4)
    score, = lm.loglikelihood_rolling([SimpleNamespace(args=("2 3 4 5 6 7 8",))])
    assert math.isfinite(score)
    assert [row["processed_tokens"] for row in lm.compression_metrics] == [4, 4, 2]
    assert [row["context_len"] for row in lm.compression_metrics] == [1, 1, 1]
    assert all(row["request_type"] == "loglikelihood_rolling" for row in lm.compression_metrics)


def test_requests_are_split_by_rank(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    lm = make_lm()
    requests = [SimpleNamespace(args=(f"{2 + i} 3", "4")) for i in range(5)]
    assert len(lm.loglikelihood(requests)) == 2  # requests[1::2]; gathering needs a process group


@pytest.mark.parametrize("decode_evict", [False, True])
def test_bfloat16_generation_and_scoring_with_compression(decode_evict):
    compressor = H2OCompressor(max_capacity_prompt=12, window_size=4, decode_evict=decode_evict, score_chunk_size=7)
    lm = make_lm(max_seq_length=64, compressor=compressor)
    lm.model.to(torch.bfloat16)
    prompt = " ".join(str(2 + i % 50) for i in range(40))
    output, = lm.generate_until([SimpleNamespace(args=(prompt, {"max_gen_toks": 4, "do_sample": False}))])
    score, _ = lm.loglikelihood([SimpleNamespace(args=(prompt, "7 8"))])[0]
    assert len(output.split()) == 4 and math.isfinite(score)
    assert lm.compression_metrics[0]["kv_retention_ratio"] < 0.5
