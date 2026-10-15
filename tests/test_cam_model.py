"""Offline end-to-end checks with actual Qwen3 GQA weights and shared request methods."""

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from models.cam.model import CaMModel
from models.cam.semanticlogkv_eval import compression_snapshot, install_adapter
from models.qwen3.evaluate import load_evaluator
from models.qwen3.prepare_base import prepare_base
from tests.test_prepare_base import source


def tiny_model():
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=32, hidden_size=24, intermediate_size=48,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=64, rope_theta=1000000,
                        tie_word_embeddings=True, attention_dropout=0.0)
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def test_no_eviction_matches_qwen_gqa_logits_and_sliced_head():
    hf = tiny_model()
    tokens = torch.tensor([[2, 3, 5, 7, 11, 13, 17, 19, 23]])
    with torch.no_grad():
        expected = hf(tokens, use_cache=False).logits
    for dense in (False, True):
        model = CaMModel(copy.deepcopy(hf), 64, start_size=2, recent_size=16,
                         prefill_chunk_size=4, dense=dense)
        model.reset_kv_cache()
        actual = model(tokens, input_pos=torch.arange(9), lm_head_start=3)
        torch.testing.assert_close(actual, expected[:, 3:], atol=2e-6, rtol=2e-5)
        counts = compression_snapshot(model, 9)
        assert counts["kv_retention_ratio"] == 1
        assert counts["kv_storage_bytes"] == 2 * 2 * 2 * 9 * 8 * 4
        assert counts["attention_score_bytes"] == (0 if dense else 2 * 2 * 9 * 4)


def test_online_compression_is_chunk_invariant_causal_and_resettable():
    hf = tiny_model()
    tokens = torch.tensor([[2, 3, 5, 7, 11, 13, 17, 19, 23, 2, 3, 5, 7]])
    models = [CaMModel(copy.deepcopy(hf), 64, start_size=1, recent_size=4, merge_size=3,
                       seed=12, prefill_chunk_size=chunk) for chunk in (1, 4)]
    outputs = []
    for model in models:
        model.reset_kv_cache()
        outputs.append(model(tokens, input_pos=torch.arange(tokens.shape[1])))
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)
    for a, b in zip(models[0].layers, models[1].layers):
        for name in ("keys", "values", "scores"):
            torch.testing.assert_close(getattr(a, name), getattr(b, name), atol=2e-6, rtol=2e-5)
    model = models[1]
    snapshot = compression_snapshot(model, 13)
    assert snapshot["retained_slots"] == 2 * 2 * 5
    assert snapshot["dense_slots"] == 2 * 2 * 13
    assert snapshot["persistent_cache_bytes"] == 2 * 2 * 5 * (2 * 8 * 4 + 4)
    model.reset_kv_cache()
    replay = model(tokens)
    torch.testing.assert_close(replay, outputs[1], atol=0, rtol=0)
    changed = tokens.clone()
    changed[:, 8:] = 29
    model.reset_kv_cache()
    causal = model(changed)
    torch.testing.assert_close(causal[:, :8], replay[:, :8], atol=0, rtol=0)
    model.clear_kv_cache()
    assert model.cache is None and all(layer.keys is None for layer in model.layers)
    with pytest.raises(RuntimeError, match="request cache"):
        model(tokens)


def test_append_positions_and_window_are_checked():
    model = CaMModel(tiny_model(), 8, start_size=1, recent_size=4, merge_size=3)
    model.reset_kv_cache()
    model(torch.tensor([[2, 3]]))
    with pytest.raises(ValueError, match="append-only"):
        model(torch.tensor([[5]]), input_pos=torch.tensor([1]))
    with pytest.raises(ValueError, match="context window"):
        model(torch.ones(1, 7, dtype=torch.long))


def test_prepared_base_loads_and_inherits_real_reference_requests(source, tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.pre_tokenizers import Whitespace
    original, _, _, experiment = source
    tokenizer = Tokenizer.from_file(str(original / "tokenizer.json"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(original / "tokenizer.json"))
    output = tmp_path / "prepared"
    prepare_base(experiment, output)
    evaluator = load_evaluator()
    native = evaluator.LogKVLM
    options = dict(start_size=1, recent_size=4, merge_size=3, seed=42, prefill_chunk_size=4)
    install_adapter(evaluator, options)
    for name in ("_score_tokens", "loglikelihood", "loglikelihood_rolling", "generate_until"):
        assert getattr(evaluator.LogKVLM, name) is getattr(native, name)
    lm = evaluator.LogKVLM(str(output), device="cpu", tokenizer_dir=str(original),
                           config_overrides={"block_size": 32})
    score, greedy = lm.loglikelihood([SimpleNamespace(args=("blue " * 9, "blue"))])[0]
    assert torch.isfinite(torch.tensor(score)) and isinstance(greedy, bool)
    assert lm.compression_metrics[0]["kv_retention_ratio"] < 1
    assert lm.model.cache is None
    outputs = lm.generate_until([SimpleNamespace(args=("blue " * 9, {"max_gen_toks": 2, "do_sample": False}))])
    assert len(outputs) == 1
    assert lm.gen_metrics[0]["generated_tokens"] >= 1
    assert lm.compression_metrics[-1]["processed_tokens"] >= 9
    assert lm.model.cache is None
    # A fresh reference import still has its original model and global metric function.
    assert load_evaluator().LogKVLM.__name__ == "LogKVLM"
