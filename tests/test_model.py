"""The self-contained Qwen3 matches a reference implementation and both checkpoint layouts."""

from functools import partial

import pytest
import torch

from h2okv.kv_cache import CompressedKVCache
from h2okv.qwen3 import load_qwen3
from tests.helpers import TINY, save_hf_checkpoint, save_litgpt_checkpoint, tiny_model


def full_cache(model, granularity="kv_head"):
    model.set_kv_cache(partial(CompressedKVCache, compressor=None, granularity=granularity))


@pytest.mark.parametrize("granularity", ["kv_head", "query_head"])
def test_cached_prefill_decode_and_continuation_match_one_dense_forward(granularity):
    model = tiny_model()
    tokens = torch.randint(0, TINY.vocab_size, (1, 24))
    with torch.no_grad():
        reference = model(tokens)
        full_cache(model, granularity)
        pieces = [model(tokens[:, :10], input_pos=torch.arange(10))]
        pieces.append(model(tokens[:, 10:17], input_pos=torch.arange(10, 17)))  # scored continuation
        for pos in range(17, 24):  # token-by-token decode
            pieces.append(model(tokens[:, pos:pos + 1], input_pos=torch.tensor([pos])))
    torch.testing.assert_close(torch.cat(pieces, dim=1), reference, atol=1e-5, rtol=1e-5)
    assert all(cache.processed_tokens == 24 and cache.length == 24 for cache in model.caches)


def test_lm_head_start_only_materialises_requested_logits():
    model = tiny_model()
    tokens = torch.randint(0, TINY.vocab_size, (1, 12))
    with torch.no_grad():
        full = model(tokens)
        full_cache(model)
        tail = model(tokens, input_pos=torch.arange(12), lm_head_start=9)
    assert tail.shape == (1, 3, TINY.vocab_size)
    torch.testing.assert_close(tail, full[:, 9:], atol=1e-5, rtol=1e-5)


def test_positions_must_continue_the_request():
    model = tiny_model()
    with pytest.raises(RuntimeError, match="set_kv_cache"):
        model(torch.tensor([[1, 2]]), input_pos=torch.arange(2))
    full_cache(model)
    model(torch.tensor([[1, 2, 3]]), input_pos=torch.arange(3))
    with pytest.raises(ValueError, match="contiguously"):
        model(torch.tensor([[4]]), input_pos=torch.tensor([5]))
    with pytest.raises(ValueError, match="max_seq_length"):
        model.max_seq_length = TINY.max_position_embeddings + 1


@pytest.mark.parametrize("tie", [False, True])
def test_litgpt_and_hf_layouts_load_to_identical_models(tmp_path, tie):
    model = tiny_model(seed=3)
    hf_dir = save_hf_checkpoint(model, tmp_path / "hf", tie=tie)  # ties the head first when requested
    lit_dir = save_litgpt_checkpoint(model, tmp_path / "lit")
    tokens = torch.randint(0, TINY.vocab_size, (1, 16))
    with torch.no_grad():
        expected = model(tokens)
        outputs = {}
        for name, directory in (("lit", lit_dir), ("hf", hf_dir)):
            loaded, info = load_qwen3(directory, max_seq_length=64, device="cpu", dtype=torch.float32)
            outputs[name] = loaded(tokens)
            assert info["weights_format"] == ("litgpt" if name == "lit" else "hf")
            if name == "hf":
                assert info["tied_lm_head"] is tie
                assert (loaded.lm_head.weight is loaded.transformer.wte.weight) is tie
    torch.testing.assert_close(outputs["lit"], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(outputs["hf"], expected, atol=1e-6, rtol=1e-6)


def test_loading_rejects_missing_or_unknown_weights(tmp_path):
    model = tiny_model()
    directory = save_litgpt_checkpoint(model, tmp_path / "lit")
    state = torch.load(directory / "lit_model.pth")
    state.pop("transformer.h.0.attn.norm_k.weight")
    torch.save(state, directory / "lit_model.pth")
    with pytest.raises(RuntimeError, match="norm_k"):
        load_qwen3(directory, 64, device="cpu", dtype=torch.float32)


def test_matches_transformers_qwen3_reference(tmp_path):
    transformers = pytest.importorskip("transformers")
    model = tiny_model(seed=5)
    directory = save_hf_checkpoint(model, tmp_path / "hf")
    reference = transformers.Qwen3ForCausalLM.from_pretrained(directory, torch_dtype=torch.float32,
                                                              attn_implementation="eager").eval()
    tokens = torch.randint(0, TINY.vocab_size, (1, 20))
    with torch.no_grad():
        expected = reference(tokens).logits
        actual = model(tokens)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
