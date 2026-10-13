"""ChunkKV wrapper against LitGPT's own dense KV-cache inference on a tiny Qwen3-shaped model."""

import pytest
import torch

import chunkkv.model as chunkkv_model
from chunkkv.core import ChunkKVSettings
from chunkkv.model import ChunkKVModel

from .conftest import tiny_config


PROMPT, DECODE = 50, 6


def decode_tokens(model, prompt, tokens):
    """Prompt prefill (generation's lm_head_start) followed by one-token decode steps."""
    logits = [model(prompt, torch.arange(prompt.size(1)), lm_head_start=prompt.size(1) - 1)]
    for step, token in enumerate(tokens):
        position = torch.tensor([prompt.size(1) + step])
        logits.append(model(token.view(1, 1), position, lm_head_start=0))
    return torch.cat(logits, dim=1)


def test_uncompressed_cache_matches_litgpt_dense_inference(gpt_pair):
    reference, gpt = gpt_pair
    model = ChunkKVModel(gpt, ChunkKVSettings(budget=PROMPT, window_size=4, chunk_size=3))
    reference.set_kv_cache(1, device="cpu", dtype=torch.float32)
    prompt, tokens = torch.randint(0, 64, (1, PROMPT)), torch.randint(0, 64, (DECODE,))
    with torch.no_grad():
        model.reset_cache()
        actual = decode_tokens(model, prompt, tokens)
        expected = decode_tokens(reference, prompt, tokens)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        counts = model.kv_compression_snapshot(PROMPT + DECODE)
        assert counts["retained_slots"] == counts["dense_slots"] == 4 * 2 * (PROMPT + DECODE)
        # A scored continuation behind an uncompressed context is one dense forward.
        model.reset_cache()
        sequence = torch.cat((prompt, tokens.view(1, -1)), dim=1)
        scored = model(sequence, torch.arange(sequence.size(1)), lm_head_start=PROMPT - 1)
        torch.testing.assert_close(scored, reference(sequence)[:, PROMPT - 1:], rtol=1e-5, atol=1e-5)


def masked_reference(reference, kept, prompt_len):
    """Dense LitGPT whose post-prompt queries cannot see the evicted prompt keys."""
    reference.set_kv_cache(1, device="cpu", dtype=torch.float32)
    evicted = torch.ones(prompt_len, dtype=torch.bool)
    evicted[kept] = False
    reference.mask_cache[..., prompt_len:, :prompt_len] &= ~evicted
    return reference


def test_compressed_decode_attends_to_kept_prompt_entries_only(gpt_pair):
    reference, gpt = gpt_pair
    settings = ChunkKVSettings(budget=20, window_size=5, chunk_size=4, reuse_layers=4)
    model = ChunkKVModel(gpt, settings)
    prompt, tokens = torch.randint(0, 64, (1, PROMPT)), torch.randint(0, 64, (DECODE,))
    with torch.no_grad():
        model.reset_cache()
        first = model(prompt, torch.arange(PROMPT), lm_head_start=PROMPT - 1)
        kept = model.request.indices.clone()  # one index set for all four layers (reuse depth 4)
        assert len(kept) == 5 + 4 * 3 and kept[-5:].tolist() == list(range(45, 50))
        rest = [model(token.view(1, 1), torch.tensor([PROMPT + i]), lm_head_start=0)
                for i, token in enumerate(tokens)]
        actual = torch.cat([first, *rest], dim=1)
        expected = decode_tokens(masked_reference(reference, kept, PROMPT), prompt, tokens)
    # The prompt itself is prefilled densely; only later tokens see the compressed cache.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(actual[:, 1:], decode_tokens(
        masked_reference(reference, torch.arange(PROMPT), PROMPT), prompt, tokens)[:, 1:])
    counts = model.kv_compression_snapshot(PROMPT + DECODE)
    groups = 4 * 2
    assert counts["exact_slots"] == 12 * groups
    assert counts["recent_slots"] == (5 + DECODE) * groups
    assert counts["retained_slots"] == (17 + DECODE) * groups
    assert counts["dense_slots"] == (PROMPT + DECODE) * groups


def test_each_layer_scores_its_own_queries_unless_indices_are_reused(gpt_pair, monkeypatch):
    reference, gpt = gpt_pair
    reference.set_kv_cache(1, device="cpu", dtype=torch.float32)
    prompt = torch.randint(0, 64, (1, PROMPT))
    with torch.no_grad():
        reference(prompt, torch.arange(PROMPT))
    dense_keys = [block.attn.kv_cache.k[:, :, :PROMPT] for block in reference.transformer.h]

    calls = []
    original = chunkkv_model.observation_scores

    def spy(q, k, window, scale):
        calls.append(k.clone())
        return original(q, k, window, scale)
    monkeypatch.setattr(chunkkv_model, "observation_scores", spy)
    model = ChunkKVModel(gpt, ChunkKVSettings(budget=21, window_size=5, chunk_size=4, reuse_layers=2))
    selections = []
    original_keep = chunkkv_model.keep_indices

    def keep_spy(*args):
        selections.append(original_keep(*args))
        return selections[-1]
    monkeypatch.setattr(chunkkv_model, "keep_indices", keep_spy)
    with torch.no_grad():
        model.reset_cache()
        model(prompt, torch.arange(PROMPT), lm_head_start=PROMPT - 1)
    # Layers 0 and 2 score with their own keys; layers 1 and 3 reuse those indices.
    assert len(calls) == 2
    torch.testing.assert_close(calls[0], dense_keys[0])
    torch.testing.assert_close(calls[1], dense_keys[2])
    for layer, attn in enumerate(model.attentions):
        indices, chunk_tokens = selections[layer // 2]
        assert attn.chunk_tokens == chunk_tokens == 16
        torch.testing.assert_close(attn.keys[:, :, :attn.length], dense_keys[layer][:, :, indices])


def test_scored_continuation_equals_prompt_prefill_then_decode(gpt_pair):
    _, gpt = gpt_pair
    model = ChunkKVModel(gpt, ChunkKVSettings(budget=18, window_size=6, chunk_size=3))
    context, continuation = torch.randint(0, 64, (1, PROMPT)), torch.randint(0, 64, (1, 5))
    sequence = torch.cat((context, continuation), dim=1)
    with torch.no_grad():
        model.reset_cache()
        scored = model(sequence, torch.arange(sequence.size(1)), lm_head_start=PROMPT - 1)
        snapshot = model.kv_compression_snapshot(sequence.size(1))
        model.reset_cache()
        stepwise = decode_tokens(model, context, continuation[0])
    assert scored.shape == (1, 6, 64)
    torch.testing.assert_close(scored, stepwise, rtol=1e-5, atol=1e-5)
    assert snapshot["retained_slots"] == (18 + 5) * 4 * 2


def test_request_lifecycle_and_input_checks(gpt_pair):
    _, gpt = gpt_pair
    model = ChunkKVModel(gpt, ChunkKVSettings(keep_ratio=0.5, window_size=2, chunk_size=2))
    tokens = torch.randint(0, 64, (1, 8))
    with pytest.raises(RuntimeError, match="request cache"):
        model(tokens, torch.arange(8))
    model.reset_cache()
    with pytest.raises(ValueError, match="contiguous"):
        model(tokens, torch.arange(1, 9))
    with pytest.raises(ValueError, match="batch size 1"):
        model(tokens.repeat(2, 1), torch.arange(8))
    with pytest.raises(ValueError, match="lm_head_start"):
        model(tokens, torch.arange(8), lm_head_start=8)
    with pytest.raises(ValueError, match="window"):
        model(torch.zeros(1, 129, dtype=torch.long), torch.arange(129))
    with torch.no_grad():
        model(tokens, torch.arange(8))
    assert model.kv_compression_snapshot(8)["retained_slots"] == 4 * 4 * 2
    with pytest.raises(ValueError, match="counts differ"):
        model.kv_compression_snapshot(9)
    model.release_cache()
    assert model.processed is None and all(attn.keys is None for attn in model.attentions)


def test_decode_capacity_grows_beyond_the_preallocated_tail(gpt_pair, monkeypatch):
    reference, gpt = gpt_pair
    monkeypatch.setattr(chunkkv_model.ChunkKVAttention, "growth", 2)
    model = ChunkKVModel(gpt, ChunkKVSettings(budget=PROMPT, window_size=4, chunk_size=3))
    reference.set_kv_cache(1, device="cpu", dtype=torch.float32)
    prompt, tokens = torch.randint(0, 64, (1, 10)), torch.randint(0, 64, (7,))
    with torch.no_grad():
        model.reset_cache()
        torch.testing.assert_close(decode_tokens(model, prompt, tokens), decode_tokens(reference, prompt, tokens),
                                   rtol=1e-5, atol=1e-5)


def test_unsupported_architectures_are_rejected():
    from litgpt.model import GPT

    with pytest.raises(ValueError, match="sliding_window_size"):
        ChunkKVModel(GPT(tiny_config(sliding_window_size=8)), ChunkKVSettings(budget=32, window_size=4))
