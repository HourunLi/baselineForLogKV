"""Small CPU checks of CaM's merge formula and its Qwen3 GQA integration."""

import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from models.cam.cache import CaMCache
from models.cam.attention import install_cam_attention


def tiny_model():
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, attention_dropout=0.0,
        _attn_implementation="eager",
    )).eval()


def run(model, ids):
    with torch.no_grad():
        return model(ids, use_cache=False, attention_mask={"full_attention": None}).logits


@pytest.mark.parametrize("start,recent,merge", [(-1, 16, 16), (0, 0, 1), (0, 1, 0), (2, 3, 4)])
def test_invalid_cache_budget(start, recent, merge):
    with pytest.raises(ValueError, match="recent_size >= merge_size"):
        CaMCache(start, recent, merge)


def test_warmup_accumulates_attention_in_fp32():
    cache = CaMCache(1, 3, 2)
    for index in range(3):
        token = torch.full((1, 2, 1, 4), float(index), dtype=torch.bfloat16)
        cache.append(token, token)
        cache.update(torch.full((1, 2, index + 1), 1 / (index + 1)))
    assert cache.scores.dtype == torch.float32
    torch.testing.assert_close(cache.scores[0, 0], torch.tensor([1 + .5 + 1/3, .5 + 1/3, 1/3]))
    assert cache.processed_tokens == 3


def test_seeded_merge_matches_formula_and_really_removes_kv():
    cache = CaMCache(1, 3, merge_size=2, seed=39)
    keys = torch.arange(20.0).reshape(1, 4, 5, 1)
    values = keys * 10
    for index in range(4):
        cache.append(keys[:, :, index:index + 1], values[:, :, index:index + 1])
        cache.update(torch.zeros(1, 4, index + 1))
    cache.append(keys[:, :, 4:5], values[:, :, 4:5])
    scores = torch.tensor([[[.0, .1, .2, .2, .5],
                            [.0, .0, .0, .0, .0],
                            [.0, .2, .1, .1, .6],
                            [.0, .1, .0, .0, .9]]])
    probability = torch.tensor([[.5, .0, 1., 1.]])
    selected = torch.bernoulli(probability, generator=torch.Generator().manual_seed(39))
    expected_values = values.clone()
    expected_values[:, :, 2:4] += values[:, :, 1:2] * selected[..., None, None] / 2
    retained = [0, 2, 3, 4]
    cache.update(scores)
    torch.testing.assert_close(cache.values, expected_values[:, :, retained])
    torch.testing.assert_close(cache.keys, keys[:, :, retained])
    torch.testing.assert_close(cache.scores, scores[:, :, retained])
    assert cache.keys.shape[2] == cache.cache_size == 4
    assert cache.processed_tokens == 5


def test_zero_sinks_repeated_evictions_and_reset_do_not_touch_global_rng():
    cache = CaMCache(0, 2, merge_size=2, seed=11)
    before = torch.random.get_rng_state().clone()
    snapshots = []
    for _ in range(2):
        cache.reset()
        assert cache.keys is None and cache.scores is None and cache.processed_tokens == 0
        for index in range(9):
            token = torch.full((1, 2, 1, 4), float(index))
            cache.append(token, token)
            cache.update(torch.ones_like(cache.scores) / cache.scores.shape[-1])
        assert cache.processed_tokens - cache.keys.shape[2] == 7
        snapshots.append(cache.values.clone())
    torch.testing.assert_close(snapshots[0], snapshots[1])
    torch.testing.assert_close(torch.random.get_rng_state(), before)


def test_uncompressed_qwen3_matches_dense_and_preserves_model_parameters():
    dense = tiny_model()
    cam = copy.deepcopy(dense)
    parameter_names = list(cam.state_dict())
    caches = install_cam_attention(cam, 2, 16, merge_size=2, seed=7)
    ids = torch.tensor([[3, 5, 7, 4, 2, 11, 13]])
    with torch.no_grad():
        expected = dense(ids, use_cache=False).logits
    actual = run(cam, ids)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert list(cam.state_dict()) == parameter_names
    assert [cache.seed for cache in caches] == [7, 8]
    assert all(cache.keys.shape == (1, 2, 7, 8) for cache in caches)


def test_first_over_budget_output_still_attends_to_the_victim():
    dense = tiny_model()
    cam = copy.deepcopy(dense)
    caches = install_cam_attention(cam, 1, 3, merge_size=2)
    ids = torch.tensor([[3, 5, 7, 4, 2]])
    with torch.no_grad():
        expected = dense(ids, use_cache=False).logits
    torch.testing.assert_close(run(cam, ids), expected, rtol=2e-5, atol=2e-6)
    assert all(cache.keys.shape[2] == 4 for cache in caches)


def test_gqa_scores_are_mean_query_head_probabilities_and_keys_are_unrotated():
    model = tiny_model()
    cache = install_cam_attention(model, 1, 8, merge_size=2)[0]
    attn = model.model.layers[0].self_attn
    hidden = torch.arange(4 * 32, dtype=torch.float32).reshape(1, 4, 32).sin()
    with torch.no_grad():
        shape = (1, 4, -1, 8)
        queries = attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1, 2)
        raw_keys = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
        cos, sin = model.model.rotary_emb(hidden, torch.arange(4).unsqueeze(0))
        queries, keys = apply_rotary_pos_emb(queries, raw_keys, cos, sin)
        scores = queries @ repeat_kv(keys, 2).transpose(-1, -2) * attn.scaling
        mask = torch.ones(4, 4, dtype=torch.bool).triu(1)
        probabilities = scores.masked_fill(mask, -torch.inf).softmax(-1)
        expected = probabilities.reshape(1, 2, 2, 4, 4).mean(2).sum(2)
        attn(hidden)
    torch.testing.assert_close(cache.scores, expected)
    torch.testing.assert_close(cache.keys, raw_keys)


def test_chunking_reset_and_future_tokens_do_not_change_prefix_outputs():
    model = tiny_model()
    caches = install_cam_attention(model, 1, 3, merge_size=2, seed=19)
    ids = torch.tensor([[3, 5, 7, 4, 2, 11, 13, 9, 8, 15]])
    whole = run(model, ids)
    final_values = [cache.values.clone() for cache in caches]
    for cache in caches:
        cache.reset()
    chunked = torch.cat([run(model, chunk) for chunk in ids.split(3, dim=1)], dim=1)
    torch.testing.assert_close(chunked, whole, atol=2e-6, rtol=2e-5)
    for cache, expected in zip(caches, final_values):
        torch.testing.assert_close(cache.values, expected, atol=2e-6, rtol=2e-5)
        cache.reset()
    changed = ids.clone()
    changed[:, 7:] = torch.tensor([1, 2, 3])
    changed_result = run(model, changed)
    torch.testing.assert_close(changed_result[:, :7], whole[:, :7], atol=2e-6, rtol=2e-5)
    assert all(cache.keys.shape[2] == 4 and cache.processed_tokens == 10 for cache in caches)


def test_external_masks_caches_and_training_are_rejected():
    model = tiny_model()
    install_cam_attention(model, 0, 2, merge_size=2)
    attn = model.model.layers[0].self_attn
    hidden = torch.zeros(1, 1, 32)
    with pytest.raises(ValueError, match="unpadded"):
        attn(hidden, attention_mask=torch.ones(1, 1))
    with pytest.raises(ValueError, match="owns its KV cache"):
        attn(hidden, past_key_values=object())
    with pytest.raises(ValueError, match="owns its KV cache"):
        attn(hidden, use_cache=True)
    attn.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        attn(hidden)
