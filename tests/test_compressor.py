"""SnapKV selection equals the upstream code, every layer keeps the same budget, and the cache drops for real."""

import math
from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from snapkv.compressor import SnapKVCompressor
from snapkv.kv_cache import CompressedKVCache
from snapkv.metrics import cache_compression_snapshot
from snapkv.qwen3 import repeat_kv
from tests.helpers import TINY, tiny_model


# ---- Reference 1: FasterDecoding/SnapKV@e216ddc snapkv/monkeypatch/snapkv_utils.py, SnapKVCluster.update_kv,
# body kept verbatim (KVCache-Factory@68cd955 SnapKVCluster.update_kv keeps the same lines for the query_head
# path). Query and key head counts are equal here, i.e. K/V already expanded to every query head.
def reference_update_kv(key_states, query_states, value_states, window_size, max_capacity_prompt,
                        kernel_size, pooling):
    assert key_states.shape[-2] == query_states.shape[-2]
    bsz, num_heads, q_len, head_dim = query_states.shape
    if q_len < max_capacity_prompt:
        return key_states, value_states
    else:
        attn_weights = torch.matmul(query_states[..., -window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -window_size:, -window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -window_size:, : -window_size].sum(dim = -2)
        if pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
        elif pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        indices = attn_cache.topk(max_capacity_prompt - window_size, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
        v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
        k_cur = key_states[:, :, -window_size:, :]
        v_cur = value_states[:, :, -window_size:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim = 2)
        value_states = torch.cat([v_past_compress, v_cur], dim = 2)
        return key_states, value_states


# ---- Reference 2: KVCache-Factory@68cd955 pyramidkv/pyramidkv_utils.py, the kv_head path of
# SnapKVCluster (_update_kv_kv_head): _reduce_group_scores, _grouped_window_attn_cache, _select_topk_kv
# kept verbatim (merge=None).
def _reduce_group_scores(scores, groups, gqa_score_agg):
    if groups == 1:
        return scores
    bsz, num_heads = scores.shape[0], scores.shape[1]
    grouped = scores.reshape(bsz, num_heads // groups, groups, *scores.shape[2:])
    if gqa_score_agg == 'mean':
        return grouped.mean(dim=2)
    elif gqa_score_agg == 'max':
        return grouped.amax(dim=2)
    elif gqa_score_agg == 'sum':
        return grouped.sum(dim=2)
    else:
        raise ValueError(f"GQA score aggregation {gqa_score_agg!r} not supported (expected 'mean', 'max' or 'sum')")


def _grouped_window_attn_cache(query_states, key_states, groups, window_size, kernel_size, pooling, gqa_score_agg):
    head_dim = query_states.shape[-1]
    key_rep = repeat_kv(key_states, groups)
    attn_weights = torch.matmul(query_states[..., -window_size:, :], key_rep.transpose(2, 3)) / math.sqrt(head_dim)
    mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    attn_weights[:, :, -window_size:, -window_size:] += mask[None, None, :, :]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights_sum = attn_weights[:, :, -window_size:, : -window_size].sum(dim=-2)
    attn_weights_sum = _reduce_group_scores(attn_weights_sum, groups, gqa_score_agg)
    if pooling == 'avgpool':
        attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    elif pooling == 'maxpool':
        attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    else:
        raise ValueError('Pooling method not supported')
    return attn_cache


def _select_topk_kv(key_states, value_states, attn_cache, capacity, window_size):
    head_dim = key_states.shape[-1]
    indices = attn_cache.topk(capacity, dim=-1).indices
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    k_past_compress = key_states[:, :, :-window_size, :].gather(dim=2, index=indices)
    v_past_compress = value_states[:, :, :-window_size, :].gather(dim=2, index=indices)
    k_cur = key_states[:, :, -window_size:, :]
    v_cur = value_states[:, :, -window_size:, :]
    key_states = torch.cat([k_past_compress, k_cur], dim=2)
    value_states = torch.cat([v_past_compress, v_cur], dim=2)
    return key_states, value_states


def reference_update_kv_kv_head(key_states, query_states, value_states, window_size, max_capacity_prompt,
                                kernel_size, pooling, gqa_score_agg):
    groups = query_states.shape[1] // key_states.shape[1]
    if query_states.shape[-2] < max_capacity_prompt:
        return key_states, value_states
    attn_cache = _grouped_window_attn_cache(query_states, key_states, groups, window_size, kernel_size, pooling,
                                            gqa_score_agg)
    return _select_topk_kv(key_states, value_states, attn_cache, max_capacity_prompt - window_size, window_size)
# ------------------------------------------------------------------------------------------


def tagged(B, H, L, D):
    """Random states whose channel 0 holds each entry's position, so a kept entry names where it came from."""
    states = torch.randn(B, H, L, D)
    states[..., 0] = torch.arange(L, dtype=states.dtype)
    return states


@pytest.mark.parametrize("granularity", ["kv_head", "query_head"])
@pytest.mark.parametrize("pooling,agg", [("maxpool", "mean"), ("avgpool", "max"), ("maxpool", "sum")])
@pytest.mark.parametrize("budget,window,kernel", [(24, 6, 5), (40, 8, 7), (89, 8, 7)])
def test_kept_entries_equal_the_upstream_snapkv(granularity, pooling, agg, budget, window, kernel):
    torch.manual_seed(0)
    B, H, H_kv, L, D = 1, 4, 2, 90, 16
    q = torch.randn(B, H, L, D)
    k, v = torch.randn(B, H_kv, L, D), tagged(B, H_kv, L, D)
    if granularity == "query_head":  # K/V expanded before selection, as the cache does in this layout
        k, v = repeat_kv(k, 2), repeat_kv(v, 2)
        ref_k, ref_v = reference_update_kv(k, q, v, window, budget, kernel, pooling)
    else:
        ref_k, ref_v = reference_update_kv_kv_head(k, q, v, window, budget, kernel, pooling, agg)
    compressor = SnapKVCompressor(3, max_capacity_prompt=budget, window_size=window, kernel_size=kernel,
                                  pooling=pooling, gqa_score_agg=agg)
    for layer in range(3):  # SnapKV: one budget for every layer
        keys, values, selected, recent = compressor.compress(layer, q, k, v)
        assert (selected, recent) == (budget - window, window) and keys.size(2) == budget
        positions = values[..., 0].long()
        # Same entries as upstream; stored in position order instead of upstream's score order.
        assert torch.equal(positions, ref_v[..., 0].long().sort(dim=-1).values)
        assert torch.equal(positions[..., -window:], torch.arange(L - window, L).expand_as(positions[..., -window:]))
        gather = positions.unsqueeze(-1).expand(-1, -1, -1, D)
        torch.testing.assert_close(keys, k.gather(2, gather), rtol=0, atol=0)
        torch.testing.assert_close(values, v.gather(2, gather), rtol=0, atol=0)
        # Attention over the kept entries does not depend on their order.
        probe = torch.randn(B, keys.size(1), 3, D)
        torch.testing.assert_close(F.scaled_dot_product_attention(probe, keys, values),
                                   F.scaled_dot_product_attention(probe, ref_k, ref_v), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("q_len,expected", [(100, 100), (2047, 2047), (2048, 2048), (2049, 2048), (32640, 2048)])
def test_every_layer_keeps_the_same_budget(q_len, expected):
    compressor = SnapKVCompressor(28, max_capacity_prompt=2048, window_size=8)
    assert compressor.budget_schedule(q_len) == [expected] * 28
    assert {compressor.layer_capacity(layer, q_len) for layer in range(28)} == {None if q_len < 2048 else 2040}


def test_short_prompt_is_kept_whole():
    compressor = SnapKVCompressor(3, max_capacity_prompt=32, window_size=8)
    q, k = torch.randn(1, 4, 20, 16), torch.randn(1, 2, 20, 16)
    keys, values, selected, recent = compressor.compress(0, q, k, k)
    assert keys is k and (selected, recent) == (12, 8)
    ref_k, _ = reference_update_kv_kv_head(k, q, k, 8, 32, 7, "maxpool", "mean")
    assert ref_k is k


@pytest.mark.parametrize("granularity,stored_heads", [("kv_head", 2), ("query_head", 4)])
def test_prefill_drops_entries_and_decode_appends(granularity, stored_heads):
    model = tiny_model()
    compressor = SnapKVCompressor(TINY.num_hidden_layers, max_capacity_prompt=16, window_size=4, kernel_size=3)
    model.set_kv_cache(partial(CompressedKVCache, compressor=compressor, granularity=granularity))
    prompt = torch.randint(0, TINY.vocab_size, (1, 50))
    with torch.no_grad():
        model(prompt, input_pos=torch.arange(50))
        assert [cache.length for cache in model.caches] == [16] * TINY.num_hidden_layers
        assert all(cache.compressed for cache in model.caches)
        for pos in range(50, 53):
            model(torch.tensor([[pos % TINY.vocab_size]]), input_pos=torch.tensor([pos]))
    counts = cache_compression_snapshot(model, 53)
    retained = (16 + 3) * TINY.num_hidden_layers
    assert counts["retained_slots"] == retained * stored_heads
    assert counts["dense_slots"] == 53 * TINY.num_hidden_layers * TINY.num_key_value_heads
    assert counts["layer_group_count"] == TINY.num_hidden_layers * TINY.num_key_value_heads
    assert counts["exact_slots"] == 12 * TINY.num_hidden_layers * stored_heads
    assert counts["recent_slots"] == (4 + 3) * TINY.num_hidden_layers * stored_heads
    assert counts["hierarchical_slots"] == 0
    assert counts["kv_retention_ratio"] == pytest.approx(counts["retained_slots"] / counts["dense_slots"])
    assert all(cache._k.size(2) >= cache.length for cache in model.caches)
    with pytest.raises(ValueError, match="expected 52"):
        cache_compression_snapshot(model, 52)


def test_compression_changes_later_tokens_but_not_the_prefill_logits():
    model = tiny_model(seed=1)
    tokens = torch.randint(0, TINY.vocab_size, (1, 60))
    compressor = SnapKVCompressor(TINY.num_hidden_layers, max_capacity_prompt=12, window_size=4, kernel_size=3)
    with torch.no_grad():
        dense = model(tokens)
        model.set_kv_cache(partial(CompressedKVCache, compressor=compressor))
        prefill = model(tokens[:, :50], input_pos=torch.arange(50))
        after = model(tokens[:, 50:], input_pos=torch.arange(50, 60))
    torch.testing.assert_close(prefill, dense[:, :50], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(after, dense[:, 50:], atol=1e-3)


def test_budget_covering_the_prompt_matches_fullkv():
    """A prompt as long as the budget keeps every entry, so later tokens see the dense cache."""
    model = tiny_model(seed=2)
    tokens = torch.randint(0, TINY.vocab_size, (1, 40))
    compressor = SnapKVCompressor(TINY.num_hidden_layers, max_capacity_prompt=32, window_size=4, kernel_size=3)
    with torch.no_grad():
        dense = model(tokens)
        model.set_kv_cache(partial(CompressedKVCache, compressor=compressor))
        model(tokens[:, :32], input_pos=torch.arange(32))
        after = model(tokens[:, 32:], input_pos=torch.arange(32, 40))
    assert not any(cache.compressed for cache in model.caches)
    torch.testing.assert_close(after, dense[:, 32:], atol=1e-5, rtol=1e-5)


def test_invalid_settings_fail_before_running():
    with pytest.raises(ValueError, match="window_size"):
        SnapKVCompressor(28, max_capacity_prompt=8, window_size=8)
    with pytest.raises(ValueError, match="odd"):
        SnapKVCompressor(28, kernel_size=4)
    with pytest.raises(ValueError, match="pooling"):
        SnapKVCompressor(28, pooling="minpool")
    with pytest.raises(ValueError, match="gqa_score_agg"):
        SnapKVCompressor(28, gqa_score_agg="median")
