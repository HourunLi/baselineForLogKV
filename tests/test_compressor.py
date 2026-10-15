"""PyramidKV selection equals KVCache-Factory's PyramidKVCluster, and the cache drops for real."""

import math
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from pyramidkv.compressor import PyramidKVCompressor
from pyramidkv.kv_cache import CompressedKVCache
from pyramidkv.metrics import cache_compression_snapshot
from pyramidkv.qwen3 import repeat_kv
from tests.helpers import TINY, tiny_model


# ---- Reference: KVCache-Factory pyramidkv/pyramidkv_utils.py, kept verbatim in logic ----
def reference_capacity(layer_idx, q_len, num_layers, budget, window, beta):
    min_num = (budget - window) // beta
    max_num = (budget - window) * 2 - min_num
    if max_num >= q_len - window:
        max_num = q_len - window
        min_num = (budget - window) * 2 - max_num
    steps = (max_num - min_num) // (num_layers - 1)
    max_capacity_prompt = max_num - layer_idx * steps
    if q_len < budget:
        return None
    if q_len < (budget - window) * 2:
        return budget - window
    return max_capacity_prompt


def reference_indices(query_states, key_states, capacity, window, kernel, pooling, agg="mean"):
    """KVCache-Factory scoring: transient repeat_kv, window softmax, group reduce, pool, top-k."""
    head_dim = query_states.shape[-1]
    groups = query_states.shape[1] // key_states.shape[1]
    key_rep = repeat_kv(key_states, groups)
    attn = torch.matmul(query_states[..., -window:, :], key_rep.transpose(2, 3)) / math.sqrt(head_dim)
    mask = torch.full((window, window), torch.finfo(attn.dtype).min, device=attn.device)
    cond = torch.arange(mask.size(-1))
    mask.masked_fill_(cond < (cond + 1).view(mask.size(-1), 1), 0)
    attn[:, :, -window:, -window:] += mask[None, None, :, :]
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
    summed = attn[:, :, -window:, :-window].sum(dim=-2)
    if groups > 1:
        grouped = summed.reshape(summed.size(0), summed.size(1) // groups, groups, -1)
        summed = {"mean": grouped.mean(2), "max": grouped.amax(2), "sum": grouped.sum(2)}[agg]
    pool = F.avg_pool1d if pooling == "avgpool" else F.max_pool1d
    scores = pool(summed, kernel_size=kernel, padding=kernel // 2, stride=1)
    return scores.topk(capacity, dim=-1).indices
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("q_len", [40, 63, 64, 100, 200, 1000, 5000])
@pytest.mark.parametrize("budget,window,beta", [(32, 8, 20), (64, 8, 5), (2048, 8, 20)])
def test_pyramid_budget_schedule_matches_reference(q_len, budget, window, beta):
    compressor = PyramidKVCompressor(28, max_capacity_prompt=budget, window_size=window, beta=beta)
    for layer in range(28):
        assert compressor.layer_capacity(layer, q_len) == reference_capacity(layer, q_len, 28, budget, window, beta)


def test_average_entries_per_layer_track_the_budget_and_decrease_with_depth():
    compressor = PyramidKVCompressor(28, max_capacity_prompt=2048, window_size=8)
    schedule = compressor.budget_schedule(32640)
    assert schedule == sorted(schedule, reverse=True) and schedule[0] > 2048 > schedule[-1]
    assert abs(sum(schedule) / len(schedule) - 2048) / 2048 < 0.01


@pytest.mark.parametrize("granularity", ["kv_head", "query_head"])
@pytest.mark.parametrize("pooling,agg", [("maxpool", "mean"), ("avgpool", "max")])
def test_selected_entries_equal_kvcache_factory(granularity, pooling, agg):
    torch.manual_seed(0)
    B, H, H_kv, L, D = 1, 4, 2, 90, 16
    q = torch.randn(B, H, L, D)
    k, v = torch.randn(B, H_kv, L, D), torch.randn(B, H_kv, L, D)
    if granularity == "query_head":  # legacy layout: K/V repeated before selection
        k, v = repeat_kv(k, 2), repeat_kv(v, 2)
    compressor = PyramidKVCompressor(3, max_capacity_prompt=24, window_size=6, kernel_size=5,
                                     pooling=pooling, beta=4, gqa_score_agg=agg)
    for layer in range(3):
        capacity = compressor.layer_capacity(layer, L)
        keys, values, selected, recent = compressor.compress(layer, q, k, v)
        indices = reference_indices(q, k, capacity, 6, 5, pooling, agg).sort(dim=-1).values
        gather = indices.unsqueeze(-1).expand(-1, -1, -1, D)
        expected_k = torch.cat((k[:, :, :-6].gather(2, gather), k[:, :, -6:]), dim=2)
        expected_v = torch.cat((v[:, :, :-6].gather(2, gather), v[:, :, -6:]), dim=2)
        assert (selected, recent) == (capacity, 6)
        torch.testing.assert_close(keys, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(values, expected_v, rtol=0, atol=0)


def test_short_prompt_is_kept_whole():
    compressor = PyramidKVCompressor(3, max_capacity_prompt=32, window_size=8)
    q, k = torch.randn(1, 4, 20, 16), torch.randn(1, 2, 20, 16)
    keys, values, selected, recent = compressor.compress(0, q, k, k)
    assert keys is k and (selected, recent) == (12, 8)


@pytest.mark.parametrize("granularity,stored_heads", [("kv_head", 2), ("query_head", 4)])
def test_prefill_drops_entries_and_decode_appends(granularity, stored_heads):
    model = tiny_model()
    compressor = PyramidKVCompressor(TINY.num_hidden_layers, max_capacity_prompt=16, window_size=4,
                                     kernel_size=3, beta=4)
    model.set_kv_cache(partial(CompressedKVCache, compressor=compressor, granularity=granularity))
    prompt = torch.randint(0, TINY.vocab_size, (1, 50))
    with torch.no_grad():
        model(prompt, input_pos=torch.arange(50))
        schedule = compressor.budget_schedule(50)
        assert [cache.length for cache in model.caches] == schedule
        for pos in range(50, 53):
            model(torch.tensor([[pos % TINY.vocab_size]]), input_pos=torch.tensor([pos]))
    counts = cache_compression_snapshot(model, 53)
    retained = sum(schedule) + 3 * TINY.num_hidden_layers
    assert counts["retained_slots"] == retained * stored_heads
    assert counts["dense_slots"] == 53 * TINY.num_hidden_layers * TINY.num_key_value_heads
    assert counts["layer_group_count"] == TINY.num_hidden_layers * TINY.num_key_value_heads
    assert counts["recent_slots"] == (4 + 3) * TINY.num_hidden_layers * stored_heads
    assert counts["hierarchical_slots"] == 0
    assert counts["kv_retention_ratio"] == pytest.approx(counts["retained_slots"] / counts["dense_slots"])
    assert all(cache._k.size(2) >= cache.length for cache in model.caches)
    with pytest.raises(ValueError, match="expected 52"):
        cache_compression_snapshot(model, 52)


def test_compression_changes_later_tokens_but_not_the_prefill_logits():
    model = tiny_model(seed=1)
    tokens = torch.randint(0, TINY.vocab_size, (1, 60))
    compressor = PyramidKVCompressor(TINY.num_hidden_layers, max_capacity_prompt=12, window_size=4, kernel_size=3)
    with torch.no_grad():
        dense = model(tokens)
        model.set_kv_cache(partial(CompressedKVCache, compressor=compressor))
        prefill = model(tokens[:, :50], input_pos=torch.arange(50))
        after = model(tokens[:, 50:], input_pos=torch.arange(50, 60))
    torch.testing.assert_close(prefill, dense[:, :50], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(after, dense[:, 50:], atol=1e-3)


def test_invalid_settings_fail_before_running():
    with pytest.raises(ValueError, match="window_size"):
        PyramidKVCompressor(28, max_capacity_prompt=8, window_size=8)
    with pytest.raises(ValueError, match="odd"):
        PyramidKVCompressor(28, kernel_size=4)
