"""H2O selection equals KVCache-Factory's H2OKVCluster; decode eviction holds the budget."""

import math
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from h2okv.compressor import H2OCompressor, accumulated_attention, attention_with_scores
from h2okv.kv_cache import CompressedKVCache, causal_attention
from h2okv.metrics import cache_compression_snapshot
from h2okv.qwen3 import repeat_kv
from tests.helpers import TINY, tiny_model


# ---- Reference: KVCache-Factory pyramidkv/pyramidkv_utils.py H2OKVCluster, logic kept verbatim ----
def reference_h2o(query_states, key_states, value_states, budget, window, agg="mean"):
    bsz, num_heads, q_len, head_dim = query_states.shape
    groups = num_heads // key_states.shape[1]
    if q_len < budget:
        return key_states, value_states
    key_rep = repeat_kv(key_states, groups)
    attn = torch.matmul(query_states, key_rep.transpose(2, 3)) / math.sqrt(head_dim)
    mask = torch.full((q_len, q_len), torch.finfo(attn.dtype).min, device=attn.device)
    cond = torch.arange(mask.size(-1))
    mask.masked_fill_(cond < (cond + 1).view(mask.size(-1), 1), 0)
    attn += mask[None, None, :, :]
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
    summed = attn[:, :, :, :-window].sum(dim=-2)
    if groups > 1:
        grouped = summed.reshape(bsz, num_heads // groups, groups, -1)
        summed = {"mean": grouped.mean(2), "max": grouped.amax(2), "sum": grouped.sum(2)}[agg]
    indices = summed.topk(budget - window, dim=-1).indices
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    k_past = key_states[:, :, :-window, :].gather(dim=2, index=indices)
    v_past = value_states[:, :, :-window, :].gather(dim=2, index=indices)
    return (torch.cat([k_past, key_states[:, :, -window:, :]], dim=2),
            torch.cat([v_past, value_states[:, :, -window:, :]], dim=2), indices[..., 0])
# ---------------------------------------------------------------------------------------------------


def full_matrix_scores(q, k, agg):
    groups = q.size(1) // k.size(1)
    L = q.size(2)
    attn = torch.matmul(q, repeat_kv(k, groups).transpose(2, 3)) / math.sqrt(q.size(-1))
    attn = attn.masked_fill(torch.ones(L, L, dtype=torch.bool).triu(1), float("-inf")).softmax(-1).sum(-2)
    grouped = attn.reshape(attn.size(0), k.size(1), groups, L)
    return {"mean": grouped.mean(2), "max": grouped.amax(2), "sum": grouped.sum(2)}[agg]


@pytest.mark.parametrize("chunk", [1, 7, 32, 1000])
@pytest.mark.parametrize("agg", ["mean", "max", "sum"])
def test_chunked_scores_equal_the_full_attention_matrix(chunk, agg):
    torch.manual_seed(0)
    q, k = torch.randn(1, 4, 50, 16), torch.randn(1, 2, 50, 16)
    torch.testing.assert_close(accumulated_attention(q, k, agg, chunk), full_matrix_scores(q, k, agg),
                               atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("granularity", ["kv_head", "query_head"])
@pytest.mark.parametrize("agg", ["mean", "max"])
def test_selected_entries_equal_kvcache_factory(granularity, agg):
    torch.manual_seed(1)
    q = torch.randn(1, 4, 80, 16)
    k, v = torch.randn(1, 2, 80, 16), torch.randn(1, 2, 80, 16)
    if granularity == "query_head":
        k, v = repeat_kv(k, 2), repeat_kv(v, 2)
    keys, values, selected, recent, scores = H2OCompressor(24, 6, agg, score_chunk_size=16).compress(0, q, k, v)
    _, _, indices = reference_h2o(q, k, v, 24, 6, agg)
    gather = indices.sort(dim=-1).values.unsqueeze(-1).expand(-1, -1, -1, 16)
    expected_k = torch.cat((k[:, :, :-6].gather(2, gather), k[:, :, -6:]), dim=2)
    expected_v = torch.cat((v[:, :, :-6].gather(2, gather), v[:, :, -6:]), dim=2)
    assert (selected, recent, scores) == (18, 6, None)
    torch.testing.assert_close(keys, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(values, expected_v, rtol=0, atol=0)


def test_short_prompt_is_kept_whole():
    q, k = torch.randn(1, 4, 20, 16), torch.randn(1, 2, 20, 16)
    keys, _, selected, recent, scores = H2OCompressor(32, 8).compress(0, q, k, k)
    assert keys is k and (selected, recent, scores) == (12, 8, None)


def test_explicit_attention_matches_sdpa_and_returns_received_attention():
    torch.manual_seed(2)
    q, k, v = torch.randn(1, 4, 3, 16), torch.randn(1, 2, 10, 16), torch.randn(1, 2, 10, 16)
    output, received = attention_with_scores(q, k, v, "sum")
    torch.testing.assert_close(output, causal_attention(q, k, v), atol=1e-5, rtol=1e-5)
    assert received.shape == (1, 2, 10)
    # Each of the 3 queries in each of the 2 grouped heads distributes probability mass 1.
    torch.testing.assert_close(received.sum(-1), torch.full((1, 2), 6.0), atol=1e-5, rtol=1e-5)
    assert torch.all(received[..., -1] > 0) and torch.all(received[..., -2] > 0)


@pytest.mark.parametrize("granularity,stored_heads", [("kv_head", 2), ("query_head", 4)])
def test_prefill_drops_entries_and_decode_appends(granularity, stored_heads):
    model = tiny_model()
    compressor = H2OCompressor(max_capacity_prompt=16, window_size=4)
    model.set_kv_cache(partial(CompressedKVCache, compressor=compressor, granularity=granularity))
    with torch.no_grad():
        model(torch.randint(0, TINY.vocab_size, (1, 50)), input_pos=torch.arange(50))
        assert [cache.length for cache in model.caches] == [16] * TINY.num_hidden_layers
        for pos in range(50, 53):
            model(torch.tensor([[pos % TINY.vocab_size]]), input_pos=torch.tensor([pos]))
    counts = cache_compression_snapshot(model, 53)
    assert counts["retained_slots"] == (16 + 3) * TINY.num_hidden_layers * stored_heads
    assert counts["recent_slots"] == (4 + 3) * TINY.num_hidden_layers * stored_heads
    assert counts["dense_slots"] == 53 * TINY.num_hidden_layers * TINY.num_key_value_heads


def test_decode_eviction_holds_the_budget():
    torch.manual_seed(3)
    compressor = H2OCompressor(max_capacity_prompt=12, window_size=4, decode_evict=True)
    cache = CompressedKVCache(0, compressor=compressor)
    cache(torch.randn(1, 4, 30, 16), torch.randn(1, 2, 30, 16), torch.randn(1, 2, 30, 16))
    assert cache.length == 12 and cache.scores.shape == (1, 2, 12)
    buffer = cache._k
    for step in range(5):
        output = cache(torch.randn(1, 4, 1, 16), torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
        assert output.shape == (1, 4, 1, 16)
        assert (cache.length, cache.num_recent, cache.num_selected, cache.evicted) == (12, 4, 8, step + 1)
    assert cache._k is buffer  # eviction rewrites the existing buffer
    snapshot = cache.snapshot()
    assert snapshot["retained_slots"] == 12 * 2 and snapshot["processed_tokens"] == 35


def test_short_prompt_grows_to_the_budget_before_evicting():
    compressor = H2OCompressor(max_capacity_prompt=12, window_size=4, decode_evict=True)
    cache = CompressedKVCache(0, compressor=compressor)
    cache(torch.randn(1, 4, 10, 16), torch.randn(1, 2, 10, 16), torch.randn(1, 2, 10, 16))
    lengths = [cache.length]
    for _ in range(4):
        cache(torch.randn(1, 4, 1, 16), torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
        lengths.append(cache.length)
    assert lengths == [10, 11, 12, 12, 12] and cache.evicted == 2


def test_decode_eviction_matches_original_h2o_bookkeeping():
    """Scores accumulate each step's attention; the minimum among non-recent entries is evicted."""
    torch.manual_seed(4)
    compressor = H2OCompressor(max_capacity_prompt=6, window_size=2, decode_evict=True, gqa_score_agg="sum")
    cache = CompressedKVCache(0, compressor=compressor)
    q, k, v = torch.randn(1, 2, 6, 8), torch.randn(1, 1, 6, 8), torch.randn(1, 1, 6, 8)
    cache(q, k, v)  # exactly at budget: nothing dropped yet
    torch.testing.assert_close(cache.scores, accumulated_attention(q, k, "sum"))
    keys, scores = cache.k.clone(), cache.scores.clone()
    nq, nk, nv = torch.randn(1, 2, 1, 8), torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)
    values = cache.v.clone()
    cache(nq, nk, nv)
    _, received = attention_with_scores(nq, torch.cat((keys, nk), 2), torch.cat((values, nv), 2), "sum")
    total = torch.cat((scores, scores.new_zeros(1, 1, 1)), -1) + received
    drop = int(total[0, 0, :5].argmin())
    expected = [i for i in range(7) if i != drop]
    torch.testing.assert_close(cache.k, torch.cat((keys, nk), 2)[:, :, expected])
    torch.testing.assert_close(cache.scores, total[:, :, expected])


def test_invalid_settings_fail_before_running():
    with pytest.raises(ValueError, match="window_size"):
        H2OCompressor(max_capacity_prompt=8, window_size=8)
    with pytest.raises(ValueError, match="gqa_score_agg"):
        H2OCompressor(gqa_score_agg="median")
