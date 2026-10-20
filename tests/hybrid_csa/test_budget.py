import pytest

from hybrid_csa.budget import theoretical_budget
from hybrid_csa.settings import HybridCSASettings


def test_32k_default_budget_counts_local_history_and_separate_index():
    b = theoretical_budget(32768)
    assert b["full_layers_zero_based"] == list(range(0, 28, 3))
    assert b["full_layer_count"] == 10 and b["csa_layer_count"] == 18
    assert b["kv_retention_ratio"] == pytest.approx((10 * 32768 + 18 * (8192 + 128)) / (28 * 32768))
    assert b["kv_compression_factor"] == pytest.approx(1.92171581769437)
    assert b["kv_plus_index_retention_ratio"] == pytest.approx(0.525390625)
    assert b["all_state_retention_ratio"] == pytest.approx(0.5255475725446429)
    assert b["index_key_scalars"] == 18 * 8192 * 64
    assert b["compressor_state_scalars"] == 18 * 4 * (2048 + 2 * 8 * 128)
    smaller_read_budget = theoretical_budget(32768, HybridCSASettings(index_topk=16))
    assert smaller_read_budget["main_kv_scalars"] == b["main_kv_scalars"]
    assert smaller_read_budget["index_key_scalars"] == b["index_key_scalars"]


@pytest.mark.parametrize("tokens", [0, 1, 3, 4, 7, 8, 127, 128, 129, 32767, 32768])
def test_partial_blocks_and_short_requests(tokens):
    b = theoretical_budget(tokens)
    assert b["compressed_slots_per_csa_layer"] == tokens // 4
    assert b["local_slots_per_csa_layer"] == min(tokens, 128)
    assert 0 <= b["compressor_tail_tokens_per_csa_layer"] <= 7
    assert b["retained_slots"] == b["exact_slots"] + b["recent_slots"] + b["hierarchical_slots"]
    assert b["kv_retention_ratio"] is None if tokens == 0 else b["kv_retention_ratio"] > 0


def test_aggregate_uses_sum_of_counts_not_mean_of_ratios():
    from hybrid_csa.semanticlogkv_eval import summarize_compression
    rows = [theoretical_budget(n) for n in (3, 32768)]
    summary = summarize_compression(rows)
    assert summary["kv_retention_ratio"] == pytest.approx(sum(r["retained_slots"] for r in rows) / sum(r["dense_slots"] for r in rows))
    assert summary["all_state_retention_ratio"] == pytest.approx(
        sum(r["main_kv_scalars"] + r["index_key_scalars"] + r["compressor_state_scalars"] for r in rows)
        / sum(r["dense_kv_scalars"] for r in rows))
