import torch

from lla_bench.metrics import allocated_byte_ratios
from lla_bench.recurrence import MatrixCache, weak_attention
from lla_bench.report import quality_rows, summary_rows


def test_shared_projections_match_expanded_heads_through_carries():
    torch.manual_seed(14)
    for groups in (1, 2):
        length, heads = 35, 4
        q, k = (torch.randn(1, groups, length, 5) for _ in range(2))
        v = torch.randn(1, heads, length, 3)
        g = -torch.rand(1, heads, length)
        lam = torch.rand(1, heads, length, length.bit_length() + 1)
        expected = weak_attention(q.repeat_interleave(heads // groups, 1),
                                  k.repeat_interleave(heads // groups, 1), v, g, lam)
        cache = MatrixCache(1, heads, 5, 3, length, "cpu")
        with torch.no_grad():
            blocks = []
            for span in (slice(0, 9), slice(9, 32), slice(32, length)):
                blocks.append(weak_attention(q[:, :, span], k[:, :, span], v[:, :, span],
                                             g[:, :, span], lam[:, :, span], cache, tile=8))
        torch.testing.assert_close(torch.cat(blocks, dim=2), expected, atol=2e-5, rtol=1e-5)


def test_report_includes_measured_long_scores_and_groups_once():
    scores = {"ppl,none": 12.5, "bits_per_byte,none": 0.7, "ppl_stderr,none": 0.2}
    group = {"alias": "BABILong 1M", "acc,none": 0.25}
    result = {"results": {"long_ppl_1m": scores, "babilong_1m": group,
                          "niah_single_1": {"1048576,none": 0.0}}, "groups": {"babilong_1m": group}}
    rows = list(quality_rows(result))
    assert [(row["section"], row["metric"], row["value"]) for row in rows] == [
        ("task", "ppl", 12.5), ("task", "bits_per_byte", 0.7), ("task", "1048576", 0.0),
        ("group", "acc", 0.25)]
    assert rows[0]["stderr"] == 0.2


def test_common_allocated_ratio_is_token_weighted_for_both_formats():
    token_bytes = 28 * 8 * 128 * 2 * 2
    budget = {"persistent_cache_bytes": 10 * token_bytes}
    compression = {"request_count": 2, "processed_tokens": 15 + 65}
    ratios = allocated_byte_ratios(budget, compression)
    assert ratios["allocated_byte_ratio"] == 0.25
    assert ratios["allocated_byte_compression_factor"] == 4
    assert allocated_byte_ratios({}, {"request_count": 0, "processed_tokens": 0})["allocated_byte_ratio"] is None
    student_bytes = 80 * 36 * 8 * 128 * 2 * 2
    assert allocated_byte_ratios(budget, dict(compression, dense_kv_bytes=student_bytes))["dense_reference_bytes"] == student_bytes
    hashes = {"doc_id": 0, "filter": "none", "doc_hash": "doc", "prompt_hash": "prompt", "target_hash": "target"}
    run = {"results": {"results": {}, "samples": {"demo": [hashes]}}, "cache_budget": budget,
           "kv_compression": {"summary": compression}, "inference_metrics": {"generation": {}, "scoring": {}}}
    digest = next(row for row in summary_rows(run) if row["metric"] == "text_set_sha256")
    hashes["prompt_hash"] = "different"
    assert next(row for row in summary_rows(run) if row["metric"] == "text_set_sha256") != digest
