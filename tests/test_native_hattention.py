import torch

from lla_bench.metrics import cache_compression_snapshot, summarize_compression
from lla_bench.model import HAttentionLM, HConfig
from lla_bench.recurrence import MatrixCache, weak_attention


def test_gated_recurrence_matches_pairwise_definition_across_chunk_boundaries():
    torch.manual_seed(17)
    t, h = 73, 2
    q, k = (torch.randn(1, h, t, 5) for _ in range(2))
    v = torch.randn(1, h, t, 3)
    decay = -torch.rand(1, h, t)
    weights = torch.rand(1, h, t, t.bit_length() + 1)
    # Direct causal reference, independent of tile partitioning or binary carries.
    expected = torch.zeros_like(v)
    for i in range(t):
        for j in range(i + 1):
            alpha = decay[:, :, j+1:i+1].sum(-1).exp()
            similarity = (q[:, :, i] * k[:, :, j]).sum(-1)
            expected[:, :, i] += ((alpha * similarity * weights[:, :, i, (i ^ j).bit_length()])
                                   .unsqueeze(-1) * v[:, :, j])
    cache = MatrixCache(1, h, 5, 3, t, "cpu")
    results = []
    with torch.no_grad():
        for a, b in zip((0, 7, 8, 26, 65), (7, 8, 26, 65, t)):
            results.append(weak_attention(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], decay[:, :, a:b],
                                          weights[:, :, a:b], cache, tile=16))
    torch.testing.assert_close(torch.cat(results, 2), expected, atol=2e-5, rtol=1e-5)


def test_native_model_streaming_and_legacy_level_extension():
    cfg = HConfig(16, 4, 2, 32, 2, 3, 8, 2, 4, 1e-5, False, True, trained_levels=4, block_size=66)
    torch.manual_seed(5)
    model = HAttentionLM(cfg).eval()
    tokens = torch.randint(32, (1, 66))
    with torch.no_grad():
        full = model(tokens)
        model.set_cache()
        chunks = [model(tokens[:, a:b], torch.arange(a, b)) for a, b in [(0, 1), (1, 18), (18, 63), (63, 66)]]
        torch.testing.assert_close(torch.cat(chunks, 1), full, rtol=1e-4, atol=2e-5)
        row = cache_compression_snapshot(model, 66)
        assert row["dense_kv_bytes"] == 66 * 28 * 8 * 128 * 4
        assert row["live_matrices"] == 2 * 4 * (66).bit_count()
        assert row["kv_retention_ratio"] is None
        assert row["live_state_bytes"] == 2 * (2 * 4 * 3 * 8 * 4 + (32 + 12) * 4 * 4)
        model.reset_cache()
        torch.testing.assert_close(model(tokens[:, :1], torch.tensor([0])), full[:, :1], atol=2e-5, rtol=1e-4)
        second = cache_compression_snapshot(model, 1)
        summary = summarize_compression([row, second])
        assert summary["live_state_byte_ratio"] == ((row["live_state_bytes"] + second["live_state_bytes"])
                                                    / (row["dense_kv_bytes"] + second["dense_kv_bytes"]))
