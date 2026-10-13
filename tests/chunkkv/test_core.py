"""ChunkKV selection against a literal, loop-based reading of Algorithm 1."""

import math

import pytest
import torch

from chunkkv.core import ChunkKVSettings, keep_indices, observation_scores, select_chunks


def reference_selection(q, k, scale, settings):
    """Explicit softmax per head and window query, then chunk sums and a stable top-k."""
    _, n_head, length, _ = q.shape
    per_group = n_head // k.size(1)
    window = min(settings.window_size, length)
    prefix = length - window
    scores = [0.0] * prefix
    for head in range(n_head):
        keys = k[0, head // per_group].double()
        for query_pos in range(prefix, length):
            logits = keys[: query_pos + 1] @ q[0, head, query_pos].double() * scale
            weights = torch.softmax(logits, dim=0)
            for j in range(prefix):
                scores[j] += weights[j].item()
    chunks = [(start, sum(scores[start:start + settings.chunk_size]))
              for start in range(0, prefix, settings.chunk_size)]
    n_chunks = (settings.prompt_budget(length) - window) // settings.chunk_size
    best = sorted(chunks, key=lambda item: -item[1])[:n_chunks]  # Python sort is stable
    kept = [j for start, _ in sorted(best) for j in range(start, min(start + settings.chunk_size, prefix))]
    return kept + list(range(prefix, length)), torch.tensor(scores)


@pytest.mark.parametrize("length,window,chunk,budget", [(37, 5, 4, 17), (40, 8, 10, 31), (23, 3, 1, 9)])
def test_selection_matches_algorithm_1(length, window, chunk, budget):
    torch.manual_seed(length)
    q, k = torch.randn(1, 4, length, 8), torch.randn(1, 2, length, 8)
    settings = ChunkKVSettings(chunk_size=chunk, window_size=window, budget=budget)
    scale = 8 ** -0.5
    expected, expected_scores = reference_selection(q, k, scale, settings)
    scores = observation_scores(q, k, window, scale)
    assert torch.allclose(scores[0].double(), expected_scores.double(), atol=1e-5)
    kept, chunk_tokens = keep_indices(scores[0], length, settings)
    assert kept.tolist() == expected
    assert chunk_tokens == len(expected) - window
    assert window < len(expected) <= budget


def test_window_is_kept_and_partial_last_chunk_is_clipped():
    scores = torch.zeros(23)
    scores[20:] = 5.0  # the partial chunk [20, 23) wins
    settings = ChunkKVSettings(chunk_size=10, window_size=4, budget=14)
    kept, chunk_tokens = keep_indices(scores, 27, settings)
    assert kept.tolist() == [20, 21, 22, 23, 24, 25, 26] and chunk_tokens == 3


def test_equal_scores_keep_earliest_chunks_and_ratio_budget():
    assert select_chunks(torch.ones(12), 3, 2).tolist() == [0, 1, 2, 3, 4, 5]
    settings = ChunkKVSettings(chunk_size=2, window_size=2, keep_ratio=0.25)
    assert settings.prompt_budget(40) == 10 and settings.compresses(40)
    assert not settings.compresses(1)
    kept, _ = keep_indices(torch.ones(38), 40, settings)
    assert len(kept) == 10 and kept[-2:].tolist() == [38, 39]


def test_short_prompts_and_observation_windows_covering_everything():
    settings = ChunkKVSettings(chunk_size=2, window_size=6, keep_ratio=0.5)
    q, k = torch.randn(1, 2, 5, 4), torch.randn(1, 1, 5, 4)
    scores = observation_scores(q, k, settings.window_size, 0.5)
    assert scores.shape == (1, 0)
    kept, chunk_tokens = keep_indices(scores[0], 5, settings)
    assert kept.tolist() == list(range(5)) and chunk_tokens == 0


def test_settings_validation_and_reuse_schedule():
    with pytest.raises(ValueError, match="exactly one"):
        ChunkKVSettings(budget=100, keep_ratio=0.1)
    with pytest.raises(ValueError, match="exactly one"):
        ChunkKVSettings()
    with pytest.raises(ValueError, match="window"):
        ChunkKVSettings(budget=70, window_size=64, chunk_size=10)
    with pytest.raises(ValueError, match="keep_ratio"):
        ChunkKVSettings(keep_ratio=0)
    settings = ChunkKVSettings(budget=128, reuse_layers=3)
    assert [layer for layer in range(8) if settings.scores_layer(layer)] == [0, 3, 6]
    assert settings.as_dict() == dict(chunk_size=10, window_size=64, budget=128, keep_ratio=None, reuse_layers=3)
    assert math.ceil(0.1 * 32768) == ChunkKVSettings(keep_ratio=0.1).prompt_budget(32768)
