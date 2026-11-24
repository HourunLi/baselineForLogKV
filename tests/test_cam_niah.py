"""Exercise the reference length planner, real fixed storage and cache-relative Qwen3 path on CPU."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from models._vendor.niahlogkv.litgpt import Config, GPT
from models.cam.niah_budget import TARGET, cache_budget
from models.cam.niah_model import NiahCaM


def test_all_ruler_passes():
    cmd = [sys.executable, "-m", "models.cam.niah", "--no-main", "--suites", "ruler", "--print-config"]
    data = json.loads(subprocess.check_output(cmd, cwd=Path(__file__).resolve().parents[1]))
    assert data["weight_role"] == "gamma_cpt"
    for row, n in zip(data["stages"], TARGET, strict=True):
        assert row["lengths"] == [n] and row["config_overrides"]["block_size"] == n + 1024
        assert row["config_overrides"].get("rope_adjustments", {}).get("factor", 1) == n / 32768
        assert ("ruler_qa_squad" in row["benchmark"]) == (n <= 131072)
        assert len(row["benchmark"].split(",")) == (13 if n <= 131072 else 12)
        assert row["limit"] == 100


@pytest.mark.parametrize("length", list(TARGET))
def test_distinct_budget_denominators(length):
    args = (length, length + 1024, 28, 8, 128, 2, 2_031_878_912)
    match = cache_budget("logkv", *args)
    assert 0 <= args[-1] - match["persistent_cache_bytes"] < match["bytes_per_entry"]
    bytes_ = cache_budget("target-bytes", *args)
    slots = cache_budget("target-slots", *args)
    target = float(TARGET[length])
    assert target <= bytes_["nominal_allocated_compression"] < target * 1.001
    assert target <= slots["nominal_logical_compression"] < target * 1.001
    assert bytes_["capacity_tokens"] < slots["capacity_tokens"]


@pytest.mark.parametrize("interval", [1, 7])
def test_litgpt_yarn_dense_prefix_and_storage_reuse(interval):
    torch.manual_seed(3)
    cfg = Config(name="tiny-qwen3", block_size=256, n_layer=2, n_embd=32, n_head=4,
                 head_size=8, n_query_groups=2, vocab_size=32, padded_vocab_size=32,
                 norm_qk=True, parallel_residual=False, rotary_percentage=1, bias=False,
                 rope_adjustments=dict(factor=2, beta_fast=32, beta_slow=1, original_max_seq_len=128),
                 mlp_class_name="LLaMAMLP", intermediate_size=64)
    dense = GPT(cfg).eval()
    cam = NiahCaM(copy.deepcopy(dense), 48, interval).eval()
    ids = torch.randint(0, 32, (1, 128))
    pos = torch.arange(ids.size(1))
    ptrs = [[t.data_ptr() for t in c.storage] for c in cam.caches]
    with torch.no_grad():
        actual = cam(ids, pos)
        torch.testing.assert_close(actual[:, :48], dense(ids[:, :48]), atol=2e-6, rtol=2e-5)
    assert cam.snapshot(128)["retained_slots"] == 48 * 4
    assert cam.storage_bytes() == 48 * 4 * (2 * 8 * 4 + 4)
    cam.reset_cache()
    assert ptrs == [[t.data_ptr() for t in c.storage] for c in cam.caches]
    if interval == 1:
        replay = torch.cat([cam(ids[:, i:i+1], pos[i:i+1]) for i in range(128)], dim=1)
    else:
        replay = cam(ids, pos)
    torch.testing.assert_close(actual, replay, atol=2e-6, rtol=2e-5)


def test_reference_lm_checkpoint_score_and_generation(tmp_path, monkeypatch):
    import yaml
    from lm_eval.api.instance import Instance
    from tests.test_prepare_base import source
    from models._vendor.niahlogkv import eval as runtime
    from models.cam.niah_model import install_adapter

    path, *_ = source.__wrapped__(tmp_path)
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/niah/qwen1.7b.yaml").read_text())
    options = {key: value for key, value in config.items() if key.startswith("log_kv_")}
    options.update(log_kv_alpha_exact_tokens=16, log_kv_alpha_span_max_tokens=4,
                   log_kv_semantic_flush_granularity=8)
    native = runtime.LogKVLM
    monkeypatch.setattr(runtime, "LogKVLM", native)
    monkeypatch.setattr(runtime, "cache_compression_snapshot", runtime.cache_compression_snapshot)
    install_adapter(runtime, 32768, "logkv", 1, 8)
    lm = runtime.LogKVLM(str(path), device="cpu", config_overrides={"block_size": 64}, **options)
    assert lm.cache_budget["persistent_cache_bytes"] == lm.model.storage_bytes()
    score, _ = lm._score_tokens([2] * 40, [2] * 12)
    assert torch.isfinite(torch.tensor(score)) and lm.compression_metrics[-1]["processed_tokens"] == 52
    request = Instance("generate_until", {}, ("blue", {"until": [], "max_gen_toks": 2}), 0)
    assert len(lm.generate_until([request])) == 1


def test_tiled_merge_propagates_across_tile_boundary(monkeypatch):
    from models.cam.cache import CaMCache
    from tests.test_cam_long_context import sequential_evictions

    cache = CaMCache(2, 16, merge_size=16)
    keys = torch.randn(1, 2, 159, 8, generator=torch.Generator().manual_seed(1))
    cache.allocate(keys, 2, 8)
    cache.append(keys, keys)
    selected = (torch.arange(141) % 5 != 0).float().expand(1, 2, -1)
    expected = sequential_evictions(cache, selected)
    monkeypatch.setattr(torch, "bernoulli", lambda probability, generator=None: selected)
    cache.update(torch.zeros_like(cache.scores))
    for actual, wanted in zip((cache.keys, cache.values, cache.scores), expected):
        torch.testing.assert_close(actual, wanted, atol=2e-6, rtol=2e-5)
