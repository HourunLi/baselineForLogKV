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
    saved = [[t.clone() for t in c.storage] for c in cam.caches]
    cam.reset_cache()
    assert ptrs == [[t.data_ptr() for t in c.storage] for c in cam.caches]
    if interval == 1:
        replay = torch.cat([cam(ids[:, i:i+1], pos[i:i+1]) for i in range(128)], dim=1)
    else:
        replay = cam(ids, pos)
    torch.testing.assert_close(actual, replay, atol=2e-6, rtol=2e-5)
    for cache, expected in zip(cam.caches, saved):
        for tensor, reference in zip(cache.storage, expected):
            torch.testing.assert_close(tensor, reference, atol=2e-5, rtol=2e-5)


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


def test_full_platform_plan_is_independent_of_previous_job(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    for name, value in dict(CAM_PYTHON=sys.executable, NIAH_ONLY="1", EVAL_LENGTHS="32k",
                            EVAL_SUITES="ruler", LIMIT="2", LONG_EVAL_LIMIT="2").items():
        monkeypatch.setenv(name, value)
    plan = json.loads(subprocess.check_output(["bash", "majob.sh", "--all", "--print-config"], cwd=root))
    rows = plan["stages"]
    assert len({r["folder"] for r in rows}) == len(rows) == 20
    assert [r["budget"] for r in rows] == ["logkv"] * 14 + ["target-bytes"] * 6
    assert [r["limit"] for r in rows] == [None, None] + [100] * 18
    assert all(r["optional"] == [] for r in rows)
    assert [r["lengths"][0] for r in rows[-6:]] == list(TARGET)
    assert [len(r["benchmark"].split(",")) for r in rows[-6:]] == [13, 13, 13, 12, 12, 12]


def test_full_ppl_uses_reference_assets_by_length(monkeypatch, tmp_path):
    from models.cam.full_data import ppl_sources

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    for key in ("LONG_PPL_DATA", "LONG_PPL_BOOKS", "LONG_PPL_BOOKS_1M", "LONG_PPL_DATA_1M"):
        monkeypatch.delenv(key, raising=False)
    short = tmp_path / "long_ppl_books/emozilla--pg19"
    ultra = tmp_path / "long_ppl_books_1m/emozilla--pg19"
    short.mkdir(parents=True)
    ultra.mkdir(parents=True)
    assert ppl_sources(262144)["LONG_PPL_BOOKS"] == str(short)
    assert ppl_sources(524288)["LONG_PPL_BOOKS"] == str(ultra)
    monkeypatch.setenv("LONG_PPL_DATA", "/data/ref")
    monkeypatch.setenv("LONG_PPL_DATA_1M", "/data/ref-ultra")
    assert ppl_sources(262144)["LONG_PPL_DATA"] == "/data/ref"
    assert ppl_sources(1048576)["LONG_PPL_DATA"] == "/data/ref-ultra"


def test_niah_preserves_user_offline_cache_paths(monkeypatch, tmp_path):
    from models.cam.niah import setup_cache_environment
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("CAM_HF_CACHE", raising=False)
    monkeypatch.setenv("RULER_CACHE_DIR", str(tmp_path / "ruler"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets"))
    monkeypatch.setenv("NLTK_DATA", "/data/shared/nltk:/data/local/nltk")
    # Isolate all os.environ updates made by the launcher.
    import os
    from unittest.mock import patch
    with patch.dict(os.environ):
        setup_cache_environment()
        assert os.environ["HF_HOME"] == str(tmp_path)
        assert os.environ["HF_DATASETS_CACHE"] == str(tmp_path / "datasets")
        assert os.environ["RULER_CACHE_DIR"] == str(tmp_path / "ruler")
        assert os.environ["NLTK_DATA"] == "/data/shared/nltk:/data/local/nltk"
        assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_managed_cluster_bounds_workers_by_visible_gpus(tmp_path):
    import os
    root = Path(__file__).resolve().parents[1]
    # Only the launcher's standalone GPU-count probe sees this simulated CUDA topology.
    (tmp_path / "torch.py").write_text(
        "from types import SimpleNamespace\n"
        "cuda = SimpleNamespace(device_count=lambda: 2, init=lambda: None, is_bf16_supported=lambda **kw: True)\n")
    executable = tmp_path / "python"
    executable.write_text('#!/bin/bash\nif [[ "$1" == "-" ]]; then\n'
                          '  PYTHONPATH="$SIM_GPU_DIR" exec "$TEST_PYTHON" -\nfi\n'
                          'echo "$MA_NUM_GPUS $LITGPT_EXPECTED_WORLD_SIZE $*"\n')
    executable.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in
           ("GPUS_PER_NODE", "NPROC_PER_NODE", "LOCAL_WORLD_SIZE", "VC_TASK_INDEX", "MA_VJ_NAME", "NODE_RANK")}
    env.update(MA_NUM_GPUS="8", MA_NUM_HOSTS="1", CAM_PYTHON=str(executable),
               TEST_PYTHON=sys.executable, SIM_GPU_DIR=str(tmp_path))
    args = ["bash", "majob.sh", "--all"]
    assert "2 2 -m models.cam.niah --all" in subprocess.check_output(args, cwd=root, env=env, text=True)
    env["GPUS_PER_NODE"] = "3"
    run = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True)
    assert run.returncode != 0 and "CUDA visible=2" in run.stderr
