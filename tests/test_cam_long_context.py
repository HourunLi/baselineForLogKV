"""64K / 128K: chunked CaM merge steps, the fixed cache budget, the long suites and the launcher."""

from argparse import Namespace
import copy
import json
import os
import subprocess

import pytest
import torch
import yaml

from models.cam import cache as cam_cache
from models.cam.cache import CaMCache
from models.cam.evaluate import METHOD_FIELDS, ROOT, cache_entries, resolve_run
from models.cam.model import CaMModel
from tests.test_cam_model import tiny_model


def sequential_evictions(cache, selected):
    """The official online rule one victim at a time (scores are not transferred, so each step's targets are the
    next merge_size entries of the original order), with the given Bernoulli outcomes."""
    keys, values, scores = cache.keys.clone(), cache.values.clone(), cache.scores.clone()
    start, merge = cache.start_size, cache.merge_size
    for k in range(selected.shape[-1]):
        contribution = values[:, :, start] * selected[..., k, None] / merge
        values[:, :, start + 1:start + 1 + merge] += contribution.unsqueeze(2)
        keep = [i for i in range(keys.shape[2]) if i != start]
        keys, values, scores = keys[:, :, keep], values[:, :, keep], scores[..., keep]
    return keys, values, scores


def test_one_step_of_many_victims_equals_evicting_them_one_by_one(monkeypatch):
    torch.manual_seed(3)
    cache = CaMCache(start_size=2, recent_size=6, merge_size=3, seed=1)
    cache.append(torch.randn(1, 2, 13, 4), torch.randn(1, 2, 13, 4))
    cache.scores = torch.rand(1, 2, 13)
    selected = torch.tensor([[[1.0, 1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0, 0.0]]])
    expected = sequential_evictions(cache, selected)
    monkeypatch.setattr(cam_cache.torch, "bernoulli", lambda probability, generator=None: selected)
    cache.update(torch.zeros(1, 2, 13))  # five entries over the budget of eight
    for actual, wanted in zip((cache.keys, cache.values, cache.scores), expected):
        torch.testing.assert_close(actual, wanted, atol=1e-6, rtol=1e-6)
    assert cache.keys.shape[2] == 8


def test_chunked_merge_steps_hold_the_budget_and_equal_dense_while_nothing_is_evicted():
    hf = tiny_model()
    tokens = torch.randint(0, 32, (1, 30))
    with torch.no_grad():
        expected = hf(tokens, use_cache=False).logits
    roomy = CaMModel(copy.deepcopy(hf), 64, start_size=2, recent_size=40, prefill_chunk_size=8, merge_interval=8)
    roomy.reset_kv_cache()
    torch.testing.assert_close(roomy(tokens, input_pos=torch.arange(30)), expected, atol=2e-6, rtol=2e-5)
    tight = CaMModel(copy.deepcopy(hf), 64, start_size=2, recent_size=6, merge_size=3, prefill_chunk_size=8,
                     merge_interval=8)
    tight.reset_kv_cache()
    for start in range(0, 30, 8):
        tight(tokens[:, start:start + 8], input_pos=torch.arange(start, min(start + 8, 30)))
        assert all(layer.keys.shape[2] <= 8 for layer in tight.layers)  # never more than the budget
    assert all(layer.processed_tokens == 30 for layer in tight.layers)


def test_cam_may_run_past_the_trained_window_but_full_kv_may_not():
    model = CaMModel(tiny_model(), 1000, start_size=2, recent_size=40, prefill_chunk_size=8, merge_interval=8)
    model.reset_kv_cache()
    model(torch.randint(0, 32, (1, 100)), input_pos=torch.arange(100))  # positions are cache slots, at most 50
    with pytest.raises(ValueError, match="position window"):
        CaMModel(tiny_model(), 1000, start_size=2, recent_size=40, dense=True)
    with pytest.raises(ValueError, match="merge step"):
        CaMModel(tiny_model(), 1000, start_size=2, recent_size=40, merge_interval=30)


@pytest.fixture
def qwen3_17b_base(tmp_path):
    """The prepared Base's config only: budgets are sized from the architecture."""
    directory = tmp_path / "semanticlogkv_base"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(dict(
        num_hidden_layers=28, num_key_value_heads=8, num_attention_heads=16, hidden_size=2048, head_dim=128)))
    return directory


def args(**overrides):
    values = dict(model=None, backend="cam", suite="long", tasks=None, output=None, limit=None)
    values.update(dict.fromkeys(METHOD_FIELDS))
    return Namespace(**(values | overrides))


@pytest.mark.parametrize("context, window, long_tasks, capacity", [
    ("64k", 66560, "babilong_64k,long_ppl_32k,long_ppl_64k", 19173),
    ("128k", 132096, "babilong_128k,long_ppl_128k", 20794),
])
def test_long_configs_fix_window_tasks_samples_and_the_logkv_budget(qwen3_17b_base, tmp_path, context, window,
                                                                     long_tasks, capacity, monkeypatch):
    monkeypatch.delenv("LONG_PPL_DATA", raising=False)
    config = ROOT / f"configs/cam_qwen3_1.7b_{context}.yaml"
    run = args(config=config, model=qwen3_17b_base, output=tmp_path)
    experiment, model, output, tasks, lengths, options, ctx, long_context, budget_bytes = resolve_run(run)
    assert ctx == context and tasks == long_tasks and lengths == [window - 1024] and run.limit == 100
    assert long_context["max_seq_length"] == window and output == tmp_path / "long"
    # raw K + V in bf16 and an fp32 score per KV head and layer fit the LogKV cache's allocated bytes
    assert options["start_size"] + options["recent_size"] == capacity == cache_entries(qwen3_17b_base, budget_bytes)
    assert 28 * 8 * capacity * (2 * 128 * 2 + 4) <= yaml.safe_load(config.read_text())["budget_bytes"]
    assert options["merge_interval"] == options["prefill_chunk_size"] == 2048
    assert os.environ["LONG_PPL_DATA"].endswith("c5e39bf32e33f9111323aa68d7d9000d22722035")
    ruler = resolve_run(args(config=config, model=qwen3_17b_base, suite="ruler"))
    assert len(ruler[3].split(",")) == 13
    with pytest.raises(ValueError, match="own eval_long configs"):
        resolve_run(args(config=config, model=qwen3_17b_base, backend="dense"))
    with pytest.raises(ValueError, match="runs at 32k"):
        resolve_run(args(config=config, model=qwen3_17b_base, suite="main"))


def test_long_task_files_are_verbatim_copies():
    import hashlib

    source = json.loads((ROOT / "tasks" / "SOURCE.json").read_text())
    for path, digest in source["sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path


def test_platform_launcher_runs_the_long_stages_of_a_long_config(tmp_path):
    executable = tmp_path / "python"
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CAM_TEST_CALLS"\n')
    executable.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("MA_", "MASTER_", "EVAL_MASTER_", "NIAH_MASTER_"))}
    env.update(CAM_PYTHON=str(executable), CAM_TEST_CALLS=str(tmp_path / "calls"), MA_NUM_GPUS="4")
    command = ["bash", str(ROOT / "majob.sh"), "configs/cam_qwen3_1.7b_64k.yaml", "--limit", "4"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert [call.split("--suite ")[1].split()[0] for call in calls] == ["long", "ruler"]


def test_long_tasks_reach_lm_eval_through_a_task_manager(monkeypatch):
    pytest.importorskip("lm_eval")
    from types import SimpleNamespace

    import lm_eval.tasks

    from models.cam.evaluate import LONG_TASKS_DIR
    from models.qwen3.evaluate import install_long_tasks

    seen = {}
    monkeypatch.setattr(lm_eval.tasks, "TaskManager", lambda include_path, metadata: (include_path, metadata))
    harness = SimpleNamespace(simple_evaluate=lambda **kwargs: seen.update(kwargs), other=1)
    evaluator = SimpleNamespace(evaluator=harness)
    install_long_tasks(evaluator, LONG_TASKS_DIR)
    evaluator.evaluator.simple_evaluate(tasks=["long_ppl_64k"], metadata={"pretrained": "base"})
    assert seen["task_manager"] == (str(LONG_TASKS_DIR), {"pretrained": "base"})
    assert evaluator.evaluator.other == 1
