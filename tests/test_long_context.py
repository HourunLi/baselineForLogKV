"""64K / 128K: YaRN, chunked cached forwards, streaming PyramidKV at fixed layer budgets, the long suites and launcher."""

import argparse
import subprocess

import pytest
import torch
import yaml

from pyramidkv import evaluate, method
from pyramidkv.compressor import PyramidKVCompressor
from pyramidkv.kv_cache import CompressedKVCache
from pyramidkv.qwen3 import Qwen3Config, build_rope_cache
from tests.helpers import TINY, litgpt_model_config, tiny_model
from tests.test_evaluate import REPO_ROOT, platform_launcher_env

QWEN3_17B = Qwen3Config(vocab_size=151936, hidden_size=2048, intermediate_size=6144, num_hidden_layers=28,
                        num_attention_heads=16, num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6,
                        rope_theta=1000000.0, max_position_embeddings=40960)


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_yarn_tables_match_transformers(factor):
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    hf = transformers.Qwen3Config(head_dim=128, hidden_size=2048, num_attention_heads=16, rope_theta=1e6,
                                  max_position_embeddings=32768,
                                  rope_scaling=dict(rope_type="yarn", factor=factor, beta_fast=32, beta_slow=1,
                                                    original_max_position_embeddings=32768))
    inv_freq, attention_factor = ROPE_INIT_FUNCTIONS["yarn"](hf, "cpu")
    angles = torch.outer(torch.arange(int(32768 * factor) + 1024).float(), inv_freq).repeat(1, 2)
    cos, sin = build_rope_cache(angles.size(0), 128, 1e6,
                                rope_scaling=dict(factor=factor, beta_fast=32, beta_slow=1, original_max_seq_len=32768))
    assert torch.allclose(cos, angles.cos() * attention_factor, atol=1e-4)
    assert torch.allclose(sin, angles.sin() * attention_factor, atol=1e-4)


def test_long_windows_need_yarn():
    with pytest.raises(ValueError, match="rope_scaling"):
        tiny_model(max_seq_length=TINY.max_position_embeddings + 1)
    model = tiny_model()
    model.rope_scaling = dict(factor=2.0, beta_fast=32, beta_slow=1, original_max_seq_len=128)
    model.max_seq_length = 2 * TINY.max_position_embeddings  # YaRN stretches the window by its factor
    assert model.cos.size(0) == 2 * TINY.max_position_embeddings


def run(model, ids, factory, chunk=None, decode=0):
    """Prefill ``ids`` (chunked when ``chunk``) and teacher-force ``decode`` more tokens one at a time."""
    model.prefill_chunk_size = chunk
    model.set_kv_cache(factory)
    prompt = ids.size(1) - decode
    with torch.no_grad():
        logits = [model(ids[:, :prompt], input_pos=torch.arange(prompt))]
        for i in range(prompt, ids.size(1)):
            logits.append(model(ids[:, i:i + 1], input_pos=torch.tensor([i])))
    return torch.cat(logits, dim=1)


def full(layer):
    return CompressedKVCache(layer)


def streaming(budget, window, layers=TINY.num_hidden_layers, **options):
    compressor = PyramidKVCompressor(num_hidden_layers=layers, max_capacity_prompt=budget, window_size=window,
                                     kernel_size=3, streaming=True, **options)
    return compressor, (lambda layer: CompressedKVCache(layer, compressor=compressor))


def test_chunked_forward_equals_one_forward():
    model = tiny_model()
    ids = torch.randint(0, TINY.vocab_size, (1, 45))
    assert torch.allclose(run(model, ids, full, chunk=8, decode=5), run(model, ids, full, decode=5), atol=1e-5)
    model.prefill_chunk_size = 8
    model.set_kv_cache(full)
    with torch.no_grad():  # logits only from lm_head_start, across chunk boundaries
        sliced = model(ids, input_pos=torch.arange(45), lm_head_start=21)
    assert sliced.shape[1] == 24


def test_streaming_pyramidkv_with_room_for_everything_is_full_attention():
    model = tiny_model()
    ids = torch.randint(0, TINY.vocab_size, (1, 50))
    compressor, factory = streaming(64, 8, beta=1)  # beta 1: a flat pyramid, 64 entries in every layer
    assert compressor.stream_schedule() == [64] * TINY.num_hidden_layers
    assert torch.allclose(run(model, ids, factory, chunk=8, decode=6), run(model, ids, full, decode=6), atol=1e-5)
    assert not any(cache.compressed for cache in model.caches)


def test_streaming_pyramidkv_keeps_each_layer_within_its_budget():
    model = tiny_model()
    ids = torch.randint(0, TINY.vocab_size, (1, 70))
    compressor, factory = streaming(16, 4)
    schedule = compressor.stream_schedule()
    assert schedule[0] > schedule[-1] and sum(schedule) <= 16 * TINY.num_hidden_layers + TINY.num_hidden_layers
    model.prefill_chunk_size = 8
    model.set_kv_cache(factory)
    with torch.no_grad():
        for start in range(0, 64, 8):  # one forward per chunk, as the chunked model does
            model(ids[:, start:start + 8], input_pos=torch.arange(start, start + 8), lm_head_start=7)
            assert all(cache.length <= budget for cache, budget in zip(model.caches, schedule))
        for i in range(64, 70):
            model(ids[:, i:i + 1], input_pos=torch.tensor([i]))
    for cache, budget in zip(model.caches, schedule):
        assert cache.length == budget and cache._k.size(2) <= budget + 8 and cache.processed_tokens == 70
        assert (cache.num_recent, cache.num_selected) == (4, budget - 4)


def test_one_streaming_selection_equals_kvcache_factory_on_a_long_prompt():
    # A prompt long enough for the full pyramid: layer_capacity equals stream_capacity in every layer, and one
    # forward selects with the same observation window, pooling and top-k as KVCache-Factory's compress().
    torch.manual_seed(2)
    compressor = PyramidKVCompressor(num_hidden_layers=3, max_capacity_prompt=12, window_size=4, kernel_size=3,
                                     streaming=True)
    q, k, v = torch.randn(1, 4, 80, 16), torch.randn(1, 2, 80, 16), torch.randn(1, 2, 80, 16)
    for layer in range(3):
        assert compressor.layer_capacity(layer, 80) == compressor.stream_capacity(layer)
        expected_k, expected_v, _, _ = compressor.compress(layer, q, k, v)
        kept_k, kept_v = compressor.stream_evict(layer, q, k, v)
        assert torch.equal(kept_k, expected_k) and torch.equal(kept_v, expected_v)


@pytest.fixture
def qwen3_17b_dir(tmp_path):
    """Model config only: budgets are sized from the architecture, no weights are read."""
    directory = tmp_path / "Qwen3-1.7B-Base"
    directory.mkdir()
    (directory / "model_config.yaml").write_text(yaml.safe_dump(litgpt_model_config(QWEN3_17B)))
    (directory / "lit_model.pth").write_bytes(b"")
    return directory


@pytest.mark.parametrize("context, window, factor, tasks, budget", [
    ("64k", 66560, 2.0, "babilong_64k,long_ppl_32k,long_ppl_64k", 19321),
    ("128k", 132096, 4.0, "babilong_128k,long_ppl_128k", 20954),
])
def test_long_configs_fix_window_yarn_tasks_and_the_logkv_budget(qwen3_17b_dir, context, window, factor, tasks,
                                                                 budget):
    config = REPO_ROOT / "configs" / f"qwen3_1.7b_{context}_pyramidkv.yaml"
    long = evaluate.parse_args(["--config", str(config), "--suite", "long", "--model", str(qwen3_17b_dir)])
    assert (long.context, long.max_seq_length, long.rope_scaling["factor"]) == (context, window, factor)
    assert long.tasks == tasks and long.limit == 100 and long.include_path == str(evaluate.LONG_TASKS_DIR)
    assert long.metadata == {"pretrained": str(qwen3_17b_dir), "max_seq_lengths": [window - 1024]}
    # The pyramid of layer budgets holds K + V in bf16 within the LogKV cache's allocated bytes, and one more
    # entry of average budget would not.
    assert long.max_capacity_prompt == budget and long.window_size == 64 and long.streaming
    budget_bytes = yaml.safe_load(config.read_text())["budget_bytes"]
    assert method.persistent_bytes(long, QWEN3_17B, 2) <= budget_bytes
    bigger = argparse.Namespace(**{**vars(long), "max_capacity_prompt": budget + 1})
    assert method.persistent_bytes(bigger, QWEN3_17B, 2) > budget_bytes
    ruler = evaluate.parse_args(["--config", str(config), "--suite", "ruler", "--model", str(qwen3_17b_dir)])
    assert len(ruler.tasks.split(",")) == 13 and "ruler_qa_hotpot" in ruler.tasks and ruler.limit == 100
    settings = evaluate.evaluation_settings(ruler, 28)
    assert settings["rope_scaling"]["factor"] == factor and settings["prefill_chunk_size"] == 2048


@pytest.mark.parametrize("argv, message", [
    (["--suite", "main", "--context", "64k"], "runs at 32k"),
    (["--suite", "ruler"], "runs at 64k/128k"),
    (["--suite", "long", "--context", "64k", "--max-seq-length", "40000"], "fixes the window"),
    (["--suite", "long", "--context", "64k", "--streaming"], "needs --prefill-chunk-size"),
])
def test_long_suites_refuse_a_mismatched_protocol(argv, message, capsys):
    with pytest.raises(SystemExit):
        evaluate.parse_args(argv)
    assert message in capsys.readouterr().err


def test_long_task_files_are_verbatim_copies():
    import hashlib
    import json

    source = json.loads((REPO_ROOT / "tasks" / "SOURCE.json").read_text())
    for path, digest in source["sha256"].items():
        assert hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() == digest, path
    names = {p.name for p in (REPO_ROOT / "tasks").rglob("*.yaml")}
    assert {"babilong_64k.yaml", "babilong_128k.yaml", "long_ppl_64k.yaml", "long_ppl_128k.yaml"} <= names


def test_platform_launcher_runs_the_long_stages_of_a_long_config(tmp_path):
    env = platform_launcher_env(tmp_path)
    env.update(MA_NUM_GPUS="4")
    command = ["bash", str(REPO_ROOT / "majob.sh"), "configs/qwen3_1.7b_64k_pyramidkv.yaml", "--limit", "4"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert [call.split("--suite ")[1].split()[0] for call in calls] == ["long", "ruler"]
    refused = subprocess.run(command + ["--context", "128k"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert refused.returncode == 2 and "not allowed" in refused.stderr
