"""64K / 128K: YaRN, chunked cached forwards, streaming H2O at a fixed budget, the long suites and launcher."""

import subprocess

import pytest
import torch
import yaml

from h2okv import evaluate
from h2okv.compressor import H2OCompressor, attention_with_scores
from h2okv.kv_cache import CompressedKVCache
from h2okv.qwen3 import Qwen3Config, build_rope_cache
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
                                  max_position_embeddings=int(32768 * factor),
                                  rope_scaling=dict(rope_type="yarn", factor=factor, beta_fast=32, beta_slow=1,
                                                    original_max_position_embeddings=32768))
    # max_position_embeddings = original x factor: transformers 4.51 (torch218) takes the factor from that ratio.
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


def streaming(budget, window):
    compressor = H2OCompressor(max_capacity_prompt=budget, window_size=window, streaming=True, score_chunk_size=3)
    return lambda layer: CompressedKVCache(layer, compressor=compressor)


def test_chunked_forward_equals_one_forward():
    model = tiny_model()
    ids = torch.randint(0, TINY.vocab_size, (1, 45))
    assert torch.allclose(run(model, ids, full, chunk=8, decode=5), run(model, ids, full, decode=5), atol=1e-5)
    model.prefill_chunk_size = 8
    model.set_kv_cache(full)
    with torch.no_grad():  # logits only from lm_head_start, across chunk boundaries
        sliced = model(ids, input_pos=torch.arange(45), lm_head_start=21)
    assert sliced.shape[1] == 24


def test_streaming_h2o_with_room_for_everything_is_full_attention():
    model = tiny_model()
    ids = torch.randint(0, TINY.vocab_size, (1, 50))
    assert torch.allclose(run(model, ids, streaming(64, 8), chunk=8, decode=6), run(model, ids, full, decode=6),
                          atol=1e-4)
    assert not any(cache.compressed for cache in model.caches)


def test_streaming_h2o_never_holds_more_than_the_budget():
    model = tiny_model()
    ids = torch.randint(0, TINY.vocab_size, (1, 70))
    model.prefill_chunk_size = 8
    model.set_kv_cache(streaming(16, 8))
    with torch.no_grad():
        for start in range(0, 64, 8):  # one forward per chunk, as the chunked model does
            model(ids[:, start:start + 8], input_pos=torch.arange(start, start + 8), lm_head_start=7)
            assert all(cache.length <= 16 and cache.k.size(2) <= 16 for cache in model.caches)
        for i in range(64, 70):
            model(ids[:, i:i + 1], input_pos=torch.tensor([i]))
    for cache in model.caches:
        assert cache.length == 16 and cache._k.size(2) <= 16 + 8 and cache.processed_tokens == 70
        assert cache.scores.shape == (1, TINY.num_key_value_heads, 16) and cache.evicted == 70 - 16
        assert (cache.num_recent, cache.num_selected) == (8, 8)


def test_chunked_scores_equal_one_block():
    torch.manual_seed(1)
    q, k, v = torch.randn(1, 4, 7, 16), torch.randn(1, 2, 20, 16), torch.randn(1, 2, 20, 16)
    whole, whole_scores = attention_with_scores(q, k, v, "mean")
    blocked, blocked_scores = attention_with_scores(q, k, v, "mean", chunk_size=3)
    assert torch.allclose(whole, blocked, atol=1e-6) and torch.allclose(whole_scores, blocked_scores, atol=1e-6)


@pytest.fixture
def qwen3_17b_dir(tmp_path):
    """Model config only: budgets are sized from the architecture, no weights are read."""
    directory = tmp_path / "Qwen3-1.7B-Base"
    directory.mkdir()
    (directory / "model_config.yaml").write_text(yaml.safe_dump(litgpt_model_config(QWEN3_17B)))
    (directory / "lit_model.pth").write_bytes(b"")
    return directory


@pytest.mark.parametrize("context, window, factor, tasks, budget", [
    ("64k", 66560, 2.0, "babilong_64k,long_ppl_32k,long_ppl_64k", 19173),
    ("128k", 132096, 4.0, "babilong_128k,long_ppl_128k", 20794),
])
def test_long_configs_fix_window_yarn_tasks_and_the_logkv_budget(qwen3_17b_dir, context, window, factor, tasks,
                                                                 budget):
    config = REPO_ROOT / "configs" / f"qwen3_1.7b_{context}_h2o.yaml"
    long = evaluate.parse_args(["--config", str(config), "--suite", "long", "--model", str(qwen3_17b_dir)])
    assert (long.context, long.max_seq_length, long.rope_scaling["factor"]) == (context, window, factor)
    assert long.tasks == tasks and long.limit == 100 and long.include_path == str(evaluate.LONG_TASKS_DIR)
    assert long.metadata == {"pretrained": str(qwen3_17b_dir), "max_seq_lengths": [window - 1024]}
    # K + V in bf16 and an fp32 score per KV head and layer must fit the LogKV cache's allocated bytes.
    assert long.max_capacity_prompt == budget and long.window_size == budget // 2 and long.streaming
    assert 28 * 8 * budget * (2 * 128 * 2 + 4) <= yaml.safe_load(config.read_text())["budget_bytes"]
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
    command = ["bash", str(REPO_ROOT / "majob.sh"), "configs/qwen3_1.7b_64k_h2o.yaml", "--limit", "4"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert [call.split("--suite ")[1].split()[0] for call in calls] == ["long", "ruler"]
    refused = subprocess.run(command + ["--context", "128k"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert refused.returncode == 2 and "not allowed" in refused.stderr
