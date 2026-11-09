"""64K / 128K: the YaRN backport, streaming ChunkKV at a fixed budget, the long suites and the launcher."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from chunkkv import evaluate as runner
from chunkkv.core import ChunkKVSettings, streaming_keep
from chunkkv.model import ChunkKVModel
from chunkkv.semanticlogkv_eval import kv_bytes_per_token

from .test_evaluate import platform_launcher_env
from .test_model import decode_tokens


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_backported_yarn_matches_transformers(factor):
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    from chunkkv._vendor.semanticlogkv.litgpt.model import build_rope_cache

    hf = transformers.Qwen3Config(head_dim=128, hidden_size=2048, num_attention_heads=16, rope_theta=1e6,
                                  max_position_embeddings=int(32768 * factor),
                                  rope_scaling=dict(rope_type="yarn", factor=factor, beta_fast=32, beta_slow=1,
                                                    original_max_position_embeddings=32768))
    # max_position_embeddings = original x factor: transformers 4.51 (torch218) takes the factor from that ratio.
    inv_freq, attention_factor = ROPE_INIT_FUNCTIONS["yarn"](hf, "cpu")
    angles = torch.outer(torch.arange(int(32768 * factor) + 1024).float(), inv_freq).repeat(1, 2)
    cos, sin = build_rope_cache(angles.size(0), 128, base=1000000, extra_config=dict(
        factor=factor, beta_fast=32, beta_slow=1, original_max_seq_len=32768))
    assert torch.allclose(cos, angles.cos() * attention_factor, atol=1e-4)
    assert torch.allclose(sin, angles.sin() * attention_factor, atol=1e-4)


def streaming(budget, **options):
    return ChunkKVSettings(budget=budget, window_size=4, chunk_size=3, streaming=True, prefill_chunk_size=8,
                           **options)


def test_streaming_with_room_for_everything_is_dense_inference(gpt_pair):
    reference, gpt = gpt_pair
    model = ChunkKVModel(gpt, streaming(64))
    reference.set_kv_cache(1, device="cpu", dtype=torch.float32)
    prompt, tokens = torch.randint(0, 64, (1, 50)), torch.randint(0, 64, (6,))
    with torch.no_grad():
        model.reset_cache()
        torch.testing.assert_close(decode_tokens(model, prompt, tokens), decode_tokens(reference, prompt, tokens),
                                   rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reuse", [1, 2])
def test_streaming_never_holds_more_than_the_budget(gpt_pair, reuse):
    _, gpt = gpt_pair
    model = ChunkKVModel(gpt, streaming(20, reuse_layers=reuse))
    prompt, tokens = torch.randint(0, 64, (1, 70)), torch.randint(0, 64, (6,))
    with torch.no_grad():
        model.reset_cache()
        for start in range(0, 64, 8):  # one forward per chunk, as the chunked request does
            model(prompt[:, start:start + 8], torch.arange(start, start + 8), lm_head_start=7)
            assert all(attn.length <= 20 for attn in model.attentions)
        for i, token in enumerate(tokens):
            model(token.view(1, 1), torch.tensor([64 + i]), lm_head_start=0)
    for attn in model.attentions:
        kept = attn.positions[:attn.length].tolist()
        assert attn.length <= 20 and attn.keys.size(2) <= 20 + 8 and kept == sorted(set(kept))
        assert kept[-4:] == list(range(66, 70))  # the window: the newest four tokens
        assert attn.chunk_tokens == attn.length - 4
    if reuse == 2:  # a reuse group shares the scoring layer's selection
        positions = [attn.positions[:attn.length].tolist() for attn in model.attentions]
        assert positions[0] == positions[1] and positions[2] == positions[3]
    counts = model.kv_compression_snapshot(70)
    assert counts["retained_slots"] == sum(attn.length for attn in model.attentions) * 2


def test_streaming_selection_keeps_whole_chunks_by_score_and_the_window():
    settings = ChunkKVSettings(budget=11, window_size=3, chunk_size=4, streaming=True, prefill_chunk_size=8)
    positions = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    scores = torch.zeros(15)
    scores[4:8] = 1.0   # chunk 2 (positions 8-11)
    scores[0:4] = 0.5   # chunk 0
    scores[8:12] = 0.1  # chunk 3 (positions 12-15)
    keep, chunk_tokens = streaming_keep(scores, positions, settings)
    # 8 older slots: chunks 2 and 0 by score; chunk 3 no longer fits; window = positions 16-18.
    assert positions[keep].tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18] and chunk_tokens == 8


@pytest.mark.parametrize("context, budget", [("64k", 19286), ("128k", 20916)])
def test_long_budgets_fit_the_logkv_cache_bytes(context, budget):
    options = yaml.safe_load((runner.BASELINE_ROOT / f"configs/chunkkv/qwen3_1.7b_{context}.yaml").read_text())
    config = SimpleNamespace(n_layer=28, n_query_groups=8, head_size=128)
    per_entry = kv_bytes_per_token(config, torch.bfloat16) + 28 * 8  # K/V of every KV group + an int64 position
    assert options["budget_bytes"] // per_entry == budget and options["streaming"]
    loaded = runner.load_chunkkv_options(runner.BASELINE_ROOT / f"configs/chunkkv/qwen3_1.7b_{context}.yaml",
                                         dict(budget=None, keep_ratio=None, chunk_size=None, window_size=None,
                                              reuse_layers=None))
    assert loaded["budget"] == "bytes" and loaded["prefill_chunk_size"] == 2048


def print_config(*argv):
    done = subprocess.run([sys.executable, "-m", "chunkkv.evaluate", *argv, "--print-config"],
                          cwd=runner.BASELINE_ROOT, capture_output=True, text=True)
    return done


@pytest.mark.parametrize("context, window, factor, long_tasks", [
    ("64k", 66560, 2.0, "babilong_64k,long_ppl_32k,long_ppl_64k"),
    ("128k", 132096, 4.0, "babilong_128k,long_ppl_128k"),
])
def test_long_suites_fix_window_yarn_tasks_and_samples(context, window, factor, long_tasks):
    long = json.loads(print_config("--context", context, "--suite", "long").stdout)
    assert long["config_overrides"] == {"block_size": window, "rope_adjustments": dict(
        factor=factor, beta_fast=32, beta_slow=1, original_max_seq_len=32768)}
    assert long["benchmark"] == long_tasks and long["limit"] == 100
    assert long["metadata"]["max_seq_lengths"] == [window - 1024]  # long_ppl reads the tokenizer from it
    assert long["chunkkv_config"].endswith(f"configs/chunkkv/qwen3_1.7b_{context}.yaml")
    assert long["chunkkv"]["streaming"] and long["long_ppl_data"].endswith("c5e39bf32e33f9111323aa68d7d9000d22722035")
    ruler = json.loads(print_config("--context", context, "--suite", "ruler").stdout)
    assert len(ruler["benchmark"].split(",")) == 13 and ruler["limit"] == 100


@pytest.mark.parametrize("argv, message", [
    (["--context", "64k", "--suite", "main"], "runs at 32k"),
    (["--suite", "ruler"], "runs at 64k/128k"),
    (["--context", "64k", "--suite", "long", "--backend", "dense"], "own eval_long configs"),
])
def test_long_suites_refuse_a_mismatched_protocol(argv, message):
    done = print_config(*argv)
    assert done.returncode != 0 and message in done.stderr


def test_long_task_files_are_verbatim_copies():
    import hashlib

    source = json.loads((runner.BASELINE_ROOT / "tasks" / "SOURCE.json").read_text())
    for path, digest in source["sha256"].items():
        assert hashlib.sha256((runner.BASELINE_ROOT / path).read_bytes()).hexdigest() == digest, path


def test_platform_launcher_runs_the_long_stages_for_a_long_context(tmp_path):
    env = {**platform_launcher_env(tmp_path), "MA_NUM_GPUS": "4"}
    command = ["bash", str(runner.BASELINE_ROOT / "majob.sh"), "--context", "64k", "--limit", "4"]
    done = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert [call.split("--suite ")[1].split()[0] for call in calls] == ["long", "ruler"]
    assert all("--context 64k" in call for call in calls)
    refused = subprocess.run(command[:2] + ["--context", "96k"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert refused.returncode == 2 and "Unknown --context" in refused.stderr
