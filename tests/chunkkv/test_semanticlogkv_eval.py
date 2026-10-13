"""ChunkKV through the real, privately loaded SemanticLogKV ``eval.py`` on CPU.

Model loading, tokenizer, truncation, generation, scoring and metric collection
are the reference code; only the tiny checkpoint and CPU device are test inputs.
"""

from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

pytest.importorskip("lm_eval")

import litgpt  # noqa: E402
from litgpt import kv_compression  # noqa: E402
from litgpt.model import GPT  # noqa: E402

from chunkkv import evaluate  # noqa: E402
from chunkkv.model import ChunkKVModel  # noqa: E402
from chunkkv.semanticlogkv_eval import compression_snapshot, kv_bytes_per_token  # noqa: E402

from .conftest import tiny_config  # noqa: E402


ROOT = Path(litgpt.__file__).resolve().parents[1]
VOCAB = ["<unk>", "<bos>", "<eos>"] + [f"w{i}" for i in range(3, 64)]
LOGKV = dict(log_kv_B=8, log_kv_recent_size=16, log_kv_prefill_block=8, log_kv_second_order_scale=0.0,
             log_kv_alpha_exact_tokens=0, log_kv_beta_novelty=False, log_kv_beta_adaptive_merge=False)


@pytest.fixture(scope="module")
def evaluator():
    return evaluate.load_evaluator(ROOT)


@pytest.fixture
def checkpoint(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    torch.manual_seed(0)
    config = tiny_config()
    torch.save(GPT(config).state_dict(), tmp_path / "lit_model.pth")
    (tmp_path / "model_config.yaml").write_text(yaml.dump(asdict(config)))
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(VOCAB)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(
        {"bos_token": "<bos>", "eos_token": "<eos>", "add_bos_token": False, "tokenizer_class": "Fast"}))
    return tmp_path


def load_lm(evaluator, checkpoint, options=None, **kwargs):
    """Construct exactly as the reference ``main`` does, on CPU."""
    native = evaluator.LogKVLM
    if options is not None:
        evaluate.install_chunkkv_adapter(evaluator, options)
    try:
        return evaluator.LogKVLM(str(checkpoint), device="cpu", config_overrides={"block_size": 128},
                                 tokenizer_dir=str(checkpoint), **kwargs)
    finally:
        evaluator.LogKVLM = native
        evaluator.cache_compression_snapshot = kv_compression.cache_compression_snapshot
        evaluator.print = print


def options(**values):
    return {**dict(chunk_size=4, window_size=6, budget=None, keep_ratio=None, reuse_layers=1), **values}


def words(n, start=0):
    return " ".join(f"w{3 + (start + 7 * i) % 61}" for i in range(n))


def requests():
    generation = SimpleNamespace(args=(words(90), {"max_gen_toks": 5, "do_sample": False, "until": ["\n"]}),
                                 task_name="niah", doc_id=0)
    choices = [SimpleNamespace(args=(words(80), " " + words(3, start=s))) for s in (1, 2)]
    return generation, choices


def run_requests(lm, evaluator):
    generation, choices = requests()
    snapshot = evaluator.cache_compression_snapshot
    evaluator.cache_compression_snapshot = compression_snapshot
    try:
        with torch.no_grad():
            return lm.generate_until([generation]), lm.loglikelihood(choices), lm.loglikelihood_rolling(
                [SimpleNamespace(args=(words(300),))])
    finally:
        evaluator.cache_compression_snapshot = snapshot


def test_uncompressed_chunkkv_reproduces_reference_dense_backend(evaluator, checkpoint):
    dense = load_lm(evaluator, checkpoint, log_kv_dense_mode=True)
    chunk = load_lm(evaluator, checkpoint, options(budget=128))
    assert isinstance(chunk.model, ChunkKVModel)
    assert chunk.cache_budget["budget_source"] == "explicit_tokens"
    expected, actual = run_requests(dense, evaluator), run_requests(chunk, evaluator)
    assert actual[0] == expected[0]
    for (score, greedy), (reference, reference_greedy) in zip(actual[1], expected[1]):
        assert score == pytest.approx(reference, abs=1e-2) and greedy == reference_greedy
    assert actual[2][0] == pytest.approx(expected[2][0], rel=1e-3)
    assert [row["retained_slots"] for row in chunk.compression_metrics] == \
        [row["dense_slots"] for row in chunk.compression_metrics]
    assert chunk.model.processed is None  # request cache released after every call


def test_compressed_requests_record_shared_metrics(evaluator, checkpoint):
    lm = load_lm(evaluator, checkpoint, options(budget=26))
    (text,), scores, (rolling,) = run_requests(lm, evaluator)
    assert isinstance(text, str) and all(math.isfinite(score) for score, _ in scores) and math.isfinite(rolling)
    generation, *choices = lm.compression_metrics[:3]
    groups = 4 * 2
    # Prompt: window 6 + five chunks of 4; four generated tokens were forwarded afterwards.
    assert generation["prompt_len"] == 90 and generation["processed_tokens"] == 94
    assert generation["exact_slots"] == 20 * groups and generation["recent_slots"] == (6 + 4) * groups
    assert generation["kv_retention_ratio"] == pytest.approx(30 / 94)
    for choice in choices:  # context compressed, the 3-token continuation appended
        assert choice["retained_slots"] == (26 + 3) * groups
    # Rolling PPL scores each window in one dense forward behind its one-token context.
    assert all(row["kv_retention_ratio"] == 1 for row in lm.compression_metrics[3:])
    assert lm.gen_metrics and lm.ppl_metrics and lm.model.processed is None


def test_logkv_budget_matches_reference_persistent_cache_bytes(evaluator, checkpoint):
    native = load_lm(evaluator, checkpoint, **LOGKV)
    native._set_eval_cache()
    reference_bytes = native.cache_budget["persistent_cache_bytes"]
    lm = load_lm(evaluator, checkpoint, options(budget="logkv"), **LOGKV)
    per_token = kv_bytes_per_token(tiny_config(), torch.bfloat16)
    assert lm.cache_budget["reference_logkv_bytes"] == reference_bytes > 0
    assert lm.cache_budget["prompt_budget_tokens"] == reference_bytes // per_token
    assert lm.cache_budget["budget_source"] == "logkv_persistent_bytes"
    assert all(block.attn.inner.kv_cache is None for block in lm.model.gpt.transformer.h)


def test_incomplete_checkpoints_and_method_switches_fail(evaluator, checkpoint):
    state = torch.load(checkpoint / "lit_model.pth")
    state.pop("lm_head.weight")
    torch.save(state, checkpoint / "lit_model.pth")
    with pytest.raises(ValueError, match="lm_head.weight"):
        load_lm(evaluator, checkpoint, options(budget=64))
    with pytest.raises(ValueError, match="dense/SWA"):
        load_lm(evaluator, checkpoint, options(budget=64), log_kv_dense_mode=True)


def test_private_adapter_leaves_reference_files_and_native_api_intact(evaluator):
    paths = [ROOT / name for name in ("eval.py", "litgpt/kv_compression.py", "litgpt/model.py")]
    contents = {path: path.read_bytes() for path in paths}
    native_class, native_snapshot = evaluator.LogKVLM, kv_compression.cache_compression_snapshot
    evaluate.install_chunkkv_adapter(evaluator, options(budget=64))
    try:
        assert evaluator.LogKVLM.__bases__ == (native_class,)
        for name in ("_score_tokens", "loglikelihood", "loglikelihood_rolling", "generate_until"):
            assert getattr(evaluator.LogKVLM, name) is getattr(native_class, name)
        assert evaluator.cache_compression_snapshot is compression_snapshot
        assert kv_compression.cache_compression_snapshot is native_snapshot
        other = evaluate.load_evaluator(ROOT)
        assert other is not evaluator and other.LogKVLM.__name__ == "LogKVLM"
    finally:
        evaluator.LogKVLM, evaluator.print = native_class, print
        evaluator.cache_compression_snapshot = native_snapshot
    assert {path: path.read_bytes() for path in paths} == contents
