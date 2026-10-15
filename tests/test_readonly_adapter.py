"""Baseline integration must leave the bundled reference runtime and its native API intact."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lm_eval")

from models._vendor.semanticlogkv.litgpt import kv_compression
from models.qwen3 import evaluate
from models.qwen3.config import RUNTIME_ROOT
from models.qwen3.semanticlogkv_eval import compression_snapshot


def test_private_adapter_does_not_modify_runtime_files_or_native_metrics():
    paths = sorted(RUNTIME_ROOT.rglob("*.py"))
    contents = {path: path.read_bytes() for path in paths}
    native_snapshot = kv_compression.cache_compression_snapshot

    evaluator = evaluate.load_evaluator()
    native_class = evaluator.LogKVLM
    evaluate.install_dms_adapter(evaluator)

    assert evaluator.LogKVLM is not native_class
    assert evaluator.LogKVLM.__bases__ == (native_class,)
    for name in ("_score_tokens", "loglikelihood", "loglikelihood_rolling", "generate_until"):
        assert getattr(evaluator.LogKVLM, name) is getattr(native_class, name)
    assert evaluator.cache_compression_snapshot is compression_snapshot
    assert kv_compression.cache_compression_snapshot is native_snapshot
    assert {path: path.read_bytes() for path in paths} == contents

    # Loading another evaluator remains native: DMS wiring belongs only to the
    # specific private module used by the baseline runner.
    native_evaluator = evaluate.load_evaluator()
    assert native_evaluator is not evaluator
    assert native_evaluator.LogKVLM.__name__ == "LogKVLM"
    assert native_evaluator.cache_compression_snapshot is native_snapshot

    cache = SimpleNamespace(k=torch.zeros(1, 2, 8, 4), is_sliding_window=False)
    model = SimpleNamespace(transformer=SimpleNamespace(h=[
        SimpleNamespace(attn=SimpleNamespace(kv_cache=cache)),
    ]))
    assert compression_snapshot(model, 5) == native_snapshot(model, 5)
