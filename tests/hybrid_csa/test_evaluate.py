"""Inputs, checkpoint gate, provenance and launch failures for the independent baseline."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from hybrid_csa import evaluate as runner
from hybrid_csa.compare_results import CONTRACT_FIELDS, compare_results


def test_backends_share_protocol_and_only_logkv_receives_logkv_settings(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(dict(ckpt_dir="/assets/base/", context_length=32768, log_kv_B=128)))
    def main(log_kv_B=512):
        pass
    evaluator = SimpleNamespace(main=main)
    records = [runner.build_eval_kwargs(evaluator, path, "/assets/logkv" if method == "logkv" else None,
                                       method, "niah_single_1", 4, tmp_path) for method in runner.BACKENDS]
    for field in ("tokenizer_dir", "benchmark", "metadata", "limit", "config_overrides"):
        assert records[0][field] == records[1][field] == records[2][field]
    assert records[0]["metadata"] == {"pretrained": "/assets/base/", "max_seq_lengths": [32768]}
    assert records[0]["checkpoint_dir"] == records[2]["checkpoint_dir"] == "/assets/base/"
    assert "log_kv_B" not in records[0] and records[1]["log_kv_B"] == 128
    with pytest.raises(ValueError, match="exactly"):
        runner.build_eval_kwargs(evaluator, path, None, "hybrid_csa", "niah_single_1", 4, tmp_path,
                                 max_seq_lengths=[8192, 32768])
    path.write_text("ckpt_dir: /base\ncontext_length: 8192\n")
    with pytest.raises(ValueError, match="32768"):
        runner.build_eval_kwargs(evaluator, path, None, "hybrid_csa", "boolq", 4, tmp_path)


def test_configuration_is_complete_and_suites_are_fixed():
    options = runner.load_hybrid_csa_options(runner.DEFAULT_HYBRID_CSA_CONFIG, {"index_topk": 64})
    assert options["index_topk"] == 64 and options["full_layer_interval"] == 3
    assert options["compress_ratio"] == 4 and options["window_size"] == 128
    assert set(options) == set(runner.HYBRID_CSA_FIELDS)
    assert len(runner.load_eval_suite(runner.DEFAULT_EVALUATION_CONFIG, "main")[0].split(",")) == 47
    assert runner.load_eval_suite(runner.DEFAULT_EVALUATION_CONFIG, "niah") == \
        ("niah_single_1,niah_single_2,niah_single_3", [32768])
    with pytest.raises(ValueError, match="compress_ratio"):
        runner.load_hybrid_csa_options(runner.DEFAULT_HYBRID_CSA_CONFIG, {"compress_ratio": 8})


def test_eval_rejects_untrained_adapter_before_importing_runtime():
    result = subprocess.run([sys.executable, "-m", "hybrid_csa.evaluate", "--suite", "niah"],
                            cwd=runner.BASELINE_ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "--adapter is required" in result.stderr


def aligned_result():
    contract = {**dict.fromkeys(CONTRACT_FIELDS, "same"), "protocol_code": {"eval.py": "same"},
                "evaluation_code": {"eval.py": "raw"}, "compatibility_code": {"model.py": "method"}}
    return {"comparison_contract": contract, "results": {"samples": {"niah": [
        dict(doc_id=0, doc_hash="doc", prompt_hash="prompt", target_hash="target",
             arguments=[["prompt", {"max_gen_toks": 16}]], resps=["answer"]),
    ]}}}


def test_compare_separates_namespace_and_method_changes_from_protocol():
    left, right = aligned_result(), aligned_result()
    right["comparison_contract"]["evaluation_code"]["eval.py"] = "namespace relocated"
    right["comparison_contract"]["compatibility_code"]["model.py"] = "different compression"
    right["results"]["samples"]["niah"][0]["resps"] = ["another answer"]
    assert compare_results(left, right) == {"niah": 1}
    for field in ("versions", "tokenizer_files", "protocol_code"):
        changed = copy.deepcopy(right)
        changed["comparison_contract"][field] = "different"
        with pytest.raises(ValueError, match=field):
            compare_results(left, changed)
    changed = copy.deepcopy(right)
    changed["results"]["samples"]["niah"][0]["arguments"][0][1]["max_gen_toks"] = 32
    with pytest.raises(ValueError, match="Actual test"):
        compare_results(left, changed)
    right["comparison_contract"].pop("protocol_code")
    with pytest.raises(ValueError, match="Historical raw hashes"):
        compare_results(left, right)


def test_namespace_normalization_preserves_real_source_changes(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir(); right.mkdir()
    (left / "eval.py").write_text("from chunkkv._vendor.semanticlogkv import data\nlimit = 4\n")
    (right / "eval.py").write_text("from hybrid_csa._vendor.semanticlogkv import data\nlimit = 4\n")
    assert runner.normalized_protocol_code(left) == runner.normalized_protocol_code(right)
    (right / "eval.py").write_text("from hybrid_csa._vendor.semanticlogkv import data\nlimit = 5\n")
    assert runner.normalized_protocol_code(left) != runner.normalized_protocol_code(right)


def test_result_annotation_records_actual_adapter_and_does_not_reuse_old_result(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    state = SimpleNamespace(enabled=True)
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        is_caching_enabled=lambda: state.enabled,
        disable_caching=lambda: setattr(state, "enabled", False),
        enable_caching=lambda: setattr(state, "enabled", True)))
    adapter = {"sha256": "actual", "stage": "sparse"}
    def install(evaluator, options, adapter_path):
        evaluator.hybrid_csa_adapter_record = adapter
    monkeypatch.setattr(runner, "install_hybrid_csa_adapter", install)
    output = tmp_path / "output"
    def main(**kwargs):
        assert not state.enabled
        (output / "eval_results_20261020_000000.json").write_text(json.dumps({
            "results": {"samples": {}}, "log_kv_B": 128, "cache_budget": {"method": "HybridCSA"}}))
    evaluator = SimpleNamespace(main=main, __file__=str(tmp_path / "runtime/eval.py"))
    kwargs = dict(output_path=str(output), tokenizer_dir="base", config_overrides={"block_size": 32768})
    runner.run_evaluation(evaluator, kwargs, "hybrid_csa", {}, adapter_path="adapter.pt")
    result = json.loads(next(output.glob("eval_results_*.json")).read_text())
    assert state.enabled and result["evaluated_adapter"] == adapter and "log_kv_B" not in result
    evaluator.main = lambda **kwargs: None
    with pytest.raises(RuntimeError, match="one new evaluation result"):
        runner.run_evaluation(evaluator, kwargs, "hybrid_csa", {}, adapter_path="adapter.pt")
