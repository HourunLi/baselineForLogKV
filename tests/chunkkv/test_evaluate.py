"""Launcher contract: shared inputs, budget options, output isolation and result checks."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from chunkkv import evaluate as runner
from chunkkv.compare_results import CONTRACT_FIELDS, compare_results


def fake_evaluator(tmp_path, config):
    def main(log_kv_B=512, log_kv_semantic_clusters=False, log_kv_dense_mode=False):
        pass
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(config))
    return SimpleNamespace(main=main, __file__=str(tmp_path / "runtime" / "eval.py"))


def test_backends_share_task_inputs_and_only_logkv_budget_reads_method_flags(tmp_path):
    config = dict(ckpt_dir=str(tmp_path / "base"), context_length=32768, log_kv_B=128,
                  log_kv_semantic_clusters=True, max_steps=1200, log_kv_diag_mode="oracle")
    evaluator = fake_evaluator(tmp_path, config)
    common = (evaluator, tmp_path / "train.yaml")
    tasks, out = "niah_single_2,niah_single_3", tmp_path / "out"
    chunk = runner.build_eval_kwargs(*common, None, "chunkkv", tasks, 4, out, {"budget": 4096})
    matched = runner.build_eval_kwargs(*common, None, "chunkkv", tasks, 4, out, {"budget": "logkv"})
    dense = runner.build_eval_kwargs(*common, None, "dense", tasks, 4, out)
    semantic = runner.build_eval_kwargs(*common, tmp_path / "cpt", "logkv", tasks, 4, out)
    method = ("log_kv_B", "log_kv_semantic_clusters")
    assert chunk["checkpoint_dir"] == dense["checkpoint_dir"] == str(tmp_path / "base")  # training-free default
    assert {k: v for k, v in matched.items() if k not in method} == chunk
    assert {k: v for k, v in dense.items() if k != "log_kv_dense_mode"} == chunk
    assert {k: v for k, v in semantic.items() if k not in method + ("checkpoint_dir",)} == \
        {k: v for k, v in chunk.items() if k != "checkpoint_dir"}
    assert semantic["log_kv_B"] == matched["log_kv_B"] == 128 and dense["log_kv_dense_mode"] is True
    assert chunk["metadata"] == {"pretrained": str(tmp_path / "base"), "max_seq_lengths": [32768]}
    assert "max_steps" not in semantic and "log_kv_diag_mode" not in semantic
    with pytest.raises(ValueError, match="SemanticLogKV checkpoint"):
        runner.build_eval_kwargs(*common, None, "logkv", tasks, 4, out)


def test_chunkkv_options_merge_yaml_and_cli(tmp_path):
    path = tmp_path / "chunkkv.yaml"
    path.write_text("chunk_size: 8\nbudget: logkv\nkeep_ratio: null\n")
    base = dict(budget=None, keep_ratio=None, chunk_size=None, window_size=None, reuse_layers=None)
    assert runner.load_chunkkv_options(path, base) == dict(
        chunk_size=8, window_size=64, budget="logkv", keep_ratio=None, reuse_layers=1)
    assert runner.load_chunkkv_options(path, {**base, "budget": "2048", "reuse_layers": 2}) == dict(
        chunk_size=8, window_size=64, budget=2048, keep_ratio=None, reuse_layers=2)
    assert runner.load_chunkkv_options(path, {**base, "keep_ratio": 0.1})["budget"] is None
    with pytest.raises(ValueError, match="window"):
        runner.load_chunkkv_options(path, {**base, "budget": "32"})
    path.write_text("budget: logkv\nkeep_ratio: 0.1\n")
    with pytest.raises(ValueError, match="exactly one"):
        runner.load_chunkkv_options(path, base)
    path.write_text("budget: 128\nchunk: 4\n")
    with pytest.raises(ValueError, match="Unknown"):
        runner.load_chunkkv_options(path, base)
    shipped = runner.load_chunkkv_options(runner.DEFAULT_CHUNKKV_CONFIG, base)
    assert shipped == dict(chunk_size=10, window_size=64, budget="logkv", keep_ratio=None, reuse_layers=1)


def test_slot_matched_budget_reads_semanticlogkv_generation_snapshots(tmp_path):
    path = tmp_path / "semantic.json"
    stats = {"request_count": 4, "retained_slots_per_layer_group": 3071.75}
    path.write_text(json.dumps({"model_backend": "logkv",
                                "kv_compression": {"by_request_type": {"generate_until": stats}}}))
    budget, reference = runner.slot_matched_budget(path)
    assert budget == 3071 and reference["requests"] == 4 and reference["retained_slots_per_layer_group"] == 3071.75
    path.write_text(json.dumps({"model_backend": "chunkkv", "kv_compression": {}}))
    with pytest.raises(ValueError, match="backend logkv"):
        runner.slot_matched_budget(path)


def test_outputs_stay_outside_reference_and_results_are_relabelled(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    installed = []
    monkeypatch.setattr(runner, "install_chunkkv_adapter", lambda evaluator, options: installed.append(options))
    reference, output = tmp_path / "runtime", tmp_path / "baseline/results"
    reference.mkdir()
    entrypoint = reference / "eval.py"
    entrypoint.write_text("unchanged source")

    def main(**kwargs):
        assert Path.cwd() == output
        Path("eval_results_cache.json").write_text("cached")
        Path("eval_results_20261013_000000.json").write_text(json.dumps({
            "log_kv_B": 128, "log_kv_dense_mode": False, "swa_window_size": None,
            "cache_budget": {"method": "ChunkKV"}, "results": {"samples": {"example": []}},
        }))
    evaluator = SimpleNamespace(main=main, __file__=str(entrypoint))
    kwargs = dict(output_path=str(output), tokenizer_dir="base", config_overrides={"block_size": 32768})
    cwd = Path.cwd()
    runner.run_evaluation(evaluator, kwargs, "chunkkv", {"example": "source"}, {"budget": 64})
    assert installed == [{"budget": 64}] and Path.cwd() == cwd
    assert entrypoint.read_text() == "unchanged source" and list(reference.iterdir()) == [entrypoint]
    result = json.loads((output / "eval_results_20261013_000000.json").read_text())
    assert result["model_backend"] == "chunkkv" and result["comparison_contract"] == {"example": "source"}
    assert not any(key.startswith("log_kv_") for key in result) and "swa_window_size" not in result
    assert result["cache_budget"] == {"method": "ChunkKV"}
    # A failed second run must not relabel existing results with a new contract.
    evaluator.main = lambda **kwargs: None
    with pytest.raises(RuntimeError, match="one new evaluation result"):
        runner.run_evaluation(evaluator, kwargs, "chunkkv", {"example": "other"}, {"budget": 64})
    with pytest.raises(ValueError, match="outside"):
        runner.run_evaluation(evaluator, {**kwargs, "output_path": str(reference / "results")}, "dense", {})
    assert not (reference / "results").exists()


def test_result_check_rejects_different_prompts_or_runtime_but_not_answers():
    result = {"comparison_contract": dict.fromkeys(CONTRACT_FIELDS, "same"), "results": {"samples": {"niah": [
        dict(doc_id=0, doc_hash="doc", prompt_hash="prompt", target_hash="target",
             arguments=[["a prompt", {"max_gen_toks": 16}]], resps=["ignored model response"]),
    ]}}}
    other = copy.deepcopy(result)
    other["results"]["samples"]["niah"][0]["resps"] = ["a different model response"]
    other["comparison_contract"]["evaluated_model"] = {"checkpoint_dir": "another checkpoint"}
    assert compare_results(result, other) == {"niah": 1}
    other["results"]["samples"]["niah"][0]["arguments"][0][1]["max_gen_toks"] = 32
    with pytest.raises(ValueError, match="Actual test"):
        compare_results(result, other)
    other = copy.deepcopy(result)
    other["comparison_contract"]["versions"] = "different"
    with pytest.raises(ValueError, match="versions"):
        compare_results(result, other)
