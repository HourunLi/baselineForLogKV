import copy
from types import SimpleNamespace
from pathlib import Path
import json

import pytest

from models.qwen3.compare_results import compare_results
from models.qwen3.evaluate import build_eval_kwargs, run_evaluation


def test_both_backends_share_inputs_but_only_semantic_uses_its_method_flags(tmp_path):
    def main(log_kv_B=512, log_kv_semantic_clusters=False):
        pass
    config = dict(ckpt_dir=str(tmp_path), context_length=32768, log_kv_B=128,
                  log_kv_semantic_clusters=True, max_steps=1200, log_kv_diag_mode="oracle")
    evaluator = SimpleNamespace(main=main, expand_env_vars=lambda x: x,
                                _load_yaml_config=lambda *args: config)
    common = (evaluator, tmp_path / "train.yaml", tmp_path / "model")
    dms = build_eval_kwargs(*common, "dms", "niah_single_2,niah_single_3", 4, tmp_path / "out")
    semantic = build_eval_kwargs(*common, "logkv", "niah_single_2,niah_single_3", 4, tmp_path / "out")
    assert dms == {k: v for k, v in semantic.items() if k not in ("log_kv_B", "log_kv_semantic_clusters")}
    assert "model_backend" not in dms and "comparison_contract" not in dms
    assert semantic["log_kv_B"] == 128 and semantic["log_kv_semantic_clusters"]
    assert dms["metadata"] == {"pretrained": str(tmp_path), "max_seq_lengths": [32768]}
    assert "max_steps" not in semantic and "log_kv_diag_mode" not in semantic
    evaluator.__file__ = str(tmp_path / "litgpt" / "eval.py")
    config["ckpt_dir"] = "checkpoints/base"
    relative = build_eval_kwargs(*common, "dms", "niah_single_2", None, tmp_path / "out")
    assert relative["tokenizer_dir"] == str(tmp_path / "litgpt/checkpoints/base")


def test_result_check_rejects_same_dataset_name_with_different_prompt_or_runtime():
    fields = ("source_weight_sha256", "source_config_files", "tokenizer_files", "evaluation_code", "compatibility_code", "versions",
              "cache", "benchmark", "metadata", "limit", "max_seq_length")
    result = {"comparison_contract": dict.fromkeys(fields, "same"), "results": {"samples": {"niah": [
        dict(doc_id=0, doc_hash="doc", prompt_hash="prompt", target_hash="target",
             arguments=[["a prompt", {"max_gen_toks": 16}]], resps=["ignored model response"]),
    ]}}}
    other = copy.deepcopy(result)
    other["results"]["samples"]["niah"][0]["resps"] = ["a different model response"]
    assert compare_results(result, other) == {"niah": 1}
    other["results"]["samples"]["niah"][0]["arguments"][0][1]["max_gen_toks"] = 32
    with pytest.raises(ValueError, match="Actual test"):
        compare_results(result, other)
    other = copy.deepcopy(result)
    other["comparison_contract"]["versions"] = "different"
    with pytest.raises(ValueError, match="versions"):
        compare_results(result, other)


def test_outputs_and_provenance_stay_outside_reference(tmp_path, monkeypatch):
    import models.qwen3.evaluate as runner
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(runner, "install_dms_adapter", lambda evaluator: None)
    reference, output = tmp_path / "litgpt", tmp_path / "baseline/results"
    reference.mkdir()
    entrypoint = reference / "eval.py"
    entrypoint.write_text("unchanged source")
    def main(**kwargs):
        assert Path.cwd() == output
        Path("eval_results_cache.json").write_text("cached")
        Path("eval_results_20261012_000000.json").write_text(json.dumps({
            "log_kv_B": 128, "swa_window_size": None, "results": {"samples": {"example": []}},
        }))
    evaluator = SimpleNamespace(main=main, __file__=str(entrypoint))
    kwargs = dict(output_path=str(output), tokenizer_dir="base", config_overrides={"block_size": 32768})
    cwd = Path.cwd()
    run_evaluation(evaluator, kwargs, "dms", {"example": "source"})
    assert Path.cwd() == cwd and entrypoint.read_text() == "unchanged source"
    assert list(reference.iterdir()) == [entrypoint]
    result = json.loads((output / "eval_results_20261012_000000.json").read_text())
    assert result["model_backend"] == "dms" and result["comparison_contract"] == {"example": "source"}
    assert "log_kv_B" not in result and "swa_window_size" not in result
    # A failed second run must not relabel existing results with a new contract.
    evaluator.main = lambda **kwargs: None
    with pytest.raises(RuntimeError, match="one new evaluation result"):
        run_evaluation(evaluator, kwargs, "dms", {"example": "different source"})
    with pytest.raises(ValueError, match="outside"):
        run_evaluation(evaluator, {**kwargs, "output_path": str(reference / "results")}, "dms", {})
    assert not (reference / "results").exists()
