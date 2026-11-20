"""Regression tests for the niahLogKV entry point, its persistent storage and the reference request API."""
import copy
import json
import subprocess
import sys

import pytest

from chunkkv.compare_results import CONTRACT_FIELDS, compare_results
from .test_semanticlogkv_eval import LOGKV, checkpoint, options, requests, words


def test_default_plan_and_length_specific_tasks():
    result = subprocess.check_output([sys.executable, "-m", "chunkkv.niah", "--print-config"])
    config = json.loads(result)
    assert config["checkpoint"] == config["tokenizer"] and config["checkpoint"].endswith("/Qwen3-1.7B-Base")
    rows = {r["folder"]: r for r in config["stages"]}
    assert len(rows) == 14 and len(rows["niah"]["benchmark"].split(",")) == 20
    assert rows["main"]["limit"] is None
    assert rows["niah"]["lengths"] == [1024, 2048, 4096, 8192, 16384, 32768]
    assert "niah_turns_32k" not in rows["long32k/main"]["benchmark"]
    assert "niah_single_1" not in rows["long32k/niah"]["benchmark"]
    for power, label in enumerate(("32k", "64k", "128k", "256k", "512k", "1m")):
        row = rows[f"long{label}/main"]
        assert row["lengths"] == [32768 << power] and row["limit"] == 100
        assert row["config_overrides"]["block_size"] == (32768 << power) + 1024
        assert row["config_overrides"].get("rope_adjustments", {}).get("factor", 1) == 2**power
        assert ("infinitebench" in row["benchmark"]) == (power <= 3)
        assert ("ruler_qa_squad" in rows[f"long{label}/niah"]["benchmark"]) == (power <= 2)


def test_reference_harness_requests_and_fixed_storage(checkpoint, monkeypatch):
    from chunkkv._vendor.niahlogkv import eval as ref
    from chunkkv.semanticlogkv_eval import compression_snapshot, make_chunkkv_lm

    cls = make_chunkkv_lm(ref, options(budget="logkv", streaming=True, prefill_chunk_size=12))
    common = dict(device="cpu", config_overrides={"block_size": 128}, tokenizer_dir=str(checkpoint), **LOGKV)
    lm = cls(str(checkpoint), **common)
    oracle = ref.LogKVLM(str(checkpoint), **common)
    oracle._set_eval_cache()
    target = oracle.cache_budget["persistent_cache_bytes"]
    budget = lm.cache_budget
    assert 0 <= target - budget["persistent_cache_bytes"] < budget["bytes_per_entry"]
    assert budget["reference_logkv_bytes"] == target
    assert budget["allocated_fraction_of_dense"] == budget["persistent_cache_bytes"] / budget["dense_cache_bytes"]

    def storage():
        return {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for layer in lm.model.attentions
                for t in (layer.keys, layer.values, layer.positions)}

    initial = storage()
    assert sum(initial.values()) == budget["persistent_cache_bytes"]
    monkeypatch.setattr(ref, "cache_compression_snapshot", compression_snapshot)
    generation, choices = requests()
    assert len(lm.generate_until([generation])) == 1
    assert len(lm.loglikelihood(choices)) == 2
    separator = "<|turn|>"
    response = lm._generate_turns(separator.join([words(55), words(5), words(4)]), separator, 2, "test")
    assert len(response.split(separator)) == 2
    assert len(lm.loglikelihood_rolling([type(generation)(args=(words(130),))])) == 1
    assert lm.compression_metrics[-1]["processed_tokens"] > 0
    assert storage() == initial


def test_reference_tasks_are_registered():
    from chunkkv._vendor.niahlogkv import eval as ref
    tm = ref._task_manager({"max_seq_lengths": [1048576]})
    for name in ["niah_turns_1m", "longbench2_1m", "long_ppl_books_1m", "babilong_1m",
                 "infinitebench_256k", "niah_single_2_words_qend", "ruler_qa_hotpot"]:
        assert name in tm.all_tasks

@pytest.mark.parametrize("size", ["1.7b", "8b"])
@pytest.mark.parametrize("backend", ["chunkkv", "logkv", "dense"])
def test_checkpoint_defaults_and_override(size, backend, tmp_path):
    command = [sys.executable, "-m", "chunkkv.niah", "--print-config", "--backend", backend, "--model-size", size]
    cfg = json.loads(subprocess.check_output(command))
    source = cfg["reference_config"]
    assert cfg["tokenizer"] == source["ckpt_dir"].rstrip("/")
    assert cfg["checkpoint"] == (cfg["tokenizer"] if backend == "chunkkv" else source["save_path"])
    assert cfg["weight_role"] == ("base" if backend == "chunkkv" else "cpt")
    override = json.loads(subprocess.check_output(command + ["--model", str(tmp_path)]))
    assert (override["checkpoint"], override["weight_role"], override["tokenizer"]) == (
        str(tmp_path), "explicit", cfg["tokenizer"])


def test_default_base_vs_cpt_comparison_preserves_protocol_checks():
    contract = dict.fromkeys(CONTRACT_FIELDS, "shared")
    contract.update(protocol="niahLogKV", effective_model_config={"n_layer": 28},
                    evaluated_model={"weight_role": "base"})
    row = dict(doc_id=1, doc_hash="doc", prompt_hash="prompt", target_hash="target", arguments=["question"])
    base = dict(model_backend="chunkkv", comparison_contract=contract, results={"samples": {"test": [row]}})
    cpt = copy.deepcopy(base)
    cpt["model_backend"] = "logkv"
    cpt["comparison_contract"].update(source_weight_sha256="cpt", source_config_files={"yaml": "cpt"},
                                      evaluated_model={"weight_role": "cpt"})
    assert compare_results(base, cpt) == {"test": 1}
    cpt["comparison_contract"]["effective_model_config"]["n_layer"] = 36
    with pytest.raises(ValueError, match="effective_model_config"):
        compare_results(base, cpt)
    base["model_backend"] = "dense"
    contract["evaluated_model"]["weight_role"] = "cpt"
    with pytest.raises(ValueError, match="source_weight_sha256"):
        compare_results(base, cpt)
