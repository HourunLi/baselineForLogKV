"""CLI configuration, result files and a CPU end-to-end run through lm-eval."""

import json
import os
import textwrap

import pytest
import yaml

from h2okv import compare_results, evaluate
from tests.helpers import save_litgpt_checkpoint, save_word_tokenizer, tiny_model


def test_yaml_config_is_overridden_by_cli_and_rejects_unknown_keys(tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump({"model": "/base/", "max-capacity-prompt": 512, "limit": 4,
                                      "tasks": "niah_single_2"}))
    args = evaluate.parse_args(["--config", str(config), "--max-capacity-prompt", "256"])
    assert args.max_capacity_prompt == 256 and args.limit == 4 and args.tasks == "niah_single_2"
    assert args.tokenizer_dir == "/base/" and args.ruler_pretrained == "/base/"  # spelling preserved
    assert args.ruler_lengths == [32768]
    settings = evaluate.evaluation_settings(args)
    assert settings["metadata"] == {"pretrained": "/base/", "max_seq_lengths": [32768]}
    assert settings["method_config"]["max_capacity_prompt"] == 256
    assert settings["method_config"]["decode_evict"] is False
    assert evaluate.parse_args(["--decode-evict"]).decode_evict is True
    config.write_text(yaml.safe_dump({"max_capacity": 1}))
    with pytest.raises(SystemExit):
        evaluate.parse_args(["--config", str(config)])


def test_budget_can_be_matched_to_a_reference_result(tmp_path):
    reference = tmp_path / "semantic.json"
    reference.write_text(json.dumps({"kv_compression": {
        "summary": {"retained_slots_per_layer_group": 999.0},
        "by_request_type": {"generate_until": {"retained_slots_per_layer_group": 2731.6}}}}))
    args = evaluate.parse_args(["--budget-from-result", str(reference)])
    assert args.max_capacity_prompt == 2732


def test_compare_rejects_different_prompts_but_ignores_answers():
    def result(prompt, answer):
        return {"results": {"samples": {"niah": [dict(doc_id=0, doc_hash="d", prompt_hash=prompt, target_hash="t",
                                                      arguments=[["p", {"max_gen_toks": 128}]], resps=[answer])]}}}
    assert compare_results.verify_same_inputs([result("a", "x"), result("a", "y")]) == {"niah": 1}
    with pytest.raises(ValueError, match="differ"):
        compare_results.verify_same_inputs([result("a", "x"), result("b", "x")])


@pytest.fixture
def tiny_run(tmp_path):
    pytest.importorskip("lm_eval")
    saved_environ = dict(os.environ)  # the CLI exports HF cache variables
    base = tmp_path / "Qwen3-tiny-Base"
    save_litgpt_checkpoint(tiny_model(max_seq_length=64), base)
    words = save_word_tokenizer(base)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    gen_rows, choice_rows = [], []
    for doc in range(4):
        context = " ".join(words[2 + (doc * 7 + i) % 60] for i in range(40))
        gen_rows.append({"input": context, "target": words[5]})
        choice_rows.append({"input": context, "choices": [words[3], words[4] + " " + words[6]], "label": doc % 2})
    for name, rows in (("gen", gen_rows), ("choice", choice_rows)):
        (tasks / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (tasks / "utils.py").write_text(textwrap.dedent("""
        import json
        from pathlib import Path

        import datasets


        def make_dataset(**kwargs):  # RULER-style generator: needs the forwarded metadata
            assert kwargs["max_seq_lengths"] == [64] and kwargs["pretrained"], kwargs
            rows = [json.loads(line) for line in (Path(__file__).parent / "gen.jsonl").read_text().splitlines()]
            return {"test": datasets.Dataset.from_list(rows)}


        def process_results(doc, results):
            return {"match": float(results[0].strip() == doc["target"])}
    """))
    common = f"dataset_path: json\ntest_split: test\nmetadata:\n  version: 1.0\n"
    (tasks / "tiny_gen.yaml").write_text(textwrap.dedent("""
        task: tiny_gen
        dataset_path: ""
        dataset_name: ""
        test_split: test
        custom_dataset: !function utils.make_dataset
        metadata:
          version: 1.0
        output_type: generate_until
        doc_to_text: "{{input}}"
        doc_to_target: "{{target}}"
        process_results: !function utils.process_results
        generation_kwargs:
          max_gen_toks: 4
          do_sample: false
          until: []
        metric_list:
          - metric: match
            aggregation: mean
            higher_is_better: true
    """))
    (tasks / "tiny_choice.yaml").write_text(common + textwrap.dedent(f"""
        task: tiny_choice
        dataset_kwargs:
          data_files:
            test: {tasks / 'choice.jsonl'}
        output_type: multiple_choice
        doc_to_text: "{{{{input}}}}"
        doc_to_choice: choices
        doc_to_target: label
        metric_list:
          - metric: acc
            aggregation: mean
            higher_is_better: true
    """))

    def run(*extra):
        output = tmp_path / ("out-" + "-".join(extra).replace("--", ""))
        evaluate.main(["--model", str(base), "--tasks", "tiny_gen,tiny_choice", "--include-path", str(tasks),
                       "--max-seq-length", "64", "--device", "cpu", "--dtype", "float32",
                       "--hf-cache-base", str(tmp_path / "hf"), "--output", str(output), *extra])
        result_file, = output.glob("eval_results_*.json")
        return result_file, json.loads(result_file.read_text())
    yield run
    os.environ.clear()
    os.environ.update(saved_environ)


def test_end_to_end_h2o_and_fullkv_share_inputs(tiny_run, capsys):
    h2o_file, h2o = tiny_run("--method", "h2o", "--max-capacity-prompt", "12", "--window-size", "4")
    evict_file, evict = tiny_run("--method", "h2o", "--max-capacity-prompt", "12", "--window-size", "4",
                                 "--decode-evict")
    full_file, full = tiny_run("--method", "fullkv")
    for result in (h2o, evict, full):
        assert set(result["results"]["results"]) == {"tiny_gen", "tiny_choice"}
        assert {"summary", "by_request_type", "samples"} <= set(result["kv_compression"])
        assert result["inference_metrics"]["generation"]["requests"] == 4
        assert result["inference_metrics"]["scoring"]["requests"] == 8  # two choices x four docs
        assert result["provenance"]["weights_sha256"]["lit_model.pth"]
        assert result["metadata"]["max_seq_lengths"] == [64]
    assert h2o["model_backend"] == "h2o" and h2o["method_config"]["max_capacity_prompt"] == 12
    generation = {name: result["kv_compression"]["by_request_type"]["generate_until"]
                  for name, result in (("h2o", h2o), ("evict", evict))}
    assert generation["h2o"]["retained_slots_per_layer_group"] == 12 + 3  # budget + generated - 1
    assert generation["evict"]["retained_slots_per_layer_group"] == 12  # constant budget
    assert full["kv_compression"]["summary"]["kv_retention_ratio"] == 1
    assert h2o_file.with_suffix(".kv_compression.csv").is_file()
    assert (h2o_file.parent / "resolved_eval.yaml").is_file()
    compare_results.main([str(full_file), str(h2o_file), str(evict_file)])
    out = capsys.readouterr().out
    assert "Verified identical test inputs" in out and '"tiny_gen": 4' in out and "h2o@12" in out
