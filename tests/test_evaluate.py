"""CLI configuration, result files and a CPU end-to-end run through lm-eval."""

import json
import os
from pathlib import Path
import subprocess
import textwrap

import pytest
import yaml

from pyramidkv import compare_results, evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]
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
    config.write_text(yaml.safe_dump({"max_capacity": 1}))
    with pytest.raises(SystemExit):
        evaluate.parse_args(["--config", str(config)])


def test_reference_suites_are_project_owned_and_niah_is_32k_only(tmp_path):
    main = evaluate.parse_args(["--suite", "main", "--output", str(tmp_path / "run")])
    tasks = main.tasks.split(",")
    assert len(tasks) == 47 and len(set(tasks)) == 47
    assert sum(task.startswith("longbench_") and task.endswith("_e") for task in tasks) == 13
    assert sum(task.startswith("longbench_") for task in tasks) == 33
    assert main.metadata is None  # the reference main stage passes no NIAH metadata
    assert main.output == str(tmp_path / "run" / "main")
    longbench = evaluate.parse_args(["--suite", "longbench"])
    assert longbench.tasks.split(",") == [task for task in tasks if task.startswith("longbench_")]
    niah = evaluate.parse_args(["--suite", "niah", "--output", str(tmp_path / "run")])
    assert niah.tasks == "niah_single_1,niah_single_2,niah_single_3"
    assert niah.metadata == {"pretrained": evaluate.DEFAULT_BASE, "max_seq_lengths": [32768]}
    assert niah.output == str(tmp_path / "run" / "niah")
    for argv in ([], ["--suite", "main", "--tasks", "boolq"], ["--suite", "niah", "--ruler-lengths", "4096"],
                 ["--tasks", "niah_single_1", "--ruler-lengths", "65536"]):
        with pytest.raises(SystemExit):
            evaluate.parse_args(argv)


def test_run_ids_never_reuse_existing_result_names(tmp_path):
    first = evaluate.new_run_id(tmp_path)
    (tmp_path / f"eval_results_{first}.json").write_text("{}")
    second = evaluate.new_run_id(tmp_path)
    assert second != first and second.startswith(first)


@pytest.mark.parametrize("nodes,rank,fail_main", [(1, 0, False), (2, 1, False), (2, 0, True)])
def test_platform_launcher_runs_both_suites_and_stops_on_failure(tmp_path, nodes, rank, fail_main):
    executable = tmp_path / "python"
    log = tmp_path / "calls"
    executable.write_text('#!/bin/bash\nprintf "%s | %s\\n" "$KV_BASELINE_EXPECTED_WORLD_SIZE" "$*" >> "$TEST_CALLS"\n'
                          'if [[ "$TEST_FAIL" = 1 && "$*" = *"--suite main"* ]]; then exit 7; fi\n')
    executable.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("MA_", "MASTER_", "MAIN_MASTER_", "NIAH_MASTER_"))}
    env.update(PYRAMIDKV_PYTHON=str(executable), TEST_CALLS=str(log), TEST_FAIL=str(int(fail_main)),
               MA_NUM_GPUS="4", MA_NUM_HOSTS=str(nodes), VC_TASK_INDEX=str(rank),
               MA_VJ_NAME="job", MA_TASK_NAME="worker", MA_MASTER_INDEX="0")
    command = ["bash", str(REPO_ROOT / "majob.sh"), "configs/qwen3_1.7b_32k_pyramidkv.yaml",
               "--limit", "4", "--output", "outputs/test"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == (7 if fail_main else 0), result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == (1 if fail_main else 2)
    for call in calls:
        assert call.startswith(f"{nodes * 4} | -m torch.distributed.run")
        assert f"--nnodes {nodes} --nproc_per_node 4 --node_rank {rank}" in call
        assert "--master_addr job-worker-0.job" in call
        assert "-m pyramidkv.evaluate --config configs/qwen3_1.7b_32k_pyramidkv.yaml" in call
        assert call.endswith("--limit 4 --output outputs/test")
    assert "--master_port 6001" in calls[0] and "--suite main" in calls[0]
    if not fail_main:
        assert "--master_port 6002" in calls[1] and "--suite niah" in calls[1]
    log.unlink()
    for extra in (["--tasks", "boolq"], ["--max-seq-length=65536"], ["--ruler-lengths", "4096"]):
        result = subprocess.run(command + extra, cwd=tmp_path, env=env, capture_output=True, text=True)
        assert result.returncode == 2 and "not allowed" in result.stderr
    env.pop("MA_VJ_NAME")
    env.update(MA_NUM_HOSTS="2", MASTER_ADDR="localhost")
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "MASTER_ADDR" in result.stderr
    assert not log.exists()


def test_budget_can_be_matched_to_a_reference_result(tmp_path):
    reference = tmp_path / "semantic.json"
    reference.write_text(json.dumps({"kv_compression": {
        "summary": {"retained_slots_per_layer_group": 999.0},
        "by_request_type": {"generate_until": {"retained_slots_per_layer_group": 2731.6}}}}))
    args = evaluate.parse_args(["--suite", "niah", "--budget-from-result", str(reference)])
    assert args.max_capacity_prompt == 2732


def test_compare_rejects_different_prompts_or_protocol_but_ignores_answers():
    def result(prompt, answer, limit=4, window=32768):
        return {"max_seq_length": window, "provenance": {"tokenizer_files": {"tokenizer.json": "t"},
                                                         "versions": {"lm-eval": "0.4.9", "datasets": "3.6.0",
                                                                      "pyarrow": "17.0.0"}},
                "results": {"config": {"limit": limit}, "versions": {"niah": 1.0},
                            "configs": {"niah": {"output_type": "generate_until", "metadata": {"x": answer}}},
                            "samples": {"niah": [dict(doc_id=0, doc_hash="d", prompt_hash=prompt, target_hash="t",
                                                      arguments=[["p", {"max_gen_toks": 128}]], resps=[answer])]}}}
    assert compare_results.verify_same_inputs([result("a", "x"), result("a", "y")]) == {"niah": 1}
    with pytest.raises(ValueError, match="differ"):
        compare_results.verify_same_inputs([result("a", "x"), result("b", "x")])
    with pytest.raises(ValueError, match="limit"):
        compare_results.verify_same_inputs([result("a", "x"), result("a", "x", limit=8)])
    with pytest.raises(ValueError, match="max_seq_length"):
        compare_results.verify_same_inputs([result("a", "x"), result("a", "x", window=16384)])
    semantic = result("a", "x")
    del semantic["max_seq_length"], semantic["provenance"]  # SemanticLogKV's own files lack these
    assert compare_results.verify_same_inputs([semantic, result("a", "y")]) == {"niah": 1}


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


        def make_dataset(**kwargs):  # NIAH-style generator: needs the forwarded metadata
            assert kwargs["max_seq_lengths"] == [64] and kwargs["pretrained"], kwargs
            rows = [json.loads(line) for line in (Path(__file__).parent / "gen.jsonl").read_text().splitlines()]
            return {"test": datasets.Dataset.from_list(rows)}


        def process_results(doc, results):
            return {"match": float(results[0].strip() == doc["target"])}
    """))
    common = f"dataset_path: json\ntest_split: test\nmetadata:\n  version: 1.0\n"
    (tasks / "niah_tiny_gen.yaml").write_text(textwrap.dedent("""
        task: niah_tiny_gen
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
        evaluate.main(["--model", str(base), "--tasks", "niah_tiny_gen,tiny_choice", "--include-path", str(tasks),
                       "--max-seq-length", "64", "--device", "cpu", "--dtype", "float32",
                       "--hf-cache-base", str(tmp_path / "hf"), "--output", str(output), *extra])
        result_file, = output.glob("eval_results_*.json")
        return result_file, json.loads(result_file.read_text())
    yield run
    os.environ.clear()
    os.environ.update(saved_environ)


def test_end_to_end_pyramidkv_and_fullkv_share_inputs(tiny_run, capsys):
    pyramid_file, pyramid = tiny_run("--method", "pyramidkv", "--max-capacity-prompt", "12", "--window-size", "4",
                                     "--kernel-size", "3")
    full_file, full = tiny_run("--method", "fullkv")
    for result in (pyramid, full):
        assert set(result["results"]["results"]) == {"niah_tiny_gen", "tiny_choice"}
        assert {"summary", "by_request_type", "samples"} <= set(result["kv_compression"])
        assert result["inference_metrics"]["generation"]["requests"] == 4
        assert result["inference_metrics"]["scoring"]["requests"] == 8  # two choices x four docs
        assert result["provenance"]["weights_sha256"]["lit_model.pth"]
        assert result["metadata"]["max_seq_lengths"] == [64]
        assert result["provenance"]["environment"]["python_executable"]
        assert result["provenance"]["topology"]["world_size"] == 1
        assert result["provenance"]["cache"]["HF_DATASETS_OFFLINE"] == "1"
    assert pyramid["model_backend"] == "pyramidkv" and pyramid["method_config"]["max_capacity_prompt"] == 12
    assert pyramid["kv_compression"]["by_request_type"]["generate_until"]["kv_retention_ratio"] < 0.5
    assert full["kv_compression"]["summary"]["kv_retention_ratio"] == 1
    assert pyramid_file.with_suffix(".kv_compression.csv").is_file()
    assert (pyramid_file.parent / f"resolved_eval_{pyramid['run_id']}.yaml").is_file()
    assert pyramid_file.name == f"eval_results_{pyramid['run_id']}.json"
    compare_results.main([str(full_file), str(pyramid_file)])
    out = capsys.readouterr().out
    assert "Verified identical test inputs" in out and '"niah_tiny_gen": 4' in out and "pyramidkv@12" in out
