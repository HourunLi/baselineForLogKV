"""Launcher contract: shared inputs, budget options, output isolation and result checks."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
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


def test_reference_suites_share_task_inputs_and_niah_lengths(tmp_path):
    evaluator = fake_evaluator(tmp_path, {"ckpt_dir": str(tmp_path / "base"), "context_length": 32768})
    main_tasks, _ = runner.load_eval_suite(runner.DEFAULT_EVALUATION_CONFIG, "main")
    long_tasks, _ = runner.load_eval_suite(runner.DEFAULT_EVALUATION_CONFIG, "longbench")
    assert len(main_tasks.split(",")) == 47
    assert len(long_tasks.split(",")) == 33
    assert long_tasks.split(",") == [task for task in main_tasks.split(",") if task.startswith("longbench_")]
    for suite in ("main", "longbench", "niah"):
        tasks, lengths = runner.load_eval_suite(runner.DEFAULT_EVALUATION_CONFIG, suite)
        records = [runner.build_eval_kwargs(
            evaluator, tmp_path / "train.yaml", tmp_path / "cpt" if backend == "logkv" else None,
            backend, tasks, 4, tmp_path / suite, {"budget": 4096}, max_seq_lengths=lengths,
        ) for backend in runner.BACKENDS]
        for key in ("benchmark", "metadata", "limit", "tokenizer_dir", "config_overrides"):
            assert records[0][key] == records[1][key] == records[2][key]
        if suite == "niah":
            assert tasks == "niah_single_1,niah_single_2,niah_single_3"
            assert records[0]["metadata"]["max_seq_lengths"] == [32768]
        else:
            assert records[0]["metadata"] is None  # main phase matches the reference eval.sh
    with pytest.raises(ValueError, match="context_length"):
        runner.build_eval_kwargs(evaluator, tmp_path / "train.yaml", None, "chunkkv", "niah_single_1",
                                 4, tmp_path, {"budget": 4096}, max_seq_lengths=[65536])


@pytest.mark.parametrize("fail_main", [False, True])
def test_full_suite_launcher_runs_separate_processes_and_stops_on_failure(tmp_path, fail_main):
    executable = tmp_path / "torchrun"
    log = tmp_path / "calls"
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CHUNKKV_TEST_CALLS"\n'
                          'if [[ "$CHUNKKV_TEST_FAIL" = 1 && "$*" = *"--suite main"* ]]; then exit 7; fi\n')
    executable.chmod(0o755)
    result = subprocess.run(
        ["bash", str(runner.BASELINE_ROOT / "scripts/chunkkv_evaluate_all.sh"),
         "--backend", "dense", "--limit", "4", "--output", str(tmp_path / "results")],
        cwd=tmp_path, env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
                          "CHUNKKV_TEST_CALLS": str(log), "CHUNKKV_TEST_FAIL": str(int(fail_main))},
        capture_output=True, text=True,
    )
    assert result.returncode == (7 if fail_main else 0), result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == (1 if fail_main else 2)
    assert "--suite main" in calls[0]
    if not fail_main:
        assert "--suite niah" in calls[1]
    assert all("--backend dense --limit 4 --output" in call for call in calls)


def platform_launcher_env(tmp_path, visible_gpus=4):
    """A fake interpreter for majob.sh: answers the CUDA device probe and logs every launch."""
    executable = tmp_path / "python"
    executable.write_text('#!/bin/bash\nif [[ "$1" = -c ]]; then\n'
                          '  [[ "$CHUNKKV_TEST_GPUS" = fail ]] && { echo "CUDA initialisation failed" >&2; exit 1; }\n'
                          '  echo "$CHUNKKV_TEST_GPUS"; exit\nfi\n'
                          'printf "%s | %s | %s\\n" "$LITGPT_EXPECTED_WORLD_SIZE" "$*" "$LD_LIBRARY_PATH" >> "$CHUNKKV_TEST_CALLS"\n'
                          'if [[ "$CHUNKKV_TEST_FAIL" = 1 && "$*" = *"--suite main"* ]]; then exit 7; fi\n')
    executable.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("MA_", "MASTER_", "EVAL_MASTER_", "NIAH_MASTER_", "VC_", "GROUP_"))
           and key not in {"NPROC_PER_NODE", "LOCAL_WORLD_SIZE", "NODE_RANK"}}
    env.update(CHUNKKV_PYTHON=str(executable), CHUNKKV_TEST_CALLS=str(tmp_path / "calls"),
               CHUNKKV_TEST_GPUS=str(visible_gpus), CHUNKKV_TEST_FAIL="0")
    return env


@pytest.mark.parametrize("requested,visible,launched", [
    ({}, 3, 3),  # a development container without platform variables uses what it has
    ({"NPROC_PER_NODE": "2"}, 3, 2),
    ({"MA_NUM_GPUS": "8"}, 3, None),
    ({}, 0, None),
    ({}, "fail", None),  # e.g. a libcuda older than torch's CUDA: stop before torchrun
])
def test_platform_launcher_defaults_to_the_visible_gpus(tmp_path, requested, visible, launched):
    env = {**platform_launcher_env(tmp_path, visible), **requested}
    command = ["bash", str(runner.BASELINE_ROOT / "majob.sh"), "--limit", "4", "--output", "outputs/test"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    if launched is None:
        expected = (1, "CUDA initialisation failed") if visible == "fail" else (2, "visible")
        assert (result.returncode, expected[1] in result.stderr) == (expected[0], True), result.stderr
        assert not (tmp_path / "calls").exists()
        return
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert len(calls) == 2
    assert all(call.startswith(f"{launched} | ") and f"--nproc_per_node {launched} " in call for call in calls)
    # SemanticLogKV's platform library order, ahead of whatever the job inherited
    inherited = env.get("LD_LIBRARY_PATH")
    expected = "/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/lib64" + (f":{inherited}" if inherited else "")
    assert all(call.endswith(f" | {expected}") for call in calls)


@pytest.mark.parametrize("nodes,rank,fail_main", [(1, 0, False), (2, 1, False), (2, 0, True)])
def test_platform_launcher_maps_ranks_ports_and_stops_on_failure(tmp_path, nodes, rank, fail_main):
    env = platform_launcher_env(tmp_path)
    log = tmp_path / "calls"
    env.update(CHUNKKV_TEST_FAIL=str(int(fail_main)), MA_NUM_GPUS="4", MA_NUM_HOSTS=str(nodes),
               VC_TASK_INDEX=str(rank), MA_VJ_NAME="job", MA_TASK_NAME="worker", MA_MASTER_INDEX="0")
    command = ["bash", str(runner.BASELINE_ROOT / "majob.sh"),
               "configs/chunkkv/experiment_qwen1.7b_32k.yaml", "--limit", "4", "--output", "outputs/test"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == (7 if fail_main else 0), result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == (1 if fail_main else 2)
    for call in calls:
        assert call.startswith(f"{nodes * 4} | -m torch.distributed.run")
        assert "--nproc_per_node 4" in call
        assert "-m chunkkv.evaluate" in call
        assert "--limit 4 --output outputs/test" in call
        if nodes == 1:  # fixed ports can be taken on the node; torchrun binds a free one
            assert "--standalone" in call and "--master_" not in call and "--nnodes" not in call
        else:
            assert f"--nnodes {nodes} --node_rank {rank}" in call
            assert "--master_addr job-worker-0.job" in call
            assert "--standalone" not in call
    assert "--suite main" in calls[0]
    if not fail_main:
        assert "--suite niah" in calls[1]
    if nodes > 1:
        assert "--master_port 6001" in calls[0]
        if not fail_main:
            assert "--master_port 6002" in calls[1]
    log.unlink()
    for extra in (["--tasks", "boolq"], ["--evaluation-config=other.yaml"], ["--experiment-config", "x.yaml"]):
        result = subprocess.run(command + extra, cwd=tmp_path, env=env, capture_output=True, text=True)
        assert result.returncode == 2 and "not allowed" in result.stderr
    assert not log.exists()
    env.pop("MA_VJ_NAME")
    env.update(MA_NUM_HOSTS="2", MASTER_ADDR="localhost")
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "MASTER_ADDR" in result.stderr
    assert not log.exists()


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


@pytest.fixture
def datasets_cache(monkeypatch):
    state = SimpleNamespace(enabled=True)
    module = SimpleNamespace(
        is_caching_enabled=lambda: state.enabled,
        disable_caching=lambda: setattr(state, "enabled", False),
        enable_caching=lambda: setattr(state, "enabled", True),
    )
    monkeypatch.setitem(sys.modules, "datasets", module)
    return state


def test_outputs_stay_outside_reference_and_results_are_relabelled(tmp_path, monkeypatch, datasets_cache):
    monkeypatch.setenv("RANK", "0")
    installed = []
    monkeypatch.setattr(runner, "install_chunkkv_adapter", lambda evaluator, options: installed.append(options))
    reference, output = tmp_path / "runtime", tmp_path / "baseline/results"
    reference.mkdir()
    entrypoint = reference / "eval.py"
    entrypoint.write_text("unchanged source")

    def main(**kwargs):
        assert Path.cwd() == output
        assert not datasets_cache.enabled
        Path("eval_results_cache.json").write_text("cached")
        Path("eval_results_20261013_000000.json").write_text(json.dumps({
            "log_kv_B": 128, "log_kv_dense_mode": False, "swa_window_size": None,
            "cache_budget": {"method": "ChunkKV"}, "results": {"samples": {"example": []}},
        }))
    evaluator = SimpleNamespace(main=main, __file__=str(entrypoint))
    kwargs = dict(output_path=str(output), tokenizer_dir="base", config_overrides={"block_size": 32768})
    cwd = Path.cwd()
    runner.run_evaluation(evaluator, kwargs, "chunkkv", {"example": "source"}, {"budget": 64},
                          resolved={"launch_id": "20261013_000000"})
    assert installed == [{"budget": 64}] and Path.cwd() == cwd
    assert datasets_cache.enabled
    assert entrypoint.read_text() == "unchanged source" and list(reference.iterdir()) == [entrypoint]
    result = json.loads((output / "eval_results_20261013_000000.json").read_text())
    assert result["model_backend"] == "chunkkv" and result["comparison_contract"] == {"example": "source"}
    assert not any(key.startswith("log_kv_") for key in result) and "swa_window_size" not in result
    assert result["cache_budget"] == {"method": "ChunkKV"}
    assert result["resolved_eval"] == {"launch_id": "20261013_000000"}
    # A failed second run must not relabel existing results with a new contract.
    evaluator.main = lambda **kwargs: None
    with pytest.raises(RuntimeError, match="one new evaluation result"):
        runner.run_evaluation(evaluator, kwargs, "chunkkv", {"example": "other"}, {"budget": 64})
    with pytest.raises(ValueError, match="outside"):
        runner.run_evaluation(evaluator, {**kwargs, "output_path": str(reference / "results")}, "dense", {})
    assert not (reference / "results").exists()


@pytest.mark.parametrize("backend", runner.BACKENDS)
@pytest.mark.parametrize("cache_enabled", [True, False])
def test_eval_failure_restores_directory_and_transform_cache_policy(
    tmp_path, monkeypatch, datasets_cache, backend, cache_enabled,
):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setattr(runner, "install_chunkkv_adapter", lambda *args: None)
    datasets_cache.enabled = cache_enabled
    cwd = Path.cwd()

    def fail(**kwargs):
        assert not datasets_cache.enabled
        raise RuntimeError("dataset failed")

    evaluator = SimpleNamespace(main=fail, __file__=str(tmp_path / "runtime/eval.py"))
    with pytest.raises(RuntimeError, match="dataset failed"):
        runner.run_evaluation(evaluator, {"output_path": str(tmp_path / "output")}, backend, {})
    assert datasets_cache.enabled == cache_enabled
    assert Path.cwd() == cwd


def test_cache_entries_from_newer_datasets_point_to_the_cache_check(tmp_path, monkeypatch, datasets_cache):
    monkeypatch.setenv("RANK", "1")

    def fail(**kwargs):
        raise ValueError("Feature type 'List' not found. Available feature types: ['Value', 'Sequence']")

    evaluator = SimpleNamespace(main=fail, __file__=str(tmp_path / "runtime/eval.py"))
    with pytest.raises(RuntimeError, match="chunkkv.check_dataset_cache") as raised:
        runner.run_evaluation(evaluator, {"output_path": str(tmp_path / "output")}, "logkv", {})
    assert "Feature type 'List'" in str(raised.value.__cause__)
    evaluator.main = lambda **kwargs: (_ for _ in ()).throw(ValueError("unrelated"))
    with pytest.raises(ValueError, match="unrelated"):
        runner.run_evaluation(evaluator, {"output_path": str(tmp_path / "output")}, "logkv", {})

    # datasets parses dataset_info.json of every cached entry; a malformed one fails there, unnamed.
    namespace = {}
    exec(compile("import json\ndef find_hash_in_cache():\n    json.loads('{\\n  \"a\": 1,\\n    ')\n",
                 "/site-packages/datasets/packaged_modules/cache/cache.py", "exec"), namespace)
    evaluator.main = lambda **kwargs: namespace["find_hash_in_cache"]()
    with pytest.raises(RuntimeError, match="not valid JSON.*chunkkv.check_dataset_cache") as raised:
        runner.run_evaluation(evaluator, {"output_path": str(tmp_path / "output")}, "logkv", {})
    assert isinstance(raised.value.__cause__, json.JSONDecodeError)
    evaluator.main = lambda **kwargs: json.loads("{")  # not the cache lookup: unchanged
    with pytest.raises(json.JSONDecodeError):
        runner.run_evaluation(evaluator, {"output_path": str(tmp_path / "output")}, "logkv", {})


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


def test_eval_extra_declares_the_requirements_the_launcher_checks():
    pyproject = (Path(runner.__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    extra = next(line for line in pyproject.splitlines() if line.startswith("eval = "))
    assert f'"datasets>={".".join(map(str, runner.EVAL_DATASETS_MIN))}"' in extra
    assert f']>={".".join(map(str, runner.EVAL_LM_EVAL_MIN))}"' in extra
    assert "\"transformers" not in pyproject  # comes with lm-eval at whatever version the environment has


@pytest.mark.parametrize("versions,message", [
    ({"lm-eval": "0.4.11", "datasets": "4.8.4"}, None),  # torch218
    ({"lm-eval": "0.4.9", "datasets": "4.8.4"}, None),
    ({"lm-eval": "0.4.11", "datasets": "5.1.0"}, None),
    ({"lm-eval": "0.4.11", "datasets": "3.6.0"}, "datasets>=4.7"),
    ({"lm-eval": "0.4.11", "datasets": "4.6.1"}, "datasets>=4.7"),
    ({"lm-eval": "0.4.8", "datasets": "4.8.4"}, "lm-eval>=0.4.9"),
])
def test_environment_check_follows_code_and_cache_requirements(monkeypatch, versions, message):
    monkeypatch.setattr(runner.importlib.metadata, "version", versions.__getitem__)
    if message is None:
        runner.check_environment()
    else:
        with pytest.raises(RuntimeError, match=message.replace(".", r"\.")):
            runner.check_environment()
