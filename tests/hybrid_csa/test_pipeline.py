import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hybrid_csa import evaluate, pipeline, train
from hybrid_csa.config import expand_env_vars


@pytest.fixture
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("HYBRID_CSA_", "BASELINE_", "MA_", "MASTER_", "EVAL_MASTER_", "NIAH_MASTER_",
                            "PREPARE_MASTER_", "VC_", "GROUP_", "HF_", "HUGGINGFACE_", "RULER_", "NLTK_")) \
                or name in ("NNODES", "NODE_RANK", "NPROC_PER_NODE", "LOCAL_WORLD_SIZE", "WORLD_SIZE", "RANK"):
            monkeypatch.delenv(name)


@pytest.fixture
def launcher(tmp_path, clean_env, monkeypatch):
    calls = tmp_path / "calls.jsonl"
    source = f"""#!{sys.executable}
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
phase = ('verify_adapter' if '--adapter' in a else 'preflight_' + a[a.index('--phase') + 1]) if 'hybrid_csa.preflight' in a else (
    'prepare' if '--prepare-distributed' in a else (
    a[a.index('--suite') + 1] if '--suite' in a else 'train'))
with open(os.environ['TEST_CALLS'], 'a') as f:
    f.write(json.dumps(dict(phase=phase, python=Path(sys.argv[0]).name, args=a,
                           world=os.environ.get('LITGPT_EXPECTED_WORLD_SIZE'),
                           base=os.environ.get('HYBRID_CSA_BASE_CHECKPOINT'))) + '\\n')
if phase == os.environ.get('TEST_FAIL'):
    sys.exit(7)
if phase.startswith('preflight') or phase == 'verify_adapter':
    print(json.dumps(dict(visible_gpus=2, adapter='validated' if phase == 'verify_adapter' else None)))
if phase == 'train' and not os.environ.get('TEST_NO_ARTIFACT'):
    p = Path(a[a.index('--output-dir')+1])
    p.mkdir(parents=True)
    (p / 'sparse.pt').write_text('test artifact')
"""
    # One stand-in for torch218: training and evaluation run in the same interpreter.
    executable = tmp_path / "torch218-python"
    executable.write_text(source)
    executable.chmod(0o755)
    monkeypatch.setenv("HYBRID_CSA_PYTHON", str(executable))
    monkeypatch.setenv("BASELINE_PYTHON", sys.executable)
    monkeypatch.setenv("TEST_CALLS", str(calls))
    monkeypatch.setenv("MA_NUM_GPUS", "2")
    monkeypatch.setenv("MA_NUM_HOSTS", "2")
    monkeypatch.setenv("VC_TASK_INDEX", "1")
    monkeypatch.setenv("MASTER_ADDR", "worker0")
    output = tmp_path / "run"

    def launch(*args):
        result = subprocess.run(["bash", str(pipeline.BASELINE_ROOT / "majob.sh"), "--output", str(output), *args],
                                cwd=tmp_path, capture_output=True, text=True)
        return result, [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    return launch, output


@pytest.mark.parametrize("fail", [None, "preflight_eval", "prepare", "train", "verify_adapter", "main"])
def test_integrated_launch_and_failure_boundaries(launcher, monkeypatch, fail):
    launch, output = launcher
    if fail:
        monkeypatch.setenv("TEST_FAIL", fail)
    result, calls = launch("--limit", "4")
    assert result.returncode == (7 if fail else 0), result.stderr
    expected = ["preflight_train", "preflight_eval", "prepare", "train", "verify_adapter", "main", "niah"]
    if fail:
        expected = expected[:expected.index(fail) + 1]
    assert [c["phase"] for c in calls] == expected
    for call in calls:
        phase = call["phase"]
        assert call["python"] == "torch218-python"
        if phase in ("prepare", "train", "main", "niah"):
            args = call["args"]
            assert args[args.index("--node_rank") + 1] == "1"
            assert args[args.index("--nproc_per_node") + 1] == ("1" if phase == "prepare" else "2")
            assert args[args.index("--master_port") + 1] == str({"prepare": 6003, "train": 6000, "main": 6001, "niah": 6002}[phase])
            assert call["world"] == "4"
        if phase in ("main", "niah"):
            assert call["args"][call["args"].index("--adapter") + 1] == str(output / "train/sparse.pt")
            assert call["args"][call["args"].index("--output") + 1] == str(output / "eval")
    if fail != "preflight_eval":
        record = json.loads((output / "pipeline_node_1.json").read_text())
        assert record["status"] == ("failed" if fail else "complete")
    else:
        assert not output.exists()


@pytest.mark.parametrize("variable", ["HYBRID_CSA_TRAIN_PYTHON", "BASELINE_TRAIN_PYTHON"])
def test_former_training_interpreter_is_refused_not_ignored(launcher, monkeypatch, variable, tmp_path):
    monkeypatch.setenv(variable, "/some/training-env/bin/python")
    result, calls = launcher[0]()
    assert result.returncode != 0 and f"unset {variable}" in result.stderr and not calls
    result, calls = launcher[0]("--train-python", "/some/training-env/bin/python")
    assert result.returncode != 0 and not calls
    train_script = pipeline.BASELINE_ROOT / "scripts/hybrid_csa_train.sh"
    result = subprocess.run(["bash", str(train_script), "--print-config"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2 and f"unset {variable}" in result.stderr


def test_missing_artifact_never_falls_back_to_base(launcher, monkeypatch):
    monkeypatch.setenv("TEST_NO_ARTIFACT", "1")
    result, calls = launcher[0]()
    assert result.returncode != 0 and "No completed sparse adapter" in result.stderr
    assert calls[-1]["phase"] == "train"


def test_dense_is_eval_only_and_overwrites_the_previous_run(launcher):
    launch, output = launcher
    result, calls = launch("--backend", "dense")
    assert result.returncode == 0, result.stderr
    assert [c["phase"] for c in calls] == ["preflight_eval", "main", "niah"]
    assert all("--adapter" not in c["args"] for c in calls)
    result, _ = launch("--backend", "dense")
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "pipeline_node_1.json").read_text())["status"] == "complete"


def test_rerun_deletes_the_earlier_results_but_nothing_else(launcher, monkeypatch, tmp_path):
    launch, output = launcher
    monkeypatch.setenv("VC_TASK_INDEX", "0")  # node 0 clears the shared result directories
    assert launch()[0].returncode == 0
    for stale in ("eval/main/old_result.json", "main/old_result.json", "train/old.pt"):
        (output / stale).parent.mkdir(parents=True, exist_ok=True)
        (output / stale).write_text("earlier run")
    (output / "pipeline_node_1.json").write_text("node 1 record")  # node 1 replaces its own
    (output / "pipeline_node_5.json").write_text("a node this run does not have")
    (output / "notes.txt").write_text("kept")
    result, _ = launch()
    assert result.returncode == 0, result.stderr
    assert not any((output / stale).exists() for stale in
                   ("eval/main/old_result.json", "main/old_result.json", "train/old.pt", "pipeline_node_5.json"))
    assert (output / "train/sparse.pt").is_file() and (output / "notes.txt").read_text() == "kept"
    assert (output / "pipeline_node_1.json").read_text() == "node 1 record"
    # An eval-only run keeps train/ and never deletes the adapter it is about to evaluate.
    result, _ = launch("--mode", "eval", "--adapter", str(output / "train/sparse.pt"))
    assert result.returncode == 0, result.stderr
    assert (output / "train/sparse.pt").is_file()
    (output / "eval").mkdir()
    (output / "eval/sparse.pt").write_text("adapter")
    result, _ = launch("--mode", "eval", "--adapter", str(output / "eval/sparse.pt"))
    assert result.returncode != 0 and "move it first" in result.stderr and (output / "eval/sparse.pt").is_file()


def test_other_nodes_leave_shared_results_to_node_zero(launcher):
    launch, output = launcher  # the fixture runs node 1 of 2
    (output / "main").mkdir(parents=True)
    (output / "main/from_node_0.json").write_text("node 0 may have written this already")
    result, _ = launch("--backend", "dense")
    assert result.returncode == 0, result.stderr
    assert (output / "main/from_node_0.json").is_file()


def test_preview_never_imports_model_libraries_or_creates_output(tmp_path, clean_env):
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'lightning', 'litdata', 'datasets', 'lm_eval', 'litgpt'}:
        raise AssertionError('Runtime imported during preview: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from hybrid_csa.pipeline import main
main()
"""
    output = tmp_path / "preview"
    result = subprocess.run([sys.executable, "-c", code, "--print-config", "--output", str(output)],
                            cwd=pipeline.BASELINE_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["mode"] == "all" and len(plan["suites"]["main"]) == 47 and len(plan["suites"]["niah"]) == 3
    assert not output.exists()


def test_plain_majob_writes_to_the_configured_directory(clean_env, monkeypatch, tmp_path):
    assert pipeline.configured_output() == Path("/home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/hybrid-csa4")
    config = pipeline.BASELINE_ROOT / "configs/hybrid_csa/test_majob_output.yaml"
    config.write_text(f"output: {tmp_path / 'ckpt'}\n")
    try:
        monkeypatch.setattr(pipeline, "JOB_CONFIG", config)
        monkeypatch.setenv("MA_VJ_NAME", "job-42")  # a platform job name does not change the directory
        monkeypatch.setenv("MA_TASK_NAME", "worker")
        plan = pipeline.build_plan(pipeline.parse_args([]))
        assert plan["output"] == str(tmp_path / "ckpt")
        assert plan["training"]["output_dir"] == str(tmp_path / "ckpt/train")
        assert (plan["eval_output"], plan["adapter"]) == (str(tmp_path / "ckpt/eval"), str(tmp_path / "ckpt/train/sparse.pt"))
        assert pipeline.build_plan(pipeline.parse_args(["--output", str(tmp_path / "x")]))["output"] == str(tmp_path / "x")
        config.write_text("output: relative/path\n")
        with pytest.raises(ValueError, match="absolute output"):
            pipeline.build_plan(pipeline.parse_args([]))
    finally:
        config.unlink()


def test_topology_defaults_and_precedence(clean_env, monkeypatch):
    assert pipeline.topology(3)["gpus"] == 3
    monkeypatch.setenv("MA_NUM_GPUS", "4")
    with pytest.raises(ValueError, match="only 3"):
        pipeline.topology(3)
    monkeypatch.setenv("MA_NUM_GPUS", "2")
    monkeypatch.setenv("MA_NUM_HOSTS", "2")
    with pytest.raises(ValueError, match="MASTER_ADDR"):
        pipeline.topology(3)
    monkeypatch.setenv("MA_VJ_NAME", "job")
    monkeypatch.setenv("MA_TASK_NAME", "worker")
    assert pipeline.topology(3)["address"] == "job-worker-0.job"
    monkeypatch.setenv("MASTER_ADDR", "explicit")
    assert pipeline.topology(3)["address"] == "explicit"
    monkeypatch.setenv("NIAH_MASTER_PORT", "6001")
    with pytest.raises(ValueError, match="distinct"):
        pipeline.topology(3)


def test_explicit_cache_children_and_required_env(clean_env, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/cache/parent")
    monkeypatch.setenv("HYBRID_CSA_HF_CACHE", "/cache/method")
    monkeypatch.setenv("HF_DATASETS_CACHE", "/cache/datasets-custom")
    monkeypatch.setenv("HF_HUB_CACHE", "/cache/hub-custom")
    evaluate.configure_environment()
    assert os.environ["HF_HOME"] == "/cache/method"
    assert os.environ["HF_DATASETS_CACHE"] == "/cache/datasets-custom"
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == "/cache/hub-custom"
    assert os.environ["HF_MODULES_CACHE"] == "/cache/method/modules"
    with pytest.raises(ValueError, match="unset"):
        expand_env_vars("${CSA_TEST_MISSING_VARIABLE}")
    monkeypatch.setenv("CSA_EMPTY", "")
    assert expand_env_vars("${CSA_EMPTY:-fallback}/${CSA_EMPTY-fallback}") == "fallback/"


@pytest.mark.parametrize("versions,allowed", [
    ({"lm-eval": "0.4.11", "datasets": "4.8.4", "pyarrow": "23.0.1"}, True),
    ({"lm-eval": "0.4.9", "datasets": "4.8.4", "pyarrow": "23.0.1"}, False),
    ({"lm-eval": "0.4.11", "datasets": "3.6.0", "pyarrow": "23.0.1"}, False),
    ({"lm-eval": "0.4.11", "datasets": "4.8.4", "pyarrow": "20.0.0"}, False),
])
def test_current_evaluation_runtime(versions, allowed, monkeypatch):
    monkeypatch.setattr(evaluate.importlib.metadata, "version", versions.__getitem__)
    if allowed:
        evaluate.check_environment()
    else:
        with pytest.raises(RuntimeError, match="without changing its packages"):
            evaluate.check_environment()


def test_prepare_reuses_once_and_propagates_errors(tmp_path, clean_env, monkeypatch):
    directory = tmp_path / "tokens"
    cfg = {"tokenized_dir": str(directory)}
    calls = []
    monkeypatch.setattr(train, "validate_prepared_data", lambda cfg, **kw: calls.append(("validate", kw)))
    def prepare(cfg):
        calls.append(("prepare", {}))
        directory.mkdir()
        (directory / train.DATA_MANIFEST).write_text("{}")
    monkeypatch.setattr(train, "prepare_data", prepare)
    train.prepare_distributed(cfg)
    train.prepare_distributed(cfg)
    assert [name for name, _ in calls].count("prepare") == 1
    assert ("validate", {"verify_source": True}) in calls
    def fail(cfg, **kw):
        raise ValueError("corpus changed")
    monkeypatch.setattr(train, "validate_prepared_data", fail)
    with pytest.raises(RuntimeError, match="corpus changed"):
        train.prepare_distributed(cfg)
