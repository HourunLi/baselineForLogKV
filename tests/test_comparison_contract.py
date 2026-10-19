"""Launcher contract: shared inputs, project-owned suites, platform job, output isolation and result checks."""

import ast
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from models.qwen3 import evaluate as runner
from models.qwen3.compare_results import CONTRACT_FIELDS, compare_results
from models.qwen3.config import BASELINE_ROOT, DEFAULT_EVALUATION_CONFIG


def fake_evaluator(tmp_path, config):
    def main(log_kv_B=512, log_kv_semantic_clusters=False):
        pass
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(config))
    return SimpleNamespace(main=main, __file__=str(tmp_path / "runtime" / "eval.py"))


def test_both_backends_share_inputs_but_only_semantic_uses_its_method_flags(tmp_path):
    config = dict(ckpt_dir=str(tmp_path) + "/", context_length=32768, log_kv_B=128,
                  log_kv_semantic_clusters=True, max_steps=1200, log_kv_diag_mode="oracle")
    evaluator = fake_evaluator(tmp_path, config)
    common = (evaluator, tmp_path / "train.yaml", tmp_path / "model")
    dms = runner.build_eval_kwargs(*common, "dms", "niah_single_2,niah_single_3", 4, tmp_path / "out")
    semantic = runner.build_eval_kwargs(*common, "logkv", "niah_single_2,niah_single_3", 4, tmp_path / "out")
    assert dms == {k: v for k, v in semantic.items() if k not in ("log_kv_B", "log_kv_semantic_clusters")}
    assert "model_backend" not in dms and "comparison_contract" not in dms
    assert semantic["log_kv_B"] == 128 and semantic["log_kv_semantic_clusters"]
    # The configured spelling (trailing slash) names the RULER cache, so it is preserved.
    assert dms["metadata"] == {"pretrained": str(tmp_path) + "/", "max_seq_lengths": [32768]}
    assert "max_steps" not in semantic and "log_kv_diag_mode" not in semantic
    config["ckpt_dir"] = "checkpoints/base"
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(config))
    relative = runner.build_eval_kwargs(*common, "dms", "niah_single_2", None, tmp_path / "out")
    assert relative["tokenizer_dir"] == str(BASELINE_ROOT / "checkpoints/base")


def test_reference_suites_share_task_inputs_and_niah_is_32k_only(tmp_path):
    evaluator = fake_evaluator(tmp_path, {"ckpt_dir": str(tmp_path / "base"), "context_length": 32768})
    main_tasks, _ = runner.load_eval_suite(DEFAULT_EVALUATION_CONFIG, "main")
    long_tasks, _ = runner.load_eval_suite(DEFAULT_EVALUATION_CONFIG, "longbench")
    assert len(main_tasks.split(",")) == 47 and len(set(main_tasks.split(","))) == 47
    assert len(long_tasks.split(",")) == 33
    assert sum(task.endswith("_e") for task in long_tasks.split(",")) == 13
    for suite in ("main", "longbench", "niah"):
        tasks, lengths = runner.load_eval_suite(DEFAULT_EVALUATION_CONFIG, suite)
        records = [runner.build_eval_kwargs(evaluator, tmp_path / "train.yaml", tmp_path / backend, backend,
                                            tasks, 4, tmp_path / suite, max_seq_lengths=lengths)
                   for backend in runner.BACKENDS]
        for key in ("benchmark", "metadata", "limit", "tokenizer_dir", "config_overrides"):
            assert records[0][key] == records[1][key]
        if suite == "niah":
            assert tasks == "niah_single_1,niah_single_2,niah_single_3"
            assert records[0]["metadata"]["max_seq_lengths"] == [32768]
        else:
            assert records[0]["metadata"] is None  # main phase matches the reference eval.sh
    with pytest.raises(ValueError, match="context_length"):
        runner.build_eval_kwargs(evaluator, tmp_path / "train.yaml", tmp_path, "dms", "niah_single_1",
                                 4, tmp_path, max_seq_lengths=[65536])


def test_project_experiment_config_keeps_the_base_window_and_no_cpt_training_fields():
    from models.qwen3.config import DEFAULT_EXPERIMENT_CONFIG, load_experiment_config

    config = load_experiment_config(DEFAULT_EXPERIMENT_CONFIG)
    assert config["context_length"] == 32768
    assert config["ckpt_dir"].endswith("Qwen3-1.7B-Base/")
    assert not {"save_path", "data_dir", "max_steps", "global_batch_size"} & set(config)


def test_print_config_reads_only_project_files_and_needs_no_model_runtime(tmp_path):
    script = r'''
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))

class BlockHeavyImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"litgpt", "utils", "torch", "lightning", "lm_eval", "transformers", "datasets"}:
            raise AssertionError(f"Unexpected runtime dependency: {fullname}")

def check_reads(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        if path.suffix in {".yaml", ".yml", ".py"} and "site-packages" not in path.parts:
            assert path.is_relative_to(root) or path.is_relative_to(Path(sys.prefix)) \
                or path.is_relative_to(Path(sys.base_prefix)), f"External file read: {path}"

sys.meta_path.insert(0, BlockHeavyImports())
sys.addaudithook(check_reads)
sys.argv = ["models.qwen3.evaluate", "--print-config", "--model", "/tmp/student", *sys.argv[2:]]
runpy.run_module("models.qwen3.evaluate", run_name="__main__")
'''
    for suite in ("main", "niah"):
        result = subprocess.run([sys.executable, "-c", script, str(BASELINE_ROOT), "--suite", suite],
                                cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        resolved = json.loads(result.stdout)
        assert resolved["config_overrides"] == {"block_size": 32768}
        assert resolved["output_path"].endswith(f"/eval_aligned/{suite}")
    outside = tmp_path / "other.yaml"
    outside.write_text("ckpt_dir: /x\ncontext_length: 32768\n")
    result = subprocess.run([sys.executable, "-m", "models.qwen3.evaluate", "--print-config", "--model", "/tmp/s",
                             "--suite", "niah", "--experiment-config", str(outside)],
                            cwd=BASELINE_ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "inside this project" in result.stderr
    result = subprocess.run([sys.executable, "-m", "models.qwen3.evaluate", "--print-config", "--model", "/tmp/s"],
                            cwd=BASELINE_ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "--suite" in result.stderr


def test_project_code_never_reaches_into_another_checkout():
    violations = []
    for path in sorted((BASELINE_ROOT / "models").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            if any(name.split(".")[0] in {"litgpt", "utils", "data"} for name in names):
                violations.append(f"{path}:{node.lineno}: external import {names}")
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "sys" and node.attr == "path"):
                violations.append(f"{path}:{node.lineno}: sys.path access")
    for path in [*BASELINE_ROOT.glob("*.sh"), *(BASELINE_ROOT / "scripts").glob("*.sh"),
                 *(BASELINE_ROOT / "configs").glob("*.yaml"), BASELINE_ROOT / "pyproject.toml"]:
        if "../litgpt" in path.read_text(encoding="utf-8"):
            violations.append(f"{path}: refers to ../litgpt")
    assert not violations, "\n".join(violations)


def platform_launcher_env(tmp_path, visible_gpus=4):
    """A fake interpreter for majob.sh: answers the CUDA device probe and logs every launch."""
    executable = tmp_path / "python"
    executable.write_text('#!/bin/bash\nif [[ "$1" = -c ]]; then\n'
                          '  [[ "$DMS_TEST_GPUS" = fail ]] && { echo "CUDA initialisation failed" >&2; exit 1; }\n'
                          '  echo "$DMS_TEST_GPUS"; exit\nfi\n'
                          'printf "%s | %s | %s\\n" "$LITGPT_EXPECTED_WORLD_SIZE" "$*" "$LD_LIBRARY_PATH" '
                          '>> "$DMS_TEST_CALLS"\n'
                          'if [[ "$DMS_TEST_FAIL" = 1 && "$*" = *"--suite main"* ]]; then exit 7; fi\n')
    executable.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("MA_", "MASTER_", "EVAL_MASTER_", "NIAH_MASTER_", "VC_", "GROUP_"))
           and key not in {"NPROC_PER_NODE", "LOCAL_WORLD_SIZE", "NODE_RANK"}}
    env.update(DMS_PYTHON=str(executable), DMS_TEST_CALLS=str(tmp_path / "calls"), DMS_TEST_GPUS=str(visible_gpus),
               DMS_TEST_FAIL="0")
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
    command = ["bash", str(BASELINE_ROOT / "majob.sh"), "--model", "outputs/student", "--limit", "4"]
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
    env.update(DMS_TEST_FAIL=str(int(fail_main)), MA_NUM_GPUS="4", MA_NUM_HOSTS=str(nodes), VC_TASK_INDEX=str(rank),
               MA_VJ_NAME="job", MA_TASK_NAME="worker", MA_MASTER_INDEX="0")
    command = ["bash", str(BASELINE_ROOT / "majob.sh"), "configs/experiment_qwen1.7b_32k.yaml",
               "--model", "outputs/student", "--limit", "4", "--output", "outputs/test"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == (7 if fail_main else 0), result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == (1 if fail_main else 2)
    for call in calls:
        assert call.startswith(f"{nodes * 4} | -m torch.distributed.run")
        assert "--nproc_per_node 4" in call
        assert "-m models.qwen3.evaluate --experiment-config configs/experiment_qwen1.7b_32k.yaml" in call
        assert "--model outputs/student --limit 4 --output outputs/test | " in call
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
    for extra in (["--tasks", "boolq"], ["--evaluation-config=x.yaml"], ["--experiment-config", "x.yaml"]):
        result = subprocess.run(command + extra, cwd=tmp_path, env=env, capture_output=True, text=True)
        assert result.returncode == 2 and "not allowed" in result.stderr
    result = subprocess.run(command[:3] + ["--limit", "4"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "--model is required" in result.stderr
    env.pop("MA_VJ_NAME")
    env.update(MA_NUM_HOSTS="2", MASTER_ADDR="localhost")
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "MASTER_ADDR" in result.stderr
    assert not log.exists()


def test_result_check_rejects_same_dataset_name_with_different_prompt_or_runtime():
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


def test_outputs_and_provenance_stay_outside_runtime(tmp_path, monkeypatch, datasets_cache):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(runner, "install_dms_adapter", lambda evaluator: None)
    reference, output = tmp_path / "runtime", tmp_path / "baseline/results"
    reference.mkdir()
    entrypoint = reference / "eval.py"
    entrypoint.write_text("unchanged source")

    def main(**kwargs):
        assert Path.cwd() == output and not datasets_cache.enabled
        Path("eval_results_cache.json").write_text("cached")
        Path("eval_results_20261012_000000.json").write_text(json.dumps({
            "log_kv_B": 128, "swa_window_size": None, "results": {"samples": {"example": []}},
        }))
    evaluator = SimpleNamespace(main=main, __file__=str(entrypoint))
    kwargs = dict(output_path=str(output), tokenizer_dir="base", config_overrides={"block_size": 32768})
    cwd = Path.cwd()
    runner.run_evaluation(evaluator, kwargs, "dms", {"example": "source"}, {"launch_id": "x"})
    assert Path.cwd() == cwd and entrypoint.read_text() == "unchanged source" and datasets_cache.enabled
    assert list(reference.iterdir()) == [entrypoint]
    result = json.loads((output / "eval_results_20261012_000000.json").read_text())
    assert result["model_backend"] == "dms" and result["comparison_contract"] == {"example": "source"}
    assert result["resolved_eval"] == {"launch_id": "x"}
    assert "log_kv_B" not in result and "swa_window_size" not in result
    # A failed second run must not relabel existing results with a new contract.
    evaluator.main = lambda **kwargs: None
    with pytest.raises(RuntimeError, match="one new evaluation result"):
        runner.run_evaluation(evaluator, kwargs, "dms", {"example": "different source"})
    with pytest.raises(ValueError, match="outside"):
        runner.run_evaluation(evaluator, {**kwargs, "output_path": str(reference / "results")}, "dms", {})
    assert not (reference / "results").exists()


def test_cache_entries_from_newer_datasets_are_reported(tmp_path, monkeypatch, datasets_cache):
    monkeypatch.setenv("RANK", "1")

    def fail(**kwargs):
        raise ValueError("Feature type 'List' not found. Available feature types: ['Value', 'Sequence']")

    evaluator = SimpleNamespace(main=fail, __file__=str(tmp_path / "runtime/eval.py"))
    with pytest.raises(RuntimeError, match="newer") as raised:
        runner.run_evaluation(evaluator, {"output_path": str(tmp_path / "output")}, "logkv", {})
    assert "Feature type 'List'" in str(raised.value.__cause__) and datasets_cache.enabled

    # datasets parses dataset_info.json of every cached entry; a malformed one fails there, unnamed.
    namespace = {}
    exec(compile("import json\ndef find_hash_in_cache():\n    json.loads('{\\n  \"a\": 1,\\n    ')\n",
                 "/site-packages/datasets/packaged_modules/cache/cache.py", "exec"), namespace)
    evaluator.main = lambda **kwargs: namespace["find_hash_in_cache"]()
    with pytest.raises(RuntimeError, match="not valid JSON") as raised:
        runner.run_evaluation(evaluator, {"output_path": str(tmp_path / "output")}, "logkv", {})
    assert isinstance(raised.value.__cause__, json.JSONDecodeError)
    evaluator.main = lambda **kwargs: json.loads("{")  # not the cache lookup: unchanged
    with pytest.raises(json.JSONDecodeError):
        runner.run_evaluation(evaluator, {"output_path": str(tmp_path / "output")}, "logkv", {})


def declared_requirements():
    """Quoted requirements of [project] dependencies and each optional-dependencies extra (comments dropped)."""
    text = (BASELINE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = "\n".join(line.split("#")[0] for line in text.split("[project]")[1].split("[tool.")[0].splitlines())
    core = project.split("dependencies = [", 1)[1].split("]\n", 1)[0]
    extras = dict(re.findall(r"^([\w-]+) = \[(.*?)\]$", project.split("[project.optional-dependencies]")[1], re.M))
    quoted = lambda block: re.findall(r'"([^"]+)"', block)
    return quoted(core), {name: quoted(block) for name, block in extras.items()}


# torch218 versions listed in docs/BASELINE_ENVIRONMENT.md section 1 (2026-10-16 snapshot).
TORCH218 = {"transformers": "4.51.3", "PyYAML": "6.0.3", "lightning": "2.6.1", "jsonargparse": "4.41.0",
            "psutil": "7.1.3", "safetensors": "0.7.0", "tokenizers": "0.21.4", "pytest": "9.1.1",
            "lm-eval": "0.4.11", "datasets": "4.8.4", "pandas": "2.3.3", "xlsxwriter": "3.2.9"}


def test_declared_requirements_install_nothing_new_into_torch218():
    core, extras = declared_requirements()
    assert set(extras) == {"test", "eval", "dms-train"}
    assert f"datasets>={'.'.join(map(str, runner.EVAL_DATASETS_MIN))}" in extras["eval"]
    assert f"lm-eval[ruler,longbench,ifeval]>={'.'.join(map(str, runner.EVAL_LM_EVAL_MIN))}" in extras["eval"]
    for requirement in core + extras["test"] + extras["eval"]:
        name = re.split(r"[\[<>=!~]", requirement)[0]
        # accelerate and sentencepiece are not in torch218; a pin would change its packages
        assert name not in {"accelerate", "sentencepiece", "openpyxl"} and "==" not in requirement, requirement
        for op, bound in re.findall(r"(>=|<=|<|>)([\d.]+)", requirement):
            if name not in TORCH218:  # not in the environment snapshot (tqdm)
                continue
            # PEP 440 compares releases padded with zeros (4.41 == 4.41.0)
            found = tuple(int(part) for part in (TORCH218[name].split(".") + ["0", "0"])[:3])
            bound = tuple(int(part) for part in (bound.split(".") + ["0", "0"])[:3])
            assert {">=": found >= bound, ">": found > bound, "<=": found <= bound, "<": found < bound}[op], requirement
    # The upstream training pins live only in the separate training environment's extra.
    from models.qwen3.check_env import TRAINING_TRANSFORMERS
    assert f"transformers=={TRAINING_TRANSFORMERS}" in extras["dms-train"]


@pytest.mark.parametrize("versions,message", [
    ({"lm-eval": "0.4.11", "datasets": "4.8.4"}, None),  # torch218
    ({"lm-eval": "0.4.9", "datasets": "4.8.4"}, None),
    ({"lm-eval": "0.4.11", "datasets": "5.1.0"}, None),
    ({"lm-eval": "0.4.11", "datasets": "4.4.2"}, "datasets>=4.7"),  # the DMS training environment
    ({"lm-eval": "0.4.11", "datasets": "3.6.0"}, "datasets>=4.7"),
    ({"lm-eval": "0.4.8", "datasets": "4.8.4"}, "lm-eval>=0.4.9"),
])
def test_environment_check_follows_code_and_cache_requirements(monkeypatch, versions, message):
    monkeypatch.setattr(runner.importlib.metadata, "version", versions.__getitem__)
    if message is None:
        runner.check_evaluation_packages()
    else:
        with pytest.raises(RuntimeError, match=message.replace(".", r"\.") + ".*torch218"):
            runner.check_evaluation_packages()
