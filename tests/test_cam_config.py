import os
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _launch(tmp_path, env=None, args=()):
    capture = tmp_path / "args"
    command = tmp_path / "torchrun"
    command.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAM_TEST_ARGS"\n')
    command.chmod(0o755)
    process_env = {key: value for key, value in os.environ.items() if key not in {
        "NPROC_PER_NODE", "NNODES", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT",
        "MA_NUM_GPUS", "LOCAL_WORLD_SIZE", "MA_NUM_HOSTS", "GROUP_WORLD_SIZE",
        "VC_TASK_INDEX", "GROUP_RANK", "MA_VJ_NAME", "MA_TASK_NAME", "MA_MASTER_INDEX",
    }}
    process_env.update(PATH=f"{tmp_path}:{process_env['PATH']}", CAM_TEST_ARGS=str(capture))
    process_env.update(env or {})
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/evaluate_cam.sh"), *args],
        env=process_env, text=True, capture_output=True, cwd=tmp_path,
    )
    return result, capture.read_text().splitlines() if capture.exists() else []


def test_default_launcher_passes_the_shared_config_and_cli_overrides(tmp_path):
    result, args = _launch(tmp_path, args=("--backend", "dense", "--suite", "niah", "--limit", "4"))
    assert result.returncode == 0, result.stderr
    assert args[:3] == ["--standalone", "--nproc_per_node", "1"]
    config = ROOT / args[args.index("--config") + 1]
    settings = yaml.safe_load(config.read_text())
    assert settings["experiment_config"] == "configs/experiment_qwen1.7b_32k.yaml"
    assert settings["evaluation_config"] == "configs/eval_suites.yaml"
    assert settings["model"] == "outputs/semanticlogkv_base"
    assert settings["start_size"] + settings["recent_size"] == 2048
    assert args[-6:] == ["--backend", "dense", "--suite", "niah", "--limit", "4"]


def test_project_task_lists_are_the_fixed_32k_protocol():
    """Snapshot of the reference eval.sh/majob.sh defaults (HourunLi/LogKV@4fd97c7), NIAH at 32K only."""
    suites = yaml.safe_load((ROOT / "configs/eval_suites.yaml").read_text())
    main = suites["main_tasks"]
    assert len(main) == 47 and len(set(main)) == 47
    longbench = [task for task in main if task.startswith("longbench_")]
    assert len(longbench) == 33 and sum(task.endswith("_e") for task in longbench) == 13
    assert main[:14] == ["boolq", "piqa", "social_iqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge",
                         "openbookqa", "mmlu", "ceval-valid", "ifeval", "truthfulqa_gen", "truthfulqa_mc1",
                         "truthfulqa_mc2"]
    assert suites["niah_tasks"] == ["niah_single_1", "niah_single_2", "niah_single_3"]
    assert suites["niah_max_seq_lengths"] == [32768]


def test_standard_multinode_launcher_uses_shared_rendezvous(tmp_path):
    result, args = _launch(tmp_path, {
        "NNODES": "2", "NPROC_PER_NODE": "4", "NODE_RANK": "1",
        "MASTER_ADDR": "worker0", "MASTER_PORT": "6123",
    })
    assert result.returncode == 0, result.stderr
    assert "--standalone" not in args
    assert args[:10] == ["--nnodes", "2", "--nproc_per_node", "4", "--node_rank", "1",
                         "--master_addr", "worker0", "--master_port", "6123"]


def test_modelarts_multinode_does_not_silently_launch_standalone(tmp_path):
    result, args = _launch(tmp_path, {"MA_NUM_HOSTS": "2"})
    assert result.returncode != 0 and not args
    assert "MASTER_ADDR" in result.stderr
    result, args = _launch(tmp_path, {"MA_NUM_HOSTS": "2", "NNODES": "1"})
    assert result.returncode != 0 and not args
    assert "isolated jobs" in result.stderr


def test_modelarts_launcher_resolves_platform_node_address(tmp_path):
    result, args = _launch(tmp_path, {
        "MA_NUM_HOSTS": "2", "MA_NUM_GPUS": "8", "VC_TASK_INDEX": "1",
        "MA_VJ_NAME": "job", "MA_TASK_NAME": "worker", "MA_MASTER_INDEX": "0",
    })
    assert result.returncode == 0, result.stderr
    assert args[:10] == ["--nnodes", "2", "--nproc_per_node", "8", "--node_rank", "1",
                         "--master_addr", "job-worker-0.job", "--master_port", "6000"]
