import os
from pathlib import Path
import re
import subprocess

import pytest
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
    result, args = _launch(tmp_path, args=("--backend", "dense", "--tasks", "niah_single_2", "--limit", "4"))
    assert result.returncode == 0, result.stderr
    assert args[:3] == ["--standalone", "--nproc_per_node", "1"]
    config = ROOT / args[args.index("--config") + 1]
    settings = yaml.safe_load(config.read_text())
    assert settings["semantic_config"].endswith("arc_attach_alpha_beta_k12_b128_2k.yaml")
    assert settings["model"] == "outputs/semanticlogkv_base"
    assert settings["tasks"][-3:] == ["niah_single_1", "niah_single_2", "niah_single_3"]
    assert settings["start_size"] + settings["recent_size"] == 2048
    assert args[-6:] == ["--backend", "dense", "--tasks", "niah_single_2", "--limit", "4"]


def test_benchmark_snapshot_matches_current_litgpt_defaults():
    settings = yaml.safe_load((ROOT / "configs/cam_qwen3_1.7b.yaml").read_text())
    reference = (ROOT / settings["litgpt_root"] / "eval.sh").resolve()
    if not reference.is_file():
        pytest.skip("Paired SemanticLogKV checkout is not available")
    source = reference.read_text()
    benchmarks = ""
    for assignment in re.findall(r'^DEFAULT_BENCHMARKS="([^"]+)"$', source, flags=re.MULTILINE):
        benchmarks = assignment.replace("${DEFAULT_BENCHMARKS}", benchmarks)
    niah = re.search(r'^NIAH_BENCHMARKS=\$\{3:-([^}]+)\}', source, flags=re.MULTILINE).group(1)
    assert settings["tasks"] == (benchmarks + "," + niah).split(",")
    lengths = re.search(r'"max_seq_lengths": \[([0-9, ]+)\]', source).group(1)
    assert settings["ruler_lengths"] == [int(value) for value in lengths.split(",")]


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
