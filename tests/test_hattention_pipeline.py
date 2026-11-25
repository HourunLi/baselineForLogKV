import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest


@pytest.mark.parametrize("failure", ["", "convert:0", "cpt:1", "preflight:1", "eval_only"])
def test_two_node_pipeline(tmp_path, failure):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "train_hattention.sh").write_text(
        'echo "$1" >> "stages.$NODE_RANK"\n'
        '[[ "$1:$NODE_RANK" != "$FAIL_STAGE" ]]\n'
    )
    (tmp_path / "eval.sh").write_text(
        'if [[ "$1" == --preflight ]]; then [[ "$FAIL_STAGE" != "preflight:$NODE_RANK" ]]; exit $?; fi\n'
        'echo eval >> "stages.$NODE_RANK"\n'
        '[[ "$1" == configs/qwen8b_hattention_eval.yaml && "$LLA_CHECKPOINT_DIR" == "$LLA_STUDENT_ROOT/cpt" ]]\n'
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1] - 2
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), MA_NUM_HOSTS="2",
               LLA_STUDENT_ROOT=str(tmp_path / "student"), FAIL_STAGE=failure,
               PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    env.pop("MA_VJ_NAME", None)
    if failure == "eval_only":
        env["LLA_PIPELINE_EVAL_ONLY"] = "1"
    workers = [subprocess.Popen([sys.executable, "-m", "lla_bench.pipeline"], cwd=tmp_path,
                                env=dict(env, VC_TASK_INDEX=str(rank))) for rank in range(2)]
    try:
        for worker in workers:
            assert (worker.wait(timeout=60) != 0) == (failure not in {"", "eval_only"})
    finally:
        for worker in workers:
            worker.kill()
            worker.wait()
    leader = ["convert"] if failure.startswith("convert") else ["convert", "prepare", "probe", "distill", "cpt"]
    if not failure:
        leader.append("eval")
    if failure == "eval_only":
        for rank in range(2):
            assert (tmp_path / f"stages.{rank}").read_text().splitlines() == ["eval"]
        return
    assert (tmp_path / "stages.0").read_text().splitlines() == leader
    other = tmp_path / "stages.1"
    assert (other.read_text().splitlines() if other.exists() else []) == leader[3:]


@pytest.mark.parametrize("stage,remaining", [("cpt", ("eval",)), ("distill", ("prepare", "cpt", "eval"))])
def test_resume_from_latest_checkpoint(tmp_path, monkeypatch, stage, remaining):
    import yaml
    from lla_bench.pipeline import remaining_stages
    from lla_bench.model import HConfig, HAttentionLM
    from lla_bench.student import export_checkpoint

    cfg = HConfig(16, 2, 1, 32, 1, 8, 8, 1, 4, 1e-6, False, True,
                  trained_levels=6, block_size=32, student=True, mlp_width=48)
    recipe = {"context_length": 32, stage: {"steps": 1200}}
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    Path("configs/qwen8b_hattention_train.yaml").write_text(yaml.safe_dump(recipe))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}")
    out = tmp_path / stage
    export_checkpoint(out, cfg, HAttentionLM(cfg).state_dict(), tmp_path,
                      dict(stage=stage, recipe=recipe, actual_steps=1200))
    assert remaining_stages(tmp_path) == remaining
    (out / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="Cannot reuse"):
        remaining_stages(tmp_path)
