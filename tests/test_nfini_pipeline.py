import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode,fail", [("--pipeline", ""), ("--check", ""), ("--pipeline", "cpt:2")])
def test_managed_cluster_entrypoint(tmp_path, mode, fail):
    repo = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    shutil.copy(repo / "configs/nfini_cpt.yaml", tmp_path / "configs")
    shutil.copy(repo / "majob.sh", tmp_path)
    (tmp_path / "scripts/hattention_env.sh").write_text('PYTHON_BIN="$TEST_PYTHON"\n')
    (tmp_path / "scripts/train_nfini.sh").write_text(
        'echo "$1" >> "stages.$NODE_RANK"\n[[ "$1:$NODE_RANK" != "$FAIL_STAGE" ]]\n')
    (tmp_path / "eval.sh").write_text(
        '[[ "$NFINI_CHECKPOINT_DIR" == "$NFINI_OUTPUT_ROOT/cpt" ]] || exit 1\n'
        'echo "${1:-eval}" >> "stages.$NODE_RANK"\n')
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1] - 2
    env = dict(os.environ, PYTHONPATH=str(repo), TEST_PYTHON=sys.executable, MA_NUM_HOSTS="4",
               MA_NUM_GPUS="8", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
               NFINI_OUTPUT_ROOT=str(tmp_path / "output"), FAIL_STAGE=fail)
    env.pop("MA_VJ_NAME", None)
    workers = [subprocess.Popen(["bash", "majob.sh", mode], cwd=tmp_path,
                                env=dict(env, VC_TASK_INDEX=str(i))) for i in range(4)]
    try:
        for worker in workers:
            assert (worker.wait(timeout=90) != 0) == bool(fail)
    finally:
        for worker in workers:
            worker.kill()
            worker.wait()
    for rank in range(4):
        expected = ["check", "--check-runtime"]
        if rank == 0:
            expected.append("--check-data")
        if mode == "--pipeline":
            if rank == 0:
                expected.append("prepare")
            expected.append("cpt")
            if not fail:
                expected.append("eval")
        assert (tmp_path / f"stages.{rank}").read_text().splitlines() == expected
