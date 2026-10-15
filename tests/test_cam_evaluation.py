from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import pytest
import yaml

from models.cam.evaluate import DEFAULT_CONFIG, METHOD_FIELDS, ROOT, resolve_run
from models.qwen3.evaluate import run_evaluation


def args(**overrides):
    values = dict(config=DEFAULT_CONFIG, model=None, backend="cam", suite="main", tasks=None, output=None)
    values.update(dict.fromkeys(METHOD_FIELDS))
    return Namespace(**(values | overrides))


def test_resolve_shared_defaults_and_explicit_smoke_options(tmp_path):
    experiment, model, output, tasks, lengths, options = resolve_run(args(output=tmp_path))
    assert experiment == ROOT / "configs/experiment_qwen1.7b_32k.yaml"
    assert model.name == "semanticlogkv_base" and output == tmp_path / "main"
    assert len(tasks.split(",")) == 47 and lengths is None
    assert options["start_size"] + options["recent_size"] == 2048
    _, _, _, tasks, lengths, _ = resolve_run(args(suite="niah"))
    assert tasks == "niah_single_1,niah_single_2,niah_single_3" and lengths == [32768]
    _, _, _, tasks, lengths, options = resolve_run(args(suite=None, tasks="niah_single_2", recent_size=100))
    assert tasks == "niah_single_2" and lengths is None and options["recent_size"] == 100
    with pytest.raises(ValueError, match="requires --model"):
        resolve_run(args(backend="logkv"))
    with pytest.raises(ValueError, match="--suite"):
        resolve_run(args(suite=None))
    with pytest.raises(ValueError, match="merge_size"):
        resolve_run(args(recent_size=8))
    outside = tmp_path / "cam.yaml"
    (tmp_path / "experiment.yaml").write_text("ckpt_dir: /x\ncontext_length: 32768\n")
    outside.write_text(yaml.safe_dump({**yaml.safe_load(DEFAULT_CONFIG.read_text()),
                                       "experiment_config": str(tmp_path / "experiment.yaml")}))
    with pytest.raises(ValueError, match="inside this project"):
        resolve_run(args(config=outside))


def test_cam_result_is_annotated_without_logkv_labels(tmp_path):
    root, output = tmp_path / "reference", tmp_path / "out"
    root.mkdir()
    def main(**kwargs):
        Path("eval_results_20261012_120000.json").write_text(json.dumps({
            "results": {"samples": {}}, "log_kv_B": 128, "swa_window_size": None,
        }))
    evaluator = SimpleNamespace(main=main, __file__=str(root / "eval.py"))
    kwargs = dict(output_path=str(output), tokenizer_dir="base", config_overrides={"block_size": 32768})
    run_evaluation(evaluator, kwargs, "cam", {"same": "contract"}, result_metadata={"cam": {"seed": 42}})
    result = json.loads(next(output.glob("*.json")).read_text())
    assert result["model_backend"] == "cam" and result["cam"]["seed"] == 42
    assert "log_kv_B" not in result and "swa_window_size" not in result


def test_launcher_sets_hf_caches_before_importing_transformers():
    command = '''
import sys
from models.cam.evaluate import resolve_run
assert "transformers" not in sys.modules
from tests.test_cam_evaluation import args
resolve_run(args())
import os
from huggingface_hub import constants
assert constants.HF_HUB_CACHE == os.environ["HF_HUB_CACHE"]
'''
    result = subprocess.run([sys.executable, "-c", command], text=True, capture_output=True,
                            env={**os.environ, "HF_HOME": "/tmp/incorrect-cam-test-cache"})
    assert result.returncode == 0, result.stderr
