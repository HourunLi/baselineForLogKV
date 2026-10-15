"""Exercise transform cache isolation with real Arrow files and independent processes."""

import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_evaluation_uses_private_transform_files_and_preserves_offline_source(tmp_path):
    datasets = pytest.importorskip("datasets")
    source = tmp_path / "source"
    datasets.Dataset.from_dict({"text": ["one", "two", "three"]}).save_to_disk(str(source))
    raw = datasets.load_from_disk(str(source))
    was_enabled = datasets.is_caching_enabled()
    datasets.enable_caching()
    try:
        mapped = raw.map(lambda row: {"text": row["text"].upper()}, new_fingerprint="shared-map")
        shared = Path(mapped.cache_files[0]["filename"])
        del mapped
        gc.collect()
        shared.write_bytes(b"")
    finally:
        if not was_enabled:
            datasets.disable_caching()
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
    script = r'''
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from datasets import load_from_disk
from chunkkv.evaluate import run_evaluation

source, output = map(Path, sys.argv[1:])
def evaluate(**kwargs):
    result = load_from_disk(str(source)).map(
        lambda row: {"text": row["text"].upper()}, new_fingerprint="shared-map")
    Path("eval_results_20261015_000000.json").write_text(json.dumps({
        "results": {}, "rows": result[:], "cache_files": result.cache_files,
    }))
run_evaluation(SimpleNamespace(main=evaluate, __file__=str(source.parent / "runtime/eval.py")),
               dict(output_path=str(output), tokenizer_dir="base", config_overrides={"block_size": 32768}),
               "dense", {})
'''
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "RANK": "0", "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
           "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    outputs = [tmp_path / f"worker-{i}" for i in range(2)]
    workers = [subprocess.Popen([sys.executable, "-c", script, str(source), str(output)],
                                cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
               for output in outputs]
    try:
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=60)
            assert worker.returncode == 0, stdout + stderr
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
    results = [json.loads((p / "eval_results_20261015_000000.json").read_text()) for p in outputs]
    assert all(result["rows"] == {"text": ["ONE", "TWO", "THREE"]} for result in results)
    directories = [Path(result["cache_files"][0]["filename"]).parent for result in results]
    assert directories[0] != directories[1]
    assert all(not directory.is_relative_to(source) for directory in directories)
    assert shared.stat().st_size == 0
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
