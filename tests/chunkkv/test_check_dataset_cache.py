"""Offline cache entries a newer `datasets` wrote are reported, and moved aside rather than deleted."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOAD = r'''
from chunkkv.evaluate import configure_environment
configure_environment()
import datasets
try:
    print(datasets.load_dataset("cais/mmlu", "abstract_algebra")["test"][0]["question"])
except ValueError as exc:
    print(exc)
'''


def test_entries_unreadable_by_installed_datasets_are_reported_and_moved_aside(tmp_path):
    datasets = pytest.importorskip("datasets")
    from datasets.arrow_writer import ArrowWriter

    # datasets>=4 writes list columns as `List`, a feature type the evaluation's datasets<4 cannot parse.
    newer_type = "List" if int(datasets.__version__.split(".")[0]) < 4 else "FeatureTypeOfNewerDatasets"
    root = tmp_path / "hf/hf_cache"
    features = datasets.Features({"question": datasets.Value("string"),
                                  "choices": datasets.Sequence(datasets.Value("string"))})

    def entry(config, name, modified, readable=True):
        directory = root / "cais___mmlu" / config / "0.0.0" / name
        directory.mkdir(parents=True)
        with ArrowWriter(features=features, path=str(directory / "mmlu-test.arrow")) as writer:
            writer.write({"question": name, "choices": ["a", "b"]})
            examples, size = writer.finalize()
        splits = datasets.SplitDict(dataset_name="mmlu")
        splits.add(datasets.SplitInfo(name="test", num_bytes=size, num_examples=examples, dataset_name="mmlu"))
        datasets.DatasetInfo(features=features, dataset_name="mmlu", config_name=config,
                             splits=splits).write_to_directory(str(directory))
        if not readable:
            info = json.loads((directory / "dataset_info.json").read_text())
            info["features"]["choices"]["_type"] = newer_type
            (directory / "dataset_info.json").write_text(json.dumps(info))
        os.utime(directory, (modified, modified))
        return directory

    older = entry("abstract_algebra", "prepared-by-v3", 1_700_000_000)
    newer = entry("abstract_algebra", "prepared-by-v4", 1_700_000_100, readable=False)
    only = entry("anatomy", "only-v4", 1_700_000_000, readable=False)
    env = {**os.environ, "CHUNKKV_HF_CACHE": str(tmp_path / "hf"),
           "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}

    def run(*args):
        return subprocess.run([sys.executable, *args], cwd=tmp_path, env=env, capture_output=True, text=True)

    # The newest entry wins, as in the evaluation's offline task loading.
    assert f"Feature type '{newer_type}' not found" in run("-c", LOAD).stdout
    report = run("-m", "chunkkv.check_dataset_cache")
    assert report.returncode == 1, report.stdout + report.stderr
    assert "cais___mmlu/abstract_algebra: offline load selects 0.0.0/prepared-by-v4" in report.stdout
    assert "readable entry kept: 0.0.0/prepared-by-v3" in report.stdout
    assert "cais___mmlu/anatomy" in report.stdout and "no readable entry" in report.stdout
    assert newer.is_dir() and "2 dataset configs fail" in report.stdout  # reporting changes nothing
    assert run("-m", "chunkkv.check_dataset_cache", "--quarantine", str(root / "aside")).returncode == 2

    quarantine = tmp_path / "quarantine"
    moved = run("-m", "chunkkv.check_dataset_cache", "--quarantine", str(quarantine))
    assert moved.returncode == 1 and "1 dataset configs fail" in moved.stdout  # anatomy has no readable entry
    assert not newer.exists() and older.is_dir() and only.is_dir()
    assert (quarantine / "cais___mmlu/abstract_algebra/0.0.0/prepared-by-v4/dataset_info.json").is_file()
    assert run("-c", LOAD).stdout.strip() == "prepared-by-v3"
