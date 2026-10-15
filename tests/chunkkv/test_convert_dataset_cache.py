"""Entries written by datasets>=4 become readable by datasets 3.x without changing their Arrow data."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from chunkkv.check_dataset_cache import Unconvertible, legacy_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / "fixtures/datasets4_cache"
LOAD = r'''
import json, sys
from chunkkv.evaluate import configure_environment
configure_environment()
import datasets
path, name, split = sys.argv[1:]
try:
    print(json.dumps(datasets.load_dataset(path, None if name == "-" else name)[split].to_list()))
except ValueError as exc:
    print(exc)
'''
LEGACY = {"Value", "ClassLabel", "Sequence", "LargeList"}


def test_list_spellings_map_to_the_datasets3_types_with_the_same_arrow_layout():
    value = {"dtype": "string", "_type": "Value"}
    written_by_v4 = {  # as datasets 4.4.2 and 5.1.0 serialize these columns
        "answers": {"feature": value, "_type": "List"},
        "targets": {"choices": {"feature": value, "_type": "List"}},
        "spans": {"feature": {"text": value}, "_type": "List"},
        "pair": {"feature": value, "length": 2, "_type": "List"},
        "nested": {"feature": {"feature": value, "_type": "List"}, "_type": "List"},
        "large": {"feature": {"feature": value, "_type": "List"}, "_type": "LargeList"},
    }
    assert legacy_features(written_by_v4, LEGACY) == {  # as datasets 3.6.0 serializes the same Arrow types
        "answers": {"feature": value, "_type": "Sequence"},
        "targets": {"choices": {"feature": value, "_type": "Sequence"}},
        "spans": [{"text": value}],
        "pair": {"feature": value, "length": 2, "_type": "Sequence"},
        "nested": {"feature": {"feature": value, "_type": "Sequence"}, "_type": "Sequence"},
        "large": {"feature": {"feature": value, "_type": "Sequence"}, "_type": "LargeList"},
    }
    with pytest.raises(Unconvertible, match="Json"):
        legacy_features({"kwargs": {"feature": {"_type": "Json"}, "_type": "List"}}, LEGACY)
    with pytest.raises(Unconvertible, match="fixed-length list of structs"):
        legacy_features({"x": {"feature": {"a": value}, "length": 2, "_type": "List"}}, LEGACY)


def test_entries_without_a_readable_fallback_are_converted_and_originals_kept(tmp_path):
    datasets = pytest.importorskip("datasets")
    if int(datasets.__version__.split(".")[0]) >= 4:
        pytest.skip("datasets>=4 reads these entries directly")
    root = tmp_path / "hf/hf_cache"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns("WRITTEN_BY"))
    env = {**os.environ, "CHUNKKV_HF_CACHE": str(tmp_path / "hf"),
           "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}

    def run(*args):
        return subprocess.run([sys.executable, *args], cwd=tmp_path, env=env, capture_output=True, text=True)

    def load(path, name, split):
        return run("-c", LOAD, path, name, split).stdout.strip()

    loads = [("truthfulqa/truthful_qa", "generation", "validation"),
             ("truthfulqa/truthful_qa", "multiple_choice", "validation"), ("example/structs", "-", "train")]
    assert all("Feature type 'List' not found" in load(*args) for args in loads)
    report = run("-m", "chunkkv.check_dataset_cache")
    assert report.returncode == 1 and "4 dataset configs fail" in report.stdout, report.stdout + report.stderr
    assert report.stdout.count("convertible offline with --quarantine DIR --convert") == 3
    assert "google___if_eval/default" in report.stdout and "feature type 'Json'" in report.stdout
    assert run("-m", "chunkkv.check_dataset_cache", "--convert").returncode == 2

    quarantine = tmp_path / "quarantine"
    converted = run("-m", "chunkkv.check_dataset_cache", "--quarantine", str(quarantine), "--convert")
    assert converted.returncode == 1 and "1 dataset configs fail" in converted.stdout, converted.stdout
    assert json.loads(load(*loads[0])) == [
        {"type": "Adversarial", "question": "q1", "best_answer": "a1",
         "correct_answers": ["a1", "a1b"], "incorrect_answers": ["x1"]},
        {"type": "Non-Adversarial", "question": "q2", "best_answer": "a2",
         "correct_answers": ["a2"], "incorrect_answers": []}]
    assert json.loads(load(*loads[1]))[0]["mc2_targets"] == {"choices": ["a", "c", "b"], "labels": [1, 1, 0]}
    assert json.loads(load(*loads[2])) == [
        {"spans": [{"text": "t", "start": 1}], "pair": [1.0, 2.0], "nested": [[1, 2], [3]], "label": 1},
        {"spans": [], "pair": [3.0, 4.0], "nested": [[]], "label": 0}]
    # Originals are moved aside byte for byte; the unconvertible Json entry is left in place.
    for original in FIXTURE.glob("*/*/*/*/*"):
        relative = original.relative_to(FIXTURE)
        if relative.parts[0] == "google___if_eval":
            assert (root / relative).read_bytes() == original.read_bytes()
        else:
            assert (quarantine / relative).read_bytes() == original.read_bytes()
    records = json.loads((quarantine / "conversions.json").read_text())
    assert len(records) == 3 and not (quarantine / ".converting").exists()
    rerun = run("-m", "chunkkv.check_dataset_cache")
    assert "google___if_eval/default" in rerun.stdout and "1 dataset configs fail" in rerun.stdout
