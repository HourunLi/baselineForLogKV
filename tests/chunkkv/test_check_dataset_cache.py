"""Offline cache entries a newer `datasets` wrote are reported, and moved aside rather than deleted."""

import json
import os
import re
import shutil
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

    # A feature type the installed datasets does not know, as datasets 4 `List` is to datasets 3.
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
    # Undoing the quarantine puts the entry back with its own modification time, so it is selected again.
    restored = run("-m", "chunkkv.check_dataset_cache", "--restore", str(quarantine))
    assert restored.returncode == 1 and "restored 1 entries" in restored.stdout, restored.stdout + restored.stderr
    assert newer.is_dir() and newer.stat().st_mtime == 1_700_000_100
    assert f"Feature type '{newer_type}' not found" in run("-c", LOAD).stdout


def test_restore_sets_recorded_converted_copies_aside_and_refuses_anything_else(tmp_path):
    pytest.importorskip("datasets")
    from chunkkv.train import sha256

    root, quarantine = tmp_path / "hf/hf_cache", tmp_path / "quarantine"

    def entry(base, relative, content, modified):
        directory = base / relative
        directory.mkdir(parents=True)
        (directory / "dataset_info.json").write_text(json.dumps({"config_name": relative.split("/")[1]}))
        (directory / "data-test.arrow").write_text(content)
        os.utime(directory, (modified, modified))
        return directory

    generation = "truthfulqa___truthful_qa/generation/0.0.0/741b"
    original = entry(quarantine, generation, "written by datasets 4.8.4", 1_700_000_200)
    copy = entry(root, generation, "converted copy", 1_800_000_000)
    moved = entry(quarantine, "cais___mmlu/anatomy/0.0.0/new", "written by datasets 4.8.4", 1_700_000_100)
    (quarantine / "conversions.json").write_text(json.dumps([{
        "entry": str(copy), "original": str(original),
        "files": {"data-test.arrow": {"original_sha256": sha256(original / "data-test.arrow"),
                                      "converted_sha256": sha256(copy / "data-test.arrow")}}}]))
    unrecorded = entry(quarantine, "ucinlp___drop/default/0.0.0/95cd", "quarantined", 1_700_000_300)
    occupied = entry(root, "ucinlp___drop/default/0.0.0/95cd", "not ours", 1_700_000_400)
    env = {**os.environ, "CHUNKKV_HF_CACHE": str(tmp_path / "hf"),
           "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}

    def restore():
        return subprocess.run([sys.executable, "-m", "chunkkv.check_dataset_cache", "--restore", str(quarantine)],
                              cwd=tmp_path, env=env, capture_output=True, text=True)

    refused = restore()
    assert refused.returncode != 0 and "not a recorded converted copy" in refused.stderr
    assert original.is_dir() and moved.is_dir() and (copy / "data-test.arrow").read_text() == "converted copy"
    shutil.move(str(occupied), str(tmp_path / "other job's entry"))

    result = restore()
    assert result.returncode == 0 and "restored 3 entries" in result.stdout, result.stdout + result.stderr
    for relative, content, modified in ((generation, "written by datasets 4.8.4", 1_700_000_200),
                                        ("cais___mmlu/anatomy/0.0.0/new", "written by datasets 4.8.4", 1_700_000_100),
                                        ("ucinlp___drop/default/0.0.0/95cd", "quarantined", 1_700_000_300)):
        assert (root / relative / "data-test.arrow").read_text() == content
        assert (root / relative).stat().st_mtime == modified
    kept = quarantine / "_converted_copies" / generation
    assert (kept / "data-test.arrow").read_text() == "converted copy" and not unrecorded.exists()


def test_broken_dataset_info_blocks_its_config_and_is_related_to_the_quarantine(tmp_path):
    datasets = pytest.importorskip("datasets")
    from datasets.arrow_writer import ArrowWriter
    from chunkkv.train import sha256

    root, quarantine = tmp_path / "hf/hf_cache", tmp_path / "quarantine"
    features = datasets.Features({"question": datasets.Value("string")})

    def entry(base, relative, modified):
        directory = base / relative
        directory.mkdir(parents=True)
        with ArrowWriter(features=features, path=str(directory / "mmlu-test.arrow")) as writer:
            writer.write({"question": relative})
            examples, size = writer.finalize()
        splits = datasets.SplitDict(dataset_name="mmlu")
        splits.add(datasets.SplitInfo(name="test", num_bytes=size, num_examples=examples, dataset_name="mmlu"))
        datasets.DatasetInfo(features=features, dataset_name="mmlu", config_name="anatomy",
                             splits=splits).write_to_directory(str(directory))
        os.utime(directory, (modified, modified))
        return directory

    relative = "cais___mmlu/anatomy/0.0.0/restored"
    newest = entry(root, "cais___mmlu/anatomy/0.0.0/newest", 1_700_000_100)
    restored = entry(root, relative, 1_700_000_000)
    original_info = (restored / "dataset_info.json").read_bytes()
    # The old --convert wrote a multi-line copy at this path; the bytes now there are that copy cut to the
    # original's length, as seen on the server.
    copy = entry(quarantine / "_converted_copies", relative, 1_700_000_000)
    converted = json.dumps(json.loads(original_info), indent=2, sort_keys=True).encode()
    (copy / "dataset_info.json").write_bytes(converted)
    (restored / "dataset_info.json").write_bytes(converted[:len(original_info)])
    os.utime(restored, (1_700_000_000, 1_700_000_000))
    (quarantine / "conversions.json").write_text(json.dumps([{
        "entry": str(restored), "original": str(quarantine / relative),
        "files": {"mmlu-test.arrow": {"original_sha256": sha256(restored / "mmlu-test.arrow"),
                                      "converted_sha256": "0" * 64}}}]))
    env = {**os.environ, "CHUNKKV_HF_CACHE": str(tmp_path / "hf"),
           "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}

    def run(*args):
        return subprocess.run([sys.executable, *args], cwd=tmp_path, env=env, capture_output=True, text=True)

    # Loading fails although the newest entry is intact: datasets parses every entry's dataset_info.json.
    loaded = run("-c", LOAD.replace("abstract_algebra", "anatomy")).stdout
    assert re.search(r"line \d+ column \d+ \(char \d+\)", loaded), loaded
    (root / "cais___mmlu/anatomy/0.0.0/partial.incomplete").mkdir()

    report = run("-m", "chunkkv.check_dataset_cache", "--compare", str(quarantine))
    out = report.stdout
    assert report.returncode == 1, out + report.stderr
    assert "cais___mmlu/anatomy: 0.0.0/restored -> JSONDecodeError: " in out
    assert "(datasets writes it as one line)" in out and "text before the error" in out
    assert f"the broken file is its first {len(original_info)} bytes" in out
    assert "cais___mmlu/anatomy: 0.0.0/partial.incomplete -> FileNotFoundError" in out
    assert "conversion cais___mmlu/anatomy/0.0.0/restored: mmlu-test.arrow original; " \
           "dataset_info.json NOT valid JSON" in out and "<- not the original entry" in out
    assert "1 dataset configs fail offline loading" in out and "1 recorded conversions" in out
    # Never moved aside: the fix depends on where the broken file came from.
    moved = run("-m", "chunkkv.check_dataset_cache", "--quarantine", str(tmp_path / "aside"))
    assert moved.returncode == 1 and restored.is_dir() and newest.is_dir() and not (tmp_path / "aside").exists()
