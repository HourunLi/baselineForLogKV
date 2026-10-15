"""Offline cache settings, RULER JSON interception and the dataset-alias fallback."""

import json
import os

import pytest

from h2okv import data_env


@pytest.fixture
def restore_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_environment_matches_the_shared_cache_layout(restore_environ):
    values = data_env.configure_hf_environment("/cache/root")
    assert values["HF_DATASETS_CACHE"] == "/cache/root/hf_cache"
    assert values["HF_HUB_CACHE"] == values["HUGGINGFACE_HUB_CACHE"] == "/cache/root/hub"
    assert values["RULER_CACHE_DIR"] == "/cache/root/ruler_cache"
    assert os.environ["HF_DATASETS_OFFLINE"] == os.environ["HF_HUB_OFFLINE"] == "1"
    assert data_env.configure_hf_environment("/x", offline=False)["HF_DATASETS_OFFLINE"] == "0"


def test_ruler_json_is_served_from_the_local_cache(tmp_path, restore_environ):
    requests = pytest.importorskip("requests")
    os.environ.update(RULER_CACHE_DIR=str(tmp_path), HF_DATASETS_OFFLINE="1")
    url, name = next(iter(data_env.RULER_URL_FILES.items()))
    (tmp_path / name).write_text(json.dumps({"data": [1]}))
    data_env.install_ruler_cache()
    assert requests.get(url).json() == {"data": [1]}
    assert requests.Session().request("GET", url).json() == {"data": [1]}
    (tmp_path / name).unlink()
    with pytest.raises(RuntimeError, match="Offline evaluation needs"):
        requests.get(url)


def test_unresolvable_alias_loads_the_legacy_cache_entry(tmp_path, monkeypatch):
    config = pytest.importorskip("datasets.config")
    monkeypatch.setattr(config, "HF_DATASETS_OFFLINE", True)
    monkeypatch.setattr(config, "HF_DATASETS_CACHE", tmp_path)
    (tmp_path / "paul_graham_essays").mkdir()
    (tmp_path / "baber___paul_graham_essays").symlink_to(tmp_path / "paul_graham_essays")
    calls = []

    def original(path, *args, **kwargs):
        calls.append(path)
        if "/" in path:
            raise FileNotFoundError(path)
        return f"loaded {path}"

    load = data_env.follow_dataset_cache_alias(original)
    assert load("baber/paul_graham_essays", split="train") == "loaded paul_graham_essays"
    assert calls == ["baber/paul_graham_essays", "paul_graham_essays"]
    with pytest.raises(FileNotFoundError):
        load("other/missing")  # no alias: the original error is kept
    monkeypatch.setattr(config, "HF_DATASETS_OFFLINE", False)
    with pytest.raises(FileNotFoundError):
        load("baber/paul_graham_essays")
