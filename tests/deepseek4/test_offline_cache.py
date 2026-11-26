import pytest


def test_social_iqa_reads_confirmed_snapshot_without_alias(tmp_path, monkeypatch, capsys):
    datasets = pytest.importorskip("datasets", minversion="4.8.4")
    pytest.importorskip("lm_eval")
    pytest.importorskip("lightning")
    from datasets.arrow_writer import ArrowWriter
    import huggingface_hub.constants as hub

    monkeypatch.setenv("DEEPSEEK4_HF_CACHE", str(tmp_path))
    monkeypatch.delenv("SOCIAL_IQA_CACHE_DIR", raising=False)
    from deepseek4._vendor.niahlogkv import eval as evaluator

    cache = tmp_path / "hf_cache"
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", cache)
    monkeypatch.setattr(datasets.config, "HF_DATASETS_OFFLINE", True)
    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(hub, "HF_HUB_OFFLINE", True)
    snapshot = (
        cache / "social_i_qa" / "default" / "0.1.0"
        / "674d85e42ac7430d3dcd4de7007feaffcb1527c535121e09bab2803fbcc925f8"
    )
    snapshot.mkdir(parents=True)
    features = datasets.Features({
        key: datasets.Value("string")
        for key in ("context", "question", "answerA", "answerB", "answerC")
    })
    features["label"] = datasets.ClassLabel(names=["1", "2", "3"])
    splits = datasets.SplitDict()
    for split in ("train", "validation"):
        example = {key: split for key in features if key != "label"}
        example["label"] = 0
        with ArrowWriter(path=str(snapshot / f"social_i_qa-{split}.arrow"), features=features) as writer:
            writer.write(example)
            _, num_bytes = writer.finalize()
        splits.add(datasets.SplitInfo(name=split, num_examples=1, num_bytes=num_bytes))
    datasets.DatasetInfo(
        dataset_name="social_i_qa", config_name="default", version="0.1.0",
        features=features, splits=splits,
    ).write_to_directory(str(snapshot))

    path = "allenai/social_i_qa"
    with pytest.raises(ConnectionError, match="OfflineModeIsEnabled"):
        datasets.load_dataset(path=path)
    with evaluator._trace_hf_cache_resolution():
        loaded = datasets.load_dataset(path=path)
        assert loaded["validation"][0]["context"] == "validation"
        assert loaded["validation"].features == features
        assert '"event": "explicit_cache_loaded"' in capsys.readouterr().err
        with pytest.raises(ConnectionError):
            datasets.load_dataset(path=path, name="other")

        relocated = tmp_path / "relocated_snapshot"
        snapshot.rename(relocated)
        monkeypatch.setenv("SOCIAL_IQA_CACHE_DIR", str(relocated))
        assert datasets.load_dataset(path=path)["train"][0]["context"] == "train"
        (relocated / "social_i_qa-validation.arrow").unlink()
        with pytest.raises(RuntimeError, match="social_i_qa-validation.arrow"):
            datasets.load_dataset(path=path)
