"""Dense-CPT control: local configuration, provenance and ordinary causal attention."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from chunkkv import config as experiment
from chunkkv import train

from .conftest import tiny_config


REAL_ROOT = train.RUNTIME_ROOT


@pytest.fixture
def reference(tmp_path, monkeypatch):
    """A standalone ChunkKV project: the only experiment YAMLs live in this project."""
    root = tmp_path / "chunkKV"
    runtime = root / "chunkkv/_vendor/semanticlogkv"
    runtime.mkdir(parents=True)
    (runtime / "demo.py").write_text("# bundled trainer\n")
    (runtime / "SOURCE.json").write_text("{}\n")
    (root / "configs").mkdir()
    (root / "configs/base.yaml").write_text(yaml.safe_dump(dict(
        ckpt_dir="weights/base", save_path=str(tmp_path / "ckpt/semantic"), max_steps=1200,
        data_dir="data/prolong", dataset_dir="data", learning_rate="5e-6",
        log_kv_semantic_clusters=True, run_eval="after", expid="semantic")))
    (root / "configs/run.yaml").write_text("config: base.yaml\nmax_steps: 600\nresume_dir: /ckpt/semantic/step_400\n")
    (root / "weights/base").mkdir(parents=True)
    (root / "weights/base/lit_model.pth").write_bytes(b"base weights")
    monkeypatch.setattr(train, "BASELINE_ROOT", root)
    monkeypatch.setattr(train, "RUNTIME_ROOT", runtime)
    monkeypatch.setattr(experiment, "BASELINE_ROOT", root)
    return root


def write_spec(path, **values):
    path.write_text(yaml.safe_dump({"experiment_config": "run.yaml", **values}))
    return path


def test_resolution_keeps_the_local_experiment_and_only_switches_off_logkv(reference):
    spec = write_spec(reference / "configs/dense.yaml", save_path="outputs/dense", overrides={"num_workers": 2})
    resolved, record = train.resolve_training_config(spec)
    assert resolved["max_steps"] == 600 and resolved["learning_rate"] == "5e-6"  # demo.py coerces it
    assert resolved["ckpt_dir"] == str(reference / "weights/base") and resolved["num_workers"] == 2
    assert resolved["data_dir"] == str(reference / "data/prolong")
    assert {key: resolved[key] for key in train.FORCED} == train.FORCED
    assert resolved["save_path"] == str(reference / "outputs/dense")
    assert resolved["tensorboard_root"] == str(reference / "outputs/dense_tensorboard")
    assert resolved["expid"] == "semantic-dense-control"
    assert list(record["experiment_config_files"]) == [str(reference / "configs/base.yaml"), str(reference / "configs/run.yaml")]
    assert record["resolved_config"] == resolved
    assert set(record["trainer_code"]) == {"demo.py", "SOURCE.json"}
    assert set(record["control_code"]) == {"train.py", "config.py"}


@pytest.mark.parametrize("save_path,overrides,message", [
    ("chunkkv/_vendor/semanticlogkv/outputs/dense", {}, "outside the bundled runtime"),
    ("{tmp}/ckpt/semantic/dense", {}, "SemanticLogKV run"),
    ("weights/base/new", {}, "Base"),
    ("/data/lihourun/semanticLogKV/outputs/dense", {}, "SemanticLogKV run"),
    ("{tmp}", {}, "SemanticLogKV run"),  # would contain the Base checkpoint and source tree
    ("outputs/dense", {"ckpt_dir": "/another/base"}, "fixed"),
    ("outputs/dense", {"log_kv_semantic_clusters": True}, "fixed"),
])
def test_outputs_and_comparison_inputs_are_protected(reference, tmp_path, save_path, overrides, message):
    spec = write_spec(reference / "configs/dense.yaml", overrides=overrides,
                      save_path=save_path.format(tmp=tmp_path))
    with pytest.raises(ValueError, match=message):
        train.resolve_training_config(spec)


def test_prepare_records_provenance_and_refuses_to_mix_runs(reference):
    spec = write_spec(reference / "configs/dense.yaml", save_path="outputs/dense")
    resolved = train.prepare(spec)
    save_path = Path(resolved["save_path"])
    manifest = json.loads((save_path / train.MANIFEST).read_text())
    assert manifest["base_weight_sha256"] == train.sha256(reference / "weights/base/lit_model.pth")
    assert yaml.safe_load((save_path / train.RESOLVED).read_text()) == resolved
    assert train.prepare(spec) == resolved  # resume of the same control
    write_spec(spec, save_path=str(save_path), overrides={"num_workers": 1})
    with pytest.raises(ValueError, match="different dense-control"):
        train.prepare(spec)
    occupied = reference / "outputs/occupied"
    occupied.mkdir()
    (occupied / "lit_model.pth").write_text("other run")
    with pytest.raises(FileExistsError):
        train.prepare(write_spec(spec, save_path=str(occupied)))


def test_run_uses_project_cwd_and_resolves_relative_override_inputs(reference, tmp_path, monkeypatch):
    spec = write_spec(reference / "configs/dense.yaml", save_path="outputs/dense",
                      overrides={"data_dir": "data/custom"})
    resolved = train.prepare(spec)
    calls = []
    def main(config):
        calls.append((Path.cwd(), yaml.safe_load(Path(config).read_text())))
    monkeypatch.setattr(train, "load_trainer", lambda: SimpleNamespace(main=main))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.chdir(tmp_path)
    train.run(spec)
    assert calls == [(reference, resolved)]
    assert calls[0][1]["data_dir"] == str(reference / "data/custom")
    assert Path.cwd() == tmp_path


def test_shipped_control_only_reads_the_project_experiment():
    resolved, record = train.resolve_training_config(train.DEFAULT_CONFIG)
    source = experiment.load_experiment_config()
    for key in ("ckpt_dir", "max_steps", "learning_rate", "min_lr", "warmup_steps", "global_batch_size",
                "micro_batch_size", "context_length", "data_dir", "arch_name"):
        assert resolved[key] == source[key]
    assert resolved["log_kv_semantic_clusters"] is False and resolved["resume_dir"] is None
    assert Path(resolved["save_path"]).is_relative_to(train.BASELINE_ROOT / "outputs")
    assert record["overrides"] == {}
    assert list(record["experiment_config_files"]) == [str(experiment.DEFAULT_EXPERIMENT_CONFIG)]
    assert {"demo.py", "data.py", "utils.py", "litgpt/model.py", "SOURCE.json"} <= set(record["trainer_code"])


def test_local_config_expands_env_and_preserves_absolute_checkpoint_string(reference, monkeypatch):
    monkeypatch.setenv("CHUNKKV_TEST_BASE", "/weights/base/")
    monkeypatch.setenv("CHUNKKV_TEST_EMPTY", "")
    monkeypatch.delenv("CHUNKKV_TEST_MISSING", raising=False)
    assert experiment.expand_env_vars({"x": ["${CHUNKKV_TEST_MISSING-fallback}", "${CHUNKKV_TEST_BASE}"],
                                       "empty": ("${CHUNKKV_TEST_EMPTY:-default}",), "number": 7}) == {
        "x": ["fallback", "/weights/base/"], "empty": ("",), "number": 7}
    config = reference / "configs/assets.yaml"
    config.write_text("ckpt_dir: ${CHUNKKV_TEST_BASE}\ndata_dir: data/corpus\n")
    assert experiment.load_experiment_config(config) == {"ckpt_dir": "/weights/base/", "data_dir": str(reference / "data/corpus")}


def test_dense_control_gpt_trains_with_ordinary_causal_attention():
    import torch
    from chunkkv._vendor.semanticlogkv.litgpt.model import GPT

    torch.manual_seed(0)
    config = tiny_config()
    reference = GPT(config)
    control = train.make_dense_gpt(GPT)(config)
    control.load_state_dict(reference.state_dict())
    # The exact call demo.py makes before its training loop.
    control.enable_log_kv_training(batch_size=1, max_seq_length=64, B=8, recent_size=16, train_block=8,
                                   semantic_clusters=False, alpha_exact_tokens=0)
    assert control.transformer.h[0].attn.kv_cache.B.startswith("none")
    assert not any(block.attn.training_log_kv for block in control.transformer.h)
    tokens = torch.randint(0, 64, (1, 33))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    torch.testing.assert_close(control(inputs, targets=targets), reference(inputs, targets=targets))
    assert set(control.state_dict()) == set(reference.state_dict())


def test_private_trainer_uses_the_dense_model_without_touching_runtime_files():
    pytest.importorskip("litdata")
    import torch
    from chunkkv._vendor.semanticlogkv.litgpt.model import GPT

    contents = {path: path.read_bytes() for path in REAL_ROOT.rglob("*.py")}
    original_path = list(sys.path)
    # demo.py sets process-wide bf16 defaults on import; keep them out of other tests.
    dtype, precision = torch.get_default_dtype(), torch.get_float32_matmul_precision()
    try:
        trainer = train.load_trainer()
    finally:
        torch.set_default_dtype(dtype)
        torch.set_float32_matmul_precision(precision)
    assert trainer.GPT.__mro__[1] is GPT and trainer.GPT.__name__ == "DenseControlGPT"
    from chunkkv._vendor.semanticlogkv.litgpt import model
    assert model.GPT is GPT
    assert sys.path == original_path
    assert {path: path.read_bytes() for path in contents} == contents
