"""Dense-CPT control: config resolution, output guards and the dense GPT switch."""

import json
from pathlib import Path

import pytest
import torch
import yaml

import litgpt
from litgpt.model import GPT

from chunkkv import train

from .conftest import tiny_config


REAL_ROOT = Path(litgpt.__file__).resolve().parents[1]


@pytest.fixture
def reference(tmp_path):
    """A minimal litgpt checkout layout with a two-file experiment chain."""
    root = tmp_path / "litgpt"
    for name in train.TRAINER_CODE:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(f"# {name}\n")
    (root / "exp").mkdir()
    (root / "exp/base.yaml").write_text(yaml.safe_dump(dict(
        ckpt_dir=str(tmp_path / "base"), save_path=str(tmp_path / "ckpt/semantic"), max_steps=1200,
        learning_rate="5e-6", log_kv_semantic_clusters=True, run_eval="after", expid="semantic")))
    (root / "exp/run.yaml").write_text("config: base.yaml\nmax_steps: 600\nresume_dir: /ckpt/semantic/step_400\n")
    (tmp_path / "base").mkdir()
    (tmp_path / "base/lit_model.pth").write_bytes(b"base weights")
    return root


def write_spec(path, **values):
    path.write_text(yaml.safe_dump({"semantic_config": "exp/run.yaml", **values}))
    return path


def test_resolution_keeps_the_experiment_and_only_switches_off_logkv(reference, tmp_path):
    spec = write_spec(tmp_path / "dense.yaml", save_path=str(tmp_path / "out/dense"),
                      overrides={"num_workers": 2})
    resolved, record = train.resolve_training_config(reference, spec, expand_env_vars=lambda x: x)
    assert resolved["max_steps"] == 600 and resolved["learning_rate"] == "5e-6"  # demo.py coerces it
    assert resolved["ckpt_dir"] == str(tmp_path / "base") and resolved["num_workers"] == 2
    assert {key: resolved[key] for key in train.FORCED} == train.FORCED
    assert resolved["save_path"] == str(tmp_path / "out/dense")
    assert resolved["tensorboard_root"] == str(tmp_path / "out/dense_tensorboard")
    assert resolved["expid"] == "semantic-dense-control"
    assert list(record["semantic_config_files"]) == [str(reference / "exp/base.yaml"), str(reference / "exp/run.yaml")]
    assert record["resolved_config"] == resolved and set(record["trainer_code"]) == set(train.TRAINER_CODE)


@pytest.mark.parametrize("save_path,overrides,message", [
    ("{root}/outputs/dense", {}, "outside litgpt"),
    ("{tmp}/ckpt/semantic/dense", {}, "SemanticLogKV run"),
    ("{tmp}", {}, "SemanticLogKV run"),  # would contain the Base checkpoint
    ("{tmp}/out/dense", {"ckpt_dir": "/another/base"}, "fixed"),
    ("{tmp}/out/dense", {"log_kv_semantic_clusters": True}, "fixed"),
])
def test_outputs_and_comparison_inputs_are_protected(reference, tmp_path, save_path, overrides, message):
    spec = write_spec(tmp_path / "dense.yaml", overrides=overrides,
                      save_path=save_path.format(root=reference, tmp=tmp_path))
    with pytest.raises(ValueError, match=message):
        train.resolve_training_config(reference, spec, expand_env_vars=lambda x: x)


def test_prepare_records_provenance_and_refuses_to_mix_runs(reference, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "reference_expand_env_vars", lambda root: (lambda x: x))
    spec = write_spec(tmp_path / "dense.yaml", save_path=str(tmp_path / "out/dense"))
    resolved = train.prepare(reference, spec)
    save_path = Path(resolved["save_path"])
    manifest = json.loads((save_path / train.MANIFEST).read_text())
    assert manifest["base_weight_sha256"] == train.sha256(tmp_path / "base/lit_model.pth")
    assert yaml.safe_load((save_path / train.RESOLVED).read_text()) == resolved
    assert train.prepare(reference, spec) == resolved  # resume of the same control
    write_spec(spec, save_path=str(save_path), overrides={"num_workers": 1})
    with pytest.raises(ValueError, match="different dense-control"):
        train.prepare(reference, spec)
    occupied = tmp_path / "out/occupied"
    occupied.mkdir()
    (occupied / "lit_model.pth").write_text("other run")
    with pytest.raises(FileExistsError):
        train.prepare(reference, write_spec(spec, save_path=str(occupied)))


def test_shipped_control_resolves_against_the_reference_experiment():
    resolved, record = train.resolve_training_config(REAL_ROOT, train.DEFAULT_CONFIG)
    source, _ = train.load_yaml_chain(REAL_ROOT / "exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml")
    for key in ("ckpt_dir", "max_steps", "learning_rate", "min_lr", "warmup_steps", "global_batch_size",
                "micro_batch_size", "context_length", "data_dir", "arch_name"):
        assert resolved[key] == source[key]
    assert resolved["log_kv_semantic_clusters"] is False and resolved["resume_dir"] is None
    assert Path(resolved["save_path"]).is_relative_to(train.BASELINE_ROOT / "outputs")
    assert record["overrides"] == {}


def test_dense_control_gpt_trains_with_ordinary_causal_attention():
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


def test_private_trainer_uses_the_dense_model_without_touching_reference_files():
    pytest.importorskip("litdata")
    paths = [REAL_ROOT / name for name in train.TRAINER_CODE]
    contents = {path: path.read_bytes() for path in paths}
    # demo.py sets process-wide bf16 defaults on import; keep them out of other tests.
    dtype, precision = torch.get_default_dtype(), torch.get_float32_matmul_precision()
    try:
        trainer = train.load_trainer(REAL_ROOT)
    finally:
        torch.set_default_dtype(dtype)
        torch.set_float32_matmul_precision(precision)
    assert trainer.GPT.__mro__[1] is GPT and trainer.GPT.__name__ == "DenseControlGPT"
    import litgpt.model
    assert litgpt.model.GPT is GPT  # the package itself still exposes the native model
    assert {path: path.read_bytes() for path in paths} == contents
