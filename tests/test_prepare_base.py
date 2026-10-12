"""Real LitGPT -> HF conversion on a tiny offline Qwen3 fixture."""

import json
from dataclasses import asdict

import pytest
import torch
import yaml
from litgpt import Config, GPT
from litgpt.scripts.convert_lit_checkpoint import copy_weights_qwen_3
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from models.qwen3.prepare_base import (
    DEFAULT_LITGPT_ROOT, build_hf_model, load_semantic_config, prepare_base, sha256_file,
    verify_prepared_base,
)


@pytest.fixture
def source(tmp_path):
    config = Config(
        name="tiny-qwen3", block_size=64, n_layer=2, n_embd=32, n_head=4,
        head_size=8, n_query_groups=2, vocab_size=32, padded_vocab_size=32,
        norm_class_name="RMSNorm", norm_eps=1e-6, norm_qk=True,
        parallel_residual=False, rotary_percentage=1.0, bias=False,
        mlp_class_name="LLaMAMLP", intermediate_size=64, rope_base=1000000,
    )
    model = GPT(config).eval()
    model.lm_head.weight = model.transformer.wte.weight
    hf_config = Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, rms_norm_eps=1e-6, rope_theta=1000000,
        tie_word_embeddings=True, bos_token_id=0, eos_token_id=0, pad_token_id=0,
    )
    path = tmp_path / "original_base"
    path.mkdir()
    torch.save({"model": model.state_dict()}, path / "lit_model.pth")
    (path / "model_config.yaml").write_text(yaml.safe_dump(asdict(config)))
    hf_config.save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<eos>": 0, "<unk>": 1, "blue": 2}, unk_token="<unk>")),
        eos_token="<eos>", unk_token="<unk>",
    )
    tokenizer.save_pretrained(path)
    (path / "generation_config.json").write_text('{"eos_token_id": 0}\n')
    base_config = tmp_path / "base.yaml"
    base_config.write_text(yaml.safe_dump({"ckpt_dir": str(path), "context_length": 64}))
    selected_config = tmp_path / "selected.yaml"
    selected_config.write_text(yaml.safe_dump({
        "config": "base.yaml", "save_path": "/must/not/read/CPT", "resume_dir": "/must/not/read/resume",
    }))
    return path, model, hf_config, selected_config


def test_exact_export_round_trip_and_provenance(source, tmp_path):
    path, model, _, experiment = source
    output = tmp_path / "hf"
    manifest = prepare_base(DEFAULT_LITGPT_ROOT, experiment, output)
    assert verify_prepared_base(output) == manifest
    restored = Qwen3ForCausalLM.from_pretrained(output, local_files_only=True).eval()
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    assert manifest["source_checkpoint_dir"] == str(path)
    assert manifest["source_weight_sha256"] == sha256_file(path / "lit_model.pth")
    assert restored.config.semanticlogkv_base["source_weight_sha256"] == manifest["source_weight_sha256"]
    assert restored.config.semanticlogkv_base["tokenizer_files"] == manifest["tokenizer_files"]
    assert json.loads((output / "semanticlogkv_base_manifest.json").read_text()) == manifest
    assert manifest["prepared_weight_files"] == {"model.safetensors": sha256_file(output / "model.safetensors")}
    for name, digest in manifest["tokenizer_files"].items():
        assert sha256_file(output / name) == digest == sha256_file(path / name)
    expected = {}
    copy_weights_qwen_3(model.config, expected, model.state_dict())
    for name, weights in restored.state_dict().items():
        torch.testing.assert_close(weights, expected[name], rtol=0, atol=0)
    # This catches QKV order / QK normalization / RoPE errors missed by shapes.
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        torch.testing.assert_close(model(tokens), restored(tokens, use_cache=False).logits, atol=2e-5, rtol=2e-5)


def test_source_tied_head_mismatch_is_rejected(source):
    _, model, config, _ = source
    state = dict(model.state_dict())
    state["lm_head.weight"] = state["lm_head.weight"] + 1
    with pytest.raises(ValueError, match="LM head differs"):
        build_hf_model(model.config, config, state, copy_weights_qwen_3)


def test_missing_weights_and_config_mismatch_are_rejected(source):
    _, model, config, _ = source
    state = dict(model.state_dict())
    del state["transformer.h.0.attn.norm_q.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        build_hf_model(model.config, config, state, copy_weights_qwen_3)
    config.rope_theta = 10000
    with pytest.raises(ValueError, match="rope_theta"):
        build_hf_model(model.config, config, model.state_dict(), copy_weights_qwen_3)


def test_existing_export_is_not_overwritten(source, tmp_path):
    _, _, _, experiment = source
    output = tmp_path / "already_exists"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep me")
    with pytest.raises(FileExistsError):
        prepare_base(DEFAULT_LITGPT_ROOT, experiment, output)
    assert marker.read_text() == "keep me"


def test_yaml_inheritance_uses_original_base(source):
    path, _, _, experiment = source
    loaded = load_semantic_config(experiment)
    assert loaded["ckpt_dir"] == str(path)
    assert loaded["context_length"] == 64


@pytest.mark.parametrize("changed", ["model.safetensors", "tokenizer.json", "config.json", "provenance"])
def test_changed_prepared_artifacts_are_rejected(source, tmp_path, changed):
    _, _, _, experiment = source
    output = tmp_path / "prepared"
    prepare_base(DEFAULT_LITGPT_ROOT, experiment, output)
    if changed == "provenance":
        config_path = output / "config.json"
        config = json.loads(config_path.read_text())
        config["semanticlogkv_base"]["source_weight_sha256"] = "different-source"
        config_path.write_text(json.dumps(config))
        error = "provenance disagrees"
    elif changed == "config.json":
        config_path = output / changed
        config = json.loads(config_path.read_text())
        config["rope_theta"] *= 2
        config_path.write_text(json.dumps(config))
        error = "config hash mismatch"
    else:
        with (output / changed).open("ab") as stream:
            stream.write(b"changed")
        error = "hash mismatch"
    with pytest.raises(ValueError, match=error):
        verify_prepared_base(output)


def test_unprepared_base_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="run models.qwen3.prepare_base first"):
        verify_prepared_base(tmp_path)


def test_converted_base_to_dms_export_to_shared_model_loader(source, tmp_path, monkeypatch):
    from dms.training.engine import DistillationModelArguments, ModelArguments, get_student_model
    from models.qwen3 import check_env
    from models.qwen3.modeling_qwen3_dms import Qwen3ForCausalLMDMS
    from models.qwen3.semanticlogkv_eval import DMSModel

    original, _, _, experiment = source
    prepared, exported = tmp_path / "prepared", tmp_path / "student_model"
    manifest = prepare_base(DEFAULT_LITGPT_ROOT, experiment, prepared)
    args = DistillationModelArguments(student=ModelArguments(model_name_or_path=str(prepared), dtype="float32"))
    student = get_student_model(args, Qwen3ForCausalLMDMS, True, {"dms_separate_alpha": True})
    student.save_pretrained(exported)
    # Loader coverage only: never execute GPU inference on this CPU fixture.
    monkeypatch.setattr(check_env, "check_environment", lambda: None)
    facade = DMSModel.from_checkpoint(exported, original, 64, "cpu")
    assert facade.hf_model.config.semanticlogkv_base["source_weight_sha256"] == manifest["source_weight_sha256"]
    assert facade.max_seq_length == 64 and not facade.training
    assert next(facade.parameters()).dtype == torch.bfloat16
