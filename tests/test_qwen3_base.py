"""Offline checks using tiny random weights; no model download or CUDA kernels."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3ForCausalLM

from dms.core import DMSTrainingStateAux
from dms.cache import DMSCache
from dms.training.data import ConfiguredTokenizer
from dms.training.engine import (
    DistillationModelArguments, ModelArguments, _configure_gradients, dms_schedule,
    get_student_model,
)
from models.qwen3.configuration_qwen3_dms import Qwen3ConfigDMS
from models.qwen3.modeling_qwen3_dms import Qwen3ForCausalLMDMS
from models.qwen3.train import (
    _create_train_dataset, export_trained_student, extract_student_model, resolve_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(**overrides):
    kwargs = dict(vocab_size=32, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=16, max_position_embeddings=2048, tie_word_embeddings=True,
                  bos_token_id=0, eos_token_id=0, pad_token_id=0, dms_separate_alpha=True,
                  dms_window_size=512, dms_paged_attention_block_size=256,
                  dms_preallocate_for_tokens=2048)
    kwargs.update(overrides)
    return Qwen3ConfigDMS(**kwargs)


@pytest.fixture
def tokenizer():
    backend = Tokenizer(WordLevel({"<eos>": 0, "<unk>": 1, "blue": 2, "key": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>",
                                  pad_token="<eos>", unk_token="<unk>")


def test_base_template_and_smoke_config(tokenizer):
    for name in ("qwen3_1.7b_base", "qwen3_1.7b_base_smoke"):
        cfg = yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text())
        assert cfg["model"]["name"] == "outputs/semanticlogkv_base"
        assert cfg["dms"]["separate_alpha"] and not cfg["dms"]["disable_eviction"]
        assert cfg["data"]["max_length"] > cfg["dms"]["window_size"] > 256
        configured = ConfiguredTokenizer(tokenizer, cfg["data"]["tokenizer_kwargs"])
        assert configured.apply_chat_template([
            {"role": "user", "content": "key"}, {"role": "assistant", "content": "blue"},
        ]) == "key\n\nblue<eos>"


def test_full_base_adapter_dimensions_without_allocating_weights():
    # Dimensions from Qwen/Qwen3-1.7B-Base/config.json, verified 2026-10-12.
    config = tiny_config(hidden_size=2048, intermediate_size=6144, num_hidden_layers=28,
                         num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                         vocab_size=151936, max_position_embeddings=32768)
    with torch.device("meta"):
        model = Qwen3ForCausalLMDMS(config)
    _, trainable = _configure_gradients(model)
    assert len(trainable) == 56
    assert all("dms_proj_alpha" in name for name in trainable)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 516096
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_load_base_weights_and_freeze(tmp_path):
    base = Qwen3ForCausalLM(tiny_config())
    base.save_pretrained(tmp_path)
    args = DistillationModelArguments(student=ModelArguments(model_name_or_path=str(tmp_path), dtype="float32"))
    model = get_student_model(args, Qwen3ForCausalLMDMS, True, {"dms_separate_alpha": True})
    assert model.lm_head.weight is model.model.embed_tokens.weight
    for name, param in model.named_parameters():
        assert param.requires_grad == ("dms_proj_alpha" in name)
        if "dms_proj_alpha" in name and "norm" not in name:
            assert torch.count_nonzero(param) == 0
    torch.testing.assert_close(model.model.embed_tokens.weight, base.model.embed_tokens.weight)


def test_export_roundtrip_preserves_adapters_and_tied_weights(tmp_path, tokenizer):
    model = Qwen3ForCausalLMDMS(tiny_config())
    model.config.dms_cr = 16
    model.config.semanticlogkv_base = {"source_weight_sha256": "original-base-digest"}
    extract_student_model(SimpleNamespace(student_model=model), tokenizer, str(tmp_path))
    restored = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True, dtype=torch.bfloat16)
    assert restored.config.dms_separate_alpha
    assert restored.config.dms_cr == 16
    assert restored.config.semanticlogkv_base == model.config.semanticlogkv_base
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    assert next(model.parameters()).dtype == torch.float32  # export must not mutate live shards
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value.bfloat16())


def test_export_gathers_all_ranks_but_writes_only_on_main(tmp_path, tokenizer):
    model = Qwen3ForCausalLMDMS(tiny_config())
    combined = SimpleNamespace(student_model=model)
    calls = []
    accelerator = SimpleNamespace(
        wait_for_everyone=lambda: calls.append("barrier"),
        get_state_dict=lambda _: {f"student_model.{k}": v for k, v in model.state_dict().items()},
        state=SimpleNamespace(fsdp_plugin=SimpleNamespace(set_state_dict_type=lambda t: calls.append(t))),
    )
    trainer = SimpleNamespace(accelerator=accelerator, model_wrapped=combined,
                              is_fsdp_enabled=True, is_world_process_zero=lambda: False)
    export_trained_student(trainer, combined, tokenizer, str(tmp_path / "model"))
    assert calls == ["barrier", "FULL_STATE_DICT", "barrier"]
    assert not (tmp_path / "model").exists()
    trainer.is_world_process_zero = lambda: True
    export_trained_student(trainer, combined, tokenizer, str(tmp_path / "model"))
    assert (tmp_path / "model/config.json").is_file()


def test_local_text_pipeline_and_empty_input(tmp_path, tokenizer, monkeypatch):
    import dms.training.data as data
    monkeypatch.setattr(data, "_DEFAULT_FILTER_NUM_PROC", None)
    monkeypatch.setattr(data, "_DEFAULT_TOKENIZE_NUM_PROC", None)
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"text": "blue key " * 40}) + "\n")
    configured = ConfiguredTokenizer(tokenizer, {"chat_template": "{{ messages[0].content }}{{ eos_token }}"})
    cfg = dict(train_file=str(path), train_samples=4, max_length=16, concat_always_start_new=False)
    ds = _create_train_dataset(cfg, configured)
    assert len(ds) == 4
    assert len(ds[0]["input_ids"]) == 16
    assert all(ds[0]["attention_mask"])
    path.write_text(json.dumps({"text": "blue"}) + "\n")
    with pytest.raises(ValueError, match="no complete contexts"):
        _create_train_dataset(cfg, configured)


def test_resume_never_silently_ignores_a_bad_path(tmp_path):
    cfg = {"hf_trainer": {"output_dir": str(tmp_path), "resume_from_checkpoint": "auto"}}
    assert resolve_checkpoint(cfg) is None
    for step in (2, 10):
        (tmp_path / f"checkpoint-{step}").mkdir()
    assert resolve_checkpoint(cfg).endswith("checkpoint-10")
    cfg["hf_trainer"]["resume_from_checkpoint"] = str(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(cfg)


def test_schedule_saturates_at_target():
    args = SimpleNamespace(max_steps=544)
    assert dms_schedule(0, args, 1, 16, 510) == (0, 1)
    assert dms_schedule(510, args, 1, 16, 510) == (15 / 16, 16)
    assert dms_schedule(544, args, 1, 16, 510) == (15 / 16, 16)


def test_compression_metric_counts_the_recent_window():
    cache = DMSCache(dms_window_size=513, max_context_length=2048)
    layer = cache[0]
    layer.cumulative_length = 1024
    layer.cont_cache.cache_seq_lengths = torch.tensor([513, 513])
    layer.paged_cache.cache_seq_lengths = torch.tensor([511, 511])
    assert cache.get_cr() == 1.0
    # One-token generation can end after prefill, before this window moves to paged storage.
    layer.cumulative_length = 100
    layer.cont_cache.cache_seq_lengths = torch.tensor([100, 100])
    layer.paged_cache.cache_seq_lengths = None
    assert cache.get_cr() == 1.0
    layer.cumulative_length = 1024
    layer.cont_cache.cache_seq_lengths = None
    layer.paged_cache.cache_seq_lengths = torch.tensor([512, 512])
    assert cache.get_cr() == 2.0


def test_teacher_matches_vanilla_and_student_has_adapter_gradients(monkeypatch):
    # Eager CPU mathematical check only; compiled CUDA kernels are covered separately.
    import models.qwen3.modeling_qwen3_dms as modeling
    monkeypatch.setattr(torch, "compile", lambda fn=None, **kw: fn if fn else lambda f: f)
    for name in ("prepare_attention_input", "post_process_attention_output"):
        fn = getattr(modeling, name)
        monkeypatch.setattr(modeling, name, fn._torchdynamo_orig_callable)
    config = tiny_config(dms_window_size=2, dms_paged_attention_block_size=1,
                         dms_initial_alpha_offset=0, dms_alpha_scale=1,
                         dms_tau=1, _attn_implementation="eager")
    vanilla = Qwen3ForCausalLM(config).eval()
    student = Qwen3ForCausalLMDMS(config).train()
    missing, unexpected = student.load_state_dict(vanilla.state_dict(), strict=False)
    assert not unexpected and all("dms_proj_alpha" in k for k in missing)
    _configure_gradients(student)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    state = DMSTrainingStateAux(True, None, 0, 0, 0, None, 0, (2, 1, 2, 8))
    reference = vanilla(ids).logits
    teacher = student(ids, dms_state=state).logits
    torch.testing.assert_close(teacher, reference, atol=2e-5, rtol=2e-5)
    state.dms_teacher_mode = False
    state.target_frac_to_close = 0.9
    state.noise = torch.zeros(state.kv_cache_shape)
    result = student(ids, dms_state=state)
    (result.logits.square().mean() + result.dms_loss).backward()
    gradients = [p.grad for n, p in student.named_parameters() if "dms_proj_alpha" in n]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    assert all(p.grad is None for n, p in student.named_parameters() if "dms_proj_alpha" not in n)
