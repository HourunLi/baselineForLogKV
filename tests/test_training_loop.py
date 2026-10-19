"""dms.training.loop on CPU with a tiny model: settings, data order, exact resume, rotation and export.

Equality with the upstream HF Trainer path itself needs accelerate, which torch218 lacks; docs/VALIDATION.md
records that comparison.
"""

import itertools
import json
import random

import pytest
import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

import dms.training.data as data
from dms.training.data import ConfiguredTokenizer
from dms.training.engine import DistillationModelArguments, DMSTrainerState, ModelArguments
from dms.training.loop import DMSTrainer, Runtime, TrainerSettings, checkpoints, epoch_indices, update_groups
from models.qwen3.modeling_qwen3_dms import Qwen3ForCausalLMDMS
from models.qwen3.train import _create_train_dataset, build_combined_model, export_trained_student
from tests.test_qwen3_base import ROOT

WORDS = [f"w{i}" for i in range(30)]
TEMPLATE = "{{ messages | map(attribute='content') | join('\n\n') }}{{ eos_token }}"
# Eviction acts within 64-token samples: a window shorter than the sample.
DMS = dict(alpha_scale=100.0, initial_alpha_offset=5.0, window_size=17, paged_attention_block_size=16,
           disable_eviction=False, separate_alpha=True, alpha_per="head", tau=0.1, initial_cr=1.0, final_cr=4.0,
           final_step=4)
TRAINER = dict(max_steps=5, learning_rate=3e-2, gradient_accumulation_steps=4, adam_beta1=0.9, adam_beta2=0.95,
               max_grad_norm=1.0, seed=42, save_steps=2, save_total_limit=2, logging_steps=1)


@pytest.mark.parametrize("name", ["qwen3_1.7b_base", "qwen3_1.7b_base_smoke"])
def test_recipe_settings_are_implemented(name):
    cfg = yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text())
    settings = TrainerSettings.from_config(cfg["trainer"])
    if name == "qwen3_1.7b_base":
        assert (settings.max_steps, settings.gradient_accumulation_steps, settings.learning_rate) == (544, 16, 3e-5)
        assert (settings.adam_beta1, settings.adam_beta2, settings.adam_epsilon) == (0.9, 0.95, 1e-8)
        assert settings.fsdp_options == {"full_shard", "offload", "auto_wrap"}
        assert settings.fsdp_config == {"use_orig_params": True, "sync_module_states": True,
                                        "activation_checkpointing": True, "state_dict_type": "FULL_STATE_DICT",
                                        "transformer_layer_cls_to_wrap": ["Qwen3DecoderLayerDMS"]}
    else:
        assert not settings.fsdp_options and settings.fsdp_config == {}


@pytest.mark.parametrize("change,message", [
    ({"lr_scheduler_type": "cosine"}, "lr_scheduler_type"),
    ({"per_device_train_batch_size": 2}, "per_device_train_batch_size"),
    ({"weight_decay": 0.1}, "weight_decay"),
    ({"gradient_checkpointing": True}, "gradient_checkpointing"),
    ({"optim": "adafactor"}, "Unsupported trainer settings"),
    ({"fsdp": "shard_grad_op auto_wrap"}, "trainer.fsdp must be"),
    ({"fsdp_config": {"use_orig_params": False}}, "use_orig_params"),
    ({"fsdp_config": {"backward_prefetch": "BACKWARD_PRE",
                      "transformer_layer_cls_to_wrap": "Qwen3DecoderLayerDMS"}}, "supports no"),
    ({"fsdp": "", "fsdp_config": {"use_orig_params": True}}, "needs trainer.fsdp"),
    ({"save_total_limit": 0}, "save_total_limit"),
])
def test_unimplemented_settings_are_refused(change, message):
    cfg = yaml.safe_load((ROOT / "configs/qwen3_1.7b_base.yaml").read_text())["trainer"]
    with pytest.raises(ValueError, match=message):
        TrainerSettings.from_config({**cfg, **change})


def test_data_order_shards_and_pads_like_accelerate():
    # 5 samples on 2 processes: process r takes positions r, r+2, ...; the epoch is padded from its start.
    order = torch.randperm(5, generator=torch.Generator().manual_seed(42 + 1)).tolist()
    assert epoch_indices(5, 42, 1, 2, 0) == [order[0], order[2], order[4]]
    assert epoch_indices(5, 42, 1, 2, 1) == [order[1], order[3], order[0]]
    assert epoch_indices(5, 42, 0, 1, 0) != epoch_indices(5, 42, 1, 1, 0)  # reshuffled every epoch
    # 10 batches, accumulation 4: groups of 4, 4 and 2, then the next epoch.
    groups = list(itertools.islice(update_groups(10, 42, 1, 0, 4), 4))
    assert [len(group) for _, group in groups] == [4, 4, 2, 4]
    assert [epoch for epoch, _ in groups] == [0.4, 0.8, 1.0, 1.4]


@pytest.fixture
def cpu_kernels(monkeypatch):
    """Eager FlexAttention and losses, and a CPU-only RNG fork for the DMS noise."""
    torch.compiler.set_stance("force_eager")
    fork = torch.random.fork_rng
    monkeypatch.setattr(torch.random, "fork_rng",
                        lambda devices=None, enabled=True, **kwargs: fork(devices=[], enabled=enabled, **kwargs))
    monkeypatch.setattr(data, "_DEFAULT_FILTER_NUM_PROC", None)
    monkeypatch.setattr(data, "_DEFAULT_TOKENIZE_NUM_PROC", None)
    yield
    torch.compiler.set_stance("default")


@pytest.fixture
def fixture(tmp_path):
    vocab = {"<eos>": 0, "<unk>": 1, **{word: i + 2 for i, word in enumerate(WORDS)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", pad_token="<eos>",
                                        unk_token="<unk>")
    base = tmp_path / "base"
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, max_position_embeddings=2048, tie_word_embeddings=True,
        bos_token_id=0, eos_token_id=0, pad_token_id=0))
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.2)
    model.save_pretrained(base)
    rng = random.Random(0)
    text = tmp_path / "data.jsonl"
    text.write_text("".join(json.dumps({"text": " ".join(rng.choice(WORDS) for _ in range(rng.randint(15, 70)))})
                            + "\n" for _ in range(40)))
    return tmp_path, base, text, tokenizer


def run(fixture, output, resume=None, runtime=None, **changes):
    tmp_path, base, text, tokenizer = fixture
    settings = TrainerSettings.from_config({**TRAINER, "output_dir": str(tmp_path / output), **changes})
    dataset = _create_train_dataset(dict(train_file=str(text), train_samples=10, max_length=64,
                                         concat_always_start_new=False),
                                    ConfiguredTokenizer(tokenizer, {"chat_template": TEMPLATE}))
    state = DMSTrainerState()
    arguments = ModelArguments(model_name_or_path=str(base), dtype="float32")
    combined = build_combined_model(DistillationModelArguments(student=arguments, teacher=arguments), settings,
                                    DMS, {"process_vocab_using_chunk": 16}, tokenizer, state)
    trainer = DMSTrainer(combined, dataset, settings, runtime or Runtime(), 64, state, resume_from=resume)
    return trainer, trainer.train()


def adapters(trainer):
    return trainer._adapter_state()


def test_resume_continues_the_uninterrupted_run_exactly(cpu_kernels, fixture):
    straight, _ = run(fixture, "straight")
    assert [path.name for path in checkpoints(fixture[0] / "straight")] == ["checkpoint-4", "checkpoint-5"]
    assert [entry["epoch"] for entry in straight.state.log_history] == [0.4, 0.8, 1.0, 1.4, 1.8]
    assert all(torch.count_nonzero(value) for name, value in adapters(straight).items() if "norm" not in name)

    run(fixture, "resumed", max_steps=2)  # stops inside the first epoch
    resumed, _ = run(fixture, "resumed", resume=str(fixture[0] / "resumed/checkpoint-2"))
    assert resumed.state.log_history == straight.state.log_history
    final, expected = adapters(resumed), adapters(straight)
    assert all(torch.equal(final[name], expected[name]) for name in expected)

    with pytest.raises(ValueError, match="different run setup"):
        run(fixture, "resumed", resume=str(fixture[0] / "resumed/checkpoint-5"), gradient_accumulation_steps=2)


def test_exported_student_carries_trained_adapters_and_record(cpu_kernels, fixture, tmp_path):
    trainer, seconds = run(fixture, "export", max_steps=2)
    tokenizer = fixture[3]
    path = tmp_path / "export/student_model"
    export_trained_student(trainer, trainer.combined, tokenizer, str(path), trainer.training_record(seconds))
    record = json.loads((path / "training_record.json").read_text())
    assert record["global_step"] == 2 and record["tokens_per_step"] == 4 * 64
    assert record["trainable_parameters"] == 2 * (64 * 2 + 64)  # per layer: alpha (KV heads x hidden) and norm
    assert record["versions"]["transformers"] and record["final_log"]["step"] == 2
    student = Qwen3ForCausalLMDMS.from_pretrained(path, torch_dtype=torch.bfloat16)
    exported = dict(student.named_parameters())
    for name, value in adapters(trainer).items():
        assert torch.equal(exported[name.removeprefix("student_model.")], value.bfloat16())
