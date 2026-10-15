"""Tiny random Qwen3 checkpoints and tokenizers for CPU tests (no downloads)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from pyramidkv.qwen3 import Qwen3, Qwen3Config

TINY = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
                   rope_theta=10000.0, max_position_embeddings=256)


def tiny_model(seed: int = 0, max_seq_length: int = 128, config: Qwen3Config = TINY) -> Qwen3:
    torch.manual_seed(seed)
    model = Qwen3(config, max_seq_length=max_seq_length)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm_1.weight") or name.endswith("norm_2.weight") or "norm_" in name or "ln_f" in name:
                param.uniform_(0.5, 1.5)
            elif param.dim() == 2:
                param.normal_(0.0, 0.2 if "wte" not in name else 1.0)
    return model.float().eval()


def litgpt_model_config(config: Qwen3Config) -> dict:
    return dict(name="tiny-qwen3", block_size=config.max_position_embeddings, vocab_size=config.vocab_size,
                padded_vocab_size=config.vocab_size, n_layer=config.num_hidden_layers,
                n_head=config.num_attention_heads, head_size=config.head_dim, n_embd=config.hidden_size,
                n_query_groups=config.num_key_value_heads, rotary_percentage=1.0, parallel_residual=False,
                bias=False, norm_class_name="RMSNorm", norm_eps=config.rms_norm_eps, norm_qk=True,
                norm_qk_type="default", mlp_class_name="LLaMAMLP", intermediate_size=config.intermediate_size,
                rope_base=int(config.rope_theta), rope_adjustments=None, sliding_window_size=None)


def hf_config(config: Qwen3Config, tie: bool) -> dict:
    return dict(architectures=["Qwen3ForCausalLM"], model_type="qwen3", vocab_size=config.vocab_size,
                hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads, head_dim=config.head_dim, hidden_act="silu",
                rms_norm_eps=config.rms_norm_eps, rope_theta=config.rope_theta, rope_scaling=None,
                max_position_embeddings=config.max_position_embeddings, attention_bias=False,
                tie_word_embeddings=tie, use_sliding_window=False, torch_dtype="float32")


def save_litgpt_checkpoint(model: Qwen3, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.clone() for k, v in model.state_dict().items()}, directory / "lit_model.pth")
    (directory / "model_config.yaml").write_text(yaml.safe_dump(litgpt_model_config(model.config)))
    return directory


def litgpt_to_hf(state: dict, config: Qwen3Config, tie: bool) -> dict:
    out = {"model.embed_tokens.weight": state["transformer.wte.weight"],
           "model.norm.weight": state["transformer.ln_f.weight"]}
    if not tie:
        out["lm_head.weight"] = state["lm_head.weight"]
    names = {"norm_1.weight": "input_layernorm.weight", "norm_2.weight": "post_attention_layernorm.weight",
             "attn.proj.weight": "self_attn.o_proj.weight", "attn.norm_q.weight": "self_attn.q_norm.weight",
             "attn.norm_k.weight": "self_attn.k_norm.weight", "mlp.fc_1.weight": "mlp.gate_proj.weight",
             "mlp.fc_2.weight": "mlp.up_proj.weight", "mlp.proj.weight": "mlp.down_proj.weight"}
    q_size = config.num_attention_heads * config.head_dim
    kv_size = config.num_key_value_heads * config.head_dim
    for layer in range(config.num_hidden_layers):
        prefix = f"transformer.h.{layer}."
        for lit, hf in names.items():
            out[f"model.layers.{layer}.{hf}"] = state[prefix + lit]
        q, k, v = state[prefix + "attn.qkv.weight"].split((q_size, kv_size, kv_size))
        out[f"model.layers.{layer}.self_attn.q_proj.weight"] = q
        out[f"model.layers.{layer}.self_attn.k_proj.weight"] = k
        out[f"model.layers.{layer}.self_attn.v_proj.weight"] = v
    return {k: v.clone().contiguous() for k, v in out.items()}


def save_hf_checkpoint(model: Qwen3, directory: Path, tie: bool = False) -> Path:
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    if tie:
        with torch.no_grad():
            model.lm_head.weight.copy_(model.transformer.wte.weight)
    save_file(litgpt_to_hf(model.state_dict(), model.config, tie), str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(hf_config(model.config, tie)))
    return directory


def save_word_tokenizer(directory: Path, vocab_size: int = 64) -> list[str]:
    """Whitespace word-level tokenizer; id 0 is ``<|endoftext|>`` used as BOS and EOS like Qwen."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders

    words = ["<|endoftext|>", "[UNK]"] + [f"w{i}" for i in range(vocab_size - 2)]
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.add_special_tokens(["<|endoftext|>"])
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps(
        {"bos_token": None, "eos_token": "<|endoftext|>", "tokenizer_class": "Qwen2Tokenizer"}))
    (directory / "generation_config.json").write_text(json.dumps({"bos_token_id": 0, "eos_token_id": 0}))
    return words
