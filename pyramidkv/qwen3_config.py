"""Qwen3 architecture settings and checkpoint-format detection, without torch.

Kept apart from :mod:`pyramidkv.qwen3` so ``--print-config`` can read the layer count
of the Base checkpoint without importing torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int  # embedding / lm_head rows (LitGPT ``padded_vocab_size``)
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int

    def __post_init__(self):
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")

    @classmethod
    def from_litgpt(cls, cfg: dict) -> "Qwen3Config":
        """Accept only the dense Qwen3 architecture; anything else changes the math."""
        unsupported = {
            "norm_class_name": ("RMSNorm", cfg.get("norm_class_name")),
            "mlp_class_name": ("LLaMAMLP", cfg.get("mlp_class_name")),
            "norm_qk": (True, cfg.get("norm_qk")),
            "norm_qk_type": ("default", cfg.get("norm_qk_type", "default")),
            "rotary_percentage": (1.0, cfg.get("rotary_percentage", 1.0)),
            "rope_condense_ratio": (1, cfg.get("rope_condense_ratio", 1)),
        }
        bad = [f"{key}={actual!r} (expected {expected!r})"
               for key, (expected, actual) in unsupported.items() if actual != expected]
        for flag in ("bias", "attn_bias", "lm_head_bias", "parallel_residual", "rope_interleave",
                     "rope_adjustments", "sliding_window_size", "attention_scores_scalar",
                     "attention_logit_softcapping", "final_logit_softcapping", "scale_embeddings",
                     "post_attention_norm", "post_mlp_norm", "shared_attention_norm"):
            if cfg.get(flag):
                bad.append(f"{flag}={cfg[flag]!r}")
        if bad:
            raise ValueError("Unsupported LitGPT architecture for Qwen3: " + ", ".join(bad))
        head_dim = cfg.get("head_size") or cfg["n_embd"] // cfg["n_head"]
        return cls(
            vocab_size=int(cfg.get("padded_vocab_size") or cfg["vocab_size"]),
            hidden_size=int(cfg["n_embd"]), intermediate_size=int(cfg["intermediate_size"]),
            num_hidden_layers=int(cfg["n_layer"]), num_attention_heads=int(cfg["n_head"]),
            num_key_value_heads=int(cfg["n_query_groups"]), head_dim=int(head_dim),
            rms_norm_eps=float(cfg["norm_eps"]), rope_theta=float(cfg["rope_base"]),
            max_position_embeddings=int(cfg["block_size"]),
        )

    @classmethod
    def from_hf(cls, cfg: dict) -> "Qwen3Config":
        bad = []
        if cfg.get("model_type", "qwen3") != "qwen3":
            bad.append(f"model_type={cfg.get('model_type')!r}")
        if cfg.get("hidden_act", "silu") != "silu":
            bad.append(f"hidden_act={cfg.get('hidden_act')!r}")
        for flag in ("rope_scaling", "attention_bias", "use_sliding_window"):
            if cfg.get(flag):
                bad.append(f"{flag}={cfg[flag]!r}")
        if bad:
            raise ValueError("Unsupported Hugging Face architecture for Qwen3: " + ", ".join(bad))
        heads = int(cfg["num_attention_heads"])
        return cls(
            vocab_size=int(cfg["vocab_size"]), hidden_size=int(cfg["hidden_size"]),
            intermediate_size=int(cfg["intermediate_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]), num_attention_heads=heads,
            num_key_value_heads=int(cfg.get("num_key_value_heads", heads)),
            head_dim=int(cfg.get("head_dim") or cfg["hidden_size"] // heads),
            rms_norm_eps=float(cfg["rms_norm_eps"]), rope_theta=float(cfg["rope_theta"]),
            max_position_embeddings=int(cfg["max_position_embeddings"]),
        )


def detect_weights_format(directory: Path) -> str:
    return "litgpt" if (directory / "lit_model.pth").is_file() else "hf"


def load_config(directory: Path, weights_format: str) -> Qwen3Config:
    if weights_format == "litgpt" and (directory / "model_config.yaml").is_file():
        return Qwen3Config.from_litgpt(yaml.safe_load((directory / "model_config.yaml").read_text()))
    if (directory / "config.json").is_file():
        return Qwen3Config.from_hf(json.loads((directory / "config.json").read_text()))
    raise FileNotFoundError(f"No model_config.yaml or config.json under {directory}")
