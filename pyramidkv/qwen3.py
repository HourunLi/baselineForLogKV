"""Self-contained dense Qwen3 decoder with a pluggable per-layer KV cache.

Parameter names follow the LitGPT layout used by SemanticLogKV's original
``lit_model.pth`` so the comparison Base loads without any conversion step or
import from that project. Hugging Face checkpoints are mapped onto the same
names. Numerics follow the reference dense path: fp32 RMSNorm statistics, RoPE
tables cast to the model dtype, grouped K/V expanded with ``repeat_interleave``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


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


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, base: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len, device=device).float(), theta).repeat(1, 2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return ((x * cos) + (rotated * sin)).to(x.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, H_kv, L, D) -> (B, H_kv * n_rep, L, D); query head h reads KV head h // n_rep."""
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        heads, groups, dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.qkv = nn.Linear(config.hidden_size, (heads + 2 * groups) * dim, bias=False)
        self.proj = nn.Linear(heads * dim, config.hidden_size, bias=False)
        self.norm_q = RMSNorm(dim, config.rms_norm_eps)
        self.norm_k = RMSNorm(dim, config.rms_norm_eps)
        self.kv_cache = None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, use_cache: bool) -> torch.Tensor:
        cfg = self.config
        B, T, _ = x.shape
        heads, groups, dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q, k, v = self.qkv(x).split((heads * dim, groups * dim, groups * dim), dim=-1)
        q = self.norm_q(q.view(B, T, heads, dim).transpose(1, 2))
        k = self.norm_k(k.view(B, T, groups, dim).transpose(1, 2))
        v = v.view(B, T, groups, dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if use_cache:
            # The cache owns storage, attention and (for compression methods) eviction.
            y = self.kv_cache(q, k, v)
        else:
            n_rep = heads // groups
            y = F.scaled_dot_product_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, heads * dim))


class MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.fc_1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc_2 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.fc_1(x)) * self.fc_2(x))


class Block(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.norm_1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.norm_2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, use_cache):
        x = x + self.attn(self.norm_1(x), cos, sin, use_cache)
        return x + self.mlp(self.norm_2(x))


class Qwen3(nn.Module):
    """``forward(idx, input_pos, input_pos_maxp1, lm_head_start)`` matches the shared evaluator call shape."""

    def __init__(self, config: Qwen3Config, max_seq_length: int | None = None):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.hidden_size),
            h=nn.ModuleList(Block(config, i) for i in range(config.num_hidden_layers)),
            ln_f=RMSNorm(config.hidden_size, config.rms_norm_eps),
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_seq_length = max_seq_length or config.max_position_embeddings

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        if not 0 < value <= self.config.max_position_embeddings:
            raise ValueError(f"max_seq_length {value} exceeds the model position window "
                             f"{self.config.max_position_embeddings}")
        self._max_seq_length = value
        device = self.lm_head.weight.device
        if device.type == "meta":
            device = None
        cos, sin = build_rope_cache(value, self.config.head_dim, self.config.rope_theta, device)
        dtype = self.lm_head.weight.dtype if self.lm_head.weight.is_floating_point() else torch.float32
        self.register_buffer("cos", cos.to(dtype), persistent=False)
        self.register_buffer("sin", sin.to(dtype), persistent=False)

    @property
    def caches(self) -> list:
        return [block.attn.kv_cache for block in self.transformer.h]

    def set_kv_cache(self, factory) -> None:
        """``factory(layer_idx)`` returns a fresh per-request cache for one layer."""
        for i, block in enumerate(self.transformer.h):
            block.attn.kv_cache = factory(i)

    def clear_kv_cache(self) -> None:
        for block in self.transformer.h:
            block.attn.kv_cache = None

    def forward(self, idx: torch.Tensor, input_pos: torch.Tensor | None = None,
                input_pos_maxp1: int | None = None, lm_head_start: int = 0) -> torch.Tensor:
        T = idx.size(1)
        use_cache = input_pos is not None
        if use_cache:
            if any(cache is None for cache in self.caches):
                raise RuntimeError("Call set_kv_cache() before forwarding with input_pos")
            expected = self.caches[0].processed_tokens
            if input_pos.numel() != T:
                raise ValueError("input_pos must have one position per input token")
            # Single-token decode on GPU skips the check to avoid a host sync per generated token.
            if (T > 1 or input_pos.device.type == "cpu") and not torch.equal(
                    input_pos, torch.arange(expected, expected + T, device=input_pos.device, dtype=input_pos.dtype)):
                raise ValueError(f"input_pos must continue the request contiguously from {expected}")
            if expected + T > self.max_seq_length:
                raise ValueError(f"Position {expected + T - 1} exceeds max_seq_length {self.max_seq_length}")
            positions = input_pos
        else:
            if T > self.max_seq_length:
                raise ValueError(f"Sequence length {T} exceeds max_seq_length {self.max_seq_length}")
            positions = torch.arange(T, device=idx.device)
        cos, sin = self.cos[positions], self.sin[positions]
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, cos, sin, use_cache)
        x = self.transformer.ln_f(x[:, lm_head_start:])
        return self.lm_head(x)


_HF_LAYER = re.compile(r"model\.layers\.(\d+)\.(.+)")
_HF_TO_LIT = {
    "input_layernorm.weight": "norm_1.weight",
    "post_attention_layernorm.weight": "norm_2.weight",
    "self_attn.o_proj.weight": "attn.proj.weight",
    "self_attn.q_norm.weight": "attn.norm_q.weight",
    "self_attn.k_norm.weight": "attn.norm_k.weight",
    "mlp.gate_proj.weight": "mlp.fc_1.weight",
    "mlp.up_proj.weight": "mlp.fc_2.weight",
    "mlp.down_proj.weight": "mlp.proj.weight",
}


def hf_to_litgpt_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename HF Qwen3 tensors to the LitGPT layout ([Q; K; V] fused, not interleaved)."""
    out, qkv = {}, {}
    for name, tensor in state.items():
        if name == "model.embed_tokens.weight":
            out["transformer.wte.weight"] = tensor
        elif name == "model.norm.weight":
            out["transformer.ln_f.weight"] = tensor
        elif name == "lm_head.weight":
            out["lm_head.weight"] = tensor
        elif match := _HF_LAYER.fullmatch(name):
            layer, rest = match.groups()
            if rest in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"):
                qkv.setdefault(layer, {})[rest.split(".")[1][0]] = tensor
            elif rest in _HF_TO_LIT:
                out[f"transformer.h.{layer}.{_HF_TO_LIT[rest]}"] = tensor
            elif not rest.endswith("rotary_emb.inv_freq"):
                raise KeyError(f"Unexpected Hugging Face tensor: {name}")
        elif not name.endswith("rotary_emb.inv_freq"):
            raise KeyError(f"Unexpected Hugging Face tensor: {name}")
    for layer, parts in qkv.items():
        if set(parts) != {"q", "k", "v"}:
            raise KeyError(f"Layer {layer} is missing one of q/k/v projections")
        out[f"transformer.h.{layer}.attn.qkv.weight"] = torch.cat((parts["q"], parts["k"], parts["v"]))
    return out


def _load_hf_tensors(directory: Path) -> dict[str, torch.Tensor]:
    for index_name, single_name, kind in (
        ("model.safetensors.index.json", "model.safetensors", "safetensors"),
        ("pytorch_model.bin.index.json", "pytorch_model.bin", "torch"),
    ):
        index = directory / index_name
        files = (sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file()
                 else [single_name] if (directory / single_name).is_file() else [])
        if not files:
            continue
        state = {}
        for name in files:
            if kind == "safetensors":
                from safetensors.torch import load_file
                state.update(load_file(str(directory / name), device="cpu"))
            else:
                state.update(torch.load(directory / name, map_location="cpu", weights_only=True))
        return state
    raise FileNotFoundError(f"No lit_model.pth, *.safetensors or pytorch_model.bin under {directory}")


def detect_weights_format(directory: Path) -> str:
    return "litgpt" if (directory / "lit_model.pth").is_file() else "hf"


def load_config(directory: Path, weights_format: str) -> Qwen3Config:
    if weights_format == "litgpt" and (directory / "model_config.yaml").is_file():
        import yaml
        return Qwen3Config.from_litgpt(yaml.safe_load((directory / "model_config.yaml").read_text()))
    if (directory / "config.json").is_file():
        return Qwen3Config.from_hf(json.loads((directory / "config.json").read_text()))
    raise FileNotFoundError(f"No model_config.yaml or config.json under {directory}")


def load_state_dict(directory: Path, weights_format: str) -> dict[str, torch.Tensor]:
    if weights_format == "hf":
        return hf_to_litgpt_state_dict(_load_hf_tensors(directory))
    path = directory / "lit_model.pth"
    if path.is_dir():
        raise FileNotFoundError(f"{path} is a sharded directory; merge it into one lit_model.pth first")
    try:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except RuntimeError:  # legacy (non-zip) serialization cannot be memory-mapped
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]  # Fabric-wrapped checkpoint
    return state


def load_qwen3(checkpoint_dir, max_seq_length: int, device="cuda", dtype=torch.bfloat16,
               weights_format: str = "auto") -> tuple[Qwen3, dict]:
    """Strictly load every transformer weight; a silently random tensor would invalidate the comparison."""
    directory = Path(checkpoint_dir).expanduser()
    if weights_format == "auto":
        weights_format = detect_weights_format(directory)
    config = load_config(directory, weights_format)
    state = load_state_dict(directory, weights_format)
    state = {k: v for k, v in state.items() if not k.endswith(("mask_cache", ".cos", ".sin")) and k not in ("cos", "sin")}
    tied = "lm_head.weight" not in state
    if tied:
        state["lm_head.weight"] = state["transformer.wte.weight"]
    with torch.device("meta"):
        model = Qwen3(config, max_seq_length=max_seq_length)
    model.load_state_dict(state, strict=True, assign=True)
    model.max_seq_length = max_seq_length  # replace the meta RoPE tables before moving
    model = model.to(device=device, dtype=dtype)
    if tied:
        model.lm_head.weight = model.transformer.wte.weight
    model.max_seq_length = max_seq_length  # fp32 tables cast once to the model dtype, as in the reference
    info = {"weights_format": weights_format, "tied_lm_head": tied, "config": config.__dict__}
    return model.eval(), info
