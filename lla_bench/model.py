"""Native HAttentionForCausalLM evaluation on plain PyTorch (published Mamba-2 variant)."""
import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .recurrence import MatrixCache, weak_attention


@dataclass
class HConfig:
    hidden_size: int
    num_heads: int
    num_hidden_layers: int
    vocab_size: int
    n_groups: int
    state_size: int
    head_dim: int
    expand: int
    conv_kernel: int
    layer_norm_epsilon: float
    use_bias: bool
    use_conv_bias: bool
    trained_levels: int = 15  # upstream MAX_SEQUENCE_LENGTH=16384, base two
    block_size: int = 32768
    inference_mlp_chunk_size: int = 2048
    student: bool = False
    mlp_width: int = 0

    @property
    def dense_geometry(self):
        return (self.num_hidden_layers, self.n_groups, self.head_dim) if self.student else (28, 8, 128)

    @classmethod
    def read(cls, path, window):
        data = json.loads(Path(path).read_text())
        if data["model_type"] == "qwen3_hattention":
            if not data["student"] or data["mlp_width"] <= 0:
                raise ValueError("invalid converted Qwen student configuration")
            saved = {field.name: data[field.name] for field in fields(cls)}
            if window is not None:
                saved["block_size"] = window
            return cls(**saved)
        if (data["model_type"] != "hattention" or data["hidden_act"] not in ("swish", "silu")
                or data["residual_in_fp32"] or data["tie_word_embeddings"]
                or data.get("time_step_limit", [0, float("inf")]) != [0, float("inf")]):
            raise ValueError("expected the native HAttention Mamba-2 architecture")
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data and f.name != "block_size"},
                   block_size=window)


class MixerCache(MatrixCache):
    def __init__(self, cfg, batch, device, dtype):
        super().__init__(batch, cfg.num_heads, cfg.state_size, cfg.head_dim, cfg.block_size, device)
        channels = cfg.num_heads * cfg.head_dim + 2 * cfg.n_groups * cfg.state_size
        self.register_buffer("conv", torch.zeros(batch, channels, cfg.conv_kernel, device=device, dtype=dtype),
                             persistent=False)

    def reset(self):
        super().reset()
        self.conv.zero_()


class Mixer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h, d, g, s = cfg.num_heads, cfg.head_dim, cfg.n_groups, cfg.state_size
        if h * d != cfg.expand * cfg.hidden_size or h % g:
            raise ValueError("incompatible native head geometry")
        self.channels = h * d + 2 * g * s
        self.in_proj = nn.Linear(cfg.hidden_size, h*d + self.channels + h*(1+cfg.trained_levels), bias=cfg.use_bias)
        self.conv1d = nn.Conv1d(self.channels, self.channels, cfg.conv_kernel,
                                groups=self.channels, bias=cfg.use_conv_bias)
        self.dt_bias = nn.Parameter(torch.ones(h))
        self.A_log = nn.Parameter(torch.arange(1, h+1).float().log())
        self.D = nn.Parameter(torch.ones(h))
        self.L = nn.Parameter(torch.ones(h, cfg.trained_levels))
        self.norm = nn.RMSNorm(h*d, eps=cfg.layer_norm_epsilon)
        self.out_proj = nn.Linear(h*d, cfg.hidden_size, bias=cfg.use_bias)
        self.cache = None

    def forward(self, hidden):
        c = self.cfg
        b, t, _ = hidden.shape
        h, d, g, s = c.num_heads, c.head_dim, c.n_groups, c.state_size
        gate, xbc, dt, dl = self.in_proj(hidden).split(
            [h*d, self.channels, h, h*c.trained_levels], dim=-1)
        xbc = xbc.transpose(1, 2)
        if self.cache is None:
            hist = F.pad(xbc, (c.conv_kernel-1, 0))
            end = t
        else:
            hist = torch.cat((self.cache.conv[..., 1:], xbc), -1)
            # If the first call is shorter than K, hist includes the zero initial history.
            self.cache.conv.copy_(hist[..., -c.conv_kernel:])
            end = self.cache.count + t
        x, key, query = F.silu(self.conv1d(hist)).transpose(1, 2).split([h*d, g*s, g*s], -1)
        x = x.reshape(b, t, h, d).transpose(1, 2)
        def heads(tensor):
            return tensor.reshape(b, t, g, s).transpose(1, 2)
        dt = F.softplus(dt + self.dt_bias).transpose(1, 2)
        log_decay = -self.A_log.float().exp()[None, :, None] * dt.float()
        lam = F.softplus(dl.reshape(b, t, h, c.trained_levels) * self.L).transpose(1, 2)
        extra = (end - 1).bit_length() + 1 - c.trained_levels
        if extra > 0:
            # Documented length extrapolation: preserve all learned read weights.
            lam = torch.cat([lam, lam[..., -1:].expand(*lam.shape[:-1], extra)], -1)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            y = weak_attention(heads(query), heads(key), x * dt.unsqueeze(-1), log_decay, lam, self.cache)
        y = y + x * self.D[None, :, None, None]
        y = y.transpose(1, 2).reshape(b, t, h*d)
        # Upstream RMSNormGated(norm_before_gate=False), full intermediate dimension.
        gated = y.float() * F.silu(gate.float())
        norm = gated * torch.rsqrt(gated.square().mean(-1, keepdim=True) + c.layer_norm_epsilon)
        return self.out_proj((norm * self.norm.weight).to(hidden.dtype))


class GatedMLP(nn.Module):
    def __init__(self, dim, size=0):
        super().__init__()
        size = size or 256 * ((int(dim * 8 / 3) + 255) // 256)
        self.gate_proj = nn.Linear(dim, size, bias=False)
        self.up_proj = nn.Linear(dim, size, bias=False)
        self.down_proj = nn.Linear(size, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mixer = Mixer(cfg)
        self.mixer_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)
        self.mlp_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)
        self.mlp = GatedMLP(cfg.hidden_size, cfg.mlp_width)

    def forward(self, hidden, chunk):
        if self.training and self.mixer.cfg.student:
            return checkpoint(self._forward, hidden, chunk, use_reentrant=False)
        return self._forward(hidden, chunk)

    def _forward(self, hidden, chunk):
        # Only the mixer carries state. Split the tokenwise MLP at the same budget as LogKV.
        hidden = hidden + self.mixer(self.mixer_norm(hidden))
        size = chunk or hidden.size(1)
        return torch.cat([x + self.mlp(self.mlp_norm(x)) for x in hidden.split(size, dim=1)], dim=1)


class HAttentionLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.max_seq_length = cfg.block_size
        self.backbone = nn.Module()
        self.backbone.embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.backbone.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.num_hidden_layers))
        self.backbone.norm_f = nn.RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    @classmethod
    def load(cls, folder, cfg, device, dtype=torch.bfloat16):
        # Load through mmap, strictly check every upstream name/shape, then cast once.
        with torch.device("meta"):
            model = cls(cfg)
        folder = Path(folder)
        index = folder / "model.safetensors.index.json"
        if index.is_file():
            mapping = json.loads(index.read_text())["weight_map"]
            state = {}
            for shard in sorted(set(mapping.values())):
                tensors = load_file(str(folder / shard))
                if set(tensors) != {k for k, v in mapping.items() if v == shard}:
                    raise ValueError(f"inconsistent tensor index for {shard}")
                state.update(tensors)
        else:
            state = load_file(str(folder / "model.safetensors"))
        model.load_state_dict(state, assign=True, strict=True)
        return model.to(device=device, dtype=dtype).eval()

    def set_cache(self, batch=1):
        p = self.lm_head.weight
        for layer in self.backbone.layers:
            layer.mixer.cache = MixerCache(self.config, batch, p.device, p.dtype)

    def reset_cache(self):
        for layer in self.backbone.layers:
            layer.mixer.cache.reset()

    def forward(self, idx, input_pos=None, input_pos_maxp1=None, lm_head_start=None, targets=None):
        cache = self.backbone.layers[0].mixer.cache
        if cache is not None:
            end = cache.count + idx.size(1)
            if end > self.max_seq_length or input_pos is None or not torch.equal(
                input_pos, torch.arange(cache.count, end, device=idx.device)
            ):
                raise ValueError("native cache requires contiguous append-only positions within the model window")
        elif input_pos is not None:
            raise ValueError("initialize the recurrent cache before scoring or generating")
        hidden = self.backbone.embeddings(idx)
        for layer in self.backbone.layers:
            hidden = layer(hidden, self.config.inference_mlp_chunk_size)
        if targets is not None:
            from .litgpt.chunked_loss import chunked_linear_cross_entropy
            return chunked_linear_cross_entropy(self.backbone.norm_f(hidden), self.lm_head.weight, targets)
        return self.lm_head(self.backbone.norm_f(hidden[:, lm_head_start or 0:]))
