"""Attention-free Qwen backbone with a Mamba-2 SSD mixer in each residual block."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def ssd_scan(x, b, c, dt, a, state, tile=64):
    """Scalar-A semiseparable scan. Matrices are at most tile x tile, even in training.

    x: (batch, heads, time, head_dim), b/c: (batch, groups, time, state_dim).
    state: (batch, heads, state_dim, head_dim). dt is positive and a is negative.
    """
    heads_per_group = x.size(1) // b.size(1)
    if x.size(2) == 1:
        keys = b[:, :, 0].float().repeat_interleave(heads_per_group, dim=1)
        reads = c[:, :, 0].float().repeat_interleave(heads_per_group, dim=1)
        step = dt[:, :, 0, None, None].float()
        state = (step * a[None, :, None, None]).exp() * state + step * keys.unsqueeze(-1) * x.float()
        return (reads.unsqueeze(-2) @ state).to(x.dtype), state

    # SSD within each tile, followed by SSD over the tile boundaries. Cap both
    # matrices independently of sequence length. Padding has zero dt (identity).
    outputs = []
    tile = min(tile, x.size(2))
    window = 8 * tile
    for start in range(0, x.size(2), window):
        length = min(window, x.size(2) - start)
        sl = slice(start, start + length)
        pad = -length % tile

        def blocks(t):
            return F.pad(t[:, :, sl].float(), (0, 0, 0, pad)).unflatten(2, (-1, tile))

        inputs, keys, reads = blocks(x), blocks(b), blocks(c)
        step = blocks(dt.unsqueeze(-1)).squeeze(-1)
        prefix = (step * a[None, :, None, None]).cumsum(-1)
        scaled = inputs * step.unsqueeze(-1)
        decay = (prefix.unsqueeze(-1) - prefix.unsqueeze(-2)).clamp_max(0).exp().tril()
        cb = (reads @ keys.transpose(-2, -1)).repeat_interleave(heads_per_group, dim=1)
        within = (cb * decay) @ scaled
        end = prefix[..., -1]
        tail = (end.unsqueeze(-1) - prefix).exp().unsqueeze(-1)
        keys = keys.repeat_interleave(heads_per_group, dim=1)
        delta = keys.transpose(-2, -1) @ (tail * scaled)

        # Carry s_before[i] = A[i-1] s_before[i-1] + delta[i-1].
        ends = end.cumsum(-1)
        begins = F.pad(ends[..., :-1], (1, 0))
        cross = (begins.unsqueeze(-1) - ends.unsqueeze(-2)).clamp_max(0).exp().tril(-1)
        before = (cross @ delta.flatten(-2)).reshape_as(delta)
        before = before + begins.exp()[..., None, None] * state.unsqueeze(2)
        reads = reads.repeat_interleave(heads_per_group, dim=1)
        y = within + (reads @ before) * prefix.exp().unsqueeze(-1)
        outputs.append(y.flatten(2, 3)[:, :, :length].to(x.dtype))
        state = before[:, :, -1] * end[:, :, -1, None, None].exp() + delta[:, :, -1]
    return torch.cat(outputs, dim=2), state


class Mamba2Mixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inner = config.mamba_inner
        self.head_dim = config.mamba_head_dim
        self.groups = config.mamba_groups
        self.state_dim = config.mamba_state_dim
        self.heads = self.inner // self.head_dim
        if self.inner % self.head_dim or self.heads % self.groups:
            raise ValueError("inner must be divisible by head_dim and heads by groups")
        self.conv_dim = self.inner + 2 * self.groups * self.state_dim
        self.in_proj = nn.Linear(config.n_embd, self.inner + self.conv_dim + self.heads, bias=False)
        self.conv = nn.Conv1d(self.conv_dim, self.conv_dim, 4, groups=self.conv_dim)
        self.out_proj = nn.Linear(self.inner, config.n_embd, bias=False)
        self.A_log = nn.Parameter(torch.empty(self.heads))
        self.dt_bias = nn.Parameter(torch.empty(self.heads))
        self.D = nn.Parameter(torch.empty(self.heads))
        self.norm_weight = nn.Parameter(torch.empty(self.inner))
        self.reset_parameters()
        self.register_buffer("conv_state", None, persistent=False)
        self.register_buffer("ssm_state", None, persistent=False)

    @torch.no_grad()
    def reset_parameters(self):
        # Also used when FSDP materializes a meta-initialized module.
        a = torch.empty_like(self.A_log, dtype=torch.float32).uniform_(1, 16)
        dt = torch.empty_like(a).uniform_(math.log(.001), math.log(.1)).exp()
        self.A_log.copy_(a.log())
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.D.fill_(1)
        self.norm_weight.fill_(1)

    def reset_cache(self, batch):
        weight = self.in_proj.weight
        if self.conv_state is None or self.conv_state.size(0) != batch:
            self.conv_state = weight.new_zeros((batch, self.conv_dim, 3))
            self.ssm_state = weight.new_zeros((batch, self.heads, self.state_dim, self.head_dim), dtype=torch.float32)
        else:
            self.conv_state.zero_()
            self.ssm_state.zero_()

    def forward(self, hidden, cached=False):
        batch, length, _ = hidden.shape
        z, xbc, dt = self.in_proj(hidden).split([self.inner, self.conv_dim, self.heads], -1)
        xbc = xbc.transpose(1, 2)
        if cached:
            xbc = torch.cat([self.conv_state, xbc], dim=2)
            self.conv_state.copy_(xbc[:, :, -3:])
            state = self.ssm_state
        else:
            xbc = F.pad(xbc, (3, 0))
            state = hidden.new_zeros((batch, self.heads, self.state_dim, self.head_dim), dtype=torch.float32)
        x, b, c = F.silu(self.conv(xbc)).transpose(1, 2).split(
            [self.inner, self.groups * self.state_dim, self.groups * self.state_dim], dim=-1
        )
        x = x.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        b = b.reshape(batch, length, self.groups, self.state_dim).transpose(1, 2)
        c = c.reshape(batch, length, self.groups, self.state_dim).transpose(1, 2)
        dt = F.softplus(dt.float() + self.dt_bias.float()).transpose(1, 2)
        y, state = ssd_scan(x, b, c, dt, -self.A_log.float().exp(), state)
        if cached:
            self.ssm_state.copy_(state)
        y = (y + x * self.D[None, :, None, None]).transpose(1, 2).reshape_as(z)
        # Gated group RMSNorm (Mamba-2 norm_before_gate=False).
        gated = (y.float() * F.silu(z.float())).reshape(batch, length, self.groups, -1)
        gated = gated * torch.rsqrt(gated.square().mean(-1, keepdim=True) + 1e-5)
        return self.out_proj(gated.reshape_as(z).to(z.dtype) * self.norm_weight)


class MambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mixer = Mamba2Mixer(config)
        self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.config = config

    def forward(self, x, cached=False):
        x = x + self.mixer(self.norm_1(x), cached)
        chunk = self.config.inference_mlp_chunk_size
        if cached and chunk and x.size(1) > chunk:
            return torch.cat([part + self.mlp(self.norm_2(part)) for part in x.split(chunk, 1)], dim=1)
        return x + self.mlp(self.norm_2(x))


class MambaLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_seq_length = config.block_size
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
            h=nn.ModuleList(MambaBlock(config) for _ in range(config.n_layer)),
            ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
        ))
        self.position = 0

    def reset_cache(self, batch=1):
        for block in self.transformer.h:
            block.mixer.reset_cache(batch)
        self.position = 0

    def cache_budget(self):
        tensors = [t for block in self.transformer.h for t in (block.mixer.conv_state, block.mixer.ssm_state)]
        return {"kind": "mamba2_conv_and_ssm", "persistent_cache_bytes": sum(t.nbytes for t in tensors)}

    def forward(self, idx, *, input_pos=None, input_pos_maxp1=None, lm_head_start=0,
                targets=None, loss_chunk_size=128):
        cached = input_pos is not None
        if cached and (self.training or input_pos_maxp1 != self.position + idx.size(1)):
            raise ValueError("cached evaluation requires contiguous inputs after reset_cache()")
        if (self.position if cached else 0) + idx.size(1) > self.max_seq_length:
            raise ValueError("sequence exceeds the evaluation/training window")
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, cached)
        if cached:
            self.position += idx.size(1)
        x = self.transformer.ln_f(x[:, lm_head_start:])
        if targets is not None:
            from litgpt.chunked_loss import chunked_linear_cross_entropy
            return chunked_linear_cross_entropy(x, self.lm_head.weight, targets, chunk_size=loss_chunk_size)
        return self.lm_head(x)
