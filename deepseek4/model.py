import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .attention import Attention, Cache, Geometry, LAYOUT


class MLP(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.fc_1 = nn.Linear(width, intermediate, bias=False)
        self.fc_2 = nn.Linear(width, intermediate, bias=False)
        self.proj = nn.Linear(intermediate, width, bias=False)

    def forward(self, x):
        return self.proj(F.silu(self.fc_1(x)) * self.fc_2(x))


class Block(nn.Module):
    def __init__(self, g, ratio, intermediate, factor):
        super().__init__()
        self.norm_1 = nn.RMSNorm(g.width, eps=1e-6)
        self.attn = Attention(g, ratio, factor)
        self.norm_2 = nn.RMSNorm(g.width, eps=1e-6)
        self.mlp = MLP(g.width, intermediate)

    def forward(self, x, start, cached):
        if self.training:
            return checkpoint(self._run, x, start, cached, use_reentrant=False)
        return self._run(x, start, cached)

    def _run(self, x, start, cached):
        delta, loss = self.attn(self.norm_1(x), start, cached)
        x = x + delta
        return x + self.mlp(self.norm_2(x)), loss


class Model(nn.Module):
    def __init__(self, config, g=None, layout=LAYOUT):
        super().__init__()
        self.config = config
        if g is None:
            if (
                config.n_embd,
                config.n_layer,
                config.intermediate_size,
                config.padded_vocab_size,
                config.n_head,
                config.n_query_groups,
                config.head_size,
            ) != (4096, 36, 12288, 151936, 32, 8, 128):
                raise ValueError("Expected Qwen3-8B-Base geometry")
            g = Geometry()
        self.g, self.max_seq_length = g, config.block_size
        self.layout = tuple(layout)
        factor = (config.rope_adjustments or {}).get("factor", 1)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, g.width),
                h=nn.ModuleList(Block(g, r, config.intermediate_size, factor) for r in layout),
                ln_f=nn.RMSNorm(g.width, eps=1e-6),
            )
        )
        self.lm_head = nn.Linear(g.width, config.padded_vocab_size, bias=False)

    def set_kv_cache(self, batch_size, max_seq_length, device, dtype):
        for block in self.transformer.h:
            block.attn.kv_cache = Cache(self.g, block.attn.ratio, max_seq_length, batch_size, device, dtype)

    def reset_kv_cache(self):
        for block in self.transformer.h:
            block.attn.kv_cache.pos = 0

    def hidden(self, ids, start, cached):
        x = self.transformer.wte(ids)
        aux = x.new_zeros((), dtype=torch.float32)
        for block in self.transformer.h:
            x, loss = block(x, start, cached)
            aux = aux + loss
        return self.transformer.ln_f(x), aux / max(1, self.layout.count(4))

    def forward(
        self, idx, input_pos=None, input_pos_maxp1=None, lm_head_start=0, *, targets=None, loss_chunk_size=1024
    ):
        if targets is not None:
            return self.training_loss(idx, targets, loss_chunk_size)
        start = 0
        if input_pos is not None:
            start = self.transformer.h[0].attn.kv_cache.pos
            if (
                input_pos.ndim != 1
                or input_pos.numel() != idx.size(1)
                or not torch.equal(input_pos, torch.arange(start, start + idx.size(1), device=idx.device))
            ):
                raise ValueError("Expected contiguous positions from the current KV cursor")
        h, _ = self.hidden(idx, start, input_pos is not None)
        return self.lm_head(h[:, lm_head_start:])

    def training_loss(self, ids, targets, chunk_size):
        from ._vendor.niahlogkv.litgpt.chunked_loss import chunked_linear_cross_entropy

        h, aux = self.hidden(ids, 0, False)
        return chunked_linear_cross_entropy(h, self.lm_head.weight, targets, chunk_size=chunk_size), aux
