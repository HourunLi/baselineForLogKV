"""Bound vocabulary-sized activations by checkpointing projection and loss together."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_linear_cross_entropy(hidden, weight, targets, bias=None, chunk_size=128, softcap=None, ignore_index=-100):
    if hidden.shape[:-1] != targets.shape:
        raise ValueError("targets must match the hidden-state token dimensions")
    if chunk_size < 0:
        raise ValueError("chunk_size must be nonnegative")
    if softcap is not None and softcap <= 0:
        raise ValueError("softcap must be positive")
    hidden = hidden.reshape(-1, hidden.size(-1))
    targets = targets.reshape(-1)
    chunk_size = chunk_size or max(1, targets.numel())

    def project_loss(x, w, y, b):
        logits = F.linear(x, w, b)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        # Accumulate CE in fp32 even with bf16-true training.
        return F.cross_entropy(logits.float(), y, ignore_index=ignore_index, reduction="sum")

    loss = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk_size):
        loss = loss + checkpoint(
            project_loss, hidden[start:start + chunk_size], weight, targets[start:start + chunk_size], bias,
            use_reentrant=False, preserve_rng_state=False,
        )
    return loss / (targets != ignore_index).sum().clamp_min(1)
