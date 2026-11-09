"""ChunkKV prompt compression (Liu et al., arXiv:2502.00299, Algorithms 1 and 2).

After a prompt is prefilled with ordinary causal attention, every layer keeps

* the observation window: the last ``window_size`` prompt tokens, verbatim;
* whole chunks of ``chunk_size`` consecutive prompt tokens chosen from the
  prefix before the window, ``(budget - window) // chunk_size`` of them.

A prefix token's score is the softmax attention it receives from the window
queries (causal inside the window, the model's own scale), summed over window
queries and over all query heads, so one chunk set serves every KV head of the
layer. Chunks start at prompt position 0 and a chunk's score is the sum of its
token scores. Kept entries stay in their original order and are raw K/V, not
merged summaries. With ``reuse_layers = N`` only layers ``0, N, 2N, ...`` score
chunks; the next ``N - 1`` layers reuse those indices (Algorithm 2).

The window is not entered into the chunk competition and is appended exactly
once, which is Algorithm 1's "replace the last w selected tokens by the window"
without duplicated entries; up to ``chunk_size - 1`` budget entries stay unused.

Streaming (``settings.streaming``, the 64K / 128K stages): the model feeds the prompt in chunks and every
forward (a prompt chunk or a decode step) ends with a selection over the kept entries plus the forward's
tokens. The forward's last ``window_size`` queries (all of them when fewer) score every entry; the newest
``window_size`` entries are the window; the older ones form chunks by absolute position (``position //
chunk_size``, the same grid as the one-shot selection), and the best chunks are kept in score order while
they fit ``budget - window``. A chunk the window has partly overtaken competes with its older members only.
"""

import math

import torch
import torch.nn.functional as F

from .settings import ChunkKVSettings


def observation_scores(q: torch.Tensor, k: torch.Tensor, window_size: int, scale: float) -> torch.Tensor:
    """Window attention received by each prefix key, summed over window queries and heads.

    ``q`` is ``(B, n_head, T, hs)`` and ``k`` is ``(B, n_query_groups, T, hs)``,
    both after RoPE. Returns float32 ``(B, T - window)``.
    """
    batch, n_head, length, head_size = q.shape
    n_groups = k.size(1)
    window = min(window_size, length)
    prefix = length - window
    scores = torch.zeros(batch, prefix, device=q.device, dtype=torch.float32)
    if prefix == 0:
        return scores
    # Window query i sits at position prefix + i and sees keys 0 .. prefix + i.
    query_pos = torch.arange(prefix, length, device=q.device)
    hidden = torch.arange(length, device=q.device) > query_pos[:, None]
    queries = q[:, :, prefix:].float().view(batch, n_groups, n_head // n_groups, window, head_size)
    # One KV group at a time bounds the float32 logits to (heads per group, window, T).
    for group in range(n_groups):
        logits = queries[:, group] @ k[:, group, None].float().transpose(-1, -2) * scale
        weights = logits.masked_fill(hidden, float("-inf")).softmax(dim=-1)
        scores += weights[..., :prefix].sum(dim=(1, 2))
    return scores


def select_chunks(scores: torch.Tensor, chunk_size: int, n_chunks: int) -> torch.Tensor:
    """Sorted positions of the ``n_chunks`` highest-scoring chunks of a 1-D score vector."""
    prefix = scores.numel()
    total = math.ceil(prefix / chunk_size)
    if n_chunks >= total:
        return torch.arange(prefix, device=scores.device)
    chunk_scores = F.pad(scores, (0, total * chunk_size - prefix)).view(total, chunk_size).sum(dim=-1)
    # Stable sort: equal scores keep the earlier chunk, independent of the top-k kernel.
    chosen = torch.sort(chunk_scores, descending=True, stable=True).indices[:n_chunks].sort().values
    positions = (chosen[:, None] * chunk_size + torch.arange(chunk_size, device=scores.device)).flatten()
    return positions[positions < prefix]


def window_scores(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    """Attention every one of the ``L`` entries receives from the ``n`` queries of ``q``, summed over queries
    and heads. The queries are the last ``n`` entries' own (causal). ``q``: (B, n_head, n, hs) after RoPE,
    ``k``: (B, n_query_groups, L, hs) after RoPE. Returns float32 (B, L)."""
    batch, n_head, n, head_size = q.shape
    n_groups, length = k.size(1), k.size(2)
    hidden = torch.ones(n, length, dtype=torch.bool, device=q.device).tril(diagonal=length - n).logical_not()
    queries = q.float().view(batch, n_groups, n_head // n_groups, n, head_size)
    scores = torch.zeros(batch, length, device=q.device, dtype=torch.float32)
    for group in range(n_groups):
        logits = queries[:, group] @ k[:, group, None].float().transpose(-1, -2) * scale
        scores += logits.masked_fill(hidden, float("-inf")).softmax(dim=-1).sum(dim=(1, 2))
    return scores


def streaming_keep(scores: torch.Tensor, positions: torch.Tensor, settings: ChunkKVSettings) -> tuple[torch.Tensor, int]:
    """Indices (position order) of the entries a streaming layer keeps, and how many are chunk tokens.

    ``scores`` and ``positions`` (ascending) describe the layer's ``L`` entries, the forward's tokens last.
    """
    length = positions.numel()
    window = min(settings.window_size, length)
    older = length - window
    chunk_ids = torch.div(positions[:older], settings.chunk_size, rounding_mode="floor")
    _, member_of, sizes = torch.unique_consecutive(chunk_ids, return_inverse=True, return_counts=True)
    chunk_scores = torch.zeros(sizes.numel(), device=scores.device, dtype=torch.float32)
    chunk_scores.index_add_(0, member_of, scores[:older].float())
    # Stable sort: equal scores keep the earlier chunk, as in the one-shot selection.
    order = torch.sort(chunk_scores, descending=True, stable=True).indices
    taken = order[torch.cumsum(sizes[order], dim=0) <= settings.budget - window]
    chosen = torch.zeros(sizes.numel(), dtype=torch.bool, device=scores.device)
    chosen[taken] = True
    kept = torch.nonzero(chosen[member_of]).flatten()
    window_positions = torch.arange(older, length, device=scores.device)
    return torch.cat((kept, window_positions)), kept.numel()


def keep_indices(scores: torch.Tensor, prompt_len: int, settings: ChunkKVSettings) -> tuple[torch.Tensor, int]:
    """Kept prompt positions (selected chunks, then the window) and the chunk-token count."""
    window = min(settings.window_size, prompt_len)
    prefix = prompt_len - window
    if scores.shape != (prefix,):
        raise ValueError("Expected one observation score per prefix token")
    n_chunks = max(0, settings.prompt_budget(prompt_len) - window) // settings.chunk_size
    chunks = select_chunks(scores, settings.chunk_size, n_chunks)
    window_positions = torch.arange(prefix, prompt_len, device=scores.device)
    return torch.cat((chunks, window_positions)), chunks.numel()
