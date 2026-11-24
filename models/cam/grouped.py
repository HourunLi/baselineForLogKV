"""Grouped causal attention with the per-KV-head scores required by CaM."""
import torch


def attend_and_merge(query, keys, cache, scale):
    batch, heads, tokens, dim = query.shape
    groups, length = keys.shape[1:3]
    queries_per_group = heads // groups
    queries = query.reshape(batch, groups, queries_per_group, tokens, dim)
    received = keys.new_zeros((batch, groups, length), dtype=torch.float32)
    outputs = []
    # Flatten the query heads into the query rows so K/V are never repeated.
    for begin in range(0, tokens, 64):
        count = min(64, tokens - begin)
        q = queries[:, :, :, begin:begin + count].reshape(batch, groups, -1, dim)
        weights = (q @ keys.transpose(-2, -1)).view(batch, heads, count, length) * scale
        if tokens > 1:
            hidden = torch.ones(count, length, dtype=torch.bool, device=keys.device).triu(
                diagonal=length - tokens + begin + 1)
            weights.masked_fill_(hidden, -torch.inf)
        probs = weights.softmax(dim=-1, dtype=torch.float32).view(batch, groups, queries_per_group, count, length)
        received.add_(probs.mean(dim=2).sum(dim=2))
        out = probs.reshape(batch, groups, -1, length).to(cache.values.dtype) @ cache.values
        outputs.append(out.reshape(batch, heads, count, dim).transpose(1, 2))
    cache.update(received)
    return torch.cat(outputs, dim=1).reshape(batch, tokens, heads * dim)
