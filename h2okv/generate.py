"""Single-prompt generation with the reference evaluator's sampling and timing rules.

Sampling: top-k crop, then temperature, then top-p; argmax unless both
``temperature > 0`` and ``top_p > 0``. Generation stops at EOS (which is kept in
the returned ids) or at ``max_returned_tokens``; task ``until`` strings are not
applied, exactly like the evaluator this baseline is compared against.
Timing: prefill ends when the first token is ready (one ``item()`` sync per
token, no extra synchronisation), decode covers the remaining tokens.
"""

from __future__ import annotations

import time

import torch


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    sorted_indices_to_remove[-1:] = 0  # always keep the most probable token
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float("-inf"))


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
           top_p: float = 1.0) -> torch.Tensor:
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    logits = logits[0, -1]
    if top_k is not None:
        values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, indices, values)
    if temperature > 0.0 and top_p > 0.0:
        logits = logits / temperature
        if top_p < 1.0:
            logits = sample_top_p(logits, top_p)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)


@torch.inference_mode()
def generate(model, prompt: torch.Tensor, max_returned_tokens: int, *, temperature: float = 1.0,
             top_k: int | None = None, top_p: float = 1.0, eos_id: int | None = None,
             timing: dict | None = None) -> torch.Tensor:
    """Return ``prompt + generated`` ids; ``timing`` receives prefill/decode seconds and token counts."""
    prompt_size = prompt.size(0)
    device = prompt.device
    if max_returned_tokens <= prompt_size:
        raise ValueError(f"Not enough space for {prompt_size} prompt tokens in {max_returned_tokens}")
    if model.max_seq_length < max_returned_tokens - 1:
        raise ValueError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")
    if timing is not None:
        timing.clear()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()

    tokens = [prompt]
    token = prompt
    input_pos = torch.arange(0, prompt_size, device=device, dtype=torch.int64)
    for current_idx in range(max_returned_tokens - prompt_size):
        logits = model(token.view(1, -1), input_pos, lm_head_start=token.numel() - 1)
        token = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p).to(dtype=torch.int64)
        tokens.append(token)
        int_token = token.item()
        if timing is not None:
            elapsed = time.perf_counter() - start
            if current_idx == 0:
                timing["prefill_time_s"] = elapsed
            timing.update(total_time_s=elapsed, decode_time_s=elapsed - timing["prefill_time_s"],
                          generated_tokens=current_idx + 1, decode_tokens=current_idx)
        if eos_id is not None and int_token == eos_id:
            break
        input_pos = torch.tensor([prompt_size + current_idx], device=device, dtype=torch.int64)
    return torch.cat([t.to(torch.int64).view(-1) for t in tokens])
