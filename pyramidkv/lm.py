"""lm-eval model with SemanticLogKV's request, scoring and metric semantics.

Implemented in this repository (nothing is imported from SemanticLogKV) so a
compressed-cache baseline sees exactly the same evaluation contract:

* tokenization: no chat template; continuation encoded with ``bos=False``;
  an empty context becomes ``[bos_id]``;
* truncation: left-truncate context to the window; generation reserves
  ``max_gen_toks`` before left-truncating the prompt;
* generation: greedy unless ``do_sample``; stops at EOS or length only
  (``until`` strings are not applied, same as the reference evaluator);
* scoring: fp32 log-softmax in 1024-position chunks; rolling windows of
  ``max_seq_length - 1`` tokens conditioned on BOS / the previous token;
* metrics: KV entries snapshot at request end before the cache is released,
  forward/prefill/decode timing boundaries and CUDA peak memory per request;
* data parallel: each rank takes ``requests[rank::world]``.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import tqdm

from . import dist_utils as du
from .generate import generate
from .kv_cache import LOGLIKELIHOOD_MODES
from .metrics import cache_compression_snapshot
from .qwen3 import load_qwen3
from .tokenizer import Tokenizer

try:
    from lm_eval.api.model import LM as _HarnessLM
except ImportError:  # unit tests and --print-config do not need the harness
    _HarnessLM = object



def _synchronize(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device) -> dict:
    if torch.device(device).type != "cuda":
        return {"peak_memory_allocated_bytes": None, "peak_memory_reserved_bytes": None}
    return {"peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def request_sample_id(req: Any, *, global_request_index: int, rank: int) -> str:
    task_name, doc_id = getattr(req, "task_name", None), getattr(req, "doc_id", None)
    idx, metadata = getattr(req, "idx", None), getattr(req, "metadata", None)
    if isinstance(metadata, dict):
        task_name = task_name or metadata.get("task_name") or metadata.get("task")
        doc_id = doc_id if doc_id is not None else metadata.get("doc_id")
        idx = idx if idx is not None else metadata.get("idx")
    parts = [f"rank{rank}", f"global_req{int(global_request_index)}"]
    if task_name is not None:
        parts.append(f"task={task_name}")
    if doc_id is not None:
        parts.append(f"doc={doc_id}")
    elif idx is not None:
        parts.append(f"idx={idx}")
    return "|".join(parts)


class CompressedKVLM(_HarnessLM):
    """``cache_factory(layer_idx)`` builds one layer's per-request cache (method-specific)."""

    def __init__(self, checkpoint_dir: str, cache_factory, *, tokenizer_dir: str | None = None,
                 max_seq_length: int = 32768, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 weights_format: str = "auto", loglikelihood_mode: str = "compress_context",
                 process_group=None, model=None, tokenizer=None):
        if _HarnessLM is not object:
            super().__init__()
        if loglikelihood_mode not in LOGLIKELIHOOD_MODES:
            raise ValueError(f"loglikelihood_mode must be one of {LOGLIKELIHOOD_MODES}")
        self.checkpoint_dir = str(checkpoint_dir)
        self.process_group = process_group
        self._device = device
        self.cache_factory = cache_factory
        self.loglikelihood_mode = loglikelihood_mode
        self.tokenizer = tokenizer or Tokenizer(tokenizer_dir or checkpoint_dir)
        if model is None:
            model, self.model_info = load_qwen3(checkpoint_dir, max_seq_length, device=device, dtype=dtype,
                                                weights_format=weights_format)
        else:
            self.model_info = {}
        self.model = model
        self.gen_metrics: list[dict] = []
        self.ppl_metrics: list[dict] = []
        self.compression_metrics: list[dict] = []

    # Request caches are dynamic: a fresh one per request, released afterwards.
    def _set_eval_cache(self) -> None:
        self.model.set_kv_cache(self.cache_factory)

    def _reset_eval_cache(self) -> None:
        self.model.clear_kv_cache()

    def all_gather_results(self, local_result_list: list, tag: str = "") -> list:
        return du.all_gather_interleaved(local_result_list, group=self.process_group, tag=tag)

    def _forward_scoring(self, inps: torch.Tensor, ctx_len: int) -> torch.Tensor:
        seq_len = inps.size(1)
        if self.loglikelihood_mode == "single_forward" or ctx_len == seq_len:
            # One prefill over context + continuation: compression happens after
            # this forward, so the scores equal FullKV (KVCache-Factory prefill semantics).
            positions = torch.arange(seq_len, device=self._device, dtype=torch.int64)
            return self.model(inps, input_pos=positions, lm_head_start=ctx_len - 1)
        # The context is prefilled and compressed; the continuation is scored against
        # the compressed cache, as a generated answer would be.
        first = self.model(inps[:, :ctx_len], input_pos=torch.arange(ctx_len, device=self._device),
                           lm_head_start=ctx_len - 1)
        rest = self.model(inps[:, ctx_len:], input_pos=torch.arange(ctx_len, seq_len, device=self._device))
        return torch.cat((first, rest), dim=1)

    def _score_tokens(self, ctx_enc: list[int], cont_enc: list[int], *,
                      request_type: str = "loglikelihood") -> tuple[float, bool]:
        max_len = self.model.max_seq_length
        if len(ctx_enc) + len(cont_enc) > max_len:
            keep_ctx_len = max_len - len(cont_enc)
            if keep_ctx_len <= 0:
                cont_enc = cont_enc[-(max_len - 1):]
                ctx_enc = []
            else:
                ctx_enc = ctx_enc[-keep_ctx_len:]
            print(f"⚠️ [{du.rank_label()}] context truncated to {len(ctx_enc)} tokens")
        if len(ctx_enc) == 0:
            ctx_enc = [self.tokenizer.bos_id]

        inps = torch.tensor([ctx_enc + cont_enc], dtype=torch.long, device=self._device)
        seq_len, ctx_len = inps.size(1), len(ctx_enc)
        with torch.no_grad():
            self._set_eval_cache()
            try:
                _synchronize(self._device)
                _reset_peak_memory(self._device)
                t0 = time.perf_counter()
                logits = self._forward_scoring(inps, ctx_len)
                _synchronize(self._device)
                t1 = time.perf_counter()
                peak_memory = _peak_memory(self._device)
                self.compression_metrics.append({
                    "request_type": request_type, "rank": du.global_rank(),
                    "context_len": ctx_len, "cont_len": len(cont_enc),
                    **cache_compression_snapshot(self.model, seq_len),
                })
            finally:
                self._reset_eval_cache()

        self.ppl_metrics.append({
            "rank": du.global_rank(), "request_type": request_type,
            "total_seq_len": seq_len, "context_len": ctx_len, "cont_len": len(cont_enc),
            "forward_time_s": t1 - t0, "forward_time_ms": (t1 - t0) * 1000,
            "forward_tokens_per_sec": seq_len / (t1 - t0) if t1 > t0 else None,
            **peak_memory,
        })

        cont_logits = logits[0, :len(cont_enc)]
        cont_targets = torch.tensor(cont_enc, dtype=torch.long, device=self._device)
        total_logprob, is_greedy, step = 0.0, True, 1024
        for i in range(0, cont_logits.size(0), step):
            block = cont_logits[i:i + step].to(torch.float32)
            target = cont_targets[i:i + step]
            log_probs = torch.log_softmax(block, dim=-1)
            total_logprob += log_probs.gather(dim=-1, index=target.unsqueeze(-1)).sum().item()
            if is_greedy:
                is_greedy = bool((block.argmax(dim=-1) == target).all().item())
        return total_logprob, is_greedy

    def loglikelihood(self, requests):
        rank, world = du.global_rank(), du.world_size()
        results = []
        desc = f"loglikelihood {du.rank_label()}"
        for req in tqdm.tqdm(requests[rank::world], desc=desc, position=rank):
            context, continuation = req.args[0], req.args[1]
            ctx_enc = self.tokenizer.encode(context).tolist()
            cont_enc = self.tokenizer.encode(continuation, bos=False).tolist()
            results.append(self._score_tokens(ctx_enc, cont_enc))
        _empty_cache()
        return self.all_gather_results(results, tag="loglikelihood")

    def generate_until(self, requests):
        rank, world = du.global_rank(), du.world_size()
        results = []
        desc = f"generate_until {du.rank_label()}"
        for local_i, req in enumerate(tqdm.tqdm(requests[rank::world], desc=desc, position=rank)):
            sample_id = request_sample_id(req, global_request_index=local_i * world + rank, rank=rank)
            prompt, gen_args = req.args[0], req.args[1]
            max_new_tokens = int(gen_args.get("max_gen_toks", gen_args.get("max_length", self.max_gen_toks)))
            do_sample = bool(gen_args.get("do_sample", False))
            temperature = float(gen_args.get("temperature", 1.0))
            top_p = float(gen_args.get("top_p", 1.0))
            top_k = gen_args.get("top_k", None)
            if not do_sample:
                temperature, top_p = 0.0, 0.0  # argmax requires both to be zero

            prompt_tensor = self.tokenizer.encode(prompt, device=self._device)
            max_len = self.model.max_seq_length
            if prompt_tensor.size(0) + max_new_tokens > max_len:
                keep_prompt_len = max_len - max_new_tokens
                prompt_tensor = prompt_tensor[-keep_prompt_len:]
                print(f"⚠️ [{du.rank_label()}] prompt truncated to {keep_prompt_len} tokens")
            prompt_len = prompt_tensor.size(0)

            with torch.no_grad():
                self._set_eval_cache()
                try:
                    _synchronize(self._device)
                    _reset_peak_memory(self._device)
                    timing = {}
                    out = generate(self.model, prompt_tensor, max_returned_tokens=prompt_len + max_new_tokens,
                                   temperature=temperature, top_k=top_k, top_p=top_p,
                                   eos_id=self.tokenizer.eos_id, timing=timing)
                    _synchronize(self._device)
                    peak_memory = _peak_memory(self._device)
                    # The last output token has not been forwarded into the cache.
                    self.compression_metrics.append({
                        "request_type": "generate_until", "rank": rank, "sample_id": sample_id,
                        "prompt_len": prompt_len, "generated_tokens": int(out.numel()) - prompt_len,
                        **cache_compression_snapshot(self.model, int(out.numel()) - 1),
                    })
                finally:
                    self._reset_eval_cache()

            generated_tokens = out[prompt_len:]
            gen_len = generated_tokens.size(0)
            total_s, prefill_s = timing["total_time_s"], timing["prefill_time_s"]
            decode_s, decode_tokens = timing["decode_time_s"], timing["decode_tokens"]
            self.gen_metrics.append({
                "rank": rank, "sample_id": sample_id, "prompt_len": prompt_len, "generated_tokens": gen_len,
                "total_time_s": total_s, "prefill_time_s": prefill_s, "ttft_ms": prefill_s * 1000,
                "prefill_tokens_per_sec": prompt_len / prefill_s if prefill_s > 0 else None,
                "decode_time_s": decode_s, "decode_tokens": decode_tokens,
                "decode_ms_per_token": decode_s / decode_tokens * 1000 if decode_tokens else None,
                "decode_tokens_per_sec": decode_tokens / decode_s if decode_tokens and decode_s > 0 else None,
                "gen_tokens_per_sec": gen_len / total_s if total_s > 0 else None,
                **peak_memory,
            })
            results.append(self.tokenizer.decode(generated_tokens))
        _empty_cache()
        return self.all_gather_results(results, tag="generate_until")

    def loglikelihood_rolling(self, requests):
        rank, world = du.global_rank(), du.world_size()
        results = []
        window = self.model.max_seq_length - 1
        desc = f"loglikelihood_rolling {du.rank_label()}"
        for req in tqdm.tqdm(requests[rank::world], desc=desc, position=rank):
            (text,) = req.args
            tokens = self.tokenizer.encode(text, bos=False).tolist()
            total = 0.0
            for start in range(0, len(tokens), window):
                chunk = tokens[start:start + window]
                ctx = [self.tokenizer.bos_id] if start == 0 else [tokens[start - 1]]
                logprob, _ = self._score_tokens(ctx, chunk, request_type="loglikelihood_rolling")
                total += logprob
            results.append(total)
        _empty_cache()
        return self.all_gather_results(results, tag="loglikelihood_rolling")

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_id

    @property
    def max_length(self):
        return self.model.max_seq_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string):
        return self.tokenizer.encode(string).tolist()

    def tok_decode(self, tokens):
        return self.tokenizer.decode(torch.tensor(tokens))
