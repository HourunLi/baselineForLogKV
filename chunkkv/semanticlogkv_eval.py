"""ChunkKV behind SemanticLogKV's ``LogKVLM``: same loading, requests, scoring and metrics."""

import hashlib
from pathlib import Path

import torch

from .core import ChunkKVSettings
from .model import ChunkKVModel


def compression_snapshot(model, processed_tokens):
    """Adapt metric collection locally; never register hooks in the reference package."""
    from chunkkv._vendor.semanticlogkv.litgpt.kv_compression import _with_ratios, cache_compression_snapshot

    if isinstance(model, ChunkKVModel):
        return _with_ratios(model.kv_compression_snapshot(processed_tokens))
    return cache_compression_snapshot(model, processed_tokens)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_keys(checkpoint_dir, model):
    """The reference loader uses strict=False; fail instead of keeping random weights."""
    state = torch.load(Path(checkpoint_dir) / "lit_model.pth", map_location="cpu", mmap=True, weights_only=False)
    state = state.get("model", state)
    expected, found = set(model.state_dict()), set(state)
    if expected != found:
        raise ValueError(f"Checkpoint keys differ from the LitGPT model: missing={sorted(expected - found)[:5]}, "
                         f"unexpected={sorted(found - expected)[:5]}")


def kv_bytes_per_token(config, dtype) -> int:
    """K and V of one token across all layers and KV groups, as stored by ChunkKV."""
    return config.n_layer * config.n_query_groups * 2 * config.head_size * torch.empty((), dtype=dtype).element_size()


def make_chunkkv_lm(evaluator, options):
    """Implement the reference LM interface, inheriting its request and scoring methods.

    ``options`` holds ChunkKVSettings fields; ``budget: "logkv"`` sizes the prompt
    budget from the persistent cache bytes of the experiment's LogKV cache, ``budget_bytes`` (64K / 128K)
    from a given byte total: the most entries per layer whose K/V across the KV groups and, when streaming,
    the shared int64 position fit it.
    """
    class ChunkKVLM(evaluator.LogKVLM):
        def __init__(self, checkpoint_dir, device="cuda", config_overrides=None,
                     log_kv_dense_mode=False, swa_window_size=None, **kwargs):
            if log_kv_dense_mode or swa_window_size is not None:
                raise ValueError("ChunkKV cannot be combined with LogKV dense/SWA switches")
            if set(config_overrides or {}) not in ({"block_size"}, {"block_size", "rope_adjustments"}):
                raise ValueError("ChunkKV requires the explicit shared block_size (and YaRN for 64K / 128K)")
            # The reference constructor loads tokenizer, LitGPT config and weights;
            # dense mode only means it skips LogKV-specific model setup.
            super().__init__(checkpoint_dir, device=device, config_overrides=config_overrides,
                             log_kv_dense_mode=True, **kwargs)
            verify_checkpoint_keys(checkpoint_dir, self.model)
            dtype = next(self.model.parameters()).dtype
            per_token = kv_bytes_per_token(self.config, dtype)
            method_options = dict(options)
            budget_bytes = method_options.pop("budget_bytes", None)
            # A streaming layer also keeps one int64 position per entry (shared by its KV groups).
            per_entry = per_token + (self.config.n_layer * 8 if method_options.get("streaming") else 0)
            budget, source = method_options.get("budget"), "prompt_ratio"
            reference_bytes = None
            if budget == "logkv":
                reference_bytes = self._reference_logkv_bytes()
                budget, source = reference_bytes // per_entry, "logkv_persistent_bytes"
            elif budget == "bytes":
                reference_bytes = int(budget_bytes)
                budget, source = reference_bytes // per_entry, "budget_bytes"
            elif budget is not None:
                source = "explicit_tokens"
            settings = ChunkKVSettings(**{**method_options, "budget": budget})
            self.model = ChunkKVModel(self.model, settings)
            self.log_kv_dense_mode, self.swa_window_size = False, None
            self.cache_budget = {
                "method": "ChunkKV", "allocation": "dynamic", "persistent_cache_bytes": None,
                "budget_source": source, "prompt_budget_tokens": settings.budget,
                "prompt_budget_bytes": settings.budget * per_token if settings.budget else None,
                "bytes_per_token": per_token, "reference_logkv_bytes": reference_bytes,
                "decode_tokens": "appended uncompressed after the compressed prompt",
                "chunkkv": settings.as_dict(),
            }
            if settings.streaming:
                # Every layer holds at most `budget` entries for the whole request, decode steps included.
                self.cache_budget.update(allocation="fixed", persistent_cache_bytes=settings.budget * per_entry,
                                         bytes_per_entry=per_entry,
                                         decode_tokens="selected with the prompt: the budget holds throughout")
            if evaluator._is_main():
                print(f"🧮 ChunkKV cache budget: {self.cache_budget}")

        def _reference_logkv_bytes(self):
            """Allocate the experiment's LogKV cache once, as SWA budget matching does."""
            self.log_kv_dense_mode = False
            try:
                super()._set_eval_cache()
                return self.cache_budget["persistent_cache_bytes"]
            finally:
                self.model.clear_kv_cache()
                self.log_kv_dense_mode, self._eval_cache_ready = True, False

        def _set_eval_cache(self):
            self.model.reset_cache()

        def _reset_eval_cache(self):
            self.model.release_cache()

    return ChunkKVLM
