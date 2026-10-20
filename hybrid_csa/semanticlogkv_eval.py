"""HybridCSA behind the fixed evaluator's LM interface, retaining request/scoring semantics."""

import hashlib
from pathlib import Path

import torch

from .settings import HybridCSASettings
from .model import HybridCSAModel


def compression_snapshot(model, processed_tokens):
    from hybrid_csa._vendor.semanticlogkv.litgpt.kv_compression import _with_ratios, cache_compression_snapshot

    if isinstance(model, HybridCSAModel):
        return _with_ratios(model.kv_compression_snapshot(processed_tokens))
    return cache_compression_snapshot(model, processed_tokens)


def summarize_compression(records):
    from ._vendor.semanticlogkv.litgpt.kv_compression import summarize_compression as summarize
    summary = summarize(records)
    fields = ("dense_kv_scalars", "main_kv_scalars", "index_key_scalars", "compressor_state_scalars")
    totals = {field: sum(row[field] for row in records) for field in fields}
    dense = totals["dense_kv_scalars"]
    combined = totals["main_kv_scalars"] + totals["index_key_scalars"]
    return {**summary, **totals,
            "basis": "theoretical elements; sum retained / sum dense across requests",
            "kv_plus_index_retention_ratio": combined / dense if dense else None,
            "all_state_retention_ratio": (combined + totals["compressor_state_scalars"]) / dense if dense else None}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_keys(checkpoint_dir, model):
    """The reference loader uses strict=False; reject incomplete original Base weights."""
    state = torch.load(Path(checkpoint_dir) / "lit_model.pth", map_location="cpu", mmap=True, weights_only=False)
    state = state.get("model", state)
    expected, found = set(model.state_dict()), set(state)
    if expected != found:
        raise ValueError(f"Checkpoint keys differ from the LitGPT model: missing={sorted(expected - found)[:5]}, "
                         f"unexpected={sorted(found - expected)[:5]}")


def make_hybrid_csa_lm(evaluator, options, adapter_path):
    """Only model/cache construction changes; generation and scoring stay inherited."""
    from .checkpoint import base_identity, load_adapter

    class HybridCSALM(evaluator.LogKVLM):
        def __init__(self, checkpoint_dir, device="cuda", config_overrides=None,
                     log_kv_dense_mode=False, swa_window_size=None, **kwargs):
            if log_kv_dense_mode or swa_window_size is not None:
                raise ValueError("HybridCSA cannot be combined with LogKV dense/SWA switches")
            if set(config_overrides or {}) != {"block_size"}:
                raise ValueError("HybridCSA requires the explicit shared block_size")
            super().__init__(checkpoint_dir, device=device, config_overrides=config_overrides,
                             log_kv_dense_mode=True, **kwargs)
            verify_checkpoint_keys(checkpoint_dir, self.model)
            settings = HybridCSASettings(**options)
            self.model = HybridCSAModel(self.model, settings)
            adapter_record = load_adapter(self.model, adapter_path, checkpoint_dir)
            base = kwargs["tokenizer_dir"]
            if Path(base).resolve() != Path(checkpoint_dir).resolve() and adapter_record["base"] != base_identity(base):
                raise ValueError("HybridCSA must use the experiment's original Base weights and tokenizer")
            self.model.eval()
            evaluator.hybrid_csa_adapter_record = adapter_record
            self.log_kv_dense_mode, self.swa_window_size = False, None
            self.cache_budget = {
                "method": "Qwen3-Hybrid-CSA4",
                "logical_budget": "theoretical full-layer KV + compressed KV + local KV; at request end",
                "auxiliary_budget": "index keys and compressor state counted separately as theoretical scalars",
                "theoretical_at_32768": self.model.theoretical_budget(32768),
                "attention_budget": "index_topk limits attended compressed slots, not stored compressed history",
                "decode_tokens": "compressed with the same causal rule as prefill",
                "full_layers_zero_based": list(settings.full_layers(self.config.n_layer)),
                "hybrid_csa": settings.as_dict(), "adapter": adapter_record,
            }
            if evaluator._is_main():
                print(f"HybridCSA cache budget: {self.cache_budget}")

        def _set_eval_cache(self):
            self.model.reset_cache()

        def _reset_eval_cache(self):
            self.model.release_cache()

    return HybridCSALM
