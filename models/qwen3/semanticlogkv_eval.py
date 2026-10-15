"""DMS model interface for the existing SemanticLogKV request/scoring pipeline."""

import hashlib
from pathlib import Path

import torch
from torch import nn


def compression_snapshot(model, processed_tokens):
    """Adapt metric collection locally; never register hooks in the reference package."""
    from models._vendor.semanticlogkv.litgpt.kv_compression import _with_ratios, cache_compression_snapshot

    if isinstance(model, DMSModel):
        return _with_ratios(model.kv_compression_snapshot(processed_tokens))
    return cache_compression_snapshot(model, processed_tokens)


def make_dms_lm(evaluator):
    """Implement the reference LM interface in this baseline, inheriting its request methods."""
    class DMSLM(evaluator.LogKVLM):
        def __init__(self, checkpoint_dir, device="cuda", config_overrides=None,
                     tokenizer_dir=None, process_group=None, log_kv_dense_mode=False,
                     swa_window_size=None, **kwargs):
            evaluator.LM.__init__(self)
            if log_kv_dense_mode or swa_window_size is not None:
                raise ValueError("DMS cannot be combined with LogKV dense/SWA switches")
            if set(config_overrides or {}) != {"block_size"}:
                raise ValueError("DMS requires the explicit shared block_size")
            self.process_group, self._device = process_group, device
            self.checkpoint_dir = checkpoint_dir
            resolved = evaluator._find_tokenizer_dir(checkpoint_dir, tokenizer_dir, group=process_group)
            self.tokenizer = evaluator.Tokenizer(resolved)
            self.model = DMSModel.from_checkpoint(checkpoint_dir, resolved,
                                                  config_overrides["block_size"], device)
            self.config = self.model.hf_model.config
            self.log_kv_dense_mode, self.swa_window_size = False, None
            self.cache_budget = {"allocation": "dynamic", "persistent_cache_bytes": None}
            self.gen_metrics, self.ppl_metrics, self.compression_metrics = [], [], []

        def _set_eval_cache(self):
            self.model.reset_kv_cache()

        def _reset_eval_cache(self):
            self.model.cache = None

    return DMSLM


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_base_source(provenance, tokenizer_dir):
    """Fail closed when a trained export cannot be tied to the comparison Base."""
    if not provenance or not provenance.get("tokenizer_files"):
        raise ValueError("Missing semanticlogkv_base provenance; prepare the shared Base and retrain/export")
    source = Path(tokenizer_dir).resolve()
    if file_sha256(source / "lit_model.pth") != provenance["source_weight_sha256"]:
        raise ValueError("DMS Base weights differ from the SemanticLogKV comparison checkpoint")
    for name, expected in provenance.get("source_config_files", {}).items():
        if file_sha256(source / name) != expected:
            raise ValueError(f"Comparison Base config differs from the DMS source: {name}")
    for name, expected in provenance["tokenizer_files"].items():
        if file_sha256(source / name) != expected:
            raise ValueError(f"Comparison tokenizer differs from the DMS Base tokenizer: {name}")


class DMSModel(nn.Module):
    """Expose LitGPT's forward/cache interface without replacing its generation loop."""

    def __init__(self, model, max_seq_length):
        super().__init__()
        if not 0 < max_seq_length <= model.config.max_position_embeddings:
            raise ValueError("Comparison context exceeds the DMS model's position window")
        self.hf_model = model
        self.max_seq_length = max_seq_length
        self.cache = None

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, tokenizer_dir, max_seq_length, device):
        from .check_env import check_environment
        from .configuration_qwen3_dms import Qwen3ConfigDMS
        from .modeling_qwen3_dms import Qwen3ForCausalLMDMS

        check_environment()
        config = Qwen3ConfigDMS.from_pretrained(checkpoint_dir, local_files_only=True)
        if not config.dms_separate_alpha:
            raise ValueError("Expected an exported DMS adapter checkpoint")
        if config.dms_disable_eviction:
            raise ValueError("DMS comparison requires eviction enabled")
        verify_base_source(getattr(config, "semanticlogkv_base", None), tokenizer_dir)
        model, loading = Qwen3ForCausalLMDMS.from_pretrained(
            checkpoint_dir, config=config, dtype=torch.bfloat16,
            local_files_only=True, output_loading_info=True,
        )
        if loading["missing_keys"] or loading["unexpected_keys"] or loading["mismatched_keys"]:
            raise ValueError(f"Incomplete DMS checkpoint: {loading}")
        # Keep the official memory-bounded prefill; request processing stays in LitGPT.
        model.config.dms_chunked_prefill = 4096
        return cls(model.to(device).eval(), max_seq_length).eval()

    def reset_kv_cache(self):
        # DMS reset releases tensors; a new cache also resets its global mode.
        self.cache = self.hf_model.get_cache()

    def forward(self, input_ids, input_pos=None, input_pos_maxp1=None, lm_head_start=0):
        if self.cache is None:
            raise RuntimeError("Initialize the DMS request cache before forwarding tokens")
        return self.hf_model(
            input_ids=input_ids, cache_position=input_pos,
            past_key_values=self.cache, use_cache=True,
            logits_to_keep=input_ids.size(1) - lm_head_start,
        ).logits

    def kv_compression_snapshot(self, processed_tokens):
        if self.cache is None or len(self.cache.layers) != self.hf_model.config.num_hidden_layers:
            raise ValueError("Cannot measure an incomplete DMS cache")
        counts = dict(processed_tokens=processed_tokens, layer_group_count=0,
                      recent_slots=0, exact_slots=0, hierarchical_slots=0,
                      retained_slots=0, dense_slots=0)
        for layer in self.cache.layers:
            if layer.get_seq_length() != processed_tokens:
                raise ValueError("DMS cache and forwarded token counts differ")
            lengths = [part.cache_seq_lengths for part in (layer.paged_cache, layer.cont_cache)
                       if part.cache_seq_lengths is not None]
            if not lengths:
                raise ValueError("DMS cache contains no valid KV lengths")
            retained = int(sum(item.sum().item() for item in lengths))
            groups = lengths[0].numel()
            recent = min(processed_tokens, layer.dms_window_size) * groups
            if not recent <= retained <= processed_tokens * groups:
                raise ValueError("Invalid DMS retained KV count")
            counts["layer_group_count"] += groups
            counts["recent_slots"] += recent
            counts["exact_slots"] += retained - recent
            counts["retained_slots"] += retained
            counts["dense_slots"] += processed_tokens * groups
        return counts
