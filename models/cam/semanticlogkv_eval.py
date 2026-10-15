"""Local adapter: reuse the reference tokenizer, request methods and metric aggregation."""

from .model import CaMModel


def compression_snapshot(model, processed_tokens):
    from models._vendor.semanticlogkv.litgpt.kv_compression import _with_ratios, cache_compression_snapshot
    if isinstance(model, CaMModel):
        return _with_ratios(model.kv_compression_snapshot(processed_tokens))
    return cache_compression_snapshot(model, processed_tokens)


def install_adapter(evaluator, options, dense=False):
    class CaMLM(evaluator.LogKVLM):
        def __init__(self, checkpoint_dir, device="cuda", config_overrides=None,
                     tokenizer_dir=None, process_group=None, log_kv_dense_mode=False,
                     swa_window_size=None, **kwargs):
            evaluator.LM.__init__(self)
            if log_kv_dense_mode or swa_window_size is not None:
                raise ValueError("Use the CaM launcher's backend selector for Full KV")
            if set(config_overrides or {}) != {"block_size"}:
                raise ValueError("CaM requires the explicit shared block_size")
            self.process_group, self._device = process_group, device
            self.checkpoint_dir = checkpoint_dir
            resolved = evaluator._find_tokenizer_dir(checkpoint_dir, tokenizer_dir, group=process_group)
            self.tokenizer = evaluator.Tokenizer(resolved)
            self.model = CaMModel.from_checkpoint(
                checkpoint_dir, resolved, config_overrides["block_size"], device, dense=dense, **options,
            )
            self.config = self.model.hf_model.config
            self.log_kv_dense_mode, self.swa_window_size = dense, None
            self.cache_budget = dict(
                allocation="dynamic", persistent_cache_bytes=None,
                measurement="per-request compression samples: raw K + V + FP32 scores; excludes RoPE and workspace",
                capacity_tokens=None if dense else options["start_size"] + options["recent_size"],
                attention_working_tokens="capacity + 1 before post-attention eviction" if not dense else None,
            )
            self.gen_metrics, self.ppl_metrics, self.compression_metrics = [], [], []

        def _set_eval_cache(self):
            self.model.reset_kv_cache()

        def _reset_eval_cache(self):
            self.model.clear_kv_cache()

    evaluator.LogKVLM = CaMLM
    evaluator.cache_compression_snapshot = compression_snapshot
    def print_backend(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("🧩 logKV "):
            args = ("🧩 Qwen3 Full KV" if dense else "🧩 CaM online / Qwen3 GQA", *args[1:])
        print(*args, **kwargs)
    evaluator.print = print_backend
