"""Qwen3 wrappers for strict online CaM and an identical-weight Full KV control.

Written against the evaluation environment's transformers (torch218: 4.51.3) and newer releases alike: the
dtype goes in as ``torch_dtype``, the Full KV cache is a plain ``DynamicCache()``, and no attention-mask
mapping is passed (CaM attends causally over its own cache and ignores a mask built for the forward).
"""

import torch
from torch import nn
from transformers import DynamicCache, Qwen3ForCausalLM

from models.qwen3.prepare_base import verify_prepared_base
from models.qwen3.provenance import verify_base_source
from .attention import install_cam_attention


class CaMModel(nn.Module):
    def __init__(self, model, max_seq_length, *, start_size=32, recent_size=2016,
                 merge_size=16, seed=42, prefill_chunk_size=256, dense=False, merge_interval=1):
        super().__init__()
        # CaM's RoPE positions are cache slots, at most the cache plus one forward, so its requests may run past
        # the trained window (64K / 128K); Full KV keeps absolute positions and stays within it.
        if max_seq_length <= 0 or (dense and max_seq_length > model.config.max_position_embeddings):
            raise ValueError("Comparison context exceeds the Qwen3 position window")
        if not dense and start_size + recent_size + merge_interval > model.config.max_position_embeddings:
            raise ValueError("CaM's cache plus one merge step must fit the Qwen3 position window")
        if prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be positive")
        if (model.config.model_type != "qwen3" or getattr(model.config, "rope_scaling", None)
                or getattr(model.model, "has_sliding_layers", False)):
            raise ValueError("Expected dense Qwen3 without sliding layers or RoPE extensions")
        self.hf_model = model.eval()
        self.max_seq_length = max_seq_length
        self.prefill_chunk_size = prefill_chunk_size
        self.dense = dense
        self.start_size, self.recent_size = start_size, recent_size
        self.merge_interval = merge_interval
        self.layers = [] if dense else install_cam_attention(
            model, start_size, recent_size, merge_size, seed, merge_interval,
        )
        self.cache = None
        self.processed_tokens = 0

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, tokenizer_dir, max_seq_length, device, **options):
        manifest = verify_prepared_base(checkpoint_dir)
        verify_base_source(manifest, tokenizer_dir)
        if str(device).startswith("cuda"):
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("CaM evaluation requires a BF16-capable CUDA GPU")
            dtype = torch.bfloat16
        elif str(device) == "cpu":
            dtype = torch.float32
        else:
            raise ValueError("Supported devices: CUDA for evaluation, CPU for small checks")
        model, loading = Qwen3ForCausalLM.from_pretrained(
            checkpoint_dir, local_files_only=True, torch_dtype=dtype,
            attn_implementation="sdpa",  # CaM replaces the attention; sdpa lets HF skip the causal mask
            output_loading_info=True,
        )
        if loading["missing_keys"] or loading["unexpected_keys"] or loading["mismatched_keys"]:
            raise ValueError(f"Incomplete Qwen3 checkpoint: {loading}")
        return cls(model.to(device), max_seq_length, **options).eval()

    def reset_kv_cache(self):
        self.processed_tokens = 0
        for layer in self.layers:
            layer.reset()
        self.cache = DynamicCache() if self.dense else True

    def clear_kv_cache(self):
        for layer in self.layers:
            layer.reset()
        self.cache = None
        self.processed_tokens = 0

    @torch.no_grad()
    def forward(self, input_ids, input_pos=None, input_pos_maxp1=None, lm_head_start=0):
        if self.cache is None:
            raise RuntimeError("Initialize the request cache before forwarding tokens")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("Shared evaluation requires one nonempty, unpadded request")
        length = input_ids.shape[1]
        expected = torch.arange(self.processed_tokens, self.processed_tokens + length, device=input_ids.device)
        if input_pos is not None and not torch.equal(input_pos, expected):
            raise ValueError("CaM input positions must be append-only and contiguous")
        if self.processed_tokens + length > self.max_seq_length or not 0 <= lm_head_start < length:
            raise ValueError("Input exceeds context window or requested logits span is invalid")
        logits = []
        for start in range(0, length, self.prefill_chunk_size):
            end = min(length, start + self.prefill_chunk_size)
            kwargs = dict(input_ids=input_ids[:, start:end], position_ids=expected[None, start:end],
                          cache_position=expected[start:end], use_cache=self.dense)
            if self.dense:
                kwargs["past_key_values"] = self.cache
            hidden = self.hf_model.model(**kwargs).last_hidden_state
            if end > lm_head_start:
                logits.append(self.hf_model.lm_head(hidden[:, max(0, lm_head_start - start):]))
        self.processed_tokens += length
        return torch.cat(logits, dim=1)

    def kv_compression_snapshot(self, processed_tokens):
        if self.cache is None or self.processed_tokens != processed_tokens:
            raise ValueError("Cache and forwarded token counts differ")
        config = self.hf_model.config
        groups = config.num_hidden_layers * config.num_key_value_heads
        if self.dense:
            if hasattr(self.cache, "layers"):  # transformers >= 4.56
                tensors = [t for layer in self.cache.layers for t in (layer.keys, layer.values)]
            else:  # 4.51 (torch218)
                tensors = [*self.cache.key_cache, *self.cache.value_cache]
            retained = processed_tokens * groups
            recent, exact = retained, 0
            score_bytes = 0
        else:
            if any(layer.processed_tokens != processed_tokens for layer in self.layers):
                raise ValueError("CaM cache token counts differ between layers")
            tensors = [t for layer in self.layers for t in (layer.keys, layer.values)]
            retained = sum(layer.keys.shape[1] * layer.keys.shape[2] for layer in self.layers)
            exact = min(processed_tokens, self.start_size) * groups
            recent = retained - exact
            score_bytes = sum(layer.scores.untyped_storage().nbytes() for layer in self.layers)
        kv_bytes = sum(t.untyped_storage().nbytes() for t in tensors)
        return dict(processed_tokens=processed_tokens, layer_group_count=groups,
                    recent_slots=recent, exact_slots=exact, hierarchical_slots=0,
                    retained_slots=retained, dense_slots=processed_tokens * groups,
                    kv_storage_bytes=kv_bytes, attention_score_bytes=score_bytes,
                    persistent_cache_bytes=kv_bytes + score_bytes)
