# Vendored for ChunkKV; local namespace changes are recorded in SOURCE.json.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self

from chunkkv._vendor.semanticlogkv.litgpt.config import Config
from chunkkv._vendor.semanticlogkv.litgpt.chunked_loss import chunked_linear_cross_entropy
from chunkkv._vendor.semanticlogkv.litgpt.log_kv_cache import (
    LogKVStreamTrainingAttention,
    LogStructuredKVCache,
    log_kv_chunk_attention,
)
from chunkkv._vendor.semanticlogkv.litgpt.log_kv_diag import DIAG as LOG_KV_DIAG, diag_block_attention
from chunkkv._vendor.semanticlogkv.litgpt.scripts.convert_hf_checkpoint import qkv_reassemble


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, block_idx) for block_idx in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.mask_cache: torch.Tensor | None = None
        self.max_seq_length = self.config.block_size

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}."
                " This is likely because the input text exceeds the supported context length of this model."
            )
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected
        if self.mask_cache is not None and self.mask_cache.shape[-1] < value:
            print(
                f"Warning: KV cache has length {self.mask_cache.shape[-1]} < {value} = max_seq_length. Call 'set_kv_cache' before doing any forwards!"
            )

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, GroupedTopkRouter):
            torch.nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        input_pos: torch.Tensor | None = None,
        input_pos_maxp1: int | None = None,
        lm_head_chunk_size: int = 0,
        lm_head_start: int | None = None,
        targets: torch.Tensor | None = None,
        loss_chunk_size: int = 128,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        If `input_pos` is provided, the KV cache uses K and V vectors for
        positions smaller than entries in `input_pos`. For efficiency, pass
        `input_pos_maxp1` as `max(input_pos) + 1` if already available from
        your forward algorithm. This slices the KV cache buffers and speeds
        up multi-head attention.

        Without `input_pos_maxp1`, the computation uses the full KV cache
        (`max_seq_length`) with masking applied. Note that inferring
        `input_pos_maxp1` from `input_pos` causes graph breaks and prevents
        compilation.

        Args:
            idx: Token indices of input sequences, shape `(B, T)`, where `B`
                is batch size.
            input_pos: Optional. Positions of input tokens. The default is
                `arange(T)`. Can have shape `(T,)` or `(B, T)` (batched index).
            input_pos_maxp1: Optional. See above.
            lm_head_chunk_size: Optional. If `lm_head_chunk_size > 0`, the final
                `lm_head` computation is done in chunks of this size.
            lm_head_start: Optional. If given, the final norm + `lm_head` are
                applied only to positions `>= lm_head_start`. Loglikelihood
                scoring needs logits only for the continuation span; at 32K
                context the full-sequence logit tensor is ~10 GB (bf16), while
                the sliced one is `cont_len x vocab`. `None` keeps the full
                output (generation, training).
            targets: Optional, already aligned labels of shape `(B, T)`; -100
                is ignored. Returns a scalar mean loss instead of logits.
            loss_chunk_size: Flattened tokens per projection/loss checkpoint.
                Used with targets; zero processes all tokens in one chunk.

        Returns:
            Logit outputs, shape `(B, T, config.padded_vocab_size)`. If
            `lm_head_chunk_size > 0`, this is a list of chunks of shape
            `(B, lm_head_chunk_size, config.padded_vocab_size)`, the final
            entry can be shorter. If `lm_head_start` is given, the second
            dimension is `T - lm_head_start` instead of `T`.

        """
        if targets is not None and (targets.shape != idx.shape or lm_head_start is not None or input_pos is not None):
            raise ValueError("training targets must match idx and require unsliced, uncached model output")
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            if input_pos.dim() > 2:
                # otherwise, things go wrong in `apply_rope`
                raise ValueError(f"input_pos must have 1 or 2 dimensions, input_pos.shape = {input_pos.shape}")
            if input_pos.shape[-1] != T:
                raise ValueError(f"input_pos.shape[-1] = {input_pos.shape[-1]} != {T} = idx.shape[1], must be the same")
            cos = batched_index_select(self.cos, 0, input_pos)
            sin = batched_index_select(self.sin, 0, input_pos)
            if input_pos.dim() == 1:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
            local_cache_masks = all(
                isinstance(block.attn.kv_cache, LogStructuredKVCache)
                or (isinstance(block.attn.kv_cache, KVCache) and block.attn.kv_cache.is_sliding_window)
                for block in self.transformer.h
            )
            if local_cache_masks:
                mask = None
            else:
                if self.mask_cache is None:
                    raise TypeError("You need to call `gpt.set_kv_cache()`")
                mask = batched_index_select(self.mask_cache, 2, input_pos)
                if mask.dim() > 4:
                    # the mask cache has a batch dim of 1 in addition to the one
                    # we get if input_pos has a batch dimension
                    mask = mask.view(*(mask.shape[0:1] + mask.shape[2:]))
                if input_pos_maxp1 is not None:
                    # Shorten final dimension so it just covers all `input_pos` entries
                    if input_pos_maxp1 > self.max_seq_length:
                        raise ValueError(f"Positions in 'input_pos' must be in [0,{self.max_seq_length})")
                    mask = mask[..., :input_pos_maxp1]
        else:
            # unsqueeze to have a batch dimension
            cos = self.cos[:T].unsqueeze(0)
            sin = self.sin[:T].unsqueeze(0)
            # `cos`, `sin` have shape (1, T, config.rope_n_elem)
            mask = None  # defaults to causal mask
            input_pos_maxp1 = None

        span_kwargs = {}
        if getattr(self, "_alpha_enabled", False):
            table = getattr(self, "alpha_span_boundaries", None)
            if table is None:
                raise RuntimeError("AlphaLogKV needs set_alpha_span_boundary_ids() before forward")
            span_kwargs["span_ends"] = table[idx].detach().cpu().tolist()
        x = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd**0.5, dtype=x.dtype)

        for block_idx, block in enumerate(self.transformer.h):
            if self.config.rope_indices is not None:
                x = block(
                    x,
                    cos[..., self.config.rope_indices[block_idx]],
                    sin[..., self.config.rope_indices[block_idx]],
                    mask,
                    input_pos,
                    input_pos_maxp1,
                    **span_kwargs,
                )
            else:
                x = block(x, cos, sin, mask, input_pos, input_pos_maxp1, **span_kwargs)
        if lm_head_start is not None:
            # Drop hidden states the caller does not need logits for BEFORE the
            # O(T x vocab) projection — the transformer stack above already ran
            # on the full sequence, so cache state / attention are unaffected.
            x = x[:, lm_head_start:]
        x = self.transformer.ln_f(x)
        if targets is not None:
            return chunked_linear_cross_entropy(
                x, self.lm_head.weight, targets, self.lm_head.bias,
                chunk_size=loss_chunk_size, softcap=self.config.final_logit_softcapping,
            )
        clamp_head = (
            partial(do_softcapping, thresh=self.config.final_logit_softcapping)
            if self.config.final_logit_softcapping is not None
            else nn.Identity()
        )
        if lm_head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            return [clamp_head(self.lm_head(x_i)) for x_i in x.split(lm_head_chunk_size, dim=1)]
        else:
            return clamp_head(self.lm_head(x))  # (B, T, padded_vocab_size)

    def set_alpha_span_boundary_ids(self, ids) -> None:
        """Install the tokenizer's fixed natural-boundary lookup once per model."""
        table = torch.zeros(self.config.padded_vocab_size, device=self.transformer.wte.weight.device, dtype=torch.bool)
        table[list(ids)] = True
        self.register_buffer("alpha_span_boundaries", table, persistent=False)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(self, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_adjustments is None:
            extra_config = None

        else:
            # Check for mutually exclusive parameter sets
            llama3_params = ["low_freq_factor", "high_freq_factor"]
            yarn_params = ["beta_fast", "beta_slow"]

            has_llama3 = any(param in self.config.rope_adjustments for param in llama3_params)
            has_yarn = any(param in self.config.rope_adjustments for param in yarn_params)

            if has_llama3 and has_yarn:
                raise ValueError(
                    "RoPE adjustments cannot contain both Llama3 parameters (low_freq_factor, high_freq_factor) "
                    "and YaRN parameters (beta_fast, beta_slow). These are mutually exclusive."
                )

            # Llama3-style RoPE
            if has_llama3:
                adjusted_params_required = ["factor", "low_freq_factor", "high_freq_factor", "original_max_seq_len"]
                params_present = [param in self.config.rope_adjustments for param in adjusted_params_required]
                if all(params_present):
                    extra_config = {name: self.config.rope_adjustments[name] for name in adjusted_params_required}
                else:
                    missing_params = [
                        param for param, present in zip(adjusted_params_required, params_present) if not present
                    ]
                    raise ValueError(
                        f"The following Llama3 RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "
                        "All Llama3 parameters must be specified together."
                    )

            # YaRN-style RoPE
            elif has_yarn:
                # Required: factor, beta_fast, beta_slow, original_max_seq_len
                # Optional: mscale, mscale_all_dim
                yarn_required_params = ["factor", "beta_fast", "beta_slow", "original_max_seq_len"]
                params_present = [param in self.config.rope_adjustments for param in yarn_required_params]

                if not all(params_present):
                    missing_params = [
                        param for param, present in zip(yarn_required_params, params_present) if not present
                    ]
                    raise ValueError(
                        f"The following YaRN RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "
                        "All YaRN required parameters must be specified together."
                    )

                extra_config = {name: self.config.rope_adjustments[name] for name in yarn_required_params}

                # Add optional YaRN parameters
                for param in ["mscale", "mscale_all_dim"]:
                    if param in self.config.rope_adjustments:
                        extra_config[param] = self.config.rope_adjustments[param]

            # Linear or standard RoPE
            elif "factor" in self.config.rope_adjustments:
                # linear RoPE
                adjusted_params_required = ["factor"]
                extra_config = {name: self.config.rope_adjustments[name] for name in adjusted_params_required}
            else:
                extra_config = None  # uses standard RoPE

        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            extra_config=extra_config,
            rope_local_base_freq=self.config.rope_local_base_freq,
        )

    def rope_cache_length(self) -> int:
        """
        Extract the head dimension (n_elem) from RoPE cache regardless of shape.

        The RoPE cache can have different shapes depending on model configuration:
        - Standard RoPE: (seq_len, n_elem) - 2D tensor
        - Dual RoPE (local/global): (seq_len, n_elem, 2) - 3D tensor

        Returns:
            int: n_elem (head dimension for RoPE)
        """
        return self.cos.size(1)

    def set_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int | None = None,
        rope_cache_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self._alpha_enabled = False
        if rope_cache_length is None:
            rope_cache_length = self.rope_cache_length()

        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size,
                max_seq_length,
                rope_cache_length,
                device,
                dtype,
            )
            block.attn._log_kv_pending = None

        if all(isinstance(block.attn.kv_cache, KVCache) and block.attn.kv_cache.is_sliding_window
               for block in self.transformer.h):
            self.mask_cache = None
        elif self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self._alpha_enabled = False
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None
            block.attn._log_kv_pending = None

    def reset_kv_cache(self) -> None:
        """Reset every layer's standard KV cache state in place — no reallocation."""
        for block in self.transformer.h:
            cache = block.attn.kv_cache
            if not isinstance(cache, KVCache):
                raise TypeError("reset_kv_cache() requires set_kv_cache() to have been called first")
            cache.reset_parameters()
            block.attn._log_kv_pending = None

    def set_log_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int | None = None,
        rope_cache_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        B: int = 512,
        recent_size: int = 1024,
        prefill_block: int = 256,
        second_order_scale: float = 1.0,
        importance_pooling: bool = False,
        importance_pooling_lambda: float = 1.0,
        importance_pooling_temperature: float = 1.0,
        semantic_clusters: bool = False,
        cluster_k_max: int = 1,
        cluster_lambda_rel: float = 1.0,
        seg_eta: float = 1.0,
        seg_g0: float = 2048.0,
        seg_gap_max: float | None = None,
        seg_block_level: int = 0,
        seg_forget: float = 0.5,
        semantic_s_h: torch.Tensor | float | None = None,
        semantic_flush_granularity: int = 2,
        semantic_cluster_chunk_size: int = 0,
        semantic_capacity_beta: float = 0.0,
        semantic_capacity_hard_cap_mult: float = 0.0,
        semantic_legacy_route: bool = False,
        semantic_unified_route: bool = False,
        semantic_merge_passes: int = 1,
        semantic_anchor_mode: str = "multi",
        semantic_pack_backend: str = "auto",
        semantic_centroid_backend: str = "sequential",
        semantic_replay_updates: bool = False,
        alpha_exact_tokens: int = 0,
        alpha_span_max_tokens: int = 64,
        beta_novelty: bool = False,
        beta_adaptive_merge: bool = False,
    ) -> None:
        """Initialize log-structured KV caches for all attention layers.

        Memory: O(B * log(N)) slots instead of standard O(N).

        Each layer gets its own LogStructuredKVCache instance.

        Args:
            batch_size: Batch size for generation
            max_seq_length: Maximum sequence length to support
            rope_cache_length: RoPE cache length (auto-detected if None)
            device: Target device
            dtype: Target dtype
            B: Number of memory slots per level (default 512)
            recent_size: Sliding window size (default 1024, recent tokens kept exact)
            prefill_block: Vectorized prefill block size. 2 reproduces the strict
                per-2-token streaming semantics; larger values batch prefill
                attention with a deviation bounded by the block size (the cache
                state trajectory is exact either way).
            second_order_scale: Coupled scale for the persisted Sigma/Gamma
                corrections. 0.0 reproduces the old first-order LogKV path;
                CPT can warm this from 0.0 to 1.0.
            importance_pooling: If True, level merges are weighted by a
                heuristic per-token importance mass (post-RoPE key L2 norm)
                instead of a uniform mean. Independent of the token-count
                weight that drives the ``log(w)`` mass bias. See
                ``LogStructuredKVCache``.
            importance_pooling_lambda: only consulted when
                ``importance_pooling`` is True. Blends the importance share
                toward the uniform/count share (1.0 = pure importance,
                0.0 = uniform). See ``LogStructuredKVCache``.
            importance_pooling_temperature: only consulted when
                ``importance_pooling`` is True. Exponent reshaping the raw
                importance heuristic before normalization (1.0 = unchanged,
                < 1.0 dampens outlier tokens). Orthogonal to
                ``importance_pooling_lambda``. See ``LogStructuredKVCache``.
        """
        self._alpha_enabled = bool(alpha_exact_tokens)
        if alpha_exact_tokens and second_order_scale != 0.:
            raise ValueError("AlphaLogKV requires second_order_scale=0")
        if rope_cache_length is None:
            rope_cache_length = self.rope_cache_length()
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        if dtype is None:
            # Default to the parameter dtype, not the process default: a bf16
            # model with fp32 cache buffers would fail on the first cat/matmul.
            dtype = next(self.parameters()).dtype
        if semantic_clusters:
            if self.config.rope_interleave:
                raise ValueError("semantic LogKV requires rope_interleave=False")
            if self.config.latent_attention is not None:
                raise ValueError("semantic LogKV does not support MultiheadLatentAttention")
            if importance_pooling:
                raise ValueError("semantic LogKV rejects importance_pooling=True")
            if torch.is_tensor(semantic_s_h) and semantic_s_h.dim() == 2 and semantic_s_h.size(0) != self.config.n_layer:
                raise ValueError(
                    f"semantic_s_h with 2 dims must be (n_layer, n_query_groups), got {tuple(semantic_s_h.shape)}"
                )

        for block_idx, block in enumerate(self.transformer.h):
            cos_cache = self.cos[..., self.config.rope_indices[block_idx]] if semantic_clusters and self.config.rope_indices is not None else self.cos
            sin_cache = self.sin[..., self.config.rope_indices[block_idx]] if semantic_clusters and self.config.rope_indices is not None else self.sin
            block_s_h = semantic_s_h[block_idx] if torch.is_tensor(semantic_s_h) and semantic_s_h.dim() == 2 else semantic_s_h
            block.attn.kv_cache = block.attn.build_log_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype,
                B=B, recent_size=recent_size,
                allocate_second_order=second_order_scale != 0.0,
                importance_pooling=importance_pooling,
                importance_pooling_lambda=importance_pooling_lambda,
                importance_pooling_temperature=importance_pooling_temperature,
                semantic_clusters=semantic_clusters,
                cluster_k_max=cluster_k_max,
                cluster_lambda_rel=cluster_lambda_rel,
                seg_eta=seg_eta,
                seg_g0=seg_g0,
                seg_gap_max=seg_gap_max,
                seg_block_level=seg_block_level,
                seg_forget=seg_forget,
                semantic_s_h=block_s_h,
                semantic_flush_granularity=semantic_flush_granularity,
                semantic_cluster_chunk_size=semantic_cluster_chunk_size,
                semantic_capacity_beta=semantic_capacity_beta,
                semantic_capacity_hard_cap_mult=semantic_capacity_hard_cap_mult,
                semantic_legacy_route=semantic_legacy_route,
                semantic_unified_route=semantic_unified_route,
                semantic_merge_passes=semantic_merge_passes,
                semantic_anchor_mode=semantic_anchor_mode,
                semantic_pack_backend=semantic_pack_backend,
                semantic_centroid_backend=semantic_centroid_backend,
                semantic_replay_updates=semantic_replay_updates,
                alpha_exact_tokens=alpha_exact_tokens,
                alpha_span_max_tokens=alpha_span_max_tokens,
                beta_novelty=beta_novelty,
                beta_adaptive_merge=beta_adaptive_merge,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
            )
            block.attn._log_kv_pending = None
            block.attn.log_kv_prefill_block = prefill_block
            block.attn.log_kv_second_order_scale = float(second_order_scale)

        # Drop any pre-existing mask_cache from prior set_kv_cache calls to avoid
        # holding stale O(N^2) bool tensors in GPU memory. With every block on
        # LogKV, GPT.forward sets mask=None and never reads it anyway.
        self.mask_cache = None

    def reset_log_kv_cache(self) -> None:
        """Reset every layer's LogKV cache state in place — no reallocation.

        LogKV buffers are sized by (B, recent_size, max_levels), independent of
        the request length, so evaluation should build them once via
        ``set_log_kv_cache()`` and reset between requests. Rebuilding per
        request re-allocates O(n_layer x (recent + B*logN)) CUDA buffers
        thousands of times, fragmenting the allocator (OOM risk on long runs).
        """
        for block in self.transformer.h:
            cache = block.attn.kv_cache
            if not isinstance(cache, LogStructuredKVCache):
                raise TypeError(
                    "reset_log_kv_cache() requires set_log_kv_cache() to have been called first"
            )
            cache.reset_parameters()
            block.attn._log_kv_pending = None

    def set_log_kv_second_order_scale(self, second_order_scale: float) -> None:
        """Set the coupled Sigma/Gamma correction scale on every LogKV layer."""
        for block in self.transformer.h:
            block.attn.log_kv_second_order_scale = float(second_order_scale)

    def enable_log_kv_training(
        self,
        batch_size: int,
        max_seq_length: int | None = None,
        rope_cache_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        B: int = 512,
        recent_size: int = 1024,
        train_block: int = 2,
        second_order_scale: float = 1.0,
        importance_pooling: bool = False,
        importance_pooling_lambda: float = 1.0,
        importance_pooling_temperature: float = 1.0,
        semantic_clusters: bool = False,
        cluster_k_max: int = 1,
        cluster_lambda_rel: float = 1.0,
        seg_eta: float = 1.0,
        seg_g0: float = 2048.0,
        seg_gap_max: float | None = None,
        seg_block_level: int = 0,
        seg_forget: float = 0.5,
        semantic_s_h: torch.Tensor | float | None = None,
        semantic_flush_granularity: int = 2,
        semantic_cluster_chunk_size: int = 0,
        semantic_capacity_beta: float = 0.0,
        semantic_capacity_hard_cap_mult: float = 0.0,
        semantic_legacy_route: bool = False,
        semantic_unified_route: bool = False,
        semantic_merge_passes: int = 1,
        allocate_second_order: bool = True,
        semantic_anchor_mode: str = "multi",
        semantic_pack_backend: str = "auto",
        semantic_centroid_backend: str = "sequential",
        semantic_replay_updates: bool = False,
        alpha_exact_tokens: int = 0,
        alpha_span_max_tokens: int = 64,
        beta_novelty: bool = False,
        beta_adaptive_merge: bool = False,
    ) -> None:
        """Attach a LogStructuredKVCache to every attention layer and switch
        each layer into ``training_log_kv`` mode.

        Unlike ``set_log_kv_cache`` (inference), this does NOT build a mask
        cache (training builds a per-chunk mask) and flips ``training_log_kv``
        on so that ``CausalSelfAttention.forward`` routes through
        ``_log_kv_training_forward`` when ``input_pos is None``.

        ``train_block`` controls low-memory replay granularity. 2 reproduces
        strict 2-token streaming; larger values trade bounded in-block
        exact-token visibility for far fewer tiny matmul launches and
        O(train_block * slots) replay memory.

        ``second_order_scale`` is a single coupled gate for the score-side
        Sigma correction and value-side Gamma correction. Use 0.0 for the old
        first-order objective and warm it to 1.0 during CPT.

        Set ``allocate_second_order=False`` only for a permanently zero gate.
        Warmup from zero to a nonzero target must leave storage enabled.

        ``importance_pooling`` is a deterministic function of k (no new
        learnable params), computed identically in training and inference, so
        turning it on here keeps train/eval consistent automatically --
        ``importance_pooling_lambda``/``importance_pooling_temperature`` are
        likewise plain scalars (no new learnable params) -- see
        ``set_log_kv_cache``.
        """
        if semantic_clusters:
            if self.config.rope_interleave:
                raise ValueError("semantic LogKV requires rope_interleave=False")
            if self.config.latent_attention is not None:
                raise ValueError("semantic LogKV does not support MultiheadLatentAttention")
            if importance_pooling:
                raise ValueError("semantic LogKV rejects importance_pooling=True")
            if torch.is_tensor(semantic_s_h) and semantic_s_h.dim() == 2 and semantic_s_h.size(0) != self.config.n_layer:
                raise ValueError(
                    f"semantic_s_h with 2 dims must be (n_layer, n_query_groups), got {tuple(semantic_s_h.shape)}"
                )
        self._alpha_enabled = bool(alpha_exact_tokens)
        if alpha_exact_tokens and second_order_scale != 0.:
            raise ValueError("AlphaLogKV requires second_order_scale=0")
        if rope_cache_length is None:
            rope_cache_length = self.rope_cache_length()
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        if dtype is None:
            # Default to the parameter dtype, not the process default (see
            # set_log_kv_cache).
            dtype = next(self.parameters()).dtype

        for block_idx, block in enumerate(self.transformer.h):
            cos_cache = self.cos[..., self.config.rope_indices[block_idx]] if semantic_clusters and self.config.rope_indices is not None else self.cos
            sin_cache = self.sin[..., self.config.rope_indices[block_idx]] if semantic_clusters and self.config.rope_indices is not None else self.sin
            block_s_h = semantic_s_h[block_idx] if torch.is_tensor(semantic_s_h) and semantic_s_h.dim() == 2 else semantic_s_h
            block.attn.kv_cache = block.attn.build_log_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype,
                B=B, recent_size=recent_size,
                allocate_second_order=allocate_second_order,
                importance_pooling=importance_pooling,
                importance_pooling_lambda=importance_pooling_lambda,
                importance_pooling_temperature=importance_pooling_temperature,
                semantic_clusters=semantic_clusters,
                cluster_k_max=cluster_k_max,
                cluster_lambda_rel=cluster_lambda_rel,
                seg_eta=seg_eta,
                seg_g0=seg_g0,
                seg_gap_max=seg_gap_max,
                seg_block_level=seg_block_level,
                seg_forget=seg_forget,
                semantic_s_h=block_s_h,
                semantic_flush_granularity=semantic_flush_granularity,
                semantic_cluster_chunk_size=semantic_cluster_chunk_size,
                semantic_capacity_beta=semantic_capacity_beta,
                semantic_capacity_hard_cap_mult=semantic_capacity_hard_cap_mult,
                semantic_legacy_route=semantic_legacy_route,
                semantic_unified_route=semantic_unified_route,
                semantic_merge_passes=semantic_merge_passes,
                semantic_anchor_mode=semantic_anchor_mode,
                semantic_pack_backend=semantic_pack_backend,
                semantic_centroid_backend=semantic_centroid_backend,
                semantic_replay_updates=semantic_replay_updates,
                alpha_exact_tokens=alpha_exact_tokens,
                alpha_span_max_tokens=alpha_span_max_tokens,
                beta_novelty=beta_novelty,
                beta_adaptive_merge=beta_adaptive_merge,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
            )
            block.attn.training_log_kv = True
            block.attn._log_kv_pending = None
            block.attn.log_kv_train_block = train_block
            block.attn.log_kv_second_order_scale = float(second_order_scale)

    def disable_log_kv_training(self) -> None:
        """Turn off logKV training mode and drop the caches."""
        self._alpha_enabled = False
        for block in self.transformer.h:
            block.attn.training_log_kv = False
            block.attn.kv_cache = None
            block.attn._log_kv_pending = None


class Block(nn.Module):
    def __init__(
        self,
        config: Config,
        block_idx: int,
    ) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = nn.Identity() if not config.norm_1 else config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = (
            CausalSelfAttention(config, block_idx)
            if not config.latent_attention
            else MultiheadLatentAttention(config, block_idx)
        )
        self.post_attention_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_attention_norm else nn.Identity()
        )
        self.norm_2 = (
            nn.Identity()
            if not config.norm_2
            else (None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps))
        )
        self.mlp = config.mlp_class(config)
        if config.first_k_dense_replace is not None and block_idx < config.first_k_dense_replace:
            self.mlp = LLaMAMLP(config)
        self.post_mlp_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_mlp_norm else nn.Identity()
        )

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
        input_pos_maxp1: int | None = None,
        span_ends: list[list[bool]] | None = None,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ──────────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓                   ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───────►    norm_2
           │  ↓                     │  ↓                   ↓
           │  attn                  │  attn                MLP
           │  ↓                     │  ↓                   ↓
           |  post_attn_norm        |  post_attn_norm      post_mlp_norm
           |  ↓                     |  ↓                   ↓
        ┌─ └► +                     └► + ◄─────────────────┘
        |     ↓
        │     norm_2
        │     ↓
        │     MLP
        │     ↓
        |     post_mlp_norm
        |     ↓
        └───► +
        """

        x_normed = self.norm_1(x)
        attention_output = self.attn(x_normed, cos, sin, mask, input_pos, input_pos_maxp1,
                                     **({"span_ends": span_ends} if span_ends is not None else {}))
        attention_output = self.post_attention_norm(attention_output)

        if self.config.parallel_residual:
            if not self.config.shared_attention_norm:
                x_normed = self.norm_2(x)
            x = attention_output + x
        else:
            x = attention_output + x
            x_normed = self.norm_2(x)

        return self.post_mlp_norm(self.mlp(x_normed)) + x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        # key, query and value projections for all heads, but in a batch
        self.qkv = nn.Linear(
            config.n_embd,
            (config.n_head + 2 * config.n_query_groups) * config.head_size,  # support for grouped/multi queries
            bias=config.bias or config.attn_bias,
        )
        # output projection
        self.proj = nn.Linear(config.head_size * config.n_head, config.n_embd, bias=config.bias)
        # disabled by default
        self.kv_cache: KVCache | LogStructuredKVCache | None = None
        # When True (training only), simulate the logKV streaming compaction
        # over the training sequence so the model learns to read compressed KV.
        self.training_log_kv: bool = False
        # LogKV inference: carry one trailing token across calls so commits stay
        # aligned to the training chunk size. Holds (k, v) — full post-RoPE key.
        self._log_kv_pending: tuple | None = None
        # LogKV inference prefill block size (see _log_kv_training_forward):
        # queries in a block share the slot state frozen at block start. 2 =
        # strict per-2-token streaming semantics; larger = fewer, larger
        # kernels with a deviation bounded by the block size. Set via
        # GPT.set_log_kv_cache(prefill_block=...).
        self.log_kv_prefill_block: int = 256
        # LogKV training replay block size. 2 is the strict-streaming reference;
        # larger blocks keep memory bounded while reducing Python/kernels at 32K.
        self.log_kv_train_block: int = 2
        # Coupled scale for Sigma/Gamma second-order LogKV corrections.
        self.log_kv_second_order_scale: float = 1.0
        # Optional Stage-0 SemanticLogKV dump recorder. This is deliberately a
        # duck-typed offline hook so the production path does not import the
        # analysis code or allocate anything unless a dump script attaches it.
        self._semantic_s0_recorder: Any | None = None
        self.apply_sliding_window_attention = False
        if config.sliding_window_size is not None and config.sliding_window_indices is not None:
            self.apply_sliding_window_attention = config.sliding_window_indices[block_idx]

        if config.norm_qk:
            norm_q_size = config.n_head * config.head_size if config.norm_qk_type == "olmo2" else config.head_size
            norm_k_size = (
                config.n_query_groups * config.head_size if config.norm_qk_type == "olmo2" else config.head_size
            )
            self.norm_q = config.norm_class(norm_q_size, eps=config.norm_eps)
            self.norm_k = config.norm_class(norm_k_size, eps=config.norm_eps)
        else:
            self.norm_q = self.norm_k = None

        if config.rope_adjustments is not None:
            mscale_all_dim = config.rope_adjustments.get("mscale_all_dim", None)
            scaling_factor = config.rope_adjustments.get("factor", None)
            if mscale_all_dim and scaling_factor:  # YaRN
                self.mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
            else:
                self.mscale = 1.0
        else:
            self.mscale = 1.0

        self.config = config
        self.block_idx = block_idx

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
        input_pos_maxp1: int | None = None,
        span_ends: list[list[bool]] | None = None,
    ) -> torch.Tensor:
        # Notation:
        # - B          | batch size
        # - T          | time-step (sequence length)
        # - C          | model's embeddings size (n_embd)
        # - C*         | attentions's embeddings size
        # - hs         | head size
        # - nh_(q,k,v) | number of heads for query, key and value
        # - n_query_groups = nh_k = nh_v | number of query groups sharing key and value heads
        # alternative notation: num_kv_groups = n_query_groups
        # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
        # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        #   │    │    │    │         │        │                 │
        # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
        # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
        # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
        # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
        # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
        # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
        #         MHA                    GQA                   MQA
        #   n_query_groups=4       n_query_groups=2      n_query_groups=1
        #
        # credit https://arxiv.org/pdf/2305.13245.pdf
        head_size = self.config.head_size
        n_head = self.config.n_head
        n_query_groups = self.config.n_query_groups
        rope_n_elem = self.config.rope_n_elem
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # Perform a single multiplication operation using a combined QKV matrix to calculate `query`, `key`, and `value`
        # instead of individually multiplying the input `x` with the respective weight matrices.
        qkv = self.qkv(x)  # (B, T, 3xC*)

        # Define query, key and value sizes.
        # If grouped/multi query is enabled, these sizes are not equal (see the diagram above).
        query_size = n_head * head_size
        key_size = value_size = n_query_groups * head_size
        # Split qkv into query, key and value matrices.
        q, k, v = qkv.split((query_size, key_size, value_size), dim=-1)  # 3x(B, T, C*)

        if self.config.norm_qk and self.config.norm_qk_type == "olmo2":
            q = self.norm_q(q)
            k = self.norm_k(k)

        # To place the num_heads (nh) dimension right after the batch (B) dimension, the first step is to decouple the
        # embedding size (C) into num_heads (nh) and head_size (hs).

        # The original GQA paper is followed here and the term query groups is used.
        # alternative notation: Query groups are also referred to as KV groups.
        q = q.view(B, T, n_head, head_size)  # (B, T, nh_q, hs)
        k = k.view(B, T, n_query_groups, head_size)  # (B, T, n_query_groups, hs)
        v = v.view(B, T, n_query_groups, head_size)  # (B, T, n_query_groups, hs)

        # The tensors `query`, `key`, and `value` are now accurately structured: within each batch element (B), there are
        # multiple heads (nh), and within each head, there is a sequence of elements (T), each represented by a vector
        # of size `hs`.
        q = q.transpose(1, 2)  # (B, nh_q, T, hs)
        k = k.transpose(1, 2)  # (B, nh_k, T, hs)
        v = v.transpose(1, 2)  # (B, nh_v, T, hs)

        if self.config.norm_qk and self.config.norm_qk_type == "default":
            q = self.norm_q(q)
            k = self.norm_k(k)

        if self._semantic_s0_recorder is not None:
            self._semantic_s0_recorder.record_pre_rope(self.block_idx, k, v)
        k_raw = k

        # Unlike standard positional embeddings rotary embeddings must be applied at every layer.
        # Partial rotary: the first rope_n_elem dims are the position channel (RoPE'd)
        # and the rest are the content channel (no RoPE). LogKV mean-pools the FULL
        # key when merging slots, so the merged position sub-channel becomes the
        # expected rotation over the span ("expected RoPE") — no per-token state.

        if self.config.rope_interleave:
            q_roped = apply_rope_interleave(q[..., :rope_n_elem], cos, sin)
            k_roped = apply_rope_interleave(k[..., :rope_n_elem], cos, sin)
        else:
            q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
            k_roped = apply_rope(k[..., :rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)  # (B, nh_q, T, hs)
        k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)  # (B, nh_k, T, hs)

        # LogKV training mode: simulate streaming compaction (C = t) so that
        # training attention matches inference decode semantics. Each chunk of
        # t tokens attends to [compact prefix (detached) + current chunk (causal)]
        # via slot attention over merged-position entries. Routed through the
        # low-memory Function (graph-free stream + backward replay): the naive
        # per-chunk graph saves O(T/2 x S) tensors per layer and OOMs at 32K.
        if self.training_log_kv and input_pos is None:
            return self._log_kv_train_lowmem_forward(q, k, v, B, T, k_raw=k_raw, span_ends=span_ends)

        # Apply kv-cache during inference.
        if input_pos is not None:
            if not isinstance(self.kv_cache, (KVCache, LogStructuredKVCache)):
                raise TypeError("You need to call `gpt.set_kv_cache()`")

            if isinstance(self.kv_cache, LogStructuredKVCache):
                self._assert_log_kv_input_pos_contiguous(input_pos, T)
                # Inference uses the same streaming chunker for prefill and decode.
                # A trailing single token is kept pending so an odd-length prompt
                # pairs with the first decode token, matching training chunks.
                return self._log_kv_training_forward(
                    q, k, v, B, T, reset_cache=False, defer_last_single=True, k_raw=k_raw, input_pos=input_pos, span_ends=span_ends
                )

            if self.apply_sliding_window_attention:
                return self._sliding_window_forward(q, k, v, input_pos)

            k, v = self.kv_cache(input_pos, k, v)

            if input_pos_maxp1 is not None and not isinstance(self.kv_cache, LogStructuredKVCache):
                # Subselect along sequence dimension (only for standard cache)
                k = k[..., :input_pos_maxp1, :]
                v = v[..., :input_pos_maxp1, :]
            # k, v: (B, nh_k, input_pos_maxp1, hs)
            # If input_pos_maxp1 is None -> max_seq_length

        # Grouped queries: balance the number of heads across all three matrices.
        # NOTE: flash attention requires it in training mode.
        # Multi-query: this step can be skipped since there is only 1 head, allowing us to use broadcasting.
        if n_query_groups != n_head and (input_pos is None or n_query_groups != 1):
            q_per_kv = n_head // n_query_groups
            k = k.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)
            v = v.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)

        if self.apply_sliding_window_attention:
            """
                  Global Window              Sliding window             Sliding window
                  attention mask      +            bias          =      attention mask
            ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐
            │ True False False False │  │ True  True  True True │  │ True  False False False │
            │ True True  False False │  │ True  True  True True │  │ True  True  False False │
            │ True True  True  False │  │ False True  True True │  │ False True  True  False │
            │ True True  True  True  │  │ False False True True │  │ False False True  True  │
            └────────────────────────┘  └───────────────────────┘  └─────────────────────────┘
            """
            if input_pos is None:
                if mask is None:
                    mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
                    mask.masked_fill_(mask.bool(), float("-inf"))
                    mask = mask.view(1, 1, *mask.shape)

                sliding_window_mask = torch.full((T, T), float("-inf"), dtype=q.dtype, device=q.device)
                for i in range(T):
                    window_start = max(0, i - self.config.sliding_window_size + 1)
                    sliding_window_mask[i, window_start : i + 1] = 0.0
                sliding_window_mask = sliding_window_mask.view(1, 1, T, T)
                mask = sliding_window_mask

        # Efficient attention using Flash Attention CUDA kernels.
        # NOTE: efficient implementation is disabled if `mask` is not None or softcapping is enabled.
        # ↓ (B, nh, T, hs) @ (B, nh, T, hs).mT --> (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        y = self.scaled_dot_product_attention(q, k, v, mask)

        # Re-assemble all head outputs side by side.
        y = y.reshape(B, T, head_size * n_head)

        # Output projection.
        return self.proj(y)  # (B, T, C)

    def _sliding_window_forward(self, q, k, v, input_pos):
        """Exact local causal attention, including prefill longer than the ring.

        Read the old ring before committing each block: overwriting it first
        loses keys needed by the block's early queries. Absolute positions mask
        both future tokens and stale ring slots, regardless of ring ordering.
        """
        cache = self.kv_cache
        batch, heads, length, _ = q.shape
        positions = input_pos.expand(batch, -1) if input_pos.ndim == 1 else input_pos
        window = cache.max_cache_len
        if cache.k.dtype != k.dtype:
            cache.k = cache.k.to(k.dtype)
        if cache.v.dtype != v.dtype:
            cache.v = cache.v.to(v.dtype)
        # ponytail: bounded SDPA blocks for correctness first; a local Flash
        # kernel can replace the explicit mask if profiling warrants it.
        chunk = min(1024, window)
        outputs = []
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            pos = positions[:, start:end]
            key_pos = torch.cat((cache.positions[:batch], pos), dim=-1)
            key_pos[:, :window].masked_fill_(key_pos[:, :window] >= pos[:, :1], -1)
            keys = torch.cat((cache.k[:batch], k[:, :, start:end]), dim=2)
            values = torch.cat((cache.v[:batch], v[:, :, start:end]), dim=2)
            distance = pos[:, :, None] - key_pos[:, None, :]
            allowed = (key_pos[:, None, :] >= 0) & (distance >= 0) & (distance < window)
            mask = q.new_zeros(batch, 1, end - start, key_pos.size(-1))
            mask.masked_fill_(~allowed[:, None], float("-inf"))
            if keys.size(1) != heads:
                repeats = heads // keys.size(1)
                keys = keys.repeat_interleave(repeats, dim=1)
                values = values.repeat_interleave(repeats, dim=1)
            outputs.append(self.scaled_dot_product_attention(q[:, :, start:end], keys, values, mask))
            cache(pos, k[:, :, start:end], v[:, :, start:end])
        return self.proj(torch.cat(outputs, dim=1).reshape(batch, length, -1))

    def _assert_log_kv_input_pos_contiguous(self, input_pos: torch.Tensor, T: int) -> None:
        """Validate the append-only LogKV cache contract.

        LogKV merges tokens in arrival order and tracks cache length with
        scalar counters. Until true indexed writes are implemented, inference
        must feed a single shared, contiguous position range.
        """
        assert isinstance(self.kv_cache, LogStructuredKVCache)

        if input_pos.dim() == 2:
            if not torch.equal(input_pos, input_pos[:1].expand_as(input_pos)):
                raise ValueError(
                    "LogKV cache does not support per-sample input_pos yet; all batch rows must share the same "
                    "contiguous positions."
                )
            input_pos = input_pos[0]
        elif input_pos.dim() != 1:
            raise ValueError(f"LogKV cache requires 1-D or shared 2-D input_pos, got shape {tuple(input_pos.shape)}.")

        pending_len = 0 if self._log_kv_pending is None else self._log_kv_pending[0].size(2)
        expected_start = self.kv_cache.token_count + pending_len
        expected = torch.arange(expected_start, expected_start + T, device=input_pos.device, dtype=input_pos.dtype)
        if not torch.equal(input_pos, expected):
            got = input_pos.detach().cpu().tolist()
            raise ValueError(
                "LogKV cache requires append-only contiguous input_pos. "
                f"Expected {expected_start}..{expected_start + T - 1}, got {got}. "
                "Reset the LogKV cache before starting a new sequence, or implement indexed LogKV writes."
            )

    def _log_kv_train_lowmem_forward(
        self,
        q: torch.Tensor,    # (B, n_head, T, hs) post-RoPE
        k: torch.Tensor,    # (B, n_query_groups, T, hs) post-RoPE
        v: torch.Tensor,    # (B, n_query_groups, T, hs)
        B: int,
        T: int,
        k_raw: torch.Tensor | None = None,
        span_ends: list[list[bool]] | None = None,
    ) -> torch.Tensor:
        """Training forward over the logKV stream with O(T + S) memory.

        ``log_kv_train_block=2`` matches the strict 2-token streaming reference.
        Larger blocks freeze the prefix state at block start and apply causal
        exact attention inside the block, matching the inference prefill
        tradeoff: the final cache state is exact, peak replay memory is
        O(train_block * slots), and the number of tiny matmul launches falls
        from T/2 to T/train_block.
        """
        # Explicit raise (not assert): must survive `python -O`.
        if not isinstance(self.kv_cache, LogStructuredKVCache):
            raise TypeError("training_log_kv requires a LogStructuredKVCache")
        cache = self.kv_cache
        cache._convert_dtype(q.dtype)
        # ``cache.second_order`` is deliberately NOT set here: LogKVStreamTrainingAttention
        # derives it from the gate it is handed, in both forward and backward, so
        # that the two passes cannot disagree. Setting it here as well would just
        # be a second source of truth to drift.
        self._log_kv_pending = None  # training never defers a tail token

        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)
        scale = scale * self.mscale * self.mscale

        train_block = max(2, min(int(self.log_kv_train_block), cache.recent_size))
        args = [
            q,
            k,
            v,
            cache,
            scale,
            train_block,
            self.log_kv_second_order_scale,
        ]
        if cache.semantic_clusters:
            if k_raw is None:
                raise ValueError("semantic LogKV training requires pre-RoPE k_raw")
            args.append(k_raw)
            args.append(span_ends)
        y = LogKVStreamTrainingAttention.apply(*args)  # (B, n_head, T, hs)
        y = y.transpose(1, 2).reshape(B, T, self.config.head_size * self.config.n_head)
        return self.proj(y)

    def _log_kv_training_forward(
        self,
        q: torch.Tensor,    # (B, n_head, T, hs) post-RoPE
        k: torch.Tensor,    # (B, n_query_groups, T, hs) post-RoPE
        v: torch.Tensor,    # (B, n_query_groups, T, hs)
        B: int,
        T: int,
        reset_cache: bool = True,
        defer_last_single: bool = False,
        k_raw: torch.Tensor | None = None,
        span_ends: list[list[bool]] | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reference forward simulating the logKV streaming compaction with a
        sliding window.

        Role: inference streaming engine (prefill blocks + pending-token
        decode pairing) and the differentiable REFERENCE implementation for
        the training semantics. Actual training routes through
        ``_log_kv_train_lowmem_forward`` — identical math, O(T + S) memory —
        and the equivalence tests pin the two together.

        Splits the sequence into chunks of 2 tokens. For each chunk the queries
        attend to:
            [compact prefix (detached)
             + sliding window from previous chunks (detached, exact)
             + current chunk exact KV (causal, with gradient)]
        via slot attention over merged-position entries (one logit per slot,
        +log w mass bias — see ``log_kv_slot_attention``). After attending, the
        chunk's KV (detached) is added to the sliding window. When the window
        overflows, the oldest 2 tokens are compacted into the hierarchy.

        Gradient flows only through the current chunk's q/k/v (and thus the qkv
        projection weights); the compacted prefix and sliding window are
        stop-gradient context, keeping activation memory bounded.

        When ``defer_last_single`` is true (LogKV inference), the final
        one-token chunk is attended immediately but not committed. It is stored
        in ``_log_kv_pending`` and paired with the next streamed token, which
        keeps odd-length prefill and subsequent decode on the same 2-token
        boundaries as training over the concatenated sequence.
        """
        # Explicit raise (not assert): this correctness precondition must
        # survive `python -O`, which strips assert statements.
        if not isinstance(self.kv_cache, LogStructuredKVCache):
            raise TypeError("training_log_kv requires a LogStructuredKVCache")
        # Any rotary_percentage is valid here: compaction mean-pools the FULL
        # post-RoPE key, so the rotated slice merges into the expected rotation
        # (Dirichlet-damped as spans widen) and any pass-through content slice
        # merges undamped. rotary_percentage < 1.0 (e.g. 0.25) keeps a
        # position-free content channel whose signal survives compaction at any
        # distance — recommended for adaptation quality — but full RoPE
        # (Qwen3 default 1.0) runs correctly: distant compact slots just fade
        # toward pure mass-bias contributions.
        cache = self.kv_cache
        if reset_cache:
            cache.reset_parameters()
            self._log_kv_pending = None
        # AFTER any reset above: this is also the real decode path
        # (reset_cache=False, same cache reused across every generated token),
        # where the guarded setter raises if the scale changed since the last
        # call on a cache that already holds compacted levels — the caller
        # must reset_parameters() to switch second-order regimes on a live
        # cache. Setting it before the reset would let a genuine change slip
        # through unexamined here but corrupt compaction on the very next carry.
        cache.second_order = self.log_kv_second_order_scale != 0.0

        # Reconcile cache buffer dtype with the activations before any read/write.
        # Inference builds the cache via set_log_kv_cache(), which may allocate
        # buffers at the process-default dtype (fp32) while the model runs in bf16;
        # without this the first get_attention_state()/add_recent() would cat/matmul
        # fp32 buffers with bf16 activations and raise. No-op once dtypes match.
        cache._convert_dtype(q.dtype)
        if cache.semantic_clusters and k_raw is None:
            raise ValueError("semantic LogKV inference requires pre-RoPE k_raw")

        t = 2  # compaction chunk size (fixed: uniform 2:1)
        n_head = self.config.n_head
        head_size = self.config.head_size

        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or head_size)
        scale = scale * self.mscale * self.mscale

        # No content/position split: q and k are the full post-RoPE vectors
        # [R(p)·(pos slice) ; content slice]. Merged slots carry both channels,
        # so a single dot product covers the content AND the position score.
        def _pos_slice(start_i: int, end_i: int) -> torch.Tensor | None:
            if not cache.semantic_clusters or input_pos is None:
                return None
            return input_pos[start_i:end_i].detach() if input_pos.dim() == 1 else input_pos[:, start_i:end_i].detach()

        outputs: list[torch.Tensor] = []
        start = 0

        pending = self._log_kv_pending if defer_last_single else None
        if pending is not None:
            if cache.semantic_clusters:
                pk, pv, pk_raw, ppos = pending[:4]
                pending_ends = pending[4] if len(pending) > 4 else None
            else:
                pk, pv = pending
                pk_raw = pk
                ppos = None
            k_b = k[:, :, :1, :]
            v_b = v[:, :, :1, :]

            # Pending token and current token are exact; the single query sees both.
            y_b = log_kv_chunk_attention(
                cache, q[:, :, :1, :], torch.cat([pk, k_b], dim=2), torch.cat([pv, v_b], dim=2),
                scale, self.log_kv_second_order_scale, causal_tail=0, reuse_prefix=True,
            )
            outputs.append(y_b)

            pos_b = _pos_slice(0, 1)
            cache.add_recent(
                torch.cat([pk, k_b.detach()], dim=2),
                torch.cat([pv, v_b.detach()], dim=2),
                k_raw=torch.cat([pk_raw, k_raw[:, :, :1, :].detach()], dim=2) if cache.semantic_clusters else None,
                input_pos=torch.cat([ppos, pos_b], dim=-1) if cache.semantic_clusters and ppos is not None and pos_b is not None else None,
                span_ends=[a + b[:1] for a, b in zip(pending_ends, span_ends)] if span_ends is not None else None,
            )
            self._log_kv_pending = None
            start = 1

        # ---- Vectorized block prefill (inference only) ----
        # Process the stream in blocks, freezing the slot state at block start:
        # in-block queries attend to [frozen compact+recent slots (fully
        # visible) + causal exact in-block tokens]. While no window flush would
        # occur inside a block this IS the exact 2-token streaming semantics
        # (each in-block query's true state equals the frozen one). Once
        # compaction is active, a query at in-block offset j instead sees up to
        # j exact tokens that strict streaming would already have compacted — a
        # bounded, strictly information-richer deviation (effective exact
        # window stretched by less than one block out of recent_size). The
        # cache state trajectory stays exact regardless: add_recent() below
        # flushes/compacts exactly as streaming would, so decode after prefill
        # sees bit-identical cache contents. A block size of 2 reproduces
        # strict streaming exactly (log_kv_prefill_block=2 for A/B checks).
        # Training keeps the per-chunk loop below for its per-chunk
        # stop-gradient structure.
        if defer_last_single:
            blk_size = max(2, min(self.log_kv_prefill_block, cache.recent_size))
            while start < T:
                block_end = min(T, start + blk_size)
                blk = block_end - start
                # Defer a lone tail token so it pairs with the next streamed
                # token (keeps the 2-token commit alignment with training).
                defer_tail = block_end == T and blk % 2 == 1
                commit_end = block_end - 1 if defer_tail else block_end

                # Diagnostic branch (score/value oracle grid; see log_kv_diag).
                # Inert unless a diag_mode(...) context set LOG_KV_DIAG.enabled.
                # Gated to fresh-prefill blocks whose frozen slots cover exactly
                # k[:, :, :start] (token_count == start, no pending/decode), the
                # precondition diag_block_attention relies on. Production numerics
                # and the cache state trajectory are unchanged.
                if LOG_KV_DIAG.enabled and start > 0 and int(cache.token_count) == start:
                    state = cache.get_attention_state(with_stats=cache.second_order)
                    y_blk = diag_block_attention(
                        q[:, :, start:block_end, :],
                        k[:, :, :start, :], v[:, :, :start, :],
                        state.slot_k, state.slot_v, state.slot_w,
                        k[:, :, start:block_end, :], v[:, :, start:block_end, :],
                        scale=scale, lam=1.0, layer=self.block_idx,
                        q_offset=start, seq_len=T,
                        slot_valid=state.slot_valid,
                        M_s=state.M_s,
                        slot_sigma_u=state.slot_sigma_u,
                        slot_sigma2=state.slot_sigma2,
                        slot_gamma_a=state.slot_gamma_a,
                        slot_gamma_b=state.slot_gamma_b,
                        slot_gamma=state.slot_gamma,
                        second_order_scale=self.log_kv_second_order_scale,
                    )
                else:
                    y_blk = log_kv_chunk_attention(
                        cache, q[:, :, start:block_end, :], k[:, :, start:block_end, :],
                        v[:, :, start:block_end, :], scale, self.log_kv_second_order_scale,
                        reuse_prefix=True,
                    )
                outputs.append(y_blk)

                if commit_end > start:
                    cache.add_recent(
                        k[:, :, start:commit_end, :].detach(),
                        v[:, :, start:commit_end, :].detach(),
                        k_raw=k_raw[:, :, start:commit_end, :].detach() if cache.semantic_clusters else None,
                        input_pos=_pos_slice(start, commit_end),
                        span_ends=[row[start:commit_end] for row in span_ends] if span_ends is not None else None,
                    )
                if defer_tail:
                    if cache.semantic_clusters:
                        self._log_kv_pending = (
                            k[:, :, commit_end:block_end, :].detach(),
                            v[:, :, commit_end:block_end, :].detach(),
                            k_raw[:, :, commit_end:block_end, :].detach(),
                            _pos_slice(commit_end, block_end),
                            [row[commit_end:block_end] for row in span_ends] if span_ends is not None else None,
                        )
                    else:
                        self._log_kv_pending = (
                            k[:, :, commit_end:block_end, :].detach(),
                            v[:, :, commit_end:block_end, :].detach(),
                        )
                start = block_end

        while start < T:
            end = min(start + t, T)
            actual_t = end - start
            defer_chunk = defer_last_single and actual_t == 1 and end == T

            k_b = k[:, :, start:end, :]  # (B, n_groups, actual_t, k_dim)
            v_b = v[:, :, start:end, :]  # (B, n_groups, actual_t, v_dim)

            y_b = log_kv_chunk_attention(
                cache, q[:, :, start:end, :], k_b, v_b,
                scale, self.log_kv_second_order_scale, reuse_prefix=not self.training,
            )
            outputs.append(y_b)

            # Add current chunk (detached) to the sliding window.
            # When the window overflows, oldest t tokens are compacted into hierarchy.
            if defer_chunk:
                if cache.semantic_clusters:
                    self._log_kv_pending = (k_b.detach(), v_b.detach(), k_raw[:, :, start:end, :].detach(), _pos_slice(start, end),
                                            [row[start:end] for row in span_ends] if span_ends is not None else None)
                else:
                    self._log_kv_pending = (k_b.detach(), v_b.detach())
            else:
                cache.add_recent(
                    k_b.detach(),
                    v_b.detach(),
                    k_raw=k_raw[:, :, start:end, :].detach() if cache.semantic_clusters else None,
                    input_pos=_pos_slice(start, end),
                    span_ends=[row[start:end] for row in span_ends] if span_ends is not None else None,
                )

            start = end

        y = torch.cat(outputs, dim=2)  # (B, n_head, T, v_dim)
        y = y.transpose(1, 2).reshape(B, T, head_size * n_head)
        return self.proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)
        scale = scale * self.mscale * self.mscale

        # with softcapping we cannot use SDPA
        if self.config.attention_logit_softcapping is not None:
            scores = q @ k.mT * scale
            scores = do_softcapping(scores, self.config.attention_logit_softcapping)
            if mask is None:
                mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            scores = scores + mask
            scores = F.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            y = scores @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
            )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "KVCache":
        if self.apply_sliding_window_attention and self.config.sliding_window_size is not None:
            effective_cache_size = min(max_seq_length, self.config.sliding_window_size)
        else:
            effective_cache_size = max_seq_length

        v_shape = (batch_size, self.config.n_query_groups, effective_cache_size, self.config.head_size)

        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError(
                    "Please pass the `rope_cache_length` parameter. "
                    "Use `rope_cache_length=model.rope_cache_length()` to extract it automatically."
                )
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                self.config.n_query_groups,
                effective_cache_size,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )

        return KVCache(
            k_shape,
            v_shape,
            device=device,
            dtype=dtype,
            is_sliding_window=self.apply_sliding_window_attention,
            sliding_window_size=self.config.sliding_window_size if self.apply_sliding_window_attention else None,
        )

    def build_log_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        B: int = 512,
        recent_size: int = 1024,
        importance_pooling: bool = False,
        importance_pooling_lambda: float = 1.0,
        importance_pooling_temperature: float = 1.0,
        semantic_clusters: bool = False,
        cluster_k_max: int = 1,
        cluster_lambda_rel: float = 1.0,
        seg_eta: float = 1.0,
        seg_g0: float = 2048.0,
        seg_gap_max: float | None = None,
        seg_block_level: int = 0,
        seg_forget: float = 0.5,
        semantic_s_h: torch.Tensor | float | None = None,
        semantic_flush_granularity: int = 2,
        semantic_cluster_chunk_size: int = 0,
        semantic_capacity_beta: float = 0.0,
        semantic_capacity_hard_cap_mult: float = 0.0,
        semantic_legacy_route: bool = False,
        semantic_unified_route: bool = False,
        semantic_merge_passes: int = 1,
        cos_cache: torch.Tensor | None = None,
        sin_cache: torch.Tensor | None = None,
        allocate_second_order: bool = True,
        semantic_anchor_mode: str = "multi",
        semantic_pack_backend: str = "auto",
        semantic_centroid_backend: str = "sequential",
        semantic_replay_updates: bool = False,
        alpha_exact_tokens: int = 0,
        alpha_span_max_tokens: int = 64,
        beta_novelty: bool = False,
        beta_adaptive_merge: bool = False,
    ) -> "LogStructuredKVCache":
        """Build a log-structured KV cache with strict O(B * log(N)) memory.

        Architecture (uniform 2:1 compaction at every level):
            Buffer:           recent_size raw tokens, no compression
            Level 0 (write):  B entries, each = 2 tokens merged
            Level 1+ (carry): B entries, each = 2 entries from prev level merged

        Slots store the FULL post-RoPE key (position sub-channel included);
        merging mean-pools it, so a slot's position embedding is the expected
        rotation over its span. No per-token buffer of any kind is allocated.
        """
        rope_n_elem = self.config.rope_n_elem
        # k_dim is the full post-RoPE key width: the RoPE'd slice AFTER
        # apply_rope, which is rope_cache_length (cos.size(1)) wide — NOT
        # rope_n_elem: for rope_n_elem == 1 the RoPE cache is widened to 2 dims
        # (HF-compat, see build_rope_cache) and apply_rope broadcasts the roped
        # slice to that width, so k/q enter attention with head_size + 1 dims.
        # Mirrors the `rope_cache_length + head_size - rope_n_elem` k-shape in
        # build_kv_cache.
        if rope_cache_length is None:
            rope_cache_length = 2 if rope_n_elem == 1 else rope_n_elem
        k_dim = rope_cache_length + self.config.head_size - rope_n_elem
        storage_k_dim = self.config.head_size if semantic_clusters else k_dim

        k_shape = (
            batch_size,
            self.config.n_query_groups,
            max_seq_length,
            storage_k_dim,
        )
        v_shape = (
            batch_size,
            self.config.n_query_groups,
            max_seq_length,
            self.config.head_size,
        )

        return LogStructuredKVCache(
            k_shape, v_shape,
            B=B, recent_size=recent_size,
            device=device, dtype=dtype,
            importance_pooling=importance_pooling,
            importance_pooling_lambda=importance_pooling_lambda,
            importance_pooling_temperature=importance_pooling_temperature,
            semantic_clusters=semantic_clusters,
            cluster_k_max=cluster_k_max,
            cluster_lambda_rel=cluster_lambda_rel,
            seg_eta=seg_eta,
            seg_g0=seg_g0,
            seg_gap_max=seg_gap_max,
            seg_block_level=seg_block_level,
            seg_forget=seg_forget,
            semantic_s_h=semantic_s_h,
            semantic_flush_granularity=semantic_flush_granularity,
            semantic_cluster_chunk_size=semantic_cluster_chunk_size,
            semantic_capacity_beta=semantic_capacity_beta,
            semantic_capacity_hard_cap_mult=semantic_capacity_hard_cap_mult,
            semantic_legacy_route=semantic_legacy_route,
            semantic_unified_route=semantic_unified_route,
            semantic_merge_passes=semantic_merge_passes,
            semantic_anchor_mode=semantic_anchor_mode,
            semantic_pack_backend=semantic_pack_backend,
            semantic_centroid_backend=semantic_centroid_backend,
            semantic_replay_updates=semantic_replay_updates,
            alpha_exact_tokens=alpha_exact_tokens,
            alpha_span_max_tokens=alpha_span_max_tokens,
            beta_novelty=beta_novelty,
            beta_adaptive_merge=beta_adaptive_merge,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rope_n_elem=rope_n_elem,
            allocate_second_order=allocate_second_order,
        )

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with legacy checkpoints."""

        for attr in ("weight", "bias"):
            legacy_key = f"{prefix}attn.{attr}"
            current_key = f"{prefix}qkv.{attr}"
            if legacy_key in state_dict:
                state_dict[current_key] = qkv_reassemble(state_dict.pop(legacy_key), self.config)

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class MultiheadLatentAttention(nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()

        self.q_a_proj = nn.Linear(config.n_embd, config.q_lora_rank, bias=config.attn_bias)
        self.q_a_norm = RMSNorm(config.q_lora_rank, eps=config.norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, config.n_head * config.qk_head_dim, bias=config.bias)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.n_embd, config.kv_lora_rank + config.qk_rope_head_dim, bias=config.attn_bias
        )
        self.kv_a_norm = RMSNorm(config.kv_lora_rank, eps=config.norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.n_query_groups * (config.qk_nope_head_dim + config.v_head_dim),
            bias=config.bias,
        )

        # output projection
        self.proj = nn.Linear(config.n_head * config.v_head_dim, config.n_embd, bias=config.bias)
        # disabled by default
        self.kv_cache: KVCache | None = None

        if config.rope_adjustments is not None:
            mscale_all_dim = config.rope_adjustments.get("mscale_all_dim", None)
            scaling_factor = config.rope_adjustments.get("factor", None)
            if mscale_all_dim and scaling_factor:  # YaRN
                self.mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
            else:
                self.mscale = 1.0
        else:
            self.mscale = 1.0

        self.config = config
        self.block_idx = block_idx

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
        input_pos_maxp1: int | None = None,
    ) -> torch.Tensor:
        # Notation:
        # - B          | batch size
        # - T          | time-step (sequence length)
        # - C          | model's embeddings size (n_embd)
        # - C*         | attentions's embeddings size
        # - hs         | head size
        # - nh_(q,k,v) | number of heads for query, key and value
        # - n_query_groups = nh_k = nh_v | number of query groups sharing key and value heads
        # alternative notation: num_kv_groups = n_query_groups
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))  # (B, T, n_head * qk_head_dim)
        q = q.view(B, T, -1, self.config.qk_head_dim)  # (B, T, n_head, qk_head_dim)
        q = q.transpose(1, 2)  # (B, n_head, T, qk_head_dim)
        q_pass, q_rot = torch.split(q, [self.config.qk_nope_head_dim, self.config.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)  # (B, T, kv_lora_rank + qk_rope_head_dim)
        k_pass, k_rot = torch.split(compressed_kv, [self.config.kv_lora_rank, self.config.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_norm(k_pass))
        k_pass = k_pass.view(B, T, self.config.n_query_groups, -1)
        k_pass = k_pass.transpose(1, 2)

        k_pass, v = torch.split(k_pass, [self.config.qk_nope_head_dim, self.config.v_head_dim], dim=-1)
        k_rot = k_rot.view(B, 1, T, self.config.qk_rope_head_dim)  # (B, 1, T, qk_rope_head_dim)

        # Unlike standard positional embeddings rotary embeddings must be applied at every layer.
        if self.config.rope_interleave:
            q_roped = apply_rope_interleave(q_rot, cos, sin)
            k_roped = apply_rope_interleave(k_rot, cos, sin)
        else:
            q_roped = apply_rope(q_rot, cos, sin)
            k_roped = apply_rope(k_rot, cos, sin)
        k_roped = k_roped.expand(*k_pass.shape[:-1], -1)  # (B, n_head, T, qk_rope_head_dim)

        q = torch.cat((q_pass, q_roped), dim=-1)
        k = torch.cat((k_pass, k_roped), dim=-1)

        # Apply kv-cache during inference.
        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)
            if input_pos_maxp1 is not None:
                # Subselect along sequence dimension
                k = k[..., :input_pos_maxp1, :]
                v = v[..., :input_pos_maxp1, :]
            # k, v: (B, nh_k, input_pos_maxp1, hs)
            # If input_pos_maxp1 is None -> max_seq_length

        # Grouped queries: balance the number of heads across all three matrices.
        # NOTE: flash attention requires it in training mode.
        # Multi-query: this step can be skipped since there is only 1 head, allowing us to use broadcasting.
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            q_per_kv = self.config.n_head // self.config.n_query_groups
            k = k.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)
            v = v.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)

        # Efficient attention using Flash Attention CUDA kernels.
        # NOTE: efficient implementation is disabled if `mask` is not None or softcapping is enabled.
        # ↓ (B, nh, T, hs) @ (B, nh, T, hs).mT --> (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        y = self.scaled_dot_product_attention(q, k, v, mask)

        # Re-assemble all head outputs side by side.
        y = y.reshape(B, T, self.config.n_head * self.config.v_head_dim)

        # Output projection.
        return self.proj(y)  # (B, T, C)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.qk_head_dim)
        scale = scale * self.mscale * self.mscale

        # with softcapping we cannot use SDPA
        if self.config.attention_logit_softcapping is not None:
            scores = q @ k.mT * scale
            scores = do_softcapping(scores, self.config.attention_logit_softcapping)
            if mask is None:
                mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            scores = scores + mask
            scores = F.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            y = scores @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
            )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "KVCache":
        v_shape = (batch_size, self.config.n_head, max_seq_length, self.config.v_head_dim)
        k_shape = (batch_size, self.config.n_head, max_seq_length, self.config.qk_head_dim)

        if rope_cache_length is not None:
            print("Warning: `rope_cache_length` has no effect on MultiheadLatentAttention!")
        if self.config.rotary_percentage != 1.0:
            print("Warning: `rotary_percentage` has no effect on MultiheadLatentAttention!")

        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config, intermediate_size: int | None = None) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.fc = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config, intermediate_size: int | None = None) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.fc_1 = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = F.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = F.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
        return self.proj(x)


class LLaMAMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = (
            nn.Linear(config.n_embd, config.n_expert, bias=False)
            if not config.n_expert_groups
            else GroupedTopkRouter(config)
        )
        self.experts = nn.ModuleList(
            LLaMAMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_expert)
        )
        if config.n_shared_expert:
            self.shared_experts = LLaMAMLP(
                config, intermediate_size=config.moe_intermediate_size * config.n_shared_expert
            )
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        residual_x = x.clone()
        x = x.view(-1, C)  # (B*T, C)
        if not self.config.n_expert_groups:
            router = self.gate(x)  # (B*T, n_expert)
            probs, indices = torch.topk(router, self.config.n_expert_per_token)  # (B*T, n_expert_per_token)
            probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        else:
            probs, indices = self.gate(x)
        if self.config.routed_scaling_factor != 1.0:
            probs = probs * self.config.routed_scaling_factor
        masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])

        y = y.view(B, T, C)
        if self.config.n_shared_expert:
            y = y + self.shared_experts(residual_x)
        return y


class GroupedTopkRouter(nn.Module):
    """
    Derived from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py.
    DeepseekV3TopkRouter class.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.n_expert, config.n_embd))
        self.register_buffer("e_score_correction_bias", torch.zeros(config.n_expert))

    @torch.no_grad()
    def get_topk_indices(self, scores: torch.Tensor) -> torch.Tensor:
        scores_for_choice = scores.view(-1, self.config.n_expert) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.config.n_expert_groups, self.config.n_expert // self.config.n_expert_groups)
            .topk(self.config.n_topk_scores_per_group, dim=-1)[0]  # Top k scores for each group
            .sum(dim=-1)
        )

        group_idx = torch.topk(group_scores, k=self.config.n_topk_groups, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.config.n_expert_groups, self.config.n_expert // self.config.n_expert_groups)
            .reshape(-1, self.config.n_expert)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.config.n_expert_per_token, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = F.linear(x.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.config.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        return topk_weights, topk_indices


# ROPE: YaRN (Yet another RoPE extensioN) scaling function for extended context
def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: torch.device | None = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: dict | None = None,
    rope_local_base_freq: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced Transformer with Rotary Position Embedding.

    Args:
        seq_len (int): Sequence length.
        n_elem (int): Number of elements (head dimension).
        device (torch.device, optional): Device for tensor allocations.
        base (int, optional): Base for computing inverse frequencies.
        condense_ratio (int, optional): Ratio to condense the position indices.
        extra_config (dict, optional): Configuration parameters for frequency adjustments (used by Llama 3.1 and 3.2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine caches for RoPE.
            Shapes are `(seq_len, n_elem)`.
    """

    # Compute the inverse frequencies theta
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Initialize attention scaling factor (modified for YaRN)
    attention_scaling = 1.0

    if extra_config is not None:
        factor = extra_config["factor"]
        # Check YaRN first (has beta_fast/beta_slow)
        if "beta_fast" in extra_config or "beta_slow" in extra_config:
            # YaRN-style RoPE scaling
            beta_fast = extra_config["beta_fast"]
            beta_slow = extra_config["beta_slow"]
            original_max_seq_len = extra_config["original_max_seq_len"]

            # Calculate attention scaling factor based on mscale and mscale_all_dim
            mscale = extra_config.get("mscale")
            mscale_all_dim = extra_config.get("mscale_all_dim")
            if mscale and mscale_all_dim:
                attention_scaling = yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)
            elif mscale_all_dim:
                attention_scaling = yarn_get_mscale(factor, mscale_all_dim)
            elif mscale:
                attention_scaling = yarn_get_mscale(factor, mscale)
            # else: attention_scaling remains 1.0

            # Create two frequency sets: extrapolation (unscaled) and interpolation (scaled)
            pos_freqs = base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem)
            theta_extrapolation = 1.0 / pos_freqs
            theta_interpolation = 1.0 / (factor * pos_freqs)

            # Find correction range based on rotation counts
            # Inverse dimension formula to find dimension based on number of rotations
            def find_correction_dim(num_rotations, dim, base_val, max_pos):
                return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base_val))

            low_dim = find_correction_dim(beta_fast, n_elem, base, original_max_seq_len)
            high_dim = find_correction_dim(beta_slow, n_elem, base, original_max_seq_len)

            # Apply truncation if specified
            if extra_config.get("truncate", True):
                low_dim = math.floor(low_dim)
                high_dim = math.ceil(high_dim)

            low_dim = max(low_dim, 0)
            high_dim = min(high_dim, n_elem // 2 - 1)

            # Create linear ramp factor for blending
            dim_range = torch.arange(n_elem // 2, device=device, dtype=torch.float32)
            if low_dim == high_dim:
                high_dim += 0.001  # Prevent singularity

            linear_func = (dim_range - low_dim) / (high_dim - low_dim)
            ramp_func = torch.clamp(linear_func, 0.0, 1.0)

            # Blend extrapolation and interpolation frequencies
            # ramp_func = 0 -> use interpolation (scaled), ramp_func = 1 -> use extrapolation (unscaled)
            theta_extrapolation_factor = ramp_func
            theta = (
                theta_interpolation * (1 - theta_extrapolation_factor)
                + theta_extrapolation * theta_extrapolation_factor
            )
        elif "original_max_seq_len" in extra_config:
            # Llama3-style RoPE scaling
            orig_context_len = extra_config["original_max_seq_len"]
            low_freq_factor = extra_config["low_freq_factor"]
            high_freq_factor = extra_config["high_freq_factor"]

            wavelen = 2 * torch.pi / theta
            ratio = orig_context_len / wavelen
            smooth_factor = (ratio - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)

            # Compute adjusted_theta without masked indexing
            adjusted_theta = (1 - smooth_factor) * (theta / factor) + smooth_factor * theta
            theta = adjusted_theta
        else:
            # Linear scaling fallback
            theta = theta / factor

    # Create position indices `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device).float() / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    # If `n_elem` is odd, the final dimension of `idx_theta` has size
    # `n_elem + 1`, so need to cut something off.
    # Due to a current bug in Hugging Face, in the case `n_elem == 1`, we leave
    # `idx_theta`, `cos`, `sin` as is. Things work out in `apply_rope` due to
    # broadcasting. If we shorten `idx_theta`, unit tests comparing to
    # Hugging Face fail.
    # https://github.com/huggingface/transformers/issues/35233
    if idx_theta.shape[-1] > n_elem > 1:
        idx_theta = idx_theta[..., :n_elem]

    # if rope_local_base_freq is given, have a separate rope value for local embedding
    # For now, we use default RoPE for local embedding
    if rope_local_base_freq is not None:
        local_theta = 1.0 / (rope_local_base_freq ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
        local_idx_theta = torch.outer(seq_idx, local_theta)
        local_idx_theta = local_idx_theta.repeat(1, 2)
        if local_idx_theta.shape[-1] > n_elem > 1:
            local_idx_theta = local_idx_theta[..., :n_elem]

        idx_theta = torch.stack((idx_theta, local_idx_theta), dim=-1)

    cos = torch.cos(idx_theta) * attention_scaling
    sin = torch.sin(idx_theta) * attention_scaling
    return cos, sin


def batched_index_select(t, dim, idx):
    """index_select for batched index and unbatched t"""
    if idx.dim() == 1:
        return torch.index_select(t, dim, idx)

    *batch_shape, idx_size = idx.shape
    res = torch.index_select(t, dim, idx.reshape(-1))  # flat index
    # split out single batch idx
    res = res.view(*t.shape[:dim], -1, idx_size, *t.shape[dim + 1 :])
    if dim > 0:
        # move batch dim to front, this is np.rollaxis(res, dim, 0) for tensors
        dims = [dim] + list(range(res.dim()))
        del dims[dim + 1]
        res = res.permute(dims)
    # unflatten batch dims
    res = res.view(*batch_shape, *res.shape[1:])
    return res


def batched_index_copy_(t, dim, idx, val):
    """Index copy for batched t, idx, val"""

    if t.device.type == "mps":
        # Normalize negative dimensions
        if dim < 0:
            dim = t.dim() + dim
        if idx.dim() == 1:
            idx_shape = [1] * val.dim()
            idx_shape[dim] = -1
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)
            t.scatter_(dim, idx_expanded, val)
            return t

        elif idx.dim() == 2:
            assert dim != 0, "Cannot index the batch dimension"
            batch_size = idx.size(0)
            idx_size = idx.size(1)
            assert batch_size == t.size(0) == val.size(0)

            idx_shape = [batch_size] + [1] * (val.dim() - 1)
            idx_shape[dim] = idx_size
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)

            t.scatter_(dim, idx_expanded, val)
            return t
        else:
            raise NotImplementedError(f"idx.dim() == {idx.dim()} not supported")

    else:
        if idx.dim() == 1:
            return t.index_copy_(dim, idx, val)

        assert idx.dim() == 2, f"multiple batch dims not yet {idx.shape=}"
        assert dim != 0, f"cannot index batch dim {dim=}"
        batch_size, idx_size = idx.shape
        assert batch_size == t.size(0)
        assert batch_size == val.size(0)

        # if we can view the batch and indexed dimensions together, we could
        # do index trickery. This is, sadly, not the case for kvcache so we
        # fall back to for loop
        for i in range(batch_size):
            unbatched_dim = dim if dim < 0 else dim - 1
            t[i].index_copy_(unbatched_dim, idx[i], val[i])
        return t


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies RoPE transform to `x`. Note that `cos`, `sin` need to have a batch
    dimension.

    Args:
        x: Input tensor, `(B, ..., T, head_size)`
        cos: Cached cosines, `(B, T, head_size)` or `(1, T, head_size)`
        sin: Cached sines, `(B, T, head_size)` or `(1, T, head_size)`

    Returns:
        Encoded tensor, `(B, ..., T, head_size)`
    """
    if cos.dim() != 3:
        raise ValueError(f"cos must be three-dimensional, but shape is {cos.shape}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos, sin must have same shape, but cos.shape={cos.shape}, sin.shape={sin.shape}")
    head_size_half = x.size(-1) // 2
    x1 = x[..., :head_size_half]  # (B, ..., T, head_size/2)
    x2 = x[..., head_size_half:]  # (B, ..., T, head_size/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, ..., T, head_size)
    dims_diff = x.dim() - cos.dim()
    if dims_diff > 0:
        # Ensure that shapes of `x`, `cos`, `sin` align
        new_shape = cos.shape[0:1] + (1,) * dims_diff + cos.shape[1:]
        cos = cos.view(*new_shape)
        sin = sin.view(*new_shape)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


def apply_rope_interleave(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings with interleaved tensor layout.

    This version rearranges the input tensor to group even/odd indices separately
    before applying the standard RoPE rotation, matching HuggingFace's
    apply_rotary_pos_emb_interleave behavior.

    Args:
        x: Input tensor of shape (..., seq_len, head_dim)
        cos: Cosine component of shape (B, seq_len, head_dim) or (1, seq_len, head_dim)
        sin: Sine component of shape (B, seq_len, head_dim) or (1, seq_len, head_dim)

    Returns:
        Tensor with RoPE applied, same shape as input
    """
    if cos.dim() != 3:
        raise ValueError(f"cos must be three-dimensional, but shape is {cos.shape}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos, sin must have same shape, but cos.shape={cos.shape}, sin.shape={sin.shape}")

    # Rearrange tensor to group even/odd indices: [x0,x1,x2,x3,...] -> [x0,x2,x4,...,x1,x3,x5,...]
    *batch_dims, d = x.shape
    x = x.view(*batch_dims, d // 2, 2).transpose(-1, -2).reshape(*batch_dims, d)

    # Standard rotation logic (same as apply_rope)
    head_size_half = x.size(-1) // 2
    x1 = x[..., :head_size_half]
    x2 = x[..., head_size_half:]
    rotated = torch.cat((-x2, x1), dim=-1)

    # Auto-detect dimension mismatch and reshape cos/sin
    dims_diff = x.dim() - cos.dim()
    if dims_diff > 0:
        new_shape = cos.shape[0:1] + (1,) * dims_diff + cos.shape[1:]
        cos = cos.view(*new_shape)
        sin = sin.view(*new_shape)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


def do_softcapping(x: torch.Tensor, thresh: float) -> torch.Tensor:
    return torch.tanh(x / thresh) * thresh


class KVCache(nn.Module):
    """
    Buffers `k`, `v` have shape
    `(batch_size, n_query_groups, max_seq_length, head_size)`.
    """

    def __init__(
        self,
        k_shape: tuple[int, int, int, int],
        v_shape: tuple[int, int, int, int],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        is_sliding_window: bool = False,
        sliding_window_size: int | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)
        self.is_sliding_window = is_sliding_window
        self.sliding_window_size = sliding_window_size
        self.max_cache_len = k_shape[2]
        if is_sliding_window:
            if self.max_cache_len < 1:
                raise ValueError("sliding window must contain at least one token")
            self.register_buffer(
                "positions", torch.full((k_shape[0], self.max_cache_len), -1, device=device, dtype=torch.long),
                persistent=False,
            )

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Writes new values `k` and `v` into the cache at the positions specified
        by `input_pos` along the sequence dimension (`max_seq_length`). The batch
        size of `k` and `v` (`bs`) must be smaller or equal to `KVCache` batch
        size. Returns the full buffers, adjusted to the batch size `bs`.

        Args:
            input_pos: Position index, `(bs, T)` or `(T,)`
            k: New values, `(bs, n_query_groups, T, head_size)`
            v: New values, `(bs, n_query_groups, T, head_size)`

        Returns:
            k_full, v_full, `(bs, n_query_groups, max_seq_length, head_size)`

        """
        # move the buffer to the activation dtype for when AMP is used
        if self.k.dtype != k.dtype:
            self.k = self.k.to(k.dtype)
        if self.v.dtype != v.dtype:
            self.v = self.v.to(v.dtype)
        # update the cache
        bs = k.size(0)
        if self.is_sliding_window:
            # Circular buffer for sliding window
            prefill_len = input_pos.shape[-1]
            if prefill_len > self.max_cache_len:
                raise ValueError(
                    f"Prefill length ({prefill_len}) exceeds the sliding window size ({self.max_cache_len}). "
                    f"This causes the ring-buffer KV cache to overwrite entries, but the attention mask is not "
                    f"rebuilt to reflect the true positions, which silently violates causality. "
                    f"Please use chunked prefill with chunk size <= {self.max_cache_len} to avoid this issue."
                )
            cache_positions = input_pos % self.max_cache_len
            positions = input_pos.expand(bs, -1) if input_pos.ndim == 1 else input_pos
            self.positions[:bs].scatter_(1, positions % self.max_cache_len, positions)
            k = batched_index_copy_(self.k[:bs, ...], -2, cache_positions, k)
            v = batched_index_copy_(self.v[:bs, ...], -2, cache_positions, v)

            max_pos = input_pos.max().item()
            if max_pos < self.max_cache_len:
                k = k[:, :, : max_pos + 1, :]
                v = v[:, :, : max_pos + 1, :]
        else:
            # Standard KV cache (global attention)
            k = batched_index_copy_(self.k[:bs, ...], -2, input_pos, k)
            v = batched_index_copy_(self.v[:bs, ...], -2, input_pos, v)

        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)
        if self.is_sliding_window:
            self.positions.fill_(-1)


def build_mask_cache(max_seq_length: int, device: torch.device | None = None) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        weight = (1 + self.weight) if self.add_unit_offset else self.weight
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
