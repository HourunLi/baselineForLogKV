# SPDX-FileCopyrightText: Copyright 2025 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Added for this baseline: `CacheLayerMixin` and `Cache` are copied from transformers 4.57.3
# (src/transformers/cache_utils.py; Cache docstring shortened and, with an older transformers, Cache
# subclassing the installed Cache); the other fallbacks are new.
"""transformers APIs the DMS code was written against (4.57.3) that older releases lack.

Evaluation runs in SemanticLogKV's environment (torch218, transformers 4.51.3); DMS training keeps the
upstream transformers 4.57.3 in its own environment. Every name below comes from the installed
transformers when it provides it, so the training environment runs the upstream classes unchanged.
Otherwise:

* ``CacheLayerMixin`` and ``Cache``: the transformers 4.57.3 classes, copied verbatim, used as a pair. The
  copied ``Cache`` subclasses the installed one only so transformers' own ``isinstance`` checks (e.g. in
  ``generate``) accept a DMS cache; it overrides every member the older class defines except
  ``get_usable_length`` and ``seen_tokens``, which DMS does not use.
* ``create_causal_mask``: without masking_utils (transformers>=4.53) it returns ``None`` for an input
  without padding (no mask, or a 2D mask of ones as from ``generate`` or the training collator), which is
  also what 4.57.3 returns there with SDPA. DMS training ignores mask values and DMS inference reads only
  the last mask row as a key padding mask, so both see all ones; padded inputs are refused.
* ``GradientCheckpointingLayer``: a plain module; checkpointing is a training feature.
* ``TransformersKwargs``: the same TypedDict, used in type hints only.
* The generic task heads are ``None``; only the causal LM is used here.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, TypedDict

import torch
from torch import nn

try:
    from transformers.cache_utils import Cache, CacheLayerMixin
except ImportError:  # transformers<4.56
    from transformers.cache_utils import Cache as _InstalledCache

    _is_torch_greater_or_equal_than_2_7 = tuple(int(part) for part in torch.__version__.split(".")[:2]) >= (2, 7)

    class CacheLayerMixin(ABC):
        """Base, abstract class for a single layer's cache."""

        is_compileable = False

        def __init__(self):
            self.keys: Optional[torch.Tensor] = None
            self.values: Optional[torch.Tensor] = None
            self.is_initialized = False

        def __repr__(self):
            return f"{self.__class__.__name__}"

        @abstractmethod
        def lazy_initialization(self, key_states: torch.Tensor): ...

        @abstractmethod
        def update(
            self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None
        ) -> tuple[torch.Tensor, torch.Tensor]: ...

        @abstractmethod
        def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]: ...

        @abstractmethod
        def get_seq_length(self) -> int: ...

        @abstractmethod
        def get_max_cache_shape(self) -> int: ...

        def offload(self):
            """Offload this layer's data to CPU device."""
            if self.is_initialized:
                self.keys = self.keys.to("cpu", non_blocking=True)
                self.values = self.values.to("cpu", non_blocking=True)

        def prefetch(self):
            """In case of layer offloading, this allows to move the data back to the layer's device ahead of time."""
            if self.is_initialized and self.keys.device != self.device:
                self.keys = self.keys.to(self.device, non_blocking=True)
                self.values = self.values.to(self.device, non_blocking=True)

        def reset(self) -> None:
            """Resets the cache values while preserving the objects"""
            if self.is_initialized:
                self.keys.zero_()
                self.values.zero_()
            # This attribute is set on several Layers
            if hasattr(self, "cumulative_length"):
                self.cumulative_length = 0

        def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
            """Reorders this layer's cache for beam search."""
            if self.get_seq_length() > 0:
                self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
                self.values = self.values.index_select(0, beam_idx.to(self.values.device))

    class Cache(_InstalledCache):
        """Container of one `CacheLayerMixin` per model layer (transformers 4.57.3)."""

        def __init__(
            self,
            layers: Optional[list[CacheLayerMixin]] = None,
            layer_class_to_replicate: Optional[type[CacheLayerMixin]] = None,
            offloading: bool = False,
            offload_only_non_sliding: bool = True,
        ):
            if layers is not None and layer_class_to_replicate is not None:
                raise ValueError(
                    "You can construct a Cache either from a list `layers` of all the predefined `CacheLayer`, or from a "
                    "`layer_class_to_replicate`, in which case the Cache will append a new layer corresponding to "
                    "`layer_class_to_replicate` for each new call to `update` with an idx not already in the Cache."
                )
            if layers is None and layer_class_to_replicate is None:
                raise ValueError(
                    "You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache."
                )
            self.layers = layers if layers is not None else []
            self.layer_class_to_replicate = layer_class_to_replicate
            self.offloading = offloading
            if self.offloading:
                self.only_non_sliding = offload_only_non_sliding
                self.prefetch_stream = torch.Stream() if _is_torch_greater_or_equal_than_2_7 else torch.cuda.Stream()

        def __repr__(self):
            return f"{self.__class__.__name__}(layers={self.layers})"

        def prefetch(self, layer_idx: int, only_non_sliding: bool = True):
            """
            Prefetch a given layer on its device. If `only_non_sliding` is True, it will try to prefetch only the layers
            which are non-sliding. If the `layer_idx` is outside the range, this will circle back to the first layers.
            Note that we use a non-default stream for this, to avoid blocking.
            """
            if only_non_sliding:
                # Try to find next non-sliding, starting at `layer_idx`
                try:
                    layer_idx = layer_idx + self.is_sliding[layer_idx:].index(False)
                # In this case, we need to circle back to the beginning
                except ValueError:
                    layer_idx = self.is_sliding.index(False)
            else:
                layer_idx = layer_idx if layer_idx < len(self.layers) else 0

            # Prefetch
            with self.prefetch_stream if _is_torch_greater_or_equal_than_2_7 else torch.cuda.stream(self.prefetch_stream):
                self.layers[layer_idx].prefetch()

        def offload(self, layer_idx: int, only_non_sliding: bool = True):
            """
            Offload a given `layer_idx`. If `only_non_sliding` is True, it will offload `layer_idx` only if it is a
            non-sliding layer. Note that we do it on the default stream, so that we ensure all earlier
            computation in the layer's `update` methods are finished.
            """
            if not (only_non_sliding and self.is_sliding[layer_idx]):
                self.layers[layer_idx].offload()

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[dict[str, Any]] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

            Parameters:
                key_states (`torch.Tensor`):
                    The new key states to cache.
                value_states (`torch.Tensor`):
                    The new value states to cache.
                layer_idx (`int`):
                    The index of the layer to cache the states for.
                cache_kwargs (`dict[str, Any]`, *optional*):
                    Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                    cache to be created.

            Return:
                A tuple containing the updated key and value states.
            """
            # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
            if self.layer_class_to_replicate is not None:
                while len(self.layers) <= layer_idx:
                    self.layers.append(self.layer_class_to_replicate())

            if self.offloading:
                # Wait for the stream to finish if needed, and start prefetching the next layer
                torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
                self.prefetch(layer_idx + 1, self.only_non_sliding)

            keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

            if self.offloading:
                self.offload(layer_idx, self.only_non_sliding)

            return keys, values

        def early_initialization(
            self, batch_size: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device
        ):
            """
            Initialize all the layers in advance (it's otherwise lazily initialized on the first `update` call).
            This is useful for our `export` recipes, as `export` needs everything in advance.
            """
            # Note that the initialization needs all dimensions (except -2), as well as device and dtype, so we use
            # this fake tensor approach. It has size 0 on the -2 dimension, so it does not allocate any data (it only
            # creates an empty tensor with correct shape, dtype and device), which is very efficient and practical
            fake_keys_tensor = torch.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype, device=device)
            # Init all layers
            for layer in self.layers:
                layer.lazy_initialization(fake_keys_tensor)

        def get_seq_length(self, layer_idx: int = 0) -> int:
            """Returns the sequence length of the cache for the given layer."""
            if layer_idx >= len(self.layers):
                return 0
            return self.layers[layer_idx].get_seq_length()

        def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
            """
            Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
            the given layer at `layer_idx`.
            The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
            """
            # For DynamicCache, where the layers are created at runtime -> if it was not yet created, the size is
            # simply the shape of `cache_position`
            if layer_idx >= len(self.layers):
                return cache_position.shape[0], 0
            return self.layers[layer_idx].get_mask_sizes(cache_position)

        def get_max_cache_shape(self, layer_idx: int = 0) -> int:
            """Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length."""
            # For DynamicCache, where the layers are created at runtime -> if it was not yet created, return -1
            # as DynamicLayer does
            if layer_idx >= len(self.layers):
                return -1
            return self.layers[layer_idx].get_max_cache_shape()

        def reset(self):
            """Recursively reset all layers tensors"""
            for layer_idx in range(len(self.layers)):
                self.layers[layer_idx].reset()

        def reorder_cache(self, beam_idx: torch.LongTensor):
            """Reorder the cache for beam search"""
            for layer_idx in range(len(self.layers)):
                self.layers[layer_idx].reorder_cache(beam_idx)

        def crop(self, max_length: int):
            """Crop the cache to the given length"""
            for layer_idx in range(len(self.layers)):
                self.layers[layer_idx].crop(max_length)

        def batch_repeat_interleave(self, repeats: int):
            """Repeat and interleave the cache"""
            for layer_idx in range(len(self.layers)):
                self.layers[layer_idx].batch_repeat_interleave(repeats)

        def batch_select_indices(self, indices: torch.Tensor):
            """Select indices from the cache"""
            for layer_idx in range(len(self.layers)):
                self.layers[layer_idx].batch_select_indices(indices)

        @property
        def max_batch_size(self) -> int:
            """Return the maximum batch size of the cache"""
            values = [layer.max_batch_size for layer in self.layers]
            if len(set(values)) > 1:
                raise ValueError(f"Max batch size is not consistent across layers: {values}")
            return values[0]

        @property
        def max_cache_len(self) -> int:
            """Return the maximum cache length of the cache"""
            values = [layer.max_cache_len for layer in self.layers]
            return max(values)

        @property
        def is_compileable(self) -> bool:
            """Return whether the cache is compileable"""
            # For DynamicCache dispatching the layers lazily (otherwise, all([]) is True)
            if len(self.layers) == 0:
                return False
            return all(layer.is_compileable for layer in self.layers)

        @property
        def is_initialized(self) -> bool:
            """Return whether the cache data is initialized"""
            return len(self.layers) > 0 and all(layer.is_initialized for layer in self.layers)

        @property
        def is_sliding(self) -> list[bool]:
            """Return whether the layers of the cache are sliding window"""
            return [getattr(layer, "is_sliding", False) for layer in self.layers]

        def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Support for backwards-compatible `past_key_values` indexing, e.g. `past_key_values[0][0].shape[2]` to get the
            sequence length.
            """
            if layer_idx < len(self.layers):
                return self.layers[layer_idx].keys, self.layers[layer_idx].values
            else:
                raise KeyError(
                    f"Cache only has {len(self.layers)} layers, attempted to access layer with index {layer_idx}"
                )

        def __iter__(self):
            """
            Support for backwards-compatible `past_key_values` iteration, e.g. `for x in past_key_values:` to iterate over
            keys and values
            """
            for layer_idx in range(len(self)):
                yield (self.layers[layer_idx].keys, self.layers[layer_idx].values)

        def __len__(self):
            """
            This value corresponds to the number of layers in the model.
            """
            # Note: for DynamicCache, layers are initialized lazily, so this will not be accurate before the first
            # forward through all the layers
            return len(self.layers)

try:
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
except ImportError:  # transformers<4.53

    def create_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, *args, **kwargs):
        """``None`` for an input without padding; DMS then attends to every key of the request."""
        if attention_mask is not None and not (attention_mask.ndim == 2 and bool(attention_mask.all())):
            raise NotImplementedError("Padded DMS inputs need transformers>=4.53 (masking_utils)")
        return None

    def create_sliding_window_causal_mask(*args, **kwargs):
        raise NotImplementedError("Sliding-window layers need transformers>=4.53 (masking_utils)")

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:  # transformers<4.52

    class GradientCheckpointingLayer(nn.Module):
        """Decoder-layer base without gradient checkpointing (a training feature)."""

        gradient_checkpointing = False

        def __call__(self, *args, **kwargs):
            if self.gradient_checkpointing and self.training:
                raise RuntimeError("Gradient checkpointing needs transformers>=4.52 (the DMS training environment)")
            return super().__call__(*args, **kwargs)

try:
    from transformers.modeling_layers import (
        GenericForQuestionAnswering,
        GenericForSequenceClassification,
        GenericForTokenClassification,
    )
except ImportError:  # transformers<4.53: the task heads are not defined
    GenericForQuestionAnswering = GenericForSequenceClassification = GenericForTokenClassification = None

try:
    from transformers.utils import TransformersKwargs
except ImportError:  # transformers<4.54

    class TransformersKwargs(TypedDict, total=False):
        """Keyword arguments of a model forward pass (type hints only)."""

        num_items_in_batch: Optional[torch.Tensor]
        output_hidden_states: Optional[bool]
        output_attentions: Optional[bool]
        output_router_logits: Optional[bool]
        cu_seq_lens_q: Optional[torch.LongTensor]
        cu_seq_lens_k: Optional[torch.LongTensor]
        max_length_q: Optional[int]
        max_length_k: Optional[int]
