"""dms.transformers_compat with the installed transformers: the APIs DMS needs, used the way DMS uses them."""

import importlib.util

import pytest
import torch
from transformers.cache_utils import Cache as InstalledCache

from dms import transformers_compat as compat
from dms.cache import DMSCache
from models.qwen3.modeling_qwen3_dms import Qwen3ForCausalLMDMS


def test_dms_cache_is_accepted_as_a_transformers_cache():
    cache = DMSCache(dms_window_size=33, max_context_length=64, block_size=16)
    # transformers' own generate() checks both before it uses a user-provided cache.
    assert isinstance(cache, InstalledCache) and Qwen3ForCausalLMDMS._supports_cache_class
    assert len(cache) == 0 and cache.get_seq_length() == 0
    cache[1]  # layers are created lazily, as in transformers 4.57.3
    assert len(cache) == 2 and cache.get_seq_length() == 0


@pytest.mark.skipif(importlib.util.find_spec("transformers.masking_utils") is not None,
                    reason="the installed transformers provides masking_utils")
def test_mask_fallback_accepts_unpadded_inputs_and_refuses_padding():
    unused = dict(config=None, input_embeds=None, cache_position=None, past_key_values=None)
    assert compat.create_causal_mask(attention_mask=None, **unused) is None
    # generate() and the training collator pass a mask of ones for an unpadded sequence
    assert compat.create_causal_mask(attention_mask=torch.ones(1, 5, dtype=torch.bool), **unused) is None
    assert compat.create_causal_mask(attention_mask=torch.ones(1, 5, dtype=torch.long), **unused) is None
    with pytest.raises(NotImplementedError, match="Padded"):
        compat.create_causal_mask(attention_mask=torch.tensor([[False, True, True]]), **unused)
