"""Prefill -> SnapKV -> decode equals independent code: Transformers' Qwen3 with the upstream selection.

The reference runs a dense prefill through ``transformers.Qwen3ForCausalLM``, replaces every layer's cache
by the upstream KVCache-Factory ``kv_head`` SnapKV output computed from that layer's post-RoPE queries (the
``update_kv`` call KVCache-Factory makes inside the prefill attention), then decodes on the compressed cache
at the true positions. This project's model and cache must give the same logits.
"""

from functools import partial

import pytest
import torch

from snapkv.compressor import SnapKVCompressor
from snapkv.kv_cache import CompressedKVCache
from tests.helpers import TINY, save_hf_checkpoint, tiny_model
from tests.test_compressor import reference_update_kv_kv_head


@pytest.mark.parametrize("pooling,agg", [("maxpool", "mean"), ("avgpool", "max")])
def test_prefill_compress_decode_matches_transformers_with_upstream_snapkv(tmp_path, monkeypatch, pooling, agg):
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen3 import modeling_qwen3

    model = tiny_model(seed=7)
    reference = transformers.Qwen3ForCausalLM.from_pretrained(
        save_hf_checkpoint(model, tmp_path / "hf"), torch_dtype=torch.float32, attn_implementation="eager").eval()
    prompt_len, decode_len, budget, window, kernel = 50, 6, 16, 4, 3
    tokens = torch.randint(0, TINY.vocab_size, (1, prompt_len + decode_len), generator=torch.Generator().manual_seed(3))

    queries = []
    rope = modeling_qwen3.apply_rotary_pos_emb

    def recording_rope(q, k, cos, sin, *args, **kwargs):
        q_embed, k_embed = rope(q, k, cos, sin, *args, **kwargs)
        queries.append(q_embed)
        return q_embed, k_embed

    cache = transformers.DynamicCache()
    with torch.no_grad():
        monkeypatch.setattr(modeling_qwen3, "apply_rotary_pos_emb", recording_rope)
        expected = [reference(tokens[:, :prompt_len], past_key_values=cache, use_cache=True).logits]
        monkeypatch.setattr(modeling_qwen3, "apply_rotary_pos_emb", rope)
        assert len(queries) == TINY.num_hidden_layers
        for layer, query in enumerate(queries):
            cache.key_cache[layer], cache.value_cache[layer] = reference_update_kv_kv_head(
                cache.key_cache[layer], query, cache.value_cache[layer], window, budget, kernel, pooling, agg)
        for pos in range(prompt_len, prompt_len + decode_len):
            position = torch.tensor([pos])
            expected.append(reference(tokens[:, pos:pos + 1], past_key_values=cache, use_cache=True,
                                      position_ids=position.view(1, 1), cache_position=position).logits)

        compressor = SnapKVCompressor(TINY.num_hidden_layers, max_capacity_prompt=budget, window_size=window,
                                      kernel_size=kernel, pooling=pooling, gqa_score_agg=agg)
        model.set_kv_cache(partial(CompressedKVCache, compressor=compressor))
        actual = [model(tokens[:, :prompt_len], input_pos=torch.arange(prompt_len))]
        for pos in range(prompt_len, prompt_len + decode_len):
            actual.append(model(tokens[:, pos:pos + 1], input_pos=torch.tensor([pos])))
        dense = tiny_model(seed=7)(tokens)

    assert [c.length for c in model.caches] == [budget + decode_len] * TINY.num_hidden_layers
    assert all(k.size(2) == budget + decode_len for k in cache.key_cache)
    torch.testing.assert_close(torch.cat(actual, dim=1), torch.cat(expected, dim=1), atol=1e-4, rtol=1e-4)
    # The decode logits really come from the compressed cache.
    assert not torch.allclose(torch.cat(actual[1:], dim=1), dense[:, prompt_len:], atol=1e-3)
