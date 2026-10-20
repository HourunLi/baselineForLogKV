"""Tiny Qwen checks for causality, GQA, streaming equivalence, and adapter gradients."""

import copy

import pytest
import torch

from hybrid_csa._vendor.semanticlogkv.litgpt.config import Config
from hybrid_csa._vendor.semanticlogkv.litgpt.model import GPT
from hybrid_csa.core import GatedCompressor
from hybrid_csa.model import HybridCSAModel
from hybrid_csa.settings import HybridCSASettings


def tiny_gpt():
    torch.manual_seed(37)
    return GPT(Config(
        name="tiny-qwen3", block_size=160, vocab_size=32, padded_vocab_size=32,
        n_layer=4, n_head=4, n_query_groups=2, n_embd=32, head_size=8,
        intermediate_size=48, norm_qk=True, mlp_class_name="LLaMAMLP",
        norm_class_name="RMSNorm", parallel_residual=False, bias=False,
        rope_base=10000, rotary_percentage=1.0,
    ))


def tiny_model(**options):
    settings = dict(window_size=5, index_topk=2, index_dim=4, index_heads=2,
                    query_chunk_size=5, prefill_chunk_size=7)
    settings.update(options)
    return HybridCSAModel(tiny_gpt(), HybridCSASettings(**settings))


def stream(model, tokens, sizes):
    model.reset_cache()
    output, position = [], 0
    for size in sizes:
        stop = min(position + size, tokens.size(1))
        if stop > position:
            output.append(model(tokens[:, position:stop], torch.arange(position, stop)))
            position = stop
    assert position == tokens.size(1)
    return torch.cat(output, dim=1)


def test_uniform_initialization_uses_previous_and_current_blocks():
    compressor = GatedCompressor(hidden_size=4, groups=2, head_dim=3)
    x = torch.randn(1, 12, 4)
    keys, values = torch.randn(1, 12, 2, 3), torch.randn(1, 12, 2, 3)
    actual = compressor(x, keys, values, torch.tensor([0, 4, 8]))
    for source, compressed in zip((keys, values), actual):
        expected = torch.stack((source[:, :4].mean(1), source[:, :8].mean(1), source[:, 4:12].mean(1)), dim=1)
        torch.testing.assert_close(compressed, expected)


def test_all_full_matches_original_qwen_with_gqa():
    reference = tiny_gpt().eval()
    model = HybridCSAModel(copy.deepcopy(reference), HybridCSASettings(full_layer_interval=1, prefill_chunk_size=7)).eval()
    tokens = torch.randint(0, 32, (1, 23))
    with torch.no_grad():
        expected = reference(tokens)
        actual = stream(model, tokens, [1, 3, 2, 9, 8])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("stage", ["compressor", "indexer", "sparse"])
def test_arbitrary_streaming_matches_full_sequence_and_is_causal(stage):
    model = tiny_model().eval()
    model.set_stage(stage)
    tokens = torch.randint(0, 32, (1, 29))
    with torch.no_grad():
        expected = model.forward_train(tokens)
        changed = tokens.clone()
        changed[:, 11:] = (changed[:, 11:] + 7) % 32
        perturbed = model.forward_train(changed)
        torch.testing.assert_close(expected[:, :11], perturbed[:, :11], atol=1e-5, rtol=1e-5)
        for sizes in ([29], [1] * 29, [1, 2, 4, 5, 3, 6, 8]):
            actual = stream(model, tokens, sizes)
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        assert all(attention.key_buffer.length == (29 if attention.full else 7) for attention in model.attentions)
    model.release_cache()


def test_prefill_slicing_preserves_requested_logits_across_window_boundary():
    model = tiny_model(window_size=128, prefill_chunk_size=31, query_chunk_size=13).eval()
    tokens = torch.randint(0, 32, (1, 139))
    with torch.no_grad():
        expected = model.forward_train(tokens)
        model.reset_cache()
        actual = model(tokens, torch.arange(139), lm_head_start=125)
    torch.testing.assert_close(actual, expected[:, 125:], atol=1e-5, rtol=1e-5)
    assert all(attention.local_k.size(1) == 128 for attention in model.attentions if not attention.full)


def test_tied_index_scores_do_not_change_selection_with_prefill_partition():
    model = tiny_model(prefill_chunk_size=31).eval()
    with torch.no_grad():
        for attention in model.attentions:
            if not attention.full:
                attention.indexer.query_proj.weight.zero_()
        tokens = torch.randint(0, 32, (1, 43))
        expected = model.forward_train(tokens)
        for sizes in ([43], [1] * 43, [3, 8, 7, 25]):
            torch.testing.assert_close(stream(model, tokens, sizes), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("stage", ["compressor", "indexer", "sparse"])
def test_checkpointed_training_preserves_gradients_and_stage_boundaries(stage):
    model = tiny_model().train()
    model.set_stage(stage)
    tokens = torch.randint(0, 32, (1, 25))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    recorded = []
    for activation_checkpointing in (False, True):
        model.zero_grad(set_to_none=True)
        lm_loss, index_loss = model.training_loss(inputs, targets, loss_chunk_size=7,
                                                  activation_checkpointing=activation_checkpointing)
        loss = index_loss if stage == "indexer" else lm_loss + index_loss
        assert torch.isfinite(loss)
        loss.backward()
        gradients = {name: parameter.grad.detach().clone() for name, parameter in model.adapter_named_parameters()
                     if parameter.grad is not None}
        for name, parameter in model.named_parameters():
            if ".compressor." not in name and ".indexer." not in name:
                assert parameter.grad is None, name
        for family in ("compressor", "indexer"):
            expected_active = stage == "sparse" or stage == family
            matching = [gradient for name, gradient in gradients.items() if f".{family}." in name]
            assert bool(matching) == expected_active
            if expected_active:
                assert all(torch.isfinite(gradient).all() for gradient in matching)
                assert any(torch.count_nonzero(gradient) for gradient in matching)
        recorded.append((loss.detach().clone(), gradients))
    torch.testing.assert_close(recorded[0][0], recorded[1][0])
    assert recorded[0][1].keys() == recorded[1][1].keys()
    for name, gradient in recorded[0][1].items():
        torch.testing.assert_close(gradient, recorded[1][1][name], atol=1e-6, rtol=1e-5)


def test_bf16_backbone_keeps_fp32_adapter_parameters_and_gradients():
    model = HybridCSAModel(
        tiny_gpt().to(dtype=torch.bfloat16),
        HybridCSASettings(window_size=5, index_topk=2, index_dim=4, index_heads=2, query_chunk_size=5),
    ).train()
    tokens = torch.randint(0, 32, (1, 25))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        lm_loss, index_loss = model.training_loss(tokens[:, :-1], tokens[:, 1:], loss_chunk_size=7)
    (lm_loss + index_loss).backward()
    for _, parameter in model.adapter_named_parameters():
        assert parameter.dtype == torch.float32
        assert parameter.grad is not None
        assert parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()
