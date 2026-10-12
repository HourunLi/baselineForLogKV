"""Real CUDA kernels, tiny random model. No downloaded weights or quality claims."""

import pytest
import torch
from types import SimpleNamespace

from dms.training.engine import (
    DMSTrainerState, CombinedModel, _configure_gradients, distillation_forward,
)
from models.qwen3.check_env import check_environment
from models.qwen3.modeling_qwen3_dms import Qwen3ForCausalLMDMS
from tests.test_qwen3_base import tiny_config, tokenizer

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def test_cuda_distillation_then_prefill_decode(tokenizer):
    check_environment()
    # Same GQA grouping and 128-dimensional heads as the full 1.7B model.
    config = tiny_config(hidden_size=256, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=128, dms_alpha_scale=1, dms_initial_alpha_offset=0,
                         dms_tau=1)
    student = Qwen3ForCausalLMDMS(config).cuda().train()
    _configure_gradients(student)
    state = DMSTrainerState()
    state.set_trainer(SimpleNamespace(state=SimpleNamespace(global_step=0),
                                     args=SimpleNamespace(process_index=0)))
    combined = CombinedModel(student, student, state, lambda _: (0.75, 4.0),
                             distillation_forward, True, tokenizer, 16)
    ids = torch.randint(2, 32, (1, 1024), device="cuda")
    optimizer = torch.optim.AdamW(combined.get_parameters_to_optimize(), lr=1e-3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = combined(ids, torch.ones_like(ids, dtype=torch.bool))
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    grads = [p.grad for p in student.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    optimizer.step()
    student.bfloat16().eval()
    with torch.inference_mode():
        for _ in range(2):  # unrelated sequences must each start with an empty cache
            cache = student.get_cache()
            output = student.generate(ids, past_key_values=cache, max_new_tokens=3,
                                      do_sample=False, eos_token_id=None, pad_token_id=0)
            assert output.shape == (1, 1027)
            assert cache.get_seq_length() == 1026
            assert cache.get_cr() >= 1
