"""Real CUDA kernels, tiny random model. No downloaded weights or quality claims."""

import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from dms.training.engine import (
    DMSTrainerState, CombinedModel, _configure_gradients, distillation_forward,
)
from dms.training.loop import init_runtime
from models.qwen3.check_env import check_environment
from models.qwen3.modeling_qwen3_dms import Qwen3ForCausalLMDMS
from models.qwen3.train import export_trained_student
from tests.test_qwen3_base import tiny_config, tokenizer
from tests.test_training_loop import adapters, fixture, run  # noqa: F401  (fixture)

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


def test_cuda_fsdp_training_resumes_and_exports(fixture, tmp_path, monkeypatch):
    """The recipe's FSDP setup (CPU offload, module-state sync, activation checkpointing) in one process."""
    check_environment(training_only=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    for name, value in dict(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK="0", WORLD_SIZE="1",
                            LOCAL_RANK="0").items():
        monkeypatch.setenv(name, value)
    runtime = init_runtime()
    fsdp = dict(fsdp="full_shard offload auto_wrap", fsdp_config=dict(
        use_orig_params=True, sync_module_states=True, activation_checkpointing=True,
        transformer_layer_cls_to_wrap=["Qwen3DecoderLayerDMS"], state_dict_type="FULL_STATE_DICT"))
    try:
        straight, _ = run(fixture, "straight", runtime=runtime, max_steps=3, **fsdp)
        run(fixture, "resumed", runtime=runtime, max_steps=2, **fsdp)
        resumed, seconds = run(fixture, "resumed", resume=str(fixture[0] / "resumed/checkpoint-2"),
                               runtime=runtime, max_steps=3, **fsdp)
        # GPU kernels need not be bitwise deterministic; a resume must still continue the same run.
        for a, b in zip(straight.state.log_history, resumed.state.log_history, strict=True):
            assert (a["step"], a["epoch"]) == (b["step"], b["epoch"])
            torch.testing.assert_close(a["loss"], b["loss"], rtol=1e-3, atol=1e-6)
        final, expected = adapters(resumed), adapters(straight)
        assert sorted(final) == sorted(expected)
        for name in expected:
            torch.testing.assert_close(final[name], expected[name], rtol=1e-3, atol=1e-5)
        path = tmp_path / "student_model"
        export_trained_student(resumed, resumed.combined, fixture[3], str(path), resumed.training_record(seconds))
        exported = dict(Qwen3ForCausalLMDMS.from_pretrained(path, torch_dtype=torch.bfloat16).named_parameters())
        for name, value in final.items():
            assert torch.equal(exported[name.removeprefix("student_model.")], value.bfloat16())
    finally:
        dist.destroy_process_group()
