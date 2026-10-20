"""Training-budget/gradient tests without downloading data or running real training."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from hybrid_csa import train


def test_config_preview_does_not_import_training_dependencies():
    source = """
import sys
class BlockRuntime:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','lightning','lm_eval','litdata','litgpt','datasets'}:
            raise RuntimeError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
from hybrid_csa.train import main
main(['--print-config'])
"""
    result = subprocess.run([sys.executable, "-c", source], cwd=train.BASELINE_ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    cfg = yaml.safe_load(result.stdout)
    assert cfg["context_length"] == 4096
    assert [x["name"] for x in cfg["stages"]] == list(train.STAGES)
    assert sum(x["steps"] for x in cfg["stages"]) * cfg["global_batch_size"] * cfg["context_length"] == 5242880


def test_training_config_refuses_original_asset_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "BASELINE_ROOT", tmp_path)
    monkeypatch.setattr(train, "require_project_file", lambda p: p)
    cfg = yaml.safe_load(train.DEFAULT_CONFIG.read_text())
    cfg["output_dir"] = cfg["base_checkpoint_dir"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="overlaps original assets"):
        train.resolve_config(path)


def test_schema_fails_instead_of_silently_dropping_rows():
    assert train.text_from_record({"text": "  hello  "}, "row") == "hello"
    assert train.text_from_record({"chapter": "book"}, "row") == "book"
    assert train.text_from_record({"messages": [{"role": "assistant", "content": "answer"}]}, "row") == "### assistant\nanswer"
    for row in (None, {}, {"text": ""}, {"messages": [{"content": None}]}):
        with pytest.raises(ValueError):
            train.text_from_record(row, "bad document")


def test_accumulation_and_learning_rate_budget():
    assert train.accumulation_steps(8, 1, 8) == 1
    assert train.accumulation_steps(8, 1, 2) == 4
    with pytest.raises(ValueError, match="divisible"):
        train.accumulation_steps(7, 1, 8)
    assert train.learning_rate(0.1, 0, 10, 2) == pytest.approx(0.05)
    assert train.learning_rate(0.1, 9, 10, 2) == pytest.approx(0.01)


def tiny_model():
    import torch

    class TinyAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.compressor = torch.nn.Parameter(torch.tensor(0.5))
            self.indexer = torch.nn.Parameter(torch.tensor(0.25))

        def adapter_named_parameters(self):
            return [("compressor", self.compressor), ("indexer", self.indexer)]

        def set_stage(self, stage):
            self.stage = stage
            self.compressor.requires_grad_(stage != "indexer")
            self.indexer.requires_grad_(stage != "compressor")

        def training_loss(self, inputs, targets, **kwargs):
            return (self.base * self.compressor * inputs.float().mean() - targets.float().mean()).square(), self.indexer.square()

    return TinyAdapter()


def tiny_loop_config():
    return dict(context_length=7, micro_batch_size=1, global_batch_size=2, grad_clip=1.0,
                loss_chunk_size=4, activation_checkpointing=True, weight_decay=0.0,
                warmup_steps=0, indexer_loss_weight=1.0)


@pytest.mark.parametrize("stage", train.STAGES)
def test_optimizer_updates_only_stage_parameters_and_records_actual_tokens(stage):
    import torch

    model = tiny_model()
    before = copy.deepcopy(model.state_dict())
    loader = [torch.arange(8).reshape(1, 8)] * 4
    result = train.train_stage(model, tiny_loop_config(), {"name": stage, "steps": 2, "learning_rate": 0.1}, loader, torch.device("cpu"))
    assert result["global_tokens"] == 28 and result["steps"] == 2
    assert torch.equal(model.base, before["base"])
    assert torch.equal(model.compressor, before["compressor"]) == (stage == "indexer")
    assert torch.equal(model.indexer, before["indexer"]) == (stage == "compressor")
    assert all(torch.isfinite(torch.tensor(item["loss"])) for item in result["history"])


def test_short_corpus_fails_instead_of_silently_repeating_it():
    import torch

    with pytest.raises(RuntimeError, match="corpus exhausted"):
        train.train_stage(tiny_model(), tiny_loop_config(), {"name": "compressor", "steps": 2, "learning_rate": 0.1},
                          [torch.arange(8).reshape(1, 8)], torch.device("cpu"))


def test_all_three_stages_backpropagate_on_real_tiny_gpt():
    import torch
    from hybrid_csa._vendor.semanticlogkv.litgpt.config import Config
    from hybrid_csa._vendor.semanticlogkv.litgpt.model import GPT
    from hybrid_csa.model import HybridCSAModel
    from hybrid_csa.settings import HybridCSASettings

    config = Config(name="tiny-qwen3", block_size=32, vocab_size=32, padded_vocab_size=32,
                    n_layer=3, n_head=2, n_query_groups=1, n_embd=16, head_size=8,
                    intermediate_size=32, norm_qk=True, mlp_class_name="LLaMAMLP", norm_class_name="RMSNorm",
                    parallel_residual=False, bias=False, rope_base=10000)
    gpt = GPT(config)
    frozen = [(parameter, parameter.detach().clone()) for parameter in gpt.parameters()]
    model = HybridCSAModel(gpt, HybridCSASettings(window_size=4, index_topk=2, index_dim=4, index_heads=1, query_chunk_size=4))
    cfg = {**tiny_loop_config(), "context_length": 15, "global_batch_size": 1}
    batch = torch.randint(0, 32, (1, 16))
    for name in train.STAGES:
        result = train.train_stage(model, cfg, {"name": name, "steps": 1, "learning_rate": 0.01}, [batch], torch.device("cpu"))
        assert result["global_tokens"] == 15
    for parameter, original in frozen:
        assert torch.equal(parameter, original)
