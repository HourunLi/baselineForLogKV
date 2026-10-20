import torch
import pytest

from hybrid_csa.checkpoint import load_adapter, save_adapter
from hybrid_csa.settings import HybridCSASettings


class Adapter:
    """Only the adapter serialization interface; numerical behavior is tested separately."""
    def __init__(self, **settings):
        self.settings = HybridCSASettings(**settings)
        self.stage = "sparse"
        self.weight = torch.randn(2, 3)

    def adapter_state_dict(self):
        return {"compressor.weight": self.weight.clone()}

    def load_adapter_state_dict(self, state):
        self.weight.copy_(state["compressor.weight"])

    def set_stage(self, stage):
        self.stage = stage


def test_adapter_bound_to_base_and_architecture_and_trained_stage(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    for name in ("lit_model.pth", "model_config.yaml", "tokenizer.json"):
        (base / name).write_text(name)
    source = Adapter()
    path = tmp_path / "adapter.pt"
    record = {"optimizer_steps": 3, "stage_steps": {"compressor": 1, "indexer": 1, "sparse": 1}}
    save_adapter(source, path, base, record)
    target = Adapter(query_chunk_size=7, prefill_chunk_size=3)
    result = load_adapter(target, path, base)
    torch.testing.assert_close(source.weight, target.weight)
    assert result["training"] == record and len(result["sha256"]) == 64
    with pytest.raises(FileExistsError):
        save_adapter(source, path, base, record)
    with pytest.raises(ValueError, match="architecture"):
        load_adapter(Adapter(window_size=64), path, base)
    (base / "lit_model.pth").write_text("different base")
    with pytest.raises(ValueError, match="fingerprints"):
        load_adapter(Adapter(), path, base)
    with pytest.raises(ValueError, match="optimizer_steps"):
        save_adapter(source, tmp_path / "untrained.pt", base, {"optimizer_steps": 0})


def test_formal_evaluation_refuses_dense_warmup_adapter(tmp_path):
    for name in ("lit_model.pth", "model_config.yaml"):
        (tmp_path / name).write_text(name)
    model = Adapter()
    model.stage = "compressor"
    save_adapter(model, tmp_path / "warmup.pt", tmp_path,
                 {"optimizer_steps": 1, "stage_steps": {"compressor": 1}})
    with pytest.raises(ValueError, match="sparse"):
        load_adapter(Adapter(), tmp_path / "warmup.pt", tmp_path)
    assert load_adapter(Adapter(), tmp_path / "warmup.pt", tmp_path, require_sparse=False)["stage"] == "compressor"
