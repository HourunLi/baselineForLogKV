from types import SimpleNamespace

import torch

from deepseek4.attention import Geometry, LAYOUT
from deepseek4.model import Model
from deepseek4.evaluate import stages


def tiny():
    g = Geometry(
        width=24,
        heads=4,
        dim=8,
        rotary=4,
        q_rank=12,
        groups=2,
        o_rank=8,
        index_heads=2,
        index_dim=6,
        topk=2,
        window=8,
    )
    cfg = SimpleNamespace(
        block_size=256, padded_vocab_size=79, intermediate_size=32, rope_adjustments={"factor": 2}
    )
    return Model(cfg, g, (0, 4, 128))


def test_flash_migration():
    cfg = SimpleNamespace(
        block_size=32768,
        padded_vocab_size=151936,
        intermediate_size=12288,
        n_embd=4096,
        n_layer=36,
        n_head=32,
        n_query_groups=8,
        head_size=128,
        rope_adjustments=None,
    )
    with torch.device("meta"):
        model = Model(cfg)
    assert model.layout == (0, 0) + (4, 128) * 17 == LAYOUT
    assert sum(p.numel() for p in model.parameters()) == 10_929_049_984


def test_causal_cache_and_reset():
    torch.manual_seed(5)
    model = tiny().eval()
    ids = torch.randint(79, (2, 145))
    with torch.no_grad():
        ref = model(ids)
        changed = ids.clone()
        changed[:, 75:] = (changed[:, 75:] + 1) % 79
        torch.testing.assert_close(model(changed)[:, :75], ref[:, :75], rtol=1e-5, atol=1e-5)
        model.set_kv_cache(2, 256, "cpu", torch.float32)
        for chunks in ([3, 1, 5, 7, 109, 2, 1, 16, 1], [1] * 145):
            offset, outputs = 0, []
            model.reset_kv_cache()
            for n in chunks:
                end = offset + n
                outputs.append(model(ids[:, offset:end], torch.arange(offset, end)))
                offset = end
            assert offset == 145
            torch.testing.assert_close(torch.cat(outputs, 1), ref, rtol=1e-5, atol=1e-5)
            assert all(b.attn.kv_cache.pos == 145 for b in model.transformer.h)


def test_training_signal():
    torch.manual_seed(4)
    model = tiny()
    ids = torch.randint(79, (2, 132))
    lm, kd, idx = model.training_loss(ids[:, :-1], ids[:, 1:], torch.randn(2, 131, 24))
    (lm + kd + idx).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.transformer.h[1].attn.iq.weight.grad.norm() > 0
    assert model.transformer.h[2].attn.compressor.gate.weight.grad.norm() > 0


def test_protocol():
    rows = stages("32k,64k,128k,256k,512k,1m", "", True, None, 100, "per_length")
    assert [len(row["benchmark"].split(",")) for row in rows[:2]] == [49, 20]
    assert len(rows) == 14
    assert rows[-1]["overrides"]["block_size"] == 1048576 + 1024
    assert rows[-1]["overrides"]["rope_adjustments"]["factor"] == 32
    assert "ruler_qa_squad" not in rows[-1]["benchmark"]
    assert "infinitebench" not in rows[-2]["benchmark"]


def test_litgpt_base_to_trained_checkpoint_and_eval(tmp_path, monkeypatch):
    """Exercise the same LitGPT weight names, distillation, save and scoring path on CPU."""
    import json
    from dataclasses import asdict
    import pytest
    import yaml

    pytest.importorskip("lightning")
    pytest.importorskip("lm_eval")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from deepseek4 import train
    from deepseek4._vendor.niahlogkv.litgpt import Config, GPT
    from deepseek4._vendor.niahlogkv import eval as evaluator

    config = Config.from_name(
        "Qwen3-8B-Base",
        n_embd=24,
        n_layer=3,
        n_head=4,
        head_size=6,
        n_query_groups=2,
        padded_vocab_size=79,
        vocab_size=79,
        intermediate_size=32,
        block_size=256,
        padding_multiple=1,
    )
    base, trained = tmp_path / "base", tmp_path / "trained"
    base.mkdir()
    trained.mkdir()
    for directory in (base, trained):
        (directory / "model_config.yaml").write_text(yaml.safe_dump(asdict(config)))
    Tokenizer(WordLevel({"[UNK]": 0, "x": 1, "[EOS]": 2}, unk_token="[UNK]")).save(str(base / "tokenizer.json"))
    (base / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "[EOS]", "bos_token": "[EOS]"})
    )
    torch.save(GPT(config).state_dict(), base / "lit_model.pth")
    g, layout = tiny().g, (0, 4, 128)
    monkeypatch.setattr(train, "Model", lambda config: Model(config, g, layout))
    monkeypatch.setattr(train, "LAYOUT", layout)
    model, teacher = train.load_models({"base_checkpoint_dir": base, "context_length": 256}, "cpu", False)
    ids = torch.randint(79, (1, 135))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with torch.no_grad():
            target = teacher(ids[:, :-1])
        losses = model.training_loss(ids[:, :-1], ids[:, 1:], target)
    sum(losses).backward()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    opt.step()
    assert all(".attn." in n for n, p in model.named_parameters() if p.requires_grad)
    train.save_model(model, trained, 0, False)
    monkeypatch.setattr(evaluator, "Model", lambda config: Model(config, g, layout))
    lm = evaluator.LogKVLM(str(trained), device="cpu", tokenizer_dir=str(base))
    score, _ = lm._score_tokens(ids[0, :130].tolist(), ids[0, 130:].tolist())
    assert float("-inf") < score < 0
    record = lm.compression_metrics[-1]
    assert record["processed_tokens"] == 135
    assert record["retained_slots"] == 3 * 8 + 135 // 4 + 135 // 128
    assert lm.cache_budget["persistent_cache_bytes"] > 0
    lm._generate_turns("x|x|x", "|", 2, "two-turn-smoke")
    turns = [r for r in lm.gen_metrics if r["request_type"] == "generate_turns"]
    assert [r["turn_index"] for r in turns] == [1, 2]
    assert lm.compression_metrics[-1]["turns"] == 2
