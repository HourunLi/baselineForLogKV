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
    lm, idx = model(ids[:, :-1], targets=ids[:, 1:], loss_chunk_size=64)
    (lm + idx).backward()
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
    """Exercise the same LitGPT weight names, full-model CPT, save and scoring path on CPU."""
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
    model = train.load_model({"base_checkpoint_dir": base, "context_length": 256}, "cpu", False)
    ids = torch.randint(79, (1, 135))
    losses = model(ids[:, :-1], targets=ids[:, 1:], loss_chunk_size=64)
    sum(losses).backward()
    opt = train.optimizer(model, {"learning_rate": 5e-6, "weight_decay": 0.1})
    opt.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
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


def test_strict_full_plan():
    from deepseek4.evaluate import full_stages

    rows = full_stages({"long_limit": 100})
    assert len(rows) == 14
    assert all(row["optional"] == [] for row in rows)
    assert len(rows[0]["benchmark"].split(",")) == 49
    assert len(rows[1]["benchmark"].split(",")) == 20
    assert rows[0]["limit"] is None
    assert {"long_ppl_1m", "long_ppl_books_1m", "niah_turns_1m", "longbench2_1m", "babilong_1m"} == set(
        rows[-2]["benchmark"].split(",")
    )
    assert "infinitebench_256k" in rows[8]["benchmark"].split(",")


def test_platform_rendezvous(monkeypatch):
    from deepseek4.evaluate import launch_args

    monkeypatch.setenv("MA_NUM_HOSTS", "2")
    monkeypatch.setenv("MA_VJ_NAME", "job")
    monkeypatch.setenv("MA_TASK_NAME", "worker")
    monkeypatch.setenv("VC_TASK_INDEX", "1")
    monkeypatch.setenv("MASTER_PORT", "6000")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    args = launch_args(8, 3)
    assert args[args.index("--master_addr") + 1] == "job-worker-0.job"
    assert args[args.index("--master_port") + 1] == "6003"


def test_all_handoff_and_stop(tmp_path, monkeypatch):
    import pytest
    from deepseek4 import pipeline

    path = tmp_path / "experiment.yaml"
    output = tmp_path / "trained"
    monkeypatch.setattr(pipeline, "config", lambda _: {"output_dir": str(output), "long_limit": 100})
    monkeypatch.setattr(pipeline, "gpu_count", lambda: 2)
    for name in ("LONG_PPL_DATA", "LONG_PPL_BOOKS"):
        monkeypatch.setenv(name, str(tmp_path / name))
    calls = []

    def run(cmd, check):
        assert check
        calls.append(cmd)

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    pipeline.run("all", path)
    assert len(calls) == 4
    assert "deepseek4.preflight" in calls[0]
    assert "--prepare" in calls[1]
    assert "deepseek4.train" in calls[2] and "--prepare" not in calls[2]
    assert calls[3][-3:] == ["--full", "--checkpoint", str(output)]

    def fail(cmd, check):
        run(cmd, check)
        if "deepseek4.train" in cmd and "--prepare" not in cmd:
            raise RuntimeError("train failed")

    calls.clear()
    monkeypatch.setattr(pipeline.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="train failed"):
        pipeline.run("all", path)
    assert len(calls) == 3


def test_reference_training_budget():
    import pytest
    from deepseek4 import train

    cfg = train.config(train.DEFAULT)
    assert train.accumulation_steps(cfg, 8) == 16
    assert cfg["steps"] * cfg["global_batch_size"] * cfg["context_length"] == 8_388_608_000
    assert train.learning_rate(cfg, 0) == pytest.approx(5e-8)
    assert train.learning_rate(cfg, 99) == train.learning_rate(cfg, 100) == 5e-6
    assert train.learning_rate(cfg, 1999) == pytest.approx(3.000001367e-6, abs=1e-15)
    with pytest.raises(ValueError):
        train.accumulation_steps(cfg, 3)


def test_reference_chunks_and_resume(tmp_path):
    import pytest

    litdata = pytest.importorskip("litdata")
    from litdata.streaming.cache import Cache
    from deepseek4.data import make_loader, reference_chunks

    (tmp_path / "tokenizer.json").write_bytes(b"test tokenizer")
    cfg = dict(
        base_checkpoint_dir=tmp_path,
        data_dir=tmp_path,
        tokenized_dir=None,
        training_data="niahlogkv_chunks",
        seed=42,
        context_length=8,
        micro_batch_size=1,
        num_workers=0,
    )
    directory = reference_chunks(cfg)
    directory.mkdir()
    writer = Cache(str(directory), chunk_size=36, item_loader=litdata.TokensLoader(block_size=9))
    for i in range(32):
        writer[i] = torch.arange(i * 9, i * 9 + 9)
    writer.done()
    writer.merge()
    loader, record = make_loader(cfg)
    assert record["path"] == str(directory)
    batches = iter(loader)
    first = next(batches)
    next(batches)
    resumed, _ = make_loader(cfg)
    resumed.load_state_dict(loader.state_dict())
    torch.testing.assert_close(next(iter(resumed)), next(batches))

    reference = litdata.StreamingDataset(
        str(directory), item_loader=litdata.TokensLoader(block_size=9), shuffle=True, seed=42
    )
    reference_loader = litdata.StreamingDataLoader(
        reference, batch_size=1, num_workers=0, pin_memory=True, drop_last=True
    )
    torch.testing.assert_close(next(iter(reference_loader)), first)
