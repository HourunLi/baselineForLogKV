import json

import pytest
import torch

from lla_bench.model import HAttentionLM, HConfig
from lla_bench.metrics import cache_compression_snapshot
from lla_bench.student import base_identity, checkpoint_report, convert, export_checkpoint, load_teacher
from lla_bench.train_student import train
from lla_bench import prolong


def test_qwen_transfer_training_export_and_streaming(tmp_path):
    pytest.importorskip("litdata")
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from litdata.streaming.writer import BinaryWriter
    from litdata import TokensLoader

    torch.manual_seed(17)
    base = tmp_path / "base"
    qcfg = Qwen3Config(hidden_size=16, num_hidden_layers=2, intermediate_size=48, head_dim=8,
                       num_attention_heads=2, num_key_value_heads=1, vocab_size=32, tie_word_embeddings=False, sliding_window=None)
    Qwen3ForCausalLM(qcfg).save_pretrained(base, max_shard_size="10KB")
    tok = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(base)

    teacher = load_teacher(base)
    student = convert(teacher, 32)
    torch.testing.assert_close(student.backbone.layers[1].mlp.gate_proj.weight,
                               teacher.model.layers[1].mlp.gate_proj.weight)
    out = tmp_path / "convert"
    identity = base_identity(base)
    export_checkpoint(out, student.config, student.state_dict(), base,
                      dict(stage="convert", teacher_path=str(base), teacher_sha256=identity))
    cache = tmp_path / prolong.cache_name(base, 32)
    cache.mkdir()
    writer = BinaryWriter(str(cache), chunk_size=16, item_loader=TokensLoader(block_size=33))
    for i in range(16):
        writer.add_item(i, torch.randint(0, 32, (33,), dtype=torch.int32))
    writer.done()
    writer.merge()

    cfg = dict(context_length=32, seed=42, num_workers=0, distill=dict(steps=1, window=8,
               global_batch_size=2, lr=0.01, weight_decay=0.0), cpt=dict(steps=1, global_batch_size=1,
               lr=0.01, min_lr=0.01, weight_decay=0.0, warmup_steps=0))
    distilled = tmp_path / "distill"
    final = tmp_path / "cpt"
    train("distill", cfg, out, distilled, base, cache, torch.device("cpu"), 1, 0)
    train("cpt", cfg, distilled, final, base, cache, torch.device("cpu"), 1, 0)
    initial, aligned, tuned = [HAttentionLM.load(path, HConfig.read(path / "config.json", 67), "cpu",
                               dtype=torch.float32) for path in (out, distilled, final)]
    for name, param in aligned.named_parameters():
        if ".mixer." not in name:
            assert torch.equal(param, initial.get_parameter(name))
    assert not torch.equal(initial.backbone.layers[0].mixer.out_proj.weight, aligned.backbone.layers[0].mixer.out_proj.weight)
    assert not torch.equal(aligned.backbone.layers[1].mlp.down_proj.weight, tuned.backbone.layers[1].mlp.down_proj.weight)
    assert json.loads((distilled / "training.json").read_text())["input_tokens"] == 16
    assert json.loads((final / "training.json").read_text())["input_tokens"] == 32
    ids = torch.randint(32, (1, 67))
    with torch.no_grad():
        full = tuned(ids)
        tuned.set_cache()
        chunks = [tuned(ids[:, a:b], torch.arange(a, b)) for a, b in ((0, 31), (31, 64), (64, 67))]
        torch.testing.assert_close(torch.cat(chunks, 1), full, atol=1e-5, rtol=1e-4)
        assert cache_compression_snapshot(tuned, 67)["dense_kv_bytes"] == 67 * 2 * 1 * 8 * 4
    from huggingface_hub import save_torch_state_dict
    shards = tmp_path / "shards"
    shards.mkdir()
    save_torch_state_dict(tuned.state_dict(), shards, max_shard_size="10KB")
    assert (shards / "model.safetensors.index.json").is_file()
    reloaded = HAttentionLM.load(shards, tuned.config, "cpu", dtype=torch.float32)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(ids), full)
    (final / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="changed since export"):
        checkpoint_report(final)
    with pytest.raises(FileExistsError):
        export_checkpoint(final, student.config, student.state_dict(), base, {})


def test_reuse_requires_complete_matching_checkpoint(tmp_path):
    import os
    from lla_bench.train_student import reuse_checkpoint

    cfg = HConfig(16, 2, 1, 32, 1, 8, 8, 1, 4, 1e-6, False, True,
                  trained_levels=6, block_size=32, student=True, mlp_width=48)
    model = HAttentionLM(cfg)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}")
    out = tmp_path / "convert"
    expected = dict(stage="convert", seed=42, teacher_sha256={"teacher": "test"})
    export_checkpoint(out, cfg, model.state_dict(), tmp_path, expected)
    config = dict(context_length=32)
    assert reuse_checkpoint(out, config, expected)
    with pytest.raises(ValueError, match="seed differs"):
        reuse_checkpoint(out, config, {**expected, "seed": 123})
    weight = out / "model.safetensors"
    st = weight.stat()
    data = weight.read_bytes()
    weight.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    os.utime(weight, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(ValueError, match="asset changed"):
        reuse_checkpoint(out, config, expected)
    weight.unlink()
    with pytest.raises(ValueError, match="Cannot reuse"):
        reuse_checkpoint(out, config, expected)
