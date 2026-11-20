import json
import os
import subprocess
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import yaml


@pytest.mark.parametrize("size", ["1.7b", "8b"])
def test_reference_plan(size):
    cfg = json.loads(subprocess.check_output(["bash", "majob.sh", f"configs/niah/qwen{size}.yaml", "--print-config"],
                                             env={**os.environ, "PYRAMIDKV_PYTHON": sys.executable}))
    rows = {r["folder"]: r for r in cfg["stages"]}
    assert cfg["weight_role"] == "base"
    assert cfg["checkpoint"] == cfg["reference_config"]["ckpt_dir"].rstrip("/") == cfg["tokenizer"]
    assert cfg["checkpoint"].endswith(f"/Qwen3-{size.upper()}-Base")
    assert len(rows) == 14
    for power, label in enumerate(("32k", "64k", "128k", "256k", "512k", "1m")):
        row = rows[f"long{label}/main"]
        assert row["config_overrides"]["block_size"] == (32768 << power) + 1024
        assert row["lengths"] == [32768 << power]
        assert ("infinitebench" in row["benchmark"]) == (power <= 3)
        assert ("ruler_qa_squad" in rows[f"long{label}/niah"]["benchmark"]) == (power <= 2)


def test_matched_cache_on_reference_requests(tmp_path, monkeypatch):
    pytest.importorskip("lm_eval")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from pyramidkv._vendor.niahlogkv import eval as ref
    from pyramidkv._vendor.niahlogkv.litgpt import Config, GPT
    from pyramidkv.niah_cache import install_backend

    torch.manual_seed(17)
    vocab = ["<unk>", "<bos>", "<eos>"] + [f"w{i}" for i in range(3, 64)]
    config = Config(name="tiny-qwen3", block_size=1024, vocab_size=64, padded_vocab_size=64,
                    n_layer=3, n_head=4, n_query_groups=2, n_embd=32, head_size=8, intermediate_size=64,
                    norm_qk=True, mlp_class_name="LLaMAMLP", norm_class_name="RMSNorm",
                    parallel_residual=False, bias=False)
    torch.save(GPT(config).state_dict(), tmp_path / "lit_model.pth")
    (tmp_path / "model_config.yaml").write_text(yaml.safe_dump(asdict(config)))
    tokenizer = Tokenizer(models.WordLevel(dict(zip(vocab, range(64))), unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(
        dict(bos_token="<bos>", eos_token="<eos>", add_bos_token=False, tokenizer_class="Fast")))
    kwargs = dict(device="cpu", config_overrides={"block_size": 1024}, tokenizer_dir=str(tmp_path),
                  log_kv_B=8, log_kv_recent_size=16, log_kv_prefill_block=32, log_kv_second_order_scale=0.)
    oracle = ref.LogKVLM(str(tmp_path), **kwargs)
    oracle._set_eval_cache()
    # Restore the evaluator module after exercising its public request methods.
    monkeypatch.setattr(ref, "LogKVLM", ref.LogKVLM)
    monkeypatch.setattr(ref, "cache_compression_snapshot", ref.cache_compression_snapshot)
    install_backend(ref, "pyramidkv", {}, dict(window_size=11, kernel_size=3, beta=12,
                                               prefill_chunk_size=16, pooling="avgpool", gqa_score_agg="max"))
    lm = ref.LogKVLM(str(tmp_path), **kwargs)
    lm._set_eval_cache()
    budget = lm.cache_budget
    assert 0 <= oracle.cache_budget["persistent_cache_bytes"] - budget["persistent_cache_bytes"] < 64
    assert budget["selection"]["window_size"] == 11 and budget["selection"]["pooling"] == "avgpool"
    assert budget["prefill_chunk"] == 16 and budget["selection"]["beta"] == 12
    assert budget["entries_per_layer"] == sorted(budget["entries_per_layer"], reverse=True)

    def storage():
        return {t.data_ptr(): t.untyped_storage().nbytes() for layer in lm.model.transformer.h
                for t in (layer.attn.pyramid_cache._k, layer.attn.pyramid_cache._v)}

    buffers = storage()
    assert sum(buffers.values()) == budget["persistent_cache_bytes"]
    prompt = " ".join(vocab[3:] * 12)
    request = SimpleNamespace(args=(prompt, {"max_gen_toks": 4, "do_sample": False}), task_name="test", doc_id=0)
    assert len(lm.generate_until([request])) == 1
    assert lm.compression_metrics[-1]["retained_slots"] < lm.compression_metrics[-1]["dense_slots"]
    assert len(lm.loglikelihood([SimpleNamespace(args=(prompt, " w4 w5"))])) == 1
    assert len(lm.loglikelihood_rolling([SimpleNamespace(args=(prompt,))])) == 1
    assert len(lm._generate_turns(prompt+"|w3|w5", "|", 2, "test").split("|")) == 2
    assert storage() == buffers
