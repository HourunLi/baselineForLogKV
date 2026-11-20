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
                                             env={**os.environ, "SNAPKV_PYTHON": sys.executable}))
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
    from snapkv._vendor.niahlogkv import eval as ref
    from snapkv._vendor.niahlogkv.litgpt import Config, GPT
    from snapkv.niah_cache import install_backend

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
    install_backend(ref, "snapkv", {}, dict(window_size=11, kernel_size=3,
                                               prefill_chunk_size=16, pooling="avgpool", gqa_score_agg="max"))
    lm = ref.LogKVLM(str(tmp_path), **kwargs)
    lm._set_eval_cache()
    budget = lm.cache_budget
    assert 0 <= oracle.cache_budget["persistent_cache_bytes"] - budget["persistent_cache_bytes"] < 64 * config.n_layer
    assert budget["selection"]["window_size"] == 11 and budget["selection"]["pooling"] == "avgpool"
    assert budget["prefill_chunk"] == 16
    assert len(set(budget["entries_per_layer"])) == 1

    def storage():
        return {t.data_ptr(): t.untyped_storage().nbytes() for layer in lm.model.transformer.h
                for t in (layer.attn.snap_cache._k, layer.attn.snap_cache._v)}

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


@pytest.mark.parametrize("new_tokens", [1, 3, 8, 16])
def test_streaming_scores_for_short_and_full_blocks(new_tokens):
    """Use dense repeated heads and an explicit causal mask as the oracle."""
    from snapkv.compressor import observation_window_scores

    rng = torch.Generator().manual_seed(41)
    q = torch.randn(2, 6, new_tokens, 16, generator=rng)
    k = torch.randn(2, 2, 39, 16, generator=rng)
    n = min(new_tokens, 8)
    logits = q[:, :, -n:] @ k.repeat_interleave(3, dim=1).transpose(-2, -1) / 4
    hidden = torch.arange(39)[None, :] > torch.arange(39 - n, 39)[:, None]
    scores = logits.masked_fill(hidden, -torch.inf).softmax(-1)[..., :-8].sum(-2)
    expected = scores.reshape(2, 2, 3, 31).mean(dim=2)
    expected = torch.nn.functional.max_pool1d(expected, 7, stride=1, padding=3)
    torch.testing.assert_close(observation_window_scores(q, k, 8, 7, "maxpool", "mean"), expected)
