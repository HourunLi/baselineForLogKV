"""Real evaluator request smoke test; requires the optional evaluation dependencies."""

from dataclasses import asdict
import json
import math
from types import SimpleNamespace

import pytest
import torch
import yaml

pytest.importorskip("lm_eval")

from hybrid_csa import evaluate
from hybrid_csa._vendor.semanticlogkv.litgpt.config import Config
from hybrid_csa._vendor.semanticlogkv.litgpt.model import GPT
from hybrid_csa.checkpoint import save_adapter
from hybrid_csa.model import HybridCSAModel
from hybrid_csa.settings import HybridCSASettings


def test_trained_adapter_runs_all_reference_request_types(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    previous_dtype = torch.get_default_dtype()
    try:
        # The private vendor demo changes global default dtype on import.
        evaluator = evaluate.load_evaluator()
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(42)
        config = Config(name="tiny-qwen3", block_size=128, vocab_size=32, padded_vocab_size=32,
                        n_layer=3, n_head=2, n_query_groups=1, n_embd=16, head_size=8,
                        intermediate_size=32, norm_qk=True, mlp_class_name="LLaMAMLP",
                        norm_class_name="RMSNorm", parallel_residual=False, bias=False, rotary_percentage=1.0)
        base = tmp_path / "base"
        base.mkdir()
        gpt = GPT(config)
        torch.save(gpt.state_dict(), base / "lit_model.pth")
        (base / "model_config.yaml").write_text(yaml.safe_dump(asdict(config)))
        vocab = ["<unk>", "<bos>", "<eos>"] + [f"w{i}" for i in range(3, 32)]
        tokenizer = Tokenizer(models.WordLevel({s: i for i, s in enumerate(vocab)}, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.save(str(base / "tokenizer.json"))
        (base / "tokenizer_config.json").write_text(json.dumps(
            {"bos_token": "<bos>", "eos_token": "<eos>", "add_bos_token": False, "tokenizer_class": "Fast"}))
        settings = HybridCSASettings(window_size=5, index_topk=2, index_dim=4, index_heads=2,
                                     query_chunk_size=7, prefill_chunk_size=13)
        model = HybridCSAModel(gpt.bfloat16(), settings).train()
        tokens = torch.randint(3, 32, (1, 24))
        for stage in ("compressor", "indexer", "sparse"):
            model.set_stage(stage)
            optimizer = torch.optim.AdamW([p for _, p in model.adapter_named_parameters() if p.requires_grad], lr=1e-3)
            model.zero_grad(set_to_none=True)
            lm, kl = model.training_loss(tokens[:, :-1], tokens[:, 1:])
            (kl if stage == "indexer" else lm + kl).backward()
            optimizer.step()
        adapter = tmp_path / "sparse.pt"
        save_adapter(model, adapter, base, {"optimizer_steps": 3,
                                          "stage_steps": dict(compressor=1, indexer=1, sparse=1)})
        native = evaluator.LogKVLM
        evaluate.install_hybrid_csa_adapter(evaluator, settings.as_dict(), adapter)
        for method in ("loglikelihood", "loglikelihood_rolling", "generate_until"):
            assert getattr(native, method) is getattr(evaluator.LogKVLM, method)
        lm = evaluator.LogKVLM(str(base), device="cpu", config_overrides={"block_size": 128}, tokenizer_dir=str(base))
        words = lambda n: " ".join(f"w{3 + i % 29}" for i in range(n))
        text, = lm.generate_until([SimpleNamespace(args=(words(90), {"max_gen_toks": 5, "do_sample": False}),
                                                   task_name="niah", doc_id=0)])
        score, = lm.loglikelihood([SimpleNamespace(args=(words(80), " " + words(3)))])
        rolling, = lm.loglikelihood_rolling([SimpleNamespace(args=(words(300),))])
        assert isinstance(text, str) and math.isfinite(score[0]) and math.isfinite(rolling)
        assert lm.model.processed is None
        assert all(row["processed_tokens"] > 0 and "index_key_scalars" in row for row in lm.compression_metrics)
        summary = evaluator.summarize_compression(lm.compression_metrics)
        assert summary["kv_retention_ratio"] == pytest.approx(summary["retained_slots"] / summary["dense_slots"])
    finally:
        torch.set_default_dtype(previous_dtype)
