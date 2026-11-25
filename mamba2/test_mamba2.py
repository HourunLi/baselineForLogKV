"""CPU regression checks for the recurrence, checkpoint preparation and LogKV requests."""
import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from litgpt.config import Config
from litgpt.model import GPT
from mamba_model import MambaLM, ssd_scan
from prepare import convert

torch.set_num_threads(1)


def tiny():
    return Config.from_name("Qwen/Qwen3-8B-Base", n_embd=16, n_layer=2, n_head=4, n_query_groups=2,
                            padded_vocab_size=16, vocab_size=16, intermediate_size=32, block_size=128,
                            mamba2=True, mamba_inner=16, mamba_head_dim=4, mamba_groups=2, mamba_state_dim=3)


@pytest.mark.parametrize("length", [1, 67, 537])
def test_ssd_against_token_recurrence(length):
    torch.manual_seed(7)
    x = torch.randn(2, 4, length, 5, requires_grad=True)
    b, c = (torch.randn(2, 2, length, 3, requires_grad=True) for _ in range(2))
    dt = torch.rand(2, 4, length, requires_grad=True)
    a = -torch.rand(4, requires_grad=True)
    initial = torch.randn(2, 4, 3, 5, requires_grad=True)
    actual, final = ssd_scan(x, b, c, dt, a, initial)
    state, outputs = initial, []
    for i in range(length):
        step = dt[:, :, i, None, None]
        key = b[:, :, i].repeat_interleave(2, dim=1).unsqueeze(-1)
        query = c[:, :, i].repeat_interleave(2, dim=1).unsqueeze(-2)
        state = state * (a[None, :, None, None] * step).exp() + key * (step * x[:, :, i:i+1])
        outputs.append(query @ state)
    ref = torch.cat(outputs, dim=2)
    torch.testing.assert_close(actual, ref, atol=8e-5, rtol=2e-5)
    torch.testing.assert_close(final, state, atol=2e-5, rtol=2e-5)
    args = (x, b, c, dt, a, initial)
    def gradients(out, carry):
        return torch.autograd.grad(out.square().mean() + carry.mean(), args)
    for got, want in zip(gradients(actual, final), gradients(ref, state)):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=3e-4)


def test_state_reset_chunks_and_cpt():
    torch.manual_seed(8)
    model = MambaLM(tiny())
    tokens = torch.randint(16, size=(2, 81))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loss = model(tokens, targets=tokens.roll(-1, 1), loss_chunk_size=9)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    model.eval()
    with torch.no_grad():
        want = model(tokens)
        for cuts in ([1, 63, 17], [33, 47, 1]):
            model.reset_cache(2)
            start, rows = 0, []
            budget = model.cache_budget()
            for length in cuts:
                end = start + length
                rows.append(model(tokens[:, start:end], input_pos=torch.arange(start,end), input_pos_maxp1=end))
                start = end
            torch.testing.assert_close(torch.cat(rows, dim=1), want, atol=1e-5, rtol=1e-5)
            assert model.cache_budget() == budget
            assert model.position == tokens.size(1)


def test_warm_start_and_harness_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGKV_EVAL_PREFILL_CHUNK", "3")
    from eval import MambaEval, summarize_inference_metrics
    base, out = tmp_path / "base", tmp_path / "converted"
    base.mkdir()
    tokenizer = Tokenizer(WordLevel({"?": 0, "bos": 1, "eos": 2, "word": 3, "answer": 4}, unk_token="?"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(base/"tokenizer.json"))
    (base/"tokenizer_config.json").write_text(json.dumps({"bos_token": "bos", "eos_token": "eos", "tokenizer_class": "PreTrainedTokenizerFast"}))
    cfg = tiny()
    dense = GPT(cfg).state_dict()
    torch.save(dense, base/"lit_model.pth")
    report = convert(cfg, base, out, 42)
    student = torch.load(out/"lit_model.pth", weights_only=True)
    copied = {k for k in student if ".mixer." not in k}
    assert all(torch.equal(student[k], dense[k]) for k in copied)
    assert report["copied_parameters"] == sum(student[k].numel() for k in copied)
    assert not any(".attn." in k for k in student)
    with pytest.raises(FileExistsError):
        convert(cfg, base, out, 42)
    lm = MambaEval(str(out), "cpu", str(base), {"block_size": 12}, None)
    def req(*args):
        return SimpleNamespace(args=args)
    ctx, cont = [3]*5, [4]*6
    with torch.no_grad():
        logits = lm.model(torch.tensor([ctx + cont]))[0, len(ctx)-1:-1].float()
        want = logits.log_softmax(-1)[:, 4].sum().item()
    ll, greedy = lm.loglikelihood([req("word "*5, "answer "*6)])[0]
    assert ll == pytest.approx(want, abs=.04)
    assert greedy == (logits.argmax(-1) == 4).all().item()
    assert len(lm.loglikelihood_rolling([req("word "*25)])) == 1
    assert [r["total_seq_len"] for r in lm.ppl_metrics[1:]] == [12, 12, 4]
    gen = lm.generate_until([req("word "*15, {"max_gen_toks": 3, "until": ["answer"]})])
    assert len(gen) == 1 and "answer" not in gen[0]
    assert lm.gen_metrics[0]["prompt_len"] == 9
    assert lm.gen_metrics[0]["truncated_prompt_tokens"] == 6
    turns = lm.generate_until([req("word|answer|word", {"turn_separator":"|", "max_gen_toks":2})])
    assert len(turns[0].split("|")) == 2
    assert [r["turn_index"] for r in lm.gen_metrics[1:]] == [1,2]
    assert lm.model.position == 0
    assert all(row["kv_retention_ratio"] is None for row in lm.compression_metrics)
    metrics = summarize_inference_metrics(lm.gen_metrics, lm.ppl_metrics)
    assert metrics["generation"]["requests"] == 3 and metrics["scoring"]["requests"] == 4


def test_hf_shards_map_to_litgpt_weights(tmp_path):
    from safetensors.torch import save_file
    from prepare import hf_names, shared_weights, specifications
    config = tiny()
    dense, expected = specifications(config)
    tensors = {src: torch.randn(dense[dst].shape) for src, dst in hf_names(config).items()}
    files = {k: f"model-{i % 2}.safetensors" for i, k in enumerate(tensors)}
    for filename in set(files.values()):
        save_file({k:v for k,v in tensors.items() if files[k] == filename}, tmp_path/filename)
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map": files}))
    (tmp_path/"config.json").write_text(json.dumps({
        "model_type": "qwen3", "hidden_size":16, "intermediate_size":32, "num_hidden_layers":2,
        "num_attention_heads":4, "num_key_value_heads":2, "tie_word_embeddings":False, "vocab_size":16,
    }))
    result = shared_weights(config, tmp_path, dense, expected)
    assert set(result) == {name for name in expected if ".mixer." not in name}
    for src, dst in hf_names(config).items():
        assert torch.equal(tensors[src], result[dst])


def test_mixer_matches_transformers_451_reference():
    from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer as Reference
    from mamba_model import Mamba2Mixer

    torch.manual_seed(23)
    cfg = tiny()
    cfg.mamba_groups = 1  # HF 4.51.3 implements whole-width gated RMSNorm.
    ours = Mamba2Mixer(cfg).eval()
    ref_cfg = Mamba2Config(hidden_size=16, expand=1, num_heads=4, head_dim=4,
                           state_size=3, n_groups=1, chunk_size=16, layer_norm_epsilon=1e-5)
    reference = Reference(ref_cfg, layer_idx=0).eval()
    mapped = {k.replace("conv.", "conv1d.").replace("norm_weight", "norm.weight"): v
              for k,v in ours.state_dict().items()}
    reference.load_state_dict(mapped, strict=True)
    x = torch.randn(1, 73, 16)
    with torch.no_grad():
        torch.testing.assert_close(ours(x), reference.torch_forward(x), atol=2e-5, rtol=2e-5)


def test_long_suite_registration_and_protocol():
    from pathlib import Path
    from long_eval_plan import LENGTHS, plan

    tasks = Path(__file__).parent / "eval_tasks"
    stages = plan(",".join(LENGTHS))
    assert [s["length"] for s in stages] == [32768, 65536, 131072, 262144, 524288, 1048576]
    for s in stages:
        length, name = s["length"], s["name"]
        assert s["config_overrides"]["block_size"] == length + 1024
        assert s["ruler_lengths"] == [length]
        suites = {"babilong", "long_ppl", "long_ppl_books", "niah_turns", "longbench2"}
        if length <= 262144:
            suites.add("infinitebench")
        assert set(s["main"]) == {f"{suite}_{name}" for suite in suites}
        assert all(list(tasks.glob(f"*/{t}.yaml")) for t in s["main"])
        assert ("ruler_qa_squad" in s["niah"]) == (length <= 131072)
    with pytest.raises(ValueError, match="InfiniteBench"):
        plan("512k,1m", "infinitebench")
