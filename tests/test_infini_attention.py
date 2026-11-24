import json

import pytest
import torch
from litdata import TokensLoader
from litdata.streaming.writer import BinaryWriter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from infini_attention.evaluation import cache_budget, snapshot
from infini_attention.model import InfiniLM
from infini_attention.train import train


@pytest.fixture
def base(tmp_path):
    torch.manual_seed(11)
    torch.set_num_threads(1)
    folder = tmp_path / "base"
    cfg = Qwen3Config(hidden_size=32, num_hidden_layers=2, intermediate_size=48, head_dim=8,
                      num_attention_heads=4, num_key_value_heads=1, vocab_size=32,
                      max_position_embeddings=32, sliding_window=None)
    Qwen3ForCausalLM(cfg).save_pretrained(folder, max_shard_size="18KB")
    tok = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "tea": 2, "red": 3}, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>",
                            bos_token="<eos>").save_pretrained(folder)
    return folder


@torch.no_grad()
def test_transfer_causality_and_memory(base):
    dense = Qwen3ForCausalLM.from_pretrained(base).eval()
    model = InfiniLM.load_local(base, base=True, segment=4, dtype=torch.float32).eval()
    ids = torch.randint(32, (2, 17))
    for name, param in dense.named_parameters():
        torch.testing.assert_close(param, model.get_parameter(name), rtol=0, atol=0)
    for layer in model.model.layers:
        layer.memory_gate.fill_(-1000)
    torch.testing.assert_close(model(ids[:, :4]), dense(ids[:, :4]).logits)
    for layer in model.model.layers:
        layer.memory_gate.zero_()
    full = model(ids)
    changed = ids.clone()
    changed[:, 7:] = (changed[:, 7:] + 3) % 32
    torch.testing.assert_close(model(changed)[:, :7], full[:, :7])
    model.set_cache(batch=2)
    budget = cache_budget(model)
    outputs = []
    for a, b in ((0, 2), (2, 3), (3, 4), (4, 11), (11, 16), (16, 17)):
        outputs.append(model(ids[:, a:b], torch.arange(a, b)))
        assert cache_budget(model) == budget
        assert snapshot(model, b)["local_kv_slots"] == 4 * (b % 4)
        if b == 4:
            # Independent segment commit from dense Qwen K/V projections.
            layer = dense.model.layers[0]
            x = layer.input_layernorm(dense.model.embed_tokens(ids[:, :4]))
            cos, sin = dense.model.rotary_emb(x, torch.arange(4)[None])
            k = layer.self_attn.k_norm(layer.self_attn.k_proj(x).view(2, 4, 1, 8).transpose(1, 2))
            a1, a2 = k.chunk(2, dim=-1)
            k = k * cos[:, None] + torch.cat((-a2, a1), -1) * sin[:, None]
            v = layer.self_attn.v_proj(x).view(2, 4, 1, 8).transpose(1, 2)
            phi = torch.nn.functional.elu(k) + 1
            cache = model.model.layers[0].cache
            torch.testing.assert_close(cache.m, torch.einsum("bhtk,bhtv->bhkv", phi, v))
            torch.testing.assert_close(cache.z[..., 0], phi.sum(2))
    torch.testing.assert_close(torch.cat(outputs, dim=1), full)
    model.reset_cache()
    torch.testing.assert_close(model(ids[:, :3], torch.arange(3)), full[:, :3])


def test_cpt_resume_export_and_harness(base, tmp_path):
    from lla_bench.eval import NativeLM
    from lm_eval.api.instance import Instance

    data = tmp_path / "data"
    data.mkdir()
    length = 16
    writer = BinaryWriter(str(data), chunk_size=32, item_loader=TokensLoader(block_size=length + 1))
    for i in range(20):
        writer.add_item(i, torch.randint(32, (length + 1,), dtype=torch.int32))
    writer.done()
    writer.merge()
    cfg = dict(context_length=length, segment_length=4, seed=42, num_workers=0, steps=2,
               global_batch_size=2, micro_batch_size=1, lr=0.001, min_lr=0.0006,
               warmup_steps=1, weight_decay=0.1, save_interval=1)
    def run(root, **kwargs):
        return train(cfg, base, data, root, torch.device("cpu"), require_8b=False, **kwargs)

    full = run(tmp_path / "full")
    root = tmp_path / "resume"
    run(root, stop_after=1)
    final = run(root, resume=root / "step-1")
    report = json.loads((final / "training.json").read_text())
    assert report["input_tokens"] == 64 and report["resume_step"] == 1
    a, b = [InfiniLM.load_local(d, dtype=torch.float32) for d in (full, final)]
    for (name, x), y in zip(a.named_parameters(), b.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0, msg=name)
    assert torch.count_nonzero(a.model.layers[0].memory_gate)
    assert not torch.equal(a.model.layers[0].self_attn.q_proj.weight,
                           Qwen3ForCausalLM.from_pretrained(base).model.layers[0].self_attn.q_proj.weight)
    harness = NativeLM(final, device="cpu", config_overrides={"block_size": 64, "rope_adjustments": {
        "original_max_seq_len": 32, "factor": 2, "beta_fast": 32, "beta_slow": 1}})
    def req(kind, args):
        return Instance(request_type=kind, doc={}, arguments=args, idx=0)
    context = "tea " * 8
    assert len(harness.loglikelihood([req("loglikelihood", (context, " red"))])) == 1
    assert len(harness.loglikelihood_rolling([req("loglikelihood_rolling", (context,))])) == 1
    assert len(harness.generate_until([req("generate_until", (context, {"max_gen_toks": 2, "until": []}))])) == 1


def test_formal_recipe_and_passes():
    from pathlib import Path
    import yaml
    from infini_attention.long_eval_plan import plan, SUITES
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/infini_cpt.yaml").read_text())
    assert cfg["steps"] * cfg["global_batch_size"] * cfg["context_length"] == 8388608000
    assert cfg["global_batch_size"] // (32 * cfg["micro_batch_size"]) == 4
    passes = plan("32k,64k,128k,256k,512k,1m", ",".join(SUITES))
    for p in passes:
        assert p["config_overrides"]["block_size"] == p["length"] + 1024
        assert bool(p["config_overrides"].get("rope_adjustments")) == (p["length"] > 32768)
        assert ("ruler_qa_squad" in p["niah"]) == (p["length"] <= 131072)
        assert any(name.startswith("infinitebench_") for name in p["main"]) == (p["length"] <= 262144)


def test_needle_depth_counts_actual_tokens(base):
    from infini_attention.depth import depth_report
    tokenizer = Tokenizer.from_file(str(base / "tokenizer.json"))
    prompt = "tea One of the special magic numbers for tea is: 77. " + "red " * 120
    n = len(tokenizer.encode(prompt))
    task = "niah_single_1"
    rows, requests = [], []
    for i, cut in enumerate([0, 20]):
        rows.append(dict(doc_id=i, doc={"max_length": 1048576, "outputs": ["77"]},
                         arguments=[[prompt, {}]], prompt_hash="test", filter="none", **{"1048576": 1-i}))
        requests.append(dict(sample_id=f"rank0|global_req{i}|task={task}|doc={i}", original_prompt_len=n,
                             truncated_prompt_tokens=cut, prompt_len=n-cut))
    report = depth_report({"results": {"samples": {task: rows}},
                           "inference_metrics": {"generate_until": requests}}, tokenizer)
    buckets = report["by_task"][task]
    assert buckets["first_10_pct"] == buckets["first_third"] == {"n": 1, "correct": 1}
    assert buckets["needle_truncated"] == {"n": 1, "correct": 0}
