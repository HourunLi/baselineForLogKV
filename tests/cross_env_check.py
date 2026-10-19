"""Check that a DMS student exported in the training environment evaluates identically in torch218.

Training runs with the upstream transformers 4.57.3 in its own environment; evaluation runs in
SemanticLogKV's environment (torch218, transformers 4.51.3) through dms.transformers_compat. Run each
step with that environment's interpreter, from the project root, on CPU (no CUDA kernels needed):

    <training python>   -m tests.cross_env_check export DIR
    <training python>   -m tests.cross_env_check run DIR train.json
    <evaluation python> -m tests.cross_env_check run DIR eval.json
    <either python>     -m tests.cross_env_check compare train.json eval.json

``export`` saves a tiny random student with non-zero adapters (so tokens are evicted) through the
project's own export (models.qwen3.train.extract_student_model). ``run`` loads it the way the evaluator
does, then scores a chunked prefill and greedy decoding through the evaluator's model interface
(DMSModel.forward, kv_compression_snapshot), with the reference FlashAttention replacement from the
upstream tests and uncompiled FlexAttention. ``compare`` requires identical logits, tokens and KV counts.
"""

import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

PROMPT_TOKENS, CHUNK, DECODE_STEPS = 96, 40, 12


def tiny_student():
    from models.qwen3.configuration_qwen3_dms import Qwen3ConfigDMS
    from models.qwen3.modeling_qwen3_dms import Qwen3ForCausalLMDMS

    torch.manual_seed(0)
    config = Qwen3ConfigDMS(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, max_position_embeddings=2048, tie_word_embeddings=True,
        bos_token_id=0, eos_token_id=1, pad_token_id=0, dms_separate_alpha=True, dms_window_size=32,
        dms_paged_attention_block_size=16, dms_preallocate_for_tokens=256)
    model = Qwen3ForCausalLMDMS(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)
        for layer in model.model.layers:  # eviction decisions follow the adapter logits
            layer.self_attn.dms_proj_alpha.weight.normal_(0.0, 1.0)
    model.config.semanticlogkv_base = {"source_weight_sha256": "cross-env-check"}
    return model


def tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({f"w{i}": i for i in range(64)}, unk_token="w2"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="w1", pad_token="w0", unk_token="w2")


def export(directory: Path):
    from models.qwen3.train import extract_student_model

    extract_student_model(SimpleNamespace(student_model=tiny_student()), tokenizer(), str(directory))


def load(directory: Path):
    """The evaluator's loading steps (DMSModel.from_checkpoint) without the CUDA and Base-identity checks."""
    import dms.attention as attention
    from dms.attention_prefill import wrapped_flex_attention
    from models.qwen3.configuration_qwen3_dms import Qwen3ConfigDMS
    from models.qwen3.modeling_qwen3_dms import Qwen3ForCausalLMDMS
    from models.qwen3.semanticlogkv_eval import DMSModel
    from tests.upstream.test_prefill_and_generate import fake_flash_attn_with_kvcache

    # CPU stand-ins for the CUDA kernels, as in the upstream reference tests.
    function = attention.dms_attn_eval_mode
    defaults = {name: parameter.default for name, parameter in inspect.signature(function).parameters.items()
                if parameter.default is not inspect.Parameter.empty}
    defaults.update(flash_attn_fn=fake_flash_attn_with_kvcache,
                    prefill_attn_fn_kwargs={"flex_attention_fn": wrapped_flex_attention})
    function.__defaults__ = tuple(defaults.values())
    config = Qwen3ConfigDMS.from_pretrained(directory, local_files_only=True)
    assert config.dms_separate_alpha and not config.dms_disable_eviction
    model, loading = Qwen3ForCausalLMDMS.from_pretrained(directory, config=config, torch_dtype=torch.bfloat16,
                                                         local_files_only=True, output_loading_info=True)
    if loading["missing_keys"] or loading["unexpected_keys"] or loading["mismatched_keys"]:
        raise ValueError(f"Incomplete DMS checkpoint: {loading}")
    assert model.lm_head.weight is model.model.embed_tokens.weight, "lm_head is not tied to the embeddings"
    model.config.dms_chunked_prefill = CHUNK
    return DMSModel(model.eval(), max_seq_length=2048)


@torch.no_grad()
def run(directory: Path, output: Path):
    model = load(directory)
    torch.manual_seed(1)
    prompt = torch.randint(2, 64, (1, PROMPT_TOKENS))
    model.reset_kv_cache()
    record = {"versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
              "steps": []}
    logits = model(prompt, input_pos=torch.arange(PROMPT_TOKENS), lm_head_start=0)
    record["prefill_logits"] = logits.float().tolist()  # every prompt position, as for loglikelihood scoring
    token, forwarded = logits[0, -1].argmax().item(), PROMPT_TOKENS
    record["steps"].append({"token": token, **model.kv_compression_snapshot(forwarded)})
    for _ in range(DECODE_STEPS):
        logits = model(torch.tensor([[token]]), input_pos=torch.tensor([forwarded]),
                       lm_head_start=0)
        forwarded += 1
        token = logits[0, -1].argmax().item()
        record["steps"].append({"token": token, "logits": logits[0, -1].float().tolist(),
                                **model.kv_compression_snapshot(forwarded)})
    output.write_text(json.dumps(record))
    last = record["steps"][-1]
    print(f"{record['versions']}: {forwarded} tokens forwarded, retained {last['retained_slots']} of "
          f"{last['dense_slots']} KV entries, tokens {[step['token'] for step in record['steps']]}")


def compare(first: Path, second: Path):
    a, b = (json.loads(path.read_text()) for path in (first, second))
    print("versions:", a["versions"], "vs", b["versions"])
    differences = []
    if a["prefill_logits"] != b["prefill_logits"]:
        differences.append("prefill logits")
    for index, (x, y) in enumerate(zip(a["steps"], b["steps"], strict=True)):
        for key in sorted(set(x) | set(y)):
            if x.get(key) != y.get(key):
                differences.append(f"step {index} {key}")
    retained = [step["retained_slots"] for step in a["steps"]]
    dense = [step["dense_slots"] for step in a["steps"]]
    print(f"retained KV entries per step {retained} of {dense}")
    if differences:
        sys.exit("DIFFERENT: " + ", ".join(differences))
    print("IDENTICAL: prefill logits, decode logits and tokens, and KV counts at every step")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("export", "run", "compare"))
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    if args.action == "export":
        export(*args.paths)
    elif args.action == "run":
        run(*args.paths)
    else:
        compare(*args.paths)


if __name__ == "__main__":
    main()
