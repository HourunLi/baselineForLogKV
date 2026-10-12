"""Generate a plain-text continuation from an exported DMS Base checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .check_env import check_environment
from .modeling_qwen3_dms import Qwen3ForCausalLMDMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local exported student_model directory")
    parser.add_argument("--prompt", required=True, help="Raw text; no chat template is applied")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--disable-eviction", action="store_true", help="Full-KV control with the same weights")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    config = json.loads((Path(args.model) / "config.json").read_text())
    if not config.get("dms_separate_alpha"):
        parser.error("Expected an exported DMS adapter model, not the original Qwen Base checkpoint")
    check_environment()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3ForCausalLMDMS.from_pretrained(
        args.model, dtype=torch.bfloat16, dms_disable_eviction=args.disable_eviction,
        dms_chunked_prefill=4096,
    ).to("cuda").eval()
    inputs = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    prompt_length = inputs.input_ids.shape[1]
    if not prompt_length or prompt_length + args.max_new_tokens > model.config.max_position_embeddings:
        parser.error("Prompt must be nonempty and prompt + output must fit max_position_embeddings")
    # A fresh cache is required for every independent prompt.
    cache = model.get_cache(preallocate_for_tokens=prompt_length + args.max_new_tokens)
    with torch.inference_mode():
        output = model.generate(
            **inputs, past_key_values=cache, max_new_tokens=args.max_new_tokens,
            do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True))
    print(json.dumps({"prompt_tokens": prompt_length, "generated_tokens": output.shape[1] - prompt_length,
                      "logical_kv_compression_ratio": cache.get_cr()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
