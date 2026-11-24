"""Size the reference cache from its buffers on CPU, or use measured evaluation JSON."""
import argparse
import json
from pathlib import Path

import torch

from .litgpt import Config
from .litgpt.model import CausalSelfAttention
from .long_eval_plan import LENGTHS


def word_first_bytes(cfg, window, B=128, recent=2048):
    # eval_full_word_first -> eval_long_word_first -> eval_gamma_surprisal_word_first_p1024
    # -> arc_gamma_k12_b128_p1024_2k -> arc_semantic_fast -> base. Every layer has the same cache.
    with torch.device("meta"):
        attn = CausalSelfAttention(cfg, 0)
    # The allocator requires RoPE geometry, but shared RoPE storage is excluded.
    rope = torch.empty(1, cfg.rope_n_elem, dtype=torch.bfloat16).expand(window, -1)
    cache = attn.build_log_kv_cache(1, window, device="cpu", dtype=torch.bfloat16, B=B, recent_size=recent,
        semantic_clusters=True, cluster_k_max=12, semantic_flush_granularity=recent,
        semantic_anchor_mode="mid", allocate_second_order=False, alpha_exact_tokens=1024,
        beta_novelty=True, beta_adaptive_merge=True, gamma_level0_reinsert=True, gamma_top_merge="lightest",
        gamma_level_slack=2, cos_cache=rope, sin_cache=rope)
    unique = {b.untyped_storage().data_ptr(): b.untyped_storage().nbytes() for name, b in cache.named_buffers()
              if name not in ("cos_cache", "sin_cache")}
    return cfg.n_layer * sum(unique.values())


def native_bytes(cfg, window):
    conv = (cfg.num_heads * cfg.head_dim + 2 * cfg.state_size * cfg.n_groups) * cfg.conv_kernel * 2
    matrix = cfg.num_heads * cfg.state_size * cfg.head_dim * window.bit_length() * 4
    return cfg.num_hidden_layers * (conv + matrix)


def main():
    from .model import HConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, help="actual LogKV JSON at the supplied --window")
    parser.add_argument("--window", type=int)
    parser.add_argument("--match-logkv", action="store_true",
                        help="size separate word_first comparison settings, B=2..128, recent/prefill/flush=512,1024,2048")
    args = parser.parse_args()
    native_cfg = HConfig.read(args.model_config, args.window or 32768)
    cfg = Config.from_name("Qwen3-8B-Base" if native_cfg.student else "Qwen3-1.7B-Base")
    if args.reference_json and not args.window:
        parser.error("--reference-json requires its matching --window")
    for window in ([args.window] if args.window else [32768, *(v + 1024 for v in LENGTHS.values())]):
        native = native_bytes(native_cfg, window)
        reference = (json.loads(args.reference_json.read_text())["cache_budget"]["persistent_cache_bytes"]
                     if args.reference_json else word_first_bytes(cfg, window))
        result = dict(window=window, native_persistent_bytes=native, reference_persistent_bytes=reference,
                      relative_gap=native / reference - 1)
        fixed = word_first_bytes(cfg, window, B=26, recent=512)
        result.update(fixed_logkv_B26_recent512_bytes=fixed, fixed_gap=native / fixed - 1)
        if args.match_logkv:
            sizes = []
            for recent in (512, 1024, 2048):
                for B in range(2, 129):
                    try:
                        sizes.append((word_first_bytes(cfg, window, B, recent), B, recent))
                    except ValueError as err:
                        if "budget is too small for alpha_exact_tokens" not in str(err):
                            raise
            closest, B, recent = min(sizes, key=lambda pair: abs(pair[0] - native))
            result.update(word_first_B=B, word_first_recent_prefill_flush=recent,
                          matched_logkv_bytes=closest, matched_gap=native / closest - 1)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
