"""Theoretical element counts, independent of tensor allocations and hardware."""

import argparse
import json

from .settings import HybridCSASettings


def theoretical_budget(tokens, settings=None, *, n_layer=28, n_query_groups=8,
                       head_size=128, n_embd=2048):
    """Batch-one Qwen cache after `tokens` have actually been forwarded.

    Main KV counts equal-width K/V pairs. Index keys and compressor working state
    are separate scalar counts; no dtype, allocator slack, or GPU-memory figures.
    """
    s = settings or HybridCSASettings()
    if tokens < 0 or min(n_layer, n_query_groups, head_size, n_embd) < 1:
        raise ValueError("tokens must be nonnegative and model dimensions positive")
    full_layers = list(s.full_layers(n_layer))
    full, csa = len(full_layers), n_layer - len(full_layers)
    closed, local = tokens // 4, min(tokens, s.window_size)
    # Previous complete block + current incomplete block, at most seven tokens.
    tail = tokens - max(0, (tokens // 4 - 1) * 4)
    dense = n_layer * tokens * n_query_groups
    exact = full * tokens * n_query_groups
    recent = csa * local * n_query_groups
    compressed = csa * closed * n_query_groups
    retained = exact + recent + compressed
    pair_width = 2 * head_size
    index_scalars = csa * closed * s.index_dim
    state_scalars = csa * tail * (n_embd + n_query_groups * pair_width)
    dense_scalars, kv_scalars = dense * pair_width, retained * pair_width
    return {
        "basis": "theoretical elements, batch=1; no allocated bytes or peak VRAM",
        "processed_tokens": tokens, "full_layers_zero_based": full_layers,
        "full_layer_count": full, "csa_layer_count": csa,
        "layer_group_count": n_layer * n_query_groups,
        "dense_slots": dense, "exact_slots": exact, "recent_slots": recent,
        "hierarchical_slots": compressed, "retained_slots": retained,
        "kv_retention_ratio": retained / dense if dense else None,
        "kv_saving_ratio": 1 - retained / dense if dense else None,
        "kv_compression_factor": dense / retained if retained else None,
        "asymptotic_kv_retention_ratio": (full + csa / 4) / n_layer,
        "dense_kv_scalars": dense_scalars, "main_kv_scalars": kv_scalars,
        "index_key_scalars": index_scalars, "compressor_state_scalars": state_scalars,
        "kv_plus_index_retention_ratio": (kv_scalars + index_scalars) / dense_scalars if dense_scalars else None,
        "all_state_retention_ratio": (kv_scalars + index_scalars + state_scalars) / dense_scalars if dense_scalars else None,
        "compressed_slots_per_csa_layer": closed, "local_slots_per_csa_layer": local,
        "compressor_tail_tokens_per_csa_layer": tail,
        "index_topk": s.index_topk,
        "index_topk_note": "read budget only; all closed compressed blocks remain stored",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--config", help="Project-owned CSA settings YAML; model geometry is Qwen3-1.7B")
    args = parser.parse_args()
    settings = None
    if args.config:
        from .evaluate import load_hybrid_csa_options
        settings = HybridCSASettings(**load_hybrid_csa_options(args.config))
    print(json.dumps(theoretical_budget(args.tokens, settings), indent=2))


if __name__ == "__main__":
    main()
