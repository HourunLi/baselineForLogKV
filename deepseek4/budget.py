"""Logical occupancy and actual buffer shapes at the six niahLogKV lengths."""

import torch

from .attention import Cache, Geometry, LAYOUT
from ._vendor.niahlogkv.long_eval_plan import LENGTHS


def budget(tokens):
    # Long-eval reserves another 1024 for generation. Occupancy uses forwarded
    # tokens; the allocated fraction denominator uses the full reserved window.
    g = Geometry()
    allocated = slots = 0
    for ratio in LAYOUT:
        cache = Cache(g, ratio, tokens + 1024, 1, "meta", torch.bfloat16)
        allocated += sum(t.numel() * t.element_size() for t in cache.buffers())
        slots += min(tokens, g.window) + (tokens // ratio if ratio else 0)
    dense = len(LAYOUT) * (tokens + 1024) * 8 * 128 * 2 * 2
    return len(LAYOUT) * tokens / slots, allocated, dense / allocated


if __name__ == "__main__":
    print("length\tlogical_factor\tallocated_cache_bytes\tqwen_dense_byte_factor")
    for name, length in LENGTHS.items():
        ratio, size, byte_factor = budget(length)
        print(f"{name}\t{ratio:.3f}\t{size}\t{byte_factor:.3f}")
