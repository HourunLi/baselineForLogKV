"""Pack a detached mid-anchor prefix and differentiable current K/V for SDPA.

Only the current chunk carries gradients in LogKV's streaming objective. The
backward is therefore a slice, not a replay of gather/RoPE/copy operations.
Triton is optional and imported only on CUDA; attention itself remains SDPA.
"""

from functools import lru_cache
import warnings

import torch
from torch.autograd.function import once_differentiable

from litgpt.log_kv_cache import _fused_on
from litgpt.log_kv_position import materialize_anchor_keys


@lru_cache(maxsize=1)
def _triton_packer():
    try:
        from litgpt.log_kv_pack_triton import pack
        return pack
    except ImportError:
        return None


def _pack_torch(cache, plan, k, v, dim, buffers=None, skip_pooled=False, recent_start=0):
    indices, anchors, _, valid = plan
    pooled = indices.size(-1)
    exact = cache.alpha_count
    recent = cache.recent_count
    length = pooled + exact + recent + k.size(2)
    if buffers is None:
        shape = (*k.shape[:2], length, dim)
        ka, va = k.new_empty(shape), v.new_empty(shape)
    else:
        ka, va = (buf[:, :, :length] for buf in buffers)
    if not skip_pooled:
        ka[:, :, :pooled].zero_()
        va[:, :, :pooled].zero_()
        if pooled:
            def gather(x):
                x = x.flatten(2, 4)
                return x.gather(2, indices[..., None].expand(*indices.shape, x.size(-1)))

            keys = materialize_anchor_keys(
                gather(cache.level_k), anchors[..., None], cache.cos_cache,
                cache.sin_cache, cache.rope_n_elem,
            ).squeeze(-2)
            ka[:, :, :pooled, :k.size(-1)] = keys.masked_fill(~valid[..., None], 0)
            va[:, :, :pooled, :v.size(-1)] = gather(cache.level_v).masked_fill(~valid[..., None], 0)
            weights = cache.level_w.flatten(2).gather(2, indices)
            ka[:, :, :pooled, k.size(-1)] = weights.clamp_min(1).log().masked_fill(~valid, -10000).to(k.dtype)
        if exact:
            valid_exact = cache.alpha_valid[:, None, :exact].expand(-1, cache.n_groups, -1)
            anchors_exact = cache.alpha_pos[:, None, :exact, None].expand(-1, cache.n_groups, -1, -1)
            keys = materialize_anchor_keys(cache.alpha_k_raw[:, :, :exact], anchors_exact,
                                           cache.cos_cache, cache.sin_cache, cache.rope_n_elem).squeeze(-2)
            ka[:, :, pooled:pooled + exact].zero_()
            va[:, :, pooled:pooled + exact].zero_()
            ka[:, :, pooled:pooled + exact, :k.size(-1)] = keys.masked_fill(~valid_exact[..., None], 0)
            va[:, :, pooled:pooled + exact, :v.size(-1)] = cache.alpha_v[:, :, :exact].masked_fill(~valid_exact[..., None], 0)
            ka[:, :, pooled:pooled + exact, k.size(-1)] = (~valid_exact).to(k.dtype) * -10000
    pooled += exact
    # Write each source directly into the final allocation. No prefix cat or pad.
    ka[:, :, pooled + recent_start:].zero_()
    va[:, :, pooled + recent_start:].zero_()
    ka[:, :, pooled + recent_start:pooled + recent, :k.size(-1)] = cache.recent_k[:, :, recent_start:recent]
    va[:, :, pooled + recent_start:pooled + recent, :v.size(-1)] = cache.recent_v[:, :, recent_start:recent]
    ka[:, :, pooled + recent:, :k.size(-1)] = k
    va[:, :, pooled + recent:, :v.size(-1)] = v
    return ka, va


class _PackMidKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, cache, plan, dim, backend, buffers, skip_pooled, recent_start):
        ctx.start = plan[0].size(-1) + cache.alpha_count + cache.recent_count
        ctx.k_dim, ctx.v_dim = k.size(-1), v.size(-1)
        if buffers is not None and (k.requires_grad or v.requires_grad):
            raise ValueError("reusable decode buffers cannot hold differentiable K/V")
        packer = _triton_packer() if _fused_on(k) and backend != "torch" else None
        if backend == "triton" and packer is None:
            raise RuntimeError("Triton packing requires CUDA and an installed Triton runtime")
        if k.is_cuda and backend == "auto" and packer is None:
            _warn_missing_triton()
        if packer is None:
            return _pack_torch(cache, plan, k, v, dim, buffers, skip_pooled, recent_start)
        return packer(cache, plan, k, v, dim, buffers, skip_pooled, recent_start)

    @staticmethod
    @once_differentiable
    def backward(ctx, dk, dv):
        # Prefix state is deliberately detached. Each current token appears once.
        gk = dk[:, :, ctx.start:, :ctx.k_dim].contiguous() if ctx.needs_input_grad[0] and dk is not None else None
        gv = dv[:, :, ctx.start:, :ctx.v_dim].contiguous() if ctx.needs_input_grad[1] and dv is not None else None
        return gk, gv, None, None, None, None, None, None, None


@lru_cache(maxsize=1)
def _warn_missing_triton():
    warnings.warn("LogKV mid packing: Triton unavailable; using the PyTorch packer", stacklevel=3)


def pack_mid_kv(cache, plan, k, v, dim, *, backend=None, buffers=None, skip_pooled=False, recent_start=0):
    """Return final padded K/V; K's extra coordinate stores log mass / masking."""
    backend = cache.semantic_pack_backend if backend is None else backend
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("packing backend must be 'auto', 'torch' or 'triton'")
    if cache.semantic_anchor_mode != "mid":
        raise ValueError("mid packer requires semantic_anchor_mode='mid'")
    if (k.ndim != 4 or v.ndim != 4 or k.shape[:3] != v.shape[:3]
            or k.shape[:2] != (cache.batch_size, cache.n_groups)
            or k.size(-1) != cache.k_dim or v.size(-1) != cache.v_dim
            or k.device != cache.level_k.device or v.device != k.device
            or k.dtype != cache.level_k.dtype or v.dtype != k.dtype):
        raise ValueError("K/V shape, device and dtype must match the cache")
    if (len(plan) != 4 or plan[0].ndim != 3 or plan[0].shape[:2] != k.shape[:2]
            or any(x.shape != plan[0].shape or x.device != k.device for x in plan)):
        raise ValueError("mid attention plan must match K/V lanes and device")
    if dim <= k.size(-1) or dim < v.size(-1):
        raise ValueError("packed dimension must hold K, its bias coordinate, and V")
    if ((skip_pooled and buffers is None) or (recent_start and not skip_pooled)
            or not 0 <= recent_start <= cache.recent_count):
        raise ValueError("prefix reuse requires a workspace and a valid recent offset")
    if buffers is not None:
        length = plan[0].size(-1) + cache.alpha_count + cache.recent_count + k.size(2)
        if len(buffers) != 2 or any(buf.ndim != 4 or buf.shape[:2] != k.shape[:2] or buf.size(2) < length or buf.size(3) != dim
               or not buf.is_contiguous() or buf.dtype != k.dtype or buf.device != k.device for buf in buffers):
            raise ValueError("decode workspace does not match the packed layout")
    return _PackMidKV.apply(
        k, v, cache, plan, dim, backend, buffers, skip_pooled, recent_start,
    )
