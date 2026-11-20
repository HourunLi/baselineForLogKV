"""Per-process host counters, with optional CUDA event spans for selected steps.

CUDA spans include stream idle/wait time, not just kernel execution. Model-level
forward/backward spans contain the LogKV spans; never add the two sets together.
alpha_select covers span selection inside route (do not add that part to the
route total again) plus the post-flush score prefetch, which runs outside route.
"""

from contextlib import contextmanager
import time

import torch


CACHE_STAGES = ("route", "replay", "plan", "pack", "attn_fwd", "attn_bwd", "alpha_select")
STEP_STAGES = ("data", "forward", "backward", "optimizer")
HOST_STATS = {f"{stage}_{suffix}": 0 for stage in CACHE_STAGES + STEP_STAGES for suffix in ("s", "n")}
_events = []
_device = None


def logkv_begin_step(*, profile_cuda=False, device=None):
    """Reset once per optimizer step, before its first accumulation micro-batch."""
    global _device
    if _events:
        raise RuntimeError("consume LogKV CUDA timings before beginning another step")
    HOST_STATS.update(dict.fromkeys(HOST_STATS, 0))
    _device = torch.device(device) if profile_cuda else None
    if _device is not None and _device.type != "cuda":
        _device = None
        raise ValueError("log_kv_profile_steps requires a CUDA training device")


@contextmanager
def logkv_timed(stage, *, cuda=True):
    stream = start = end = None
    if cuda and _device is not None:
        stream = torch.cuda.current_stream(_device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        HOST_STATS[f"{stage}_s"] += time.perf_counter() - t0
        HOST_STATS[f"{stage}_n"] += 1
        if start is not None:
            end.record(stream)
            _events.append((stage, start, end))


def logkv_take_host_stats():
    """Consume counters; only selected profiling steps synchronize the GPU."""
    global _device
    out = dict(HOST_STATS)
    out["cuda_profiled"] = _device is not None
    if _device is not None:
        torch.cuda.synchronize(_device)
        for stage in CACHE_STAGES + STEP_STAGES:
            out[f"{stage}_cuda_s"] = 0.0
        for stage, start, end in _events:
            out[f"{stage}_cuda_s"] += start.elapsed_time(end) / 1000.0
    # Compatibility for consumers of the old combined input-preparation metric.
    out["attn_s"] = out["plan_s"] + out["pack_s"]
    out["attn_n"] = out["pack_n"]
    _events.clear()
    _device = None
    HOST_STATS.update(dict.fromkeys(HOST_STATS, 0))
    return out
