"""Rank helpers for one-model-per-GPU evaluation with CPU (Gloo) result exchange.

Each rank evaluates ``requests[rank::world]``; results are gathered as Python
objects and re-interleaved into the original request order. Waiting ranks
print a heartbeat instead of blocking silently, so long-context stragglers are
not mistaken for a hung job by cluster watchdogs.
"""

from __future__ import annotations

import os
import time

import torch.distributed as dist


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def global_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else _env_int("RANK", 0)


def world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else _env_int("WORLD_SIZE", 1)


def local_rank() -> int:
    return _env_int("LOCAL_RANK", global_rank())


def rank_label() -> str:
    return f"GlobalRank {global_rank()}/{world_size()} (local {local_rank()})"


def is_main() -> bool:
    return global_rank() == 0


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def broadcast_object(obj, src: int = 0, group=None):
    if not dist_ready():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src, group=group)
    return box[0]


_SYNC_SEQ = 0


def wait_all_ranks(stage: str) -> float:
    """Barrier on the default store with periodic reports of missing ranks."""
    global _SYNC_SEQ
    if not dist_ready():
        return 0.0
    _SYNC_SEQ += 1
    key = f"kv_baseline_sync/{_SYNC_SEQ}"
    world = world_size()
    store = dist.distributed_c10d._get_default_store()
    store.set(f"{key}/rank{global_rank()}", "1")
    store.add(key, 1)
    heartbeat = max(_env_int("KV_BASELINE_SYNC_HEARTBEAT_S", 300), 1)
    timeout = _env_int("KV_BASELINE_SYNC_TIMEOUT_S", 7200)
    start, next_report = time.perf_counter(), heartbeat
    while store.add(key, 0) < world:
        waited = time.perf_counter() - start
        if (0 < timeout <= waited) or waited >= next_report:
            missing = [r for r in range(world) if not store.check([f"{key}/rank{r}"])]
            message = f"[{rank_label()}] {stage}: waited {waited:.0f}s, missing ranks {missing}"
            if 0 < timeout <= waited:
                raise RuntimeError(f"{message} (KV_BASELINE_SYNC_TIMEOUT_S={timeout})")
            print(f"⏳ {message}", flush=True)
            next_report += heartbeat
        time.sleep(1.0)
    return time.perf_counter() - start


def all_gather_interleaved(local: list, group=None, tag: str = "") -> list:
    """Gather each rank's ``requests[rank::world]`` results back into request order."""
    if not dist_ready():
        return local
    print(f"⏳ [{rank_label()}] {tag}: {len(local)} local results done, waiting for other ranks", flush=True)
    wait_all_ranks(f"{tag} gather")
    gathered = [None] * world_size()
    dist.all_gather_object(gathered, local, group=group)
    out = []
    for i in range(max(len(rows) for rows in gathered)):
        for rows in gathered:
            if i < len(rows):
                out.append(rows[i])
    return out
