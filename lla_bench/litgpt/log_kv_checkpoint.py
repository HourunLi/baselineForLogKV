"""Bind routing logs to one non-reentrant Block checkpoint invocation.

The checkpoint frame owns the records, not a layer-global queue. It retains no
full Q/K/V or ladder snapshots; op-log tensors are shared with the original
context, together with CPU replay schedules. When enabled, a fifth record owns
input entries and routing-state updates on their original device. Recompute contexts can
be entered again for retain_graph backward.
"""

from contextvars import ContextVar
from functools import partial
import weakref

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.utils.checkpoint import checkpoint


_route_pass = ContextVar("logkv_checkpoint_route_pass", default=None)


class _RouteContext:
    def __init__(self, records, recompute):
        self.records = records
        self.recompute = recompute

    def __enter__(self):
        # A fresh cursor per entry supports repeated backward on the same graph.
        self.token = _route_pass.set([self.records, self.recompute, 0])

    def __exit__(self, exc_type, exc, tb):
        state = _route_pass.get()
        _route_pass.reset(self.token)
        if not self.recompute and exc_type is not None:
            self.records.clear()
        # PyTorch may stop recompute early once saved tensors are restored.
        if self.recompute and exc_type is None and state[2] != len(self.records):
            raise RuntimeError("LogKV checkpoint recompute did not consume all routing records")


def logkv_checkpoint_contexts():
    records = []
    return _RouteContext(records, False), _RouteContext(records, True)


def checkpoint_route_replay(cache, signature):
    """Return the matching log during recompute; never silently reroute."""
    state = _route_pass.get()
    if state is None:
        return None  # ordinary forward, outside an enabled checkpoint
    if not cache.semantic_clusters or cache.K_max <= 1:
        raise RuntimeError("LogKV checkpoint routing replay requires semantic clusters with K > 1")
    records, recompute, cursor = state
    if not recompute:
        return None
    if cursor >= len(records):
        raise RuntimeError("LogKV checkpoint recompute has no matching routing record")
    cache_ref, expected, log = records[cursor]
    if cache_ref() is not cache or signature != expected:
        raise RuntimeError("LogKV checkpoint routing record does not match this cache/input/configuration")
    if any(part is None for part in log):
        raise RuntimeError("LogKV checkpoint routing record is incomplete")
    state[2] += 1
    return log


def checkpoint_record_routes(cache, signature, log):
    state = _route_pass.get()
    if state is not None and not state[1]:
        if any(part is None for part in log):
            raise RuntimeError("LogKV checkpoint forward did not produce a complete routing record")
        state[0].append((weakref.ref(cache), signature, log))


def enable_logkv_checkpoint_replay(model, block_type):
    """Attach to existing Fabric/PyTorch wrappers, preserving checkpoint options."""
    updates = []
    blocks = [module for module in model.modules() if isinstance(module, block_type)]
    wrappers = [module for module in model.modules()
                if isinstance(module, CheckpointWrapper)
                and isinstance(module._checkpoint_wrapped_module, block_type)]
    if not blocks or len(wrappers) != len(blocks):
        raise RuntimeError("LogKV routing replay requires a checkpoint wrapper for every Block")
    for wrapper in wrappers:
        fn = wrapper.checkpoint_fn
        if not isinstance(fn, partial) or fn.func is not checkpoint or fn.keywords.get("use_reentrant") is not False:
            raise RuntimeError("LogKV routing replay requires the standard non-reentrant PyTorch checkpoint")
        options = dict(fn.keywords)
        if options.get("context_fn") not in (None, logkv_checkpoint_contexts) or options.get("debug", False):
            raise RuntimeError("LogKV routing replay cannot replace a custom/debug checkpoint context")
        options["context_fn"] = logkv_checkpoint_contexts
        updates.append((wrapper, partial(fn.func, *fn.args, **options)))
    for wrapper, fn in updates:
        wrapper.checkpoint_fn = fn
    return len(wrappers)
