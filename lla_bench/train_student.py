"""Explicit conversion, layerwise distillation and 32K CPT for the Qwen HAttention student."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, MixedPrecision, StateDictType
from torch.nn import functional as F
import yaml

from . import prolong
from .model import HAttentionLM, HConfig
from .student import base_identity, checkpoint_report, convert, export_checkpoint, file_hash, load_teacher


def distill_backward(teacher, mixers, ids, accumulation):
    """Consume each teacher activation in its own hook, freeing the student graph before the next layer."""
    losses, hooks = [], []

    def attach(mixer):
        def align(module, args, kwargs, output):
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            target = output[0].detach().float()
            with torch.enable_grad():
                predicted = mixer(hidden.detach())
                loss = F.mse_loss(predicted.float(), target) / target.square().mean().clamp_min(1e-12)
                (loss / (len(mixers) * accumulation)).backward()
            losses.append(loss.detach())
        return align

    try:
        for block, mixer in zip(teacher.layers, mixers, strict=True):
            hooks.append(block.self_attn.register_forward_hook(attach(mixer), with_kwargs=True))
        with torch.no_grad():
            teacher(input_ids=ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    return torch.stack(losses).mean()


def token_batches(data, length, window):
    if length % window:
        raise ValueError("distillation window must divide the packed training length")
    while True:
        seen = False
        for batch in data:
            seen = True
            if batch.shape != (1, length + 1):
                raise ValueError(f"expected one packed sequence of {length + 1} tokens, got {batch.shape}")
            for offset in range(0, length, window):
                yield batch[:, offset:offset + window + 1].long()
        if not seen:
            raise ValueError("empty ProLong epoch on this rank")


def runtime_report():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return dict(commit=commit.stdout.strip(), python=sys.version, cuda=torch.version.cuda,
                versions={p: version(p) for p in ("torch", "transformers", "safetensors", "litdata", "huggingface-hub", "tokenizers", "pyarrow")},
                source_sha256={p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name("model.py"),
                               Path(__file__).with_name("recurrence.py"), Path(prolong.__file__),
                               Path(__file__).with_name("student.py"))})


def train(stage, config, parent, out, base, cache, device, world, rank, steps_override=None):
    distilling = stage == "distill"
    probe = stage == "probe"
    previous = checkpoint_report(parent)
    expected = "distill" if stage == "cpt" else "convert"
    if previous["stage"] != expected:
        raise ValueError(f"{stage} requires a {expected} checkpoint")
    length = config["context_length"]
    cfg = HConfig.read(parent / "config.json", None)
    if cfg.block_size != length or cfg.trained_levels != (length - 1).bit_length() + 1:
        raise ValueError("CPT length differs from the converted hierarchy")
    if not (cache / "index.json").is_file():
        raise FileNotFoundError(f"missing ProLong index in {cache}; check the cache path or run the prepare stage")
    data = prolong.loader(cache, length, config["seed"], config["num_workers"])
    recipe = config["distill" if distilling else "cpt"]
    window = recipe["window"] if distilling else length
    steps = 1 if probe else (steps_override or recipe["steps"])
    batch_size = world if probe else recipe["global_batch_size"]
    if steps < 1 or batch_size < world or batch_size % world:
        raise ValueError("steps must be positive and global_batch_size must be a positive multiple of world size")
    accumulation = batch_size // world
    stream = token_batches(data, length, window)
    student = HAttentionLM.load(parent, cfg, "cpu", dtype=None)
    if probe:
        student.backbone.layers = torch.nn.ModuleList([student.backbone.layers[0]])
    gpu = device.type == "cuda"
    sharded = world > 1
    precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    def wrap(module):
        module = module.float()
        return FSDP(module, device_id=device, mixed_precision=precision if gpu else None) if sharded else module.to(device)

    if distilling:
        if rank == 0 and base_identity(base) != previous["teacher_sha256"]:
            raise ValueError(f"teacher files differ from conversion in {base}")
        teacher = load_teacher(base).model.to(device=device, dtype=torch.bfloat16 if gpu else torch.float32)
        student.requires_grad_(False)
        for layer in student.backbone.layers:
            layer.mixer = wrap(layer.mixer.requires_grad_(True))
        mixers = [layer.mixer for layer in student.backbone.layers]
        model = student
    else:
        student.train().requires_grad_(not probe)
        for i, layer in enumerate(student.backbone.layers):
            student.backbone.layers[i] = wrap(layer.requires_grad_(True))
        model = wrap(student)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=recipe["lr"], weight_decay=recipe["weight_decay"], foreach=False)
    digest = hashlib.sha256()
    if gpu:
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for step in range(1, steps + 1):
        if not distilling:
            warmup = recipe["warmup_steps"]
            progress = max(0, step - warmup) / max(1, recipe["steps"] - warmup)
            lr = (recipe["lr"] * step / warmup if step <= warmup else recipe["min_lr"]
                  + 0.5 * (recipe["lr"] - recipe["min_lr"]) * (1 + math.cos(math.pi * min(1, progress))))
            for group in optimizer.param_groups:
                group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        mean = torch.zeros((), device=device)
        for _ in range(accumulation):
            batch = next(stream)
            digest.update(batch.contiguous().numpy().tobytes())
            batch = batch.to(device)
            ids = batch[:, :-1]
            # FSDP casts parameter computations. The unsharded CUDA path keeps FP32 master weights.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=gpu and not sharded):
                if distilling:
                    loss = distill_backward(teacher, mixers, ids, accumulation)
                else:
                    loss = model(ids, targets=batch[:, 1:])
                    (loss / accumulation).backward()
            mean += loss.detach() / accumulation
        if world > 1:
            dist.all_reduce(mean)
            mean /= world
        if not torch.isfinite(mean):
            raise FloatingPointError(f"non-finite {stage} loss at step {step}")
        optimizer.step()
        if rank == 0:
            print(f"{stage} step={step}/{steps} loss={mean.item():.6g} lr={optimizer.param_groups[0]['lr']:.3g}",
                  flush=True)
    if gpu:
        torch.cuda.synchronize(device)
    stats = dict(rank=rank, device=torch.cuda.get_device_name(device) if gpu else "CPU",
                 wall_seconds=time.monotonic() - started, packed_input_sha256=digest.hexdigest(),
                 peak_allocated=torch.cuda.max_memory_allocated(device) if gpu else None,
                 peak_reserved=torch.cuda.max_memory_reserved(device) if gpu else None)
    ranks = [stats]
    if world > 1:
        ranks = [None] * world
        dist.all_gather_object(ranks, stats)
    report = dict(stage=stage, recipe=config, last_loss=mean.item(), parent=str(parent.resolve()),
                  parent_manifest_sha256=file_hash(parent / "training.json"), teacher_sha256=previous["teacher_sha256"],
                  teacher_path=previous["teacher_path"], cache=str(cache.resolve()), cache_index_sha256=file_hash(cache / "index.json"),
                  actual_steps=steps, input_tokens=steps * batch_size * window, global_batch_size=batch_size,
                  world_size=world, ranks=ranks, **runtime_report())
    report["training_stages"] = {**previous.get("training_stages", {}), stage: dict(
        steps=steps, input_tokens=report["input_tokens"], global_batch_size=batch_size, sequence_length=window,
        last_loss=report["last_loss"], gpu_hours=sum(r["wall_seconds"] for r in ranks) / 3600 if gpu else None)}
    if probe:
        if rank == 0:
            print(json.dumps(report, indent=2))
        return
    ctx = (FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                FullStateDictConfig(rank0_only=True, offload_to_cpu=gpu)) if sharded else nullcontext())
    with ctx:
        state = model.state_dict()
    if rank == 0:
        export_checkpoint(out, cfg, state, parent, report)


def reuse_checkpoint(out, config, expected):
    if not out.exists():
        return False
    print(f"Verifying existing checkpoint: {out}", flush=True)
    try:
        report = checkpoint_report(out, verify_all=True)
        for key, value in expected.items():
            if report[key] != value:
                raise ValueError(f"{key} differs from the requested run")
        cfg = HConfig.read(out / "config.json", None)
        if not cfg.student or cfg.block_size != config["context_length"]:
            raise ValueError("student configuration differs from the requested conversion")
        # Strictly check tensor names and shapes via CPU mmap without copying the weights.
        HAttentionLM.load(out, cfg, "cpu", dtype=None)
    except (OSError, KeyError, ValueError, RuntimeError, TypeError) as exc:
        raise ValueError(f"Cannot reuse {out}: {exc}. Inspect the files or use a fresh LLA_STUDENT_ROOT.") from exc
    print(f"Verified complete checkpoint, skipping {out.name}: {out}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("convert", "prepare", "distill", "probe", "cpt"))
    parser.add_argument("--config", type=Path, default=Path("configs/qwen8b_hattention_train.yaml"))
    parser.add_argument("--steps", type=int, help="shorten this distill/CPT run; use a separate output root")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    base = Path(os.getenv("LLA_QWEN_BASE", config["base"])).resolve()
    root = Path(os.getenv("LLA_STUDENT_ROOT", config["output_root"])).resolve()
    raw = Path(os.getenv("LLA_PROLONG_DATA", config["prolong"])).resolve()
    private_cache = root / "prolong_cache"
    override = os.getenv("LLA_PROLONG_CACHE_ROOT")
    cache_root = Path(override).resolve() if override else raw
    if args.print_config:
        print(json.dumps({**config, "stage": args.stage, "base": str(base), "output_root": str(root),
                          "prolong": str(raw), "read_cache_root": str(cache_root), "private_cache_root": str(private_cache),
                          "steps_override": args.steps}, indent=2))
        return
    stage = args.stage
    out = root / stage
    if args.steps is not None and (stage not in {"distill", "cpt"} or args.steps < 1):
        parser.error("--steps requires distill or cpt and a positive value")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    torch.manual_seed(config["seed"])
    if stage == "convert":
        if world != 1:
            parser.error("conversion is a single process")
        identity = base_identity(base)
        if reuse_checkpoint(out, config, dict(stage=stage, seed=config["seed"], teacher_sha256=identity)):
            return
        teacher = load_teacher(base)
        c = teacher.config
        if (c.num_hidden_layers, c.hidden_size, c.intermediate_size, c.num_attention_heads,
                c.num_key_value_heads, c.head_dim) != (36, 4096, 12288, 32, 8, 128):
            raise ValueError(f"{base} does not have the Qwen3-8B-Base geometry")
        student = convert(teacher, config["context_length"])
        export_checkpoint(out, student.config, student.state_dict(), base,
                          dict(stage=stage, seed=config["seed"], teacher_path=str(base),
                               teacher_sha256=identity, **runtime_report()))
        return
    parent = root / ("distill" if stage == "cpt" else "convert")
    name = prolong.cache_name(base if stage == "prepare" else parent, config["context_length"])
    cache = cache_root / name
    if not override and not cache.exists():
        cache = private_cache / name
    if stage == "prepare":
        if world != 1:
            parser.error("data preparation is a single process")
        if (cache / "index.json").is_file():
            prolong.validate_cache(cache)
            print(f"Using existing ProLong cache: {cache}", flush=True)
            return
        prolong.prepare(raw, cache, base, config["context_length"], max(1, config["num_workers"]))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("formal training and the 32K probe require a CUDA GPU")
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl")
    try:
        skip = [False]
        if rank == 0 and stage in {"distill", "cpt"} and out.exists():
            expected = dict(stage=stage, recipe=config, actual_steps=args.steps or config[stage]["steps"],
                            parent_manifest_sha256=file_hash(parent / "training.json"),
                            cache_index_sha256=file_hash(cache / "index.json"))
            skip[0] = reuse_checkpoint(out, config, expected)
        if world > 1:
            dist.broadcast_object_list(skip, src=0)
        if not skip[0]:
            train(stage, config, parent, out, base, cache, device, world, rank, args.steps)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
