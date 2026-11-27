"""Full-model Qwen3-8B + V4 attention CPT on the reference ProLong stream."""

import argparse
from dataclasses import asdict, fields
from datetime import timedelta
import importlib.metadata
import json
import math
import random
import shutil
import fcntl
import os
from pathlib import Path
import subprocess
import time

import torch
import torch.distributed as dist
import yaml

from .data import base_identity, sha256, make_loader, prepare_distributed
from .attention import Geometry, LAYOUT
from .model import Model, Block

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "configs/deepseek4/qwen8b.yaml"


def config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    for name in ("base_checkpoint_dir", "data_dir", "tokenized_dir", "output_dir"):
        value = os.environ.get("DEEPSEEK4_" + name.upper(), cfg[name])
        cfg[name] = str(Path(value).expanduser().resolve()) if value else None
    if min(cfg["steps"], cfg["micro_batch_size"], cfg["context_length"]) < 1 or cfg["context_length"] > 32768:
        raise ValueError("Invalid training length, steps or micro batch size")
    return cfg


def load_model(cfg, device, distributed):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from ._vendor.niahlogkv.litgpt import Config

    base = Path(cfg["base_checkpoint_dir"])
    base_config = yaml.safe_load((base / "model_config.yaml").read_text())
    allowed = {f.name for f in fields(Config)}
    arch = Config(**{k: v for k, v in base_config.items() if k in allowed})
    if arch.rope_adjustments or arch.rope_base != 1000000 or arch.parallel_residual or arch.norm_eps != 1e-6:
        raise ValueError("Expected original Qwen3-8B-Base geometry and RoPE")
    arch.block_size = cfg["context_length"]
    weights = torch.load(base / "lit_model.pth", map_location="cpu", mmap=True, weights_only=True)
    weights = weights.get("model", weights)
    with torch.device("meta"):
        student = Model(arch)
    for name in ("transformer.wte", "transformer.ln_f", "lm_head"):
        module = student.get_submodule(name)
        module.load_state_dict({"weight": weights[name + ".weight"]}, assign=True)
        module.to(device=device, dtype=torch.bfloat16)
    # Shard each initialized block before materializing the next. The root
    # FSDP unit owns the trainable embedding, final norm and output head.
    for i, ratio in enumerate(LAYOUT):
        block = Block(student.g, ratio, arch.intermediate_size, 1)
        prefix = f"transformer.h.{i}."
        for name in ("norm_1", "norm_2", "mlp"):
            module = getattr(block, name)
            module.load_state_dict({k: weights[prefix + name + "." + k] for k in module.state_dict()})
        block.to(dtype=torch.bfloat16)
        student.transformer.h[i] = (
            FSDP(block, device_id=device, use_orig_params=True) if distributed else block.to(device)
        )
    return FSDP(student, device_id=device, use_orig_params=True) if distributed else student


def save_model(model, directory, rank, distributed):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from contextlib import nullcontext

    state = {}
    context = (
        FSDP.summon_full_params(model, recurse=False, writeback=False, rank0_only=True)
        if distributed
        else nullcontext()
    )
    with context:
        root = model.module if distributed else model
        for name in ("transformer.wte", "transformer.ln_f", "lm_head"):
            if rank == 0:
                state[name + ".weight"] = root.get_submodule(name).weight.detach().to("cpu", copy=True)
    for i, block in enumerate(model.transformer.h):
        # Copy while the gathered storage is live. This avoids FSDP's full
        # state-dict offload hook and bounds gathering to one block at a time.
        context = (
            FSDP.summon_full_params(block, recurse=False, writeback=False, rank0_only=True)
            if distributed
            else nullcontext()
        )
        with context:
            if rank == 0:
                module = block.module if distributed else block
                for key, tensor in module.state_dict().items():
                    state[f"transformer.h.{i}.{key}"] = tensor.to(device="cpu", dtype=torch.bfloat16, copy=True)
    if rank == 0:
        torch.save(state, directory / "lit_model.pth")


def accumulation_steps(cfg, world):
    batch = cfg["global_batch_size"]
    per_round = world * cfg["micro_batch_size"]
    if batch < per_round or batch % per_round:
        raise ValueError(
            f"global_batch_size={batch} must be divisible by world_size × micro_batch_size={per_round}"
        )
    return batch // per_round


def learning_rate(cfg, completed):
    warmup, peak = cfg["warmup_steps"], cfg["learning_rate"]
    if completed < warmup:
        return peak * (completed + 1) / warmup
    progress = (completed - warmup) / max(1, cfg["steps"] - warmup)
    return cfg["min_lr"] + (peak - cfg["min_lr"]) * (1 + math.cos(math.pi * progress)) / 2


def optimizer(model, cfg):
    decay, plain = [], []
    for name, p in model.named_parameters():
        (plain if any(k in name.lower() for k in ("bias", "norm", "ln_")) else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": plain, "weight_decay": 0.0}],
        lr=cfg["learning_rate"],
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
    )


def save_progress(output, step, model, opt, loader, elapsed, rank, world):
    # Local original-parameter shards and optimizer use the same rank layout on resume.
    folder = output / f"checkpoint-{step:08d}"
    folder.mkdir(exist_ok=True)
    if (folder / "complete").exists():
        raise FileExistsError(f"Checkpoint already published: {folder}")
    torch.save(
        dict(
            params={name: p.detach().to("cpu", copy=True) for name, p in model.named_parameters()},
            optimizer=opt.state_dict(),
            loader=loader.state_dict(),
            elapsed=elapsed,
        ),
        folder / f"rank-{rank}.pt",
    )
    if world > 1:
        dist.barrier()
    if rank == 0:
        (folder / "complete").touch(exist_ok=False)
    if world > 1:
        dist.barrier()


def latest_checkpoint(output):
    digits = "[0-9]" * 8
    # Older runs published step directories by rename. New runs publish a marker last.
    folders = [p for p in output.glob("step-" + digits) if p.is_dir()]
    folders.extend(p for p in output.glob("checkpoint-" + digits) if (p / "complete").is_file())
    return max(folders, key=lambda p: int(p.name.split("-")[1]), default=None)


def restore_progress(folder, model, opt, loader, rank):
    state = torch.load(folder / f"rank-{rank}.pt", map_location="cpu", weights_only=True, mmap=True)
    params = dict(model.named_parameters())
    if params.keys() != state["params"].keys():
        raise ValueError("Checkpoint parameter names differ from the model")
    with torch.no_grad():
        for name, param in params.items():
            shard = state["params"][name]
            if param.shape != shard.shape:
                raise ValueError(f"Checkpoint shard shape changed: {name}")
            param.copy_(shard)
    opt.load_state_dict(state["optimizer"])
    loader.load_state_dict(state["loader"])
    return state["elapsed"]


def initialize_recipe(output, manifest):
    """Start in a pre-created directory, preserving unrelated ModelArts files."""
    recipe = output / "recipe.json"
    reset = output / ".cpt-reset"
    record = output / "training.json"
    for key in ("base_checkpoint_dir", "data_dir", "tokenized_dir"):
        path = manifest.get("config", {}).get(key)
        if path:
            asset = Path(path).resolve()
            destination = output.resolve()
            if asset.is_relative_to(destination) or destination.is_relative_to(asset):
                raise ValueError(f"Training output {output} overlaps {key}={path}")
    if recipe.is_symlink() or reset.is_symlink():
        raise FileExistsError(f"Expected regular training metadata in {output}")
    if recipe.is_file() and not reset.exists():
        try:
            previous = json.loads(recipe.read_text())
        except json.JSONDecodeError:
            previous = None
        if previous == manifest:
            if not record.exists():
                return
            try:
                json.loads(record.read_text())
            except json.JSONDecodeError:
                final = output / f"checkpoint-{manifest['config']['steps']:08d}" / "complete"
                if final.is_file():
                    return  # Retry final export from the published last step.

    # Keep the marker until cleanup and publication both finish. A retry under
    # the output lock must never resume a partly deleted generation.
    reset.touch()
    record.unlink(missing_ok=True)
    reserved = {"training.json.tmp", "model_config.yaml", "lit_model.pth", "lit_model.pth.tmp", "evaluation"}
    for entry in output.iterdir():
        if (entry.name in reserved or entry.match("step-" + "[0-9]" * 8)
                or entry.match("checkpoint-" + "[0-9]" * 8) or entry.match("steps-from-*.jsonl")
                or entry.name.startswith(".checkpoint-")):
            if entry.is_symlink() or not entry.is_dir():
                entry.unlink()
            else:
                shutil.rmtree(entry)
    recipe.write_text(json.dumps(manifest, indent=2))
    reset.unlink()
    print(f"Starting a new CPT run in {output} from the Base checkpoint", flush=True)


def train(cfg):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires BF16 CUDA")
    world, rank = int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("RANK", 0))
    accumulation = accumulation_steps(cfg, world)
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    multi = world > 1
    if multi:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    lock = None
    try:
        torch.manual_seed(cfg["seed"])
        random.seed(cfg["seed"])
        output = Path(cfg["output_dir"])
        loader, data = make_loader(cfg)
        if multi:
            inputs = [None] * world
            dist.all_gather_object(inputs, (cfg, data))
            if any(entry != inputs[0] for entry in inputs):
                raise ValueError("Ranks resolved different training configs or ProLong assets")
        if len(loader) * cfg["num_epochs"] < accumulation * cfg["steps"]:
            raise ValueError(
                "ProLong stream is too short to reach the formal budget within the reference epoch limit"
            )
        base = Path(cfg["base_checkpoint_dir"])
        start = [None]
        if not rank:
            output.mkdir(parents=True, exist_ok=True)
            lock = (output / ".cpt.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = dict(
                method="qwen3-v4flash-36",
                config=cfg,
                layout=list(LAYOUT),
                attention=asdict(Geometry()),
                base=base_identity(base),
                data=data,
                world_size=world,
                accumulation_steps=accumulation,
                precision="bf16-true",
                objective="next_token_ce + csa_index_ce",
                trainable="all",
                device=torch.cuda.get_device_name(device),
                cuda=torch.version.cuda,
                versions={
                    p: importlib.metadata.version(p) for p in ("torch", "litdata", "lightning", "tokenizers")
                },
                git=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "deepseek4").glob("*.py")},
            )
            record = output / "training.json"
            initialize_recipe(output, manifest)
            checkpoint = latest_checkpoint(output)
            if checkpoint is not None:
                start[0] = checkpoint.name
        if multi:
            dist.broadcast_object_list(start, src=0)
        checkpoint_name = start[0]
        completed = int(checkpoint_name.split("-")[1]) if checkpoint_name else 0
        model = load_model(cfg, device, multi)
        opt = optimizer(model, cfg)
        previous_time = (
            restore_progress(output / checkpoint_name, model, opt, loader, rank) if checkpoint_name else 0
        )
        if not rank:
            # Retain superseded attempts separately so throughput and wasted work remain auditable.
            log = (output / f"steps-from-{completed:08d}-{time.time_ns()}.jsonl").open("x", buffering=1)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        batches = iter(loader)
        tokens_per_update = cfg["global_batch_size"] * cfg["context_length"]
        for step in range(completed + 1, cfg["steps"] + 1):
            opt.zero_grad(set_to_none=True)
            totals = torch.zeros(2, dtype=torch.float32, device=device)
            for _ in range(accumulation):
                batch = next(batches, None)
                if batch is None:
                    if loader.current_epoch >= cfg["num_epochs"]:
                        raise RuntimeError(
                            f"Reference epoch limit reached before update {step}; formal budget incomplete"
                        )
                    batches = iter(loader)
                    batch = next(batches)
                if batch.shape != (cfg["micro_batch_size"], cfg["context_length"] + 1):
                    raise ValueError(f"Unexpected training batch {batch.shape}")
                batch = batch.to(device, dtype=torch.long)
                ce, idx = model(batch[:, :-1], targets=batch[:, 1:], loss_chunk_size=cfg["entropy_chunk_size"])
                ((ce + idx) / accumulation).backward()
                totals += torch.stack((ce.detach(), idx.detach()))
            if multi:
                dist.all_reduce(totals)
            totals /= world * accumulation
            if not totals.isfinite().all():
                raise FloatingPointError(f"Nonfinite loss at update {step}")
            lr = learning_rate(cfg, step - 1)
            for group in opt.param_groups:
                group["lr"] = lr
            opt.step()
            elapsed = previous_time + time.monotonic() - started
            if not rank:
                metrics = dict(
                    step=step,
                    lm=totals[0].item(),
                    index_ce=totals[1].item(),
                    lr=lr,
                    tokens=step * tokens_per_update,
                    gpu_hours=elapsed * world / 3600,
                    attempt_seconds=time.monotonic() - started,
                    rank0_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                )
                print(metrics, flush=True)
                log.write(json.dumps(metrics) + "\n")
            if step % cfg["save_interval"] == 0 or step == cfg["steps"]:
                save_progress(output, step, model, opt, loader, elapsed, rank, world)
        opt.zero_grad(set_to_none=True)
        del opt
        save_model(model, output, rank, multi)
        if not rank:
            log.close()
            (output / "model_config.yaml").write_bytes((base / "model_config.yaml").read_bytes())
            # Earlier failed attempts may have progressed beyond the last published checkpoint.
            # Count their reported wall times as well as work on the final trajectory.
            attempt_seconds = time.monotonic() - started
            for earlier in output.glob("steps-from-*.jsonl"):
                if earlier == Path(log.name):
                    continue
                last_seconds = 0
                with earlier.open() as stream:
                    for line in stream:
                        if line.endswith("\n"):
                            last_seconds = json.loads(line)["attempt_seconds"]
                attempt_seconds += last_seconds
            manifest.update(
                steps=cfg["steps"],
                tokens=cfg["steps"] * tokens_per_update,
                gpu_hours=attempt_seconds * world / 3600,
                weight_sha256=sha256(output / "lit_model.pth"),
            )
            record.write_text(json.dumps(manifest, indent=2))
        if multi:
            dist.barrier()
    finally:
        if lock is not None:
            lock.close()
        if multi:
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    cfg = config(args.config)
    if args.prepare:
        prepare_distributed(cfg)
    else:
        train(cfg)
