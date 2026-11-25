"""Qwen3-8B frozen teacher, frozen Qwen MLPs and trainable V4 attention."""

import argparse
from dataclasses import asdict, fields
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import torch.distributed as dist
from torch import nn
import yaml

from .data import base_identity, sha256, make_loader, prepare_distributed
from .attention import Geometry, LAYOUT
from .model import Model, Block

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "configs/deepseek4/qwen8b.yaml"


def config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    for name in ("base_checkpoint_dir", "data_dir", "tokenized_dir", "output_dir"):
        cfg[name] = str(Path(os.environ.get("DEEPSEEK4_" + name.upper(), cfg[name])).expanduser().resolve())
    if min(cfg["steps"], cfg["micro_batch_size"], cfg["context_length"]) < 1 or cfg["context_length"] > 32768:
        raise ValueError("Invalid training length, steps or micro batch size")
    return cfg


def load_models(cfg, device, distributed):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from ._vendor.niahlogkv.litgpt import Config, GPT

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
        teacher = GPT(arch)
        student = Model(arch)
    teacher.load_state_dict(weights, strict=True, assign=True)
    teacher.lm_head = nn.Identity()
    # Regenerate the two nonpersistent RoPE buffers constructed on meta.
    teacher.cos, teacher.sin = teacher.rope_cache(device=device)
    teacher.to(device=device, dtype=torch.bfloat16).requires_grad_(False).eval()
    for mod, name in (
        (student.transformer.wte, "transformer.wte"),
        (student.transformer.ln_f, "transformer.ln_f"),
        (student.lm_head, "lm_head"),
    ):
        mod.load_state_dict({"weight": weights[name + ".weight"]}, assign=True)
        mod.to(device=device, dtype=torch.bfloat16).requires_grad_(False)
    # Materialize and shard one block at a time. Frozen weights and trainable
    # weights use FP32 FSDP master storage with BF16 computation.
    for i, ratio in enumerate(LAYOUT):
        block = Block(student.g, ratio, arch.intermediate_size, 1)
        prefix = f"transformer.h.{i}."
        for name in ("norm_1", "norm_2", "mlp"):
            module = getattr(block, name)
            module.load_state_dict({k: weights[prefix + name + "." + k] for k in module.state_dict()})
            module.requires_grad_(False)
        if distributed:
            block = FSDP(
                block,
                device_id=device,
                use_orig_params=True,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            )
        else:
            block.to(device)
        student.transformer.h[i] = block
    return student, teacher


def save_model(model, directory, rank, distributed):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
    from contextlib import nullcontext

    state = {}
    for i, block in enumerate(model.transformer.h):
        context = (
            FSDP.state_dict_type(
                block, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            )
            if distributed
            else nullcontext()
        )
        with context:
            part = block.state_dict()
        if rank == 0:
            state.update({f"transformer.h.{i}.{k}": t.to("cpu", torch.bfloat16) for k, t in part.items()})
    if rank == 0:
        for name in ("transformer.wte", "transformer.ln_f", "lm_head"):
            state[name + ".weight"] = model.get_submodule(name).weight.detach().cpu()
        with (directory / "lit_model.pth").open("xb") as f:
            torch.save(state, f)


def train(cfg):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires BF16 CUDA")
    world, rank = int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("RANK", 0))
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    multi = world > 1
    if multi:
        dist.init_process_group("nccl")
    try:
        torch.manual_seed(cfg["seed"])
        output = Path(cfg["output_dir"])
        if not rank:
            output.mkdir(parents=True, exist_ok=False)
        if multi:
            dist.barrier()
        loader, data = make_loader(cfg)
        model, teacher = load_models(cfg, device, multi)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"], foreach=False
        )
        base = Path(cfg["base_checkpoint_dir"])
        if not rank:
            manifest = dict(
                method="qwen3-v4flash-36",
                config=cfg,
                layout=LAYOUT,
                attention=asdict(Geometry()),
                base=base_identity(base),
                data=data,
                world_size=world,
                device=torch.cuda.get_device_name(device),
                cuda=torch.version.cuda,
                versions={
                    p: importlib.metadata.version(p) for p in ("torch", "litdata", "lightning", "tokenizers")
                },
                git=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "deepseek4").glob("*.py")},
            )
        if multi:
            dist.barrier()
        started = time.monotonic()
        batches = iter(loader)
        for step in range(1, cfg["steps"] + 1):
            batch = next(batches, None)
            if batch is None:
                raise RuntimeError("Prepared data exhausted before the requested number of updates")
            batch = batch.to(device, dtype=torch.long)
            if batch.shape != (cfg["micro_batch_size"], cfg["context_length"] + 1):
                raise ValueError(f"Unexpected training batch {batch.shape}")
            ids = batch[:, :-1]
            opt.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                target = teacher(ids)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ce, kd, idx = model.training_loss(ids, batch[:, 1:], target)
                loss = ce + kd + idx
            if not loss.isfinite():
                raise FloatingPointError(f"Nonfinite loss at update {step}")
            loss.backward()
            params = [p for p in model.parameters() if p.grad is not None]
            grad_sq = torch.stack([p.grad.detach().norm().square() for p in params]).sum()
            if multi:
                dist.all_reduce(grad_sq)
            if not grad_sq.isfinite():
                raise FloatingPointError(f"Nonfinite gradients at update {step}")
            scale = grad_sq.clamp_min(1).rsqrt()
            for p in params:
                p.grad.mul_(scale)
            opt.step()
            values = torch.stack((ce, kd, idx)).detach()
            if multi:
                dist.all_reduce(values)
                values /= world
            if not rank:
                metrics = dict(
                    step=step,
                    lm=values[0].item(),
                    teacher_kl=values[1].item(),
                    index_ce=values[2].item(),
                    tokens=world * step * ids.numel(),
                )
                print(metrics, flush=True)
                with (output / "steps.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
        elapsed = time.monotonic() - started
        save_model(model, output, rank, multi)
        if not rank:
            (output / "model_config.yaml").write_bytes((base / "model_config.yaml").read_bytes())
            manifest.update(
                steps=step,
                tokens=metrics["tokens"],
                gpu_hours=elapsed * world / 3600,
                weight_sha256=sha256(output / "lit_model.pth"),
            )
            (output / "training.json").write_text(json.dumps(manifest, indent=2))
    finally:
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
