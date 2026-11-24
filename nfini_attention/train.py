"""Full-parameter ProLong CPT. Checkpoints are published only after all ranks finish saving."""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import socket
import tempfile
import uuid
import time
from contextlib import nullcontext
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch
from torch.distributed.elastic.multiprocessing.errors import record
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import yaml
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from lla_bench import prolong
from lla_bench.student import base_identity, file_hash
from .model import InfiniLM, SegmentLayer


def code_identity():
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return dict(commit=commit, source_sha256={p.name: file_hash(p) for p in
                (Path(__file__), here / "model.py", Path(prolong.__file__))})


def learning_rate(step, cfg):
    # The reference schedules with zero-based optimizer step before the update.
    warm = cfg["warmup_steps"]
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    progress = (step - warm) / max(1, cfg["steps"] - warm)
    return cfg["min_lr"] + (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * progress)) / 2


def distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def gather(value):
    if not distributed():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def save_progress(folder, model, optimizer, loader, meta, rank):
    folder.parent.mkdir(parents=True, exist_ok=True)
    tmp = [tempfile.mkdtemp(dir=folder.parent, prefix=".saving-") if rank == 0 else None]
    if distributed():
        dist.broadcast_object_list(tmp, src=0)
    stage = Path(tmp[0])
    model_state, optim_state = get_state_dict(model, optimizer)
    dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=stage / "state")
    torch.save(loader.state_dict(), stage / f"data-{rank}.pt")
    if distributed():
        dist.barrier()
    if rank == 0:
        (stage / "progress.json").write_text(json.dumps(meta, indent=2))
        stage.rename(folder)
    if distributed():
        dist.barrier()


def restore(folder, model, optimizer, loader, identity, rank):
    meta = json.loads((folder / "progress.json").read_text())
    if meta["identity"] != identity:
        raise ValueError("resume requires the same base, recipe, cache and world size")
    model_state, optim_state = get_state_dict(model, optimizer)
    dcp.load({"model": model_state, "optimizer": optim_state}, checkpoint_id=folder / "state")
    set_state_dict(model, optimizer, model_state_dict=model_state, optim_state_dict=optim_state)
    loader.load_state_dict(torch.load(folder / f"data-{rank}.pt", map_location="cpu", weights_only=False))
    return meta


def export(folder, model, state, base, report):
    with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
        stage = Path(tmp) / "export"
        model.save_pretrained(stage, state_dict=state, max_shard_size="2GB", safe_serialization=True)
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                     "special_tokens_map.json", "added_tokens.json", "generation_config.json"):
            if (base / name).is_file():
                shutil.copy2(base / name, stage / name)
        files = list(stage.iterdir())
        report["files_sha256"] = {p.name: file_hash(p) for p in files}
        report["file_stats"] = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in files}
        (stage / "training.json").write_text(json.dumps(report, indent=2))
        stage.rename(folder)


def check_base_config(config, base):
    expected = dict(model_type="qwen3", num_hidden_layers=36, hidden_size=4096, intermediate_size=12288,
                    num_attention_heads=32, num_key_value_heads=8, head_dim=128, vocab_size=151936,
                    tie_word_embeddings=False, max_position_embeddings=32768)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Expected original HF Qwen3-8B-Base geometry at {base}")


def train(cfg, base, cache, root, device, *, resume=None, stop_after=None, require_8b=True):
    rank = dist.get_rank() if distributed() else 0
    world = dist.get_world_size() if distributed() else 1
    length = cfg["context_length"]
    batch = cfg["global_batch_size"]
    if cfg["micro_batch_size"] != 1 or batch < world or batch % world:
        raise ValueError("micro batch must be 1 and global batch a positive multiple of world size")
    target = cfg["steps"] if stop_after is None else stop_after
    if not 1 <= target <= cfg["steps"] or not 0 <= cfg["warmup_steps"] <= cfg["steps"]:
        raise ValueError("invalid training schedule")
    # A short trial has its own export name and never becomes the default evaluation checkpoint.
    out = root / ("cpt" if target == cfg["steps"] else f"trial-{target}")
    root.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    loader = prolong.loader(cache, length, cfg["seed"], cfg["num_workers"])
    torch.manual_seed(cfg["seed"])
    gpu = device.type == "cuda"
    model = InfiniLM.load_local(base, base=True, segment=cfg["segment_length"],
                                dtype=torch.bfloat16 if gpu else torch.float32)
    c = model.config
    if require_8b:
        check_base_config(c.to_dict(), base)
    if length > c.max_position_embeddings:
        raise ValueError("CPT exceeds the base RoPE context")
    model.train().requires_grad_(True)
    count = sum(p.numel() for p in model.parameters())
    # Avoid all ranks reading the 16 GB of base weights a second time for hashing.
    source = [base_identity(base) if rank == 0 else None]
    if distributed():
        dist.broadcast_object_list(source, src=0)
    identity = dict(recipe=cfg, base_sha256=source[0], world_size=world,
                    cache=str(cache.resolve()), cache_index_sha256=file_hash(cache / "index.json"))
    started = time.monotonic()
    if gpu:
        torch.cuda.reset_peak_memory_stats(device)
    if distributed():
        model = FSDP(model, auto_wrap_policy=partial(transformer_auto_wrap_policy,
                     transformer_layer_cls={SegmentLayer}), use_orig_params=True, device_id=device)
    else:
        model.to(device)
    groups = [[], []]
    for name, p in model.named_parameters():
        no_decay = any(k in name.lower() for k in ("bias", "norm", "ln_"))
        groups[int(no_decay)].append(p)
    optimizer = torch.optim.AdamW([{"params": groups[0], "weight_decay": cfg["weight_decay"]},
                                   {"params": groups[1], "weight_decay": 0.0}],
                                  lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8, foreach=False)
    progress = dict(step=0, input_tokens=0, gpu_hours=0.0, identity=identity)
    if resume:
        progress = restore(resume, model, optimizer, loader, identity, rank)
    begin = progress["step"]
    if begin >= target:
        raise ValueError("resume step must precede requested stop")
    previous_hours = progress["gpu_hours"]
    stream = iter(loader)
    digest = hashlib.sha256()
    for step in range(begin, target):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, cfg)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        step_tokens = 0
        for _ in range(batch // world):
            try:
                block = next(stream)
            except StopIteration:
                stream = iter(loader)
                block = next(stream)
            if tuple(block.shape) != (1, length + 1):
                raise ValueError(f"malformed packed ProLong block: {block.shape}")
            digest.update(block.contiguous().numpy().tobytes())
            block = block.to(device=device, dtype=torch.long)
            loss = model(block[:, :-1], targets=block[:, 1:])
            (loss / (batch // world)).backward()
            loss_sum += loss.detach()
            step_tokens += block.size(1) - 1
        counts = torch.tensor(step_tokens, device=device, dtype=torch.int64)
        if distributed():
            dist.all_reduce(counts)
            dist.all_reduce(loss_sum)
        if not torch.isfinite(loss_sum) or counts.item() != batch * length:
            raise RuntimeError("non-finite loss or inconsistent distributed token count")
        optimizer.step()
        if gpu:
            torch.cuda.synchronize(device)
        progress.update(step=step + 1, input_tokens=progress["input_tokens"] + counts.item(),
                        last_loss=loss_sum.item() / batch,
                        gpu_hours=previous_hours + (time.monotonic() - started) * world / 3600 if gpu else 0.0)
        if rank == 0:
            print(f"step={step+1} input_tokens={progress['input_tokens']} loss={progress['last_loss']:.6f}", flush=True)
        if (step + 1) % cfg["save_interval"] == 0 and step + 1 < cfg["steps"]:
            save_progress(root / f"step-{step+1}", model, optimizer, loader, progress, rank)
    ranks = gather(dict(rank=rank, input_sha256_since_resume=digest.hexdigest(),
                        wall_seconds=time.monotonic() - started,
                        peak_allocated=torch.cuda.max_memory_allocated(device) if gpu else None,
                        peak_reserved=torch.cuda.max_memory_reserved(device) if gpu else None))
    ctx = FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                               FullStateDictConfig(rank0_only=True, offload_to_cpu=gpu)) if distributed() else nullcontext()
    with ctx:
        state = model.state_dict()
    if rank == 0:
        export(out, model.module if distributed() else model, state, base,
               dict(**progress, method="nfini-attention", base=str(base), resume_step=begin, ranks=ranks,
                    trainable_parameters=count, precision="bf16-true" if gpu else "float32",
                    **code_identity(),
                    versions={k: version(k) for k in ("torch", "transformers", "litdata", "safetensors")}))
    return out


def check_storage_marker(path, *, publish=False, timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        try:
            if publish:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as marker:
                    marker.write(b"nfini")
            elif path.read_bytes() != b"nfini":
                raise RuntimeError(f"Unexpected shared probe contents at {path}")
            return
        except FileNotFoundError as exc:
            if time.monotonic() >= deadline:
                action = "publish" if publish else "read"
                raise RuntimeError(f"{socket.gethostname()} rank={os.getenv('RANK')} could not {action} "
                                   f"{path} within {timeout}s; verify the shared output mount on both nodes") from exc
            time.sleep(1)


def check_job(base, raw, cache, root, device):
    from litdata import StreamingDataset, StreamingDataLoader, TokensLoader  # verify the training loader imports
    from safetensors import safe_open
    from transformers import AutoTokenizer

    rank = dist.get_rank() if dist.is_initialized() else 0
    local = int(os.environ["LOCAL_WORLD_SIZE"])
    expected = int(os.environ.get("MA_NUM_HOSTS", "1")) * local
    if local > torch.cuda.device_count() or not torch.cuda.is_bf16_supported():
        raise ValueError(f"Expected {local} visible CUDA devices with BF16")
    value = torch.ones(1, device=device, dtype=torch.bfloat16)
    if distributed():
        dist.all_reduce(value)
    if value.item() != expected:
        raise RuntimeError(f"NCCL sum failed for expected world size {expected}")
    if device.index == 0:
        print(f"[rank {rank}] NCCL all_reduce passed, world_size={expected}; checking shared storage", flush=True)
    # Each host must read a file actually created by the leader on shared storage.
    marker = [None]
    if rank == 0:
        marker[0] = f"storage-check-{uuid.uuid4().hex}"
        check_storage_marker(root / marker[0], publish=True)
    if distributed():
        dist.broadcast_object_list(marker, src=0)
    try:
        if device.index == 0:
            check_storage_marker(root / marker[0])
        if distributed():
            dist.barrier()  # All hosts have read the marker before the leader can remove it.
        if device.index == 0:
            if (root / "cpt").exists() or any(root.glob("step-*")):
                raise FileExistsError(f"Existing training output at {root}; use a fresh NFINI_OUTPUT_ROOT "
                                      "or use the standalone resume/evaluation entry point")
            config = json.loads((base / "config.json").read_text())
            check_base_config(config, base)
            if "infini_segment_length" in config:
                raise ValueError(f"Expected original HF Qwen3-8B-Base at {base}")
            if not (base / "tokenizer_config.json").is_file():
                raise FileNotFoundError(base / "tokenizer_config.json")
            AutoTokenizer.from_pretrained(base, local_files_only=True)
            index = base / "model.safetensors.index.json"
            shards = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
            for shard in shards:
                with safe_open(base / shard, framework="pt", device="cpu") as weights:
                    if not weights.keys():
                        raise ValueError(f"Empty weights: {base / shard}")
            if (cache / "index.json").is_file():
                prolong.check_chunks(cache)
                source = cache
            elif os.environ.get("NFINI_PROLONG_CACHE_ROOT") or cache.exists() or not next(prolong.raw_files(raw), None):
                raise FileNotFoundError(f"No usable ProLong data at cache={cache}, raw={raw}")
            else:
                source = raw
            print(f"[node {rank // local}] {torch.cuda.get_device_name(device)} | torch {torch.__version__} | "
                  f"CUDA {torch.version.cuda} | NCCL {torch.cuda.nccl.version()} | transformers {version('transformers')} | "
                  f"litdata {version('litdata')} | base={base} | ProLong={source}",
                  flush=True)
    finally:
        if rank == 0:
            (root / marker[0]).unlink(missing_ok=True)


@record
def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("prepare", "cpt", "check"))
    cli.add_argument("--config", type=Path, default=Path("configs/nfini_cpt.yaml"))
    cli.add_argument("--steps", type=int, help="stop at this optimizer step with the original LR schedule")
    cli.add_argument("--resume", type=Path, help="complete step directory from this recipe")
    cli.add_argument("--print-config", action="store_true")
    args = cli.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    base = Path(os.environ.get("NFINI_QWEN_BASE", cfg["base"])).resolve()
    raw = Path(os.environ.get("NFINI_PROLONG_DATA", cfg["prolong"])).resolve()
    root = Path(os.environ.get("NFINI_OUTPUT_ROOT", cfg["output_root"])).resolve()
    if args.print_config:
        print(json.dumps({**cfg, "base": str(base), "prolong": str(raw), "output_root": str(root),
                          "target_input_tokens": cfg["context_length"] * cfg["global_batch_size"] * (args.steps or cfg["steps"])}, indent=2))
        return
    name = prolong.cache_name(base, cfg["context_length"])
    override = os.environ.get("NFINI_PROLONG_CACHE_ROOT")
    cache = Path(override or raw) / name
    if not override and not cache.exists():
        cache = root / "prolong_cache" / name
    if args.stage == "prepare":
        if (cache / "index.json").is_file():
            print(f"Using existing packed data: {cache}")
        elif override:
            raise FileNotFoundError(f"Missing shared cache index in {cache}")
        else:
            prolong.prepare(raw, cache, base, cfg["context_length"], max(1, cfg["num_workers"]))
        return
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
    try:
        if args.stage == "check":
            check_job(base, raw, cache, root, device)
            return
        train(cfg, base, cache, root, device, resume=args.resume, stop_after=args.steps)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
