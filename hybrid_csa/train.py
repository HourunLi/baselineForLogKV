"""Offline, adapter-only CSA adaptation. Preparation and training are explicit operations.

Each stage starts a fresh optimizer; --adapter continues weights, not an exact
optimizer/data-loader resume. No NIAH examples are used or generated here.
"""

import argparse
from contextlib import nullcontext
from functools import partial
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import yaml

from .config import BASELINE_ROOT, environment_path, expand_env_vars, require_project_file

DEFAULT_CONFIG = BASELINE_ROOT / "configs/hybrid_csa/train_qwen1.7b_32k.yaml"
STAGES = ("compressor", "indexer", "sparse")
DATA_MANIFEST = "hybrid_csa_data.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute(path):
    path = Path(str(path)).expanduser()
    return (path if path.is_absolute() else BASELINE_ROOT / path).resolve()


def resolve_config(path=DEFAULT_CONFIG, overrides=None):
    path = require_project_file(path)
    cfg = expand_env_vars(yaml.safe_load(path.read_text()) or {})
    for field, names in {
        "base_checkpoint_dir": ("HYBRID_CSA_BASE_CHECKPOINT", "BASELINE_BASE_CHECKPOINT"),
        "data_dir": ("HYBRID_CSA_DATA_DIR", "BASELINE_DATA_DIR"),
        "tokenized_dir": ("HYBRID_CSA_TOKENIZED_DIR",),
        "output_dir": ("HYBRID_CSA_TRAIN_OUTPUT", "BASELINE_OUTPUT_ROOT"),
    }.items():
        value = environment_path(*names)
        if value is not None:
            cfg[field] = value
    cfg.update({key: str(value) for key, value in (overrides or {}).items() if value is not None})
    required = {"base_checkpoint_dir", "data_dir", "tokenized_dir", "output_dir", "context_length",
                "global_batch_size", "micro_batch_size", "seed", "num_workers", "prepare_workers",
                "precision", "activation_checkpointing", "loss_chunk_size", "weight_decay", "grad_clip",
                "indexer_loss_weight", "warmup_steps", "settings", "stages"}
    if set(cfg) != required:
        raise ValueError(f"Missing keys: {sorted(required - cfg.keys())}; unknown keys: {sorted(cfg.keys() - required)}")
    for name in ("base_checkpoint_dir", "data_dir", "tokenized_dir", "output_dir"):
        cfg[name] = str(Path(cfg[name]).expanduser()) if Path(cfg[name]).expanduser().is_absolute() else str(absolute(cfg[name]))
    if not isinstance(cfg["context_length"], int) or not 2048 < cfg["context_length"] <= 32768:
        raise ValueError("Adaptation context must exceed top-k * 4 (2048) and stay within the 32768 window")
    for key in ("global_batch_size", "micro_batch_size", "loss_chunk_size", "prepare_workers"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["num_workers"] < 0 or cfg["warmup_steps"] < 0 or cfg["grad_clip"] <= 0:
        raise ValueError("num_workers/warmup_steps must be nonnegative and grad_clip positive")
    if cfg["precision"] != "bf16":
        raise ValueError("This recipe requires bf16; train adapters in fp32 under bf16 autocast")
    from .settings import HybridCSASettings
    settings = HybridCSASettings(**cfg["settings"])
    if cfg["context_length"] <= settings.index_topk * settings.compress_ratio:
        raise ValueError("Adaptation context must exercise sparse selection: context_length > index_topk * compress_ratio")
    if [stage["name"] for stage in cfg["stages"]] != list(STAGES):
        raise ValueError(f"Stages must be {STAGES} in order")
    for stage in cfg["stages"]:
        if set(stage) != {"name", "steps", "learning_rate"} or stage["steps"] < 1:
            raise ValueError("Each stage needs name, positive steps, and learning_rate")
        stage["learning_rate"] = float(stage["learning_rate"])
        if not 0 < stage["learning_rate"] < 1:
            raise ValueError("learning_rate must be between 0 and 1")
    protected = [absolute(cfg[k]) for k in ("base_checkpoint_dir", "data_dir")]
    protected += [BASELINE_ROOT / "hybrid_csa", Path("/data/lihourun/semanticLogKV")]
    protected += [Path(os.environ[k]).resolve() for k in ("HF_HOME", "HF_DATASETS_CACHE", "RULER_CACHE_DIR")
                  if os.environ.get(k)]
    outputs = [absolute(cfg[k]) for k in ("tokenized_dir", "output_dir")]
    for destination in outputs:
        if any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in protected):
            raise ValueError(f"Output overlaps original assets or runtime: {destination}")
    if outputs[0].is_relative_to(outputs[1]) or outputs[1].is_relative_to(outputs[0]):
        raise ValueError("tokenized_dir and output_dir must be separate directories")
    return cfg


def tokenizer_identity(directory):
    names = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "generation_config.json")
    files = {name: sha256(Path(directory) / name) for name in names if (Path(directory) / name).is_file()}
    if not {"tokenizer.json", "tokenizer.model"} & files.keys():
        raise FileNotFoundError(f"No tokenizer in {directory}")
    return files


def text_from_record(record, source):
    """Explicit local corpus schema: never skip a malformed/empty document silently."""
    if not isinstance(record, dict):
        raise ValueError(f"Expected a document object in {source}")
    fields = [name for name in ("text", "chapter", "content", "code")
              if isinstance(record.get(name), str) and record[name].strip()]
    if fields:
        return record[fields[0]].strip()
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        parts = []
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError(f"Invalid messages document in {source}")
            parts.append(f"### {message.get('role', 'user')}\n{message['content']}")
        return "\n\n".join(parts)
    raise ValueError(f"No nonempty text/chapter/content/code/messages in {source}")


def is_litdata_output(directory):
    """A directory LitData's optimize wrote: its index.json lists chunks and the writer config."""
    index = Path(directory) / "index.json"
    try:
        record = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(record, dict) and isinstance(record.get("chunks"), list) and isinstance(record.get("config"), dict)


def corpus_files(source):
    """Raw documents under ``source`` and the LitData output directories skipped there.

    A shared corpus can hold another run's prepared chunks (SemanticLogKV writes litdata_<key>_ctx<N>/ into its
    data_dir); their index.json is not a document. Symlinked directories are not followed.
    """
    files, skipped = [], []
    for directory, subdirectories, names in os.walk(source):
        directory = Path(directory)
        if "index.json" in names and is_litdata_output(directory):
            skipped.append(directory)
            subdirectories.clear()
            continue
        subdirectories.sort()
        files += [directory / name for name in names if Path(name).suffix in {".parquet", ".jsonl", ".json"}]
    return sorted(files), sorted(skipped)


def tokenize_file(path, tokenizer_dir):
    from ._vendor.semanticlogkv.litgpt.tokenizer import Tokenizer

    tokenizer = Tokenizer(tokenizer_dir)
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        columns = [name for name in ("text", "chapter", "content", "code", "messages")
                   if name in parquet.schema_arrow.names]
        if not columns:
            raise ValueError(f"Missing supported document column in {path}")
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                yield tokenizer.encode(text_from_record(row, str(path)), bos=False, eos=True)
    else:
        with path.open(encoding="utf-8") as source:
            first = next((line for line in source if line.strip()), "")
            source.seek(0)
            if first.lstrip().startswith("["):
                rows = json.load(source)
                for row in rows:
                    yield tokenizer.encode(text_from_record(row, str(path)), bos=False, eos=True)
            else:
                for number, line in enumerate(source, 1):
                    if line.strip():
                        row = json.loads(line)
                        yield tokenizer.encode(text_from_record(row, f"{path}:{number}"), bos=False, eos=True)


def prepare_data(cfg):
    """Single-process explicit preparation into a new project-owned directory."""
    from litdata import optimize
    from litdata.streaming import TokensLoader

    import torch.distributed as dist
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 and not (dist.is_initialized() and dist.get_rank() == 0):
        raise ValueError("Run --prepare-data once, outside torchrun")
    source = Path(cfg["data_dir"])
    if not source.is_dir():
        raise FileNotFoundError(f"Local corpus does not exist: {source}")
    files, skipped = corpus_files(source)
    for directory in skipped:
        print(f"Skipping prepared LitData chunks, not raw documents: {directory}", flush=True)
    if not files:
        raise FileNotFoundError(f"No local parquet/jsonl/json documents in {source}"
                                + (f" (skipped LitData outputs: {[str(d) for d in skipped]})" if skipped else ""))
    destination = Path(cfg["tokenized_dir"])
    if (destination / DATA_MANIFEST).exists():
        raise FileExistsError(f"Refusing to overwrite tokenized data: {destination}")
    if destination.exists() and any(destination.iterdir()):
        # No manifest: an earlier preparation stopped before finishing, so nothing in it can be used.
        print(f"Removing an incomplete earlier preparation: {destination}", flush=True)
        shutil.rmtree(destination)
    record = {"schema": 1, "context_length": cfg["context_length"],
              "tokenizer": tokenizer_identity(cfg["base_checkpoint_dir"]),
              "documents": {str(p.relative_to(source)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files},
              "data_dir": str(source), "skipped_litdata_outputs": [str(d.relative_to(source)) for d in skipped],
              "text_fields_in_priority_order": ["text", "chapter", "content", "code", "messages"],
              "bos": False, "eos": True, "preparer_sha256": sha256(__file__)}
    destination.mkdir(parents=True, exist_ok=True)
    optimize(fn=partial(tokenize_file, tokenizer_dir=cfg["base_checkpoint_dir"]),
             inputs=[str(p) for p in files], output_dir=str(destination),
             num_workers=min(cfg["prepare_workers"], len(files)), chunk_bytes="200MB",
             item_loader=TokensLoader(block_size=cfg["context_length"] + 1))
    record["index_sha256"] = sha256(destination / "index.json")
    (destination / DATA_MANIFEST).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(f"Prepared offline token data: {destination}")


def validate_prepared_data(cfg, verify_source=False):
    directory = Path(cfg["tokenized_dir"])
    manifest = directory / DATA_MANIFEST
    if not manifest.is_file() or not (directory / "index.json").is_file():
        raise FileNotFoundError(f"Missing prepared data at {directory}; run python -m hybrid_csa.train --prepare-data first")
    record = json.loads(manifest.read_text())
    if record["tokenizer"] != tokenizer_identity(cfg["base_checkpoint_dir"]):
        raise ValueError("Prepared data tokenizer differs from original Base tokenizer")
    if record["context_length"] != cfg["context_length"] or record["index_sha256"] != sha256(directory / "index.json"):
        raise ValueError("Prepared data length/index changed; prepare a new directory")
    if Path(record["data_dir"]).resolve() != Path(cfg["data_dir"]).resolve():
        raise ValueError("Prepared data came from a different raw corpus; choose a new tokenized_dir")
    if verify_source:
        # The same files with the same sizes. Their SHA-256 were recorded once at preparation; re-reading a
        # 100+ GB shared corpus before every reuse would cost more than the run itself.
        source = Path(cfg["data_dir"])
        files, _ = corpus_files(source)
        actual = {str(p.relative_to(source)): p.stat().st_size for p in files}
        if actual != {name: entry["bytes"] for name, entry in record["documents"].items()}:
            raise ValueError("Raw corpus files or sizes changed since preparation; choose a new tokenized_dir")
    return record


def prepare_distributed(cfg):
    """One process per node; rank zero prepares, all nodes receive success or failure."""
    from datetime import timedelta
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("gloo", timeout=timedelta(hours=24))
    try:
        if world > 1:
            configurations = [None] * world
            dist.all_gather_object(configurations, cfg)
            if any(item != cfg for item in configurations):
                raise ValueError("Nodes resolved different training configs or asset/output paths")
        result = [None]
        if world == 1 or dist.get_rank() == 0:
            try:
                directory = Path(cfg["tokenized_dir"])
                if (directory / DATA_MANIFEST).is_file():
                    validate_prepared_data(cfg, verify_source=True)
                    print(f"Reusing verified prepared data: {directory}", flush=True)
                else:
                    prepare_data(cfg)
                result[0] = {"ok": True}
            except Exception as error:
                result[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        if world > 1:
            dist.broadcast_object_list(result, src=0)
        if not result[0]["ok"]:
            raise RuntimeError(f"Shared data preparation failed: {result[0]['error']}")
        # Every node must see the completed shared files before it can train.
        error_message = None
        try:
            validate_prepared_data(cfg)
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
        if world > 1:
            errors = [None] * world
            dist.all_gather_object(errors, error_message)
        else:
            errors = [error_message]
        if any(errors):
            raise RuntimeError(f"Prepared files are not readable on every node: {errors}")
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def make_loader(cfg):
    from litdata import StreamingDataLoader, StreamingDataset, TokensLoader

    record = validate_prepared_data(cfg)
    directory = Path(cfg["tokenized_dir"])
    # LitData detects the initialized torch distributed world and shards across ranks
    # and loader workers. A second DistributedSampler would incorrectly shard twice.
    dataset = StreamingDataset(input_dir=str(directory), item_loader=TokensLoader(block_size=cfg["context_length"] + 1),
                               shuffle=True, seed=cfg["seed"], drop_last=True)
    loader = StreamingDataLoader(dataset, batch_size=cfg["micro_batch_size"], num_workers=cfg["num_workers"],
                                 pin_memory=True, drop_last=True)
    return loader, record


def accumulation_steps(global_batch_size, micro_batch_size, world_size):
    divisor = micro_batch_size * world_size
    if global_batch_size % divisor:
        raise ValueError(f"global_batch_size {global_batch_size} must be divisible by micro_batch_size "
                         f"{micro_batch_size} * WORLD_SIZE {world_size}")
    return global_batch_size // divisor


def stage_objective(name, lm_loss, indexer_kl, weight):
    if name == "indexer":
        return indexer_kl
    if name == "compressor":
        return lm_loss
    return lm_loss + weight * indexer_kl


def learning_rate(base, step, steps, warmup):
    warmup = min(warmup, steps)
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(steps - warmup - 1, 1)
    return base * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)


def train_stage(model, cfg, stage, loader, device, world_size=1, rank=0):
    """The same loop supports tiny CPU checks and formal BF16 CUDA adaptation."""
    import torch
    from torch.nn.parallel import DistributedDataParallel

    model.set_stage(stage["name"])
    parameters = [p for _, p in model.adapter_named_parameters() if p.requires_grad]
    if not parameters:
        raise ValueError(f"No trainable adapter parameters in stage {stage['name']}")
    for parameter in parameters:
        if parameter.dtype != torch.float32:
            raise ValueError("Trainable adapter weights must remain fp32")
    optimizer = torch.optim.AdamW(parameters, lr=stage["learning_rate"], weight_decay=cfg["weight_decay"])

    class LossForward(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, inputs, targets):
            return self.wrapped.training_loss(inputs, targets, loss_chunk_size=cfg["loss_chunk_size"],
                                              activation_checkpointing=cfg["activation_checkpointing"])

    wrapped = LossForward(model)
    if world_size > 1:
        wrapped = DistributedDataParallel(wrapped, device_ids=[device.index], broadcast_buffers=False)
    wrapped.train()
    accumulation = accumulation_steps(cfg["global_batch_size"], cfg["micro_batch_size"], world_size)
    iterator = iter(loader)
    tokens = 0
    started = time.monotonic()
    history = []
    for step in range(stage["steps"]):
        optimizer.zero_grad(set_to_none=True)
        rate = learning_rate(stage["learning_rate"], step, stage["steps"], cfg["warmup_steps"])
        for group in optimizer.param_groups:
            group["lr"] = rate
        sums = torch.zeros(3, device=device)
        for micro in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError("Prepared corpus exhausted; training never silently repeats documents. Reduce steps or prepare more data.") from error
            batch = batch.to(device=device, dtype=torch.long, non_blocking=True)
            if batch.shape != (cfg["micro_batch_size"], cfg["context_length"] + 1):
                raise ValueError(f"Expected packed token batch {(cfg['micro_batch_size'], cfg['context_length'] + 1)}, got {tuple(batch.shape)}")
            sync = wrapped.no_sync() if world_size > 1 and micro + 1 < accumulation else nullcontext()
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with sync, autocast:
                lm, kl = wrapped(batch[:, :-1], batch[:, 1:])
                loss = stage_objective(stage["name"], lm, kl, cfg["indexer_loss_weight"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {stage['name']} loss at step {step + 1}")
                (loss / accumulation).backward()
            sums += torch.stack((loss.detach(), lm.detach(), kl.detach())) / accumulation
            tokens += batch[:, :-1].numel() * world_size
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        if world_size > 1:
            torch.distributed.all_reduce(sums)
            sums /= world_size
        entry = {"stage": stage["name"], "step": step + 1, "loss": sums[0].item(),
                 "lm_loss": sums[1].item(), "indexer_kl": sums[2].item(), "lr": rate,
                 "grad_norm": grad_norm.item(), "global_tokens": tokens}
        history.append(entry)
        if rank == 0:
            print(json.dumps(entry), flush=True)
    elapsed = time.monotonic() - started
    return {"stage": stage["name"], "steps": stage["steps"], "global_tokens": tokens,
            "wall_seconds": elapsed, "gpu_hours": elapsed * world_size / 3600 if device.type == "cuda" else 0,
            "optimizer": "AdamW, fresh state at every stage", "history": history}


def run(cfg, adapter=None, start_stage="compressor"):
    import torch
    from ._vendor.semanticlogkv.litgpt.config import Config
    from ._vendor.semanticlogkv.litgpt.model import GPT
    from .checkpoint import load_adapter, save_adapter
    from .model import HybridCSAModel
    from .settings import HybridCSASettings

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Formal training requires CUDA with BF16 support; use tiny CPU tests for implementation checks")
    if start_stage != "compressor" and adapter is None:
        raise ValueError("Starting after compressor requires --adapter from a previous adaptation stage")
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group("nccl")
    try:
        accumulation_steps(cfg["global_batch_size"], cfg["micro_batch_size"], world)
        torch.manual_seed(cfg["seed"])
        base = Path(cfg["base_checkpoint_dir"])
        config = Config.from_file(base / "model_config.yaml")
        if cfg["context_length"] > config.block_size:
            raise ValueError("Training context exceeds original model context")
        gpt = GPT(config)
        state = torch.load(base / "lit_model.pth", map_location="cpu", weights_only=True, mmap=True)
        gpt.load_state_dict(state, strict=True)
        del state
        gpt.requires_grad_(False)
        gpt.to(device=device, dtype=torch.bfloat16)
        model = HybridCSAModel(gpt, HybridCSASettings(**cfg["settings"]))
        model.to(device=device)
        for _, parameter in model.adapter_named_parameters():
            parameter.data = parameter.data.float()
        parent_adapter = None
        if adapter:
            parent_adapter = load_adapter(model, adapter, base_checkpoint_dir=base, require_sparse=False)
            previous = {"compressor": {"compressor"}, "indexer": {"compressor", "indexer"}, "sparse": {"indexer", "sparse"}}
            if model.stage not in previous[start_stage]:
                raise ValueError(f"Cannot start {start_stage} from a {model.stage} adapter")
        loader, data_record = make_loader(cfg)
        output = Path(cfg["output_dir"])
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
            (output / "resolved_train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        if world > 1:
            torch.distributed.barrier()
        versions = {}
        for name in ("torch", "litdata", "lightning", "tokenizers", "pyarrow", "PyYAML"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASELINE_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        record = {"resolved_config": cfg, "versions": versions, "python": platform.python_version(),
                  "cuda": torch.version.cuda, "world_size": world, "device": torch.cuda.get_device_name(device),
                  "git_commit": revision, "code_sha256": {str(p.relative_to(BASELINE_ROOT)): sha256(p) for p in sorted((BASELINE_ROOT / "hybrid_csa").rglob("*.py"))},
                  "tokenized_data": data_record, "adapter_input": parent_adapter,
                  "resume_semantics": "adapter weights only; optimizer and data iterator restart; not exact resume",
                  "frozen_original_base": True, "training_corpus_purpose": "general language modeling, no NIAH tuning",
                  "adapter_parameter_count": sum(p.numel() for _, p in model.adapter_named_parameters()),
                  "original_parameter_count": sum(p.numel() for p in model.parameters()) - sum(p.numel() for _, p in model.adapter_named_parameters()),
                  "completed_stages": [], "optimizer_steps": 0, "stage_steps": {name: 0 for name in STAGES}}
        if rank == 0:
            (output / "training_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        for stage in cfg["stages"][STAGES.index(start_stage):]:
            result = train_stage(model, cfg, stage, loader, device, world, rank)
            record["completed_stages"].append(result)
            record["optimizer_steps"] += result["steps"]
            record["stage_steps"][stage["name"]] += result["steps"]
            if rank == 0:
                save_adapter(model, output / f"{stage['name']}.pt", base_checkpoint_dir=base, training_record=record)
                with (output / "metrics.jsonl").open("a") as stream:
                    for entry in result["history"]:
                        stream.write(json.dumps(entry) + "\n")
                (output / "training_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            if world > 1:
                torch.distributed.barrier()
    finally:
        if world > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--print-config", action="store_true")
    operation.add_argument("--prepare-data", action="store_true")
    operation.add_argument("--prepare-distributed", action="store_true",
                           help="Launcher-only: coordinate shared preparation with one process per node")
    for name in ("base_checkpoint_dir", "data_dir", "tokenized_dir", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--start-stage", choices=STAGES, default="compressor")
    args = parser.parse_args(argv)
    cfg = resolve_config(args.config, {name: getattr(args, name).resolve() if getattr(args, name) else None for name in
                                     ("base_checkpoint_dir", "data_dir", "tokenized_dir", "output_dir")})
    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    else:
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        if args.prepare_distributed:
            prepare_distributed(cfg)
        elif args.prepare_data:
            prepare_data(cfg)
        else:
            run(cfg, args.adapter, args.start_stage)


if __name__ == "__main__":
    main()
