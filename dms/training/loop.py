# Written for the SemanticLogKV baseline (not part of NVIDIA's code); see docs/UPSTREAM.md.
"""DMS distillation training without HF Trainer or accelerate, so it runs in SemanticLogKV's torch218.

The upstream recipe trains with transformers 4.57.3's Trainer and accelerate 1.4.0. For the settings the
training YAMLs use, this loop does what they do:

- Data: every epoch, ``torch.randperm`` drawn from a generator seeded with ``seed + epoch`` (accelerate's
  SeedableRandomSampler; ``seed`` is also the global seed, set first). With N processes, process r takes
  positions r, r+N, ... and the epoch is padded from its start so every process gets the same number of
  batches (BatchSamplerShard, even_batches). Batches hold one packed sample of ``data.max_length`` tokens.
- Updates: each epoch is cut into groups of ``gradient_accumulation_steps`` batches, the last one possibly
  shorter. Each loss is divided by its group's size; gradients are synchronized on the group's last batch
  only (no_sync before it), clipped to ``max_grad_norm``, and fused AdamW with a constant learning rate
  steps once. DMS noise seeds and the compression schedule read the update counter (``global_step``) and the
  position in the group (``grad_acc_step``) where DMSGradAccCallback set them.
- Precision and sharding: bf16 autocast around the forward. With ``fsdp`` set, torch FSDP with the settings
  accelerate derives from the same YAML: bf16 parameters, gradient reduction and buffers; optional CPU
  offload; one unit per decoder layer; non-reentrant activation checkpointing per layer; no prefetch;
  limit_all_gathers; use_orig_params; sync_module_states. Without it, DDP across processes, or the plain
  module in one process.
- Logs: per logging step, the mean loss since the last log, the gradient norm, the learning rate, the epoch
  and the last batch's DMS metrics averaged over processes (``gl_*``), as ModifiedTrainer logged them.

Deliberate differences: a checkpoint holds the trained adapter weights, the optimizer state and the step
instead of a full model copy (the frozen weights are the prepared Base, verified again on resume), and resume
continues exactly where the batch sequence stopped (HF Trainer skips ``global_step % (batches_per_epoch //
accumulation)`` groups, which lands elsewhere when an epoch's batch count is not a multiple of the
accumulation). Values other than the recipe's are refused rather than ignored.
"""

import contextlib
import functools
import itertools
import json
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field, fields
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from dms.logging import get_logger

logger = get_logger("Loop")

CHECKPOINT_NAME = re.compile(r"checkpoint-(\d+)")
STUDENT_PREFIX = "student_model."

# HF TrainingArguments fields whose other values this loop does not implement.
_FIXED_SETTINGS = {
    "per_device_train_batch_size": 1, "weight_decay": 0.0, "warmup_steps": 0, "lr_scheduler_type": "constant",
    "save_strategy": "steps", "logging_strategy": "steps", "gradient_checkpointing": False, "tf32": False,
    "bf16": True, "report_to": "none",
}
_FSDP_OPTIONS = {"full_shard", "offload", "auto_wrap"}
_FIXED_FSDP_CONFIG = {"use_orig_params": True, "state_dict_type": "FULL_STATE_DICT"}
_FSDP_CONFIG_DEFAULTS = {"sync_module_states": True, "activation_checkpointing": False}


@dataclass
class TrainerSettings:
    """The ``trainer:`` section of a training YAML, with HF TrainingArguments names, defaults and meanings."""

    output_dir: str
    max_steps: int
    learning_rate: float
    gradient_accumulation_steps: int = 1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    seed: int = 42
    save_steps: int = 500
    save_total_limit: int | None = None
    logging_steps: int = 500
    fsdp: str = ""
    fsdp_config: dict = field(default_factory=dict)
    resume_from_checkpoint: str | None = None
    run_name: str | None = None
    # Global rank, which DMS noise seeding reads as TrainingArguments.process_index.
    process_index: int = field(default=0, init=False, compare=False)

    @classmethod
    def from_config(cls, section: dict) -> "TrainerSettings":
        section = dict(section)
        for key, value in _FIXED_SETTINGS.items():
            if key in section and section.pop(key) != value:
                raise ValueError(f"trainer.{key} must be {value!r}; other values are not implemented")
        unknown = sorted(set(section) - {item.name for item in fields(cls) if item.init})
        if unknown:
            raise ValueError(f"Unsupported trainer settings: {unknown}")
        settings = cls(**section)
        settings.fsdp_config = settings._checked_fsdp_config()
        for name in ("max_steps", "gradient_accumulation_steps", "save_steps", "logging_steps"):
            if getattr(settings, name) < 1:
                raise ValueError(f"trainer.{name} must be positive")
        if settings.save_total_limit is not None and settings.save_total_limit < 1:
            raise ValueError("trainer.save_total_limit must be positive or null")
        return settings

    @property
    def fsdp_options(self) -> set[str]:
        return set(self.fsdp.split())

    def _checked_fsdp_config(self) -> dict:
        options, config = self.fsdp_options, dict(self.fsdp_config)
        if not options:
            if config:
                raise ValueError("trainer.fsdp_config needs trainer.fsdp")
            return {}
        if not options <= _FSDP_OPTIONS or not {"full_shard", "auto_wrap"} <= options:
            raise ValueError(f"trainer.fsdp must be 'full_shard auto_wrap', optionally with 'offload': {self.fsdp!r}")
        for key, value in _FIXED_FSDP_CONFIG.items():
            if config.setdefault(key, value) != value:
                raise ValueError(f"trainer.fsdp_config.{key} must be {value!r}; other values are not implemented")
        for key, value in _FSDP_CONFIG_DEFAULTS.items():
            config.setdefault(key, value)
        layers = config.get("transformer_layer_cls_to_wrap")
        config["transformer_layer_cls_to_wrap"] = [layers] if isinstance(layers, str) else layers
        unknown = sorted(set(config) - set(_FIXED_FSDP_CONFIG) - set(_FSDP_CONFIG_DEFAULTS)
                         - {"transformer_layer_cls_to_wrap"})
        if unknown or not config["transformer_layer_cls_to_wrap"]:
            raise ValueError(f"trainer.fsdp_config needs transformer_layer_cls_to_wrap and supports no {unknown}")
        return config


@dataclass
class Runtime:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self):
        if dist.is_initialized():
            dist.barrier()


def init_runtime() -> Runtime:
    """Join the torchrun process group (also a group of one, as accelerate does), or run in this process."""
    cuda = torch.cuda.is_available()
    if "LOCAL_RANK" not in os.environ:
        return Runtime(device=torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu"))
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl" if cuda else "gloo")
    return Runtime(dist.get_rank(), local_rank, dist.get_world_size(), device)


def epoch_indices(num_samples: int, seed: int, epoch: int, world_size: int, rank: int) -> list[int]:
    """This process's sample indices for one epoch, in the order HF Trainer with accelerate gives them."""
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    order = torch.randperm(num_samples, generator=generator).tolist()
    padding = -num_samples % world_size
    order += (order * world_size)[:padding]
    return order[rank::world_size]


def update_groups(num_samples: int, seed: int, world_size: int, rank: int, accumulation: int):
    """Yield ``(epoch progress after the group, sample indices)`` for every optimizer update, without end."""
    for epoch in itertools.count():
        indices = epoch_indices(num_samples, seed, epoch, world_size, rank)
        for start in range(0, len(indices), accumulation):
            end = min(start + accumulation, len(indices))
            yield epoch + end / len(indices), indices[start:end]


def checkpoints(output_dir: Path) -> list[Path]:
    """Complete checkpoints in ``output_dir``, oldest step first."""
    found = [(int(match.group(1)), path) for path in Path(output_dir).glob("checkpoint-*")
             if (match := CHECKPOINT_NAME.fullmatch(path.name)) and path.is_dir()]
    return [path for _, path in sorted(found)]


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainState:
    global_step: int = 0
    epoch: float = 0.0
    log_history: list[dict] = field(default_factory=list)


class DMSTrainer:
    """Train a CombinedModel's DMS adapters; exposes ``state.global_step`` and ``args.process_index``."""

    def __init__(self, combined_model, train_dataset, settings: TrainerSettings, runtime: Runtime,
                 max_length: int, trainer_state, resume_from: str | None = None):
        self.combined, self.dataset, self.args, self.runtime = combined_model, train_dataset, settings, runtime
        self.max_length, self.trainer_state = max_length, trainer_state
        settings.process_index = runtime.rank
        self.state = TrainState()
        self.output_dir = Path(settings.output_dir)
        trainable = [(name, param.numel()) for name, param in combined_model.named_parameters() if param.requires_grad]
        if not trainable:
            raise ValueError("No trainable DMS parameters")
        # Counted before wrapping: FSDP leaves each process a shard of every parameter.
        self.trainable, self.trainable_numel = [name for name, _ in trainable], sum(n for _, n in trainable)
        _set_seed(settings.seed)
        if runtime.device.type == "cuda":  # tf32: false, applied as TrainingArguments applies it
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        trainer_state.set_trainer(self)
        resumed = self._load_checkpoint(Path(resume_from)) if resume_from else None
        self.model = self._wrap(combined_model)
        self.optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad], lr=settings.learning_rate,
            betas=(settings.adam_beta1, settings.adam_beta2), eps=settings.adam_epsilon, weight_decay=0.0,
            fused=True)
        if resumed is not None:
            self._load_optimizer_state(resumed)

    # ----- distribution -------------------------------------------------------------------------------

    @property
    def is_fsdp(self) -> bool:
        return bool(self.args.fsdp_options)

    def _wrap(self, model):
        if self.is_fsdp:
            return self._wrap_fsdp(model)
        model = model.to(self.runtime.device)
        if self.runtime.world_size == 1:
            return model
        # HF Trainer asks for find_unused_parameters on a model that is not a PreTrainedModel.
        device_ids = [self.runtime.device] if self.runtime.device.type == "cuda" else None
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids,
                                                         find_unused_parameters=True)

    def _wrap_fsdp(self, model):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
        from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        config = self.args.fsdp_config
        names = set(config["transformer_layer_cls_to_wrap"])
        layer_classes = {type(module) for module in model.modules() if type(module).__name__ in names}
        if {cls.__name__ for cls in layer_classes} != names:
            raise ValueError(f"Model has no layer class among {sorted(names)}")
        policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_classes)
        # Model stays where it was loaded (CPU); FSDP moves each unit to device_id. Mixed precision is
        # accelerate's bf16 policy, buffers included.
        wrapped = FSDP(
            model, sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params="offload" in self.args.fsdp_options),
            auto_wrap_policy=policy,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
                                           buffer_dtype=torch.bfloat16),
            sync_module_states=config["sync_module_states"], backward_prefetch=None, forward_prefetch=False,
            use_orig_params=True, limit_all_gathers=True, device_id=self.runtime.device,
        )
        if config["activation_checkpointing"]:
            apply_activation_checkpointing(
                wrapped, auto_wrap_policy=policy,
                checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper,
                                                        checkpoint_impl=CheckpointImpl.NO_REENTRANT))
        return wrapped

    def _fsdp_full_state(self, rank0_only: bool = True):
        from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        offload = self.runtime.device.type != "cpu"  # gathered weights go to CPU (already there on a CPU run)
        return FSDP.state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=offload, rank0_only=rank0_only),
            FullOptimStateDictConfig(offload_to_cpu=offload, rank0_only=rank0_only))

    def full_state_dict(self) -> dict[str, torch.Tensor]:
        """The CombinedModel's complete weights on the main process (empty elsewhere under FSDP)."""
        if self.is_fsdp:
            with self._fsdp_full_state():
                return self.model.state_dict()
        return self.combined.state_dict()

    def _adapter_state(self) -> dict[str, torch.Tensor]:
        """The trained weights, complete, on the main process."""
        if self.is_fsdp:
            state = self.full_state_dict()
            return {name: state[name].float().clone() for name in self.trainable} if self.runtime.is_main else {}
        parameters = dict(self.combined.named_parameters())
        return {name: parameters[name].detach().float().cpu().clone() for name in self.trainable}

    def _all_mean(self, values: list[float]) -> list[float]:
        if not dist.is_initialized():
            return values
        tensor = torch.tensor(values, dtype=torch.float32, device=self.runtime.device)
        dist.all_reduce(tensor)  # gloo has no AVG
        return (tensor / self.runtime.world_size).tolist()

    # ----- training -----------------------------------------------------------------------------------

    def _batch(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long)
        attention_mask = torch.tensor([sample["attention_mask"]], dtype=torch.bool)
        if input_ids.shape[1] != self.max_length or not attention_mask.all():
            raise ValueError("Every packed training sample must fill data.max_length without padding")
        return {"input_ids": input_ids.to(self.runtime.device),
                "attention_mask": attention_mask.to(self.runtime.device)}

    def _clip_gradients(self):
        if not self.args.max_grad_norm > 0:
            return None
        if self.is_fsdp:
            return self.model.clip_grad_norm_(self.args.max_grad_norm)
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

    def train(self):
        args, device = self.args, self.runtime.device
        groups = update_groups(len(self.dataset), args.seed, self.runtime.world_size, self.runtime.rank,
                               args.gradient_accumulation_steps)
        groups = itertools.islice(groups, self.state.global_step, args.max_steps)
        loss_since_log, steps_since_log, started = 0.0, 0, time.time()
        self.model.train()
        for epoch, group in groups:
            self.trainer_state.reset_grad_acc_step()
            group_loss = torch.zeros((), dtype=torch.float32, device=device)
            for position, index in enumerate(group):
                last = position == len(group) - 1
                batch = self._batch(index)
                sync = contextlib.nullcontext() if last or not hasattr(self.model, "no_sync") else self.model.no_sync()
                with sync:
                    with torch.autocast(device.type, dtype=torch.bfloat16):
                        loss = self.model(**batch)["loss"]
                    loss = loss / len(group)
                    loss.backward()
                group_loss += loss.detach().float()
                if not last:
                    self.trainer_state.increment_grad_acc_step()
            grad_norm = self._clip_gradients()
            self.optimizer.step()
            self.model.zero_grad(set_to_none=True)
            self.state.global_step += 1
            self.state.epoch = epoch
            loss_since_log += group_loss.item()
            steps_since_log += 1
            final = self.state.global_step == args.max_steps
            if self.state.global_step % args.logging_steps == 0 or final:
                self._log(loss_since_log / steps_since_log, grad_norm)
                loss_since_log, steps_since_log = 0.0, 0
            if self.state.global_step % args.save_steps == 0 or final:
                self.save_checkpoint()
        return time.time() - started

    def _log(self, loss: float, grad_norm):
        custom = dict(self.trainer_state.logs)
        values = self._all_mean([loss, *custom.values()])
        record = {"loss": values[0],
                  "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                  "learning_rate": self.args.learning_rate}
        record.update({f"gl_{key}": value for key, value in zip(custom, values[1:])})
        record.update(epoch=self.state.epoch, step=self.state.global_step)
        self.state.log_history.append(record)
        if self.runtime.is_main:
            logger.info(json.dumps(record))

    # ----- checkpoints --------------------------------------------------------------------------------

    def _run_identity(self) -> dict:
        base = getattr(self.combined.student_model.config, "semanticlogkv_base", None) or {}
        return {"world_size": self.runtime.world_size, "seed": self.args.seed,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps, "fsdp": self.args.fsdp,
                "num_samples": len(self.dataset), "max_length": self.max_length,
                "base_weight_sha256": base.get("source_weight_sha256")}

    def save_checkpoint(self) -> Path:
        adapters = self._adapter_state()
        if self.is_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with self._fsdp_full_state():
                optimizer = FSDP.optim_state_dict(self.model, self.optimizer)
        else:
            optimizer = self.optimizer.state_dict()
        target = self.output_dir / f"checkpoint-{self.state.global_step}"
        if self.runtime.is_main:
            staging = target.with_name(target.name + ".tmp")
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            torch.save(adapters, staging / "adapters.pt")
            torch.save(optimizer, staging / "optimizer.pt")
            record = {**asdict(self.state), "run": self._run_identity(), "settings": asdict(self.args)}
            (staging / "trainer_state.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            shutil.rmtree(target, ignore_errors=True)
            staging.rename(target)  # only complete checkpoints carry the checkpoint-N name
            if self.args.save_total_limit is not None:
                for old in checkpoints(self.output_dir)[:-self.args.save_total_limit]:
                    shutil.rmtree(old)
            logger.info(f"Saved {target}")
        self.runtime.barrier()
        return target

    def _load_checkpoint(self, path: Path) -> Path:
        """Restore the adapters (before wrapping) and the step; the optimizer follows once wrapped."""
        record = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        if record["run"] != self._run_identity():
            raise ValueError(f"Checkpoint {path} comes from a different run setup: {record['run']} "
                             f"vs {self._run_identity()}")
        adapters = torch.load(path / "adapters.pt", map_location="cpu", weights_only=True)
        if sorted(adapters) != sorted(self.trainable):
            raise ValueError(f"Checkpoint {path} holds other trainable weights than this model")
        result = self.combined.load_state_dict(adapters, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"Unexpected checkpoint weights: {result.unexpected_keys}")
        self.state = TrainState(record["global_step"], record["epoch"], record["log_history"])
        logger.info(f"Resuming from {path} at step {self.state.global_step}")
        return path

    def _load_optimizer_state(self, path: Path):
        state = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
        if self.is_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with self._fsdp_full_state(rank0_only=False):
                state = FSDP.optim_state_dict_to_load(self.model, self.optimizer, state)
        self.optimizer.load_state_dict(state)

    def training_record(self, runtime_seconds: float) -> dict:
        """What was trained, where, and at what cost; stored with the exported student."""
        device = self.runtime.device
        return {
            "global_step": self.state.global_step, "run": self._run_identity(), "settings": asdict(self.args),
            "trainable_parameters": self.trainable_numel,
            "tokens_per_step": self.args.gradient_accumulation_steps * self.runtime.world_size * self.max_length,
            "runtime_seconds_this_launch": runtime_seconds,
            "devices": [torch.cuda.get_device_name(device)] * self.runtime.world_size
            if device.type == "cuda" else [device.type] * self.runtime.world_size,
            "versions": {name: version(name) for name in ("torch", "transformers", "datasets", "tokenizers")},
            "final_log": self.state.log_history[-1] if self.state.log_history else None,
        }
