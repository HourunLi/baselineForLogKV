# Vendored for ChunkKV; local namespace changes are recorded in SOURCE.json.
"""
logKV adaptation CPT: continue pre-training under simulated compressed-KV
(log-structured) streaming attention, so the model learns to read merged slots.
Run on already-pretrained base weights. This script has no dense route.

Usage:
    # Single GPU debug
    python demo.py --config exp/qwen0.6b-4k/debug.yaml

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 demo.py --config exp/qwen0.6b-32k/cpt-base.yaml

Architecture:
    - Training: model(idx) routes through the logKV streaming simulation
      (enable_log_kv_training; chunked slot attention, 2:1 compaction)
    - Inference (eval): log-structured KV cache with O(recent + B*log N) memory
      via model.set_log_kv_cache() + model(idx, input_pos=input_pos)
"""

import os
import sys
import re
import inspect
import shutil
import glob
import json
import tempfile
import time
from pathlib import Path
import yaml
from dataclasses import asdict
import torch
import lightning as L
from datetime import datetime
from chunkkv._vendor.semanticlogkv.litgpt import Config
from chunkkv._vendor.semanticlogkv.litgpt.model import GPT
from chunkkv._vendor.semanticlogkv.litgpt.log_kv_checkpoint import enable_logkv_checkpoint_replay
from chunkkv._vendor.semanticlogkv.litgpt.utils import get_log_kv_second_order_scale, load_checkpoint
from chunkkv._vendor.semanticlogkv.litgpt.log_kv_timing import (
    CACHE_STAGES, STEP_STAGES, logkv_begin_step, logkv_take_host_stats, logkv_timed,
)
import random
import numpy as np
from lightning.fabric.loggers import TensorBoardLogger
from chunkkv._vendor.semanticlogkv.litgpt.model import Block
import math
from lightning.fabric.strategies import DDPStrategy, FSDPStrategy
from datetime import timedelta
from litdata.streaming import (
    CombinedStreamingDataset,
    StreamingDataLoader,
    StreamingDataset,
    TokensLoader,
)

from chunkkv._vendor.semanticlogkv.data import litdata_chunks_dir, tokenizer_cache_key
from chunkkv._vendor.semanticlogkv.utils import *

torch.set_float32_matmul_precision("high")
torch.set_default_dtype(torch.bfloat16)


def _unique_save_dir(save_path: str) -> str:
    p = Path(os.path.expandvars(os.path.expanduser(save_path)))
    try:
        p = p.resolve()
    except OSError:
        p = Path(os.path.abspath(p))
    parent = p.parent
    base = p.name
    cand = p
    n = 2
    # Only bump when the candidate already holds a checkpoint, so a pre-existing
    # but checkpoint-less directory (TensorBoard logs, a failed run, a manually
    # created dir) is reused. This keeps the saved path equal to the YAML
    # `save_path` that the pipeline (majob.sh) later evaluates, instead of
    # silently diverging to `_v2` and leaving the eval pointed at an empty dir.
    while (cand / "lit_model.pth").exists():
        cand = parent / f"{base}_v{n}"
        n += 1
    return str(cand)


_STEP_CHECKPOINT_RE = re.compile(r"^step[_-](\d+)(?:_v\d+)?$")
_CHECKPOINT_META_FILENAME = "checkpoint_meta.yaml"


def _normal_path(path: str | os.PathLike) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    try:
        return p.resolve()
    except OSError:
        return Path(os.path.abspath(p))


def _training_speed_metrics(fabric, *, elapsed_s: float, tokens: int, samples: int) -> dict:
    """Sum actual work, use the slowest rank's step time and largest device peak."""
    counts = torch.tensor([tokens, samples], dtype=torch.int64, device=fabric.device)
    is_cuda = fabric.device.type == "cuda"
    spans = torch.tensor([
        elapsed_s,
        torch.cuda.max_memory_allocated(fabric.device) if is_cuda else 0,
        torch.cuda.max_memory_reserved(fabric.device) if is_cuda else 0,
    ], dtype=torch.float64, device=fabric.device)
    if fabric.world_size > 1:
        counts = fabric.all_reduce(counts, reduce_op="sum")
        spans = fabric.all_reduce(spans, reduce_op="max")
    total_tokens, total_samples = counts.tolist()
    step_time, peak_allocated, peak_reserved = spans.tolist()
    metrics = {
        "train/step_time_s": step_time,
        "train/tokens": total_tokens,
        "train/samples": total_samples,
        "train/tokens_per_sec": total_tokens / step_time,
        "train/tokens_per_sec_per_device": total_tokens / step_time / fabric.world_size,
        "train/samples_per_sec": total_samples / step_time,
    }
    if is_cuda:
        metrics["train/peak_memory_allocated_bytes"] = int(peak_allocated)
        metrics["train/peak_memory_reserved_bytes"] = int(peak_reserved)
    return metrics


def _checkpoint_exists(path: Path) -> bool:
    # FSDP/DCP checkpoints may be directories named lit_model.pth; non-FSDP
    # checkpoints are regular files.
    try:
        if path.is_file():
            return path.stat().st_size > 0
        if path.is_dir():
            return any(path.iterdir())
    except OSError:
        return False
    return False


def _read_checkpoint_metadata(checkpoint_dir: Path) -> dict:
    metadata_path = checkpoint_dir / _CHECKPOINT_META_FILENAME
    if not metadata_path.is_file():
        return {}
    try:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _write_checkpoint_metadata(
    checkpoint_dir: Path | str,
    *,
    global_step: int,
    max_steps: int,
    data_epoch: int,
    num_epochs: int,
    finished: bool,
    kind: str,
    in_progress: bool = False,
) -> None:
    checkpoint_dir = _normal_path(checkpoint_dir)
    completed_epochs = max(0, min(int(num_epochs), int(data_epoch) - 1))
    metadata = {
        "global_step": int(global_step),
        "max_steps": int(max_steps),
        "data_epoch": int(data_epoch),
        "completed_epochs": completed_epochs,
        "num_epochs": int(num_epochs),
        "finished": bool(finished),
        "kind": kind,
        "in_progress": bool(in_progress),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(checkpoint_dir / _CHECKPOINT_META_FILENAME, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _replace_path(src: Path, dst: Path) -> None:
    if dst.is_dir():
        shutil.rmtree(dst)
    elif dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def _checkpoint_is_ready(checkpoint_path: Path) -> bool:
    if not _checkpoint_exists(checkpoint_path):
        return False
    metadata = _read_checkpoint_metadata(checkpoint_path.parent)
    return not bool(metadata.get("in_progress", False))


def _checkpoint_step(checkpoint_path: Path) -> int:
    metadata = _read_checkpoint_metadata(checkpoint_path.parent)
    step = metadata.get("global_step")
    if step is not None:
        try:
            return int(step)
        except (TypeError, ValueError):
            pass
    match = _STEP_CHECKPOINT_RE.fullmatch(checkpoint_path.parent.name)
    if match:
        return int(match.group(1))
    # Legacy direct checkpoints predate checkpoint_meta.yaml; treat them as
    # final so existing completed runs keep the old "do not retrain" behavior.
    return 10**18


def _checkpoint_sort_key(checkpoint_path: Path) -> tuple[int, float]:
    step = _checkpoint_step(checkpoint_path)
    try:
        mtime = checkpoint_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return step, mtime


def _latest_training_checkpoint(save_path: str) -> Path | None:
    root = _normal_path(save_path)
    candidates: list[Path] = []
    final_checkpoint = root / "lit_model.pth"
    if _checkpoint_is_ready(final_checkpoint):
        candidates.append(final_checkpoint)
    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir() or _STEP_CHECKPOINT_RE.fullmatch(child.name) is None:
                continue
            checkpoint = child / "lit_model.pth"
            if _checkpoint_is_ready(checkpoint):
                candidates.append(checkpoint)
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_sort_key)


def _copy_tokenizer_and_configs(src_dirs: list[str], dst: str) -> None:
    """Copy *.json / *.model (tokenizer + HF configs) into the save dir so it is
    self-contained for eval (eval.py builds its Tokenizer from the save dir).
    Later dirs take precedence on filename clashes, so pass the tokenizer_dir
    last — it may differ from the weights checkpoint_dir."""
    files: dict[str, str] = {}
    searched: list[str] = []
    for d in src_dirs:
        searched.append(str(d))
        for p in glob.glob(f"{d}/*.json") + glob.glob(f"{d}/*.model"):
            files[os.path.basename(p)] = p
    if "tokenizer.json" not in files and "tokenizer.model" not in files:
        raise FileNotFoundError(
            "No tokenizer.json / tokenizer.model found while saving checkpoint. "
            f"Searched: {searched}. Set tokenizer_dir in the YAML to the base model "
            "directory that contains the tokenizer files."
        )
    for p in files.values():
        shutil.copy(p, dst)


def _distributed_looks_multi_node() -> bool:
    try:
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        lws = int(os.environ.get("LOCAL_WORLD_SIZE", str(ws)))
    except ValueError:
        return False
    return ws > lws


def _nvidia_smi_field(query: str) -> str | None:
    """Return a single `nvidia-smi --query-gpu` field for GPU 0, or None if the
    tool is missing / errors. Used only for diagnostics, so failures are silent."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    first = out.stdout.strip().splitlines()
    return first[0].strip() if first else None


def cuda_preflight() -> None:
    """Fail fast with a readable message when the CUDA runtime cannot initialize.

    ``torch._C._cuda_init`` is lazy: the first GPU touch (here, ``fabric.launch()``)
    is where a driver/runtime mismatch surfaces, as an opaque traceback. This
    prints a torch/CUDA/driver fingerprint up front and, if the device is
    unreachable, raises a RuntimeError that names the likely cause (NVIDIA driver
    too old for the CUDA version baked into the installed torch wheel) instead of
    letting the raw ``_cuda_init`` error propagate.
    """
    driver = _nvidia_smi_field("driver_version")
    print("[cuda-preflight] "
          f"torch={torch.__version__} "
          f"torch.version.cuda={torch.version.cuda} "
          f"driver={driver or 'n/a'} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
          flush=True)

    # Try to actually reach the device. torch.cuda.is_available() swallows the
    # underlying error, so provoke it directly to capture the real message.
    err: BaseException | None = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() returned False")
        torch.zeros(1, device="cuda")  # forces torch._C._cuda_init
    except BaseException as e:  # noqa: BLE001 - want the raw reason for the report
        err = e

    if err is None:
        print(f"[cuda-preflight] OK: {torch.cuda.device_count()} device(s) visible, "
              f"compiled for CUDA {torch.version.cuda}", flush=True)
        return

    if torch.version.cuda is None:
        hint = ("This torch build is CPU-only (torch.version.cuda is None). Install a "
                "CUDA build of torch that matches the node's driver.")
    else:
        hint = (
            f"torch was built for CUDA {torch.version.cuda}; the NVIDIA driver "
            f"({driver or 'unknown'}) must support at least that CUDA version. "
            "Run `nvidia-smi` and compare its top-right 'CUDA Version' with "
            f"torch.version.cuda={torch.version.cuda}. If the driver's max is lower, "
            "either install a torch wheel built for an older CUDA (e.g. a +cu118 "
            "build) or upgrade the node's driver. Also verify CUDA_VISIBLE_DEVICES "
            "points at a real GPU and that this node actually has one."
        )
    raise RuntimeError(
        f"CUDA is not usable on this node: {type(err).__name__}: {err}\n"
        f"[cuda-preflight] {hint}"
    ) from err


def get_lr(current_step, total_steps, warmup_steps, max_lr, min_lr):
    """Cosine LR schedule with linear warmup."""
    if current_step < warmup_steps:
        return max_lr * (current_step + 1) / max(warmup_steps, 1)
    if current_step > total_steps:
        return min_lr
    decay_ratio = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def build_train_dataset(
    *,
    context_length: int,
    seed: int,
    tok_dir: str,
    data_dir: str,
    dataset_dir: str,
    data_mix_yaml: str | None,
) -> StreamingDataset | CombinedStreamingDataset:
    """Single-source or multi-source (YAML-weighted) streaming dataset."""
    block = context_length + 1

    def _stream(path: str) -> StreamingDataset:
        return StreamingDataset(
            input_dir=path,
            item_loader=TokensLoader(block_size=block),
            shuffle=True,
            seed=seed,
        )

    if not data_mix_yaml:
        chunks = litdata_chunks_dir(tok_dir, data_dir, dataset_dir, context_length)
        if not os.path.isdir(chunks) or not any(os.scandir(chunks)):
            raise FileNotFoundError(
                f"LitData cache not found: {chunks}\nRun: python data.py ... --context_length {context_length}"
            )
        return _stream(chunks)

    yml = os.path.normpath(os.path.expanduser(data_mix_yaml))
    with open(yml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Data mix YAML must be a dict: {yml}")
    try:
        raw_paths, raw_w = cfg["paths"], cfg["weights"]
    except KeyError as e:
        raise ValueError(f"Data mix YAML must contain paths and weights: {yml}") from e
    if len(raw_paths) != len(raw_w):
        raise ValueError("paths and weights must have same length")
    base = cfg.get("base_dir")
    base = os.path.abspath(os.path.expanduser(str(base))) if base else ""

    def _resolve_yaml_path(p: str) -> str:
        p_exp = os.path.expanduser(str(p).strip())
        full = os.path.join(base, p_exp) if base else p_exp
        full = os.path.abspath(os.path.normpath(full))
        if not os.path.isdir(full):
            raise FileNotFoundError(f"paths entry is not a directory: {full!r}")
        key = tokenizer_cache_key(tok_dir)
        exact = os.path.join(full, f"litdata_{key}_ctx{context_length}")
        if os.path.isdir(exact) and any(os.scandir(exact)):
            return os.path.abspath(exact)
        raise FileNotFoundError(
            f"Not found: {exact!r} (run data.py with matching tokenizer and context_length first)"
        )

    dirs: list[str] = []
    for p in raw_paths:
        full = _resolve_yaml_path(str(p))
        if not os.path.isdir(full) or not any(os.scandir(full)):
            raise FileNotFoundError(f"Invalid LitData dir: {full}")
        dirs.append(full)

    s = float(sum(raw_w))
    if s <= 0:
        raise ValueError("weights sum must be positive")
    weights = tuple(float(w) / s for w in raw_w)
    iterate = bool(cfg.get("iterate_over_all", False))

    if len(dirs) == 1:
        return _stream(dirs[0])
    return CombinedStreamingDataset(
        datasets=[_stream(d) for d in dirs],
        seed=seed,
        weights=weights,
        iterate_over_all=iterate,
    )


def _load_yaml_config(yaml_path: str, model_dir: str) -> dict:
    """Load a YAML config file that may refer to another YAML via 'config:' key.

    Supports nesting: if the YAML has a 'config:' key pointing to another YAML,
    the base config is loaded and overridden with the current YAML's values.
    The 'config:' path is relative to the current YAML file.
    """
    yaml_path = os.path.normpath(os.path.expanduser(yaml_path))
    if not os.path.isfile(yaml_path):
        # A silent {} here used to mean every setting fell back to its
        # hardcoded default (0.6B arch, relative checkpoints/ path, ...)
        # with no warning that the config never loaded at all.
        raise FileNotFoundError(
            f"config file not found: {yaml_path!r} (resolved from cwd {os.getcwd()!r}); "
            "check the --config path and the working directory it's run from."
        )

    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}

    if "config" in cfg:
        base_name = cfg.pop("config")
        # Resolve relative to the current YAML file's directory
        base_dir = os.path.dirname(yaml_path)
        base_path = os.path.join(base_dir, base_name)
        base_cfg = _load_yaml_config(base_path, model_dir)
        merged = {**base_cfg, **cfg}
        return merged

    return cfg


# PyYAML implements YAML 1.1: scientific notation WITHOUT a decimal point
# ("2e-5", "5e-6") does not match its float resolver and loads as *str*.
# Configs in exp/ use exactly that style, so numeric params (learning_rate,
# min_lr, ...) would arrive as strings and crash AdamW / the LR schedule.
_SCI_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d*)?[eE][-+]?\d+")


def _coerce_yaml_sci_floats(cfg: dict) -> dict:
    """Convert top-level str values that are scientific-notation numbers to float."""
    return {
        k: float(v) if isinstance(v, str) and _SCI_FLOAT_RE.fullmatch(v) else v
        for k, v in cfg.items()
    }


@auto_expand_env_vars
def main(
    # ── Model ──
    arch_name: str = "Qwen/Qwen3-0.6B-Base",
    context_length: int = 4096,
    ckpt_dir: str | None = None,
    resume_dir: str | None = None,
    auto_resume: bool = False,
    # ── Training ──
    global_batch_size: int = 32,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    max_steps: int = 10,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.1,
    num_devices: int = 1,
    warmup_steps: int = 5,
    min_lr: float = 2e-6,
    entropy_chunk_size: int = 0,
    # ── Data ──
    dataset_name: str = "debug",
    dataset_dir: str = "data",
    data_dir: str = "data",
    tokenizer_dir: str | None = None,
    data_shuffle_seed: int = 42,
    data_mix_yaml: str | None = None,
    num_workers: int = 16,
    # ── IO ──
    save_ckpt: bool = False,
    save_path: str = "./ckpt/cpt",
    save_interval: int = 500,
    merged_state_staging_dir: str | None = None,
    enable_tensorboard: bool = True,
    tensorboard_root: str = "./tb",
    # ── Experiment ──
    expid: str = "debug",
    # ── Log-structured KV cache (always on) ──
    # This script IS the logKV adaptation phase: training always simulates the
    # compressed-KV streaming attention so the model learns to read merged
    # slots. Run it after dense pretraining, on already-pretrained base weights.
    log_kv_B: int = 512,
    log_kv_recent_size: int = 1024,
    # Training replay block size. 2 = strict 2-token streaming semantics but
    # extremely slow at 32K; 64/128/256 keep memory bounded and greatly reduce
    # tiny matmul/autograd replay launches.
    log_kv_train_block: int = 128,
    # Eval-time prefill block size, forwarded to eval.py by run_eval (inference
    # only, does not affect training). 2 = strict 2-token streaming semantics;
    # larger = faster prefill with a bounded, block-size-limited deviation.
    log_kv_prefill_block: int = 256,
    # Coupled gate for score-side Sigma and value-side Gamma corrections. The
    # CPT path warms this from 0 to the target value to avoid an immediate
    # attention-distribution jump at step 0. Set
    # log_kv_second_order_warmup_steps=0 to start at the target value.
    log_kv_second_order_scale: float = 1.0,
    log_kv_second_order_warmup_steps: int = 10,
    # Importance-weighted level pooling (training AND eval; a deterministic
    # function of k, no new learnable params, so both paths stay consistent
    # automatically). Independent of the token-count weight driving the
    # log(w) mass bias. See LogStructuredKVCache.
    log_kv_importance_pooling: bool = False,
    # Only consulted when log_kv_importance_pooling=True. Blends the
    # importance share toward the uniform/count share (1.0 = pure
    # importance, 0.0 = uniform). See LogStructuredKVCache.
    log_kv_importance_pooling_lambda: float = 1.0,
    # Only consulted when log_kv_importance_pooling=True, orthogonal to
    # lambda: exponent reshaping the raw importance heuristic before
    # normalization (1.0 = unchanged, < 1.0 dampens outlier tokens).
    # See LogStructuredKVCache.
    log_kv_importance_pooling_temperature: float = 1.0,
    log_kv_semantic_clusters: bool = False,
    log_kv_cluster_k_max: int = 1,
    log_kv_cluster_lambda_rel: float = 1.0,
    log_kv_seg_eta: float = 1.0,
    log_kv_seg_g0: float = 2048.0,
    log_kv_seg_gap_max: float | None = None,
    log_kv_seg_block_level: int = 0,
    log_kv_seg_forget: float = 0.5,
    log_kv_semantic_s_h_path: str | None = None,
    log_kv_semantic_flush_granularity: int = 2,
    log_kv_semantic_cluster_chunk_size: int = 0,
    log_kv_semantic_capacity_beta: float = 0.0,
    log_kv_semantic_capacity_hard_cap_mult: float = 0.0,
    # Opt in to the pre-batching router: one Ward merge per orphan and a
    # per-cluster ladder walk. Kept for A/B against the fast path only.
    log_kv_semantic_legacy_route: bool = False,
    log_kv_semantic_unified_route: bool = False,
    log_kv_semantic_merge_passes: int = 1,
    log_kv_semantic_anchor_mode: str = "multi",
    log_kv_semantic_pack_backend: str = "auto",
    log_kv_semantic_centroid_backend: str = "sequential",
    log_kv_semantic_replay_updates: bool = False,
    # Alpha exact spans + Beta scoring/merging are on by default for semantic
    # clusters; set log_kv_alpha_exact_tokens: 0 (or the beta flags false) to disable.
    log_kv_alpha_exact_tokens: int = 256,
    log_kv_alpha_span_max_tokens: int = 64,
    log_kv_beta_novelty: bool = True,
    log_kv_beta_adaptive_merge: bool = True,
    log_kv_profile_steps: list[int] | None = None,
    activation_checkpointing: bool = True,
    # ── Eval ──
    run_eval: str = "",  # "before" | "after" | "both"
    eval_benchmark: str = "debug",
    # ── YAML config ──
    config: str | None = None,
):
    # ── Load YAML config if provided ──
    # NOTE: `locals()[k] = v` does NOT write back to a function's real locals in
    # CPython (locals() returns a snapshot), so YAML overrides must be applied by
    # explicit re-binding below. A YAML value overrides the CLI/default value
    # unless it is null (None), matching the previous `if v is not None` intent.
    _yaml: dict = {}
    if config is not None:
        # expand_env_vars：bash 风格 ${VAR} / ${VAR-default} 展开（含继承合并后的
        # 全部值）。必须与 majob.sh 的 bash 展开、eval.py 的装载一致，否则
        # save_path 会被写成字面 ${...} 目录、eval 却去展开后的路径找权重。
        _yaml = _coerce_yaml_sci_floats(
            expand_env_vars(_load_yaml_config(os.path.join(os.getcwd(), config), "") or {})
        )
        _banned = sorted(_k for _k in _yaml if _k.startswith("eval_log_kv_"))
        if _banned:
            raise ValueError(
                "eval_log_kv_* YAML keys are no longer supported; train and eval share the same LogKV config. "
                f"Remove: {', '.join(_banned)}"
            )
        _valid = set(inspect.signature(main).parameters)
        for _k in _yaml:
            if _k != "config" and _k not in _valid:
                print(f"[demo] WARNING: unknown YAML key ignored: {_k}")

    def _o(name, current):
        v = _yaml.get(name)
        return v if v is not None else current

    arch_name = _o("arch_name", arch_name)
    context_length = _o("context_length", context_length)
    ckpt_dir = _o("ckpt_dir", ckpt_dir)
    resume_dir = _o("resume_dir", resume_dir)
    auto_resume = _o("auto_resume", auto_resume)
    global_batch_size = _o("global_batch_size", global_batch_size)
    micro_batch_size = _o("micro_batch_size", micro_batch_size)
    num_epochs = _o("num_epochs", num_epochs)
    max_steps = _o("max_steps", max_steps)
    learning_rate = _o("learning_rate", learning_rate)
    weight_decay = _o("weight_decay", weight_decay)
    num_devices = _o("num_devices", num_devices)
    warmup_steps = _o("warmup_steps", warmup_steps)
    min_lr = _o("min_lr", min_lr)
    entropy_chunk_size = _o("entropy_chunk_size", entropy_chunk_size)
    dataset_name = _o("dataset_name", dataset_name)
    dataset_dir = _o("dataset_dir", dataset_dir)
    data_dir = _o("data_dir", data_dir)
    tokenizer_dir = _o("tokenizer_dir", tokenizer_dir)
    data_shuffle_seed = _o("data_shuffle_seed", data_shuffle_seed)
    data_mix_yaml = _o("data_mix_yaml", data_mix_yaml)
    num_workers = _o("num_workers", num_workers)
    save_ckpt = _o("save_ckpt", save_ckpt)
    save_path = _o("save_path", save_path)
    save_interval = _o("save_interval", save_interval)
    merged_state_staging_dir = _o("merged_state_staging_dir", merged_state_staging_dir)
    enable_tensorboard = _o("enable_tensorboard", enable_tensorboard)
    tensorboard_root = _o("tensorboard_root", tensorboard_root)
    expid = _o("expid", expid)
    log_kv_B = _o("log_kv_B", log_kv_B)
    log_kv_recent_size = _o("log_kv_recent_size", log_kv_recent_size)
    log_kv_train_block = _o("log_kv_train_block", log_kv_train_block)
    log_kv_prefill_block = _o("log_kv_prefill_block", log_kv_prefill_block)
    log_kv_second_order_scale = _o("log_kv_second_order_scale", log_kv_second_order_scale)
    log_kv_second_order_warmup_steps = _o(
        "log_kv_second_order_warmup_steps", log_kv_second_order_warmup_steps
    )
    log_kv_importance_pooling = bool(_o("log_kv_importance_pooling", log_kv_importance_pooling))
    log_kv_importance_pooling_lambda = float(
        _o("log_kv_importance_pooling_lambda", log_kv_importance_pooling_lambda)
    )
    log_kv_importance_pooling_temperature = float(
        _o("log_kv_importance_pooling_temperature", log_kv_importance_pooling_temperature)
    )
    log_kv_semantic_clusters = bool(_o("log_kv_semantic_clusters", log_kv_semantic_clusters))
    log_kv_semantic_anchor_mode = _o("log_kv_semantic_anchor_mode", log_kv_semantic_anchor_mode)
    log_kv_semantic_pack_backend = _o("log_kv_semantic_pack_backend", log_kv_semantic_pack_backend)
    log_kv_semantic_centroid_backend = _o("log_kv_semantic_centroid_backend", log_kv_semantic_centroid_backend)
    log_kv_semantic_replay_updates = bool(_o("log_kv_semantic_replay_updates", log_kv_semantic_replay_updates))
    log_kv_alpha_exact_tokens = int(_o("log_kv_alpha_exact_tokens", log_kv_alpha_exact_tokens))
    log_kv_alpha_span_max_tokens = int(_o("log_kv_alpha_span_max_tokens", log_kv_alpha_span_max_tokens))
    log_kv_beta_novelty = bool(_o("log_kv_beta_novelty", log_kv_beta_novelty))
    log_kv_beta_adaptive_merge = bool(_o("log_kv_beta_adaptive_merge", log_kv_beta_adaptive_merge))
    if not log_kv_semantic_clusters:
        log_kv_alpha_exact_tokens = 0  # exact spans live on semantic clusters only
    if not log_kv_alpha_exact_tokens:
        # Beta extends the Alpha exact pool; disabling Alpha disables both Beta changes.
        log_kv_beta_novelty = log_kv_beta_adaptive_merge = False
    log_kv_profile_steps = _o("log_kv_profile_steps", log_kv_profile_steps)
    if log_kv_profile_steps is not None and (
        not isinstance(log_kv_profile_steps, (list, tuple))
        or any(type(step) is not int or step < 1 for step in log_kv_profile_steps)
    ):
        raise ValueError("log_kv_profile_steps must contain positive, one-based optimizer step numbers")
    profile_steps = set(log_kv_profile_steps or [])
    log_kv_cluster_k_max = int(_o("log_kv_cluster_k_max", log_kv_cluster_k_max))
    log_kv_cluster_lambda_rel = float(_o("log_kv_cluster_lambda_rel", log_kv_cluster_lambda_rel))
    log_kv_seg_eta = float(_o("log_kv_seg_eta", log_kv_seg_eta))
    log_kv_seg_g0 = float(_o("log_kv_seg_g0", log_kv_seg_g0))
    log_kv_seg_gap_max = _o("log_kv_seg_gap_max", log_kv_seg_gap_max)
    log_kv_seg_gap_max = None if log_kv_seg_gap_max is None else float(log_kv_seg_gap_max)
    log_kv_seg_block_level = int(_o("log_kv_seg_block_level", log_kv_seg_block_level))
    log_kv_seg_forget = float(_o("log_kv_seg_forget", log_kv_seg_forget))
    log_kv_semantic_s_h_path = _o("log_kv_semantic_s_h_path", log_kv_semantic_s_h_path)
    log_kv_semantic_flush_granularity = int(
        _o("log_kv_semantic_flush_granularity", log_kv_semantic_flush_granularity)
    )
    log_kv_semantic_cluster_chunk_size = int(
        _o("log_kv_semantic_cluster_chunk_size", log_kv_semantic_cluster_chunk_size)
    )
    log_kv_semantic_capacity_beta = float(
        _o("log_kv_semantic_capacity_beta", log_kv_semantic_capacity_beta)
    )
    log_kv_semantic_capacity_hard_cap_mult = float(
        _o("log_kv_semantic_capacity_hard_cap_mult", log_kv_semantic_capacity_hard_cap_mult)
    )
    log_kv_semantic_legacy_route = bool(
        _o("log_kv_semantic_legacy_route", log_kv_semantic_legacy_route)
    )
    log_kv_semantic_unified_route = bool(
        _o("log_kv_semantic_unified_route", log_kv_semantic_unified_route)
    )
    log_kv_semantic_merge_passes = int(_o("log_kv_semantic_merge_passes", log_kv_semantic_merge_passes))
    activation_checkpointing = bool(_o("activation_checkpointing", activation_checkpointing))
    run_eval = _o("run_eval", run_eval)
    eval_benchmark = _o("eval_benchmark", eval_benchmark)

    log_kv_second_order_scale = float(log_kv_second_order_scale)
    log_kv_second_order_warmup_steps = int(log_kv_second_order_warmup_steps)
    # Validate early, before launching a long distributed job.
    get_log_kv_second_order_scale(
        0, log_kv_second_order_warmup_steps, log_kv_second_order_scale
    )

    # Fail fast: run_eval="after"/"both" evaluates save_path, which is only
    # written when save_ckpt is true. Catch the contradiction here instead of
    # training for hours and then dying in eval on a missing tokenizer/weights.
    if run_eval in ("after", "both") and not save_ckpt:
        raise ValueError(
            f"run_eval={run_eval!r} evaluates the saved checkpoint, but save_ckpt is false — "
            "nothing would be written to save_path. Set save_ckpt: true in the YAML "
            "(or set run_eval: '' to skip the post-training eval)."
        )

    # 1. Set seeds
    set_random_seeds(42)
    print(f"Training Python: {sys.executable}")
    loggers = []
    if enable_tensorboard:
        try:
            loggers.append(TensorBoardLogger(root_dir=tensorboard_root, name=f"{expid}_{arch_name.replace('/', '-')}"))
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"TensorBoard logger initialization failed in {sys.executable}. "
                f"Install the backend with: {sys.executable} -m pip install 'tensorboard>=2.14', "
                "or explicitly set enable_tensorboard: false."
            ) from exc

    # 2. Fabric setup
    config_obj = Config.from_name(arch_name)
    config_obj.block_size = context_length
    use_fsdp = context_length > 4096
    print("Use FSDP:", use_fsdp)
    if not use_fsdp:
        strategy = DDPStrategy(timeout=timedelta(days=3650))
    else:
        strategy = FSDPStrategy(
            sharding_strategy="SHARD_GRAD_OP",
            state_dict_type="full",
            auto_wrap_policy={Block},
            # Streaming replay bounds attention's inner graph, but Q/K/V,
            # RMSNorm and projection activations still accumulate across layers.
            activation_checkpointing_policy={Block} if activation_checkpointing else None,
            timeout=timedelta(days=3650),
        )
    # Surface a driver/runtime mismatch here, as a readable error, before the
    # opaque torch._C._cuda_init traceback inside fabric.launch().
    cuda_preflight()
    fabric = L.Fabric(
        accelerator="cuda",
        # torchrun has already chosen the per-node process count; YAML cannot resize it.
        devices=int(os.environ.get("LOCAL_WORLD_SIZE", num_devices)),
        num_nodes=int(os.environ.get("GROUP_WORLD_SIZE", 1)),
        strategy=strategy,
        precision="bf16-true",
        loggers=loggers,
    )
    fabric.launch()
    fabric.print("Tensorboard root:", tensorboard_root)
    # Checkpoint 去向尽早入日志：save_path 为相对路径时跟随作业的 cwd，
    # 训练完“找不到 checkpoint”十有八九是在另一个目录里找。save_ckpt=False
    # 时这里就是唯一会明说“不会保存”的地方。
    fabric.print(
        f"save_ckpt={save_ckpt} | auto_resume={auto_resume} | save_path="
        f"{Path(os.path.expandvars(os.path.expanduser(save_path))).absolute()} "
        f"(cwd={os.getcwd()})"
        + ("" if save_ckpt else " | ⚠️ save_ckpt=False：本次训练不会写任何 checkpoint")
    )

    fabric.print(f"Model config initialized: {config_obj.name}")

    checkpoint_dir = f"checkpoints/{arch_name}"
    if ckpt_dir is not None:
        checkpoint_dir = ckpt_dir
    initial_load_dir = checkpoint_dir if resume_dir is None else resume_dir
    initial_ckpt_path = Path(initial_load_dir) / "lit_model.pth"
    resume_ckpt_path = None
    if resume_dir is None and auto_resume:
        resume_ckpt_path = _latest_training_checkpoint(save_path)
        if resume_ckpt_path is not None:
            fabric.print(f"Auto-resume checkpoint found: {resume_ckpt_path}")
        else:
            fabric.print("auto_resume=True, but no checkpoint was found under save_path; loading initial weights.")
    elif resume_dir is not None and auto_resume:
        fabric.print("auto_resume=True is ignored because resume_dir is set; loading resume_dir as the weight source.")

    selected_ckpt_path = resume_ckpt_path if resume_ckpt_path is not None else initial_ckpt_path
    source_kind = "auto_resume" if resume_ckpt_path is not None else ("resume_dir" if resume_dir is not None else "ckpt_dir/base")
    fabric.print(f"Training checkpoint source ({source_kind}): {selected_ckpt_path}")
    fabric.print(f"Training checkpoint output: {_normal_path(save_path)}")
    if not _checkpoint_exists(selected_ckpt_path):
        raise FileNotFoundError(
            f"Training input checkpoint ({source_kind}) does not exist or is empty: {selected_ckpt_path}. "
            "Check ckpt_dir/resume_dir. save_path is the output directory; it is only selected "
            "as an input when explicitly requested by resume_dir or when auto_resume finds a checkpoint."
        )

    with fabric.init_module(empty_init=True):
        model = GPT(config_obj)
    model = fabric.setup_module(model)
    if use_fsdp and activation_checkpointing and log_kv_semantic_clusters and log_kv_cluster_k_max > 1:
        checkpoint_blocks = enable_logkv_checkpoint_replay(model, Block)
        fabric.print(f"LogKV checkpoint routing replay: enabled for {checkpoint_blocks} Blocks (strict, no rerouting fallback)")

    # ── Optimizer with weight decay groups ──
    decay_params = []
    no_decay_params = []
    no_decay_keywords = ["bias", "norm", "ln_"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    fabric.print(f"Weight decay params: {len(decay_params)} tensors")
    fabric.print(f"No weight decay params: {len(no_decay_params)} tensors")

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8
    )
    optimizer = fabric.setup_optimizers(optimizer)
    state = {"model": model, "optimizer": optimizer, "global_step": 0, "data_epoch": 1}

    load_dir = initial_load_dir
    if resume_ckpt_path is not None:
        fabric.print(f"Resuming full training state from {resume_ckpt_path}...")
        fabric.load(resume_ckpt_path, state)
        load_dir = str(resume_ckpt_path.parent)
        loaded_global_step = state.get("global_step", 0) or 0
        if isinstance(loaded_global_step, torch.Tensor):
            loaded_global_step = loaded_global_step.item()
        fabric.print(f"Training state restored at global_step={int(loaded_global_step)}.")
    else:
        fabric.print(f"Loading checkpoint from {initial_ckpt_path}...")
        # Single load: load_checkpoint streams weights into the (FSDP-)wrapped model.
        # A prior torch.load() here only to peek at the state dict doubled peak CPU
        # memory for large checkpoints without being used.
        load_checkpoint(fabric, model, initial_ckpt_path, strict=True)
        fabric.print("Weights loaded successfully.")

    # ── Evaluation helper ──
    # Same code path AND same semantics as a standalone `eval.py --config
    # exp/.../eval.yaml` run and as majob.sh: the checkpoint is evaluated under
    # the same compressed-KV (logKV) attention it was trained with, with the
    # training-time B/recent_size and the same prefill block. Results land in
    # {save_path}/evaluate (majob.sh's EVAL_OUTPUT_DIR) — also for the "before"
    # eval, whose base checkpoint dir may be read-only; each timestamped JSON
    # records its own checkpoint_dir. tokenizer_dir is forwarded for checkpoints
    # whose save dir lacks tokenizer files.
    def _run_eval(ckpt, benchmark):
        from chunkkv._vendor.semanticlogkv.eval import main as eval_main
        eval_main(
            checkpoint_dir=ckpt,
            benchmark=benchmark,
            output_path=f"{save_path}/evaluate",
            log_kv_B=log_kv_B,
            log_kv_recent_size=log_kv_recent_size,
            log_kv_prefill_block=log_kv_prefill_block,
            log_kv_second_order_scale=log_kv_second_order_scale,
            log_kv_importance_pooling=log_kv_importance_pooling,
            log_kv_importance_pooling_lambda=log_kv_importance_pooling_lambda,
            log_kv_importance_pooling_temperature=log_kv_importance_pooling_temperature,
            log_kv_semantic_clusters=log_kv_semantic_clusters,
            log_kv_cluster_k_max=log_kv_cluster_k_max,
            log_kv_cluster_lambda_rel=log_kv_cluster_lambda_rel,
            log_kv_seg_eta=log_kv_seg_eta,
            log_kv_seg_g0=log_kv_seg_g0,
            log_kv_seg_gap_max=log_kv_seg_gap_max,
            log_kv_seg_block_level=log_kv_seg_block_level,
            log_kv_seg_forget=log_kv_seg_forget,
            log_kv_semantic_s_h_path=log_kv_semantic_s_h_path,
            log_kv_semantic_flush_granularity=log_kv_semantic_flush_granularity,
            log_kv_semantic_cluster_chunk_size=log_kv_semantic_cluster_chunk_size,
            log_kv_semantic_capacity_beta=log_kv_semantic_capacity_beta,
            log_kv_semantic_capacity_hard_cap_mult=log_kv_semantic_capacity_hard_cap_mult,
            log_kv_semantic_legacy_route=log_kv_semantic_legacy_route,
            log_kv_semantic_unified_route=log_kv_semantic_unified_route,
            log_kv_semantic_merge_passes=log_kv_semantic_merge_passes,
            log_kv_semantic_anchor_mode=log_kv_semantic_anchor_mode,
            log_kv_semantic_pack_backend=log_kv_semantic_pack_backend,
            log_kv_semantic_centroid_backend=log_kv_semantic_centroid_backend,
            log_kv_semantic_replay_updates=log_kv_semantic_replay_updates,
            log_kv_alpha_exact_tokens=log_kv_alpha_exact_tokens,
            log_kv_alpha_span_max_tokens=log_kv_alpha_span_max_tokens,
            log_kv_beta_novelty=log_kv_beta_novelty,
            log_kv_beta_adaptive_merge=log_kv_beta_adaptive_merge,
            tokenizer_dir=tokenizer_dir,
        )

    if run_eval in ("before", "both"):
        _run_eval(load_dir, eval_benchmark)

    # ── Data loading ──
    tok_dir = tokenizer_dir if tokenizer_dir is not None else load_dir
    train_dataset = build_train_dataset(
        context_length=context_length,
        seed=data_shuffle_seed,
        tok_dir=tok_dir,
        data_dir=data_dir,
        dataset_dir=dataset_dir,
        data_mix_yaml=data_mix_yaml,
    )
    if isinstance(train_dataset, CombinedStreamingDataset):
        fabric.print(f"[Rank {fabric.global_rank}] Training data: CombinedStreamingDataset (multi-source)")
    else:
        fabric.print(f"[Rank {fabric.global_rank}] Training data: StreamingDataset (single source)")
    fabric.barrier()
    dataloader = StreamingDataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )
    dataloader = fabric.setup_dataloaders(dataloader)

    def _save_training_checkpoint(
        checkpoint_dir_path: str | os.PathLike,
        *,
        global_step: int,
        data_epoch: int,
        finished: bool,
        kind: str,
        atomic_model: bool = False,
    ) -> str:
        checkpoint_dir_path = _normal_path(checkpoint_dir_path)
        checkpoint_file = checkpoint_dir_path / "lit_model.pth"
        save_file = checkpoint_dir_path / "lit_model.pth.tmp" if atomic_model else checkpoint_file
        if fabric.global_rank == 0:
            os.makedirs(checkpoint_dir_path, exist_ok=True)
            if atomic_model:
                _remove_path(save_file)
            else:
                _write_checkpoint_metadata(
                    checkpoint_dir_path,
                    global_step=global_step,
                    max_steps=max_steps,
                    data_epoch=data_epoch,
                    num_epochs=num_epochs,
                    finished=finished,
                    kind=kind,
                    in_progress=True,
                )
        fabric.barrier()

        state["global_step"] = global_step
        state["data_epoch"] = data_epoch
        fabric.print(f"Saving {kind} checkpoint to {checkpoint_file} ...")
        fabric.save(str(save_file), state)
        fabric.barrier()

        if fabric.global_rank == 0:
            if not _checkpoint_exists(save_file):
                raise RuntimeError(
                    f"fabric.save 已返回，但 {save_file} 不存在或为空——"
                    "检查磁盘配额 / 共享文件系统同步状态。"
                )
            if atomic_model:
                _replace_path(save_file, checkpoint_file)
            if not _checkpoint_exists(checkpoint_file):
                raise RuntimeError(
                    f"fabric.save 已返回，但 {checkpoint_file} 不存在或为空——"
                    "检查磁盘配额 / 共享文件系统同步状态。"
                )
            _copy_tokenizer_and_configs([checkpoint_dir, tok_dir], str(checkpoint_dir_path))
            with open(checkpoint_dir_path / "model_config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(asdict(config_obj), f)
            _write_checkpoint_metadata(
                checkpoint_dir_path,
                global_step=global_step,
                max_steps=max_steps,
                data_epoch=data_epoch,
                num_epochs=num_epochs,
                finished=finished,
                kind=kind,
            )
        fabric.barrier()
        return str(checkpoint_dir_path)

    fabric.print("Starting Continue Pretraining...")
    model.train()

    loaded_global_step = state.get("global_step", 0) or 0
    if isinstance(loaded_global_step, torch.Tensor):
        loaded_global_step = loaded_global_step.item()
    global_step = int(loaded_global_step)
    initial_second_order_scale = get_log_kv_second_order_scale(
        global_step, log_kv_second_order_warmup_steps, log_kv_second_order_scale
    )
    semantic_s_h = (
        load_log_kv_semantic_s_h(
            log_kv_semantic_s_h_path,
            n_layer=config_obj.n_layer,
            n_groups=config_obj.n_query_groups,
        )
        if log_kv_semantic_clusters
        else None
    )

    # Always simulate the logKV compressed-KV streaming attention during
    # training — this script only supports the logKV adaptation route.
    if log_kv_alpha_exact_tokens:
        from chunkkv._vendor.semanticlogkv.litgpt.tokenizer import Tokenizer
        model.set_alpha_span_boundary_ids(Tokenizer(tok_dir).alpha_span_boundary_ids)
    model.enable_log_kv_training(
        batch_size=micro_batch_size,
        max_seq_length=context_length,
        device=fabric.device,
        # Allocate cache buffers in the activation dtype instead of relying on
        # the process default; keeps them aligned with the bf16-true params.
        dtype=next(model.parameters()).dtype,
        B=log_kv_B,
        recent_size=log_kv_recent_size,
        train_block=log_kv_train_block,
        second_order_scale=initial_second_order_scale,
        allocate_second_order=log_kv_second_order_scale != 0.0,
        importance_pooling=log_kv_importance_pooling,
        importance_pooling_lambda=log_kv_importance_pooling_lambda,
        importance_pooling_temperature=log_kv_importance_pooling_temperature,
        semantic_clusters=log_kv_semantic_clusters,
        cluster_k_max=log_kv_cluster_k_max,
        cluster_lambda_rel=log_kv_cluster_lambda_rel,
        seg_eta=log_kv_seg_eta,
        seg_g0=log_kv_seg_g0,
        seg_gap_max=log_kv_seg_gap_max,
        seg_block_level=log_kv_seg_block_level,
        seg_forget=log_kv_seg_forget,
        semantic_s_h=semantic_s_h,
        semantic_flush_granularity=log_kv_semantic_flush_granularity,
        semantic_cluster_chunk_size=log_kv_semantic_cluster_chunk_size,
        semantic_capacity_beta=log_kv_semantic_capacity_beta,
        semantic_capacity_hard_cap_mult=log_kv_semantic_capacity_hard_cap_mult,
        semantic_legacy_route=log_kv_semantic_legacy_route,
        semantic_unified_route=log_kv_semantic_unified_route,
        semantic_merge_passes=log_kv_semantic_merge_passes,
        semantic_anchor_mode=log_kv_semantic_anchor_mode,
        semantic_pack_backend=log_kv_semantic_pack_backend,
        semantic_centroid_backend=log_kv_semantic_centroid_backend,
        semantic_replay_updates=log_kv_semantic_replay_updates,
        alpha_exact_tokens=log_kv_alpha_exact_tokens,
        alpha_span_max_tokens=log_kv_alpha_span_max_tokens,
        beta_novelty=log_kv_beta_novelty,
        beta_adaptive_merge=log_kv_beta_adaptive_merge,
    )
    effective_log_kv_train_block = max(2, min(int(log_kv_train_block), int(log_kv_recent_size)))
    fabric.print(
        f"logKV training ENABLED: B={log_kv_B}, "
        f"recent_size={log_kv_recent_size}, "
        f"train_block={log_kv_train_block} "
        f"(effective={effective_log_kv_train_block}), "
        f"second_order_scale={initial_second_order_scale:.4f} "
        f"(target={log_kv_second_order_scale:.4f}, "
        f"warmup_steps={log_kv_second_order_warmup_steps}), "
        f"blocks/seq={math.ceil(context_length / effective_log_kv_train_block)}, "
        f"importance_pooling={log_kv_importance_pooling} "
        f"(lambda={log_kv_importance_pooling_lambda}, temperature={log_kv_importance_pooling_temperature}), "
        f"semantic={log_kv_semantic_clusters} "
        f"(K={log_kv_cluster_k_max}, lambda_rel={log_kv_cluster_lambda_rel}, "
        f"g_max={log_kv_seg_gap_max}, l_block={log_kv_seg_block_level}, "
        f"flush={log_kv_semantic_flush_granularity}, tree_chunk={log_kv_semantic_cluster_chunk_size}, "
        f"s_h={log_kv_semantic_s_h_path}, "
        f"capacity_beta={log_kv_semantic_capacity_beta}, "
        f"hard_cap_mult={log_kv_semantic_capacity_hard_cap_mult}, "
        f"legacy_route={log_kv_semantic_legacy_route}, "
        f"unified_route={log_kv_semantic_unified_route}, "
        f"merge_passes={log_kv_semantic_merge_passes}, "
        f"anchors={log_kv_semantic_anchor_mode}, pack={log_kv_semantic_pack_backend}, "
        f"centroid={log_kv_semantic_centroid_backend}, "
        f"replay_updates={log_kv_semantic_replay_updates}, "
        f"alpha_exact={log_kv_alpha_exact_tokens}, alpha_span_max={log_kv_alpha_span_max_tokens}, "
        f"beta_novelty={log_kv_beta_novelty}, beta_adaptive_merge={log_kv_beta_adaptive_merge}, "
        f"effective_B={model.transformer.h[0].attn.kv_cache.B})"
    )

    gradient_accumulation_steps = max(1, global_batch_size // (micro_batch_size * fabric.world_size))
    optimizer.zero_grad(set_to_none=True)
    step_active = False
    fabric.print(
        f"LogKV timing: host counters every optimizer step; CUDA profile steps={sorted(profile_steps)}. "
        "Stage breakdowns are rank 0 only; throughput uses all ranks and synchronized step boundaries. "
        "CUDA spans include stream waits; forward/backward totals contain the LogKV breakdown."
    )
    step_stats = MicroStepMeanStats()
    if global_step > 0:
        fabric.print(f"Continuing from global_step={global_step}; target max_steps={max_steps}.")
    total_steps = max_steps
    training_finished = False
    micro_batch_idx = 0
    loaded_data_epoch = state.get("data_epoch", 1) or 1
    if isinstance(loaded_data_epoch, torch.Tensor):
        loaded_data_epoch = loaded_data_epoch.item()
    data_epoch = int(loaded_data_epoch)
    loader_iter = iter(dataloader)
    run_start_step = global_step
    run_time_s = 0.0
    run_tokens = 0
    speed_log_path = _normal_path(save_path) / "training_metrics.jsonl"
    if fabric.global_rank == 0:
        speed_log_path.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Training speed metrics: {speed_log_path} (appended; cumulative values cover this run only)")

    while global_step < max_steps and not training_finished:
        if not step_active:
            if fabric.device.type == "cuda":
                torch.cuda.synchronize(fabric.device)
                torch.cuda.reset_peak_memory_stats(fabric.device)
            if fabric.world_size > 1:
                fabric.barrier()  # Keep rank-0 logging and checkpoint I/O outside the next measured step.
            step_tokens = step_samples = 0
            step_start_time = time.perf_counter()
            logkv_begin_step(profile_cuda=global_step + 1 in profile_steps, device=fabric.device)
            step_active = True
        try:
            with logkv_timed("data", cuda=False):
                train_data = next(loader_iter)
                inputs = train_data[:, 0:context_length].contiguous().long()
                targets = train_data[:, 1:context_length + 1].contiguous().long()
        except StopIteration:
            data_epoch += 1
            if data_epoch > num_epochs:
                fabric.print(f"Data exhausted after {num_epochs} epochs, step={global_step}/{max_steps}. Stopping.")
                break
            loader_iter = iter(dataloader)
            continue

        is_accumulating = (micro_batch_idx + 1) % gradient_accumulation_steps != 0
        micro_batch_idx += 1
        step_tokens += inputs.numel()
        step_samples += inputs.size(0)
        current_second_order_scale = get_log_kv_second_order_scale(
            global_step, log_kv_second_order_warmup_steps, log_kv_second_order_scale
        )
        model.set_log_kv_second_order_scale(current_second_order_scale)

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            # Routes through _log_kv_train_lowmem_forward (training_log_kv is on
            # and input_pos is None): chunked slot attention over the simulated
            # compressed-KV stream, not a standard dense causal forward. The
            # low-memory Function streams the forward without a graph and
            # replays block-by-block in backward, so per-layer activation
            # memory is O(T + train_block*S) instead of the naive O(T/2*S).
            with logkv_timed("forward"):
                loss = model(inputs, targets=targets, loss_chunk_size=entropy_chunk_size)
                step_stats.accumulate(loss=loss.detach().item())

            loss = loss / gradient_accumulation_steps
            with logkv_timed("backward"):
                fabric.backward(loss)

        if not is_accumulating:
            current_lr = get_lr(global_step, total_steps, warmup_steps, learning_rate, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            with logkv_timed("optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            kv = logkv_take_host_stats()
            if fabric.device.type == "cuda" and not kv["cuda_profiled"]:
                torch.cuda.synchronize(fabric.device)
            step_time = time.perf_counter() - step_start_time
            speed_metrics = _training_speed_metrics(
                fabric, elapsed_s=step_time, tokens=step_tokens, samples=step_samples,
            )
            run_time_s += speed_metrics["train/step_time_s"]
            run_tokens += speed_metrics["train/tokens"]
            speed_metrics.update({
                "train/run_time_s": run_time_s,
                "train/run_tokens": run_tokens,
                "train/run_tokens_per_sec": run_tokens / run_time_s,
            })
            now = datetime.now()
            avgs = step_stats.averages()
            metrics_txt = " | ".join(f"{k}: {v:.4f}" for k, v in sorted(avgs.items()))
            if metrics_txt:
                metrics_txt = metrics_txt + " | "
            kv_txt = "logKV_host: " + " + ".join(
                f"{stage} {kv[stage + '_s']:.3f}s/{kv[stage + '_n']}" for stage in CACHE_STAGES
            ) + " | "
            fabric.print(
                f"[{now.strftime('%H:%M:%S')}] "
                f"Epoch {data_epoch} | Step {global_step + 1} | "
                f"{metrics_txt}"
                f"2nd_scale: {current_second_order_scale:.4f} | "
                f"{kv_txt}"
                f"Time (slowest rank): {speed_metrics['train/step_time_s']:.2f}s | "
                f"tokens/s: {speed_metrics['train/tokens_per_sec']:.1f} global, "
                f"{speed_metrics['train/tokens_per_sec_per_device']:.1f}/device | "
                f"run: {run_time_s:.1f}s, {speed_metrics['train/run_tokens_per_sec']:.1f} tokens/s"
            )
            if "train/peak_memory_allocated_bytes" in speed_metrics:
                fabric.print(
                    f"Peak CUDA memory (largest rank): "
                    f"{speed_metrics['train/peak_memory_allocated_bytes'] / 2**30:.2f} GiB allocated, "
                    f"{speed_metrics['train/peak_memory_reserved_bytes'] / 2**30:.2f} GiB reserved"
                )
            fabric.print("step_host (inclusive): " + " | ".join(
                f"{stage} {kv[stage + '_s']:.3f}s" for stage in STEP_STAGES
            ))
            if kv["cuda_profiled"]:
                fabric.print("logKV_cuda: " + " | ".join(
                    f"{stage} {kv[stage + '_cuda_s']:.3f}s" for stage in CACHE_STAGES
                ))
                fabric.print("step_cuda (inclusive): " + " | ".join(
                    f"{stage} {kv[stage + '_cuda_s']:.3f}s" for stage in STEP_STAGES if stage != "data"
                ))
            timing_metrics = {"train/step_host_s": step_time, "train/logkv_cuda_profiled": int(kv["cuda_profiled"])}
            for stage in CACHE_STAGES + STEP_STAGES:
                prefix = "logkv" if stage in CACHE_STAGES else "step"
                timing_metrics[f"train/{prefix}_host_{stage}_s"] = kv[f"{stage}_s"]
                timing_metrics[f"train/{prefix}_{stage}_calls"] = kv[f"{stage}_n"]
                if kv["cuda_profiled"] and stage != "data":
                    timing_metrics[f"train/{prefix}_cuda_{stage}_s"] = kv[f"{stage}_cuda_s"]
            timing_metrics["train/logkv_host_attn_state_s"] = kv["attn_s"]
            timing_metrics.update(speed_metrics)
            if fabric.global_rank == 0:
                fabric.log_dict(timing_metrics, step=global_step + 1)
                with speed_log_path.open("a", encoding="utf-8") as speed_log:
                    speed_log.write(json.dumps({
                        "step": global_step + 1, "run_start_step": run_start_step,
                        "timestamp": now.isoformat(), "world_size": fabric.world_size,
                        **timing_metrics,
                    }) + "\n")
            for name, val in avgs.items():
                fabric.log(f"train/{name}", val, step=global_step + 1)
            fabric.log("train/learning_rate", current_lr, step=global_step + 1)
            fabric.log("train/log_kv_second_order_scale", current_second_order_scale, step=global_step + 1)

            step_stats.reset()
            global_step += 1
            step_active = False

            if global_step >= max_steps:
                fabric.print(f"Reached max_steps={max_steps}. Training complete.")
                training_finished = True

            if save_ckpt and save_interval > 0 and global_step % save_interval == 0 and not training_finished:
                step_save_path = f"{save_path}/step_{global_step}"
                step_save_path = _unique_save_dir(step_save_path)
                step_save_path = _save_training_checkpoint(
                    step_save_path,
                    global_step=global_step,
                    data_epoch=data_epoch,
                    finished=False,
                    kind="interval",
                )
                fabric.print(f"Checkpoint saved to {step_save_path}")
                latest_save_path = _save_training_checkpoint(
                    save_path,
                    global_step=global_step,
                    data_epoch=data_epoch,
                    finished=False,
                    kind="latest",
                    atomic_model=True,
                )
                fabric.print(f"Latest checkpoint updated at {latest_save_path}")

    # Discard an incomplete accumulation window before any final save/evaluation.
    if step_active:
        logkv_take_host_stats()

    # ── Final save ──
    if save_ckpt:
        save_path = _save_training_checkpoint(
            save_path,
            global_step=global_step,
            data_epoch=data_epoch,
            finished=True,
            kind="final",
            atomic_model=True,
        )
        if fabric.global_rank == 0:
            fabric.print(f"Tokenizer and model_config written to {save_path}")
            fabric.print("Done.")

    if run_eval in ("after", "both"):
        _run_eval(save_path, eval_benchmark)

    fabric.print("Training finished!")
    try:
        fabric.barrier()
    except Exception as e:  # noqa: BLE001
        fabric.print(f"Final training barrier failed before teardown; continuing shutdown: {e}")
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_before_destroy = torch.distributed.get_rank()
            torch.distributed.destroy_process_group()
            if rank_before_destroy == 0:
                print("Distributed process group destroyed after training.", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"Distributed process group teardown after training failed; ignoring during shutdown: {e}", flush=True)


if __name__ == "__main__":
    run_cli(main)
