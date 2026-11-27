"""
Evaluation pipeline for logKV models.

Usage:
    # Single GPU
    python eval.py --checkpoint_dir ./ckpt/qwen0.6b-32k-cpt-base --benchmark piqa

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 eval.py --checkpoint_dir ./ckpt/... --benchmark "boolq,piqa,..."

    # With YAML config
    python eval.py --config exp/qwen1.7b-32k/eval.yaml

除 logKV 相关内容（LogKVLM、YAML --config 机制、tokenizer 回退）外，
本文件与 kv 分支的 eval.py 逐段对齐（环境变量、输出、指标收集等）。
"""

from __future__ import annotations

import os
import re
import csv
import glob
import json
import builtins
import sys
import warnings
import logging
import time
import inspect
import contextlib
from datetime import datetime, timedelta
from pathlib import Path

import yaml
import torch
import torch.distributed as dist
import numpy as np
from typing import Any

import tqdm

from snapkv._vendor.niahlogkv.utils import *


# ==========================================
# 🧩 分布式小工具
# ==========================================
# 约定（与 demo.py 的 checkpoint 保存一致）：**任何"看文件系统"的判定只由
# rank 0 做，再 broadcast 给其它 rank**。共享盘（OBS/NFS）的元数据缓存在各节点
# 上的可见时刻不一致，若每个 rank 自己 stat/glob，就可能出现分支不一致 ——
# 一部分 rank 进了集合通信（barrier / all_gather）而另一部分跳过或直接抛异常
# 退出，剩下的 rank 就会等待通信超时。评测只交换 CPU 对象，使用 Gloo；
# 保留长等待窗口以容纳不同 rank 的长上下文计算时差。

def _before_stop(text, until):
    """Return the generated text preceding the earliest nonempty lm-eval stop string."""
    if isinstance(until, str):
        until = [until]
    end = len(text)
    for stop in until or ():
        if stop:
            index = text.find(stop)
            if index >= 0:
                end = min(end, index)
    return text[:end]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return _env_int("RANK", 0)


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return _env_int("WORLD_SIZE", 1)


def _local_rank() -> int:
    return _env_int("LOCAL_RANK", _global_rank())


def _rank_label() -> str:
    return f"GlobalRank {_global_rank()}/{_world_size()} (local {_local_rank()})"


def _code_version() -> dict:
    """The git commit this eval ran from and whether tracked files differed from it (None outside git)."""
    import subprocess
    root = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(dirty)}


def _eval_verbose() -> bool:
    return os.environ.get("LOGKV_EVAL_VERBOSE", "0") == "1"


def _eval_print(*args, **kwargs) -> None:
    if _is_main() or _eval_verbose():
        print(*args, **kwargs)


def _eval_progress(items, desc: str):
    """Rank 0's shard as one short line at the start, at every 10% and at the end (with the time left), and at
    most every 5 minutes in between when items are slow; never synchronizes model calls."""
    if _eval_verbose():
        yield from tqdm.tqdm(items, desc=f"{desc} {_rank_label()}", position=_global_rank())
        return
    if not _is_main():
        yield from items
        return
    total = len(items)
    label = f"{desc} [{_rank_label()} 分片]"
    started = last_report = time.perf_counter()
    print(f"⏳ {label}: 0/{total}", flush=True)
    step = max(1, -(-total // 10))
    for completed, item in enumerate(items, 1):
        yield item
        now = time.perf_counter()
        if completed == total or completed % step == 0 or now - last_report >= 300:
            left = (now - started) / completed * (total - completed)
            print(f"⏳ {label}: {completed}/{total} ({100 * completed // total}%)，已用 {now - started:.0f}s"
                  + (f"，约剩 {left:.0f}s" if completed < total else ""), flush=True)
            last_report = now


@contextlib.contextmanager
def _harness_logging():
    """Silence harness progress/INFO only during evaluation, preserving warnings and errors."""
    if _eval_verbose():
        yield
        return
    names = ("lm_eval", "datasets", "huggingface_hub", "transformers")
    loggers = {logging.getLogger(name) for name in names}
    loggers.update(logger for name, logger in logging.Logger.manager.loggerDict.copy().items()
                   if isinstance(logger, logging.Logger) and name.startswith(tuple(n + "." for n in names)))
    levels = {logger: logger.level for logger in loggers}
    original_init = tqdm.tqdm.__init__

    def quiet_init(self, *args, **kwargs):
        kwargs["disable"] = True
        original_init(self, *args, **kwargs)

    original_print = builtins.print

    def quiet_print(*args, **kwargs):
        # lm-eval's RULER builders print as they size each sample ("Max length ... | Current length ...
        # | Noises: ...", once per try and per rank); drop prints made from those modules only.
        # By the caller's file: lm-eval loads a task's utils through its YAML (!function) under the bare
        # module name (vt_utils), so __name__ does not say where the code lives.
        caller = sys._getframe(1)
        where = caller.f_code.co_filename.replace("\\", "/")
        if "/lm_eval/tasks/ruler/" in where or caller.f_globals.get("__name__", "").startswith("lm_eval.tasks.ruler"):
            return
        original_print(*args, **kwargs)

    # Rank 0 keeps warnings (but datasets' offline-cache notices, one per dataset load); the other ranks keep
    # errors only, and Python warnings stay on rank 0. Errors and tracebacks are never filtered.
    def floor(logger):
        quiet = not _is_main() or logger.name == "datasets" or logger.name.startswith("datasets.")
        return logging.ERROR if quiet else logging.WARNING

    warning_filters = warnings.filters[:]
    try:
        for logger in loggers:
            logger.setLevel(max(logger.getEffectiveLevel(), floor(logger)))
        if not _is_main():
            warnings.simplefilter("ignore")
        # lm-eval imports tqdm in multiple modules, including locally during bootstrap.
        tqdm.tqdm.__init__ = quiet_init
        builtins.print = quiet_print
        yield
    finally:
        warnings.filters[:] = warning_filters
        builtins.print = original_print
        tqdm.tqdm.__init__ = original_init
        for logger, level in levels.items():
            logger.setLevel(level)


def _request_sample_id(req: Any, *, global_request_index: int, rank: int) -> str:
    """Stable cross-diagnostic id for one actual generate_until model call."""
    task_name = getattr(req, "task_name", None)
    doc_id = getattr(req, "doc_id", None)
    idx = getattr(req, "idx", None)
    metadata = getattr(req, "metadata", None)
    if isinstance(metadata, dict):
        task_name = task_name or metadata.get("task_name") or metadata.get("task")
        doc_id = doc_id if doc_id is not None else metadata.get("doc_id")
        idx = idx if idx is not None else metadata.get("idx")

    parts = [f"rank{rank}", f"global_req{int(global_request_index)}"]
    if task_name is not None:
        parts.append(f"task={task_name}")
    if doc_id is not None:
        parts.append(f"doc={doc_id}")
    elif idx is not None:
        parts.append(f"idx={idx}")
    return "|".join(str(part) for part in parts)


_DIST_ENV_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "LITGPT_EXPECTED_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID",
)


def _dist_env_text() -> str:
    return " ".join(f"{key}={os.environ.get(key, '<unset>')}" for key in _DIST_ENV_KEYS)


def _expected_world_size() -> int:
    for key in ("LITGPT_EXPECTED_WORLD_SIZE", "EXPECTED_WORLD_SIZE"):
        value = _env_int(key, 0)
        if value > 0:
            return value
    return 0


def _check_expected_world_size(stage: str) -> None:
    expected = _expected_world_size()
    if expected <= 0:
        return
    actual = _world_size()
    if actual != expected:
        raise RuntimeError(
            f"{stage}: distributed world size mismatch: actual={actual}, expected={expected}. "
            f"This usually means torchrun did not rendezvous across all nodes. env: {_dist_env_text()}"
        )


def _rendezvous_snapshot(stage: str) -> None:
    if not _eval_verbose():
        return
    import socket

    ts = datetime.now().strftime("%H:%M:%S")
    if dist.is_available() and dist.is_initialized():
        dist_text = (
            f"dist_initialized=True backend={dist.get_backend()} "
            f"rank={dist.get_rank()} world={dist.get_world_size()}"
        )
    else:
        dist_text = "dist_initialized=False backend=<unset> rank=<unset> world=<unset>"
    print(f"🧭 [{ts}][{socket.gethostname()}][{stage}] {dist_text} env: {_dist_env_text()}", flush=True)


def _is_main() -> bool:
    """全局 rank == 0（多节点下每个节点都有一个 local_rank 0，不能用 local_rank 判定）。"""
    return _global_rank() == 0


def _dist_ready() -> bool:
    """真正处于多进程集合通信状态时才为 True（单卡下所有集合操作退化为 no-op）。"""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _bcast_device(group=None) -> torch.device | None:
    """NCCL 后端下 broadcast_object_list 必须显式给设备，否则会走默认卡导致串扰。"""
    if dist.get_backend(group) == "nccl" and torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return None


def _broadcast_obj(obj: Any, src: int = 0, group=None) -> Any:
    """把 rank ``src`` 上的任意可 pickle 对象广播给所有 rank（单卡时原样返回）。"""
    if not _dist_ready():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src, group=group, device=_bcast_device(group))
    return box[0]


_SYNC_SEQ = 0


def _wait_all_ranks(stage: str) -> float:
    """所有 rank 汇合，返回本 rank 的等待秒数；等待期间定期打印仍未到达的 rank。

    不用 ``monitored_barrier``：它等待时进程没有任何输出、GPU 利用率恒定，ModelArts
    卡死检测（进程 IO 不变 + GPU 利用率不变，默认 30 min）会把"等最慢的 rank / 等 rank 0
    写结果"判成卡死，且要到 12h 超时才报告缺席 rank。这里改在默认进程组的 TCPStore 上
    计数：CPU 轮询，每 ``LOGKV_EVAL_SYNC_HEARTBEAT_S``（默认 300s）打印一次缺席 rank；
    超过 ``LOGKV_EVAL_SYNC_TIMEOUT_S``（默认 7200s，<=0 不限）仍未到齐就报错。
    所有 rank 必须按相同顺序调用（键按调用序号区分）。
    """
    global _SYNC_SEQ
    if not _dist_ready():
        return 0.0
    _SYNC_SEQ += 1
    key = f"logkv_eval_sync/{_SYNC_SEQ}"
    world = _world_size()
    store = dist.distributed_c10d._get_default_store()
    store.set(f"{key}/rank{_global_rank()}", "1")
    store.add(key, 1)
    heartbeat_s = max(_env_int("LOGKV_EVAL_SYNC_HEARTBEAT_S", 300), 1)
    timeout_s = _env_int("LOGKV_EVAL_SYNC_TIMEOUT_S", 7200)
    t0 = time.perf_counter()
    next_report = heartbeat_s
    while store.add(key, 0) < world:
        waited = time.perf_counter() - t0
        timed_out = 0 < timeout_s <= waited
        if timed_out or waited >= next_report:
            missing = [r for r in range(world) if not store.check([f"{key}/rank{r}"])]
            message = f"[{_rank_label()}] {stage}: 已等待 {waited:.0f}s，仍未到达的 rank: {missing}"
            if timed_out:
                raise RuntimeError(f"{message}（超过 LOGKV_EVAL_SYNC_TIMEOUT_S={timeout_s}）")
            print(f"⏳ {message}", flush=True)
            next_report += heartbeat_s
        time.sleep(1.0)
    return time.perf_counter() - t0


def _hb(stage: str) -> None:
    """阶段里程碑默认只打印主进程；详细模式带上所有 rank。"""
    import socket
    host = socket.gethostname()
    ts = datetime.now().strftime("%H:%M:%S")
    _eval_print(f"💓 [{ts}][{_rank_label()}][{host}] {stage}", flush=True)


class SafeJSONEncoder(json.JSONEncoder):
    """处理无法直接序列化的对象（numpy、torch、函数等）"""
    def default(self, obj):
        # numpy 类型
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        # torch 类型
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        # 函数、方法、可调用对象
        if callable(obj):
            return f"<function: {obj.__name__ if hasattr(obj, '__name__') else str(obj)}>"
        # 其他特殊类型
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        # 类实例
        if hasattr(obj, '__dict__'):
            return f"<{obj.__class__.__name__} object>"
        # 默认处理
        return super().default(obj)


def extract_results_to_csv(results: dict, benchmark: str, output_path: Path) -> None:
    """
    从 results 中提取各数据集的 acc 或 acc_norm 数值，输出为 CSV 文件。
    - HellaSwag 数据集：提取 acc_norm,none
    - 其他数据集：提取 acc,none
    """
    csv_rows = []
    tasks = benchmark.split(",") if benchmark != "debug" else ["piqa"]

    results_data = results.get("results", {})

    for task in tasks:
        task = task.strip()
        task_results = results_data.get(task, {})

        # HellaSwag 使用 acc_norm，其他使用 acc
        if task.lower() in ["hellaswag", "arc-c", "openbookqa"]:
            metric_key = "acc_norm,none"
            metric_name = "acc_norm"
        else:
            metric_key = "acc,none"
            metric_name = "acc"

        value = task_results.get(metric_key, None)

        csv_rows.append({
            "dataset": task,
            # "metric": metric_name,
            "value": value,
        })

    # 写入 CSV
    csv_file = output_path.with_suffix(".csv")
    try:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "value"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"📊 CSV 结果已保存到: {csv_file}")
    except Exception as e:
        print(f"❌ 保存 CSV 失败: {e}")

BASE = os.environ.get('SNAPKV_HF_CACHE', os.environ.get('HF_HOME', '/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache'))
os.environ['HF_HOME'] = BASE
os.environ['HF_DATASETS_CACHE'] = f'{BASE}/hf_cache'
os.environ['HF_EVALUATE_CACHE'] = f'{BASE}/evaluate'
os.environ['HF_MODULES_CACHE'] = f'{BASE}/modules'
os.environ['HUGGINGFACE_HUB_CACHE'] = f'{BASE}/hub'
os.environ['HF_HUB_CACHE'] = f'{BASE}/hub'
os.environ['RULER_CACHE_DIR'] = f'{BASE}/ruler_cache'
os.environ['NLTK_DATA'] = f'{BASE}/nltk_data'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_DATASETS_IN_MEMORY_MAX_SIZE'] = '0'
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# 每个 rank 各自独立加载一遍数据集，HF datasets 的 "Found the latest cached
# dataset configuration ..." 日志和内部 tqdm 进度条会被重复打印 rank 份，
# 跟评测本身的计算进度混在一起。这里关掉，计算进度由 _eval_progress 报告。
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
import datasets.config as _hf_datasets_config
import huggingface_hub.constants as _hf_hub_constants

if _eval_verbose():
    print(f"[{_rank_label()}] 环境变量 datasets: {os.environ.get('HF_DATASETS_CACHE')}", flush=True)
    print(f"[{_rank_label()}] 库实际使用 datasets: {_hf_datasets_config.HF_DATASETS_CACHE}", flush=True)
    print(f"[{_rank_label()}] 环境变量 hub: {os.environ.get('HF_HUB_CACHE')}", flush=True)
    print(f"[{_rank_label()}] 库实际使用 hub: {_hf_hub_constants.HF_HUB_CACHE}", flush=True)

try:
    import datasets as _hf_datasets
    _hf_datasets.disable_progress_bars()
    _hf_datasets.logging.set_verbosity_error()
except Exception:
    pass

import dataclasses

from snapkv._vendor.niahlogkv.litgpt.config import Config
from snapkv._vendor.niahlogkv.litgpt.alpha_log_kv import WORD_SCORES
from snapkv._vendor.niahlogkv.litgpt.model import GPT
from snapkv._vendor.niahlogkv.litgpt.tokenizer import Tokenizer
from lm_eval import evaluator
from lm_eval.api.model import LM

# Repository tasks (e.g. eval_tasks/niah_cue) beside lm-eval's own.
EVAL_TASKS_DIR = Path(__file__).resolve().parent / "eval_tasks"


# Tasks named for a context length (eval_tasks/babilong_long, long_ppl, longbench2_long, niah_turns, infinitebench);
# a long_ppl_books_<L>_<point> task is named for its configuration's length L.
_LENGTH_NAMED_TASK = re.compile(
    r"^(babilong(?:_qa\d+)?|long_ppl(?:_books)?|longbench2|niah_turns|infinitebench(?:_[a-z_]+)?)_(\d+)([km])(?:_\d+[km])?$")


def _check_task_lengths(tasks, max_seq_length):
    """A task named for a context length must fit the model's window: longer inputs are cut from the start
    without a word, and the result would still carry the name. LongBench v2's per-length groups take their
    budget from the window itself, so for them the window must not be larger either (block_size = length plus
    a generation margin of at most 4096). InfiniteBench cuts its prompts to the length itself and generates in a
    1024-token margin beyond it (eval_tasks/infinitebench), so it needs that margin."""
    for task in tasks:
        match = _LENGTH_NAMED_TASK.match(task.strip())
        if not match:
            continue
        length = int(match.group(2)) * (1024 if match.group(3) == "k" else 1024 * 1024)
        need = length + (1024 if match.group(1).startswith("infinitebench") else 0)  # long_eval_plan.MARGIN
        if max_seq_length < need or (match.group(1) == "longbench2" and max_seq_length > length + 4096):
            raise ValueError(f"task {task} needs a model window of {need} tokens"
                             + (f" (at most {length + 4096})" if match.group(1) == "longbench2" else "")
                             + f", but max_seq_length is {max_seq_length}: set config_overrides.block_size")


def _longbench2_files_problem():
    """LongBench v2 is required; report missing subsets and damaged cache files before loading a model."""
    from snapkv._vendor.niahlogkv.litgpt.longbench2_patch import data_problem

    return data_problem()


INFINITEBENCH_FILES = ("passkey", "number_string", "kv_retrieval", "longbook_sum_eng", "longbook_qa_eng",
                       "longbook_choice_eng", "longbook_qa_chn", "longdialogue_qa_eng", "code_debug", "math_find")


def _infinitebench_files_problem():
    """Why InfiniteBench's data cannot be read offline, or None: its tasks read the jsonl files from
    INFINITEBENCH_DIR, else <HF_HOME>/infinitebench, else the hub cache, following flattened links to their blobs
    (eval_tasks/infinitebench: locate). The message says what this machine holds for the first missing file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "infinitebench_utils", EVAL_TASKS_DIR / "infinitebench" / "infinitebench_utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    missing, first = [], ""
    for name in INFINITEBENCH_FILES:
        path, seen = utils.locate(name)
        if path is None:
            missing.append(name)
            first = first or seen
    if missing:
        return (f"InfiniteBench files missing on this machine: {missing[:3]}"
                + (f" and {len(missing) - 3} more" if len(missing) > 3 else "")
                + f". Seen for {missing[0]}: {first}. (unused/fetch_long_eval_data.py --infinitebench with this HF_HOME,"
                + " then make sure this machine sees it; or INFINITEBENCH_DIR)")
    return None


def _check_task_data(tasks):
    """A task asked for by name whose data is not on disk fails here, with what to run, instead of in every rank's
    dataset loading (InfiniteBench reads its jsonl files offline from INFINITEBENCH_DIR or the hub cache)."""
    if any(t.strip() == "longbench2" or t.strip().startswith("longbench2_") for t in tasks):
        problem = _longbench2_files_problem()
        if problem:
            raise FileNotFoundError(problem)
    if any(t.strip().startswith("infinitebench_") for t in tasks):
        problem = _infinitebench_files_problem()
        if problem:
            raise FileNotFoundError(problem)


def _drop_empty_lengths(results):
    """Remove RULER's per-length metrics that had no sample (value -1).

    lm-eval's RULER scores every sample under its default length too (DEFAULT_SEQ_LENGTHS, 4096) with -1, so a
    run at other lengths reports "4096 ... -1.0000" for every task. Such an entry says only that no sample had
    that length; it is dropped (with its stderr) from the table and the saved results."""
    for section in ("results", "groups"):
        for metrics in (results.get(section) or {}).values():
            empty = [k for k, v in metrics.items()
                     if k.split(",")[0].isdigit() and isinstance(v, (int, float)) and v == -1]
            for key in empty:
                metric, _, filt = key.partition(",")
                metrics.pop(key, None)
                metrics.pop(f"{metric}_stderr,{filt}", None)
    return results


def _skip_optional_tasks(tasks, max_seq_length):
    """The launchers' default lists mark tasks that some checkpoints or setups cannot serve (EVAL_OPTIONAL_TASKS,
    e.g. the 32K turns and book curve): such a task is left out, with the reason, when the model's window is
    shorter than its length (a 4K run) or its data is not given; a task asked for by name must fit or fails
    (_check_task_lengths)."""
    optional = {t for t in os.environ.get("EVAL_OPTIONAL_TASKS", "").split(",") if t}
    kept = []
    for task in tasks:
        reason = None
        if task in optional and not task.startswith("longbench2_"):
            try:
                _check_task_lengths([task], max_seq_length)
            except ValueError as error:
                reason = str(error)
            if reason is None and task.startswith("long_ppl_books") and not os.environ.get("LONG_PPL_BOOKS"):
                reason = "LONG_PPL_BOOKS is not set (unused/fetch_long_eval_data.py --ppl-books-hub emozilla/pg19)"
            elif reason is None and task.startswith("long_ppl_") and not task.startswith("long_ppl_books") \
                    and not os.environ.get("LONG_PPL_DATA"):
                reason = "LONG_PPL_DATA is not set (unused/fetch_long_eval_data.py --ppl-hub emozilla/pg19-test)"
            elif reason is None and task.startswith("infinitebench_"):
                reason = _infinitebench_files_problem()
        if reason is None:
            kept.append(task)
        elif _is_main():
            print(f"⚠️ [eval] 默认清单跳过 {task}: {reason}", flush=True)
    return kept


def _task_manager(metadata):
    """lm-eval's task index plus ``eval_tasks/``, carrying ``metadata`` as ``simple_evaluate`` would."""
    from lm_eval.tasks import TaskManager

    if not EVAL_TASKS_DIR.is_dir() or "metadata" not in inspect.signature(TaskManager).parameters:
        return None  # simple_evaluate builds its own, with lm-eval's tasks only.
    return TaskManager(include_path=str(EVAL_TASKS_DIR), metadata=dict(metadata or {}))
from snapkv._vendor.niahlogkv.litgpt.generate.base import generate as litgpt_generate
from snapkv._vendor.niahlogkv.litgpt.kv_compression import cache_compression_snapshot, summarize_compression
from snapkv._vendor.niahlogkv.litgpt.log_kv_diag import DIAG as LOG_KV_DIAG, diag_mode
from snapkv._vendor.niahlogkv.litgpt.ruler_dataset_cache import evaluation_sample_limit
from snapkv._vendor.niahlogkv.litgpt.ruler_patch import apply_patch
apply_patch()
from snapkv._vendor.niahlogkv.litgpt.longbench2_patch import apply_patch as apply_longbench2_patch  # noqa: E402
apply_longbench2_patch()
from snapkv._vendor.niahlogkv.litgpt.hf_cache_patch import apply_patch as apply_hf_cache_patch  # noqa: E402
apply_hf_cache_patch()  # offline loads skip <hash>.incomplete folders

_CONFIG_FIELDS = {f.name for f in dataclasses.fields(Config)}


def _unreadable_dataset_infos(root, dataset):
    """Cached ``dataset_info.json`` files of ``dataset`` that are not valid JSON.

    Offline, ``datasets`` reads every cached version of the dataset to pick a
    configuration, so one broken file fails the whole load without naming it.
    Size and modification time tell a cut-short copy, and when it was written.
    """
    try:
        from datasets.naming import camelcase_to_snakecase
    except ImportError:  # Diagnostics only; the cache itself needs datasets.
        camelcase_to_snakecase = str.lower
    name = camelcase_to_snakecase(str(dataset).rstrip("/").split("/")[-1])
    root = Path(root)
    try:
        entries = [entry for entry in root.iterdir() if entry.name == name or entry.name.endswith("___" + name)]
    except OSError as exc:
        return [dict(path=str(root), error=f"{type(exc).__name__}: {exc}")]
    broken = []
    for entry in sorted(entries):
        # <dataset>/<config>/<version>/<hash>/dataset_info.json
        for info in sorted(entry.glob("*/*/*/dataset_info.json")):
            try:
                json.loads(info.read_text(encoding="utf-8"))
                continue
            except json.JSONDecodeError as exc:
                lines = exc.doc.splitlines()
                broken.append(dict(path=str(info), error=str(exc),
                                   context=lines[max(0, exc.lineno - 3):exc.lineno + 1]))
            except (OSError, UnicodeDecodeError) as exc:
                broken.append(dict(path=str(info), error=f"{type(exc).__name__}: {exc}"))
            try:
                stat = info.stat()
                broken[-1].update(size=stat.st_size, modified=time.strftime(
                    "%Y-%m-%d %H:%M:%S %z", time.localtime(stat.st_mtime)))
            except OSError:
                pass
    return broken


def _walk_cache_path(path):
    """lstat each component of ``path`` in order and stop at the first one that cannot be entered.

    ENOTDIR only says *some* parent is not a directory; this names which one, what it
    really is in this process (symlink target, file contents), and which mount it lives on.
    """
    import re
    import stat

    def kind(mode):
        for name in ("DIR", "REG", "LNK", "FIFO", "SOCK", "CHR", "BLK"):
            if getattr(stat, f"S_IS{name}")(mode):
                return name.lower()
        return oct(stat.S_IFMT(mode))

    def error(exc):
        return f"{type(exc).__name__}: {exc}"

    mounts = []
    try:
        with open("/proc/self/mountinfo", encoding="utf-8", errors="backslashreplace") as stream:
            for line in stream:
                fields = line.split()
                tail = fields.index("-")
                mounts.append(dict(point=re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]),
                                   root=fields[3], dev=fields[2], options=fields[5],
                                   fstype=fields[tail + 1], source=fields[tail + 2]))
    except (OSError, ValueError, IndexError):
        pass

    parts = Path(os.path.abspath(path)).parts
    steps, previous_dev, last_dir = [], None, None
    for index in range(len(parts)):
        current = Path(*parts[:index + 1])
        step = dict(path=str(current))
        steps.append(step)
        try:
            info = os.lstat(current)
        except OSError as exc:
            step.update(error=error(exc), blocks_traversal=True)
            break
        dev = f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}"
        step.update(type=kind(info.st_mode), mode=oct(stat.S_IMODE(info.st_mode)),
                    size=info.st_size, dev=dev)
        if dev != previous_dev:
            # The entry lives on the fs holding its parent's real directory.
            physical = os.path.join(os.path.realpath(current.parent), current.name)
            covering = [m for m in mounts if m["point"] == "/" or physical == m["point"]
                        or physical.startswith(m["point"] + "/")]
            if covering:
                step["mount"] = max(covering, key=lambda m: len(m["point"]))
            previous_dev = dev
        entered = info
        if stat.S_ISLNK(info.st_mode):
            try:
                step["link"] = os.readlink(current)
                step["resolved"] = os.path.realpath(current)
                entered = os.stat(current)
            except OSError as exc:
                step.update(target_error=error(exc), blocks_traversal=True)
                break
            step["target_type"] = kind(entered.st_mode)
        if stat.S_ISDIR(entered.st_mode):
            last_dir = current
        elif index < len(parts) - 1:
            step["blocks_traversal"] = True
            if stat.S_ISREG(entered.st_mode):
                # An object store without symlink support may surface a link as a small file.
                try:
                    with open(current, "rb") as stream:
                        step["head"] = stream.read(256).decode("utf-8", "backslashreplace")
                except OSError as exc:
                    step["head_error"] = error(exc)
            try:
                step["xattrs"] = os.listxattr(current, follow_symlinks=False)
            except OSError:
                pass
            break
    if last_dir is not None:
        try:
            names = sorted(os.listdir(last_dir))
            steps.append(dict(listing_of=str(last_dir), entries=names[:50], total=len(names)))
        except OSError as exc:
            steps.append(dict(listing_of=str(last_dir), error=error(exc)))
    return steps


@contextlib.contextmanager
def _trace_hf_cache_resolution():
    """Log hidden cache errors; follow cache aliases that this mount cannot resolve."""
    import functools
    import importlib
    import socket
    import sys
    import traceback

    load = importlib.import_module("datasets.load")
    cfg = importlib.import_module("datasets.config")
    hub = importlib.import_module("huggingface_hub.constants")
    factory = getattr(load, "CachedDatasetModuleFactory", None)
    identity = dict(rank=_global_rank(), host=socket.gethostname(), pid=os.getpid(), python=sys.executable)

    def emit(event, **details):
        if event == "runtime" and not _eval_verbose():
            return
        # One line per event keeps rank and exception together in aggregated logs.
        print("[hf-cache] " + json.dumps(dict(event=event, **identity, **details),
                                        ensure_ascii=False, default=str), file=sys.stderr, flush=True)

    packages = {}
    for name in ("datasets", "huggingface_hub", "lm_eval"):
        module = importlib.import_module(name)
        packages[name] = dict(version=getattr(module, "__version__", None), file=module.__file__)
    emit("runtime", script=str(Path(__file__).resolve()), cwd=os.getcwd(), packages=packages,
         datasets_cache=cfg.HF_DATASETS_CACHE, hub_cache=hub.HF_HUB_CACHE,
         datasets_offline=cfg.HF_DATASETS_OFFLINE, hub_offline=hub.HF_HUB_OFFLINE,
         environment={key: os.environ.get(key) for key in (
             "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "HF_MODULES_CACHE",
             "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "HF_ENDPOINT", "PYTHONPATH", "PYTHON_EXEC")},
         cache_trace_supported=factory is not None)
    if factory is None:
        yield
        return

    datasets_module = importlib.import_module("datasets")
    original_load = datasets_module.load_dataset
    original = factory.get_module
    cache_misses = set()

    @functools.wraps(original)
    def traced_get_module(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except Exception as exc:
            if isinstance(exc, FileNotFoundError):
                cache_misses.add(self.name)
            emit("local_cache_failure", dataset=self.name, cache_dir_argument=self.cache_dir,
                 effective_cache_dir=os.path.expanduser(str(self.cache_dir or cfg.HF_DATASETS_CACHE)),
                 traceback=traceback.format_exc())
            raise

    def unusable_alias(root, path):
        """Return (alias, target) when the cache entry for ``namespace/name`` stands in for ``name``."""
        from datasets.naming import camelcase_to_snakecase

        namespace, name = path.split("/")
        alias = root / f"{namespace}___{camelcase_to_snakecase(name)}"
        target = root / camelcase_to_snakecase(name)
        try:
            # Anything this process can enter must be the legacy directory itself (a working link);
            # a real directory of its own is not an alias, and its failure is a different problem.
            if (not os.path.lexists(alias) or not target.is_dir()
                    or alias.is_dir() and not os.path.samefile(alias, target)):
                return None
        except OSError:
            return None
        return alias, target

    @functools.wraps(original_load)
    def load_with_local_fallback(*args, **kwargs):
        cache_misses.clear()
        try:
            return original_load(*args, **kwargs)
        except json.JSONDecodeError:
            path = args[0] if args else kwargs.get("path")
            root = Path(kwargs.get("cache_dir") or cfg.HF_DATASETS_CACHE).expanduser()
            emit("dataset_info_unreadable", dataset=path, cache_root=str(root),
                 files=_unreadable_dataset_infos(root, path),
                 fix="repair or re-copy the listed dataset_info.json; a duplicate hash directory may be removed")
            raise
        except (ConnectionError, FileNotFoundError):
            path = args[0] if args else kwargs.get("path")
            if not (cfg.HF_DATASETS_OFFLINE and isinstance(path, str) and path.count("/") == 1
                    and path in cache_misses):
                raise
            root = Path(kwargs.get("cache_dir") or cfg.HF_DATASETS_CACHE).expanduser()
            found = unusable_alias(root, path)
            legacy = path.split("/")[1]
            packaged = importlib.import_module("datasets.packaged_modules")._PACKAGED_DATASETS_MODULES
            # The legacy name must reach the same cache lookup, not a packaged builder or a local folder.
            if found is None or legacy in packaged or os.path.exists(legacy):
                raise
            alias, target = found
            config = args[1] if len(args) > 1 else kwargs.get("name")
            emit("cache_alias_unusable", dataset=path, alias=str(alias), target=str(target),
                 walk=_walk_cache_path(alias / (config or "default")))
            # Loading under the legacy name is what following namespace___name -> name would do.
            if args:
                result = original_load(legacy, *args[1:], **kwargs)
            else:
                result = original_load(**dict(kwargs, path=legacy))
            emit("cache_alias_followed", dataset=path, loaded_as=legacy, directory=str(target),
                 num_rows=getattr(result, "num_rows", None))
            return result

    factory.get_module = traced_get_module
    datasets_module.load_dataset = load_with_local_fallback
    try:
        yield
    finally:
        factory.get_module = original
        datasets_module.load_dataset = original_load


def _checkpoint_file_for_loading(path: Path, wait_s: float = 120.0) -> Path:
    """Resolve checkpoint links flattened to path text by a bucket mount, without writing to that mount.

    Keep the selected checkpoint: following its reference is allowed; substituting an older step is not.
    Non-main ranks share one bounded metadata-visibility wait across the entire reference chain.
    """
    import socket

    origin = Path(os.path.abspath(os.path.expandvars(os.path.expanduser(str(path)))))
    current, chain, seen = origin, [], set()
    deadline = time.monotonic() + max(wait_s, 0) if _dist_ready() and not _is_main() else None

    def details(message):
        return (f"{message}. Requested checkpoint: {origin}; reference chain: {' -> '.join(chain)}; "
                f"observed on host {socket.gethostname()} [{_rank_label()}]. "
                "Copy the actual binary checkpoint as an ordinary file to the selected location; "
                "an older step will not be substituted automatically.")

    for _ in range(16):
        name = str(current)
        chain.append(name)
        if name in seen:
            raise ValueError(details("Checkpoint reference cycle"))
        seen.add(name)
        waited = False
        while not (current.exists() or current.is_symlink()):
            remaining = deadline - time.monotonic() if deadline is not None else 0
            if remaining <= 0:
                raise FileNotFoundError(details(f"Checkpoint file or reference target is missing: {current}"))
            time.sleep(min(2.0, remaining))
            waited = True
        if waited:
            print(f"[eval][{_rank_label()}] 共享盘元数据延迟，等待后已看到 {current}", flush=True)
        if current.is_symlink():
            target = os.readlink(current)
        else:
            if current.is_dir():
                raise FileNotFoundError(details(
                    f"{current} is a directory (FSDP/DCP shards); first consolidate it with "
                    "utils.convert_and_replace_fsdp_ckpt"))
            with open(current, "rb") as stream:
                head = stream.read(4097)  # a path placeholder is small; never read the full weights here
            if not head:
                raise ValueError(details(f"Checkpoint file is empty: {current}"))
            target = None
            if len(head) <= 4096:
                try:
                    text = head.decode("utf-8").strip("\x00\r\n\t ")
                except UnicodeDecodeError:
                    text = ""
                if text.startswith("version https://git-lfs.github.com/spec/v1"):
                    raise ValueError(details(f"{current} contains a Git LFS pointer, not checkpoint weights"))
                if (text and "://" not in text and not any(ord(c) < 32 for c in text)
                        and (text.startswith(("/", "./", "../")) or "/" in text
                             or text.endswith((".pth", ".pt", ".ckpt")))):
                    target = text
            if target is None:
                return current
        next_path = Path(target)
        if not next_path.is_absolute():
            next_path = current.parent / next_path
        current = Path(os.path.abspath(next_path))
        if _is_main():
            print(f"[eval] Checkpoint reference: {name} -> {current}", flush=True)
    raise ValueError(details("Too many checkpoint reference hops (limit 16)"))


def _load_lit_model_checkpoint(
    checkpoint_dir: str, map_location: str | torch.device, wait_s: float = 120.0
) -> Any:
    """加载 ``{checkpoint_dir}/lit_model.pth``（单文件 ``torch.save``，与 demo FSDP ``state_dict_type='full'`` 一致）。

    多机场景下 rank 0 已经确认过文件存在（见 ``_resolve_checkpoint_dir``），但其它
    节点的共享盘元数据缓存可能还没刷新，包括路径占位文件指向的真实文件。直接抛 FileNotFoundError 会让这个 rank
    单独退出，其余 rank 卡在后续集合通信里等到超时；所以非 rank0 上先轮询等待
    ``wait_s`` 秒。单进程运行时不等待，行为与之前一致（立刻报错）。
    """
    import pickle
    import socket

    requested = Path(checkpoint_dir).expanduser() / "lit_model.pth"
    lit_path = _checkpoint_file_for_loading(requested, wait_s=wait_s)
    if _is_main():
        print(f"[eval] Checkpoint binary: {lit_path} ({lit_path.stat().st_size} bytes)", flush=True)
    try:
        return torch.load(str(lit_path), map_location=map_location, weights_only=False)
    except (pickle.UnpicklingError, EOFError) as error:
        with open(lit_path, "rb") as stream:
            head = stream.read(160)
        raise ValueError(
            f"Checkpoint is not a readable torch.save binary: requested={requested}, actual={lit_path}, "
            f"size={lit_path.stat().st_size}, starts={head!r}, host={socket.gethostname()}, rank={_rank_label()}. "
            "Restore the actual binary weights as an ordinary file at this path; no older step was selected."
        ) from error


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _filter_config_dict(d: dict[str, Any]) -> dict[str, Any]:
    dropped = sorted(k for k in d if k not in _CONFIG_FIELDS)
    if dropped and (_is_main() or _eval_verbose()):
        print(f"[eval] 过滤掉非 Config 字段: {dropped}")
    return {k: v for k, v in d.items() if k in _CONFIG_FIELDS}


def _normalize_training_config_dict(d: dict[str, Any]) -> tuple[dict[str, Any], list[int] | None]:
    """训练 exp / CLI 里常用 research_*_layers_str；Config 只接受 research_prefill_swa_layers 等。"""
    out = dict(d)
    identity_layers: list[int] | None = None
    if "research_swa_layers_str" in out:
        out["research_prefill_swa_layers"] = _parse_csv_ints(out.pop("research_swa_layers_str"))
    return out, identity_layers


def _configure_inference_memory(config, overrides=None):
    """Use the same token-wise MLP budget for dense and compressed evaluation, without changing attention."""
    value = os.environ.get("LOGKV_EVAL_MLP_CHUNK", (overrides or {}).get("inference_mlp_chunk_size", 2048))
    try:
        chunk = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("LOGKV_EVAL_MLP_CHUNK / inference_mlp_chunk_size must be an integer >= 0") from error
    if chunk < 0:
        raise ValueError("LOGKV_EVAL_MLP_CHUNK / inference_mlp_chunk_size must be an integer >= 0")
    config.inference_mlp_chunk_size = chunk
    return chunk


def _score_continuation_logits(logits, targets):
    # Bound the fp32 log-softmax workspace while scoring every continuation token.
    total_logprob = 0.0
    is_greedy = True
    for i in range(0, logits.size(0), 1024):
        block = logits[i:i + 1024].float()
        target = targets[i:i + 1024]
        log_probs = torch.log_softmax(block, dim=-1)
        total_logprob += log_probs.gather(dim=-1, index=target.unsqueeze(-1)).sum().item()
        if is_greedy:
            is_greedy = bool((block.argmax(dim=-1) == target).all().item())
    return total_logprob, is_greedy


def _config_from_yaml_and_overrides(config_path: str, overrides: dict[str, Any] | None) -> Config:
    with open(config_path, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    if base is None:
        raise ValueError(f"{config_path} is empty or invalid YAML.")
    merged = {**base, **(overrides or {})}
    merged, identity_layers = _normalize_training_config_dict(merged)
    merged = _filter_config_dict(merged)
    cfg = Config(**merged)
    if identity_layers is not None:
        cfg.research_prefill_identity_layers = identity_layers
    return cfg


def _config_for_evaluation(checkpoint_dir: str, overrides: dict[str, Any] | None = None,
                           arch_name: str | None = None) -> Config:
    """Prefer the saved architecture; use the explicit model name for legacy experiment directories."""
    config_path = Path(checkpoint_dir) / "model_config.yaml"
    if config_path.is_file():
        return _config_from_yaml_and_overrides(str(config_path), overrides)
    fallback = {"name": arch_name or Path(checkpoint_dir).name, **(overrides or {})}
    return Config.from_name(**_filter_config_dict(fallback))


# ==========================================
# 🧩 logKV 专属：YAML --config 机制（与 demo.py 相同约定：
# YAML 非 null 值覆盖 CLI；支持 "config:" 继承）
# ==========================================

def _load_yaml_config(yaml_path: str, model_dir: str) -> dict:
    """Load YAML config with inheritance ('config:' key overrides)."""
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
        base_dir = os.path.dirname(yaml_path)
        base_path = os.path.join(base_dir, base_name)
        base_cfg = _load_yaml_config(base_path, model_dir)
        return {**base_cfg, **cfg}

    return cfg


# PyYAML implements YAML 1.1: scientific notation WITHOUT a decimal point
# ("2e-5") does not match its float resolver and loads as *str*. Coerce such
# top-level values so numeric params never receive strings.
_SCI_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d*)?[eE][-+]?\d+")


def _coerce_yaml_sci_floats(cfg: dict) -> dict:
    """Convert top-level str values that are scientific-notation numbers to float."""
    return {
        k: float(v) if isinstance(v, str) and _SCI_FLOAT_RE.fullmatch(v) else v
        for k, v in cfg.items()
    }


# ==========================================
# 🧩 logKV 专属：tokenizer 目录回退解析
# ==========================================

def _probe_tokenizer_dir(checkpoint_dir: str, tokenizer_dir: str | None) -> Path:
    """Resolve a directory that actually holds tokenizer.json / tokenizer.model.

    The training save dir normally receives a copy of the tokenizer files
    (demo.py copies them next to lit_model.pth), but older checkpoints,
    interrupted runs, or ``save_ckpt: false`` leave it without one. litgpt's
    ``Tokenizer`` then raises a bare ``NotImplementedError``, so resolve the
    directory up front and fail with an actionable message instead.

    Search order:
      1. explicit ``tokenizer_dir`` (CLI/YAML),
      2. ``checkpoint_dir`` itself,
      3. the base model dir recorded in ``model_config.yaml``
         (demo.py convention: checkpoints/<hf org>/<hf name>).

    仅供 ``_find_tokenizer_dir`` 在 rank 0 上调用 —— 探测结果由 rank 0 广播，
    不要在各 rank 上分别调用（分支不一致会挂死，见文件顶部的分布式约定）。
    """
    candidates: list[tuple[str, Path]] = []
    if tokenizer_dir is not None:
        candidates.append(("--tokenizer_dir", Path(tokenizer_dir)))
    candidates.append(("checkpoint_dir", Path(checkpoint_dir)))

    cfg_path = Path(checkpoint_dir) / "model_config.yaml"
    if cfg_path.is_file():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            hf = cfg.get("hf_config") or {}
            if hf.get("org") and hf.get("name"):
                candidates.append(
                    ("base checkpoint (from model_config.yaml)",
                     Path("checkpoints") / hf["org"] / hf["name"])
                )
        except Exception as e:  # noqa: BLE001 — fallback probing only
            print(f"[eval] WARNING: could not read {cfg_path} for tokenizer fallback: {e}")

    for label, d in candidates:
        if (d / "tokenizer.json").is_file() or (d / "tokenizer.model").is_file():
            return d

    tried = "\n".join(f"  - {label}: {d.resolve()}" for label, d in candidates)
    raise FileNotFoundError(
        "No tokenizer.json / tokenizer.model found. Searched:\n"
        f"{tried}\n"
        "Fix: pass --tokenizer_dir <dir containing the tokenizer files>, or copy "
        "the base model's tokenizer files into the checkpoint dir. Note that a "
        "training run with `save_ckpt: false` writes NO checkpoint (no weights, "
        "no tokenizer) — the train->eval pipeline requires `save_ckpt: true`."
    )


def _find_tokenizer_dir(checkpoint_dir: str, tokenizer_dir: str | None, group=None) -> Path:
    """rank 0 探测 tokenizer 目录并广播结论；失败信息也一并广播。

    每个 rank 自己探测时，共享盘元数据缓存不一致会让一部分 rank 找到目录、另一部分
    抛 FileNotFoundError 单独退出，剩下的 rank 就卡在后面的集合通信里。这里让 rank 0
    独自判定，成功广播路径、失败广播错误信息，保证所有 rank 要么一起继续、要么一起
    以同一条报错退出。
    """
    payload: tuple[str, str] | None = None
    if _is_main():
        try:
            payload = ("ok", str(_probe_tokenizer_dir(checkpoint_dir, tokenizer_dir)))
        except FileNotFoundError as e:
            payload = ("err", str(e))

    kind, value = _broadcast_obj(payload, group=group)
    if kind == "err":
        raise FileNotFoundError(value)
    return Path(value)


def _synchronize_inference_device(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _reset_inference_peak_memory(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _inference_peak_memory(device) -> dict:
    if torch.device(device).type != "cuda":
        return {"peak_memory_allocated_bytes": None, "peak_memory_reserved_bytes": None}
    return {
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def summarize_inference_metrics(gen_metrics: list[dict], ppl_metrics: list[dict]) -> dict:
    """Pool token counts and stage times across requests, never average their rates."""
    def rate(tokens, seconds):
        return tokens / seconds if tokens > 0 and seconds > 0 else None

    def peak_memory(rows):
        return {
            key: max((row[key] for row in rows if row[key] is not None), default=None)
            for key in ("peak_memory_allocated_bytes", "peak_memory_reserved_bytes")
        }

    prompt_tokens = sum(row["prompt_len"] for row in gen_metrics)
    generated_tokens = sum(row["generated_tokens"] for row in gen_metrics)
    decode_tokens = sum(row["decode_tokens"] for row in gen_metrics)
    total_time = sum(row["total_time_s"] for row in gen_metrics)
    prefill_time = sum(row["prefill_time_s"] for row in gen_metrics)
    decode_time = sum(row["decode_time_s"] for row in gen_metrics)
    forward_tokens = sum(row["total_seq_len"] for row in ppl_metrics)
    forward_time = sum(row["forward_time_s"] for row in ppl_metrics)
    return {
        "throughput_scope": "Effective single-device throughput: sum(tokens) / sum(request stage seconds) across all ranks; not aggregate cluster throughput.",
        "timing_scope": "Model calls only, excluding tokenization, cache setup/reset, result gathering, and file output. Prefill includes first-token sampling; decode excludes the first generated token. Each multi-turn answer is one generation request, with only newly forwarded tokens counted as its prompt. Generation counts include EOS; no warmup requests are discarded.",
        "memory_scope": "Maximum per-request CUDA allocator peak across ranks, including resident model and cache; not KV-only memory or total device memory. CPU values are null.",
        "generation": {
            "requests": len(gen_metrics),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "decode_tokens": decode_tokens,
            "total_time_s": total_time,
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "avg_ttft_ms": 1000 * prefill_time / len(gen_metrics) if gen_metrics else None,
            "prefill_tokens_per_sec": rate(prompt_tokens, prefill_time),
            "decode_tokens_per_sec": rate(decode_tokens, decode_time),
            "gen_tokens_per_sec": rate(generated_tokens, total_time),
            "decode_ms_per_token": 1000 * decode_time / decode_tokens if decode_tokens else None,
            **peak_memory(gen_metrics),
        },
        "scoring": {
            "requests": len(ppl_metrics),
            "input_tokens": forward_tokens,
            "forward_time_s": forward_time,
            "forward_tokens_per_sec": rate(forward_tokens, forward_time),
            **peak_memory(ppl_metrics),
        },
    }


class LogKVLM(LM):
    """LM wrapper that scores and generates through LogKV, SWA, or dense KV cache.

    By default, loglikelihood scoring and generate_until use the merged-position
    slot cache from the LogKV training path: full post-RoPE keys are stored as
    exact recent tokens or compressed slots, and slot attention scores one logit
    per slot with the log(w) mass bias. With ``log_kv_dense_mode=True``, the same
    request/scoring code builds the native full KV cache instead, so attention is
    ordinary causal dense attention with no LogKV slots or corrections.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        config_overrides: dict[str, Any] | None = None,
        log_kv_B: int = 512,
        log_kv_recent_size: int = 1024,
        log_kv_prefill_block: int = 256,
        log_kv_second_order_scale: float = 1.0,
        log_kv_dense_mode: bool = False,
        swa_window_size: int | None = None,
        log_kv_importance_pooling: bool = False,
        log_kv_importance_pooling_lambda: float = 1.0,
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
        log_kv_semantic_legacy_route: bool = False,
        log_kv_semantic_unified_route: bool = False,
        log_kv_semantic_merge_passes: int = 1,
        log_kv_semantic_anchor_mode: str = "multi",
        log_kv_semantic_pack_backend: str = "auto",
        log_kv_semantic_centroid_backend: str = "sequential",
        log_kv_semantic_replay_updates: bool = False,
        log_kv_alpha_exact_tokens: int = 0,
        log_kv_alpha_span_max_tokens: int = 64,
        log_kv_beta_novelty: bool = False,
        log_kv_beta_adaptive_merge: bool = False,
        log_kv_gamma_level0_reinsert: bool = False,
        log_kv_gamma_top_merge: str = "fold",
        log_kv_gamma_level_slack: int = 2,
        log_kv_gamma_cue_tokens: int = 0,
        log_kv_gamma_literals: bool = False,
        log_kv_gamma_cue_guard: bool = False,
        log_kv_gamma_query_cue: bool = False,
        log_kv_gamma_query_detector: str | None = None,
        log_kv_gamma_query_detector_options: dict | None = None,
        log_kv_gamma_query_pool_tokens: int = 0,
        log_kv_retention_heads: str | None = None,
        log_kv_gamma_surprisal: str | None = None,
        tokenizer_dir: str | None = None,
        process_group=None,
        arch_name: str | None = None,
    ):
        super().__init__()
        self.process_group = process_group
        self._device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_kv_B = log_kv_B
        self.log_kv_recent_size = log_kv_recent_size
        self.log_kv_prefill_block = log_kv_prefill_block
        self.log_kv_second_order_scale = float(log_kv_second_order_scale)
        self.log_kv_dense_mode = bool(log_kv_dense_mode)
        if swa_window_size is not None and (swa_window_size < 0 or log_kv_dense_mode):
            raise ValueError("swa_window_size must be >= 0 and cannot be combined with dense mode")
        self.swa_window_size = swa_window_size
        self.cache_budget = {}
        self._truncation_counts = {}
        if swa_window_size:
            config_overrides = {**(config_overrides or {}), "sliding_window_size": swa_window_size,
                                "sliding_window_indices": None}
        self.log_kv_importance_pooling = bool(log_kv_importance_pooling)
        self.log_kv_importance_pooling_lambda = float(log_kv_importance_pooling_lambda)
        self.log_kv_importance_pooling_temperature = float(log_kv_importance_pooling_temperature)
        self.log_kv_semantic_clusters = bool(log_kv_semantic_clusters)
        self.log_kv_cluster_k_max = int(log_kv_cluster_k_max)
        self.log_kv_cluster_lambda_rel = float(log_kv_cluster_lambda_rel)
        self.log_kv_seg_eta = float(log_kv_seg_eta)
        self.log_kv_seg_g0 = float(log_kv_seg_g0)
        self.log_kv_seg_gap_max = None if log_kv_seg_gap_max is None else float(log_kv_seg_gap_max)
        self.log_kv_seg_block_level = int(log_kv_seg_block_level)
        self.log_kv_seg_forget = float(log_kv_seg_forget)
        self.log_kv_semantic_s_h_path = log_kv_semantic_s_h_path
        self.log_kv_semantic_flush_granularity = int(log_kv_semantic_flush_granularity)
        self.log_kv_semantic_cluster_chunk_size = int(log_kv_semantic_cluster_chunk_size)
        self.log_kv_semantic_capacity_beta = float(log_kv_semantic_capacity_beta)
        self.log_kv_semantic_capacity_hard_cap_mult = float(log_kv_semantic_capacity_hard_cap_mult)
        self.log_kv_semantic_legacy_route = bool(log_kv_semantic_legacy_route)
        self.log_kv_semantic_unified_route = bool(log_kv_semantic_unified_route)
        self.log_kv_semantic_merge_passes = int(log_kv_semantic_merge_passes)
        self.log_kv_semantic_anchor_mode = log_kv_semantic_anchor_mode
        self.log_kv_semantic_pack_backend = log_kv_semantic_pack_backend
        self.log_kv_semantic_centroid_backend = log_kv_semantic_centroid_backend
        self.log_kv_semantic_replay_updates = log_kv_semantic_replay_updates
        self.log_kv_alpha_exact_tokens = log_kv_alpha_exact_tokens
        self.log_kv_alpha_span_max_tokens = log_kv_alpha_span_max_tokens
        self.log_kv_beta_novelty = bool(log_kv_beta_novelty)
        self.log_kv_beta_adaptive_merge = bool(log_kv_beta_adaptive_merge)
        self.log_kv_gamma_level0_reinsert = bool(log_kv_gamma_level0_reinsert)
        self.log_kv_gamma_top_merge = str(log_kv_gamma_top_merge)
        self.log_kv_gamma_level_slack = int(log_kv_gamma_level_slack)
        self.log_kv_gamma_cue_tokens = int(log_kv_gamma_cue_tokens)
        self.log_kv_gamma_literals = bool(log_kv_gamma_literals)
        self.log_kv_gamma_cue_guard = bool(log_kv_gamma_cue_guard)
        self.log_kv_gamma_query_cue = bool(log_kv_gamma_query_cue)
        self.log_kv_gamma_query_detector = log_kv_gamma_query_detector or None
        self._query_detector_options = dict(log_kv_gamma_query_detector_options or {})
        self._query_detector = None
        self.log_kv_gamma_query_pool_tokens = int(log_kv_gamma_query_pool_tokens)
        if self.log_kv_gamma_query_detector:
            from snapkv._vendor.niahlogkv.litgpt.query_detector import validate_local_model
            validate_local_model(self.log_kv_gamma_query_detector)
            if not self.log_kv_gamma_query_cue or self.log_kv_gamma_cue_tokens:
                raise ValueError("dynamic detector needs query_cue=true and cue_tokens=0 (no fixed head/tail cues)")
        self.log_kv_retention_heads = log_kv_retention_heads or None
        self.log_kv_gamma_surprisal = log_kv_gamma_surprisal or None

        # 控制打印：在多卡下尽量只让主进程打印，防止刷屏
        is_master = _is_main()

        resolved_tok_dir = _find_tokenizer_dir(checkpoint_dir, tokenizer_dir, group=process_group)
        if is_master:
            print(f"🔤 正在加载 tokenizer: {resolved_tok_dir}")
        self.tokenizer = Tokenizer(resolved_tok_dir)

        # ==========================================
        # 🌟 智能 Config 路由加载 (YAML 版本)
        # ==========================================
        config_path = os.path.join(checkpoint_dir, "model_config.yaml")

        if os.path.exists(config_path):
            if is_master: print(f"📄 发现专属架构 YAML 配置文件: {config_path}，正在自动同步架构...")
        else:
            if is_master: print("⚠️ 未发现训练期保存的 YAML 配置文件，使用显式架构名或已注册的目录名初始化...")
        self.config = _config_for_evaluation(checkpoint_dir, config_overrides, arch_name)
        # The length pass reserves an extra 1024 positions for generation.
        self._dense_256k_eval = (
            self.log_kv_dense_mode and self.config.block_size in (262144, 263168)
            and not self.config.latent_attention and self.config.sliding_window_size is None
            and self.config.attention_logit_softcapping is None
        )

        chunk = _configure_inference_memory(self.config, config_overrides)
        if is_master:
            print(f"🧩 Inference MLP chunk: {chunk} tokens (0 = full sequence); attention/KV policy is unchanged")
            if self._dense_256k_eval:
                print(f"🧩 Dense 256K: {self._prefill_chunk()} tokens per forward, fused GQA, full KV")

        # ==========================================
        self.log_kv_semantic_s_h = (
            load_log_kv_semantic_s_h(
                self.log_kv_semantic_s_h_path,
                n_layer=self.config.n_layer,
                n_groups=self.config.n_query_groups,
            )
            if self.log_kv_semantic_clusters
            else None
        )

        mode_name = "全程 SWA" if swa_window_size is not None else (
            "dense 标准 KV 注意力" if self.log_kv_dense_mode else "logKV 压缩注意力"
        )
        if is_master: print(f"🔧 正在初始化 Transformer ({mode_name})...")
        self.model = GPT(self.config).to(device).bfloat16()
        if log_kv_alpha_exact_tokens:
            self.model.set_alpha_span_boundary_ids(
                self.tokenizer.alpha_span_boundary_ids,
                self.tokenizer.alpha_line_end_ids if self.log_kv_gamma_cue_tokens else None,
                self.tokenizer.gamma_literal_texts
                if self.log_kv_gamma_literals or self.log_kv_gamma_surprisal in WORD_SCORES else None,
            )
        # Query-aware cue: host table of line-ending tokens, built once; each request's
        # question is cut from the last CUE_QUERY_TOKENS prompt ids only.
        self._query_line_ends = self._query_line_end_table() if self.log_kv_gamma_query_cue else None
        # niahLogKV: frozen learned retention heads, loaded apart from the LM weights.
        self._retention_heads = None
        if self.log_kv_retention_heads:
            from snapkv._vendor.niahlogkv.litgpt.retention import RetentionHeads
            self._retention_heads = RetentionHeads.load(self.log_kv_retention_heads, device=device)
            self.model.set_retention_heads(self._retention_heads)
            if is_master:
                print(f"🎯 retention heads {self.log_kv_retention_heads} (tag {self._retention_heads.tag()})")

        if is_master: print(f"🔄 正在加载权重...")
        checkpoint = _load_lit_model_checkpoint(checkpoint_dir, map_location=device)

        # 🌟 核心修复：检查是不是被包裹过的 checkpoint 字典
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            if is_master: print("📦 检测到 Fabric Checkpoint，已自动提取 model 权重。")
        else:
            state_dict = checkpoint

        # Same-checkpoint comparisons require every parameter to load.
        load_result = self.model.load_state_dict(state_dict, strict=True)

        if is_master:
            print(f"✅ 权重加载完毕！缺失的 keys: {len(load_result.missing_keys)} 个")
            # 如果 missing_keys 极其多（比如几百个），说明加载又失败了

        self.model.eval()

        # 推理时延指标收集
        self.gen_metrics: list[dict] = []      # generate_until
        self.ppl_metrics: list[dict] = []      # loglikelihood
        self.compression_metrics: list[dict] = []

        # Eval cache is built lazily once and reset per request (see
        # _set_eval_cache) — never re-allocated per sample.
        self._eval_cache_ready = False

    def _set_eval_cache(self) -> None:
        """Install (once) and reset the selected inference cache.

        LogKV buffer sizes depend only on (B, recent_size, max_levels) — not on
        the request length — so the cache is built ONCE at the model's full
        window and reset in place before each request. Rebuilding per request
        re-allocates O(n_layer x (recent + B*logN)) CUDA buffers thousands of
        times over a benchmark run and fragments the allocator (OOM risk).
        ``max_seq_length`` only bounds the append-only token counter and level
        hierarchy, so sizing it at the model's window covers every request.
        Dense mode mirrors this reuse policy with the standard O(N) KV cache.
        """
        if self._eval_cache_ready:
            self._reset_eval_cache()
            return
        dtype = next(self.model.parameters()).dtype
        max_seq_length = self.model.max_seq_length
        if self.log_kv_dense_mode or self.swa_window_size:
            cache_options = {"fused_dense_attention": True} if getattr(self, "_dense_256k_eval", False) else {}
            self.model.set_kv_cache(
                batch_size=1,
                max_seq_length=max_seq_length,
                device=self._device,
                dtype=dtype,
                **cache_options,
            )
            self._eval_cache_ready = True
            self._report_cache_budget()
            return
        self.model.set_log_kv_cache(
            batch_size=1,
            max_seq_length=max_seq_length,
            device=self._device,
            dtype=dtype,
            B=self.log_kv_B,
            recent_size=max(2, min(self.log_kv_recent_size, max_seq_length)),
            prefill_block=self.log_kv_prefill_block,
            second_order_scale=self.log_kv_second_order_scale,
            importance_pooling=self.log_kv_importance_pooling,
            importance_pooling_lambda=self.log_kv_importance_pooling_lambda,
            importance_pooling_temperature=self.log_kv_importance_pooling_temperature,
            semantic_clusters=self.log_kv_semantic_clusters,
            cluster_k_max=self.log_kv_cluster_k_max,
            cluster_lambda_rel=self.log_kv_cluster_lambda_rel,
            seg_eta=self.log_kv_seg_eta,
            seg_g0=self.log_kv_seg_g0,
            seg_gap_max=self.log_kv_seg_gap_max,
            seg_block_level=self.log_kv_seg_block_level,
            seg_forget=self.log_kv_seg_forget,
            semantic_s_h=self.log_kv_semantic_s_h,
            semantic_flush_granularity=self.log_kv_semantic_flush_granularity,
            semantic_cluster_chunk_size=self.log_kv_semantic_cluster_chunk_size,
            semantic_capacity_beta=self.log_kv_semantic_capacity_beta,
            semantic_capacity_hard_cap_mult=self.log_kv_semantic_capacity_hard_cap_mult,
            semantic_legacy_route=self.log_kv_semantic_legacy_route,
            semantic_unified_route=self.log_kv_semantic_unified_route,
            semantic_merge_passes=self.log_kv_semantic_merge_passes,
            semantic_anchor_mode=self.log_kv_semantic_anchor_mode,
            semantic_pack_backend=self.log_kv_semantic_pack_backend,
            semantic_centroid_backend=self.log_kv_semantic_centroid_backend,
            semantic_replay_updates=self.log_kv_semantic_replay_updates,
            alpha_exact_tokens=self.log_kv_alpha_exact_tokens,
            alpha_span_max_tokens=self.log_kv_alpha_span_max_tokens,
            beta_novelty=self.log_kv_beta_novelty,
            beta_adaptive_merge=self.log_kv_beta_adaptive_merge,
            gamma_level0_reinsert=self.log_kv_gamma_level0_reinsert,
            gamma_top_merge=self.log_kv_gamma_top_merge,
            gamma_level_slack=self.log_kv_gamma_level_slack,
            gamma_cue_tokens=self.log_kv_gamma_cue_tokens,
            gamma_literals=self.log_kv_gamma_literals,
            gamma_cue_guard=self.log_kv_gamma_cue_guard,
            gamma_query_cue=self.log_kv_gamma_query_cue,
            gamma_query_pool_tokens=self.log_kv_gamma_query_pool_tokens,
            retention_head=getattr(self, "_retention_heads", None) is not None,
            gamma_surprisal=self.log_kv_gamma_surprisal,
        )
        if self.swa_window_size == 0:
            # Match allocated persistent cache buffers, including metadata and
            # spare ladder/recent capacity. RoPE tables are common model state.
            reference_bytes = self._cache_storage_bytes()
            first = self.model.transformer.h[0].attn.kv_cache
            bytes_per_position = self.config.n_layer * (
                self.config.n_query_groups * (first.slot_k_dim + self.config.head_size)
                * torch.empty((), dtype=dtype).element_size() + 8
            )
            self.swa_window_size = min(max_seq_length, reference_bytes // bytes_per_position)
            if self.swa_window_size < 1:
                raise ValueError("LogKV budget cannot fit one SWA position")
            self.cache_budget["reference_logkv_bytes"] = reference_bytes
            self.model.clear_kv_cache()
            self.config.sliding_window_size = self.swa_window_size
            self.config.sliding_window_indices = [1] * self.config.n_layer
            for block in self.model.transformer.h:
                block.attn.apply_sliding_window_attention = True
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_seq_length, device=self._device, dtype=dtype)
        self._eval_cache_ready = True
        self._report_cache_budget()

    def _cache_storage_bytes(self) -> int:
        seen = {}
        for block in self.model.transformer.h:
            for name, tensor in block.attn.kv_cache.named_buffers():
                if name in ("cos_cache", "sin_cache"):
                    continue
                storage = tensor.untyped_storage()
                seen[storage.data_ptr()] = storage.nbytes()
        return sum(seen.values())

    def _report_cache_budget(self) -> None:
        self.cache_budget.update(persistent_cache_bytes=self._cache_storage_bytes(),
                                 swa_window_size=self.swa_window_size)
        # Qwen3 full attention stores one key and one value per KV group, token and layer.
        dense_bytes = (self.model.max_seq_length * self.config.n_layer * self.config.n_query_groups
                       * 2 * self.config.head_size * next(self.model.parameters()).element_size())
        self.cache_budget.update(dense_cache_bytes=dense_bytes,
                                 allocated_fraction_of_dense=self.cache_budget["persistent_cache_bytes"] / dense_bytes)
        first = self.model.transformer.h[0].attn.kv_cache
        if getattr(first, "alpha_exact_tokens", 0):
            self.cache_budget.update(alpha_exact_tokens=first.alpha_exact_tokens, effective_B=first.B,
                                     alpha_span_max_tokens=first.alpha_span_max_tokens,
                                     beta_novelty=first.beta_novelty, beta_adaptive_merge=first.beta_adaptive_merge)
        if getattr(first, "semantic_clusters", False):
            self.cache_budget.update(effective_B=first.B, levels=first.L_alloc,
                                     gamma_level0_reinsert=first.gamma_level0_reinsert,
                                     gamma_top_merge=first.gamma_top_merge, gamma_level_slack=first.gamma_level_slack,
                                     gamma_cue_tokens=getattr(first, "gamma_cue_tokens", 0),
                                     gamma_literals=getattr(first, "gamma_literals", False),
                                     gamma_cue_guard=getattr(first, "gamma_cue_guard", False),
                                     gamma_query_cue=getattr(first, "gamma_query_cue", False))
        if _is_main():
            print(f"🧮 persistent cache budget (excluding shared RoPE / transient workspace): {self.cache_budget}")

    def _reset_eval_cache(self) -> None:
        if self.log_kv_dense_mode or self.swa_window_size is not None:
            self.model.reset_kv_cache()
        else:
            self.model.reset_log_kv_cache()

    # ==========================================
    # 🌟 分布式结果收集
    # ==========================================
    def _warn_truncation(self, kind: str, kept: int) -> None:
        count = self._truncation_counts.get(kind, 0) + 1
        self._truncation_counts[kind] = count
        if count == 1 or _eval_verbose():
            print(f"⚠️ [{_rank_label()}] {kind}: 触发截断，保留 context/prompt 长度 {kept}；"
                  "本轮截断总数将在汇总时报告", flush=True)

    def all_gather_results(self, local_result_list: list, tag: str = ""):
        for kind, count in self._truncation_counts.items():
            print(f"⚠️ [{_rank_label()}] {kind}: 本 rank 本轮共截断 {count} 次", flush=True)
        self._truncation_counts.clear()
        if torch.cuda.is_available():  # how close this request type came to the card's limit (long contexts)
            peak = torch.cuda.max_memory_allocated() / 2**30
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30
            _eval_print(f"[{_rank_label()}] {tag}: 显存峰值 {peak:.1f} / {total:.1f} GiB", flush=True)
            torch.cuda.reset_peak_memory_stats()
        if not _dist_ready():
            return local_result_list

        # 结果只有 float/bool/str，不必经 GPU/NCCL 序列化与通信。
        # 汇合等待长上下文的负载差异；等待期间的心跳与超时见 _wait_all_ranks。
        _eval_print(f"⏳ [{_rank_label()}] {tag} 本 rank {len(local_result_list)} 条已完成，等待其它 rank", flush=True)
        wait_s = _wait_all_ranks(f"{tag} 汇总")
        _eval_print(
            f"🤝 [{_rank_label()}] {tag} 全部 rank 已就绪，等待用时 {wait_s:.1f}s；开始 Gloo 结果汇总",
            flush=True,
        )

        dp_size = _world_size()
        all_results_list = [None for _ in range(dp_size)]
        t0 = time.perf_counter()
        dist.all_gather_object(all_results_list, local_result_list, group=self.process_group)

        final_results = []
        max_load = max(len(r) for r in all_results_list)
        for i in range(max_load):
            for rank_id in range(dp_size):
                if i < len(all_results_list[rank_id]):
                    final_results.append(all_results_list[rank_id][i])
        _eval_print(
            f"✅ [{_rank_label()}] {tag} Gloo 结果汇总完成，共 {len(final_results)} 条，"
            f"用时 {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return final_results

    # ==========================================
    # 🌟 核心 1：PPL 与 选择题评测
    # ==========================================
    def _prefill_chunk(self) -> int:
        """Tokens per forward when a long input enters the LogKV cache (LOGKV_EVAL_PREFILL_CHUNK, default 32768;
        0 = one forward). The cache streams, so chunked input gives bit-identical logits (tests/test_eval_compression.py),
        and the activations stay bounded. Dense 256K uses bounded chunks with fused lower-right causal attention."""
        if getattr(self, "log_kv_dense_mode", False):
            return 2048 if getattr(self, "_dense_256k_eval", False) else 0
        return int(os.environ.get("LOGKV_EVAL_PREFILL_CHUNK", "32768"))

    def _forward(self, tokens, start, lm_head_start):
        """``self.model`` on ``tokens`` at positions start..start+T-1, logits from ``lm_head_start`` on, in chunks
        of ``_prefill_chunk()`` tokens when the input is longer."""
        T, chunk = int(tokens.numel()), self._prefill_chunk()
        if not chunk or T <= chunk:
            return self.model(tokens[None], input_pos=torch.arange(start, start + T, device=self._device),
                              input_pos_maxp1=start + T, lm_head_start=lm_head_start)
        kept = []
        for a in range(0, T, chunk):
            b = min(a + chunk, T)
            first = min(max(lm_head_start - a, 0), b - a - 1)  # a chunk before the scored span keeps one row
            out = self.model(tokens[None, a:b], input_pos=torch.arange(start + a, start + b, device=self._device),
                             input_pos_maxp1=start + b, lm_head_start=first)
            if b > lm_head_start:
                kept.append(out[:, max(lm_head_start - a, 0) - first:])
        return torch.cat(kept, dim=1)

    def _generate_chunked(self, prompt, max_returned_tokens, temperature, top_k, top_p, *, timing=None):
        """litgpt's generate for a prompt prefilled in chunks (_forward): the same tokens, the prompt included and
        the last token (eos or the length limit) not forwarded."""
        from snapkv._vendor.niahlogkv.litgpt.generate.base import sample

        if timing is not None:
            timing.clear()
            _synchronize_inference_device(self._device)
            generation_start = time.perf_counter()
        eos, out = self.tokenizer.eos_id, [prompt]
        logits, pos = self._forward(prompt, 0, prompt.numel() - 1), int(prompt.numel())
        for step in range(max_returned_tokens - pos):
            if step:
                logits = self.model(token.view(1, 1), input_pos=torch.tensor([pos], device=self._device),
                                    input_pos_maxp1=pos + 1, lm_head_start=0)
                pos += 1
            token = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p).to(dtype=torch.int64)
            out.append(token.view(-1))
            int_token = token.item()
            if timing is not None:
                elapsed = time.perf_counter() - generation_start
                if step == 0:
                    timing["prefill_time_s"] = elapsed
                timing.update(
                    total_time_s=elapsed,
                    decode_time_s=elapsed - timing["prefill_time_s"],
                    generated_tokens=step + 1,
                    decode_tokens=step,
                )
            if eos is not None and int_token == eos:
                break
        return torch.cat(out)

    def _score_tokens(
        self, ctx_enc: list[int], cont_enc: list[int], *, request_type: str = "loglikelihood"
    ) -> tuple[float, bool]:
        """Forward (context + continuation) and return
        ``(sum log p(continuation | context), is_greedy)``.

        Left-truncates so the sequence fits ``max_seq_length``. If the
        continuation alone meets/exceeds the window, the context is dropped and
        the continuation is left-truncated to its last ``max_len - 1`` tokens —
        scoring is then partial but the forward stays in bounds.
        """
        # 🌟 安全阀：如果 题干 + 选项 > max_seq_length，必须切掉题干最前面的部分
        max_len = self.model.max_seq_length
        if len(ctx_enc) + len(cont_enc) > max_len:
            keep_ctx_len = max_len - len(cont_enc)
            if keep_ctx_len <= 0:
                cont_enc = cont_enc[-(max_len - 1):]
                ctx_enc = []
            else:
                ctx_enc = ctx_enc[-keep_ctx_len:]
            self._warn_truncation(request_type, len(ctx_enc))

        if len(ctx_enc) == 0:
            ctx_enc = [self.tokenizer.bos_id]

        inps = torch.tensor([ctx_enc + cont_enc], dtype=torch.long, device=self._device)
        seq_len = inps.size(1)
        ctx_len = len(ctx_enc)

        with torch.no_grad():
            # Score through whichever eval cache was selected. In LogKV mode,
            # ``input_pos`` must be append-only contiguous because slot
            # compaction is order-based rather than indexed; dense KV mode uses
            # the same positions for the native full cache.
            self._set_eval_cache()
            try:
                _synchronize_inference_device(self._device)
                _reset_inference_peak_memory(self._device)
                if getattr(self, "_dense_256k_eval", False):
                    score, is_greedy, forward_time = self._score_dense_256k(inps[0], ctx_len)
                else:
                    t0 = time.perf_counter()
                    input_pos = torch.arange(seq_len, device=self._device, dtype=torch.int64)
                    # Only the scoring span needs logits.
                    logits = self._forward(inps[0], 0, ctx_len - 1) if self._prefill_chunk() \
                        else self.model(inps, input_pos=input_pos, lm_head_start=ctx_len - 1)
                    _synchronize_inference_device(self._device)
                    forward_time = time.perf_counter() - t0
                peak_memory = _inference_peak_memory(self._device)
                self.compression_metrics.append({
                    "request_type": request_type, "rank": _global_rank(),
                    "context_len": ctx_len, "cont_len": len(cont_enc),
                    **cache_compression_snapshot(self.model, seq_len),
                })
            finally:
                # In-place state reset (defensive: the next request resets again
                # via _set_eval_cache). Keeping the buffers avoids re-allocation.
                self._reset_eval_cache()

        self.ppl_metrics.append({
            "rank": _global_rank(), "request_type": request_type,
            "total_seq_len": seq_len, "context_len": ctx_len,
            "cont_len": len(cont_enc),
            "forward_time_s": forward_time,
            "forward_time_ms": forward_time * 1000,
            "forward_tokens_per_sec": seq_len / forward_time if forward_time > 0 else None,
            **peak_memory,
        })

        if getattr(self, "_dense_256k_eval", False):
            return score, is_greedy

        # ``logits`` starts at position ctx_len-1 (lm_head_start); its first
        # len(cont_enc) rows are the positions [ctx_len-1, seq_len-2] that
        # predict the continuation tokens.
        cont_logits = logits[0, : len(cont_enc)]
        cont_targets = torch.tensor(cont_enc, dtype=torch.long, device=self._device)

        return _score_continuation_logits(cont_logits, cont_targets)

    def _score_dense_256k(self, tokens, ctx_len):
        """Consume each chunk's logits before prefill advances to the next chunk."""
        total_logprob, forward_time = 0.0, 0.0
        is_greedy = True
        length, chunk = tokens.numel(), self._prefill_chunk()
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            first = min(end, max(start, ctx_len - 1))
            _synchronize_inference_device(self._device)
            t0 = time.perf_counter()
            logits = self._forward(tokens[start:end], start, first - start)
            _synchronize_inference_device(self._device)
            forward_time += time.perf_counter() - t0
            # Position i predicts i+1, so the final input position is never scored.
            stop = min(end, length - 1)
            if first < stop:
                score, greedy = _score_continuation_logits(logits[0, :stop - first], tokens[first + 1:stop + 1])
                total_logprob += score
                is_greedy &= greedy
            del logits
        return total_logprob, is_greedy, forward_time

    def loglikelihood(self, requests):
        dp_rank = _global_rank()
        dp_size = _world_size()

        local_requests = requests[dp_rank::dp_size]
        results = []

        for req in _eval_progress(local_requests, "loglikelihood"):
            context, continuation = req.args[0], req.args[1]

            ctx_enc = self.tokenizer.encode(context).tolist()
            # Llama 3 等 use_bos=True：若对 continuation 再 encode 一次会多一个 BOS，拼接后破坏
            # loglikelihood 对齐；Qwen 通常无 BOS，bos=False 与默认行为一致。
            cont_enc = self.tokenizer.encode(continuation, bos=False).tolist()

            results.append(self._score_tokens(ctx_enc, cont_enc))

        # 清理 CUDA 缓存，避免 all_gather 时 OOM
        torch.cuda.empty_cache()
        return self.all_gather_results(results, tag="loglikelihood")

    # ==========================================
    # 🌟 核心 2：自回归生成任务 (LongBench)
    # ==========================================
    def _query_line_end_table(self):
        """Line-ending tokens over every id the model embeds.

        ``tokenizer.vocab_size`` leaves out added tokens, and the end-of-text
        token (a line end) is one of them.
        """
        ends = list(self.tokenizer.alpha_line_end_ids)
        table = np.zeros(max(self.config.padded_vocab_size, max(ends, default=-1) + 1), dtype=bool)
        table[ends] = True
        return table

    def _set_gamma_query(self, prompt_tensor, prompt_text=None) -> None:
        """Detect visible task spans, or use the legacy closing-question cue when detection is off."""
        from snapkv._vendor.niahlogkv.litgpt.alpha_log_kv import CUE_QUERY_TOKENS, query_region

        if getattr(self, "log_kv_gamma_query_detector", None):
            from snapkv._vendor.niahlogkv.litgpt.query_detector import QueryDetector, cue_ids
            if self._query_detector is None:
                options = dict(self._query_detector_options)
                if options.get("device") == "model":
                    options["device"] = str(self._device)
                self._query_detector = QueryDetector(self.log_kv_gamma_query_detector, **options)
            text = self.tokenizer.decode(prompt_tensor) if prompt_text is None else prompt_text
            spans = self._query_detector.detect(text)
            self.model.set_gamma_queries([cue_ids(text, spans, self.tokenizer)])
            return
        tail = prompt_tensor[-CUE_QUERY_TOKENS:].tolist()
        self.model.set_gamma_queries([query_region(tail, self._query_line_ends)])

    def _generate_turns(self, prompt, separator, max_new_tokens, sample_id):
        """Questions asked one after another on one cache (SCBench's multi-turn mode).

        ``prompt`` is the shared context, then each question, joined by ``separator``. The context is
        prefilled before any question is known (no query cue: a compressed cache must serve whatever is
        asked later); each question and each greedy answer are appended to the same cache, and an answer
        ends at a line end (or the end of text) or after ``max_new_tokens``. A context too long for the
        window loses its start. Returns the answers joined by ``separator``.
        """
        context, *questions = prompt.split(separator)
        stops = set(self.tokenizer.alpha_line_end_ids)
        asked = [self.tokenizer.encode(q, device=self._device, bos=False) for q in questions]
        pending = self.tokenizer.encode(context, device=self._device)
        room = self.model.max_seq_length - sum(q.numel() for q in asked) - len(asked) * max_new_tokens
        context_was_truncated = pending.numel() > room
        if pending.numel() > room:
            pending = pending[-max(room, 1):]
            self._warn_truncation("generate_until", pending.numel())
        context_len, pos, answers = int(pending.numel()), 0, []
        with torch.no_grad():
            self._set_eval_cache()
            if (getattr(self, "_query_line_ends", None) is not None
                    and getattr(self.model, "log_kv_cue_stream", None) is not None):
                if getattr(self, "log_kv_gamma_query_detector", None):
                    # Only the context visible at cache creation; never inspect future turns.
                    self._set_gamma_query(pending, None if context_was_truncated else context)
                else:
                    self.model.set_gamma_queries([None])
            try:
                for turn_index, question in enumerate(asked, start=1):
                    # The previous answer's last token has not been forwarded yet: it leads this chunk.
                    pending = torch.cat([pending, question])
                    prompt_len = int(pending.numel())
                    answer = []
                    _synchronize_inference_device(self._device)
                    _reset_inference_peak_memory(self._device)
                    generation_start = time.perf_counter()
                    prefill_time_s = total_time_s = 0.0
                    for step in range(max_new_tokens):
                        T = pending.numel()
                        logits = self._forward(pending, pos, T - 1)
                        pos += T
                        token = int(logits[0, -1].argmax())
                        # Reading the sampled token synchronizes the device, as in _generate_chunked.
                        total_time_s = time.perf_counter() - generation_start
                        if step == 0:
                            prefill_time_s = total_time_s
                        answer.append(token)
                        pending = torch.tensor([token], device=self._device, dtype=torch.long)
                        if token in stops:
                            break
                    _synchronize_inference_device(self._device)
                    peak_memory = _inference_peak_memory(self._device)
                    gen_len = len(answer)
                    decode_tokens = max(gen_len - 1, 0)
                    decode_time_s = total_time_s - prefill_time_s
                    # Each answer has its own TTFT/decode phase, while the cache stays shared across turns.
                    self.gen_metrics.append({
                        "request_type": "generate_turns", "rank": _global_rank(), "sample_id": sample_id,
                        "turn_index": turn_index, "turns": len(asked), "context_len": context_len,
                        "processed_tokens": pos, "prompt_len": prompt_len, "generated_tokens": gen_len,
                        "total_time_s": total_time_s, "prefill_time_s": prefill_time_s,
                        "ttft_ms": prefill_time_s * 1000,
                        "prefill_tokens_per_sec": prompt_len / prefill_time_s if prefill_time_s > 0 else None,
                        "decode_time_s": decode_time_s, "decode_tokens": decode_tokens,
                        "decode_ms_per_token": decode_time_s / decode_tokens * 1000 if decode_tokens else None,
                        "decode_tokens_per_sec": decode_tokens / decode_time_s if decode_tokens and decode_time_s > 0 else None,
                        "gen_tokens_per_sec": gen_len / total_time_s if total_time_s > 0 else None,
                        **peak_memory,
                    })
                    answers.append(self.tokenizer.decode(torch.tensor(answer, dtype=torch.long)))
                self.compression_metrics.append({
                    "request_type": "generate_turns", "rank": _global_rank(), "sample_id": sample_id,
                    "prompt_len": context_len, "turns": len(asked), **cache_compression_snapshot(self.model, pos),
                })
            finally:
                self._reset_eval_cache()
        return separator.join(answers)

    def generate_until(self, requests):
        dp_rank = _global_rank()
        dp_size = _world_size()

        local_requests = requests[dp_rank::dp_size]
        results = []

        for local_i, req in enumerate(_eval_progress(local_requests, "generate_until")):
            global_request_index = local_i * dp_size + dp_rank
            sample_id = _request_sample_id(req, global_request_index=global_request_index, rank=dp_rank)
            prompt = req.args[0]
            gen_args = req.args[1]

            # 解析 lm-eval 的 gen_kwargs：`until` 是停词（字符串列表），长度上限用 `max_gen_toks`（见 lm-eval model_guide）
            max_new_tokens = int(gen_args.get("max_gen_toks", gen_args.get("max_length", self.max_gen_toks)))
            do_sample = bool(gen_args.get("do_sample", False))
            temperature = float(gen_args.get("temperature", 1.0))
            top_p = float(gen_args.get("top_p", 1.0))
            top_k = gen_args.get("top_k", None)
            if not do_sample:
                # litgpt.generate.sample 仅在 temperature<=0 且 top_p<=0 时走 argmax；仅设 temperature=0 而 top_p=1 仍会多项式采样
                temperature = 0.0
                top_p = 0.0
            if gen_args.get("turn_separator"):  # several questions on one cache (eval_tasks/niah_turns)
                results.append(self._generate_turns(prompt, gen_args["turn_separator"], max_new_tokens, sample_id))
                continue

            prompt_tensor = self.tokenizer.encode(prompt, device=self._device)
            original_prompt_tokens = int(prompt_tensor.size(0))
            prompt_token_offset = 0

            # 🌟 安全阀：为生成的新 Token 预留空间
            max_len = self.model.max_seq_length
            if prompt_tensor.size(0) + max_new_tokens > max_len:
                keep_prompt_len = max_len - max_new_tokens
                prompt_token_offset = int(prompt_tensor.size(0) - keep_prompt_len)
                prompt_tensor = prompt_tensor[-keep_prompt_len:]
                self._warn_truncation("generate_until", keep_prompt_len)

            total_max_len = prompt_tensor.size(0) + max_new_tokens
            prompt_len = prompt_tensor.size(0)

            with torch.no_grad():
                # 🧩 长上下文生成使用所选 cache（LogKV slot cache 或 dense KV）。
                # 建一次、按请求原地重置，不逐样本重分配 — 见 _set_eval_cache。
                self._set_eval_cache()
                if (getattr(self, "_query_line_ends", None) is not None
                        and getattr(self.model, "log_kv_cue_stream", None) is not None):
                    self._set_gamma_query(prompt_tensor, prompt if prompt_token_offset == 0 else None)
                try:
                    _synchronize_inference_device(self._device)
                    _reset_inference_peak_memory(self._device)
                    timing = {}
                    if prompt_len > (self._prefill_chunk() or prompt_len):  # a long prompt: prefill in chunks
                        out = self._generate_chunked(
                            prompt_tensor, total_max_len, temperature, top_k, top_p, timing=timing,
                        )
                    else:
                        out = litgpt_generate(
                            self.model,
                            prompt_tensor,
                            max_returned_tokens=total_max_len,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            eos_id=self.tokenizer.eos_id,
                            timing=timing,
                        )
                    _synchronize_inference_device(self._device)
                    peak_memory = _inference_peak_memory(self._device)
                    # The last output token has not been forwarded into the cache.
                    self.compression_metrics.append({
                        "request_type": "generate_until", "rank": dp_rank, "sample_id": sample_id,
                        "prompt_len": prompt_len, "generated_tokens": int(out.numel()) - prompt_len,
                        **cache_compression_snapshot(self.model, int(out.numel()) - 1),
                    })
                finally:
                    self._reset_eval_cache()

            # 截取新生成的部分并解码
            generated_tokens = out[prompt_tensor.size(0):]
            gen_len = generated_tokens.size(0)
            total_time_s = timing["total_time_s"]
            prefill_time_s = timing["prefill_time_s"]
            decode_time_s = timing["decode_time_s"]
            decode_tokens = timing["decode_tokens"]
            self.gen_metrics.append({
                "rank": dp_rank, "sample_id": sample_id,
                "prompt_len": prompt_len, "generated_tokens": gen_len,
                "original_prompt_len": original_prompt_tokens, "truncated_prompt_tokens": prompt_token_offset,
                "total_time_s": total_time_s,
                "prefill_time_s": prefill_time_s,
                "ttft_ms": prefill_time_s * 1000,
                "prefill_tokens_per_sec": prompt_len / prefill_time_s if prefill_time_s > 0 else None,
                "decode_time_s": decode_time_s,
                "decode_tokens": decode_tokens,
                "decode_ms_per_token": decode_time_s / decode_tokens * 1000 if decode_tokens else None,
                "decode_tokens_per_sec": decode_tokens / decode_time_s if decode_tokens and decode_time_s > 0 else None,
                "gen_tokens_per_sec": gen_len / total_time_s if total_time_s > 0 else None,
                **peak_memory,
            })

            decoded = self.tokenizer.decode(generated_tokens)
            results.append(_before_stop(decoded, gen_args.get("until")))

        # 清理 CUDA 缓存，避免 all_gather 时 OOM
        torch.cuda.empty_cache()
        return self.all_gather_results(results, tag="generate_until")

    def loglikelihood_rolling(self, requests):
        """Rolling log-likelihood over full documents (lm-eval PPL tasks, e.g.
        wikitext). Returns one summed log-prob (scalar float) per request,
        matching lm-eval's reference implementations — returning tuples here
        breaks perplexity aggregation downstream.

        Long documents are scored in non-overlapping windows of max_seq_length:
        each window's tokens are conditioned on the window prefix (the first
        window starts from BOS). This matches lm-eval's standard rolling-window
        scoring; a ``pass`` stub would return None and break PPL tasks.
        """
        dp_rank = _global_rank()
        dp_size = _world_size()

        local_requests = requests[dp_rank::dp_size]
        results = []

        max_len = self.model.max_seq_length
        for req in _eval_progress(local_requests, "loglikelihood_rolling"):
            (text,) = req.args
            tokens = self.tokenizer.encode(text, bos=False).tolist()

            total_logprob = 0.0
            # Non-overlapping windows; window size max_len - 1 leaves room for
            # the 1-token context (BOS or the previous window's last token).
            window = max_len - 1
            for start in range(0, len(tokens), window):
                chunk = tokens[start : start + window]
                if start == 0:
                    ctx = [self.tokenizer.bos_id]
                else:
                    ctx = [tokens[start - 1]]
                logprob, _ = self._score_tokens(ctx, chunk, request_type="loglikelihood_rolling")
                total_logprob += logprob

            results.append(total_logprob)

        torch.cuda.empty_cache()
        return self.all_gather_results(results, tag="loglikelihood_rolling")

    @property
    def eot_token_id(self): return self.tokenizer.eos_id
    @property
    def max_length(self): return self.model.max_seq_length
    @property
    def max_gen_toks(self): return 256
    @property
    def batch_size(self): return 1
    @property
    def device(self): return self._device
    def tok_encode(self, string): return self.tokenizer.encode(string).tolist()
    def tok_decode(self, tokens): return self.tokenizer.decode(torch.tensor(tokens))

def _resolve_checkpoint_dir(checkpoint_dir: str, group=None) -> str:
    """优先使用指定目录自身的 lit_model.pth；仅在它不存在时回退到最大的 step_* 保存点。

    glob + exists() 必须只由 rank 0 判定再广播：训练刚写完最后一个 step_* 时，
    其它节点的共享盘元数据缓存可能还看不到它，各 rank 各自 glob 就会选到**不同的
    checkpoint**（评测结果静默不可比），或者一部分 rank 找不到权重直接退出、把其余
    rank 留在集合通信里等到超时。
    """
    resolved: str | None = None
    if _is_main():
        base = Path(os.path.expandvars(os.path.expanduser(checkpoint_dir)))
        if (base / "lit_model.pth").exists():
            resolved = str(base)
            print(f"Checkpoint weights: {base / 'lit_model.pth'}", flush=True)
        else:
            step_dirs = [
                d for d in base.glob("step_*")
                if d.is_dir() and (d / "lit_model.pth").exists()
            ]
            if step_dirs:
                max_dir = max(step_dirs, key=lambda d: int(d.name.split("_")[1]))
                print(f"🔍 检测到分段 checkpoint，自动选择最新: {max_dir}")
                resolved = str(max_dir)
            else:
                resolved = checkpoint_dir

    return _broadcast_obj(resolved, group=group)


@auto_expand_env_vars
def main(
    checkpoint_dir: str = "checkpoints/Qwen/Qwen3-0.6B-Base",
    benchmark: str = "debug",
    config_overrides: dict[str, Any] | None = None,
    output_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    # ── 🧩 logKV / dense KV eval ──
    # true = 使用原生 O(N) 标准 KVCache 跑普通 causal dense attention；
    # false = 默认 LogKV 压缩注意力路径。dense 模式只用于手动基线对比。
    log_kv_dense_mode: bool = False,
    # None = LogKV; 0 = SWA matched to LogKV's allocated cache bytes; >0 = explicit window.
    swa_window_size: int | None = None,
    log_kv_B: int = 512,
    log_kv_recent_size: int = 1024,
    # prefill 分块大小：块内 query 共享块首冻结的 slot 状态。2 = 严格 2-token
    # 流式语义（用于 A/B 验证近似偏差）；越大越快，偏差上界 = 块内 query 比严格
    # 流式多看到 < block 个未压缩 token（缓存状态轨迹两者严格一致）。
    log_kv_prefill_block: int = 256,
    # Must match the CPT target scale, not the warm-up intermediate value.
    # 0.0 reproduces the first-order LogKV eval path; 1.0 enables full
    # Sigma/Gamma second-order corrections.
    log_kv_second_order_scale: float = 1.0,
    # 重要性加权池化：level 合并时按启发式每 token 重要性（post-RoPE key
    # L2 范数）加权，而非均匀 mean-pool；不影响 log(w) mass bias（token 计数
    # 独立跟踪）。是 k 的纯函数，无新增可学参数，训练/推理天然一致。
    log_kv_importance_pooling: bool = False,
    # 只在 log_kv_importance_pooling=True 时生效：把重要性份额往均匀/计数
    # 份额上混合，1.0（默认）= 纯重要性，0.0 数值上等价于均匀池化，中间值
    # 插值。2026-08-14 决定性实验发现纯重要性（lambda=1.0）在 niah 上比
    # vanilla 均匀池化倒退（虽然 ACC/LongBench 涨了），加这个参数是为了低成本
    # 扫一遍 lambda 找 niah 不倒退的甜点（见 CLAUDE.md 6.5）。
    log_kv_importance_pooling_lambda: float = 1.0,
    # 同样只在 log_kv_importance_pooling=True 时生效，跟 lambda 正交：对原始
    # 重要性启发式（key 范数）取指数再归一化，1.0（默认）=不变，<1.0 压缩
    # 极端值的动态范围（比如 attention-sink 式异常高范数 token），不像 lambda
    # 整体往均匀分布混合那样连带压掉中等显著性 token 的信号；两个参数可以同时
    # 设置（先 temperature 重塑，再 lambda 混合）。
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
    # Gamma (on by default): delayed Alpha evictions re-sort only ladder level 0 instead
    # of re-merging the whole cluster, and a saturated top level merges its lightest
    # adjacent pair instead of folding into slot 0. level_slack 2 keeps the classic
    # spare levels; fewer spends the same bytes on a wider B' (more live entries).
    log_kv_gamma_level0_reinsert: bool = True,
    log_kv_gamma_top_merge: str = "lightest",
    log_kv_gamma_level_slack: int = 2,
    # Gamma cues (on by default): exact-pool selection also values spans that repeat
    # rare tokens of the stream's opening region (up to this many tokens, cut at its
    # last line end); 0 keeps the K/V-only span scores.
    log_kv_gamma_cue_tokens: int = 64,
    # Gamma literals (off by default): exact-pool selection also values spans holding the
    # first occurrence of a random-looking literal (digit strings, codes).
    log_kv_gamma_literals: bool = False,
    # Gamma cue guard (off by default): cues need an opening region that ends at a line end,
    # and a cue counted more than 16 times gives no evidence. Only with cues on.
    log_kv_gamma_cue_guard: bool = False,
    # Gamma query-aware cue (off by default): the closing question of each generation prompt
    # (its last 64 tokens, after their first line end) is a cue too; literals then rank below cues.
    log_kv_gamma_query_cue: bool = False,
    log_kv_gamma_query_detector: str | None = None,
    log_kv_gamma_query_detector_options: dict | None = None,
    log_kv_gamma_query_pool_tokens: int = 0,
    # niahLogKV learned retention heads (train_retention.py output); None = off. They score
    # Alpha's exact spans query-agnostically and hold no KV.
    log_kv_retention_heads: str | None = None,
    # niahLogKV surprisal admission (litgpt.alpha_log_kv.SURPRISAL_SCORES; None = off): the exact pool
    # ranks spans by their tokens' surprisal under the model itself (prefill runs in recent-window
    # chunks); the word scores count a span by its most informative words.
    log_kv_gamma_surprisal: str | None = None,
    # ── 🧩 logKV：tokenizer 回退（checkpoint 目录缺 tokenizer 文件时用）──
    tokenizer_dir: str | None = None,
    # ── 只跑一小批样本（Phase 0 诊断用；见 log_kv_diag_mode）。int = 绝对条数，
    # float in (0, 1) = lm-eval 的抽样比例。None = 跑全量。
    limit: int | float | None = None,
    # ── 🧩 logKV 诊断（score/value oracle 归因网格；见 litgpt.log_kv_diag）──
    # None/"off" = 不诊断（默认，零开销）。其余取值：
    # baseline/s_oracle/v_oracle/gamma_only/exact/dense —— 见 log_kv_diag 模块文档。
    log_kv_diag_mode: str | None = None,
    # 诊断汇总 JSON 的落盘目录；缺省时退回 output_path。
    log_kv_diag_output: str | None = None,
    # ── 分层消融（D1 悬而未决的 marginal vs cumulative 问题；见 log_kv_diag）──
    # 层号 >= 此值强制走 exact，其余层走 log_kv_diag_mode（通常是 baseline），
    # 在同一次真实前向里测。None = 不启用（默认）。典型 sweep：28 - N，
    # N ∈ {0,1,2,4,7,14,21,28}（28 为本模型层数，按实际改）。
    log_kv_diag_exact_from_layer: int | None = None,
    # ── D5 重分箱：peakiness 只统计序列最后这么多个 token 的 query ──
    # （question/生成阶段紧邻的 prefill 尾部），None = 不过滤（默认，旧行为）。
    log_kv_diag_peak_window_from_end: int | None = None,
    # ── width 门控消融（D1 显示 rank-1 统计量在 slot_width>=4 起明显失真；
    # 见 log_kv_diag）── slot_width > 此值的槽位强制走 1 阶路径，不管
    # second_order_scale；None = 不启用（默认）。不改变 baseline 实际输出/
    # 传播的隐状态,只在 by_layer_output 里新增一个 err_width_gated 语料,
    # 和 err_baseline/err_baseline_1st_order 在同一次前向、同一组隐状态上
    # 直接可比。典型 sweep：{2, 4, 8, 16, None}。
    log_kv_diag_second_order_max_width: int | None = None,
    # ── layer 门控消融（同上；两者是 AND 关系，可单独用也可联合用）── 层号 >
    # 此值的层强制走 1 阶路径，不管 second_order_scale；None = 不启用（默认）。
    # 典型 sweep：{7, 14, 21, 27, None}（配合已有的 exact_from_layer 分层消融
    # 结果来选阈值）。
    log_kv_diag_second_order_max_layer: int | None = None,
    # ── 🧩 logKV：YAML config ──
    config: str | None = None,
    # Base architecture for legacy checkpoints without model_config.yaml; never inferred from experiment tags.
    arch_name: str | None = None,
):
    # ── 🧩 logKV：加载 YAML config（非 null 值覆盖 CLI，同 demo.py 约定）──
    # NOTE: `locals()[k] = v` does NOT write back to a function's real locals in
    # CPython, so YAML overrides must be applied by explicit re-binding.
    _yaml: dict = {}
    if config is not None:
        # expand_env_vars：bash 风格 ${VAR} / ${VAR-default} 展开，与 demo.py 的
        # YAML 装载、majob.sh 的 bash 展开共用同一语义（详见 utils.expand_env_vars）。
        _yaml = _coerce_yaml_sci_floats(
            expand_env_vars(_load_yaml_config(os.path.join(os.getcwd(), config), checkpoint_dir) or {})
        )
        _banned = sorted(_k for _k in _yaml if _k.startswith("eval_log_kv_"))
        if _banned:
            raise ValueError(
                "eval_log_kv_* YAML keys are no longer supported; use log_kv_* for the single eval LogKV config. "
                f"Remove: {', '.join(_banned)}"
            )
        _valid = set(inspect.signature(main).parameters)
        for _k in _yaml:
            if _k != "config" and _k not in _valid and (_is_main() or _eval_verbose()):
                print(f"[eval] WARNING: unknown YAML key ignored: {_k}")

    def _o(name, current):
        v = _yaml.get(name)
        return v if v is not None else current

    checkpoint_dir = _o("checkpoint_dir", checkpoint_dir)
    arch_name = _o("arch_name", arch_name)
    benchmark = _o("benchmark", benchmark)
    config_overrides = _o("config_overrides", config_overrides)
    output_path = _o("output_path", output_path)
    metadata = _o("metadata", metadata)
    log_kv_dense_mode = bool(_o("log_kv_dense_mode", log_kv_dense_mode))
    swa_window_size = _o("swa_window_size", swa_window_size)
    log_kv_B = _o("log_kv_B", log_kv_B)
    log_kv_recent_size = _o("log_kv_recent_size", log_kv_recent_size)
    log_kv_prefill_block = _o("log_kv_prefill_block", log_kv_prefill_block)
    log_kv_second_order_scale = float(_o("log_kv_second_order_scale", log_kv_second_order_scale))
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
    log_kv_gamma_level0_reinsert = bool(_o("log_kv_gamma_level0_reinsert", log_kv_gamma_level0_reinsert))
    log_kv_gamma_top_merge = str(_o("log_kv_gamma_top_merge", log_kv_gamma_top_merge))
    log_kv_gamma_level_slack = int(_o("log_kv_gamma_level_slack", log_kv_gamma_level_slack))
    log_kv_gamma_cue_tokens = int(_o("log_kv_gamma_cue_tokens", log_kv_gamma_cue_tokens))
    log_kv_gamma_literals = bool(_o("log_kv_gamma_literals", log_kv_gamma_literals))
    log_kv_gamma_cue_guard = bool(_o("log_kv_gamma_cue_guard", log_kv_gamma_cue_guard))
    log_kv_gamma_query_cue = bool(_o("log_kv_gamma_query_cue", log_kv_gamma_query_cue))
    log_kv_gamma_query_detector = _o("log_kv_gamma_query_detector", log_kv_gamma_query_detector) or None
    log_kv_gamma_query_detector_options = _o("log_kv_gamma_query_detector_options", log_kv_gamma_query_detector_options)
    log_kv_gamma_query_pool_tokens = int(_o("log_kv_gamma_query_pool_tokens", log_kv_gamma_query_pool_tokens))
    log_kv_retention_heads = _o("log_kv_retention_heads", log_kv_retention_heads) or None
    log_kv_gamma_surprisal = _o("log_kv_gamma_surprisal", log_kv_gamma_surprisal) or None
    if not log_kv_semantic_clusters:
        log_kv_alpha_exact_tokens = 0  # exact spans live on semantic clusters only
    if not log_kv_alpha_exact_tokens:
        # Beta and Gamma cues and literals score the Alpha exact pool; disabling Alpha disables them.
        log_kv_beta_novelty = log_kv_beta_adaptive_merge = False
        log_kv_gamma_cue_tokens = 0
        log_kv_gamma_literals = False
        log_kv_gamma_query_cue = False
        log_kv_gamma_query_detector = None
        log_kv_gamma_query_pool_tokens = 0
        log_kv_retention_heads = None
        log_kv_gamma_surprisal = None
    if not (log_kv_gamma_cue_tokens or log_kv_gamma_query_cue):
        log_kv_gamma_cue_guard = False  # the guard only acts on cues
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
    tokenizer_dir = _o("tokenizer_dir", tokenizer_dir)
    limit = _o("limit", limit)
    log_kv_diag_mode = _o("log_kv_diag_mode", log_kv_diag_mode)
    log_kv_diag_output = _o("log_kv_diag_output", log_kv_diag_output)
    log_kv_diag_exact_from_layer = _o("log_kv_diag_exact_from_layer", log_kv_diag_exact_from_layer)
    log_kv_diag_peak_window_from_end = _o(
        "log_kv_diag_peak_window_from_end", log_kv_diag_peak_window_from_end
    )
    log_kv_diag_second_order_max_width = _o(
        "log_kv_diag_second_order_max_width", log_kv_diag_second_order_max_width
    )
    log_kv_diag_second_order_max_layer = _o(
        "log_kv_diag_second_order_max_layer", log_kv_diag_second_order_max_layer
    )

    # LongBench v2 cannot be dropped from a requested evaluation, even if an old launcher marks it optional.
    # Check its data before CUDA initialization and checkpoint/model loading.
    longbench2_tasks = [task for task in benchmark.split(",")
                       if task.strip() == "longbench2" or task.strip().startswith("longbench2_")]
    if longbench2_tasks and _is_main():
        from snapkv._vendor.niahlogkv.litgpt.longbench2_patch import plain_folder
        print(f"LongBench v2 evaluation data (ordinary files): {plain_folder()}", flush=True)
    _check_task_data(longbench2_tasks)

    local_rank = _local_rank()
    world_size = _world_size()
    if world_size > 1 or _expected_world_size() > 0:
        _rendezvous_snapshot("eval startup before init_process_group")
        _check_expected_world_size("eval startup before init_process_group")

    # 进程组可能由外部创建（demo.py 用 Fabric 起训练后直接 in-process 调 eval.main），
    # 这时**不能**由 eval 销毁它，否则训练侧后续的集合通信全炸。只有 eval 自己建的
    # 才由 eval 负责销毁。
    pg_owned_here = False
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    diag_active = log_kv_diag_mode not in (None, "off")
    if log_kv_dense_mode or swa_window_size is not None:
        if diag_active:
            raise ValueError(
                "dense/SWA mode is incompatible with log_kv_diag_mode: "
                "LogKV diagnostics require LogStructuredKVCache slots."
            )

    # 多节点时用全局 rank==0（每个节点都有一个 local_rank 0，用它会重复打印）
    if _is_main():
        print(f"🚀 启动魔改版评估管线 | 任务: {benchmark}")
        if swa_window_size is not None:
            print(f"🧩 SWA during prefill and decode | window: {swa_window_size} (0 = match LogKV bytes)")
        elif log_kv_dense_mode:
            print(
                "🧩 dense 标准 KV 注意力 | 使用 GPT.set_kv_cache() 原生 causal attention；"
                "忽略 log_kv_B/recent_size/prefill_block/second_order_scale 等 LogKV 参数"
            )
        else:
            print(
                f"🧩 logKV 压缩注意力 | B: {log_kv_B} | recent_size: {log_kv_recent_size} | "
                f"prefill_block: {log_kv_prefill_block} | "
                f"second_order_scale: {log_kv_second_order_scale} | "
                f"importance_pooling: {log_kv_importance_pooling} "
                f"(lambda={log_kv_importance_pooling_lambda}, "
                f"temperature={log_kv_importance_pooling_temperature}) | "
                f"semantic: {log_kv_semantic_clusters} "
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
                f"beta_novelty={log_kv_beta_novelty}, beta_adaptive_merge={log_kv_beta_adaptive_merge}, "
                f"gamma_level0_reinsert={log_kv_gamma_level0_reinsert}, gamma_top_merge={log_kv_gamma_top_merge}, "
                f"gamma_level_slack={log_kv_gamma_level_slack}, gamma_cue_tokens={log_kv_gamma_cue_tokens}, "
                f"gamma_literals={log_kv_gamma_literals}, gamma_cue_guard={log_kv_gamma_cue_guard}, "
                f"gamma_query_cue={log_kv_gamma_query_cue}, retention_heads={log_kv_retention_heads}, "
                f"gamma_surprisal={log_kv_gamma_surprisal})"
            )
        if diag_active:
            print(
                f"🔬 诊断模式: {log_kv_diag_mode} | limit: {limit} | "
                f"exact_from_layer: {log_kv_diag_exact_from_layer} | "
                f"peak_window_from_end: {log_kv_diag_peak_window_from_end} | "
                f"second_order_max_width: {log_kv_diag_second_order_max_width} | "
                f"second_order_max_layer: {log_kv_diag_second_order_max_layer} | "
                "每 rank 各自累积统计量，不跨 rank 聚合"
            )

    # 仅本 rank 正常完成时参加收尾同步；异常直接退出，不用 barrier 掩盖原始错误。
    eval_done = False
    eval_group = None
    eval_group_owned_here = False
    try:
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="gloo", timeout=timedelta(hours=12))
            pg_owned_here = True
        if world_size > 1:
            _rendezvous_snapshot("eval startup after init_process_group")
            _check_expected_world_size("eval startup after init_process_group")
            _hb("进程组就绪（rendezvous 完成）")
        if _dist_ready():
            if pg_owned_here:
                eval_group = dist.group.WORLD
            else:
                # 训练内调用时保留 Fabric 的默认组，只创建/销毁评测自己的 CPU 组。
                eval_group = dist.new_group(backend="gloo", timeout=timedelta(hours=12))
                eval_group_owned_here = True
            _hb("评测 Gloo 通信组就绪（CPU 对象通信，超时 12 小时）")
        checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir, group=eval_group)

        lm_model = LogKVLM(
            checkpoint_dir,
            device=device,
            config_overrides=config_overrides,
            arch_name=arch_name,
            log_kv_B=log_kv_B,
            log_kv_recent_size=log_kv_recent_size,
            log_kv_prefill_block=log_kv_prefill_block,
            log_kv_second_order_scale=log_kv_second_order_scale,
            log_kv_dense_mode=log_kv_dense_mode,
            swa_window_size=swa_window_size,
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
            log_kv_gamma_level0_reinsert=log_kv_gamma_level0_reinsert,
            log_kv_gamma_top_merge=log_kv_gamma_top_merge,
            log_kv_gamma_level_slack=log_kv_gamma_level_slack,
            log_kv_gamma_cue_tokens=log_kv_gamma_cue_tokens,
            log_kv_gamma_literals=log_kv_gamma_literals,
            log_kv_gamma_cue_guard=log_kv_gamma_cue_guard,
            log_kv_gamma_query_cue=log_kv_gamma_query_cue,
            log_kv_gamma_query_detector=log_kv_gamma_query_detector,
            log_kv_gamma_query_detector_options=log_kv_gamma_query_detector_options,
            log_kv_gamma_query_pool_tokens=log_kv_gamma_query_pool_tokens,
            log_kv_retention_heads=log_kv_retention_heads,
            log_kv_gamma_surprisal=log_kv_gamma_surprisal,
            tokenizer_dir=tokenizer_dir,
            process_group=eval_group,
        )
        if world_size > 1:
            _hb("checkpoint + tokenizer + 模型加载完成，即将进入 simple_evaluate")
        if benchmark != "debug":
            benchmark = ",".join(_skip_optional_tasks(benchmark.split(","), lm_model.max_length))
            _check_task_lengths(benchmark.split(","), lm_model.max_length)
            _check_task_data(benchmark.split(","))

        with (
            _harness_logging(),
            _trace_hf_cache_resolution(),
            evaluation_sample_limit(limit),
            diag_mode(
                log_kv_diag_mode,
                exact_from_layer=log_kv_diag_exact_from_layer,
                peak_window_from_end=log_kv_diag_peak_window_from_end,
                second_order_max_width=log_kv_diag_second_order_max_width,
                second_order_max_layer=log_kv_diag_second_order_max_layer,
            ) if diag_active else contextlib.nullcontext()
        ):
            results = evaluator.simple_evaluate(
                model=lm_model,
                tasks=["piqa"] if benchmark == "debug" else benchmark.split(","),
                task_manager=_task_manager(metadata),
                confirm_run_unsafe_code=True,
                batch_size=1,
                metadata=metadata,
                limit=limit,
                log_samples=True,
            )
        _hb("simple_evaluate 已返回")
        if results:
            _drop_empty_lengths(results)
        compression_samples = lm_model.all_gather_results(lm_model.compression_metrics, tag="KV compression")
        speed_samples = lm_model.all_gather_results(
            [{"generation": lm_model.gen_metrics, "scoring": lm_model.ppl_metrics}], tag="inference speed",
        )

        # 落盘阶段：只有 rank 0 写文件（建目录、写 json/csv/xlsx）。其它 rank 什么都
        # 不做，直接到下面的 barrier 等 rank 0 写完 —— 各 rank 同时往共享盘写同名文件
        # 会互相截断，而 mkdir/exists 各判各的又会引入分支不一致。
        if _is_main():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            gen_metrics = [row for rank in speed_samples for row in rank["generation"]]
            ppl_metrics = [row for rank in speed_samples for row in rank["scoring"]]
            inference_summary = summarize_inference_metrics(gen_metrics, ppl_metrics)
            inference_report = {
                **inference_summary,
                "world_size": world_size,
                "inference_mlp_chunk_size": lm_model.config.inference_mlp_chunk_size,
                "generate_until": gen_metrics,
                "loglikelihood": ppl_metrics,
            }
            for kind in ("generation", "scoring"):
                if inference_summary[kind]["requests"]:
                    print(f"⏱ 推理速度（{kind}，全部 rank 样本，单设备有效吞吐）: "
                          + json.dumps(inference_summary[kind], ensure_ascii=False), flush=True)
            compression_summary = summarize_compression(compression_samples)
            compression_report = {
                "measurement_point": "request_end_before_cache_reset",
                "slot_definition": "valid stored KV entries across all layers, batches and KV groups; "
                                   "includes recent, deferred tokens and Alpha pending spans; excludes padding, "
                                   "virtual attention anchors and workspace",
                "summary": compression_summary,
                "by_request_type": {
                    kind: summarize_compression([row for row in compression_samples if row["request_type"] == kind])
                    for kind in sorted({row["request_type"] for row in compression_samples})
                },
                "samples": compression_samples,
            }
            if compression_summary["request_count"]:
                print(
                    f"🧮 KV 槽位压缩（全部 rank，{compression_summary['request_count']} 次调用，结束时快照）: "
                    f"保留比例 {compression_summary['kv_retention_ratio']:.2%}，"
                    f"节省比例 {compression_summary['kv_saving_ratio']:.2%}，"
                    f"压缩倍数 {compression_summary['kv_compression_factor']:.2f}x",
                    flush=True,
                )
            else:
                print("🧮 KV 槽位压缩: 本次没有模型调用，未产生可统计数据", flush=True)

            # 🌟 诊断汇总（score/value oracle 归因；见 litgpt.log_kv_diag）：每次调用
            # 对应一个 (task 类型, mode) 组合，独立落一份 JSON —— 不与常规 results 合并，
            # 也不跨 rank 聚合（诊断样本量小，单卡跑就够，见上面的启动打印）。
            if diag_active:
                diag_dir = Path(log_kv_diag_output or output_path or ".").expanduser()
                diag_dir.mkdir(parents=True, exist_ok=True)
                tag = log_kv_diag_mode
                if log_kv_second_order_scale != 1.0:
                    tag += f"_sos{log_kv_second_order_scale:g}"
                if log_kv_diag_exact_from_layer is not None:
                    tag += f"_efl{log_kv_diag_exact_from_layer}"
                if log_kv_diag_peak_window_from_end is not None:
                    tag += f"_pw{log_kv_diag_peak_window_from_end}"
                if log_kv_diag_second_order_max_width is not None:
                    tag += f"_w{log_kv_diag_second_order_max_width}"
                if log_kv_diag_second_order_max_layer is not None:
                    tag += f"_l{log_kv_diag_second_order_max_layer}"
                # checkpoint 目录名嵌进文件名——不同 checkpoint（比如 naive vs
                # warmup）的诊断产物即使不小心落进同一个 output_path，靠文件名
                # 也能分清，不用回头翻时间戳猜是哪次跑的（历史上 D5 数据被搞混过一次）。
                ckpt_name = Path(checkpoint_dir).name
                diag_file = diag_dir / f"diag_{ckpt_name}_{tag}_{benchmark.replace(',', '+')}_{ts}.json"
                with open(diag_file, "w", encoding="utf-8") as f:
                    json.dump(LOG_KV_DIAG.summary(), f, indent=2, ensure_ascii=False)
                print(f"🔬 诊断汇总已保存到: {diag_file}")

            # 🌟 第一步：立即保存原始 results 对象，便于后续恢复
            results_cache_file = Path("eval_results_cache.json")
            try:
                with open(results_cache_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
                print(f"💾 原始 results 已缓存到: {results_cache_file}")
            except Exception as e:
                print(f"❌ 缓存 results 失败: {e}")
                print(f"⚠️ 尝试使用备用方案...")
                try:
                    # 备用方案：先转换为字符串表示
                    results_str = str(results)
                    with open(results_cache_file, "w", encoding="utf-8") as f:
                        json.dump({"results_str": results_str}, f, indent=2, ensure_ascii=False)
                    print(f"💾 results 已以字符串形式缓存到: {results_cache_file}")
                except Exception as e2:
                    print(f"❌ 备用方案也失败了: {e2}")
                    return

            # 🌟 第二步：打印表格
            from lm_eval.utils import make_table
            print(make_table(results))

            # 🌟 第三步：如果指定了输出路径，保存完整的JSON输出
            if output_path is not None:
                json_output = {
                    "timestamp": ts,
                    **_code_version(),
                    "benchmark": benchmark,
                    "checkpoint_dir": checkpoint_dir,
                    "log_kv_dense_mode": log_kv_dense_mode,
                    "inference_mlp_chunk_size": lm_model.config.inference_mlp_chunk_size,
                    "swa_window_size": lm_model.swa_window_size,
                    "cache_budget": lm_model.cache_budget,
                    **getattr(lm_model, "baseline_provenance", {}),
                    "kv_compression": compression_report,
                    "inference_metrics": inference_report,
                    "log_kv_B": log_kv_B,
                    "log_kv_recent_size": log_kv_recent_size,
                    "log_kv_prefill_block": log_kv_prefill_block,
                    "log_kv_second_order_scale": log_kv_second_order_scale,
                    "metadata": metadata,
                    "log_kv_importance_pooling": log_kv_importance_pooling,
                    "log_kv_importance_pooling_lambda": log_kv_importance_pooling_lambda,
                    "log_kv_importance_pooling_temperature": log_kv_importance_pooling_temperature,
                    "log_kv_semantic_clusters": log_kv_semantic_clusters,
                    "log_kv_cluster_k_max": log_kv_cluster_k_max,
                    "log_kv_cluster_lambda_rel": log_kv_cluster_lambda_rel,
                    "log_kv_seg_eta": log_kv_seg_eta,
                    "log_kv_seg_g0": log_kv_seg_g0,
                    "log_kv_seg_gap_max": log_kv_seg_gap_max,
                    "log_kv_seg_block_level": log_kv_seg_block_level,
                    "log_kv_seg_forget": log_kv_seg_forget,
                    "log_kv_semantic_s_h_path": log_kv_semantic_s_h_path,
                    "log_kv_semantic_flush_granularity": log_kv_semantic_flush_granularity,
                    "log_kv_semantic_cluster_chunk_size": log_kv_semantic_cluster_chunk_size,
                    "log_kv_semantic_capacity_beta": log_kv_semantic_capacity_beta,
                    "log_kv_semantic_capacity_hard_cap_mult": log_kv_semantic_capacity_hard_cap_mult,
                    "log_kv_semantic_legacy_route": log_kv_semantic_legacy_route,
                    "log_kv_semantic_unified_route": log_kv_semantic_unified_route,
                    "log_kv_semantic_merge_passes": log_kv_semantic_merge_passes,
                    "log_kv_semantic_anchor_mode": log_kv_semantic_anchor_mode,
                    "log_kv_semantic_pack_backend": log_kv_semantic_pack_backend,
                    "log_kv_semantic_centroid_backend": log_kv_semantic_centroid_backend,
                    "log_kv_semantic_replay_updates": log_kv_semantic_replay_updates,
                    "log_kv_alpha_exact_tokens": log_kv_alpha_exact_tokens,
                    "log_kv_alpha_span_max_tokens": log_kv_alpha_span_max_tokens,
                    "log_kv_beta_novelty": log_kv_beta_novelty,
                    "log_kv_beta_adaptive_merge": log_kv_beta_adaptive_merge,
                    "log_kv_gamma_level0_reinsert": log_kv_gamma_level0_reinsert,
                    "log_kv_gamma_top_merge": log_kv_gamma_top_merge,
                    "log_kv_gamma_level_slack": log_kv_gamma_level_slack,
                    "log_kv_gamma_cue_tokens": log_kv_gamma_cue_tokens,
                    "log_kv_gamma_literals": log_kv_gamma_literals,
                    "log_kv_gamma_cue_guard": log_kv_gamma_cue_guard,
                    "log_kv_gamma_query_cue": log_kv_gamma_query_cue,
                    "log_kv_gamma_query_detector": log_kv_gamma_query_detector,
                    "log_kv_gamma_query_detector_options": log_kv_gamma_query_detector_options,
                    "log_kv_gamma_query_pool_tokens": log_kv_gamma_query_pool_tokens,
                    "query_detector_stats": (lm_model._query_detector.stats()
                                             if getattr(lm_model, "_query_detector", None) is not None else None),
                    "log_kv_retention_heads": log_kv_retention_heads,
                    "log_kv_gamma_surprisal": log_kv_gamma_surprisal,
                    "log_kv_retention_tag": (lm_model._retention_heads.tag()
                                             if getattr(lm_model, "_retention_heads", None) is not None else None),
                    "results": results,
                }

                base = Path(output_path).expanduser()
                if base.suffix.lower() == ".json":
                    output_file = base.with_name(f"{base.stem}_{ts}{base.suffix}")
                else:
                    output_file = base / f"eval_results_{ts}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)

                if compression_samples:
                    compression_csv = output_file.with_suffix(".kv_compression.csv")
                    columns = list(dict.fromkeys(key for row in compression_samples for key in row))
                    with open(compression_csv, "w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=columns)
                        writer.writeheader()
                        writer.writerows(compression_samples)
                    print(f"📊 KV 槽位压缩明细已保存至 {compression_csv}")

                try:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(json_output, f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
                    print(f"✅ 完整结果已保存到: {output_file}")
                except Exception as e:
                    print(f"❌ 保存完整结果失败: {e}")
                    print(f"⚠️ 尝试备用方案...")
                    try:
                        json_output["results"] = str(results)
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(json_output, f, indent=2, ensure_ascii=False)
                        print(f"✅ 完整结果已以备用方案保存到: {output_file}")
                    except Exception as e2:
                        print(f"❌ 备用方案也失败了: {e2}")

                # 🌟 第四步：保存推理时延 xlsx
                inference_xlsx = output_file.with_suffix(".inference_metrics.xlsx")
                try:
                    import pandas as pd
                    with pd.ExcelWriter(inference_xlsx) as writer:
                        pd.DataFrame([
                            {"request_type": kind,
                             "inference_mlp_chunk_size": lm_model.config.inference_mlp_chunk_size,
                             **inference_summary[kind]}
                            for kind in ("generation", "scoring")
                        ]).to_excel(writer, sheet_name="speed_summary", index=False)
                        if gen_metrics:
                            gen_df = pd.DataFrame(gen_metrics)
                            gen_df.to_excel(writer, sheet_name="generate_until", index=False)
                            # 按 prompt 长度分桶统计
                            gen_df["prompt_bucket"] = pd.cut(gen_df["prompt_len"],
                                bins=[0, 1024, 4096, 8192, 16384, 32768, 999999],
                                labels=["0-1K", "1K-4K", "4K-8K", "8K-16K", "16K-32K", "32K+"])
                            summary = pd.DataFrame([
                                {"prompt_bucket": str(bucket), **summarize_inference_metrics(
                                    rows.to_dict("records"), [],
                                )["generation"]}
                                for bucket, rows in gen_df.groupby("prompt_bucket", observed=True)
                            ])
                            summary.to_excel(writer, sheet_name="gen_by_bucket", index=False)
                        if ppl_metrics:
                            ppl_df = pd.DataFrame(ppl_metrics)
                            ppl_df.to_excel(writer, sheet_name="loglikelihood", index=False)
                    print(f"📊 推理时延指标已保存至 {inference_xlsx}")
                except Exception as e:
                    print(f"⚠️ 推理时延 xlsx 保存失败: {e}")

                # 🌟 第五步：生成 CSV 结果文件
                extract_results_to_csv(results, benchmark, output_file)

        eval_done = True
    finally:
        try:
            if _dist_ready():
                if eval_done:
                    wait_s = _wait_all_ranks("收尾同步")
                    _eval_print(f"🤝 [{_rank_label()}] 收尾同步完成，用时 {wait_s:.1f}s", flush=True)
                else:
                    print(f"⚠️ [{_rank_label()}] 评测未正常结束，跳过收尾 barrier", flush=True)
        finally:
            # 清理不能掩盖原始异常，也不能销毁训练持有的默认组。
            if eval_group_owned_here:
                try:
                    dist.destroy_process_group(eval_group)
                except Exception as e:  # noqa: BLE001
                    print(f"⚠️ [{_rank_label()}] 销毁评测 Gloo 组失败: {e}", flush=True)
            if pg_owned_here and dist.is_available() and dist.is_initialized():
                try:
                    rank_before_destroy = _global_rank()
                    dist.destroy_process_group()
                    if rank_before_destroy == 0:
                        print("✅ 分布式进程组已销毁，评测进程可以正常退出了", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"⚠️ [{_rank_label()}] 销毁进程组失败: {e}", flush=True)


@auto_expand_env_vars
def output_from_cache(
    cache_file: str = "eval_results_cache.json",
    benchmark: str = "debug",
    checkpoint_dir: str = "checkpoints/Qwen/Qwen3-0.6B-Base",
    output_path: str | None = None,
):
    """从缓存文件读取 results，重新进行输出处理（无需重新运行评测）"""
    cache_path = Path(cache_file)
    if not cache_path.exists():
        print(f"❌ 缓存文件不存在: {cache_path}")
        return

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        # 检查是否是备用方案保存的（字符串形式）
        if "results_str" in cache_data:
            print(f"⚠️ 缓存是字符串形式，无法恢复为结构化数据")
            print(f"缓存内容: {cache_data['results_str'][:500]}...")
            return

        results = _drop_empty_lengths(cache_data)
        print(f"✅ 从缓存读取 results: {cache_path}")
    except Exception as e:
        print(f"❌ 读取缓存失败: {e}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 打印表格
    try:
        from lm_eval.utils import make_table
        print(make_table(results))
    except Exception as e:
        print(f"⚠️ 打印表格失败: {e}")

    # 保存完整的JSON输出
    if output_path is not None:
        json_output = {
            "timestamp": ts,
            **_code_version(),
            "benchmark": benchmark,
            "checkpoint_dir": checkpoint_dir,
            "log_kv_dense_mode": log_kv_dense_mode,
            "results": results,
        }

        base = Path(output_path).expanduser()
        if base.suffix.lower() == ".json":
            output_file = base.with_name(f"{base.stem}_{ts}{base.suffix}")
        else:
            output_file = base / f"eval_results_{ts}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
            print(f"✅ 完整结果已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")

        # 生成 CSV 结果文件
        extract_results_to_csv(results, benchmark, output_file)

if __name__ == "__main__":
    run_cli(main)
