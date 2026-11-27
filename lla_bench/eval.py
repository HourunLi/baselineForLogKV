from __future__ import annotations

import os
import re
import csv
import glob
import hashlib
import json
import builtins
import sys
import warnings
import logging
import time
import inspect
import contextlib
from importlib.metadata import version
from datetime import datetime, timedelta
from pathlib import Path

BASE = os.environ.get("HF_HOME", "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache")
os.environ['HF_HOME'] = BASE
os.environ['HF_DATASETS_CACHE'] = f'{BASE}/hf_cache'
os.environ['HF_EVALUATE_CACHE'] = f'{BASE}/evaluate'
os.environ['HF_MODULES_CACHE'] = f'{BASE}/modules'
os.environ['HUGGINGFACE_HUB_CACHE'] = f'{BASE}/hub'
os.environ['HF_HUB_CACHE'] = f'{BASE}/hub'
os.environ['RULER_CACHE_DIR'] = f'{BASE}/ruler_cache'
os.environ.setdefault("NLTK_DATA", f"{BASE}/nltk_data")
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

import yaml
import torch
import torch.distributed as dist
import numpy as np
from typing import Any

import tqdm

from lla_bench.utils import *


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
        raise RuntimeError(f"保存 CSV 失败: {csv_file}") from e


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

from lla_bench.model import HAttentionLM, HConfig
from lla_bench.litgpt.tokenizer import Tokenizer
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
    from lla_bench.litgpt.longbench2_patch import data_problem

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
from lla_bench.litgpt.generate.base import generate as litgpt_generate
from lla_bench.metrics import cache_compression_snapshot, summarize_compression
from lla_bench.litgpt.ruler_dataset_cache import evaluation_sample_limit
from lla_bench.litgpt.ruler_patch import apply_patch
apply_patch()
from lla_bench.litgpt.longbench2_patch import apply_patch as apply_longbench2_patch  # noqa: E402
apply_longbench2_patch()
from lla_bench.litgpt.hf_cache_patch import apply_patch as apply_hf_cache_patch  # noqa: E402
apply_hf_cache_patch()  # offline loads skip <hash>.incomplete folders




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


class NativeLM(LM):
    """The niahLogKV request/scoring harness using the selected native model."""

    def __init__(self, checkpoint_dir, device="cuda", config_overrides=None, tokenizer_dir=None, process_group=None):
        super().__init__()
        self.process_group = process_group
        self._device = device
        self.checkpoint_dir = checkpoint_dir
        self.cache_budget = {}
        self._truncation_counts = {}
        options = config_overrides or {}
        resolved = _find_tokenizer_dir(checkpoint_dir, tokenizer_dir, group=process_group)
        self.resolved_tokenizer_dir = str(resolved)
        self.tokenizer = Tokenizer(resolved)
        self.infini = "infini_segment_length" in json.loads((Path(checkpoint_dir) / "config.json").read_text())
        if self.infini:
            from infini_attention.evaluation import load_model
            self.model, self.config, self.checkpoint_source = load_model(checkpoint_dir, device, options)
            _configure_inference_memory(self.config, options)
            self.model.mlp_chunk = self.config.inference_mlp_chunk_size
        else:
            if set(options) - {"block_size", "inference_mlp_chunk_size"}:
                raise ValueError(f"unsupported native model options: {options}")
            self.config = HConfig.read(Path(checkpoint_dir) / "config.json", int(options.get("block_size", 32768)))
            _configure_inference_memory(self.config, options)
            self.checkpoint_source = None
            manifest = Path(checkpoint_dir) / "hattention_source.json"
            if manifest.is_file():
                source = json.loads(manifest.read_text())
                # If a prepared file was subsequently replaced, drop its published identity.
                if all((Path(checkpoint_dir) / name).is_file()
                       and (Path(checkpoint_dir) / name).stat().st_size == entry["size"]
                       and (Path(checkpoint_dir) / name).stat().st_mtime_ns == entry["mtime_ns"]
                       for name, entry in source["files"].items()):
                    self.checkpoint_source = source
            if self.config.student:
                from .student import checkpoint_report
                self.checkpoint_source = checkpoint_report(Path(checkpoint_dir))
            self.model = HAttentionLM.load(checkpoint_dir, self.config, device)
        # 推理时延指标收集
        self.gen_metrics: list[dict] = []      # generate_until
        self.ppl_metrics: list[dict] = []      # loglikelihood
        self.compression_metrics: list[dict] = []

        # Eval cache is built lazily once and reset per request (see
        # _set_eval_cache) — never re-allocated per sample.
        self._eval_cache_ready = False

    def _set_eval_cache(self) -> None:
        if self._eval_cache_ready:
            self._reset_eval_cache()
            return
        self.model.set_cache()
        if self.infini:
            from infini_attention.evaluation import cache_budget
            self.cache_budget = cache_budget(self.model)
            self._eval_cache_ready = True
            return
        caches = [layer.mixer.cache for layer in self.model.backbone.layers]
        self.cache_budget = dict(persistent_cache_bytes=sum(buf.numel() * buf.element_size()
                                 for cache in caches for buf in cache.buffers()),
                                 allocated_matrix_levels=caches[0].states.size(0),
                                 trained_read_levels=self.config.trained_levels,
                                 state_dtype="float32", conv_dtype=str(caches[0].conv.dtype),
                                 window=self.model.max_seq_length)
        self._eval_cache_ready = True

    def _reset_eval_cache(self):
        self.model.reset_cache()

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
        chunk = int(os.environ.get("LOGKV_EVAL_PREFILL_CHUNK", "32768"))
        if chunk < 0:
            raise ValueError("LOGKV_EVAL_PREFILL_CHUNK must be nonnegative")
        return chunk

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
        from lla_bench.litgpt.generate.base import sample

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
        self, ctx_enc: list[int], cont_enc: list[int], *, request_type: str = "loglikelihood", sample_id=None
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

        cpu_ids = torch.tensor([ctx_enc + cont_enc], dtype=torch.long)
        input_hash = hashlib.sha256(cpu_ids.numpy().astype("<i8", copy=False)).hexdigest()
        inps = cpu_ids.to(self._device)
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
                if self._prefill_chunk() and seq_len > self._prefill_chunk():
                    score, is_greedy, forward_time = self._score_chunks(inps[0], ctx_len)
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
                    "context_len": ctx_len, "cont_len": len(cont_enc), "sample_id": sample_id,
                    "scoring_input_ids_sha256": input_hash,
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

        if self._prefill_chunk() and seq_len > self._prefill_chunk():
            return score, is_greedy

        # ``logits`` starts at position ctx_len-1 (lm_head_start); its first
        # len(cont_enc) rows are the positions [ctx_len-1, seq_len-2] that
        # predict the continuation tokens.
        cont_logits = logits[0, : len(cont_enc)]
        cont_targets = torch.tensor(cont_enc, dtype=torch.long, device=self._device)

        return _score_continuation_logits(cont_logits, cont_targets)

    def _score_chunks(self, tokens, ctx_len):
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

        for i, req in enumerate(_eval_progress(local_requests, "loglikelihood")):
            context, continuation = req.args[0], req.args[1]

            ctx_enc = self.tokenizer.encode(context).tolist()
            # Llama 3 等 use_bos=True：若对 continuation 再 encode 一次会多一个 BOS，拼接后破坏
            # loglikelihood 对齐；Qwen 通常无 BOS，bos=False 与默认行为一致。
            cont_enc = self.tokenizer.encode(continuation, bos=False).tolist()

            sample_id = _request_sample_id(req, global_request_index=i * dp_size + dp_rank, rank=dp_rank)
            results.append(self._score_tokens(ctx_enc, cont_enc, sample_id=sample_id))

        # 清理 CUDA 缓存，避免 all_gather 时 OOM
        torch.cuda.empty_cache()
        return self.all_gather_results(results, tag="loglikelihood")

    # ==========================================
    # 🌟 核心 2：自回归生成任务 (LongBench)
    # ==========================================
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

            prompt_tensor = self.tokenizer.encode(prompt)
            original_prompt_tokens = int(prompt_tensor.size(0))
            prompt_token_offset = 0

            # 🌟 安全阀：为生成的新 Token 预留空间
            max_len = self.model.max_seq_length
            if prompt_tensor.size(0) + max_new_tokens > max_len:
                keep_prompt_len = max_len - max_new_tokens
                prompt_token_offset = int(prompt_tensor.size(0) - keep_prompt_len)
                prompt_tensor = prompt_tensor[-keep_prompt_len:]
                self._warn_truncation("generate_until", keep_prompt_len)

            prompt_hash = hashlib.sha256(prompt_tensor.numpy().astype("<i8", copy=False)).hexdigest()
            prompt_tensor = prompt_tensor.to(self._device)
            total_max_len = prompt_tensor.size(0) + max_new_tokens
            prompt_len = prompt_tensor.size(0)

            with torch.no_grad():
                # 🧩 长上下文生成使用所选 cache（LogKV slot cache 或 dense KV）。
                # 建一次、按请求原地重置，不逐样本重分配 — 见 _set_eval_cache。
                self._set_eval_cache()
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
                "prompt_ids_sha256": prompt_hash,
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
        for i, req in enumerate(_eval_progress(local_requests, "loglikelihood_rolling")):
            sample_id = _request_sample_id(req, global_request_index=i * dp_size + dp_rank, rank=dp_rank)
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
                logprob, _ = self._score_tokens(ctx, chunk, request_type="loglikelihood_rolling", sample_id=sample_id)
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

@auto_expand_env_vars
def main(
    checkpoint_dir: str = "checkpoints/hattention-mamba2",
    benchmark: str = "debug",
    config_overrides: dict[str, Any] | None = None,
    output_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    limit: int | float | None = None,
    config: str | None = None,
    tokenizer_dir: str | None = None,
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
    benchmark = _o("benchmark", benchmark)
    config_overrides = _o("config_overrides", config_overrides)
    output_path = _o("output_path", output_path)
    metadata = _o("metadata", metadata)
    tokenizer_dir = _o("tokenizer_dir", tokenizer_dir)
    limit = _o("limit", limit)

    # LongBench v2 cannot be dropped from a requested evaluation, even if an old launcher marks it optional.
    # Check its data before CUDA initialization and checkpoint/model loading.
    longbench2_tasks = [task for task in benchmark.split(",")
                       if task.strip() == "longbench2" or task.strip().startswith("longbench2_")]
    if longbench2_tasks and _is_main():
        from lla_bench.litgpt.longbench2_patch import plain_folder
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
    device = os.environ.get("LLA_EVAL_DEVICE", "cuda")
    if device == "cuda":
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    elif device != "cpu" or world_size != 1:
        raise ValueError("LLA_EVAL_DEVICE must be cuda or single-process cpu")

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
        lm_model = NativeLM(checkpoint_dir, device=device, config_overrides=config_overrides,
                            tokenizer_dir=tokenizer_dir, process_group=eval_group)
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
                "slot_definition": ("local_kv_slots counts exact per-head entries; compressed history has no token slots"
                                    if lm_model.infini else "matrix states do not retain discrete KV pairs; slot ratios are null"),
                "byte_definition": "live local KV + committed FP32 memory and normalizer / dense BF16 KV bytes" if lm_model.infini else "live FP32 matrices + BF16 convolution history / reference dense BF16 KV bytes at the same token count",
                "dense_reference_layers_groups_head_dim": lm_model.config.dense_geometry,
                "summary": compression_summary,
                "by_request_type": {
                    kind: summarize_compression([row for row in compression_samples if row["request_type"] == kind])
                    for kind in sorted({row["request_type"] for row in compression_samples})
                },
                "samples": compression_samples,
            }
            print("Cache state bytes:", compression_summary, flush=True)

            # 🌟 第一步：立即保存原始 results 对象，便于后续恢复
            result_base = Path(output_path or ".")
            cache_dir = result_base.parent if result_base.suffix == ".json" else result_base
            results_cache_file = cache_dir / "eval_results_cache.json"
            results_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with results_cache_file.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, cls=SafeJSONEncoder)

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
                    "method": "Infini-Attention" if lm_model.infini else "qwen3_hattention_mamba2" if lm_model.config.student else "upstream_hattention_mamba2",
                    "checkpoint_source": lm_model.checkpoint_source,
                    "evaluation_device": str(lm_model.device),
                    "model_dtype": str(lm_model.model.lm_head.weight.dtype),
                    "device_name": torch.cuda.get_device_name() if torch.device(lm_model.device).type == "cuda" else "CPU",
                    "level_extrapolation": None if lm_model.infini else "share_deepest_trained_lambda",
                    "position_encoding": ("Qwen3 RoPE" if lm_model.infini else "native SSM (no RoPE)"),
                    "checkpoint_parameters": sum(p.numel() for p in lm_model.model.parameters()),
                    "model_config": dataclasses.asdict(lm_model.config),
                    "tokenizer_dir": lm_model.resolved_tokenizer_dir,
                    "prefill_chunk_size": lm_model._prefill_chunk(),
                    "package_versions": {pkg: version(pkg) for pkg in
                                         ("torch", "lm-eval", "datasets", "transformers", "huggingface-hub")},
                    "inference_mlp_chunk_size": lm_model.config.inference_mlp_chunk_size,
                    "cache_budget": lm_model.cache_budget,
                    "kv_compression": compression_report,
                    "inference_metrics": inference_report,
                    "metadata": metadata,
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

                with output_file.open("w", encoding="utf-8") as f:
                    json.dump(json_output, f, ensure_ascii=False, cls=SafeJSONEncoder)
                print(f"完整结果已保存到 {output_file}", flush=True)
                if lm_model.infini:
                    from infini_attention.report import write_summary
                    with output_file.with_suffix(".summary.csv").open("w", newline="", encoding="utf-8") as stream:
                        write_summary(stream, str(output_file), json_output)

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
                                bins=[0, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, float("inf")],
                                labels=["0-1K", "1K-4K", "4K-8K", "8K-16K", "16K-32K", "32K-64K", "64K-128K", "128K-256K", "256K-512K", "512K-1M", ">1M"])
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


if __name__ == "__main__":
    run_cli(main)
