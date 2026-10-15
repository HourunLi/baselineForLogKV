# Vendored for the DMS baseline; local namespace changes are recorded in SOURCE.json.
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

from models._vendor.semanticlogkv.utils import *


# ==========================================
# 🧩 分布式小工具
# ==========================================
# 约定（与 demo.py 的 checkpoint 保存一致）：**任何"看文件系统"的判定只由
# rank 0 做，再 broadcast 给其它 rank**。共享盘（OBS/NFS）的元数据缓存在各节点
# 上的可见时刻不一致，若每个 rank 自己 stat/glob，就可能出现分支不一致 ——
# 一部分 rank 进了集合通信（barrier / all_gather）而另一部分跳过或直接抛异常
# 退出，剩下的 rank 就会等待通信超时。评测只交换 CPU 对象，使用 Gloo；
# 保留长等待窗口以容纳不同 rank 的长上下文计算时差。

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


def _tqdm_position() -> int:
    # Use global rank so aggregated multi-node stdout gets one tqdm row per
    # process instead of collapsing every node onto local ranks 0..7.
    return _global_rank()


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
    """无条件心跳打印（所有 rank，不受 is_main 限制），排查多机卡死用。

    标记 dist.init_process_group / 模型加载完成这类一次性里程碑（真正的逐条
    计算进度由 loglikelihood/generate_until 里每个 rank 各自的 tqdm 负责，
    不需要在这里重复打印）。带 hostname，方便从日志里按 rank 分组核对。
    """
    import socket
    host = socket.gethostname()
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"💓 [{ts}][{_rank_label()}][{host}] {stage}", flush=True)


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

# if 'HF_DATASETS_CACHE' not in os.environ and 'PKU' not in os.environ:
print("设置HF_DATASETS_CACHE环境变量...")

BASE = os.environ.get('DMS_HF_CACHE', '/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache')
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
# 跟评测本身的计算进度混在一起。这里关掉，只留下面每个 rank 自己的计算进度。
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
import datasets.config as _hf_datasets_config
import huggingface_hub.constants as _hf_hub_constants

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

from models._vendor.semanticlogkv.litgpt.config import Config
from models._vendor.semanticlogkv.litgpt.model import GPT
from models._vendor.semanticlogkv.litgpt.tokenizer import Tokenizer
from lm_eval import evaluator
from lm_eval.api.model import LM
from models._vendor.semanticlogkv.litgpt.generate.base import generate as litgpt_generate
from models._vendor.semanticlogkv.litgpt.kv_compression import cache_compression_snapshot, summarize_compression
from models._vendor.semanticlogkv.litgpt.log_kv_diag import DIAG as LOG_KV_DIAG, diag_mode
from models._vendor.semanticlogkv.litgpt.ruler_patch import apply_patch
apply_patch()

_CONFIG_FIELDS = {f.name for f in dataclasses.fields(Config)}


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


def _load_lit_model_checkpoint(
    checkpoint_dir: str, map_location: str | torch.device, wait_s: float = 120.0
) -> Any:
    """加载 ``{checkpoint_dir}/lit_model.pth``（单文件 ``torch.save``，与 demo FSDP ``state_dict_type='full'`` 一致）。

    多机场景下 rank 0 已经确认过文件存在（见 ``_resolve_checkpoint_dir``），但其它
    节点的共享盘元数据缓存可能还没刷新。此时直接抛 FileNotFoundError 会让这个 rank
    单独退出，其余 rank 卡在后续集合通信里等到超时；所以非 rank0 上先轮询等待
    ``wait_s`` 秒。单进程运行时不等待，行为与之前一致（立刻报错）。
    """
    lit_path = Path(checkpoint_dir).expanduser() / "lit_model.pth"
    if not lit_path.exists() and _dist_ready() and not _is_main():
        deadline = time.time() + wait_s
        while time.time() < deadline and not lit_path.exists():
            time.sleep(2.0)
        if lit_path.exists():
            print(f"[eval][{_rank_label()}] 共享盘元数据延迟，等待后已看到 {lit_path}")
    if not lit_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {lit_path}")
    if lit_path.is_dir():
        raise FileNotFoundError(
            f"{lit_path} 为目录（旧版 FSDP/DCP 分片）。请先在仓库根目录对同一路径调用 "
            "`utils.convert_and_replace_fsdp_ckpt` 合并为单文件 lit_model.pth，再跑 eval。"
        )
    return torch.load(str(lit_path), map_location=map_location, weights_only=False)


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _filter_config_dict(d: dict[str, Any]) -> dict[str, Any]:
    dropped = sorted(k for k in d if k not in _CONFIG_FIELDS)
    if dropped:
        print(f"[eval] 过滤掉非 Config 字段: {dropped}")
    return {k: v for k, v in d.items() if k in _CONFIG_FIELDS}


def _normalize_training_config_dict(d: dict[str, Any]) -> tuple[dict[str, Any], list[int] | None]:
    """训练 exp / CLI 里常用 research_*_layers_str；Config 只接受 research_prefill_swa_layers 等。"""
    out = dict(d)
    identity_layers: list[int] | None = None
    if "research_swa_layers_str" in out:
        out["research_prefill_swa_layers"] = _parse_csv_ints(out.pop("research_swa_layers_str"))
    return out, identity_layers


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
        "timing_scope": "Model calls only, excluding tokenization, cache setup/reset, result gathering, and file output. Prefill includes first-token sampling; decode excludes the first generated token. Generation counts include EOS; no warmup requests are discarded.",
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
        tokenizer_dir: str | None = None,
        process_group=None,
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
            self.config = _config_from_yaml_and_overrides(config_path, config_overrides)
        else:
            if is_master: print("⚠️ 未发现训练期保存的 YAML 配置文件，正在使用备用参数初始化...")
            fallback_kw: dict[str, Any] = {"name": checkpoint_dir.split("/")[-1]}
            if config_overrides:
                fallback_kw.update(config_overrides)
            fallback_kw = _filter_config_dict(fallback_kw)
            self.config = Config.from_name(**fallback_kw)

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
            self.model.set_alpha_span_boundary_ids(self.tokenizer.alpha_span_boundary_ids)

        if is_master: print(f"🔄 正在加载权重...")
        checkpoint = _load_lit_model_checkpoint(checkpoint_dir, map_location=device)

        # 🌟 核心修复：检查是不是被包裹过的 checkpoint 字典
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            if is_master: print("📦 检测到 Fabric Checkpoint，已自动提取 model 权重。")
        else:
            state_dict = checkpoint

        # 🌟 强烈建议：捕获并打印一下加载结果，看看是不是真的加载成功了！
        load_result = self.model.load_state_dict(state_dict, strict=False)

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
            self.model.set_kv_cache(
                batch_size=1,
                max_seq_length=max_seq_length,
                device=self._device,
                dtype=dtype,
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
        first = self.model.transformer.h[0].attn.kv_cache
        if getattr(first, "alpha_exact_tokens", 0):
            self.cache_budget.update(alpha_exact_tokens=first.alpha_exact_tokens, effective_B=first.B,
                                     alpha_span_max_tokens=first.alpha_span_max_tokens,
                                     beta_novelty=first.beta_novelty, beta_adaptive_merge=first.beta_adaptive_merge)
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
    def all_gather_results(self, local_result_list: list, tag: str = ""):
        if not _dist_ready():
            return local_result_list

        # 结果只有 float/bool/str，不必经 GPU/NCCL 序列化与通信。
        # 汇合等待长上下文的负载差异；等待期间的心跳与超时见 _wait_all_ranks。
        print(f"⏳ [{_rank_label()}] {tag} 本 rank {len(local_result_list)} 条已完成，等待其它 rank", flush=True)
        wait_s = _wait_all_ranks(f"{tag} 汇总")
        print(
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
        print(
            f"✅ [{_rank_label()}] {tag} Gloo 结果汇总完成，共 {len(final_results)} 条，"
            f"用时 {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return final_results

    # ==========================================
    # 🌟 核心 1：PPL 与 选择题评测
    # ==========================================
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
        rank_label = _rank_label()

        # 🌟 安全阀：如果 题干 + 选项 > max_seq_length，必须切掉题干最前面的部分
        max_len = self.model.max_seq_length
        if len(ctx_enc) + len(cont_enc) > max_len:
            keep_ctx_len = max_len - len(cont_enc)
            if keep_ctx_len <= 0:
                cont_enc = cont_enc[-(max_len - 1):]
                ctx_enc = []
            else:
                ctx_enc = ctx_enc[-keep_ctx_len:]
            print(f"⚠️ [{rank_label}] 警告: 触发截断，剩余 context 长度: {len(ctx_enc)}")

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
                t0 = time.perf_counter()
                input_pos = torch.arange(seq_len, device=self._device, dtype=torch.int64)
                # lm_head_start: materialize logits only for the scoring span
                # [ctx_len-1, seq_len) — full-sequence logits at 32K are ~10 GB
                # bf16, the sliced tensor is (cont_len + 1) x vocab.
                logits = self.model(inps, input_pos=input_pos, lm_head_start=ctx_len - 1)
                _synchronize_inference_device(self._device)
                t1 = time.perf_counter()
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
            "forward_time_s": t1 - t0,
            "forward_time_ms": (t1 - t0) * 1000,
            "forward_tokens_per_sec": seq_len / (t1 - t0) if t1 > t0 else None,
            **peak_memory,
        })

        # ``logits`` starts at position ctx_len-1 (lm_head_start); its first
        # len(cont_enc) rows are the positions [ctx_len-1, seq_len-2] that
        # predict the continuation tokens.
        cont_logits = logits[0, : len(cont_enc)]
        cont_targets = torch.tensor(cont_enc, dtype=torch.long, device=self._device)

        # fp32 log-softmax in position-chunks: rolling-PPL requests score a
        # full window (cont_len ~ max_seq_length), and a one-shot fp32
        # log_softmax over cont_len x vocab would peak at ~19 GB at 32K. fp32
        # (rather than the model's bf16) keeps per-token log-probs accurate —
        # they are summed over thousands of tokens downstream.
        total_logprob = 0.0
        is_greedy = True
        step = 1024
        for i in range(0, cont_logits.size(0), step):
            blk = cont_logits[i : i + step].to(torch.float32)
            tgt = cont_targets[i : i + step]
            log_probs = torch.log_softmax(blk, dim=-1)
            total_logprob += log_probs.gather(dim=-1, index=tgt.unsqueeze(-1)).sum().item()
            if is_greedy:
                is_greedy = bool((blk.argmax(dim=-1) == tgt).all().item())
        return total_logprob, is_greedy

    def loglikelihood(self, requests):
        dp_rank = _global_rank()
        dp_size = _world_size()

        local_requests = requests[dp_rank::dp_size]
        results = []

        # 每个 rank 都显示自己的进度条（不再只有 rank 0 可见）。
        desc = f"loglikelihood GlobalRank {dp_rank}/{dp_size} (local {_local_rank()})"
        for req in tqdm.tqdm(local_requests, desc=desc, position=_tqdm_position()):
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
    def generate_until(self, requests):
        dp_rank = _global_rank()
        dp_size = _world_size()

        local_requests = requests[dp_rank::dp_size]
        results = []

        # 每个 rank 都显示自己的进度条（不再只有 rank 0 可见）。
        desc = f"generate_until GlobalRank {dp_rank}/{dp_size} (local {_local_rank()})"
        for local_i, req in enumerate(tqdm.tqdm(local_requests, desc=desc, position=_tqdm_position())):
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
                # models._vendor.semanticlogkv.litgpt.generate.sample 仅在 temperature<=0 且 top_p<=0 时走 argmax；仅设 temperature=0 而 top_p=1 仍会多项式采样
                temperature = 0.0
                top_p = 0.0

            prompt_tensor = self.tokenizer.encode(prompt, device=self._device)
            original_prompt_tokens = int(prompt_tensor.size(0))
            prompt_token_offset = 0

            # 🌟 安全阀：为生成的新 Token 预留空间
            max_len = self.model.max_seq_length
            if prompt_tensor.size(0) + max_new_tokens > max_len:
                keep_prompt_len = max_len - max_new_tokens
                prompt_token_offset = int(prompt_tensor.size(0) - keep_prompt_len)
                prompt_tensor = prompt_tensor[-keep_prompt_len:]
                print(f"⚠️ [{_rank_label()}] 警告: 触发生成截断，Prompt 被切至: {keep_prompt_len}")

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
            results.append(decoded)

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
        # 每个 rank 都显示自己的进度条（不再只有 rank 0 可见）。
        desc = f"loglikelihood_rolling GlobalRank {dp_rank}/{dp_size} (local {_local_rank()})"
        for req in tqdm.tqdm(local_requests, desc=desc, position=_tqdm_position()):
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
    """自动检测分段 checkpoint。若存在 step_* 子目录（含 lit_model.pth）则选最大 step，否则直接用原路径。

    glob + exists() 必须只由 rank 0 判定再广播：训练刚写完最后一个 step_* 时，
    其它节点的共享盘元数据缓存可能还看不到它，各 rank 各自 glob 就会选到**不同的
    checkpoint**（评测结果静默不可比），或者一部分 rank 找不到权重直接退出、把其余
    rank 留在集合通信里等到超时。
    """
    resolved: str | None = None
    if _is_main():
        base = Path(os.path.expandvars(os.path.expanduser(checkpoint_dir)))
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
    # ── 🧩 logKV：tokenizer 回退（checkpoint 目录缺 tokenizer 文件时用）──
    tokenizer_dir: str | None = None,
    # ── 只跑一小批样本（Phase 0 诊断用；见 log_kv_diag_mode）。int = 绝对条数，
    # float in (0, 1) = lm-eval 的抽样比例。None = 跑全量。
    limit: int | float | None = None,
    # ── 🧩 logKV 诊断（score/value oracle 归因网格；见 models._vendor.semanticlogkv.litgpt.log_kv_diag）──
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
            if _k != "config" and _k not in _valid:
                print(f"[eval] WARNING: unknown YAML key ignored: {_k}")

    def _o(name, current):
        v = _yaml.get(name)
        return v if v is not None else current

    checkpoint_dir = _o("checkpoint_dir", checkpoint_dir)
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
    if not log_kv_semantic_clusters:
        log_kv_alpha_exact_tokens = 0  # exact spans live on semantic clusters only
    if not log_kv_alpha_exact_tokens:
        # Beta extends the Alpha exact pool; disabling Alpha disables both Beta changes.
        log_kv_beta_novelty = log_kv_beta_adaptive_merge = False
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
                f"beta_novelty={log_kv_beta_novelty}, beta_adaptive_merge={log_kv_beta_adaptive_merge})"
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
            tokenizer_dir=tokenizer_dir,
            process_group=eval_group,
        )
        if world_size > 1:
            _hb("checkpoint + tokenizer + 模型加载完成，即将进入 simple_evaluate")

        with (
            _trace_hf_cache_resolution(),
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
                confirm_run_unsafe_code=True,
                batch_size=1,
                metadata=metadata,
                limit=limit,
                log_samples=True,
            )
        _hb("simple_evaluate 已返回")
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

            # 🌟 诊断汇总（score/value oracle 归因；见 models._vendor.semanticlogkv.litgpt.log_kv_diag）：每次调用
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
                    "benchmark": benchmark,
                    "checkpoint_dir": checkpoint_dir,
                    "log_kv_dense_mode": log_kv_dense_mode,
                    "swa_window_size": lm_model.swa_window_size,
                    "cache_budget": lm_model.cache_budget,
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
                            {"request_type": kind, **inference_summary[kind]}
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
                    print(f"🤝 [{_rank_label()}] 收尾同步完成，用时 {wait_s:.1f}s", flush=True)
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

        results = cache_data
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
