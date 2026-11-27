from __future__ import annotations

import argparse
import os
import json
import re
import functools
import inspect
from typing import Any
import torch
import random
import numpy as np
import yaml
from litgpt.tokenizer import Tokenizer
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
import shutil


def load_log_kv_semantic_s_h(path: str | os.PathLike | None, n_layer: int, n_groups: int) -> torch.Tensor | None:
    """Load SemanticLogKV key-scale calibration as ``(n_layer, n_groups)`` fp32."""
    if path in (None, ""):
        return None
    p = os.path.expandvars(os.path.expanduser(os.fspath(path)))
    if p.endswith((".pt", ".pth")):
        obj = torch.load(p, map_location="cpu")
    else:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)

    if isinstance(obj, dict) and "key_scale" in obj:
        rows = []
        for layer in range(int(n_layer)):
            try:
                rows.append(obj["key_scale"][str(layer)]["s_h"])
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{p}: missing key_scale[{layer!r}]['s_h']") from exc
        t = torch.as_tensor(rows, dtype=torch.float32)
    else:
        if isinstance(obj, dict):
            obj = obj.get("semantic_s_h", obj.get("s_h"))
        if obj is None:
            raise ValueError(f"{p}: expected Stage0 manifest key_scale or s_h/semantic_s_h tensor data")
        t = torch.as_tensor(obj, dtype=torch.float32)

    if t.dim() == 1:
        if t.numel() != n_groups:
            raise ValueError(f"{p}: s_h shape {tuple(t.shape)} does not match n_groups={n_groups}")
        t = t.unsqueeze(0).expand(n_layer, -1).contiguous()
    elif tuple(t.shape) != (n_layer, n_groups):
        raise ValueError(f"{p}: s_h shape {tuple(t.shape)} does not match ({n_layer}, {n_groups})")
    if not bool(torch.isfinite(t).all()) or not bool((t > 0).all()):
        raise ValueError(f"{p}: s_h must be finite and > 0")
    return t.contiguous()


def convert_and_replace_fsdp_ckpt(ckpt_root: str):
    """
    将指定目录下的 FSDP/DCP checkpoint 转换为单文件，并使用 shutil 自动完成文件替换。
    假设原分片保存在名为 'lit_model.pth' 的【文件夹】中。
    """
    # 1. 定义路径
    original_dcp_dir = os.path.join(ckpt_root, "lit_model.pth")
    merged_file_temp = os.path.join(ckpt_root, "lit_model_merged.pth")
    backup_dcp_dir = os.path.join(ckpt_root, "lit_model_fsdp_dir.bak")

    # 2. 基础检查
    if not os.path.exists(original_dcp_dir):
        print(f"❌ 找不到源路径: {original_dcp_dir}")
        return
    if not os.path.isdir(original_dcp_dir):
        print(f"⚠️ {original_dcp_dir} 已经是一个文件，无需合并和替换！")
        return

    print(f"📦 [1/3] 正在读取分布式分片目录: {original_dcp_dir}")
    print("⏳ 正在聚合权重（此过程会在 CPU 内存中拼装完整模型，请耐心等待）...")

    try:
        # 3. 执行核心转换逻辑 (存为一个临时文件)
        dcp_to_torch_save(original_dcp_dir, merged_file_temp)
        
        file_size_gb = os.path.getsize(merged_file_temp) / (1024 ** 3)
        print(f"✅ 转换成功！合并后的单文件大小: {file_size_gb:.2f} GB")

        # 4. 偷梁换柱 (使用 shutil 转移文件)
        print("\n🔄 [2/3] 正在执行目录与文件的替换操作...")
        
        # 为了防止多次运行报错，检查是否已经有备份的旧目录，有则清理
        if os.path.exists(backup_dcp_dir):
            print(f"   * 发现残留的旧备份目录 {backup_dcp_dir}，正在清理...")
            shutil.rmtree(backup_dcp_dir)
        
        # (A) 备份原 FSDP 目录 (将文件夹 lit_model.pth 移动并重命名为 lit_model_fsdp_dir.bak)
        print("   -> 备份原目录: lit_model.pth ===> lit_model_fsdp_dir.bak")
        shutil.move(original_dcp_dir, backup_dcp_dir)
        
        # (B) 将新生成的单文件重命名为代码期望的名字
        print("   -> 替换新文件: lit_model_merged.pth ===> lit_model.pth")
        shutil.move(merged_file_temp, original_dcp_dir)

        print("\n🎉 [3/3] 全部完成！目录结构已自动整理完毕。")
        print(f"👉 现在的 lit_model.pth 是一个纯净的单文件。你可以直接去运行 eval.py 了！")

    except Exception as e:
        print(f"\n❌ 操作过程中发生致命错误: {e}")
        # 安全回滚逻辑：清理可能生成的、损坏的临时单文件
        if os.path.exists(merged_file_temp):
            os.remove(merged_file_temp)
            print("🧹 已自动清理未完成的残缺临时文件。")
        print("💡 提示：你的原始 FSDP 目录结构未受影响，请检查分片数据是否在保存时损坏。")

def _expand_single_string(text: str) -> str:
    """底层的单字符串替换逻辑"""
    def replace_fn(match):
        inner = match.group(1)
        if ':-' in inner:
            var_name, default_val = inner.split(':-', 1)
        elif '-' in inner:
            var_name, default_val = inner.split('-', 1)
        else:
            var_name, default_val = inner, ""
        return os.environ.get(var_name, default_val)
    return re.sub(r'\$\{([^}^{]+)\}', replace_fn, text)

def _deep_expand(obj: Any) -> Any:
    """递归遍历：支持嵌套的字典、列表、元组，精准爆破所有字符串"""
    if isinstance(obj, str):
        return _expand_single_string(obj)
    elif isinstance(obj, dict):
        return {k: _deep_expand(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_expand(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_deep_expand(v) for v in obj)
    # 如果是 int, float, bool 等基本类型，直接原样返回
    return obj

def auto_expand_env_vars(func):
    """
    魔法装饰器：拦截 CLI / Python 调用传进来的所有参数，清洗后再喂给目标函数。
    functools.wraps 极其关键，它能保留原函数的 Type Hint 签名，
    保证 CLI 辅助函数依然能正常解析！
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        expanded_args = _deep_expand(args)
        expanded_kwargs = _deep_expand(kwargs)
        return func(*expanded_args, **expanded_kwargs)
    return wrapper


def expand_env_vars(obj: Any) -> Any:
    """递归展开 ${VAR} / ${VAR-default} / ${VAR:-default}（bash 风格）。

    这是 YAML 路径展开约定的唯一实现（`from utils import *` 可见的公开名）。
    训练（demo.py）与评测（eval.py）的 YAML 装载、CLI 参数注入必须共用它：
    Python 自带的 os.path.expandvars 不认识 `${VAR-default}`（变量名含 `-`
    时查不到就原样保留），曾导致 demo 把 checkpoint 存进字面名为
    `lihourun` 的目录，而 majob.sh 用 bash 展开后的路径去
    评测，两边指向不同目录。幂等：已展开的字符串不含 `${...}`，再过一遍是
    no-op。
    """
    return _deep_expand(obj)


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _coerce_cli_value(value: str, param: inspect.Parameter) -> Any:
    if value.lower() in ("none", "null"):
        return None

    default = param.default
    annotation = "" if param.annotation is inspect.Parameter.empty else str(param.annotation)

    # Container annotations such as list[int] also contain scalar type names.
    if isinstance(default, (dict, list, tuple)) or "dict" in annotation or "list" in annotation or "tuple" in annotation:
        return yaml.safe_load(value)
    if isinstance(default, bool) or "bool" in annotation:
        return _str_to_bool(value)
    if (isinstance(default, int) and not isinstance(default, bool)) or "int" in annotation:
        return int(value)
    if isinstance(default, float) or "float" in annotation:
        return float(value)
    return value


def run_cli(func):
    """Small CLI runner that keeps ``--config`` available as a normal argument.

    ``jsonargparse.CLI`` reserves ``--config`` for its own config-file action,
    which collides with these scripts' explicit ``config`` parameter. This thin
    parser preserves the existing command style while still handling basic
    command-line overrides such as ``--log_kv_B 512`` and ``--max_steps 10``.
    """
    signature = inspect.signature(func)
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", "--local-rank", dest="_local_rank", default=None, help=argparse.SUPPRESS)

    for name, param in signature.parameters.items():
        if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        default = param.default
        annotation = "" if param.annotation is inspect.Parameter.empty else str(param.annotation)
        if isinstance(default, bool) or "bool" in annotation:
            parser.add_argument(f"--{name}", nargs="?", const="true", default=None)
        else:
            parser.add_argument(f"--{name}", default=None)

    args = vars(parser.parse_args())
    args.pop("_local_rank", None)
    kwargs = {
        name: _coerce_cli_value(value, signature.parameters[name])
        for name, value in args.items()
        if value is not None
    }
    # 与 YAML 装载同一套 bash 风格展开（${VAR} / ${VAR-default}）：
    # 单引号传入的 --save_path '${MY_REAL_NAME}/...' 也能得到一致语义。
    kwargs = _deep_expand(kwargs)
    return func(**kwargs)


class MicroStepMeanStats:
    """
    同一 global step 内按 micro batch 累加；每个指标各自维护 sum / count。
    accumulate 里没传的指标本步不更新；averages() 只返回本 step 内至少收到过一次样本的指标。
    构造时的名字仅用于预置键（reset 后会清空，之后仍可在 accumulate 里动态出现新键名）。
    """

    def __init__(self, *metric_names: str) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        for k in metric_names:
            self._sums[k] = 0.0
            self._counts[k] = 0

    def accumulate(self, **values: float) -> None:
        for k, v in values.items():
            if k not in self._sums:
                self._sums[k] = 0.0
                self._counts[k] = 0
            self._sums[k] += float(v)
            self._counts[k] += 1

    def averages(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts[k] > 0}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# set random seeds
def set_random_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CPTBinDataset(torch.utils.data.Dataset):
    """
    极速读取预编译的 .bin 文件。CPU 开销几乎为 0。
    """
    def __init__(self, bin_path: str, seq_len: int):
        # 魔法所在：memmap 不会把大文件一次性读进内存，而是按需在硬盘和内存间滑动
        self.data = np.memmap(bin_path, dtype=np.int32, mode='r')
        self.seq_len = seq_len
        self.chunk_size = seq_len + 1
        
        # 精确计算这个文件一共能切出多少个完整的 Batch
        self.total_chunks = len(self.data) // self.chunk_size

    def __len__(self):
        # 现在 Dataloader 终于知道你有多少数据了！
        return self.total_chunks

    def __getitem__(self, idx: int):
        # 直接用索引在极速数组上切片
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        
        # 截取数据并转为 int64 张量 (PyTorch Embedding 层的硬性要求)
        chunk = self.data[start_idx:end_idx]
        chunk_tensor = torch.from_numpy(chunk.astype(np.int64))
        
        # 返回 inputs 和 targets
        return chunk_tensor[:-1], chunk_tensor[1:]


class CPTOnlineJsonlDataset(torch.utils.data.Dataset):
    """
    支持在线 Tokenize 的 JSONL Dataset。
    在内部初始化 Tokenizer，并在 __getitem__ 时实时编码文本。
    """
    def __init__(self, jsonl_path: str, model_dir: str, seq_len: int, text_key: str = "content"):
        self.jsonl_path = jsonl_path
        self.seq_len = seq_len
        self.chunk_size = seq_len + 1  # 额外保留一位用于预测 target
        self.text_key = text_key
        
        print(f"⏳ [Dataset] 正在初始化 Tokenizer ({model_dir})...")
        # 直接在 Dataset 内部实例化 Tokenizer
        self.tokenizer = Tokenizer(model_dir)
        
        # 尝试获取 pad_id，如果你的 Tokenizer 没有定义 pad_id，默认使用 0
        self.pad_id = getattr(self.tokenizer, 'pad_id', 0)

        print(f"⏳ [Dataset] 正在扫描 {jsonl_path} 构建索引...")
        self.line_offsets = []
        with open(jsonl_path, 'rb') as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line)
                
        self.total_lines = len(self.line_offsets)
        print(f"✅ [Dataset] 索引构建完成，共发现 {self.total_lines} 条记录。")

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx: int):
        # 1. 读取单行
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            f.seek(self.line_offsets[idx])
            line_str = f.readline()
            line_data = json.loads(line_str)
            
        text = line_data.get(self.text_key, "")
        
        # 2. 实时 Tokenize (包含 EOS 结束符)
        tokens = self.tokenizer.encode(text, bos=False, eos=True)
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
            
        # 3. 截断 (Truncation)
        # 如果长度超过了所需的最大容量 (seq_len + 1)
        if len(tokens) > self.chunk_size:
            tokens = tokens[:self.chunk_size]
            
        # 4. 分离 Input 和 Label
        # 这里非常重要：先错位分离，再各自 Padding
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # 5. 独立填充 (Padding)
        if len(input_ids) < self.seq_len:
            pad_len = self.seq_len - len(input_ids)
            # Inputs 用正常的 pad_id 填充 (保证 Embedding 不报错)
            input_ids = input_ids + [self.pad_id] * pad_len
            # Labels 用 -100 填充 (让 CrossEntropyLoss 忽略这些位置的 Loss 计算)
            labels = labels + [-100] * pad_len
            
        # 6. 转换为 Tensor 并返回
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        return input_tensor, label_tensor
