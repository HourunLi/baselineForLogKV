"""
parquet / jsonl / json → LitData 流式 chunks（litdata.optimize）。

``data_dir`` 下 ``rglob`` 递归收集 ``*.parquet`` / ``*.jsonl`` / ``*.json``（``.json`` 常为 NDJSON：每行一个 dict），一次写入同一 litdata 目录。
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import partial
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset

from litgpt.tokenizer import Tokenizer

DEFAULT_CHUNK_BYTES = "200MB"


def tokenizer_cache_key(tokenizer_dir: str) -> str:
    """tokenizer 文件内容哈希短 id。"""
    p = Path(os.path.normpath(tokenizer_dir)).resolve()
    h = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
        f = p / name
        if f.is_file():
            h.update(name.encode())
            h.update(f.read_bytes())
    if not (p / "tokenizer.json").is_file() and not (p / "tokenizer.model").is_file():
        raise FileNotFoundError(
            f"在 {tokenizer_dir} 下未找到 tokenizer.json 或 tokenizer.model，无法确定 tokenizer 身份。"
        )
    return h.hexdigest()[:16]


def litdata_chunks_dir(
    tokenizer_dir: str, data_dir: str | None, dataset_dir: str, context_length: int
) -> str:
    """LitData optimize 输出目录（与训练共用；含 context_length）。"""
    key = tokenizer_cache_key(tokenizer_dir)
    root = data_dir if data_dir else dataset_dir
    return os.path.join(root, f"litdata_{key}_ctx{context_length}")


def _parquet_format_for_source_paths(paths: list[str]) -> str:
    """路径含 ``tulu`` → ``messages``；``textbookchapters`` → 正文列在 tokenize 里按 schema 选；否则 ``text`` 列。"""
    if paths and any("tulu" in os.path.normpath(p).lower() for p in paths):
        return "tulu_messages"
    if paths and any("textbookchapters" in os.path.normpath(p).lower() for p in paths):
        return "chapter"
    return "text"


def _parquet_body_column(pf: pq.ParquetFile) -> str:
    """TextbookChapters 等语料正文列多为 ``text``，少数为 ``chapter`` / ``content``。"""
    names = {f.name for f in pf.schema_arrow}
    for c in ("text", "chapter", "content"):
        if c in names:
            return c
    raise ValueError(f"parquet 缺少正文列 text/chapter/content，实际列: {sorted(names)}")


def _collect_data_files(data_dir: str) -> list[str]:
    """递归收集 ``*.parquet`` / ``*.jsonl`` / ``*.json``。"""
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data_dir 不是目录: {data_dir}")
    files = list(root.rglob("*.parquet")) + list(root.rglob("*.jsonl")) + list(root.rglob("*.json"))
    return sorted({str(p) for p in files})


def _iter_json_dicts(path: str):
    """``.json`` 多为 NDJSON（每行一个完整 dict）；仅当以 ``[`` 开头时才整文件解析 JSON 数组。"""
    with open(path, encoding="utf-8") as f:
        head = ""
        for line in f:
            s = line.strip()
            if s:
                head = s
                break
        else:
            return
        if head[0] == "[":
            f.seek(0)
            data = json.loads(f.read())
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
            return
        f.seek(0)
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError:
                continue


def tulu_messages_to_text(messages: object) -> str | None:
    """``messages`` → 单段文本（``### role`` 分段）。"""
    if messages is None:
        return None
    if isinstance(messages, float) and pd.isna(messages):
        return None
    if not messages:
        return None
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user"))
        content = m.get("content")
        if content is None:
            continue
        text = str(content).strip()
        if not text:
            continue
        parts.append(f"### {role}\n{text}")
    if not parts:
        return None
    return "\n\n".join(parts)


def tokenize_source_file(path: str, tokenizer: Tokenizer, parquet_format: str | None = None):
    """
    供 litdata.optimize 调用：每个样本一条序列（bos=False, eos=True）。

    Parquet：``text`` 模式固定读 ``text``；``chapter`` 模式按文件 schema 选 ``text`` / ``chapter`` / ``content``。
    ``parquet_format=None``（默认）时按**当前文件自身路径**逐文件推断格式。
    混合语料（如 tulu + 普通 text parquet 同在一个 data_dir）必须逐文件推断：
    若用所有路径统一决定一个格式，普通 parquet 会被误按 ``messages`` 列读取。
    json/jsonl：``content`` / ``text`` / ``code``。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        if parquet_format is None:
            parquet_format = _parquet_format_for_source_paths([path])
        pf = pq.ParquetFile(path)
        if parquet_format == "tulu_messages":
            col = "messages"
        elif parquet_format == "chapter":
            col = _parquet_body_column(pf)
        elif parquet_format == "text":
            col = "text"
        else:
            raise ValueError(
                f"未知 parquet_format: {parquet_format!r}（支持 text、chapter、tulu_messages）"
            )
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=[col])
            pc = table.column(col)
            for i in range(len(pc)):
                raw = pc[i].as_py()
                if parquet_format == "tulu_messages":
                    text = tulu_messages_to_text(raw)
                else:
                    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                        continue
                    text = str(raw).strip()
                if not text:
                    continue
                yield tokenizer.encode(text, bos=False, eos=True)
    elif ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        continue
                    text = (obj.get("content") or obj.get("text") or obj.get("code") or "")
                    text = str(text).strip()
                    if not text:
                        continue
                    yield tokenizer.encode(text, bos=False, eos=True)
                except json.JSONDecodeError:
                    continue
    elif ext == ".json":
        for obj in _iter_json_dicts(path):
            text = (obj.get("content") or obj.get("text") or obj.get("code") or "")
            text = str(text).strip()
            if not text:
                continue
            yield tokenizer.encode(text, bos=False, eos=True)
    else:
        raise ValueError(f"不支持的文件类型: {path}")


def _validate_tokenizer(tokenizer: Tokenizer | None) -> None:
    if tokenizer is None:
        raise ValueError("Tokenizer 为 None，请传入有效的 tokenizer_dir。")


def prepare_tokenized_data(
    dataset_dir: str,
    dataset_name: str,
    tokenizer_dir: str,
    data_dir: str | None = None,
    context_length: int = 4096,
    chunk_bytes: str = DEFAULT_CHUNK_BYTES,
    num_workers: int | None = None,
) -> str:
    """
    使用 litdata.optimize 将数据编译为流式 chunks。

    ``context_length`` 为训练序列长度；内部 ``TokensLoader(block_size=context_length + 1)`` 与 LitGPT 一致。
    返回 optimize 输出目录路径。
    """
    from litdata import optimize
    from litdata.streaming import TokensLoader

    os.makedirs(dataset_dir, exist_ok=True)
    out_dir = litdata_chunks_dir(tokenizer_dir, data_dir, dataset_dir, context_length)
    block_size = context_length + 1

    all_files: list[str] = []
    if data_dir and os.path.isdir(data_dir):
        all_files = _collect_data_files(data_dir)
        if not all_files:
            raise FileNotFoundError(f"在 {data_dir} 下（递归）没有找到 .parquet / .jsonl / .json。")
    else:
        single_parquet = os.path.join(dataset_dir, f"{dataset_name}.parquet")
        if not os.path.exists(single_parquet):
            if dataset_name == "debug":
                print("🌐 正在拉取 FineWeb-Edu (sample)...")
                eng_stream = load_dataset(
                    "HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True
                )
                mixed_dataset = Dataset.from_list(list(eng_stream.take(10000))).select_columns(["text"])
                mixed_dataset.to_parquet(single_parquet)
            else:
                raise NotImplementedError(f"数据集 {dataset_name} 尚未实现")
        all_files = [single_parquet]

    tokenizer = Tokenizer(tokenizer_dir)
    _validate_tokenizer(tokenizer)

    max_workers = num_workers if num_workers is not None else max(1, (os.cpu_count() or 1) - 1)
    use_workers = min(max_workers, len(all_files)) if all_files else 1

    if os.path.isdir(out_dir) and any(os.scandir(out_dir)):
        print(
            f"⚠️ 已存在 LitData 缓存: {out_dir}\n"
            "若需更换语料或 tokenizer，请删除该目录后重跑。"
        )
        return out_dir

    os.makedirs(out_dir, exist_ok=True)
    print(f"🚀 litdata.optimize → {out_dir}（chunk_bytes={chunk_bytes}, workers={use_workers}）")

    optimize(
        # parquet_format=None → tokenize_source_file 按每个文件自身路径推断格式，
        # 避免混合语料（tulu + 普通 text parquet）被统一成错误 schema。
        fn=partial(tokenize_source_file, tokenizer=tokenizer),
        inputs=all_files,
        output_dir=out_dir,
        num_workers=use_workers,
        chunk_bytes=chunk_bytes,
        item_loader=TokensLoader(block_size=block_size),
    )
    print("✅ optimize 完成。")
    return out_dir


def main(
    tokenizer_dir: str = "checkpoints/Qwen/Qwen3-0.6B-Base",
    dataset_name: str = "debug",
    dataset_dir: str = "data",
    data_dir: str = "data",
    context_length: int = 4096,
    chunk_bytes: str = DEFAULT_CHUNK_BYTES,
    num_workers: int | None = None,
):
    """离线预处理：python data.py --tokenizer_dir ..."""
    out = prepare_tokenized_data(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        tokenizer_dir=tokenizer_dir,
        data_dir=data_dir,
        context_length=context_length,
        chunk_bytes=chunk_bytes,
        num_workers=num_workers,
    )
    print(f"✅ 输出目录: {out}")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
