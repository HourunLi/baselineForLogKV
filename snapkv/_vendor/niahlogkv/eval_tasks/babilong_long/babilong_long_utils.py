"""BABILong qa1-qa5 at a fixed context length (lm-eval's babilong tasks take one length per run from
--metadata and do not name it in their results; these name it in the task, e.g. babilong_qa1_64k).

The data is RMT-team/babilong-1k-samples (one config per length: 0k ... 128k, splits qa1 ... qa5);
BABILONG_DATASET may point at a local copy of it instead. Longer tasks name their own dataset in
dataset_kwargs (256k: RMT-team/babilong, 100 samples per length, up to 10M tokens by lm-eval's README),
which BABILONG_DATASET does not replace.
"""
import os
import json
import re
from pathlib import Path

import datasets


def process_results(doc, results):
    """Same case-insensitive substring metric as lm-eval's babilong/common_utils.py."""
    prediction = re.sub(r"[\x00-\x1f]", "\n", results[0].strip()).strip()
    return {"acc": float(doc.get("target", "").strip().lower() in prediction.lower())}


def plain_folder():
    home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return Path(os.environ.get("BABILONG_DIR") or home / "babilong").expanduser()


def plain_paths(length, split):
    name = "1M" if str(length).lower() == "1m" else str(length).lower()
    base = plain_folder() / "data" / split / name
    return base.with_suffix(".arrow"), base.with_suffix(".json")


def load_plain(length, split):
    """Memory-map prepared ordinary Arrow files; manual ordinary JSON copies also work without a builder."""
    arrow, source = plain_paths(length, split)
    path = arrow if arrow.exists() or arrow.is_symlink() else source
    if path.is_symlink():
        raise ValueError(f"BABILong requires an ordinary file, not a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(
            f"BABILong {length}/{split} missing in BABILONG_DIR={plain_folder()}. Run "
            "bash diag.sh unused/fetch_long_eval_data.py --lengths 512k,1m --no-ruler-qa, "
            "with the evaluation job's HF_HOME and BABILONG_DIR.")
    try:
        if path == arrow:
            data = datasets.Dataset.from_file(str(path), split=split, in_memory=False)
        else:
            with path.open(encoding="utf-8") as stream:
                data = datasets.Dataset.from_list(json.load(stream), split=split)
        if not len(data) or not {"input", "question", "target"} <= set(data.column_names):
            raise ValueError("empty data or missing input/question/target columns")
        return data
    except Exception as error:
        raise ValueError(f"BABILong {length}/{split}: unreadable ordinary data file {path}: {error}") from error


def load_dataset(**kwargs):
    length, split = kwargs["length"], kwargs["qa_split"]
    arrow, source = plain_paths(length, split)
    if str(length).lower() in ("512k", "1m") or arrow.exists() or source.exists():
        return {split: load_plain(length, split)}
    path = kwargs.get("dataset") or os.environ.get("BABILONG_DATASET") or "RMT-team/babilong-1k-samples"
    return {split: datasets.load_dataset(path, name=length, split=split)}
