"""BABILong qa1-qa5 at a fixed context length (lm-eval's babilong tasks take one length per run from
--metadata and do not name it in their results; these name it in the task, e.g. babilong_qa1_64k).

The data is RMT-team/babilong-1k-samples (one config per length: 0k ... 128k, splits qa1 ... qa5).
BABILONG_DATASET may point at a local copy of that dataset instead.
"""
import os

import datasets
from lm_eval.tasks.babilong.common_utils import process_results  # noqa: F401  (the same string match)


def load_dataset(**kwargs):
    length, split = kwargs["length"], kwargs["qa_split"]
    path = os.environ.get("BABILONG_DATASET") or "RMT-team/babilong-1k-samples"
    return {split: datasets.load_dataset(path, name=length, split=split)}
