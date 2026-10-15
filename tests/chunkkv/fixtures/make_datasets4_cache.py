"""Regenerate datasets4_cache/: offline cache entries written by datasets>=4 (synthetic rows).

Run with a datasets>=4 interpreter; IFEval's `Json` column needs datasets>=4.7. Entries use the
layout `load_dataset` reads offline: <namespace>___<name>/<config>/0.0.0/<hash>/.
"""

import json
from pathlib import Path
import shutil
import sys

import datasets
from datasets import ClassLabel, Features, List, Value
from datasets.arrow_writer import ArrowWriter

ROOT = Path(__file__).resolve().parent / "datasets4_cache"
TARGETS = {"choices": List(Value("string")), "labels": List(Value("int32"))}
ENTRIES = {
    ("truthfulqa___truthful_qa", "generation", "validation"): (
        Features({"type": Value("string"), "question": Value("string"), "best_answer": Value("string"),
                  "correct_answers": List(Value("string")), "incorrect_answers": List(Value("string"))}),
        [{"type": "Adversarial", "question": "q1", "best_answer": "a1",
          "correct_answers": ["a1", "a1b"], "incorrect_answers": ["x1"]},
         {"type": "Non-Adversarial", "question": "q2", "best_answer": "a2",
          "correct_answers": ["a2"], "incorrect_answers": []}]),
    ("truthfulqa___truthful_qa", "multiple_choice", "validation"): (
        Features({"question": Value("string"), "mc1_targets": TARGETS, "mc2_targets": TARGETS}),
        [{"question": "q1", "mc1_targets": {"choices": ["a", "b"], "labels": [1, 0]},
          "mc2_targets": {"choices": ["a", "c", "b"], "labels": [1, 1, 0]}}]),
    ("example___structs", "default", "train"): (
        Features({"spans": List({"text": Value("string"), "start": Value("int32")}),
                  "pair": List(Value("float32"), length=2), "nested": List(List(Value("int64"))),
                  "label": ClassLabel(names=["no", "yes"])}),
        [{"spans": [{"text": "t", "start": 1}], "pair": [1.0, 2.0], "nested": [[1, 2], [3]], "label": 1},
         {"spans": [], "pair": [3.0, 4.0], "nested": [[]], "label": 0}]),
}
if hasattr(datasets, "Json"):
    ENTRIES[("google___if_eval", "default", "train")] = (
        Features({"prompt": Value("string"), "kwargs": List(datasets.Json())}),
        [{"prompt": "p", "kwargs": [{"num_words": 3}, {"language": "en"}]}])

if int(datasets.__version__.split(".")[0]) < 4:
    sys.exit("Run with datasets>=4")
for index, ((directory, config, split), (features, rows)) in enumerate(sorted(ENTRIES.items())):
    entry = ROOT / directory / config / "0.0.0" / (f"{index + 1:x}" * 40)
    shutil.rmtree(entry, ignore_errors=True)
    entry.mkdir(parents=True)
    name = directory.split("___")[1]
    with ArrowWriter(features=features, path=str(entry / f"{name}-{split}.arrow")) as writer:
        for row in rows:
            writer.write(row)
        examples, size = writer.finalize()
    splits = datasets.SplitDict(dataset_name=name)
    splits.add(datasets.SplitInfo(name=split, num_bytes=size, num_examples=examples, dataset_name=name))
    datasets.DatasetInfo(features=features, dataset_name=name, config_name=config,
                         splits=splits).write_to_directory(str(entry))
    print(entry.relative_to(ROOT), json.dumps(features.to_dict()))
(ROOT / "WRITTEN_BY").write_text(f"datasets {datasets.__version__}\n")
