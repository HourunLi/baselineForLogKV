"""Export the original lm-eval scores and resource summaries from either evaluation JSON."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


FIELDS = ("run", "section", "task", "metric", "filter", "value", "stderr")


def quality_rows(results):
    # lm-eval also includes aggregated groups in "results". Emit those once.
    groups = results.get("groups") or {}
    for section, tasks in (("task", results["results"]), ("group", groups)):
        for task, scores in tasks.items():
            if section == "task" and task in groups:
                continue
            for key, value in scores.items():
                metric, sep, filt = key.partition(",")
                if not sep or metric.endswith("_stderr"):
                    continue
                yield dict(section=section, task=task, metric=metric, filter=filt, value=value,
                           stderr=scores.get(f"{metric}_stderr,{filt}"))


def summary_rows(run):
    yield from quality_rows(run["results"])
    for task, samples in (run["results"].get("samples") or {}).items():
        yield dict(section="inputs", task=task, metric="logged_rows", value=len(samples))
        fields = ("doc_id", "filter", "doc_hash", "prompt_hash", "target_hash")
        if samples and all(all(row.get(key) is not None for key in fields) for row in samples):
            # The same texts may be split into different tokens or truncated differently by a model.
            rows = sorted(tuple(row[key] for key in fields) for row in samples)
            digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
            yield dict(section="inputs", task=task, metric="text_set_sha256", value=digest)
    for section, metrics in (("cache_budget", run["cache_budget"]),
                             ("compression", run["kv_compression"]["summary"]),
                             ("generation", run["inference_metrics"]["generation"]),
                             ("scoring", run["inference_metrics"]["scoring"])):
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) or value is None:
                yield dict(section=section, metric=metric, value=value)


def write_summary(stream, name, run, *, header=True):
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    if header:
        writer.writeheader()
    for row in summary_rows(run):
        writer.writerow({"run": name, **row})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="Full eval_results_<timestamp>.json files")
    args = parser.parse_args()
    for index, path in enumerate(args.results):
        with path.open(encoding="utf-8") as stream:
            write_summary(sys.stdout, str(path), json.load(stream), header=index == 0)


if __name__ == "__main__":
    main()
