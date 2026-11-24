"""Exercise the offline data loaders for the evaluation passes requested by eval.sh."""
import json
import os
import sys

from lla_bench.eval import _task_manager
from lla_bench.unused import check_long_eval_data as checks
from .long_eval_plan import LENGTHS


def main(main_tasks, passes):
    problems = []

    def report(ok, message):
        print(f"{'OK' if ok else 'MISSING'} {message}", flush=True)
        if not ok:
            problems.append(message)

    base = os.environ["NFINI_QWEN_BASE"]
    lengths = {}
    manager = _task_manager({"pretrained": base, "tokenizer": base})
    for task in main_tasks.split(","):
        if task == "long_ppl_books_32k":
            lengths.setdefault("books", set()).add("32k")
        elif task not in ("none", "niah_turns_32k"):
            try:
                report(bool(manager.load_task_or_group(task)), f"main {task}")
            except Exception as e:
                report(False, f"main {task}: {e}")

    length_names = {n: name for name, n in LENGTHS.items()}
    for line in passes:
        _, _, window, tasks, needles, _ = line.split("|")
        length = length_names[json.loads(window)[0]]
        for task in tasks.split(","):
            family = task.rsplit("_", 1)[0]
            suite = {"long_ppl": "ppl", "long_ppl_books": "books", "niah_turns": "turns"}.get(family, family)
            if suite != "none":
                lengths.setdefault(suite, set()).add(length)
        if needles != "none":
            lengths.setdefault("ruler", set()).add(length)

    if "babilong" in lengths:
        checks.check_babilong(report, sorted(lengths["babilong"]))
    if "longbench2" in lengths:
        checks.check_longbench2(report)
    extended = {"512k", "1m"}
    if "ruler" in lengths or "turns" in lengths:
        checks.check_ruler_qa(report, bool((lengths.get("ruler", set()) | lengths.get("turns", set())) & extended))
    if lengths.get("ruler", set()) & extended:
        checks.check_ruler_cwe(report)
    if "infinitebench" in lengths:
        checks.check_infinitebench(report)
    for long_group in (False, True):
        if long_group:
            for key in ("DATA", "BOOKS", "DOCS"):
                os.environ[f"LONG_PPL_{key}"] = os.environ[f"NFINI_LONG_PPL_{key}_1M"]
        for suite in ("ppl", "books"):
            selected = sorted(n for n in lengths.get(suite, ()) if (n in extended) == long_group)
            if selected:
                checks.check_ppl(report, base, selected, [suite], require_book_count=False)
    if problems:
        raise SystemExit(f"{len(problems)} offline evaluation data checks failed")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
