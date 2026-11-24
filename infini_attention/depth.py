"""Break down official 1M single-needle scores by needle position in the actual retained prompt."""
import argparse
import json
import re
from pathlib import Path

from tokenizers import Tokenizer


NEEDLE = re.compile(r"One of the special magic (?:numbers|uuids) for \S+ is: (\S+)\.")
TASKS = ("niah_single_1", "niah_single_2", "niah_single_3")


def depth_report(report, tokenizer):
    requests = {}
    for row in report["inference_metrics"]["generate_until"]:
        fields = dict(part.split("=", 1) for part in row["sample_id"].split("|") if "=" in part)
        if fields.get("task") in TASKS:
            key = fields["task"], int(fields["doc"])
            if key in requests:
                raise ValueError(f"ambiguous generation log for {key}")
            requests[key] = row
    by_task = {}
    samples = []
    for task in TASKS:
        buckets = {name: {"n": 0, "correct": 0} for name in
                   ("first_third", "middle_third", "last_third", "first_10_pct", "needle_truncated")}
        for sample in report["results"]["samples"].get(task, []):
            doc = sample["doc"]
            if int(doc["max_length"]) != 1048576:
                continue
            prompt = sample["arguments"][0][0]
            matches = [m for m in NEEDLE.finditer(prompt) if m.group(1) in doc["outputs"]]
            if len(matches) != 1:
                raise ValueError(f"expected one target needle in {task}, doc {sample['doc_id']}")
            row = requests[task, sample["doc_id"]]
            encoded = tokenizer.encode(prompt)
            # Account for the harness inserting or removing a single initial BOS.
            bos_shift = row["original_prompt_len"] - len(encoded)
            if abs(bos_shift) > 1 or row["original_prompt_len"] - row["truncated_prompt_tokens"] != row["prompt_len"]:
                raise ValueError("tokenization disagrees with the recorded request length")
            pos = encoded.char_to_token(matches[0].start()) + bos_shift - row["truncated_prompt_tokens"]
            depth = pos / row["prompt_len"] if pos >= 0 else None
            names = ["needle_truncated"] if depth is None else [
                ("first_third", "middle_third", "last_third")[min(2, int(3 * depth))]]
            if depth is not None and depth < 0.1:
                names.append("first_10_pct")
            correct = sample["1048576"]
            if correct not in (0, 1):
                raise ValueError("expected a binary official single-needle score")
            for name in names:
                buckets[name]["n"] += 1
                buckets[name]["correct"] += int(correct)
            samples.append(dict(task=task, doc_id=sample["doc_id"], prompt_hash=sample["prompt_hash"],
                                filter=sample["filter"], prompt_len=row["prompt_len"], needle_token=pos,
                                depth=depth, correct=int(correct)))
        if any(b["n"] for b in buckets.values()):
            by_task[task] = buckets
    if not samples:
        raise ValueError("no 1M single-needle samples in this result file")
    return {"by_task": by_task, "samples": samples}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("result", type=Path)
    cli.add_argument("--tokenizer", type=Path, help="directory containing the exact evaluation tokenizer")
    args = cli.parse_args()
    report = json.loads(args.result.read_text())
    folder = args.tokenizer or Path(report["tokenizer_dir"])
    print(json.dumps(depth_report(report, Tokenizer.from_file(str(folder / "tokenizer.json"))), indent=2))


if __name__ == "__main__":
    main()
