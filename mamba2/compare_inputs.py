"""Check the actual lm-eval inputs in a matching pair of LogKV/Mamba-2 stage JSON files."""
import argparse
import hashlib
import json
from pathlib import Path


def sample_signatures(result):
    samples = result["results"].get("samples")
    if not samples:
        raise ValueError("Missing lm-eval samples")
    signatures = {}
    for task, rows in samples.items():
        if not rows:
            raise ValueError(f"Empty task {task}")
        items = []
        for row in rows:
            inputs = [row[k] for k in ("doc_id", "doc_hash", "prompt_hash", "target_hash", "arguments")]
            items.append(hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
        signatures[task] = sorted(items)
    return signatures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    args = p.parse_args()
    def read(path):
        with path.open(encoding="utf-8") as stream:
            return sample_signatures(json.load(stream))
    reference, candidate = read(args.reference), read(args.candidate)
    for task in sorted(reference.keys() | candidate.keys()):
        if reference.get(task) != candidate.get(task):
            raise SystemExit(f"Task, sample, target, prompt or generation arguments differ for {task}")
    print(f"Identical logged inputs for {len(reference)} tasks and {sum(map(len, reference.values()))} samples.")


if __name__ == "__main__":
    main()
