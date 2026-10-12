"""Verify that two shared-evaluator result files used the same comparison inputs."""

import argparse
import hashlib
import json
from pathlib import Path


def sample_signatures(result):
    samples = result["results"].get("samples")
    if not samples:
        raise ValueError("Result has no logged samples; cannot verify actual test inputs")
    signatures = {}
    for task, rows in samples.items():
        if not rows:
            raise ValueError(f"No samples for {task}")
        # Arguments include the exact prompt/continuation and generation settings.
        records = [{key: row[key] for key in
                    ("doc_id", "doc_hash", "prompt_hash", "target_hash", "arguments")} for row in rows]
        signatures[task] = sorted(hashlib.sha256(json.dumps(row, sort_keys=True,
                                  ensure_ascii=False).encode()).hexdigest() for row in records)
    return signatures


def compare_results(left, right):
    a, b = left.get("comparison_contract"), right.get("comparison_contract")
    if not a or not b:
        raise ValueError("Both results must be produced by the aligned evaluation launcher")
    fields = ("source_weight_sha256", "source_config_files", "tokenizer_files", "evaluation_code", "compatibility_code", "versions",
              "cache", "benchmark", "metadata", "limit", "max_seq_length")
    differences = [name for name in fields if a[name] != b[name]]
    if differences:
        raise ValueError("Comparison contract differs: " + ", ".join(differences))
    first, second = sample_signatures(left), sample_signatures(right)
    if first != second:
        raise ValueError("Actual test samples, prompts, targets or generation parameters differ")
    return {task: len(rows) for task, rows in first.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    counts = compare_results(json.loads(args.left.read_text()), json.loads(args.right.read_text()))
    print("Verified identical Base source, tokenizer, evaluator, runtime and actual test inputs:")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
