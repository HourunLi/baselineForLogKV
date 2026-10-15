"""Verify that two aligned result files used the same inputs, then show their KV budgets."""

import argparse
import hashlib
import json
from pathlib import Path


CONTRACT_FIELDS = ("source_weight_sha256", "source_config_files", "tokenizer_files", "evaluation_code",
                   "compatibility_code", "versions", "cache", "benchmark", "metadata", "limit", "max_seq_length")


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
    differences = [name for name in CONTRACT_FIELDS if a[name] != b[name]]
    if differences:
        raise ValueError("Comparison contract differs: " + ", ".join(differences))
    first, second = sample_signatures(left), sample_signatures(right)
    if first != second:
        raise ValueError("Actual test samples, prompts, targets or generation parameters differ")
    return {task: len(rows) for task, rows in first.items()}


def budget_summary(result):
    summary = (result.get("kv_compression") or {}).get("summary") or {}
    return {
        "model_backend": result.get("model_backend"),
        "checkpoint": (result["comparison_contract"].get("evaluated_model") or {}).get("checkpoint_dir"),
        "kv_retention_ratio": summary.get("kv_retention_ratio"),
        "retained_slots_per_layer_group": summary.get("retained_slots_per_layer_group"),
        "cache_budget": result.get("cache_budget"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    left, right = (json.loads(path.read_text()) for path in (args.left, args.right))
    counts = compare_results(left, right)
    print("Protocol check passed: identical Base source, tokenizer, evaluator, runtime and actual test inputs:")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print("Method differences (expected; not part of the protocol check). KV budgets are request-end")
    print("snapshots; compare them before comparing scores:")
    print(json.dumps([budget_summary(left), budget_summary(right)], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
