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
    fields = CONTRACT_FIELDS
    if a.get("protocol") == b.get("protocol") == "niahLogKV":
        role_pair = {(result.get("model_backend"), (contract.get("evaluated_model") or {}).get("weight_role"))
                     for result, contract in ((left, a), (right, b))}
        # Only the default Base-vs-CPT comparison intentionally changes weights.
        # Compare the resolved model configuration across differently serialized checkpoint YAMLs.
        if role_pair in ({("chunkkv", "base"), ("logkv", "cpt")}, {("chunkkv", "base"), ("dense", "cpt")}):
            if not a.get("effective_model_config") or a["effective_model_config"] != b.get("effective_model_config"):
                raise ValueError("Comparison contract differs: effective_model_config")
            fields = tuple(k for k in fields if k not in ("source_weight_sha256", "source_config_files"))
    differences = [name for name in fields if a[name] != b[name]]
    if a.get("protocol") != b.get("protocol") or a.get("rope_adjustments") != b.get("rope_adjustments"):
        differences.append("protocol/rope_adjustments")
    if differences:
        raise ValueError("Comparison contract differs: " + ", ".join(differences))
    first, second = sample_signatures(left), sample_signatures(right)
    if first != second:
        raise ValueError("Actual test samples, prompts, targets or generation parameters differ")
    return {task: len(rows) for task, rows in first.items()}


def budget_summary(result):
    summary = (result.get("kv_compression") or {}).get("summary") or {}
    model = result["comparison_contract"].get("evaluated_model") or {}
    return {
        "model_backend": result.get("model_backend"),
        "checkpoint": model.get("checkpoint_dir"),
        "weight_role": model.get("weight_role"),
        "weight_sha256": model.get("lit_model_sha256"),
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
    print("Verified matching evaluation protocol and actual test inputs:")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print("KV budgets (request-end snapshots; compare them before comparing scores):")
    print(json.dumps([budget_summary(left), budget_summary(right)], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
