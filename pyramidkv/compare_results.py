"""Check that result files used the same evaluation protocol and test inputs, then tabulate them.

Works on this baseline's results and on SemanticLogKV ``eval_results_*.json``
files alike: both keep lm-eval's per-sample log (doc/prompt/target hashes and
the exact request arguments) plus ``kv_compression`` and ``inference_metrics``.
Protocol fields (samples, task versions and settings, limit, window, tokenizer,
package versions) must agree; method fields (backend, budget, weights) are
reported separately and are expected to differ.

    python -m pyramidkv.compare_results semantic.json pyramidkv.json [h2o.json ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SAMPLE_KEYS = ("doc_id", "doc_hash", "prompt_hash", "target_hash", "arguments")
# Task settings that change prompts, generation or scoring; ``metadata`` only names caches.
TASK_CONFIG_KEYS = ("output_type", "num_fewshot", "generation_kwargs", "target_delimiter",
                    "fewshot_delimiter", "repeats")
# Package versions that define tasks, data, tokenization or numerics; lm-eval's version decides the task
# definitions (e.g. LongBench differs between 0.4.9 and 0.4.11), so results from different versions never compare.
VERSION_FIELDS = ("torch", "transformers", "datasets", "pyarrow", "lm-eval", "tokenizers")


def _recorded_version(name: str):
    return lambda result: ((result.get("provenance") or {}).get("versions") or {}).get(name)


# Recorded by this project's results only; SemanticLogKV's own files leave them unverified.
PROVENANCE_FIELDS = {
    "max_seq_length": lambda result: result.get("max_seq_length"),
    "tokenizer_files": lambda result: (result.get("provenance") or {}).get("tokenizer_files"),
    **{name: _recorded_version(name) for name in VERSION_FIELDS},
}


def sample_signatures(result: dict) -> dict[str, list[str]]:
    samples = (result.get("results") or {}).get("samples")
    if not samples:
        raise ValueError("Result has no logged samples; cannot verify the actual test inputs")
    signatures = {}
    for task, rows in samples.items():
        if not rows:
            raise ValueError(f"No samples logged for {task}")
        missing = [key for key in SAMPLE_KEYS if key not in rows[0]]
        if missing:
            raise ValueError(f"{task} samples lack {missing}")
        signatures[task] = sorted(
            hashlib.sha256(json.dumps({key: row[key] for key in SAMPLE_KEYS}, sort_keys=True,
                                      ensure_ascii=False).encode()).hexdigest()
            for row in rows)
    return signatures


def protocol_record(result: dict) -> dict:
    """Harness-level settings both kinds of result files carry inside lm-eval's ``results``."""
    harness = result.get("results") or {}
    configs = harness.get("configs") or {}
    return {
        "limit": (harness.get("config") or {}).get("limit"),
        "n-samples": harness.get("n-samples"),
        "task_versions": harness.get("versions"),
        "task_configs": {task: {key: config.get(key) for key in TASK_CONFIG_KEYS}
                         for task, config in sorted(configs.items())},
    }


def protocol_differences(results: list[dict]) -> tuple[list[str], list[str]]:
    """Return (fields that differ, fields not recorded in every file)."""
    differing, unverified = [], []
    records = [protocol_record(result) for result in results]
    for key in records[0]:
        values = [json.dumps(record[key], sort_keys=True, default=str) for record in records]
        if len(set(values)) > 1:
            differing.append(key)
    for key, read in PROVENANCE_FIELDS.items():
        values = [read(result) for result in results]
        if any(value is None for value in values):
            unverified.append(key)
        elif len({json.dumps(value, sort_keys=True) for value in values}) > 1:
            differing.append(key)
    return differing, unverified


def verify_same_inputs(results: list[dict]) -> dict[str, int]:
    """Raise unless every file evaluated the same tasks on the same prompts/targets/arguments."""
    first = sample_signatures(results[0])
    for index, other in enumerate(results[1:], start=1):
        current = sample_signatures(other)
        if set(current) != set(first):
            raise ValueError(f"File {index} evaluated tasks {sorted(current)}, expected {sorted(first)}")
        for task in first:
            if current[task] != first[task]:
                raise ValueError(f"File {index}: {task} samples, prompts, targets or generation "
                                 "arguments differ from file 0")
    differing, unverified = protocol_differences(results)
    if differing:
        raise ValueError("Evaluation protocol differs between files: " + ", ".join(differing))
    if unverified:
        print("⚠️ protocol fields not recorded in every file (not verified): " + ", ".join(unverified))
    for key in ("metadata",):
        values = [json.dumps(result.get(key), sort_keys=True) for result in results]
        if len(set(values)) > 1:
            print(f"⚠️ {key} differs between files (inputs still verified identical): {values}")
    return {task: len(rows) for task, rows in first.items()}


def method_record(result: dict) -> dict:
    """Fields expected to differ between methods; shown, never used to accept a comparison."""
    provenance = result.get("provenance") or {}
    return {"model_backend": result.get("model_backend"), "checkpoint_dir": result.get("checkpoint_dir"),
            "weights_sha256": provenance.get("weights_sha256"), "method_config": result.get("method_config"),
            "cache_budget": result.get("cache_budget")}


def _label(result: dict, path: Path) -> str:
    backend = result.get("model_backend")
    if backend is None:  # SemanticLogKV's own evaluator
        backend = ("dense" if result.get("log_kv_dense_mode") else
                   f"swa{result['swa_window_size']}" if result.get("swa_window_size") is not None else "logkv")
    config = result.get("method_config") or {}
    budget = config.get("max_capacity_prompt")
    return f"{backend}" + (f"@{budget}" if budget else "") + f" ({path.name})"


def summary_rows(result: dict) -> dict[str, object]:
    row = {}
    for task, metrics in sorted(((result.get("results") or {}).get("results") or {}).items()):
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and "_stderr" not in key:
                row[f"{task}/{key}"] = value
    compression = (result.get("kv_compression") or {}).get("summary") or {}
    for key in ("kv_retention_ratio", "kv_compression_factor", "retained_slots_per_layer_group"):
        row[f"kv/{key}"] = compression.get(key)
    generation = (result.get("inference_metrics") or {}).get("generation") or {}
    for key in ("avg_ttft_ms", "prefill_tokens_per_sec", "decode_ms_per_token", "peak_memory_allocated_bytes"):
        row[f"speed/{key}"] = generation.get(key)
    return row


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--skip-input-check", action="store_true", help="Only print the table")
    args = parser.parse_args(argv)
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    if len(loaded) > 1 and not args.skip_input_check:
        counts = verify_same_inputs(loaded)
        print("Verified identical test inputs (samples, prompts, targets, generation arguments) "
              "and evaluation protocol:")
        print(json.dumps(counts, ensure_ascii=False))
        print("Method differences (expected; not part of the protocol check):")
        print(json.dumps([method_record(result) for result in loaded], ensure_ascii=False, indent=2, default=str))
    rows = [summary_rows(result) for result in loaded]
    labels = [_label(result, path) for result, path in zip(loaded, args.results)]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    width = max(len(key) for key in keys) if keys else 10
    print("\n" + "metric".ljust(width) + " | " + " | ".join(labels))
    for key in keys:
        print(key.ljust(width) + " | " + " | ".join(_fmt(row.get(key)) for row in rows))


if __name__ == "__main__":
    main()
