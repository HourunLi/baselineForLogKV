"""Check that result files used identical test inputs, then tabulate them side by side.

Works on this baseline's results and on SemanticLogKV ``eval_results_*.json``
files alike: both keep lm-eval's per-sample log (doc/prompt/target hashes and
the exact request arguments) plus ``kv_compression`` and ``inference_metrics``.

    python -m pyramidkv.compare_results semantic.json pyramidkv.json [h2o.json ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SAMPLE_KEYS = ("doc_id", "doc_hash", "prompt_hash", "target_hash", "arguments")


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
    for key in ("metadata",):
        values = [json.dumps(result.get(key), sort_keys=True) for result in results]
        if len(set(values)) > 1:
            print(f"⚠️ {key} differs between files (inputs still verified identical): {values}")
    return {task: len(rows) for task, rows in first.items()}


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
        print("Verified identical test inputs (samples, prompts, targets, generation arguments):")
        print(json.dumps(counts, ensure_ascii=False))
    rows = [summary_rows(result) for result in loaded]
    labels = [_label(result, path) for result, path in zip(loaded, args.results)]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    width = max(len(key) for key in keys) if keys else 10
    print("\n" + "metric".ljust(width) + " | " + " | ".join(labels))
    for key in keys:
        print(key.ljust(width) + " | " + " | ".join(_fmt(row.get(key)) for row in rows))


if __name__ == "__main__":
    main()
