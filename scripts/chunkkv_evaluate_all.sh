#!/usr/bin/env bash
# SemanticLogKV task lists: main benchmarks, then 32K-only NIAH.
set -euo pipefail
cd "$(dirname "$0")/.."
for arg in "$@"; do
    case "$arg" in
        --tasks|--tasks=*|--suite|--suite=*)
            echo "Use scripts/chunkkv_evaluate.sh for a selected --tasks or --suite group." >&2
            exit 2
            ;;
        -h|--help)
            echo "Usage: bash scripts/chunkkv_evaluate_all.sh [--backend chunkkv|dense|logkv] [--model DIR] [--limit N] [--output ROOT] ..."
            echo "Runs main benchmarks and NIAH separately, writing ROOT/main and ROOT/niah."
            exit 0
            ;;
    esac
done
bash scripts/chunkkv_evaluate.sh --suite main "$@"
bash scripts/chunkkv_evaluate.sh --suite niah "$@"
