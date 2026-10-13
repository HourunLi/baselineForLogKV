#!/usr/bin/env bash
# Usage: bash scripts/chunkkv_evaluate.sh [--backend chunkkv|logkv|dense] [--model DIR] [--limit 4] ...
set -euo pipefail
cd "$(dirname "$0")/.."
torchrun --standalone --nproc_per_node "${NPROC_PER_NODE:-1}" -m chunkkv.evaluate "$@"
