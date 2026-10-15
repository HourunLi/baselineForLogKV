#!/usr/bin/env bash
# One model per GPU on one machine; NPROC_PER_NODE selects the number of local GPUs (default 1).
# Pass --suite main|niah|longbench for a reference task group or --tasks for a diagnostic check.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:?Usage: bash scripts/evaluate.sh configs/qwen3_1.7b_32k_pyramidkv.yaml --suite niah [--limit 4] [--method fullkv] ...}
shift
torchrun --standalone --nproc_per_node "${NPROC_PER_NODE:-1}" \
    -m pyramidkv.evaluate --config "$CONFIG" "$@"
