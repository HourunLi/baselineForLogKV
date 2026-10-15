#!/usr/bin/env bash
# One model per GPU on one machine; NPROC_PER_NODE selects the number of local GPUs (default 1).
# Pass --suite main|niah|longbench for a reference task group or --tasks for a diagnostic check.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_PATH=${1:?Usage: bash scripts/evaluate.sh /path/to/student_model --suite niah [--limit 4] [--backend logkv]}
shift
torchrun --standalone --nproc_per_node "${NPROC_PER_NODE:-1}" \
    -m models.qwen3.evaluate --model "$MODEL_PATH" "$@"
