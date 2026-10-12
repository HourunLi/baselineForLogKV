#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_PATH=${1:?Usage: bash scripts/evaluate.sh /path/to/student_model [--limit 4] [--backend logkv]}
shift
torchrun --standalone --nproc_per_node "${NPROC_PER_NODE:-1}" \
    -m models.qwen3.evaluate --model "$MODEL_PATH" "$@"
