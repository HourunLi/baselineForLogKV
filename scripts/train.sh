#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:-configs/qwen3_1.7b_base.yaml}
test -f "$CONFIG" || { echo "Config not found: $CONFIG" >&2; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
python -m models.qwen3.check_env --training-only
python -m models.qwen3.train --config "$CONFIG" --prepare-dataset-only
torchrun --standalone --nproc_per_node "${NPROC_PER_NODE:-1}" \
    -m models.qwen3.train --config "$CONFIG"
