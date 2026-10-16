#!/usr/bin/env bash
# One model per GPU on one machine; NPROC_PER_NODE selects the number of local GPUs (default 1).
# Runs from the project root with the active interpreter (SemanticLogKV's torch218) or H2O_PYTHON.
# Pass --suite main|niah|longbench for a reference task group or --tasks for a diagnostic check.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:?Usage: bash scripts/evaluate.sh configs/qwen3_1.7b_32k_h2o.yaml --suite niah [--limit 4] [--method fullkv] ...}
shift
# CUDA 12.8 library order of SemanticLogKV's eval.sh, as in majob.sh (docs/BASELINE_ENVIRONMENT.md section 1).
export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
"${H2O_PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE:-1}" \
    -m h2okv.evaluate --config "$CONFIG" "$@"
