#!/usr/bin/env bash
# One model per GPU on one machine; NPROC_PER_NODE selects the number of local GPUs (default 1).
# Runs from the project root with the active interpreter (SemanticLogKV's torch218) or DMS_PYTHON.
# Pass --suite main|niah|longbench for a reference task group or --tasks for a diagnostic check.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_PATH=${1:?Usage: bash scripts/evaluate.sh /path/to/student_model --suite niah [--limit 4] [--backend logkv]}
shift
# CUDA 12.8 library order of SemanticLogKV's eval.sh, as in majob.sh (docs/BASELINE_ENVIRONMENT.md section 1).
export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
"${DMS_PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE:-1}" \
    -m models.qwen3.evaluate --model "$MODEL_PATH" "$@"
