#!/usr/bin/env bash
# Train only, on one machine, in SemanticLogKV's environment (torch218) or DMS_PYTHON; NPROC_PER_NODE selects the
# number of local GPUs (default 1). bash majob.sh --train CONFIG trains and then evaluates in one job.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:-configs/qwen3_1.7b_base.yaml}
test -f "$CONFIG" || { echo "Config not found: $CONFIG" >&2; exit 1; }
PYTHON=${DMS_PYTHON:-python}
# CUDA 12.8 library order of SemanticLogKV's eval.sh, as in majob.sh (docs/BASELINE_ENVIRONMENT.md section 1).
export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
"$PYTHON" -m models.qwen3.check_env --training-only
"$PYTHON" -m models.qwen3.train --config "$CONFIG" --prepare-dataset-only
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE:-1}" \
    -m models.qwen3.train --config "$CONFIG"
