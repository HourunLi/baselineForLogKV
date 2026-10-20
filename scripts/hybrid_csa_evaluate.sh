#!/usr/bin/env bash
# A selected diagnostic suite on one node; use majob.sh for platform multi-node jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/hybrid_csa_runtime.sh
PYTHON=${HYBRID_CSA_PYTHON:-${BASELINE_EVAL_PYTHON:-${BASELINE_PYTHON:-/home/ma-user/anaconda3/envs/torch218/bin/python}}}
for arg in "$@"; do
    if [[ "$arg" == --print-config || "$arg" == --help || "$arg" == -h ]]; then
        exec "$PYTHON" -m hybrid_csa.evaluate "$@"
    fi
done
if (( ${MA_NUM_HOSTS:-${NNODES:-${GROUP_WORLD_SIZE:-1}}} != 1 )); then
    echo "Use majob.sh --mode eval for multi-node evaluation" >&2; exit 2
fi
VISIBLE=$("$PYTHON" -c 'import torch; torch.cuda.init(); print(torch.cuda.device_count())')
NPROC=${MA_NUM_GPUS:-${NPROC_PER_NODE:-${LOCAL_WORLD_SIZE:-$VISIBLE}}}
[[ "$NPROC" =~ ^[1-9][0-9]*$ ]] && (( NPROC <= VISIBLE )) || { echo "Invalid GPU process count: $NPROC" >&2; exit 2; }
export LITGPT_EXPECTED_WORLD_SIZE=$NPROC
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" -m hybrid_csa.evaluate "$@"
