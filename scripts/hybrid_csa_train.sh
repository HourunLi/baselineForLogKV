#!/usr/bin/env bash
# Activate a separate training environment first. Run once per node for training.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:-configs/hybrid_csa/train_qwen1.7b_32k.yaml}
shift $(( $# > 0 ? 1 : 0 ))
PYTHON=${HYBRID_CSA_TRAIN_PYTHON:-${BASELINE_TRAIN_PYTHON:-${BASELINE_PYTHON:-python}}}
source scripts/hybrid_csa_runtime.sh
for argument in "$@"; do
    if [[ "$argument" == --print-config || "$argument" == --prepare-data ]]; then
        exec "$PYTHON" -m hybrid_csa.train --config "$CONFIG" "$@"
    fi
done
# CUDA init catches stale libcuda even when NVML still reports all visible cards.
VISIBLE=$("$PYTHON" -c 'import torch; torch.cuda.init(); print(torch.cuda.device_count())')
NPROC=${MA_NUM_GPUS:-${NPROC_PER_NODE:-${LOCAL_WORLD_SIZE:-$VISIBLE}}}
NODES=${MA_NUM_HOSTS:-${NNODES:-${GROUP_WORLD_SIZE:-1}}}
RANK=${VC_TASK_INDEX:-${NODE_RANK:-${GROUP_RANK:-0}}}
if (( NPROC < 1 || NPROC > VISIBLE )); then
    echo "Requested $NPROC GPUs but only $VISIBLE are visible" >&2; exit 1
fi
if (( NODES == 1 )); then
    exec "$PYTHON" -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node "$NPROC" \
        -m hybrid_csa.train --config "$CONFIG" "$@"
fi
if [[ -z "${MASTER_ADDR:-}" && -n "${MA_VJ_NAME:-}" ]]; then
    MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME:?MA_TASK_NAME is required}-${MA_MASTER_INDEX:-0}.${MA_VJ_NAME}"
else
    MASTER_ADDR=${MASTER_ADDR:?Multi-node training requires MASTER_ADDR}
fi
if [[ "$MASTER_ADDR" == localhost || "$MASTER_ADDR" == 127.0.0.1 || "$MASTER_ADDR" == ::1 ]]; then
    echo "Multi-node MASTER_ADDR must not be localhost" >&2; exit 1
fi
exec "$PYTHON" -m torch.distributed.run --nnodes "$NODES" --nproc_per_node "$NPROC" --node_rank "$RANK" \
    --master_addr "$MASTER_ADDR" --master_port "${MASTER_PORT:-6000}" \
    -m hybrid_csa.train --config "$CONFIG" "$@"
