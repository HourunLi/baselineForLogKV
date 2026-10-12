#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

NPROC_PER_NODE=${NPROC_PER_NODE:-${MA_NUM_GPUS:-${LOCAL_WORLD_SIZE:-1}}}
NNODES=${NNODES:-${MA_NUM_HOSTS:-${GROUP_WORLD_SIZE:-1}}}
NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX:-${GROUP_RANK:-0}}}
MASTER_PORT=${MASTER_PORT:-6000}
for name in NPROC_PER_NODE NNODES NODE_RANK MASTER_PORT; do
    if [[ ! ${!name} =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${name} must be a non-negative integer." >&2
        exit 1
    fi
done
if (( NPROC_PER_NODE < 1 || NNODES < 1 || NODE_RANK >= NNODES || MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
    echo "ERROR: invalid process count, node rank, or master port." >&2
    exit 1
fi
if [[ ${MA_NUM_HOSTS:-1} != 1 && $NNODES == 1 ]]; then
    echo "ERROR: MA_NUM_HOSTS indicates multiple hosts; NNODES=1 would launch isolated jobs." >&2
    exit 1
fi

if (( NNODES == 1 )); then
    launch=(--standalone --nproc_per_node "$NPROC_PER_NODE")
else
    if [[ -z ${MASTER_ADDR:-} && -n ${MA_VJ_NAME:-} && -n ${MA_TASK_NAME:-} ]]; then
        MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-${MA_MASTER_INDEX:-0}.${MA_VJ_NAME}"
    fi
    if [[ -z ${MASTER_ADDR:-} || $MASTER_ADDR == localhost || $MASTER_ADDR == 127.0.0.1 ]]; then
        echo "ERROR: multi-node evaluation requires a shared MASTER_ADDR." >&2
        exit 1
    fi
    launch=(--nnodes "$NNODES" --nproc_per_node "$NPROC_PER_NODE"
            --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")
fi
exec torchrun "${launch[@]}" -m models.cam.evaluate --config configs/cam_qwen3_1.7b.yaml "$@"
