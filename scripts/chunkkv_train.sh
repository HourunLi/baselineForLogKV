#!/usr/bin/env bash
# Dense CPT control. Multi-node: run on every node with NNODES/NODE_RANK/MASTER_ADDR set.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:-configs/chunkkv/dense_cpt_qwen1.7b_32k.yaml}
test -f "$CONFIG" || { echo "Config not found: $CONFIG" >&2; exit 1; }
shift $(( $# > 0 ? 1 : 0 ))
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
if [ "${NODE_RANK:-0}" = "0" ]; then
    python -m chunkkv.train --config "$CONFIG" --prepare "$@"
fi
torchrun --nnodes "${NNODES:-1}" --nproc_per_node "${NPROC_PER_NODE:-8}" --node_rank "${NODE_RANK:-0}" \
    --master_addr "${MASTER_ADDR:-localhost}" --master_port "${MASTER_PORT:-6000}" \
    -m chunkkv.train --config "$CONFIG" "$@"
