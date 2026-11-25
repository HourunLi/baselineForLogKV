#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:-eval}
if (( $# )); then shift; fi
python_bin=${DEEPSEEK4_PYTHON:-/opt/conda/envs/torch218/bin/python}
case "$mode" in
  eval) exec "$python_bin" -m deepseek4.evaluate "$@" ;;
  prepare|train)
    if [[ "$mode" == prepare ]]; then
      nproc=1
      set -- --prepare "$@"
    else
      nproc=${NPROC_PER_NODE:-${MA_NUM_GPUS:-$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')}}
      (( nproc > 0 )) || { echo "Training requires visible CUDA devices" >&2; exit 1; }
    fi
    nodes=${MA_NUM_HOSTS:-${NNODES:-1}}
    rendezvous=(--standalone)
    if (( nodes > 1 )); then
      rendezvous=(--nnodes "$nodes" --node_rank "${VC_TASK_INDEX:-${NODE_RANK:?}}" --master_addr "${MASTER_ADDR:?}" --master_port "${MASTER_PORT:-6001}")
    fi
    exec "$python_bin" -m torch.distributed.run "${rendezvous[@]}" --nproc_per_node "$nproc" -m deepseek4.train "$@"
    ;;
  *) echo 'Usage: bash majob.sh {prepare|train|eval} [options]' >&2; exit 2 ;;
esac
