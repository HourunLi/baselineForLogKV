#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/nccl_env.sh
python_bin=${NFINI_TRAIN_PYTHON:-python}
if [[ -z ${NFINI_TRAIN_PYTHON:-} && -d /opt/conda ]]; then
    python_bin=/opt/conda/envs/torch218/bin/python
fi
if ! command -v "$python_bin" >/dev/null; then
    echo "Training interpreter unavailable: $python_bin. Set NFINI_TRAIN_PYTHON to the prepared environment." >&2
    exit 1
fi
echo "Training Python: $(command -v "$python_bin")" >&2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false LITDATA_DISABLE_VERSION_CHECK=1
export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
for arg in "$@"; do
    if [[ $arg == --print-config ]]; then exec "$python_bin" -m nfini_attention.train "$@"; fi
done
case "${1:-}" in
    prepare)
        if [[ ${VC_TASK_INDEX:-${NODE_RANK:-${GROUP_RANK:-0}}} != 0 ]]; then exit 0; fi
        exec "$python_bin" -m nfini_attention.train "$@" ;;
    cpt|check) ;;
    *) echo "Usage: bash majob.sh --train {prepare|cpt} [--config FILE] [--steps N]" >&2; exit 2 ;;
esac
gpus=${GPUS_PER_NODE:-${MA_NUM_GPUS:-$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')}}
nodes=${MA_NUM_HOSTS:-${GROUP_WORLD_SIZE:-1}}
addr=${MASTER_ADDR:-localhost}
if [[ -n ${MA_VJ_NAME:-} ]]; then addr="${MA_VJ_NAME}-${MA_TASK_NAME}-${MA_MASTER_INDEX:-0}.${MA_VJ_NAME}"; fi
if [[ ! $gpus =~ ^[1-9][0-9]*$ ]]; then echo "Training needs a positive GPU count; found $gpus" >&2; exit 1; fi
if [[ $nodes != 1 && $addr == localhost ]]; then
    echo "Multi-node training needs a reachable MASTER_ADDR" >&2
    exit 1
fi
exec "$python_bin" -m torch.distributed.run --nproc_per_node "$gpus" --nnodes "$nodes" \
    --node_rank "${VC_TASK_INDEX:-${NODE_RANK:-${GROUP_RANK:-0}}}" --master_addr "$addr" --master_port "${MASTER_PORT:-6000}" \
    -m nfini_attention.train "$@"
