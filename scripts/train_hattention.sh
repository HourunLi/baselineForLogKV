#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin=${LLA_TRAIN_PYTHON:-python}
if [[ -z ${LLA_TRAIN_PYTHON:-} && -d /home/ma-user/anaconda3 ]]; then
    python_bin=/home/ma-user/anaconda3/envs/torch218/bin/python
fi
if ! command -v "$python_bin" >/dev/null; then
    echo "Training interpreter unavailable: $python_bin. Set LLA_TRAIN_PYTHON to the prepared environment." >&2
    exit 1
fi
echo "Training Python: $(command -v "$python_bin")" >&2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false LITDATA_DISABLE_VERSION_CHECK=1
export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
for arg in "$@"; do
    if [[ $arg == --print-config ]]; then exec "$python_bin" -m lla_bench.train_student "$@"; fi
done
case "${1:-}" in
    pipeline) exec "$python_bin" -m lla_bench.pipeline ;;
    convert|prepare|probe)
        if [[ ${VC_TASK_INDEX:-${NODE_RANK:-0}} != 0 ]]; then exit 0; fi
        exec "$python_bin" -m lla_bench.train_student "$@" ;;
    distill|cpt) ;;
    *) echo "Usage: bash majob.sh --train {convert|prepare|probe|distill|cpt} [--config FILE] [--steps N]" >&2; exit 2 ;;
esac
gpus=${GPUS_PER_NODE:-${MA_NUM_GPUS:-$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')}}
nodes=${MA_NUM_HOSTS:-1}
addr=${MASTER_ADDR:-localhost}
if [[ -n ${MA_VJ_NAME:-} ]]; then addr="${MA_VJ_NAME}-${MA_TASK_NAME}-${MA_MASTER_INDEX:-0}.${MA_VJ_NAME}"; fi
if [[ ! $gpus =~ ^[1-9][0-9]*$ ]]; then echo "Training needs a positive GPU count; found $gpus" >&2; exit 1; fi
if [[ $nodes != 1 && $addr == localhost ]]; then
    echo "Multi-node training needs a reachable MASTER_ADDR" >&2
    exit 1
fi
exec "$python_bin" -m torch.distributed.run --nproc_per_node "$gpus" --nnodes "$nodes" \
    --node_rank "${VC_TASK_INDEX:-${NODE_RANK:-0}}" --master_addr "$addr" --master_port "${MASTER_PORT:-6000}" \
    -m lla_bench.train_student "$@"
