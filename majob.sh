#!/usr/bin/env bash
# Training-platform entry point for evaluating an exported DMS student; it never trains.
# Run once per node: main suite (47 entries), then NIAH (3 entries, 32768 only); a failed stage stops the job.
set -eo pipefail
cd "$(dirname "$0")"

usage="Usage: bash majob.sh [configs/experiment_qwen1.7b_32k.yaml] --model DIR [--backend dms|logkv] [--limit N] [--output DIR]"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "$usage"
    echo "Evaluates configs/eval_suites.yaml in two stages. Default environment: conda torch218;"
    echo "set DMS_PYTHON to use an existing interpreter. Train first with scripts/train.sh."
    exit 0
fi
CONFIG_FILE=configs/experiment_qwen1.7b_32k.yaml
if [[ $# -gt 0 && "$1" != -* ]]; then
    CONFIG_FILE=$1
    shift
fi
[[ -f "$CONFIG_FILE" ]] || { echo "Missing experiment YAML: $CONFIG_FILE" >&2; exit 2; }
has_model=0
# The task lists, NIAH lengths and experiment YAML are fixed for a formal run.
for arg in "$@"; do
    case "$arg" in
        --model|--model=*) has_model=1 ;;
        --suite|--suite=*|--tasks|--tasks=*|--evaluation-config|--evaluation-config=*|\
        --experiment-config|--experiment-config=*|--print-config)
            echo "majob.sh runs both full suites; $arg is not allowed here (use models.qwen3.evaluate)." >&2
            exit 2 ;;
    esac
done
(( has_model )) || { echo "--model is required: pass the exported DMS student (majob.sh never trains)." >&2; exit 2; }

# Conda activation scripts may read unset variables; enable nounset afterwards.
if [[ -z "${DMS_PYTHON:-}" ]]; then
    source /home/ma-user/anaconda3/bin/activate "${DMS_CONDA_ENV:-torch218}"
    DMS_PYTHON=$(python -c 'import sys; print(sys.executable)')
fi
set -u
GPUS_PER_NODE=${MA_NUM_GPUS:-${NPROC_PER_NODE:-${LOCAL_WORLD_SIZE:-8}}}
NUM_NODES=${MA_NUM_HOSTS:-${GROUP_WORLD_SIZE:-1}}
NODE_RANK=${VC_TASK_INDEX:-${NODE_RANK:-${GROUP_RANK:-0}}}
MASTER_ADDR=${MASTER_ADDR:-localhost}
if [[ -n "${MA_VJ_NAME:-}" ]]; then
    MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME:?MA_TASK_NAME is required}-${MA_MASTER_INDEX:-0}.${MA_VJ_NAME}"
fi
MASTER_PORT=${MASTER_PORT:-6000}
for name in GPUS_PER_NODE NUM_NODES NODE_RANK MASTER_PORT; do
    [[ "${!name}" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "$name must be an integer: ${!name}" >&2; exit 2; }
done
EVAL_MASTER_PORT=${EVAL_MASTER_PORT:-$((MASTER_PORT + 1))}
NIAH_MASTER_PORT=${NIAH_MASTER_PORT:-$((MASTER_PORT + 2))}
for name in EVAL_MASTER_PORT NIAH_MASTER_PORT; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] && (( ${!name} <= 65535 )) || {
        echo "$name must be a port in 1..65535" >&2; exit 2;
    }
done
(( GPUS_PER_NODE > 0 && NUM_NODES > 0 && NODE_RANK < NUM_NODES )) || {
    echo "Invalid GPU count, node count or node rank" >&2; exit 2;
}
if (( NUM_NODES > 1 )) && [[ "$MASTER_ADDR" == localhost || "$MASTER_ADDR" == 127.* || "$MASTER_ADDR" == ::1 ]]; then
    echo "Multi-node evaluation requires a shared, reachable MASTER_ADDR" >&2
    exit 2
fi
[[ "$EVAL_MASTER_PORT" != "$NIAH_MASTER_PORT" ]] || { echo "Evaluation stages need distinct ports" >&2; exit 2; }
# The bundled evaluator checks the rendezvous against this expected world size.
export LITGPT_EXPECTED_WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}

echo "Python=$DMS_PYTHON node=$NODE_RANK/$NUM_NODES GPUs/node=$GPUS_PER_NODE world_size=$LITGPT_EXPECTED_WORLD_SIZE"
for suite in main niah; do
    port=$EVAL_MASTER_PORT
    [[ "$suite" != niah ]] || port=$NIAH_MASTER_PORT
    echo "Evaluating $suite at $MASTER_ADDR:$port using $CONFIG_FILE"
    "$DMS_PYTHON" -m torch.distributed.run \
        --nnodes "$NUM_NODES" --nproc_per_node "$GPUS_PER_NODE" \
        --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$port" \
        -m models.qwen3.evaluate --experiment-config "$CONFIG_FILE" --suite "$suite" "$@"
done
