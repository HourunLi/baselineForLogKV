#!/usr/bin/env bash
# Training-platform entry point for Base + ChunkKV evaluation (no CPT). Run once per node:
# main suite (47 entries), then NIAH (3 entries, 32768 only); a failed stage stops the job.
set -eo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "Usage: bash majob.sh [configs/chunkkv/experiment_qwen1.7b_32k.yaml] [evaluation options]"
    echo "Runs main benchmarks and 32K NIAH. Default environment: torch218."
    echo "Set CHUNKKV_PYTHON to an existing interpreter to skip conda activation."
    exit 0
fi
CONFIG_FILE=configs/chunkkv/experiment_qwen1.7b_32k.yaml
if [[ $# -gt 0 && "$1" != -* ]]; then
    CONFIG_FILE=$1
    shift
fi
[[ -f "$CONFIG_FILE" ]] || { echo "Missing experiment YAML: $CONFIG_FILE" >&2; exit 2; }
# The task lists, NIAH lengths and experiment YAML are fixed for a formal run.
for arg in "$@"; do
    case "$arg" in
        --suite|--suite=*|--tasks|--tasks=*|--evaluation-config|--evaluation-config=*|\
        --experiment-config|--experiment-config=*|--semantic-config|--semantic-config=*|--print-config)
            echo "majob.sh runs both full suites; $arg is not allowed here (use chunkkv.evaluate)." >&2
            exit 2 ;;
    esac
done

# Conda activation scripts may read unset variables; enable nounset afterwards.
if [[ -z "${CHUNKKV_PYTHON:-}" ]]; then
    source /home/ma-user/anaconda3/bin/activate "${CHUNKKV_CONDA_ENV:-torch218}"
    CHUNKKV_PYTHON=$(python -c 'import sys; print(sys.executable)')
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
export LITGPT_EXPECTED_WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}

echo "Python=$CHUNKKV_PYTHON node=$NODE_RANK/$NUM_NODES GPUs/node=$GPUS_PER_NODE world_size=$LITGPT_EXPECTED_WORLD_SIZE"
for suite in main niah; do
    port=$EVAL_MASTER_PORT
    [[ "$suite" != niah ]] || port=$NIAH_MASTER_PORT
    echo "Evaluating $suite at $MASTER_ADDR:$port using $CONFIG_FILE"
    "$CHUNKKV_PYTHON" -m torch.distributed.run \
        --nnodes "$NUM_NODES" --nproc_per_node "$GPUS_PER_NODE" \
        --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$port" \
        -m chunkkv.evaluate --experiment-config "$CONFIG_FILE" --suite "$suite" "$@"
done
