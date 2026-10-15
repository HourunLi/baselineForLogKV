#!/usr/bin/env bash
# Training-platform entry point: run once per node. Evaluates the main suite, then 32K NIAH.
set -eo pipefail
cd "$(dirname "$0")"

usage="Usage: bash majob.sh [configs/qwen3_1.7b_32k_pyramidkv.yaml] [evaluation options, e.g. --limit 4 --output DIR]"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "$usage"
    echo "Runs configs/eval_suites.yaml: main (47 entries), then NIAH (3 entries, 32768 only)."
    echo "Default environment: conda torch218; set PYRAMIDKV_PYTHON to use an existing interpreter."
    exit 0
fi
CONFIG_FILE=configs/qwen3_1.7b_32k_pyramidkv.yaml
if [[ $# -gt 0 && "$1" != -* ]]; then
    CONFIG_FILE=$1
    shift
fi
[[ -f "$CONFIG_FILE" ]] || { echo "Missing evaluation YAML: $CONFIG_FILE" >&2; exit 2; }
# The formal protocol fixes tasks, NIAH lengths, window and offline data; only the
# method, budget, output and a debugging --limit may be chosen here.
for arg in "$@"; do
    case "$arg" in
        --suite|--suite=*|--tasks|--tasks=*|--eval-suites|--eval-suites=*|--include-path|--include-path=*|\
        --ruler-lengths|--ruler-lengths=*|--ruler-pretrained|--ruler-pretrained=*|\
        --max-seq-length|--max-seq-length=*|--print-config)
            echo "majob.sh runs the fixed main and NIAH suites; $arg is not allowed here." >&2
            echo "Use scripts/evaluate.sh for selected tasks or diagnostics." >&2
            exit 2 ;;
    esac
done

# Conda activation scripts may read unset variables; enable nounset afterwards.
if [[ -z "${PYRAMIDKV_PYTHON:-}" ]]; then
    source /home/ma-user/anaconda3/bin/activate "${PYRAMIDKV_CONDA_ENV:-torch218}"
    PYRAMIDKV_PYTHON=$(python -c 'import sys; print(sys.executable)')
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
MAIN_MASTER_PORT=${MAIN_MASTER_PORT:-$((MASTER_PORT + 1))}
NIAH_MASTER_PORT=${NIAH_MASTER_PORT:-$((MASTER_PORT + 2))}
for name in MAIN_MASTER_PORT NIAH_MASTER_PORT; do
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
[[ "$MAIN_MASTER_PORT" != "$NIAH_MASTER_PORT" ]] || { echo "Evaluation stages need distinct ports" >&2; exit 2; }
export KV_BASELINE_EXPECTED_WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}

echo "Python=$PYRAMIDKV_PYTHON node=$NODE_RANK/$NUM_NODES GPUs/node=$GPUS_PER_NODE world_size=$KV_BASELINE_EXPECTED_WORLD_SIZE"
for suite in main niah; do
    port=$MAIN_MASTER_PORT
    [[ "$suite" != niah ]] || port=$NIAH_MASTER_PORT
    echo "Evaluating suite '$suite' at $MASTER_ADDR:$port using $CONFIG_FILE"
    # set -e stops here when a stage fails, so NIAH never runs after a failed main stage.
    "$PYRAMIDKV_PYTHON" -m torch.distributed.run \
        --nnodes "$NUM_NODES" --nproc_per_node "$GPUS_PER_NODE" \
        --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$port" \
        -m pyramidkv.evaluate --config "$CONFIG_FILE" --suite "$suite" "$@"
done
