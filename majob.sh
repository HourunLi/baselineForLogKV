#!/usr/bin/env bash
# managed_cluster evaluation entry, once per node. --all runs both complete CaM experiments.
set -eo pipefail
cd "$(dirname "$0")"

usage="Usage: bash majob.sh [configs/cam_qwen3_1.7b.yaml] [--backend cam|dense|logkv] [--model DIR] [--limit N] [--output DIR]"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "Usage: bash majob.sh [--backend cam|dense|logkv] [--lengths 32k,...,1m] [--suites LIST] [--print-config]"
    echo "bash majob.sh --all runs the full CaM suite and six target-budget RULER passes."
    echo "Supplying an original YAML runs the legacy protocol:"
    echo "$usage"
    echo "Only the legacy YAML path uses the HF-converted Base."
    echo "Default environment: conda torch218; set CAM_PYTHON to use an existing interpreter."
    exit 0
fi
# A positional legacy YAML selects the existing HF/SemanticLogKV entry point.
niah_protocol=1
[[ $# -gt 0 && "$1" != -* ]] && niah_protocol=0
CONFIG_FILE=configs/cam_qwen3_1.7b.yaml
if [[ $# -gt 0 && "$1" != -* ]]; then
    CONFIG_FILE=$1
    shift
fi
[[ -f "$CONFIG_FILE" ]] || { echo "Missing CaM YAML: $CONFIG_FILE" >&2; exit 2; }
# The task lists, NIAH lengths and experiment YAML are fixed for a formal run.
for arg in "$@"; do
    (( niah_protocol )) && break
    case "$arg" in
        --suite|--suite=*|--tasks|--tasks=*|--config|--config=*|--print-config)
            echo "majob.sh runs the full suites of the config's context; $arg is not allowed here (use models.cam.evaluate)." >&2
            exit 2 ;;
    esac
done

# Conda activation scripts may read unset variables; enable nounset afterwards.
if [[ -z "${CAM_PYTHON:-}" ]]; then
    source /opt/conda/bin/activate "${CAM_CONDA_ENV:-torch218}"
    CAM_PYTHON=$(python -c 'import sys; print(sys.executable)')
fi
set -u
# Runtime library order of SemanticLogKV's eval.sh/majob.sh (as on the other baseline branches): a training job's
# default LD_LIBRARY_PATH can reach an older libcuda first ("driver ... too old"); torch218's cu128 build needs these.
export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0} NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
if (( niah_protocol )); then
    for arg in "$@"; do
        if [[ "$arg" == --print-config ]]; then exec "$CAM_PYTHON" -m models.cam.niah "$@"; fi
    done
fi
if (( niah_protocol )); then
    export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
    export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2} NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}
    GPUS_PER_NODE=$("$CAM_PYTHON" - <<'PY_GPU'
import os
import sys
import torch

visible = torch.cuda.device_count()
allocated = int(os.environ.get("MA_NUM_GPUS") or visible)
available = min(visible, allocated)
explicit = os.getenv("GPUS_PER_NODE") or os.getenv("NPROC_PER_NODE") or os.getenv("LOCAL_WORLD_SIZE")
workers = int(explicit) if explicit else available
if not 1 <= workers <= available:
    sys.exit(f"Invalid GPU count: requested={workers}, CUDA visible={visible}, MA_NUM_GPUS={allocated}")
torch.cuda.init()
if not torch.cuda.is_bf16_supported(including_emulation=False):
    sys.exit("The selected Qwen3 evaluation needs a BF16-capable CUDA device")
print(workers)
PY_GPU
)
else
    GPUS_PER_NODE=${MA_NUM_GPUS:-${NPROC_PER_NODE:-${LOCAL_WORLD_SIZE:-}}}
    if [[ -z "$GPUS_PER_NODE" ]]; then
        GPUS_PER_NODE=$("$CAM_PYTHON" -c 'import torch; print(torch.cuda.device_count())')
    fi
fi
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

if (( niah_protocol )); then
    export MA_NUM_HOSTS="$NUM_NODES" MA_NUM_GPUS="$GPUS_PER_NODE" NODE_RANK MASTER_ADDR EVAL_MASTER_PORT
    exec "$CAM_PYTHON" -m models.cam.niah "$@"
fi

# The YAML's top-level context: key (32k when absent) picks the stages.
CONTEXT=$(sed -n -E 's/^context:[[:space:]]*["'"'"']?([0-9a-z]+)["'"'"']?[[:space:]]*(#.*)?$/\1/p' "$CONFIG_FILE" | tail -n 1)
case "${CONTEXT:-32k}" in
    32k) SUITES=(main niah) ;;
    64k|128k) SUITES=(long ruler) ;;
    *) echo "Unknown context '$CONTEXT' in $CONFIG_FILE (32k, 64k or 128k)" >&2; exit 2 ;;
esac

echo "Python=$CAM_PYTHON node=$NODE_RANK/$NUM_NODES GPUs/node=$GPUS_PER_NODE world_size=$LITGPT_EXPECTED_WORLD_SIZE"
echo "Context ${CONTEXT:-32k}: suites ${SUITES[*]}"
for suite in "${SUITES[@]}"; do
    port=$EVAL_MASTER_PORT
    [[ "$suite" != "${SUITES[1]}" ]] || port=$NIAH_MASTER_PORT
    echo "Evaluating $suite at $MASTER_ADDR:$port using $CONFIG_FILE"
    "$CAM_PYTHON" -m torch.distributed.run \
        --nnodes "$NUM_NODES" --nproc_per_node "$GPUS_PER_NODE" \
        --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$port" \
        -m models.cam.evaluate --config "$CONFIG_FILE" --suite "$suite" "$@"
done
