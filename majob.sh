#!/usr/bin/env bash
# Training-platform entry point. Run once per node; a failed stage stops the job.
# --model DIR evaluates an exported DMS student: main suite (47 entries), then NIAH (3 entries, 32768 only).
# --train YAML (one node) first prepares the shared Base if it is missing, prepares the data and trains, all in
# the same environment, then evaluates the student it exported. Without --train, nothing is trained.
set -eo pipefail
cd "$(dirname "$0")"

usage="Usage: bash majob.sh [configs/experiment_qwen1.7b_32k.yaml] (--model DIR | --train TRAIN_YAML) [--backend dms|logkv] [--limit N] [--output DIR]"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "$usage"
    echo "Evaluates configs/eval_suites.yaml in two stages, after training with TRAIN_YAML when --train is given"
    echo "(e.g. configs/qwen3_1.7b_base.yaml). Default environment: conda torch218;"
    echo "set DMS_PYTHON to use an existing interpreter."
    exit 0
fi
CONFIG_FILE=configs/experiment_qwen1.7b_32k.yaml
if [[ $# -gt 0 && "$1" != -* ]]; then
    CONFIG_FILE=$1
    shift
fi
[[ -f "$CONFIG_FILE" ]] || { echo "Missing experiment YAML: $CONFIG_FILE" >&2; exit 2; }
has_model=0
TRAIN_CONFIG=
eval_args=()
# The task lists, NIAH lengths and experiment YAML are fixed for a formal run.
while (( $# )); do
    case "$1" in
        --train) [[ $# -ge 2 ]] || { echo "--train needs a training YAML" >&2; exit 2; }
                 TRAIN_CONFIG=$2; shift 2; continue ;;
        --train=*) TRAIN_CONFIG=${1#--train=}; shift; continue ;;
        --model|--model=*) has_model=1 ;;
        --suite|--suite=*|--tasks|--tasks=*|--evaluation-config|--evaluation-config=*|\
        --experiment-config|--experiment-config=*|--print-config)
            echo "majob.sh runs both full suites; $1 is not allowed here (use models.qwen3.evaluate)." >&2
            exit 2 ;;
    esac
    eval_args+=("$1")
    shift
done
if [[ -n "$TRAIN_CONFIG" ]]; then
    (( ! has_model )) || { echo "--train evaluates the student it trains; do not also pass --model." >&2; exit 2; }
    [[ -f "$TRAIN_CONFIG" ]] || { echo "Missing training YAML: $TRAIN_CONFIG" >&2; exit 2; }
else
    (( has_model )) || { echo "--model is required: pass the exported DMS student, or --train TRAIN_YAML." >&2; exit 2; }
fi

# Conda activation scripts may read unset variables; enable nounset afterwards.
if [[ -z "${DMS_PYTHON:-}" ]]; then
    source /home/ma-user/anaconda3/bin/activate "${DMS_CONDA_ENV:-torch218}"
    DMS_PYTHON=$(python -c 'import sys; print(sys.executable)')
fi
set -u
# Runtime settings of SemanticLogKV's eval.sh/majob.sh (commit 4fd97c7) on this platform. A training job's
# default LD_LIBRARY_PATH can reach an older libcuda first; torch218's cu128 build needs these libraries first.
# Its NVTE_*/CUDNN_PATH settings serve Transformer Engine, which this project does not use.
export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
export CUDNN_LOGERR_DBG=${CUDNN_LOGERR_DBG:-1} CUDNN_LOGDEST_DBG=${CUDNN_LOGDEST_DBG:-stderr}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0} NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2} NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}
# Initialise CUDA once here (not just count devices through NVML), so a driver/library mismatch stops the job
# before torchrun starts, naming the libcuda that was loaded. Prints the devices this node can use.
CUDA_PROBE='
import ctypes, os, sys
import torch
try:
    torch.cuda.init()
except Exception as exc:
    try:
        version = ctypes.c_int()
        ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(version))
        path = next((line.split()[-1] for line in open("/proc/self/maps") if "libcuda.so" in line), "?")
        found = f"loaded {path}, CUDA driver API {version.value}"
    except OSError as missing:
        found = f"libcuda.so.1 not loadable ({missing})"
    paths = os.environ.get("LD_LIBRARY_PATH", "")
    sys.exit(f"CUDA initialisation failed: {exc}\n{found}; torch needs CUDA {torch.version.cuda}\n"
             f"LD_LIBRARY_PATH={paths}")
print(torch.cuda.device_count())
'
VISIBLE_GPUS=$("$DMS_PYTHON" -c "$CUDA_PROBE")
[[ "$VISIBLE_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "No CUDA device is visible to $DMS_PYTHON" >&2; exit 2; }
GPUS_PER_NODE=${MA_NUM_GPUS:-${NPROC_PER_NODE:-${LOCAL_WORLD_SIZE:-$VISIBLE_GPUS}}}
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
(( GPUS_PER_NODE <= VISIBLE_GPUS )) || {
    echo "$GPUS_PER_NODE GPU processes requested but only $VISIBLE_GPUS CUDA devices are visible;" \
         "set MA_NUM_GPUS or NPROC_PER_NODE to at most $VISIBLE_GPUS" >&2; exit 2;
}
if (( NUM_NODES > 1 )) && [[ "$MASTER_ADDR" == localhost || "$MASTER_ADDR" == 127.* || "$MASTER_ADDR" == ::1 ]]; then
    echo "Multi-node evaluation requires a shared, reachable MASTER_ADDR" >&2
    exit 2
fi
[[ "$EVAL_MASTER_PORT" != "$NIAH_MASTER_PORT" ]] || { echo "Evaluation stages need distinct ports" >&2; exit 2; }
if [[ -n "$TRAIN_CONFIG" ]] && (( NUM_NODES > 1 )); then
    echo "--train runs on one node; train first, then evaluate on several nodes with --model" >&2
    exit 2
fi
# The bundled evaluator checks the rendezvous against this expected world size.
export LITGPT_EXPECTED_WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}

echo "Python=$DMS_PYTHON node=$NODE_RANK/$NUM_NODES GPUs/node=$GPUS_PER_NODE (visible $VISIBLE_GPUS) world_size=$LITGPT_EXPECTED_WORLD_SIZE"
if [[ -n "$TRAIN_CONFIG" ]]; then
    # Base path and output directory as the training code reads them (relative to the project root).
    mapfile -t train_paths < <("$DMS_PYTHON" -c 'import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
print(config["model"]["name"]); print(config["trainer"]["output_dir"])' "$TRAIN_CONFIG")
    (( ${#train_paths[@]} == 2 )) || { echo "Cannot read model.name and trainer.output_dir from $TRAIN_CONFIG" >&2; exit 2; }
    base_dir=${train_paths[0]}
    student_dir=${train_paths[1]}/student_model
    if [[ ! -e "$base_dir" ]]; then
        echo "Preparing the shared Base in $base_dir from $CONFIG_FILE"
        "$DMS_PYTHON" -m models.qwen3.prepare_base --experiment-config "$CONFIG_FILE" --output "$base_dir"
    fi
    echo "Preparing training data for $TRAIN_CONFIG"
    "$DMS_PYTHON" -m models.qwen3.train --config "$TRAIN_CONFIG" --prepare-dataset-only
    echo "Training $TRAIN_CONFIG on $GPUS_PER_NODE GPUs (free local port)"
    "$DMS_PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$GPUS_PER_NODE" \
        -m models.qwen3.train --config "$TRAIN_CONFIG"
    [[ -f "$student_dir/config.json" ]] || { echo "Training exported no student in $student_dir" >&2; exit 1; }
    eval_args+=(--model "$student_dir")
fi
for suite in main niah; do
    if (( NUM_NODES == 1 )); then
        # A fixed port may already be taken on the node; standalone rendezvous binds a free one.
        rendezvous=(--standalone)
        echo "Evaluating $suite on one node (free local port) using $CONFIG_FILE"
    else
        port=$EVAL_MASTER_PORT
        [[ "$suite" != niah ]] || port=$NIAH_MASTER_PORT
        rendezvous=(--nnodes "$NUM_NODES" --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$port")
        echo "Evaluating $suite at $MASTER_ADDR:$port using $CONFIG_FILE"
    fi
    # set -e stops here when a stage fails, so NIAH never runs after a failed main stage.
    "$DMS_PYTHON" -m torch.distributed.run "${rendezvous[@]}" --nproc_per_node "$GPUS_PER_NODE" \
        -m models.qwen3.evaluate --experiment-config "$CONFIG_FILE" --suite "$suite" "${eval_args[@]}"
done
