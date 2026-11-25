#!/bin/bash
# source /home/miniconda3/bin/activate megatron-lm-014
if ! source /opt/conda/bin/activate torch218; then
    echo "❌ 无法激活 torch218，停止执行，避免使用错误的 Python 环境。"
    exit 1
fi

# Set caches before Python imports Hugging Face and freezes its cache paths.
export HF_HOME=${HF_HOME:-/workspace/data/hf_cache}
export LONGBENCH2_DIR=${LONGBENCH2_DIR:-${HF_HOME}/longbench2}
export BABILONG_DIR=${BABILONG_DIR:-${HF_HOME}/babilong}
export HF_DATASETS_CACHE="${HF_HOME}/hf_cache"
export HF_EVALUATE_CACHE="${HF_HOME}/evaluate"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export RULER_CACHE_DIR="${HF_HOME}/ruler_cache"
export NLTK_DATA="${HF_HOME}/nltk_data"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

PYTHON_BIN=$(python -c 'import sys; print(sys.executable)') || exit 1
echo "🐍 流水线 Python: ${PYTHON_BIN}"
echo "LongBench v2 data directory (ordinary files): ${LONGBENCH2_DIR}"
# DeepSeek 3Bv2 Sandwich Training Script with YAML Configuration
# Optimized for H200 GPUs (141GB VRAM)

export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDNN_LOGERR_DBG=1
export CUDNN_LOGDEST_DBG=stderr
# logKV 流式 attention 每 chunk 产生大量不等长的小分配（评测数千条样本、训练
# T/2 个 chunk），expandable_segments 让分配器按段扩展而非整块缓存，
# 显著缓解长时运行的碎片化 OOM。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

export NVTE_WITH_USER_CUDA=1 # must turn on when multiple-nodes
export CUDNN_PATH=/usr # must turn on when multiple-nodes
export NVTE_CUDA_INCLUDE_DIR=/usr/local/cuda-12.8/include # must turn on when multiple-nodes

export NCCL_NVLS_ENABLE=0
export NCCL_NET_PLUGIN=none
# IB/RoCE ACK 超时是指数而非秒数：4.096us x 2^N，网卡只取低 5 bit（有效 1..31）。
# 12000 截成 uint8 是 224，低 5 bit 为 0 = 无限等待：尾包丢失后永不重传，NCCL
# 内核以 100% 利用率空等，表现为随机卡死。22 约 17s 重传一次，重试耗尽后报错。
export NCCL_IB_TIMEOUT=22
# LogKV 路由决策与注意力重叠（侧流）。先在目标 GPU 上运行
# python unused/check_route_overlap.py，输出 bitwise_equal=true 后再设为 1。
export LOGKV_ROUTE_OVERLAP=${LOGKV_ROUTE_OVERLAP:-0}
export NCCL_NET_GDR_LEVEL=2  # Enable GPUDirect RDMA if RDMA is available # optim0129
export NCCL_MIN_NCHANNELS=4  # Increase NCCL channels # optim0129

# ==============================================================================
# Distributed Training Setup
# ==============================================================================
# Detect GPUs visible to this job (including CUDA_VISIBLE_DEVICES); managed_cluster
# allocation is an upper bound. GPUS_PER_NODE may explicitly select fewer GPUs.
GPUS_PER_NODE=$("${PYTHON_BIN}" - <<'PY_GPU_COUNT'
import os
import sys

import torch

visible = torch.cuda.device_count()
if visible < 1:
    raise SystemExit("ERROR: no CUDA GPUs are visible; check CUDA_VISIBLE_DEVICES and the CUDA runtime.")
try:
    allocated = int(os.environ.get("MA_NUM_GPUS") or visible)
    available = min(visible, allocated)
    selected = int(os.environ.get("GPUS_PER_NODE") or available)
except ValueError:
    raise SystemExit("ERROR: GPUS_PER_NODE and MA_NUM_GPUS must be positive integers.")
if allocated < 1 or not 1 <= selected <= available:
    raise SystemExit(
        f"ERROR: GPUS_PER_NODE={selected} must be within 1..{available} "
        f"(visible={visible}, MA_NUM_GPUS={allocated})."
    )
print(f"GPU detection: visible={visible}, MA_NUM_GPUS={allocated}, using={selected}", file=sys.stderr)
print(selected)
PY_GPU_COUNT
) || exit 1
NUM_NODES=${MA_NUM_HOSTS:-${GROUP_WORLD_SIZE:-1}}
MASTER_ADDR=${MASTER_ADDR:-localhost}
if [ -n "${MA_VJ_NAME:-}" ]; then
    MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-${MA_MASTER_INDEX:-0}.${MA_VJ_NAME}"
fi
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${VC_TASK_INDEX:-${NODE_RANK:-${GROUP_RANK:-0}}}

for name in GPUS_PER_NODE NUM_NODES NODE_RANK MASTER_PORT; do
    value=${!name}
    if [[ -z "${value}" || "${value}" == *[!0-9]* ]]; then
        echo "❌ 致命错误：${name} 必须是非负整数，当前值为 '${value}'"
        exit 1
    fi
done

if [ "${NUM_NODES}" -gt 1 ] && [ "${MASTER_ADDR}" = "localhost" ]; then
    echo "❌ 致命错误：NUM_NODES=${NUM_NODES} 但 MASTER_ADDR=localhost。"
    echo "   多节点 torchrun 必须使用所有节点可访问的 master 地址；否则每个节点会各自组成 world_size=${GPUS_PER_NODE}。"
    exit 1
fi

TRAIN_MASTER_PORT=${TRAIN_MASTER_PORT:-${MASTER_PORT}}
EVAL_MASTER_PORT=${EVAL_MASTER_PORT:-$((MASTER_PORT + 1))}
NIAH_MASTER_PORT=${NIAH_MASTER_PORT:-$((MASTER_PORT + 2))}
export LITGPT_EXPECTED_WORLD_SIZE=${LITGPT_EXPECTED_WORLD_SIZE:-$((NUM_NODES * GPUS_PER_NODE))}

echo "🌍 正在启动多机多卡训练: Node ${NODE_RANK} / ${NUM_NODES}"
echo "🧮 期望 world size: ${LITGPT_EXPECTED_WORLD_SIZE} (= ${NUM_NODES} nodes × ${GPUS_PER_NODE} gpus)"
echo "🔗 Train rendezvous: ${MASTER_ADDR}:${TRAIN_MASTER_PORT}"
echo "🔗 Eval rendezvous: ${MASTER_ADDR}:${EVAL_MASTER_PORT}"
echo "🔗 NIAH rendezvous: ${MASTER_ADDR}:${NIAH_MASTER_PORT}"

set -e
exec "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${NUM_NODES}" --nproc_per_node="${GPUS_PER_NODE}" --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" --master_port="${TRAIN_MASTER_PORT}" demo.py --config "${1:-exp/cpt.yaml}"
