#!/bin/bash
# source /home/miniconda3/bin/activate megatron-lm-014
if ! source /home/ma-user/anaconda3/bin/activate torch218; then
    echo "❌ 无法激活 torch218，停止执行，避免使用错误的 Python 环境。"
    exit 1
fi

# Set caches before Python imports Hugging Face and freezes its cache paths.
export HF_HOME=${HF_HOME:-/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache}
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
export PYTHONUNBUFFERED=1
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
# 单进程诊断脚本（unused/niah_surprisal.py、unused/reuse_label_stats.py 等）：
# 环境与 majob.sh 相同（torch218、HF/RULER/NLTK 离线缓存、CUDA），只用一张 GPU。
#   bash diag.sh unused/niah_surprisal.py --config exp/qwen1.7b-32k/arc_gamma_k12_b128_p1024_2k.yaml --out s.jsonl
# 选卡：CUDA_VISIBLE_DEVICES=3 bash diag.sh ...
# ==============================================================================
if [ -z "$1" ]; then
    echo "❌ 致命错误：未提供要运行的脚本！"
    echo "💡 正确用法：bash $0 <脚本.py> [参数 ...]"
    exit 1
fi
if [ ! -f "$1" ]; then
    echo "❌ 致命错误：找不到脚本 $1"
    exit 1
fi
echo "🔬 诊断: ${PYTHON_BIN} $*"
"${PYTHON_BIN}" "$@"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "❌ 诊断失败 (Exit Code: ${STATUS})；请查看上方第一个 Traceback。"
    exit "${STATUS}"
fi
echo "🎉 诊断结束：$1"
