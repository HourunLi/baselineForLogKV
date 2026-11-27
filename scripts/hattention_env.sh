if ! source /home/ma-user/anaconda3/bin/activate torch218; then
    echo "ERROR: failed to activate torch218."
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
export NLTK_DATA="${NLTK_DATA:-${HF_HOME}/nltk_data}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

PYTHON_BIN=$(python -c 'import sys; print(sys.executable)') || exit 1
echo "Eval Python: ${PYTHON_BIN}"
echo "LongBench v2 data directory (ordinary files): ${LONGBENCH2_DIR}"

export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDNN_LOGERR_DBG=1
export CUDNN_LOGDEST_DBG=stderr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

export NVTE_WITH_USER_CUDA=1
export CUDNN_PATH=/usr
export NVTE_CUDA_INCLUDE_DIR=/usr/local/cuda-12.8/include

source scripts/nccl_env.sh
# IB/RoCE ACK 超时是指数而非秒数：4.096us x 2^N，网卡只取低 5 bit（有效 1..31）。
# 12000 截成 uint8 是 224，低 5 bit 为 0 = 无限等待：尾包丢失后永不重传，NCCL
# 内核以 100% 利用率空等，表现为随机卡死。22 约 17s 重传一次，重试耗尽后报错。
# LogKV 路由决策与注意力重叠（侧流）。先在目标 GPU 上运行
# python unused/check_route_overlap.py，输出 bitwise_equal=true 后再设为 1。
export LOGKV_ROUTE_OVERLAP=${LOGKV_ROUTE_OVERLAP:-0}

