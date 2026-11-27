#!/usr/bin/env bash
# ModelArts launches this script once on each node.
set -euo pipefail
cd "$(dirname "$0")"
python_bin=${DEEPSEEK4_PYTHON:-/home/ma-user/anaconda3/envs/torch218/bin/python}
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22} NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none} NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4} CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Set these before importing any HF library in the parent or torchrun workers.
export HF_HOME=${DEEPSEEK4_HF_CACHE:-${HF_HOME:-/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache}}
export HF_DATASETS_CACHE="$HF_HOME/hf_cache" HF_EVALUATE_CACHE="$HF_HOME/evaluate"
export HF_MODULES_CACHE="$HF_HOME/modules" HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export RULER_CACHE_DIR="$HF_HOME/ruler_cache" NLTK_DATA="$HF_HOME/nltk_data"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_IN_MEMORY_MAX_SIZE=0 HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1
export BABILONG_DIR=${BABILONG_DIR:-$HF_HOME/babilong} LONGBENCH2_DIR=${LONGBENCH2_DIR:-$HF_HOME/longbench2}

mode=${1:-eval}
if (( $# )); then shift; fi
case "$mode" in
  eval) exec "$python_bin" -m deepseek4.evaluate "$@" ;;
  all|check|prepare|train) exec "$python_bin" -m deepseek4.pipeline "$mode" "$@" ;;
  *) echo 'Usage: bash majob.sh {all|check|prepare|train|eval} [options]' >&2; exit 2 ;;
esac
