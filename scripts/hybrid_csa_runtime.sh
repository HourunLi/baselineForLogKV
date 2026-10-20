#!/usr/bin/env bash
# Shared CUDA/library setup; does not activate or alter any Python environment.
export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
export CUDNN_LOGERR_DBG=${CUDNN_LOGERR_DBG:-1} CUDNN_LOGDEST_DBG=${CUDNN_LOGDEST_DBG:-stderr}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0} NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2} NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TMPDIR=${TMPDIR:-/tmp}
