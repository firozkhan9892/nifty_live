#!/bin/bash
pip install -r requirements-vercel.txt
# Remove NVIDIA CUDA libs pulled by xgboost (they add ~300MB)
pip uninstall -y nvidia-nccl-cu12 nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cusparse-cu12 nvidia-cuda-runtime-cu12 2>/dev/null || true
