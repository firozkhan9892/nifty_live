#!/bin/bash
# Install deps except xgboost (handled separately to avoid nvidia cuda)
pip install fastapi python-telegram-bot smartapi-python pyotp yfinance pandas numpy scikit-learn python-dotenv pytz requests
# Install xgboost without CUDA deps
pip install xgboost --no-deps 2>/dev/null || pip install xgboost
pip install scipy 2>/dev/null || true
# Remove any NVIDIA CUDA libs accidentally pulled
pip uninstall -y nvidia-nccl-cu12 nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cusparse-cu12 nvidia-cuda-runtime-cu12 2>/dev/null || true
pip list --format=columns 2>/dev/null | grep -i nvidia || echo "No nvidia packages"
