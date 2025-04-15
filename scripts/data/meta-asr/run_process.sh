#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0  # Adjust this to your GPU number

# Add error handling
set -e

python3 parquet_data_process.py 2>&1 | tee processing.log