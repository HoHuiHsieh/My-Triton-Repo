# !/bin/bash
# This script is used to build the engine for the model using TensorRT.
# It converts the checkpoint to a format suitable for TensorRT and then builds the engine.
# Ensure you have the required environment variables set before running this script.
# Usage: ./build-engine.sh

# FP8 Post-Training Quantization
python ./src/quantize.py \
    --model_dir ./model \
    --output_dir ./checkpoint/FP8 \
    --dtype bfloat16 \
    --qformat fp8 \
    --kv_cache_dtype fp8 \
    --calib_size 512 \
&& \
trtllm-build --checkpoint_dir ./checkpoint/FP8 \
    --output_dir ./engine/FP8 \
    --gemm_plugin auto \
    --max_batch_size 16
