# Llama-3.1-8B-Instruct on NVIDIA Triton Inference Server

This repository contains all necessary components to deploy and serve Meta's Llama-3.1-8B-Instruct model using NVIDIA Triton Inference Server with TensorRT-LLM acceleration.

[![NVIDIA Triton](https://img.shields.io/badge/NVIDIA-Triton_Inference_Server-76B900?style=flat-square&logo=nvidia)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
[![Llama 3.1](https://img.shields.io/badge/Meta-Llama_3.1-0467DF?style=flat-square&logo=meta)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
[![TensorRT LLM](https://img.shields.io/badge/NVIDIA-TensorRT--LLM-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Quick Start Guide](#quick-start-guide)
  - [0. Download the Model](#0-download-the-model)
  - [1. Build the TensorRT Engine](#1-build-the-tensorrt-engine)
  - [2. Configure the Triton Model Repository](#2-configure-the-triton-model-repository)
  - [3. Start the Triton Inference Server](#3-start-the-triton-inference-server)
- [Frontend Integration](#frontend-integration)
- [Customization](#customization)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Overview

This project enables efficient deployment of Meta's Llama-3.1-8B-Instruct model using NVIDIA's TensorRT-LLM for optimized inference and the Triton Inference Server for scalable serving. The implementation includes FP8 quantization to improve efficiency while maintaining model quality.

The pipeline includes:
- Downloading the Llama-3.1-8B-Instruct model from Hugging Face
- Building a TensorRT-LLM engine optimized for your GPU
- Setting up a complete Triton Inference Server pipeline with preprocessing, inference, and postprocessing stages

## Prerequisites

- NVIDIA GPU with compute capability 8.0 or higher (Ampere architecture or newer)
- Docker with NVIDIA Container Toolkit installed and configured
- Minimum 16GB GPU memory (24GB+ recommended)
- At least 50GB free disk space for model weights and TensorRT engines
- Git and Git LFS for model downloading
- Internet connection for downloading the model

## Repository Structure

```
├── checkpoint/         # Converted and quantized model checkpoint files
├── engine/             # Generated TensorRT-LLM engine files
├── model/              # Original model weights (populated by download script)
├── raw-repository/     # Template files for Triton model repository
├── repository/         # Generated Triton model repository structure (created during setup)
│   ├── llama-3.1-8b-instruct/  # Main model ensemble configuration
│   ├── preprocessing/   # Input preprocessing pipeline
│   ├── tokenize/        # Tokenization component
│   ├── tensorrt_llm/    # TensorRT-LLM inference engine
│   ├── postprocessing/  # Output processing and formatting
│   └── usage_counter/   # Usage tracking component
├── script/             # Utility scripts
│   ├── build-engine.sh  # Script to build the TensorRT engine
│   ├── download.sh      # Script to download model from Hugging Face
│   ├── fill-template.sh # Script to configure the model repository
│   └── start-tritonserver.sh # Script to launch Triton server container
└── src/                # Python source code
    ├── convert_checkpoint.py  # Convert HF model to TRT-LLM format
    ├── fill_template.py       # Configure Triton model repository
    ├── launch_triton_server.py # Start Triton server with proper settings
    ├── quantize.py            # Quantize model to FP8/BF16
    └── summarize_long.py      # Utility for long text summarization
```

## Quick Start Guide

### 0. Download the Model

First, download the Llama-3.1-8B-Instruct model from Hugging Face:

1. Start the container for model preparation:
   ```bash
   bash script/start-tritonserver.sh
   ```

2. Attach to the container bash:
   ```bash
   docker exec -it triton-llama-3.1-8b-instruct bash
   ```

3. Download the model:
   ```bash
   bash script/download.sh
   ```
   > **Note**: This script requires git and git-lfs to be installed, and will download the model from Hugging Face to the `./model` directory.

### 1. Build the TensorRT Engine

With the model downloaded, now build the TensorRT engine:

1. If not already in the container, attach to it:
   ```bash
   docker exec -it triton-llama-3.1-8b-instruct bash
   ```

2. Run the engine build script:
   ```bash
   bash script/build-engine.sh
   ```
   > **Note**: You can modify build parameters in `script/build-engine.sh` to adjust quantization settings. By default, this uses FP8 quantization with a bfloat16 base type and FP8 KV cache.
    
3. Verify that engine and checkpoint files were created in the `./checkpoint/FP8` and `./engine/FP8` directories.

### 2. Configure the Triton Model Repository

Configure the Triton model repository using the templates:

1. If not already in the container, attach to it:
   ```bash
   docker exec -it triton-llama-3.1-8b-instruct bash
   ```

2. Update the repository settings by filling in the model template:
   ```bash
   bash script/fill-template.sh
   ```
   > **Note**: This script copies the `raw-repository` templates to `repository` and configures them with appropriate settings. You can edit environment variables in `script/fill-template.sh` to customize memory usage, batch sizes, and other parameters.

### 3. Start the Triton Inference Server

Now start the Triton server with your configured model:

1. Exit the current container (type `exit`) if you're inside it, and start a deployment container:
   ```bash
   bash script/start-tritonserver.sh --deploy
   ```

2. Check the container logs to verify successful startup:
   ```bash
   docker logs -f triton-llama-3.1-8b-instruct
   ```

3. Once the server is running, it exposes the following endpoints:
   - HTTP endpoint: `http://localhost:8000`
   - gRPC endpoint: `localhost:8001`
   - Metrics endpoint: `http://localhost:8002/metrics`

4. To monitor server performance and resource usage:
   ```bash
   watch -n 1 nvidia-smi
   ```
## Frontend Integration

This model can be deployed with [My-OpenAI-Frontend](https://github.com/HoHuiHsieh/My-OpenAI-Frontend.git), which provides an OpenAI-compatible API interface. This allows you to:

- Use standard OpenAI API clients with your local deployment
- Build applications with a familiar API structure
- Integrate with existing tools that support OpenAI's API format

## Customization

- **Model quantization**: Edit the quantization parameters in `script/build-engine.sh` to adjust precision:
  ```bash
  # For FP8 precision (better performance, slightly lower quality)
  python ./src/quantize.py --qformat fp8 --kv_cache_dtype fp8
  ```

- **Server configuration**: Modify `script/start-tritonserver.sh` to change:
  - Port mappings (`-p` options)
  - Container name (`CONTAINER_NAME` variable)
  - Volume mounts (`-v` options)
  - Docker image version (upgrade to newer Triton versions)

- **Memory usage**: Adjust in `script/fill-template.sh`:
  ```bash
  # Change GPU memory allocation (0.0-1.0)
  export GPU_MEM_FRACTION=0.9
  ```

- **Model pipeline**: Edit templates in the `raw-repository/` directory to customize:
  - Preprocessing and tokenization settings
  - Inference parameters and batch settings
  - Postprocessing behavior

## Performance Considerations

- **Memory usage**: FP8 quantization reduces memory footprint significantly
- **Batch size**: Adjust `MAX_BATCH_SIZE` in `fill-template.sh` to optimize throughput
- **GPU utilization**: Monitor with `nvidia-smi` and adjust `GPU_MEM_FRACTION` accordingly
- **Response time**: The `max_queue_delay_microseconds` parameter affects latency/throughput tradeoff

## Troubleshooting

- **Server fails to start**: 
  - Check GPU memory usage with `nvidia-smi`
  - Ensure there's enough GPU memory available
  - Reduce `GPU_MEM_FRACTION` or `MAX_BATCH_SIZE` in `script/fill-template.sh`

- **Model loading errors**:
  - Check that all model files were downloaded correctly
  - Ensure the `model` directory contains all required files
  - Check the container has access to the model directory

- **Build failures**:
  - Ensure all dependencies are installed correctly as specified in `requirements.txt`
  - Check CUDA and TensorRT versions are compatible
  - Verify GPU has enough memory for the build process

- **Low performance**:
  - Try different quantization settings in `build-engine.sh`
  - Adjust batch size and memory fraction in `fill-template.sh`
  - Check for other processes using GPU resources
