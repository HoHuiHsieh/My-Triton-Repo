# NVIDIA NV-Embed-v2 Model on NVIDIA Triton Inference Server

This repository contains all necessary components to deploy and serve NVIDIA's Embedding v2 model using NVIDIA Triton Inference Server with PyTorch acceleration.

[![NVIDIA Triton](https://img.shields.io/badge/NVIDIA-Triton_Inference_Server-76B900?style=flat-square&logo=nvidia)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
[![NVIDIA Embed](https://img.shields.io/badge/NVIDIA-NV--Embed--v2-76B900?style=flat-square&logo=nvidia)](https://huggingface.co/nvidia/NV-Embed-v2)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Quick Start Guide](#quick-start-guide)
  - [0. Download the Model](#0-download-the-model)
  - [1. Configure the Triton Model Repository](#1-configure-the-triton-model-repository)
  - [2. Start the Triton Inference Server](#2-start-the-triton-inference-server)
- [Frontend Integration](#frontend-integration)
- [Customization](#customization)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Overview

This project enables efficient deployment of NVIDIA's NV-Embed-v2 model ([nvidia/NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2)) using NVIDIA's Triton Inference Server for scalable serving. The implementation includes a Python backend that uses PyTorch for inference.

The pipeline includes:
- Configuring the NVIDIA NV-Embed-v2 model for Triton Inference Server
- Optimized implementation for generating embeddings from textual input

## Prerequisites

- NVIDIA GPU with compute capability 7.0 or higher (Volta architecture or newer)
- Docker with NVIDIA Container Toolkit installed and configured
- Minimum 8GB GPU memory (16GB+ recommended)
- At least 20GB free disk space for model weights
- Internet connection for downloading the model

## Repository Structure

```
├── repository/            # Generated Triton model repository structure
│   ├── nv-embed-v2/       # Main model configuration
│   │   ├── config.pbtxt   # Triton model configuration
│   │   └── 1/             # Model version 1
│   │       └── model.py   # Python inference backend
├── script/                # Utility scripts
│   ├── start-tritonserver.sh # Script to launch Triton server container
│   └── download.sh        # Script to download model from Hugging Face
├── model/                 # Directory for model weights
```

## Quick Start Guide

### 0. Download the Model

First, download the NVIDIA NV-Embed-v2 model from Hugging Face:

1. Start the container for model preparation:
   ```bash
   bash script/start-tritonserver.sh
   ```

2. Attach to the container bash:
   ```bash
   docker exec -it triton-nv-embed-v2 bash
   ```

3. Download the model:
   ```bash
   # Execute the download script to fetch the model from Hugging Face
   bash script/download.sh
   ```
   
   > **Note**: This script requires git and git-lfs to be installed, and will download the NV-Embed-v2 model from Hugging Face to the `./model` directory.

### 1. Configure the Triton Model Repository

The Triton model repository is already configured in the `/workspace/repository` directory. You can review and modify the configuration in `/workspace/repository/nv-embed-v2/config.pbtxt` if needed.

### 2. Start the Triton Inference Server

Now start the Triton server with your configured model:

1. Exit the current container (type `exit`) if you're inside it, and start a deployment container:
   ```bash
   bash script/start-tritonserver.sh --deploy
   ```

2. Check the container logs to verify successful startup:
   ```bash
   docker logs -f triton-nv-embed-v2
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

This model can be integrated with various applications that need text embeddings, such as:

- Vector databases like Milvus, Pinecone, or FAISS
- Semantic search applications
- Text similarity comparison tools
- Recommendation systems
- Custom applications via the Triton client API

## Customization

- **Server configuration**: Modify `script/start-tritonserver.sh` to change:
  - Port mappings (`-p` options)
  - Container name (`CONTAINER_NAME` variable)
  - Volume mounts (`-v` options)
  - Docker image version (upgrade to newer Triton versions)

- **Model configuration**: Edit `repository/nv-embed-v2/config.pbtxt` to customize:
  - Batch size settings
  - Input and output tensor configurations
  - Instance group count for scaling

- **Model behavior**: Modify `repository/nv-embed-v2/1/model.py` to customize:
  - Preprocessing steps
  - Embedding generation logic
  - Maximum input length
  - Instruction prefixing for queries

## Performance Considerations

- **Batch size**: Adjust `max_batch_size` in `config.pbtxt` to optimize throughput
- **GPU utilization**: Monitor with `nvidia-smi` and adjust instance count accordingly
- **Memory usage**: The model uses PyTorch for inference, adjust batch size if you encounter OOM errors
- **Response time**: Consider the trade-off between batch size and latency for your use case

## Troubleshooting

- **Server fails to start**: 
  - Check GPU memory usage with `nvidia-smi`
  - Ensure there's enough GPU memory available
  - Verify the model files are properly downloaded

- **Model loading errors**:
  - Check that all model files were downloaded correctly
  - Ensure the `model` directory contains all required files
  - Check the container has access to the model directory

- **Inference errors**:
  - Verify input formats match the expected formats in `model.py`
  - Check token length is within the model's maximum length
  - Inspect server logs for detailed error messages

- **Low performance**:
  - Try adjusting batch size in `config.pbtxt`
  - Consider increasing instance count for multi-GPU setups
  - Check for other processes using GPU resources
