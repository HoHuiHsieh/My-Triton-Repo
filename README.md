# My-Triton-Repo

This repository provides examples of deploying AI models using NVIDIA Triton Inference Server for home and small-scale deployments. The setup leverages TensorRT-LLM for Large Language Models (LLMs) and PyTorch for embedding models, all packaged with Docker for easy containerized deployment.

[![NVIDIA Triton](https://img.shields.io/badge/NVIDIA-Triton_Inference_Server-76B900?style=flat-square&logo=nvidia)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
[![TensorRT LLM](https://img.shields.io/badge/NVIDIA-TensorRT--LLM-76B900?style=flat-square&logo=nvidia)](https://github.com/NVIDIA/TensorRT-LLM)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![Llama 3.1](https://img.shields.io/badge/Meta-Llama_3.1-0467DF?style=flat-square&logo=meta)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
[![NVIDIA Embed](https://img.shields.io/badge/NVIDIA-NV--Embed--v2-76B900?style=flat-square&logo=nvidia)](https://huggingface.co/nvidia/NV-Embed-v2)

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Available Models](#available-models)
- [Getting Started](#getting-started)
- [Frontend Integration](#frontend-integration)
- [System Requirements](#system-requirements)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project demonstrates how to efficiently deploy and serve AI models using NVIDIA's Triton Inference Server. The repository includes complete deployment configurations for:

- State-of-the-art Large Language Models (LLMs) with TensorRT-LLM acceleration
- Embedding models for vector search and semantic analysis
- Optimized inference pipelines with preprocessing and postprocessing

By following the examples in this repository, you can set up a production-quality AI inference system on your own hardware, with performance optimizations like quantization and batching to maximize throughput.

## Repository Structure

```
├── meta-llama-3.1-8B-Instruct/  # Meta's Llama 3.1 8B LLM implementation
│   ├── checkpoint/              # Converted model checkpoints 
│   ├── engine/                  # TensorRT-LLM engine files
│   ├── model/                   # Original model weights
│   ├── raw-repository/          # Triton model template files
│   ├── repository/              # Configured Triton model repository
│   ├── script/                  # Utility scripts for setup and deployment
│   └── src/                     # Python source code for model preparation
├── my-agent/                    # Multi-functional AI agent implementation
│   ├── docker/                  # Docker configuration files
│   ├── repository/              # Triton model repository with agent modules
│   ├── script/                  # Setup and deployment scripts
│   └── test/                    # Test and demo scripts
├── nv-embed-v2/                 # NVIDIA's Embedding v2 model implementation
│   ├── model/                   # Model weights directory
│   ├── repository/              # Triton model repository
│   └── script/                  # Setup and deployment scripts
├── LICENSE                      # License file for the repository
└── README.md                    # This readme file
```

## Available Models

This repository currently includes the following models:

### Meta/Llama-3.1-8B-Instruct

A state-of-the-art open-weights language model from Meta, deployed with TensorRT-LLM optimization:
- FP8 quantized for efficient inference
- Complete Triton pipeline with preprocessing, inference, and postprocessing
- Optimized for consumer-grade GPUs (RTX series)

[➡️ Llama 3.1 Deployment Guide](meta-llama-3.1-8B-Instruct/README.md)

### My-Agent

A modular AI agent built with LangChain and LangGraph frameworks:
- Document retrieval module for accessing and searching knowledge bases
- Python code generation capabilities for creating and executing code
- Planning module for decomposing complex tasks
- Integration with PostgreSQL for vector storage
- Containerized deployment with Docker and Docker Compose

[➡️ My-Agent Deployment Guide](my-agent/README.md)

### NVIDIA/NV-Embed-v2

NVIDIA's embedding model for generating text representations:
- 768-dimensional embeddings suitable for vector databases
- PyTorch-based inference backend
- Optimized for semantic search and text similarity

[➡️ NV-Embed Deployment Guide](nv-embed-v2/README.md)

## Getting Started

To get started with any model in this repository:

1. Clone this repository:
   ```bash
   git clone https://github.com/HoHuiHsieh/My-Triton-Repo.git
   cd My-Triton-Repo
   ```

2. Navigate to the model directory you want to deploy:
   ```bash
   cd meta-llama-3.1-8B-Instruct  # or nv-embed-v2
   ```

3. Follow the model-specific README instructions for downloading, setting up, and deploying the model.

## Frontend Integration

These models can be deployed with [My-OpenAI-Frontend](https://github.com/HoHuiHsieh/My-OpenAI-Frontend.git), which provides an OpenAI-compatible API interface. This allows you to:

- Use standard OpenAI API clients with your local deployment
- Build applications with a familiar API structure
- Integrate with existing tools that support OpenAI's API format

## System Requirements

- **GPU**: NVIDIA GPU with compute capability 8.0+ for LLMs (RTX 3000 series or newer), 7.0+ for embedding models
- **RAM**: Minimum 16GB system RAM, 24GB+ recommended
- **GPU Memory**: 
  - Llama 3.1 8B: Minimum 16GB VRAM (24GB+ recommended)
  - NV-Embed-v2: Minimum 8GB VRAM
  - My-Agent: CPU execution with optional GPU acceleration
- **Storage**: At least 50GB free disk space for model weights and engines
- **Software**: 
  - Docker with NVIDIA Container Toolkit
  - NVIDIA drivers version 575 or later
  - Docker Compose for My-Agent
  - Internet connection for My-Agent's OpenAI API access

## License

This repository is provided under the GPL-3.0. 
The models themselves may have their own licenses - please refer to the original model sources for details:

- [Meta Llama 3.1 License](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [NVIDIA NV-Embed-v2 License](https://huggingface.co/nvidia/NV-Embed-v2)

## Contact

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/hohuihsieh)
