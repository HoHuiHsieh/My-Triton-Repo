# My-Agent-In-Triton-Repo

This repository contains all necessary components to deploy and serve a multi-functional AI agent using NVIDIA Triton Inference Server with modular components for document retrieval, Python code generation, and planning capabilities.

[![NVIDIA Triton](https://img.shields.io/badge/NVIDIA-Triton_Inference_Server-76B900?style=flat-square&logo=nvidia)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-25A162?style=flat-square)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Framework-3693F3?style=flat-square)](https://python.langchain.com/docs/langgraph)

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Quick Start Guide](#quick-start-guide)
  - [1. Start the Services](#1-start-the-services)
  - [2. Run the Triton Server](#2-run-the-triton-server)
  - [3. Test the Agent](#3-test-the-agent)
- [Agent Capabilities](#agent-capabilities)
- [Customization](#customization)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Overview

This project enables efficient deployment of a modular AI agent using NVIDIA's Triton Inference Server for scalable serving. The agent is built with LangChain and LangGraph frameworks, providing a flexible architecture for different capabilities including document retrieval, Python code generation, and complex planning.

The pipeline includes:
- A document retrieval module for accessing and searching knowledge bases
- A Python code generator for creating and executing Python code
- A planning module for decomposing complex tasks
- Integration with PostgreSQL database for vector storage
- All components deployed as containerized services using Docker and Docker Compose

## Prerequisites

- NVIDIA GPU (optional but recommended for better performance)
- Docker and Docker Compose installed and configured
- OpenAI API key for LLM access
- At least 8GB RAM
- Internet connection for pulling container images and accessing external APIs

## Repository Structure

```
├── docker/               # Docker configuration files
│   ├── docker-compose.yml  # Docker Compose configuration
│   └── Dockerfile          # Docker image definition
├── repository/           # Triton model repository structure
│   └── my-agent-for-test/  # Agent model configuration
│       ├── config.pbtxt    # Triton model configuration
│       └── 1/              # Model version directory
│           ├── model.py    # Triton model implementation
│           └── my_agent/   # Agent implementation modules
│               ├── doc_retriever/  # Document retrieval module
│               ├── py_coder/       # Python code generation module
│               ├── planner.py      # Planning module
│               ├── state.py        # Agent state management
│               ├── summarize.py    # Summarization utilities
│               └── __init__.py     # Module initialization
├── script/               # Utility scripts
│   ├── serve.sh          # Script to serve models in Triton
│   └── start.sh          # Script to start all services
└── test/                 # Test and demo scripts
    ├── agent_demo.py     # Demo for the complete agent
    ├── doc_retriever_demo.py # Demo for document retriever
    ├── py_coder_demo.py  # Demo for Python code generation
    └── utils.py          # Utility functions for testing
```

## Quick Start Guide

### 1. Start the Services

Start the Docker containers using the provided script:

```bash
bash script/start.sh
```

This will:
- Build and start the Triton server container
- Start the PostgreSQL database with pgvector extension
- Configure the necessary volumes and network connections

### 2. Run the Triton Server

Once the containers are running, you need to start the Triton server:

1. Access the container's shell:
   ```bash
   docker exec -it my-agent-in-triton bash
   ```

2. Start the Triton server:
   ```bash
   bash script/serve.sh
   ```

3. The server will be accessible at:
   - HTTP endpoint: `http://localhost:8000`

### 3. Test the Agent

You can test the agent using the provided demo scripts:

1. Access the container shell (if not already in it):
   ```bash
   docker exec -it my-agent-in-triton bash
   ```

2. Run one of the demo scripts:
   ```bash
   python test/agent_demo.py
   ```

You can also run specific module demos:
- `python test/doc_retriever_demo.py` for document retrieval capabilities
- `python test/py_coder_demo.py` for Python code generation testing

## Agent Capabilities

The agent integrates several key capabilities:

- **Document Retrieval**: Search and retrieve information from document sources
- **Python Code Generation**: Generate and execute Python code based on requirements
- **Planning**: Break down complex tasks into manageable steps
- **State Management**: Maintain conversation and execution state across interactions
- **LLM Integration**: Connect with large language models via OpenAI API

## Customization

- **Agent modules**: Add or modify capabilities by editing or adding modules in `repository/my-agent-for-test/1/my_agent/`
- **Server configuration**: Modify `script/serve.sh` to change Triton server parameters:
  ```bash
  # Example customization
  tritonserver \
    --model-repository=/workspace/repository \
    --model-control-mode=poll \
    --repository-poll-secs=5 \
    --http-thread-count=8
  ```

- **Docker settings**: Edit `docker/docker-compose.yml` to adjust:
  - Port mappings
  - Resource allocations
  - Volume mounts
  - Container configurations

## Performance Considerations

- **Database optimization**: Fine-tune PostgreSQL settings for vector operations in `docker/docker-compose.yml`
- **Model caching**: Enable model caching in Triton server to improve response times
- **Resource allocation**: Adjust container memory and CPU limits based on workload
- **Connection pooling**: Implement connection pooling for database operations in high-load scenarios

## Troubleshooting

- **Server fails to start**: 
  - Check Docker logs: `docker logs my-agent-in-triton`
  - Verify port availability: `netstat -tuln | grep 8000`
  - Ensure resource availability: `docker stats`

- **Agent errors**:
  - Check OpenAI API key is correctly set
  - Verify PostgreSQL connection: `docker exec -it postgres psql -U postgresql -d postgres`
  - Inspect Triton server logs: `docker exec -it my-agent-in-triton cat /logs/triton.log`

- **Performance issues**:
  - Monitor resource usage: `docker stats`
  - Check database query performance
  - Consider scaling up container resources in `docker/docker-compose.yml`