[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/HoHuiHsieh-607b70170/)

# NV Triton Service with Hugging Face LLM using TensorRT-LLM and Docker
This repository provides a complete example of deploying a Hugging Face large language model (LLM) as an NVIDIA Triton Inference Server service. The setup leverages TensorRT-LLM for optimized model execution and Docker for easy containerized deployment.

## Features:
- **End-to-End Deployment:** Streamlines the process of integrating Hugging Face LLMs with NVIDIA Triton for efficient inference.
- **TensorRT-LLM Acceleration:** Utilizes TensorRT-LLM to enhance performance and reduce latency.
- **Containerized Environment:** Uses Docker to simplify setup and ensure reproducibility.
- **Scalability & Optimization:** Leverages Triton’s dynamic batching and model management for scalable inference.

## Getting Started
Follow the step-by-step instructions to build, configure, and deploy your own Triton inference service for Hugging Face models.

📌 Ideal for developers, AI researchers, and engineers looking to optimize LLM inference with NVIDIA’s AI stack.

## Prerequisites

- Docker
- Git LFS
- jq (for JSON processing)

## Steps

### 1. Download LLM from Huggingface

```bash
export MODEL_PATH="../models/Llama-3.2-1B"

git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.2-1B "$MODEL_PATH"
```

### 2. Get TensorRT-LLM

```bash
export WORKDIR="$PWD"
export TRTLLM_DIR="3rdparty/TensorRT-LLM"
export TRTLLM_TAG="v0.17.0"

if [ -d "$WORKDIR/$TRTLLM_DIR" ]; then
    echo "$TRTLLM_DIR does exist."
    cd "$WORKDIR/$TRTLLM_DIR"
    git fetch
    git checkout "$TRTLLM_TAG"
else
    git clone https://github.com/NVIDIA/TensorRT-LLM.git "$TRTLLM_DIR"
    cd "$WORKDIR/$TRTLLM_DIR"
    git checkout "$TRTLLM_TAG"
fi
```

### 3. Get Triton Inference Server

```bash
export TRITON_DIR="3rdparty/Triton-trtllm_backend"
export TRITON_TAG="v0.17.0"

if [ -d "$WORKDIR/$TRITON_DIR" ]; then
    echo "$TRITON_DIR does exist."
    cd "$WORKDIR/$TRITON_DIR"
    git fetch
    git checkout "$TRITON_TAG"
else
    git clone https://github.com/triton-inference-server/tensorrtllm_backend.git "$TRITON_DIR"
    cd "$WORKDIR/$TRITON_DIR"
    git checkout "$TRITON_TAG"
fi
```

### 4. Build trtllm engine

```bash
export TRTLLM_EXAMPLE_MODEL_DIR="$TRTLLM_DIR/examples/llama"
export MODEL_CKPT_PATH="model/ckpt"
export MODEL_ENGINE_PATH="model/engine"
export BUILDER_SCRIPT="script/build-llama-trtllm.sh"

cd $WORKDIR
docker build --target builder_container \
    -t llm/trtllm-builder:$TRTLLM_TAG \
    --build-arg HOST_TRTLLM_DIR=$TRTLLM_EXAMPLE_MODEL_DIR \
    .
docker run -it --rm --gpus all \
    -v $WORKDIR/$MODEL_PATH:/workspace/model \
    -v $WORKDIR/$MODEL_CKPT_PATH:/workspace/ckpt \
    -v $WORKDIR/$MODEL_ENGINE_PATH:/workspace/engine \
    -v $WORKDIR/$BUILDER_SCRIPT:/workspace/build.sh \
    -w /workspace \
    llm/trtllm-builder:$TRTLLM_TAG ./build.sh
```

### 5. Serve model with Triton Inference Server

```bash
export CONTAINER_TAG="self-host-llm/triton-llama3.2-1b:latest"
export TRITON_REPO_DIR="3rdparty/Triton-trtllm_backend/all_models/inflight_batcher_llm"
export CONFIG_FILL_SCRIPT="script/fill_template.py"
export RUN_SERVE_SCRIPT="script/launch_triton_server.py"

cd $WORKDIR
docker build --no-cache --target server_container \
    -t $CONTAINER_TAG \
    --build-arg HOST_REPO_DIR=$TRITON_REPO_DIR \
    --build-arg CONFIG_FILL_SCRIPT=$CONFIG_FILL_SCRIPT \
    --build-arg RUN_SERVE_SCRIPT=$RUN_SERVE_SCRIPT \
    .
docker run -itd --rm --gpus "device=0" \
    -v $WORKDIR/$MODEL_ENGINE_PATH:/workspace/engine \
    -v $WORKDIR/$MODEL_PATH:/workspace/tokenizor \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -w /workspace \
    $CONTAINER_TAG 
```

### 6. Test Triton Inference Server

```bash
RESPONSE=$(curl -s -w "\nHTTP_STATUS_CODE:%{http_code}\n" -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "id":"0",
        "inputs":[
            {
                "name":"text_input",
                "shape":[1,1],
                "datatype":"BYTES",
                "parameters":{},
                "data":[["Who are you?"]]
            },
            {
                "name":"max_tokens",
                "shape":[1,1],
                "datatype":"INT32",
                "parameters":{},
                "data":[[128]]
            }
        ]
    }' \
    http://localhost:8000/v2/models/ensemble/infer)

HTTP_STATUS=$(echo "$RESPONSE" | grep "HTTP_STATUS_CODE" | awk -F: '{print $2}')
BODY=$(echo "$RESPONSE" | sed '/HTTP_STATUS_CODE/d')

if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "Request was successful. Response:"
    echo "$BODY" | jq
else
    echo "Request failed with status code $HTTP_STATUS. Response:"
    echo "$BODY"
fi
