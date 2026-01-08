$WORKSPACE = "/workspace"

# Run the Docker container with the specified configurations
docker run -itd --rm --gpus '"device=0"' `
    --name tensorrtllm-gemma-3-1b-it `
    -v ${PWD}/../../Huggingface/gemma-3-1b-it:${WORKSPACE}/model/gemma-3-1b-it `
    -v ${PWD}:${WORKSPACE}/src `
    -p 8000:8000 `
    -w $WORKSPACE `
    -e NCCL_DEBUG=WARN `
    --ipc=host `
    --ulimit memlock=-1 `
    --ulimit stack=67108864 `
    tensorrt-llm-hf:1.2.0rc6 `
    trtllm-serve ${WORKSPACE}/model/gemma-3-1b-it `
        --host 0.0.0.0 `
        --port 8000
