# !/bin/bash

# Load environment variables from .env file
# if [ -f .env ]; then
#     export $(grep -v '^#' .env | xargs)
# fi
export WORKSPACE=/workspace

# Run the Docker container with the specified configurations
docker run -itd --rm --gpus '"device=0"' \
    --name tritonserver-whisper-tiny \
    -v ${PWD}/../../Huggingface/whisper-tiny:${WORKSPACE}/model/whisper-tiny \
    -v ${PWD}/repository:${WORKSPACE}/repository \
    -v ${PWD}:${WORKSPACE}/src \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -w $WORKSPACE \
    tritonserver:25.12-whisper-tiny \
    tritonserver --model-repository=${WORKSPACE}/repository \
                 --model-control-mode=poll \
                 --repository-poll-secs=1