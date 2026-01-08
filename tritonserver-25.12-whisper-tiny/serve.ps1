# PowerShell script to run Triton server for whisper-tiny
# Usage: .\serve.ps1

# Set WORKSPACE environment variable
$env:WORKSPACE = "/workspace"

# Get current directory path and convert to Unix-style path for Docker
$currentPath = (Get-Location).Path -replace '\\', '/'
$driveLetter = $currentPath.Substring(0, 1).ToLower()
$unixPath = $currentPath.Substring(2) -replace '^\\', '/'
$dockerPath = "/$driveLetter$unixPath"

# Run the Docker container with the specified configurations
docker run -itd --rm --gpus '"device=0"' `
    --name tritonserver-whisper-tiny `
    -v "${dockerPath}/../../Huggingface/whisper-tiny:${env:WORKSPACE}/model/whisper-tiny" `
    -v "${dockerPath}/repository:${env:WORKSPACE}/repository" `
    -v "${dockerPath}:${env:WORKSPACE}/src" `
    -p 8000:8000 `
    -p 8001:8001 `
    -p 8002:8002 `
    -w $env:WORKSPACE `
    tritonserver:25.12-whisper-tiny `
    tritonserver --model-repository=${env:WORKSPACE}/repository `
                 --model-control-mode=poll `
                 --repository-poll-secs=1
