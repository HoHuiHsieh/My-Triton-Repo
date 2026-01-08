# PowerShell script to build Docker image
# Usage: .\build.ps1

# Set image name and Dockerfile path
$imageName = "tritonserver:25.12-transformers"
$dockerfile = "Dockerfile"

# Build the Docker image
docker build -t $imageName -f $dockerfile .
