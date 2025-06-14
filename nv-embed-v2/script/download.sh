#!/bin/bash
# This script is used to download the NV-Embed-v2 Instruct model from Hugging Face.

# Ensure you have git and git-lfs installed before running this script.
apt update && apt install -y git git-lfs

# Initialize git-lfs if not already done
git lfs install

# Clone the NV-Embed-v2 Instruct model repository from Hugging Face
git clone git clone https://huggingface.co/nvidia/NV-Embed-v2

# Move the downloaded model to the desired directory
mv -r ./NV-Embed-v2 ./model