#!/bin/bash
# This script is used to download the Llama 3.1 8B Instruct model from Hugging Face.

# Ensure you have git and git-lfs installed before running this script.
apt update && apt install -y git git-lfs

# Initialize git-lfs if not already done
git lfs install

# Clone the Llama 3.1 8B Instruct model repository from Hugging Face
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

# Move the downloaded model to the desired directory
mv -r ./Llama-3.1-8B-Instruct ./model