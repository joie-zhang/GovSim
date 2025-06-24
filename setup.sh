#!/bin/bash

# Create conda environment
conda create -n GovSim python=3.11.5 -y

# Initialize conda for bash (needed for activation to work in scripts)
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate GovSim

# Install PyTorch for macOS (CPU-only, no CUDA)
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch -y

# Skip CUDA toolkit installation on macOS (not supported)
# conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit cuda -y

# Install other conda packages
conda install conda-forge::weasyprint -y
conda install -c conda-forge python-kaleido -y

# Install pip packages
pip install -r pathfinder/requirements.txt
pip install auto-gptq
pip install bitsandbytes

pip install -r requirements.txt
pip install "numpy==1.26.4"

pip install transformers

echo "Setup complete! To use the environment, run: conda activate GovSim"