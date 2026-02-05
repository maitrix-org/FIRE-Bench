FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip git \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install general libs from PyPI
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir numpy pandas datasets transformers

# Install PyTorch CUDA 12.1 wheels from the PyTorch index
RUN python3 -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121
