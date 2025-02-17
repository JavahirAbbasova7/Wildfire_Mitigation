# Start from a base image with CUDA and cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive 

# Install required tools and Python 3.9
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    vim \
    ca-certificates \
    software-properties-common \
    build-essential \
    xpra \
    tmux \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y python3.9 python3.9-dev python3.9-distutils \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install dependencies with Python 3.9's pip
RUN python -m pip install --no-cache-dir simfire
# Install additional library for OpenCV compatibility
RUN apt update

# Set working directory
WORKDIR /app