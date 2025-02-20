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
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    llvm \
    make \
    tk-dev \
    xz-utils \
    zlib1g-dev \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y python3.9 python3.9-dev python3.9-distutils \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone simharness repository
RUN git clone https://github.com/mitrefireline/simharness.git /app/simharness

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/simharness/requirements.txt
RUN pip install ray
RUN pip install pyarrow
RUN pip install pandas
RUN pip install ray[tune]
RUN pip install ray[rllib]