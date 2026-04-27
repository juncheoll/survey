# Dockerfile.base
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    curl \
    wget \
    ca-certificates \
    build-essential \
    pkg-config \
    cmake \
    ninja-build \
    numactl \
    libaio-dev \
    openmpi-bin \
    libopenmpi-dev \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Make python3 point to python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip and install uv for per-framework env management
RUN python3 -m pip install --upgrade pip setuptools wheel uv

# Git LFS init
RUN git lfs install

# Clone FlexLLMGen and install
RUN git clone https://github.com/FMInference/FlexLLMGen.git
RUN uv venv /opt/venvs/flexgen
RUN uv pip install --python /opt/venvs/flexgen/bin/python -e /FlexLLMGen/.
RUN uv pip install --python /opt/venvs/flexgen/bin/python transformers

# Set up automatic activation and cd in .bashrc
RUN echo "source /opt/venvs/flexgen/bin/activate" >> /root/.bashrc && \
    echo "cd /FlexLLMGen" >> /root/.bashrc

# Set working directory
WORKDIR /FlexLLMGen

COPY run_flexgen_benchmark.sh run_flexgen_benchmark.sh
RUN chmod +x run_flexgen_benchmark.sh

CMD ["/bin/bash"]