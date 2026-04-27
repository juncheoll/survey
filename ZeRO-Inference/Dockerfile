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

# Clone DeepSpeedExamples and modify the script
RUN git clone https://github.com/deepspeedai/DeepSpeedExamples.git
RUN sed -i '22s/.*/try:\n    from transformers.integrations.deepspeed import HfDeepSpeedConfig\nexcept Exception:\n    from transformers.deepspeed import HfDeepSpeedConfig/' /DeepSpeedExamples/inference/huggingface/zero_inference/run_model.py

# Create uv venv and install requirements
RUN uv venv /opt/venvs/zero
RUN uv pip install --python /opt/venvs/zero/bin/python -r /DeepSpeedExamples/inference/huggingface/zero_inference/requirements.txt

# Install mpi4py for deepspeed
RUN uv pip install --python /opt/venvs/zero/bin/python mpi4py

# Set up automatic activation and cd in .bashrc
RUN echo "source /opt/venvs/zero/bin/activate" >> /root/.bashrc && \
    echo "cd /DeepSpeedExamples/inference/huggingface/zero_inference" >> /root/.bashrc

# Set working directory
WORKDIR /DeepSpeedExamples/inference/huggingface/zero_inference

COPY run_inference_benchmark.sh /DeepSpeedExamples/inference/huggingface/zero_inference/run_inference_benchmark.sh
RUN chmod +x /DeepSpeedExamples/inference/huggingface/zero_inference/run_inference_benchmark.sh

CMD ["/bin/bash"]