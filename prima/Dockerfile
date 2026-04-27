# Dockerfile.base
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-9 \
    make \
    cmake \
    fio \
    git \
    wget \
    libzmq3-dev \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install HiGHS
RUN git clone https://github.com/ERGO-Code/HiGHS.git && \
    cd HiGHS && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf HiGHS

# Clone and build PRIMA.cpp
RUN git clone https://gitee.com/zonghang-li/prima.cpp.git
RUN cd prima.cpp && make GGML_CUDA=1 USE_HIGHS=1 -j$(nproc)

# Set working directory
WORKDIR /prima.cpp

CMD ["/bin/bash"]