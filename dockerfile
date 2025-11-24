# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    TORCH_CUDA_ARCH_LIST="9.0" \
    PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" \
    PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
COPY apt-packages.txt /tmp/apt-packages.txt
RUN apt-get update && \
    apt-get install -y --allow-change-held-packages --no-install-recommends $(xargs -r < /tmp/apt-packages.txt) && \
    rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------
RUN python3 -m pip install --upgrade pip==24.0 setuptools==80.9.0 wheel==0.45.1

COPY requirements-base.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN python3 -m pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    torchaudio==2.1.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

RUN python3 -m pip install --no-cache-dir \
    torch-scatter==2.1.2+pt21cu121 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# ------------------------------------------------------------
# Project code
# ------------------------------------------------------------
WORKDIR /
COPY . /OpenDriveVLA/
RUN mkdir -p /workspace && ln -s /OpenDriveVLA /workspace/OpenDriveVLA

# Build mmcv from source (editable)
RUN cd /OpenDriveVLA/third_party/mmcv_1_7_2 && \
    export CUDA_HOME=/usr/local/cuda && \
    MMCV_WITH_OPS=1 python3 -m pip install -v -e . --no-build-isolation

# Install mmdetection3d (editable, no deps)
RUN cd /OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6 && \
    python3 -m pip install -e . --no-build-isolation --no-deps

# Environment
ENV PYTHONPATH=/OpenDriveVLA:/OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6:/workspace/OpenDriveVLA:/workspace/OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6

WORKDIR /OpenDriveVLA

CMD ["/bin/bash"]
