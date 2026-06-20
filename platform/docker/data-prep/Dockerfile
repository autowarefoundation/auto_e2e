FROM public.ecr.aws/docker/library/python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# PyTorch CPU-only + torchvision
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Data pipeline deps
RUN pip install --no-cache-dir \
    "flytekit>=1.14,<2" \
    "lerobot>=0.5.1" \
    huggingface_hub \
    datasets \
    webdataset \
    Pillow \
    numpy \
    av \
    timm \
    scipy \
    boto3

COPY Model/ /workspace/Model/
COPY platform/pipelines/ /workspace/platform/pipelines/

ENV PYTHONPATH=/workspace/Model:/workspace
ENV PYTHONUNBUFFERED=1
