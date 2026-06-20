FROM public.ecr.aws/docker/library/python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# CPU-only torch first (lerobot-dataset omits torch, so ours stays)
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

# lerobot-dataset = standalone LeRobotDataset with relaxed deps (no torch override)
RUN pip install --no-cache-dir \
    "flytekit>=1.14,<2" \
    lerobot-dataset \
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
