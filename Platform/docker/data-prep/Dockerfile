FROM public.ecr.aws/docker/library/python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# CPU-only torch first (lerobot-dataset omits torch, so ours stays)
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

# lerobot-dataset = standalone LeRobotDataset with relaxed deps (no torch override)
# boto3<1.36: unpinned boto3 makes pip backtrack for many minutes — it downloads
# metadata for dozens of recent boto3 releases (1.43.x → …) trying to satisfy
# flytekit 1.14's botocore/s3transfer ceiling. Capping boto3 lets the resolver
# land immediately on a compatible 1.35.x. boto3 is used by label_cache.py (S3).
RUN pip install --no-cache-dir \
    "flytekit>=1.14,<2" \
    kubernetes \
    lerobot-dataset \
    physical_ai_av \
    huggingface_hub \
    datasets \
    webdataset \
    Pillow \
    numpy \
    av \
    timm \
    scipy \
    "boto3<1.36" \
    pyarrow

COPY Model/ /workspace/Model/
COPY Platform/pipelines/ /workspace/Platform/pipelines/

ENV PYTHONPATH=/workspace/Model:/workspace
ENV PYTHONUNBUFFERED=1
