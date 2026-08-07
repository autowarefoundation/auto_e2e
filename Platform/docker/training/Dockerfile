FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

ENV HF_HOME=/opt/model-cache/huggingface
ENV TORCH_HOME=/opt/model-cache/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    timm==1.0.27 \
    mlflow-skinny==2.22.0 \
    webdataset \
    flytekit==1.16.24 \
    flytekitplugins-ray==1.16.24 \
    "ray[train]==2.46.0" \
    fsspec==2026.6.0 \
    s3fs==2026.6.0 \
    gcsfs==2026.7.0 \
    adlfs==2026.5.0 \
    kubernetes \
    pyyaml \
    numpy \
    pyproj==3.7.2 \
    boto3==1.43.0

# Cache the exact Stage A backbone in the immutable training image so all
# workers avoid identical runtime downloads.
RUN python -c \
    "import timm; timm.create_model('swinv2_tiny_window8_256', pretrained=True, features_only=True)"

COPY Model/ /workspace/Model/
COPY Platform/pipelines/ /workspace/Platform/pipelines/

ENV PYTHONPATH=/workspace/Model:/workspace
ENV PYTHONUNBUFFERED=1
ENV RAY_TRAIN_V2_ENABLED=1
