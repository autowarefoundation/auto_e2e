FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

ENV HF_HOME=/opt/model-cache/huggingface
ENV TORCH_HOME=/opt/model-cache/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    timm==1.0.27 \
    mlflow-skinny==3.13.0 \
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

RUN python -c \
    "import torch; assert torch.__version__.startswith('2.7.1'); assert torch.version.cuda == '12.8'"

# Cache the exact Stage A backbone in the immutable training image so all
# workers avoid identical runtime downloads.
# AutoBEV's DINOv2 ViT-B/14 (registers) weights are baked in: training pods do
# not reach Hugging Face.
RUN python -c \
    "import timm; timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True, num_classes=0)"

COPY Model/ /workspace/Model/
COPY Platform/pipelines/ /workspace/Platform/pipelines/

ENV PYTHONPATH=/workspace/Model:/workspace
ENV PYTHONUNBUFFERED=1
ENV RAY_TRAIN_V2_ENABLED=1
