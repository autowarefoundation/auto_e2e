FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir \
    timm==1.0.27 \
    mlflow-skinny==2.22.0 \
    webdataset \
    flytekit==1.16.24 \
    fsspec==2026.6.0 \
    s3fs==2026.6.0 \
    gcsfs==2026.7.0 \
    adlfs==2026.5.0 \
    kubernetes \
    pyyaml \
    numpy \
    pyproj \
    boto3==1.43.0

COPY Model/ /workspace/Model/
COPY Platform/pipelines/ /workspace/Platform/pipelines/

ENV PYTHONPATH=/workspace/Model:/workspace
ENV PYTHONUNBUFFERED=1
