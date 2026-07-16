FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir \
    timm==1.0.27 \
    mlflow-skinny==2.22.0 \
    webdataset \
    flytekit==1.14.9 \
    kubernetes \
    pyyaml \
    numpy \
    pyproj \
    boto3

COPY Model/ /workspace/Model/
COPY Platform/pipelines/ /workspace/Platform/pipelines/

ENV PYTHONPATH=/workspace/Model:/workspace
ENV PYTHONUNBUFFERED=1
