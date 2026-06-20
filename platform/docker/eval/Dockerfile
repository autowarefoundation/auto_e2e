FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir \
    timm==1.0.27 \
    mlflow-skinny==2.22.0 \
    webdataset \
    flytekit \
    pyyaml \
    numpy

COPY Model/ /workspace/Model/
COPY platform/pipelines/ /workspace/platform/pipelines/

ENV PYTHONPATH=/workspace/Model:/workspace
ENV PYTHONUNBUFFERED=1
