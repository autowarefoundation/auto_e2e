FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir timm==1.0.27 \
    mlflow-skinny==2.22.0 webdataset flytekit

COPY Model/ /workspace/Model/

ENV PYTHONPATH=/workspace/Model
ENV PYTHONUNBUFFERED=1

