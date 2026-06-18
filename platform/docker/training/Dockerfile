FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir timm==1.0.27 pytest==9.0.3 \
    mlflow-skinny==2.22.0 flytekit==1.16.23 \
    flytekitplugins-kfpytorch==1.16.23

COPY Model/ /workspace/Model/

ENV PYTHONPATH=/workspace/Model
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "Model/training/train.py"]
