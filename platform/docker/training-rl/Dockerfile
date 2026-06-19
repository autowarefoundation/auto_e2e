FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir timm==1.0.27 \
    mlflow-skinny==2.22.0 webdataset \
    stable-baselines3[extra] gymnasium \
    carla==0.9.15 boto3

COPY Model/ /workspace/Model/

ENV PYTHONPATH=/workspace/Model
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "Model/training/train_rl.py"]
