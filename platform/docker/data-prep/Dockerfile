FROM public.ecr.aws/docker/library/python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# PyTorch CPU-only
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# lerobot without hardware deps (evdev etc) — install --no-deps then add what we need
RUN pip install --no-cache-dir lerobot==0.5.1 --no-deps && \
    pip install --no-cache-dir datasets huggingface_hub safetensors pillow numpy

# flytekit + data pipeline deps
RUN pip install --no-cache-dir "flytekit>=1.14,<2" boto3 pyarrow scipy webdataset

# physical_ai_av (best-effort)
RUN pip install --no-cache-dir physical_ai_av --no-deps 2>/dev/null || true

COPY Model/ /workspace/Model/
COPY platform/pipelines/ /workspace/platform/pipelines/

ENV PYTHONPATH=/workspace/Model:/workspace
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]
