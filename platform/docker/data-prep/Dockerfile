FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# No GPU used; base provides PyTorch (needed by lerobot) + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir \
    lerobot==0.5.1 \
    flytekit==1.16.23 \
    boto3 pyarrow pandas scipy pillow \
    webdataset

# physical_ai_av installed separately (may need --no-deps if conflicts)
RUN pip install --no-cache-dir physical_ai_av || \
    pip install --no-cache-dir --no-deps physical_ai_av

COPY Model/ /workspace/Model/
COPY platform/pipelines/ /workspace/platform/pipelines/

ENV PYTHONPATH=/workspace/Model:/workspace
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]
