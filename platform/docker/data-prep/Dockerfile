FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install PyTorch CPU-only first (lerobot requires it)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install flytekit first (has strict version bounds)
RUN pip install --no-cache-dir flytekit==1.16.23 boto3 pyarrow pandas scipy pillow webdataset

# Install lerobot (needs torch + Python 3.12)
RUN pip install --no-cache-dir lerobot==0.5.1 --no-deps && \
    pip install --no-cache-dir datasets huggingface_hub safetensors

# physical_ai_av (best-effort, non-critical for L2D ingest)
RUN pip install --no-cache-dir physical_ai_av --no-deps 2>/dev/null || true

COPY Model/ /workspace/Model/
COPY platform/pipelines/ /workspace/platform/pipelines/

ENV PYTHONPATH=/workspace/Model:/workspace
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]
