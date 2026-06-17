FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir lerobot physical_ai_av

COPY Model/ /workspace/Model/

ENV PYTHONPATH=/workspace/Model
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]
