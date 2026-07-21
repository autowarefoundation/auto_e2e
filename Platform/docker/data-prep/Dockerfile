FROM public.ecr.aws/docker/library/python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git git-lfs && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# CPU-only torch first (lerobot-dataset omits torch, so ours stays)
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

# Install flytekit FIRST and PINNED, so its transitive deps (botocore/s3transfer,
# adlfs, gcsfs, …) are resolved by ONE package's constraints. An unpinned
# "flytekit>=1.14,<2" mixed with the rest let pip try the newest boto3/adlfs and
# backtrack for many minutes (downloading metadata for dozens of releases in
# descending order) before landing on a compatible set. Pinning flytekit to a
# concrete 1.14.x makes the resolve deterministic and fast.
RUN pip install --no-cache-dir "flytekit==1.14.9"

# KITScenes API + the matching vendored Python 3.12 Lanelet2 wheel. Pinning the
# SDK commit keeps the official scene split and on-disk parser contract aligned
# with the immutable dataset revision used by the Flyte cache key.
ARG KITSCENES_SDK_REVISION=7765cdec5490894266070ab46e23724b58b3da42
RUN git lfs install --system \
    && GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/KIT-MRT/kitscenes.git /tmp/kitscenes \
    && GIT_LFS_SKIP_SMUDGE=1 git -C /tmp/kitscenes checkout "${KITSCENES_SDK_REVISION}" \
    && git -C /tmp/kitscenes lfs pull \
    && cd /tmp/kitscenes \
    && pip install --no-cache-dir ".[map]" \
    && rm -rf /tmp/kitscenes

# KITScenes requires NumPy 1.x, while lerobot-dataset 0.5.0's package metadata
# requires NumPy 2.x even though its dataset runtime remains NumPy 1-compatible.
# Install its non-conflicting runtime dependencies explicitly, then install the
# dataset-only wheel without letting pip replace the KITScenes ABI.
RUN pip install --no-cache-dir \
    kubernetes \
    physical_ai_av \
    "huggingface_hub>=1.0,<2.0" \
    "datasets==4.8.5" \
    "fsspec==2026.2.0" \
    "s3fs==2026.2.0" \
    "gcsfs==2026.2.0" \
    webdataset \
    Pillow \
    "numpy<2.0" \
    "opencv-python-headless<4.10" \
    "av>=15.0,<16.0" \
    "accelerate>=1.10,<2.0" \
    "deepdiff>=7.0,<9.0" \
    "imageio[ffmpeg]>=2.34,<3.0" \
    "jsonlines>=4.0,<5.0" \
    "packaging>=24.2,<26.0" \
    "termcolor>=2.4,<4.0" \
    timm \
    scipy \
    pyproj \
    "boto3==1.43.0" \
    pyarrow \
    && pip install --no-cache-dir --no-deps "lerobot-dataset==0.5.0"

COPY Model/ /workspace/Model/
COPY Platform/pipelines/ /workspace/Platform/pipelines/
COPY Tools/trajectory_visualization/ /workspace/Tools/trajectory_visualization/

ENV PYTHONPATH=/workspace/Model:/workspace
ENV PYTHONUNBUFFERED=1
