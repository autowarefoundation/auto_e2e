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

# The rest: lerobot-dataset = standalone LeRobotDataset with relaxed deps (no
# torch override). boto3 is used by label_cache.py (S3); it's already pulled in
# compatible by flytekit above, but list it so the dependency is explicit.
RUN pip install --no-cache-dir \
    kubernetes \
    lerobot-dataset \
    physical_ai_av \
    huggingface_hub \
    datasets \
    webdataset \
    Pillow \
    "numpy<2.0" \
    "opencv-python-headless<4.10" \
    av \
    timm \
    scipy \
    pyproj \
    boto3 \
    pyarrow

COPY Model/ /workspace/Model/
COPY Platform/pipelines/ /workspace/Platform/pipelines/

ENV PYTHONPATH=/workspace/Model:/workspace
ENV PYTHONUNBUFFERED=1
