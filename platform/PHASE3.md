# AutoE2E Phase 3 Platform Design: Data Pipeline

Status: DRAFT — ready for review.

## Goal

Raw OSS datasets (video + sensor logs) are automatically converted to a
**training-ready format** (pre-extracted JPEG frames + egomotion parquet +
manifest) on S3. Training jobs read directly from the pre-extracted format — no
video decode at training time.

## Problem Statement

Current training (`L2DDataset`, `NvidiaAVDataset`) decodes video on-the-fly
from HuggingFace / local disk. This causes:

1. **Slow DataLoader**: video decode is CPU-bound, starves the GPU
2. **Not cloud-native**: lerobot/physical_ai_av SDKs expect local files or HF cache
3. **No versioning**: no way to reproduce a training run's exact data state
4. **No parallelism**: episode processing is serial, cannot leverage Flyte map_task

### Specific problem: NVIDIA PhysicalAI frame loading (Issue #30)

The `load_camera_frame()` in `nvidia_physical_ai/camera.py` reads the **entire
video file** into memory on every `__getitem__` call (`video_path.read_bytes()` +
`SeekVideoReader`), decodes a single frame, then discards everything. For a
typical 20s clip at 30fps with 7 cameras:

- 4,200 frames exist per clip, but only ~72 are valid sample points (10Hz
  egomotion with 64-step history/future windows)
- Each `__getitem__` re-reads 7 video files (hundreds of MB) to extract 7 frames
- With `DataLoader(num_workers=4)`: multiple full video files resident in memory
- TorchCodec objects don't serialize cleanly across DataLoader workers

**Resolution**: The pre-extraction step (§2) computes valid sample points
offline, extracts ONLY those frames as JPEG, and bundles into WebDataset shards.
At training time, zero video decode occurs. This is Option 3 from Issue #30
("Pre-extracted aligned-frame cache") implemented at platform scale.

## Solution Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  Flyte Data Ingest Pipeline (CPU nodes, parallel per episode)     │
│                                                                   │
│  1. ingest_episode   HF download (lerobot) / SDK fetch (nvidia)   │
│  2. extract_frames   Video → JPEG (ffmpeg, 256x256, quality 90)   │
│  3. extract_ego      Parse egomotion → 10Hz parquet               │
│  4. upload_to_s3     Frames + parquet → S3 training-ready layout  │
│                                                                   │
│  Parallelism: Flyte map_task — one task per episode/clip          │
│  Container: data-prep image (ffmpeg + lerobot + physical_ai_av)   │
│  NodePool: general-purpose CPU (Auto Mode default, auto-scaled)   │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│  Training-Ready Format on S3                                      │
│                                                                   │
│  s3://datasets/{dataset_name}/{version}/                          │
│  ├── manifest.json          (sample count, camera list, dims)     │
│  ├── splits/                                                      │
│  │   ├── train.json         (list of sample_ids)                  │
│  │   └── val.json                                                 │
│  ├── frames/                                                      │
│  │   └── {episode_id}/{frame_idx}/                                │
│  │       ├── cam_0.jpg      (front_wide, 256x256)                 │
│  │       ├── cam_1.jpg      (front_tele)                          │
│  │       ├── ...                                                  │
│  │       └── cam_6.jpg      (rear_tele or bev_map)                │
│  ├── egomotion/                                                   │
│  │   └── {episode_id}.parquet  (10Hz: timestamp, x, y, yaw, v)   │
│  └── metadata/                                                    │
│      ├── camera_params.json (intrinsics per camera)               │
│      └── dataset_info.json  (source, license, creation date)      │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│  PreExtractedDataset (new unified DataLoader)                     │
│                                                                   │
│  - Reads manifest.json → enumerate samples                        │
│  - Loads JPEG from S3 (via Mountpoint for S3 CSI or boto3)        │
│  - Loads egomotion from parquet (per-episode, cached)             │
│  - Returns same dict shape as L2DDataset/NvidiaAVDataset          │
│  - No video decode, no lerobot dependency at train time           │
│  - Compatible with DDP (file-based, deterministic sharding)       │
└───────────────────────────────────────────────────────────────────┘
```

## Detailed Design

### 1. Training-Ready Format Specification

**Egomotion parquet schema** (per episode, 10Hz):
```
timestamp_s: float64   — seconds from episode start
x: float32            — longitudinal position (m)
y: float32            — lateral position (m)
yaw: float32          — heading (rad)
vx: float32           — longitudinal velocity (m/s)
vy: float32           — lateral velocity (m/s)
yaw_rate: float32     — yaw rate (rad/s)
```

**Manifest schema** (`manifest.json`):
```json
{
  "dataset": "l2d",
  "version": "v1.0",
  "num_samples": 12345,
  "num_episodes": 42,
  "cameras": ["cam_0", "cam_1", "cam_2", "cam_3", "cam_4", "cam_5", "cam_6"],
  "frame_size": [256, 256],
  "egomotion_hz": 10,
  "history_steps": 16,
  "future_steps": 64,
  "created_at": "2026-06-19T00:00:00Z"
}
```

**Split files** (`splits/train.json`, `splits/val.json`):
```json
[
  {"episode_id": "ep_000", "frame_idx": 16},
  {"episode_id": "ep_000", "frame_idx": 17},
  ...
]
```

### 2. Flyte Data Ingest Workflow

```python
# platform/pipelines/data_ingest/workflow.py

@task(container_image=DATA_PREP_IMAGE, requests=Resources(cpu="4", mem="16Gi"))
def ingest_l2d_episode(episode_idx: int, repo_id: str, output_prefix: str) -> str:
    """Download one L2D episode, extract frames + egomotion, upload to S3."""
    ...

@workflow
def ingest_l2d(
    repo_id: str = "yaak-ai/L2D",
    episodes: List[int] = None,  # None = all
    output_bucket: str = "auto-e2e-platform-datasets-381491877296",
    version: str = "v1.0",
) -> str:
    """Full L2D ingest: parallel per episode → manifest → split."""
    episode_results = map_task(ingest_l2d_episode)(
        episode_idx=episodes, repo_id=[repo_id]*len(episodes), ...
    )
    manifest = build_manifest(episode_results, version)
    return manifest
```

Each `ingest_*_episode` task:
1. Downloads one episode from HF (lerobot) or NVIDIA SDK
2. **Computes valid sample points**: parses egomotion to find frames with
   sufficient history (16 steps) and future (64 steps) context. Only these
   frames are extracted — not every frame in the video. (This resolves Issue #30:
   NVIDIA clips have 4200 frames but only ~72 valid sample points.)
3. Decodes ONLY valid-sample camera frames → JPEG at 256×256 (ffmpeg `-ss` seek
   to exact timestamp, quality 90). No full-video decode.
4. Extracts egomotion window (history + future) for each valid sample → .npy
5. Uploads to S3 in the training-ready layout
6. Returns episode metadata (valid sample count, frame indices)

### 3. Data Access: WebDataset Tar Shards + EBS Prefetch

**Problem with Mountpoint S3 CSI / per-file S3 reads**: Training DataLoader does
random-access reads of many small files (30KB JPEG x 7 cameras x batch). Each
file triggers a separate S3 GET (~10ms). With batch_size=8, that's 56 GETs per
step — GPU starves waiting for I/O.

**Solution: WebDataset tar shards stored on S3, prefetched to EBS before training.**

```
S3 (cold storage, source of truth):
  s3://datasets/{name}/{version}/shards/
  ├── train-000000.tar  (100MB, ~3000 samples bundled)
  ├── train-000001.tar
  ├── ...
  └── val-000000.tar

Training Pod (hot storage, EBS PVC):
  /data/shards/          ← init container syncs relevant shards from S3
  ├── train-000000.tar
  ├── train-000001.tar
  └── ...
```

**Pipeline**:
1. Flyte ingest writes individual frames + parquet to S3 (training-ready format)
2. A **shard-packing task** (Flyte, post-ingest) bundles samples into WebDataset
   tar files (~100MB each). Each tar entry = one sample (7 JPEG + ego .npy).
3. Training pod **init container** syncs shard tars from S3 → EBS PVC
   (`aws s3 sync`). ~60s for 10GB dataset at 1 Gbps.
4. Training DataLoader reads from local EBS via `webdataset.WebLoader` —
   sequential tar reads, no per-file S3 GET, disk throughput saturated.

**WebDataset shard format** (inside each .tar):
```
000000.cam_0.jpg
000000.cam_1.jpg
...
000000.cam_6.jpg
000000.ego.npy          # (history_steps + future_steps, 7) float32
000000.meta.json        # {"episode_id": "...", "frame_idx": 42}
000001.cam_0.jpg
...
```

**Why WebDataset**:
- Sequential reads → EBS gp3 throughput (125-1000 MB/s) fully utilized
- Built-in shuffling (shard-level + in-shard)
- Native PyTorch DataLoader integration (`wds.WebLoader`)
- DDP-friendly: each worker reads different shards (no overlap)
- Proven at scale (Google, LAION, many AD datasets)

**EBS PVC for shard cache** (per training job):
```yaml
# Attached to PyTorchJob via volumeClaimTemplates or pre-created PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dataset-cache
  namespace: auto-e2e-training
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: auto-ebs-sc
  resources:
    requests:
      storage: 50Gi  # sized per dataset (L2D ~7GB, NVIDIA ~30GB)
```

**Init container** in PyTorchJob:
```yaml
initContainers:
  - name: fetch-shards
    image: amazon/aws-cli:latest
    command: ["sh", "-c"]
    args:
      - aws s3 sync s3://datasets/l2d/v1.0/shards/ /data/shards/ --exclude "val-*"
    volumeMounts:
      - name: dataset-cache
        mountPath: /data
```

**Performance estimate** (L2D, ~150 episodes, ~7GB shards):
- S3 sync: ~60s at 1 Gbps (EKS private subnet → S3 via gateway endpoint)
- DataLoader: sequential tar from gp3 EBS → 125 MB/s baseline, 500 MB/s burst
- With num_workers=4 reading 4 shards in parallel: GPU never starves
- Compare: per-file S3 GET would be 56 × 10ms = 560ms/step (unusable)

### 3b. PreExtractedDataset (WebDataset-backed)

```python
# Model/data_parsing/pre_extracted.py
import webdataset as wds
import torch, numpy as np
from torchvision import transforms

def make_training_dataloader(shard_dir: str, batch_size: int, num_workers: int = 4):
    """WebDataset DataLoader reading from local EBS shard cache."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def decode_sample(sample):
        frames = [transform(sample[f"cam_{i}.jpg"]) for i in range(7)]
        ego = np.frombuffer(sample["ego.npy"], dtype=np.float32).reshape(-1, 7)
        # Split ego into history (first 16 rows) and future (next 64 rows)
        ego_history = torch.from_numpy(ego[:16].flatten())
        trajectory_target = torch.from_numpy(ego[16:].flatten())
        return {
            "visual_tiles": torch.stack(frames),
            "egomotion_history": ego_history,
            "visual_history": torch.zeros(896),
            "trajectory_target": trajectory_target,
        }

    dataset = (
        wds.WebDataset(f"{shard_dir}/train-{{000000..000099}}.tar")
        .shuffle(1000)
        .map(decode_sample)
    )
    return wds.WebLoader(dataset, batch_size=batch_size, num_workers=num_workers)
```


### 4. LakeFS (Data Versioning) — Deferred

Original Phase 3 plan includes LakeFS for dataset versioning. After review:

**Decision: Defer LakeFS to Phase 3.5 or later.**

Reasoning:
- S3 versioning on the datasets bucket is already enabled (Phase 1)
- The `{version}` folder in the training-ready format provides manual versioning
- LakeFS adds operational complexity (another stateful service, Helm chart, RDS
  usage) for a benefit (branch-per-experiment) we don't yet need
- Priority: get data pipeline working end-to-end first, add LakeFS when we need
  to branch datasets for A/B comparisons

When we do add LakeFS, it wraps the existing S3 buckets — transparent to the
DataLoader (just changes the S3 endpoint).

### 5. Data Prep Dockerfile Enhancement

Current `platform/docker/data-prep/Dockerfile` is minimal. Phase 3 additions:

```dockerfile
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt \
    lerobot physical_ai_av flytekit==1.16.23 boto3 pyarrow

COPY Model/ /workspace/Model/
COPY platform/pipelines/ /workspace/platform/pipelines/

ENV PYTHONPATH=/workspace/Model:/workspace
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python"]
```

### 6. Mountpoint for S3 CSI Driver (Ingest Tasks Only)

Used by Flyte data ingest tasks to write processed frames directly to S3.
**NOT used for training DataLoader** (WebDataset + EBS instead).

EKS Auto Mode includes the EBS CSI driver but NOT S3 CSI. Install separately:

```hcl
# modules/storage/s3-csi.tf
resource "aws_eks_addon" "s3_csi" {
  cluster_name = var.cluster_name
  addon_name   = "aws-mountpoint-s3-csi-driver"
}
```

PersistentVolume for the datasets bucket (used by ingest pods):
```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: datasets-s3-pv
spec:
  capacity:
    storage: 1Ti  # notional
  accessModes: [ReadWriteMany]
  csi:
    driver: s3.csi.aws.com
    volumeHandle: s3-datasets
    volumeAttributes:
      bucketName: auto-e2e-platform-datasets-<ACCOUNT_ID>
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: datasets-s3
  namespace: auto-e2e-training
spec:
  accessModes: [ReadWriteMany]
  storageClassName: ""
  resources:
    requests:
      storage: 1Ti
  volumeName: datasets-s3-pv
```

### 7. Implementation Plan (Ordered)

1. **PreExtractedDataset** — write the unified DataLoader (`Model/data_parsing/pre_extracted.py`). Test locally with a small mock dataset.

2. **Flyte data_ingest workflow** — `platform/pipelines/data_ingest/workflow.py`. L2D first (simpler, lerobot provides direct access). map_task per episode.

3. **Data-prep Dockerfile** — update with flytekit + pyarrow + boto3.

4. **Mountpoint for S3 CSI** — Terraform addon + PV/PVC manifests.

5. **Register & run L2D ingest** — `pyflyte register`, trigger from Flyte UI. Verify frames appear in S3.

6. **Verify training reads pre-extracted** — modify train.py to accept `--dataset-format=pre_extracted --manifest-path=s3://...`. Run a 1-epoch training job.

7. **NVIDIA PhysicalAI ingest** — second workflow, same pattern, uses download_dataset.py + ffmpeg extraction.

8. **Train/val split logic** — 80/20 random split per episode, saved in splits/.

9. **End-to-end verify** — Flyte ingest pipeline → S3 pre-extracted → training PyTorchJob reads it → checkpoint to S3 → MLflow logs.

### 8. Cost Considerations

- Data ingest runs on CPU nodes (Auto Mode general-purpose). No GPU cost.
- map_task parallelism: L2D has ~150 episodes → 150 parallel tasks → completes in
  minutes not hours. Karpenter scales CPU nodes automatically.
- S3 storage (valid samples only, NOT every frame):
  - L2D: ~72 valid samples/episode × 7 cameras × 30KB = ~15 MB/episode.
    150 episodes ≈ 2.3 GB shards.
  - NVIDIA: ~72 valid samples/clip × 7 cameras × 30KB = ~15 MB/clip.
    Even 500 clips = ~7.5 GB shards.
  - Extraction of ONLY valid samples (not all 4200 frames/clip) reduces storage
    and processing by ~58x vs naive full-frame extraction.
- EBS PVC for training: 50 Gi gp3 ≈ $4/month. Created per-job, deleted after.

### 9. Open Questions

1. **Frame rate for extraction**: L2D source is 10Hz (CAN bus aligned). NVIDIA is
   30fps but egomotion is 100Hz. Extract at 10Hz (aligned to ego) or full fps?
   → **Proposal**: Extract at 10Hz for both. Reduces storage 3x for NVIDIA and
   matches the trajectory prediction frequency.

2. **Image format**: JPEG vs WebP vs PNG. JPEG quality 90 is the right tradeoff
   (30KB, fast decode, minimal quality loss for 256x256 inputs).

3. **Should train.py support both old (lerobot) and new (pre-extracted) paths?**
   → **Yes**: gate via `--dataset-format` arg. Default remains lerobot for EC2
   development; `pre_extracted` for EKS production training.

4. **S3 access pattern for DataLoader**: ~~Mountpoint CSI vs direct boto3?~~
   → **WebDataset tar shards on EBS** (decided). Init container syncs shards
   from S3 to local EBS PVC before training starts. Sequential tar reads
   give full disk bandwidth. Mountpoint CSI kept only for Flyte ingest tasks.
