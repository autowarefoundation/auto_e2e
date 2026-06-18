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
2. Decodes all camera videos → JPEG at 256x256 (ffmpeg subprocess, quality 90)
3. Extracts egomotion parquet at 10Hz (resampled from source rate)
4. Uploads to S3 in the training-ready layout
5. Returns episode metadata (sample count, frame count)

### 3. PreExtractedDataset (Unified DataLoader)

```python
# Model/data_parsing/pre_extracted.py

class PreExtractedDataset(Dataset):
    """Reads from the training-ready S3 format. No video decode."""

    def __init__(self, manifest_path: str, split: str = "train", s3_client=None):
        self.manifest = json.load(open(manifest_path))
        self.samples = json.load(open(f"{base}/splits/{split}.json"))
        ...

    def __getitem__(self, idx) -> dict:
        sample = self.samples[idx]
        frames = [load_jpeg(f"frames/{sample['episode_id']}/{sample['frame_idx']}/cam_{i}.jpg")
                  for i in range(self.manifest['num_cameras'])]
        ego = load_parquet_window(sample['episode_id'], sample['frame_idx'])
        return {
            "visual_tiles": torch.stack(frames),
            "egomotion_history": ego_history,
            "visual_history": torch.zeros(896),
            "trajectory_target": ego_future,
        }
```

S3 access options (ranked by preference):
1. **Mountpoint for S3 CSI**: S3 bucket mounted as filesystem in pod. Zero code
   change, Dataset reads local paths. Read-only, sequential access patterns OK
   for training.
2. **s3fs / boto3**: Explicit S3 reads with caching. More code but works anywhere.
3. **Pre-download to PVC**: Download entire dataset to EBS before training. Simple
   but slow for large datasets and wastes storage.

Recommendation: **Mountpoint for S3 CSI** (Phase 1 already configured the CSI
driver via EKS Auto Mode block storage; S3 CSI is a separate addon to enable).

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

### 6. Mountpoint for S3 CSI Driver

EKS Auto Mode includes the EBS CSI driver but NOT S3 CSI. Install separately:

```hcl
# modules/storage/s3-csi.tf
resource "aws_eks_addon" "s3_csi" {
  cluster_name = var.cluster_name
  addon_name   = "aws-mountpoint-s3-csi-driver"
}
```

PersistentVolume for the datasets bucket:
```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: datasets-s3-pv
spec:
  capacity:
    storage: 1Ti  # notional
  accessModes: [ReadOnlyMany]
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
  accessModes: [ReadOnlyMany]
  storageClassName: ""
  resources:
    requests:
      storage: 1Ti
  volumeName: datasets-s3-pv
```

Training pods mount this PVC and read JPEG/parquet as local files.

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
- S3 storage: JPEG at 256x256 quality 90 ≈ 30KB/frame. 7 cameras × 10Hz × 20s
  episode = 1400 frames/episode ≈ 42MB/episode. 150 episodes ≈ 6.3 GB total (tiny).
- NVIDIA PhysicalAI: 7 cameras × 30fps × 20s = 4200 frames/clip at 256x256 ≈ 126MB.
  Unknown total clips but likely < 100 GB.

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

4. **S3 access pattern for DataLoader**: Mountpoint CSI vs direct boto3?
   → **Mountpoint CSI** preferred (zero code change, kernel-level caching).
   Fallback to boto3 if CSI has issues with random-access JPEG reads.
