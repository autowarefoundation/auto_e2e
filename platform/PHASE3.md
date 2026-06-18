# AutoE2E Phase 3 Platform Design: Data Pipeline

Status: DRAFT — ready for review.

## Goal

Raw OSS datasets (video + sensor logs) are automatically converted to a
**training-ready format** (pre-extracted JPEG frames + egomotion parquet +
manifest) on S3. Training jobs read directly from the pre-extracted format — no
video decode at training time.

## Problem Statement

**全てのデータセットに共通する問題**: raw dataset はそれぞれ独自フォーマット
(動画 + センサーログ、HF LeRobot、SDK zip、etc) で提供されるが、training DataLoader
が求めるのは「valid sample point のフレーム + egomotion window」だけ。

共通の非効率:
1. **不要フレームの処理**: 数千フレーム中、実際にtrainingで使うのはhistory/future
   窓が成立する一部のみ (例: 4200フレーム中72)
2. **毎回decode**: DataLoader の `__getitem__` ごとに動画/アーカイブを再読み込み
3. **format差異の学習時コスト**: データセットごとの parse ロジックが hot path に混入
4. **クラウド非対応**: SDK がローカルファイルを前提

これは L2D (lerobot の動画デコード)、NVIDIA PhysicalAI (#30: 毎回 read_bytes +
SeekVideoReader)、KIT Scenes (lanelet2 map rasterize + pose file parse) 全てに
共通する。今後データセットが増えても同じ問題が発生する。

**解決方針**: データセットごとに異なる **Ingest Adapter** が raw → unified format
に変換する。変換後は全て同一の WebDataset shard 形式。学習コードはデータセットの
出自を知らない。

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

### 2. Flyte Data Ingest Workflow (Pluggable Adapter Design)

データセットごとに raw format は異なるが、出力 (unified shard format) は共通。
差分を **IngestAdapter** プロトコルに閉じ込めて、workflow ロジックは共通化する。

```
┌─────────────────────────────────────────────────────────────────┐
│  Dataset-specific Adapter (protocol)                            │
│                                                                 │
│  class IngestAdapter(Protocol):                                 │
│      def list_episodes(self) -> List[EpisodeRef]                │
│      def download_episode(self, ref) -> Path                    │
│      def compute_valid_samples(self, episode_path) -> List[...] │
│      def extract_frame(self, episode_path, sample, cam) -> img  │
│      def extract_egomotion(self, episode_path, sample) -> array │
│                                                                 │
│  Implementations:                                               │
│    L2DAdapter        — lerobot HF, video decode via lerobot     │
│    NvidiaAVAdapter   — SDK zip, ffmpeg -ss seek decode          │
│    KitScenesAdapter  — rosbag/parquet, pose-based ego           │
│    (future)          — any new dataset: implement 5 methods     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼  (all adapters produce the same output)
┌─────────────────────────────────────────────────────────────────┐
│  Common Pipeline (dataset-agnostic)                             │
│                                                                 │
│  for each episode (parallel via map_task):                      │
│    1. adapter.download_episode(ref)                             │
│    2. valid_samples = adapter.compute_valid_samples(...)        │
│    3. for sample in valid_samples:                              │
│         frames = [adapter.extract_frame(..., cam)               │
│                   for cam in cameras]                           │
│         ego = adapter.extract_egomotion(...)                    │
│         → append to shard buffer                                │
│    4. flush shard buffer → .tar → upload S3                     │
│                                                                 │
│  After all episodes:                                            │
│    5. build manifest.json + train/val splits                    │
└─────────────────────────────────────────────────────────────────┘
```

**Flyte Workflow 構造**:
```python
# platform/pipelines/data_ingest/workflow.py

@task(container_image=DATA_PREP_IMAGE, requests=Resources(cpu="4", mem="16Gi"))
def ingest_episode(adapter_name: str, episode_ref: str, output_prefix: str) -> EpisodeResult:
    """Generic: instantiate adapter by name, process one episode."""
    adapter = get_adapter(adapter_name)  # registry lookup
    ...

@workflow
def ingest_dataset(
    adapter_name: str = "l2d",           # "l2d" | "nvidia_av" | "kit_scenes" | ...
    output_bucket: str = "...",
    version: str = "v1.0",
    episode_limit: int = 0,              # 0 = all
) -> str:
    episodes = list_episodes(adapter_name, episode_limit)
    results = map_task(ingest_episode)(
        adapter_name=[adapter_name]*len(episodes),
        episode_ref=episodes, ...
    )
    return build_manifest_and_shards(results, version)
```

**Adapter Registry** (`platform/pipelines/data_ingest/adapters/__init__.py`):
```python
ADAPTER_REGISTRY: dict[str, type[IngestAdapter]] = {
    "l2d": L2DAdapter,
    "nvidia_av": NvidiaAVAdapter,
    "kit_scenes": KitScenesAdapter,
}

def get_adapter(name: str) -> IngestAdapter:
    return ADAPTER_REGISTRY[name]()
```

新しいデータセットを追加するとき:
1. `adapters/new_dataset.py` に `IngestAdapter` の5メソッドを実装
2. `ADAPTER_REGISTRY` に1行追加
3. Flyte UI から `adapter_name="new_dataset"` で起動 — workflow 変更不要

Each `ingest_episode` task (adapter-agnostic logic):
1. `adapter.download_episode(ref)` — raw データ取得
2. **`adapter.compute_valid_samples()`** — history/future 窓が成立する
   sample points を計算。全フレームではなく必要なものだけ特定 (Issue #30 解決)
3. `adapter.extract_frame()` — 該当タイムスタンプのみデコード → JPEG 256×256
4. `adapter.extract_egomotion()` — history + future window を切り出し → .npy
5. WebDataset shard buffer に追加、100MB で flush → S3 upload

**各 Adapter の差分**:

| Adapter | Download | Valid Sample 計算 | Frame 抽出 | Egomotion |
|---|---|---|---|---|
| L2D | `LeRobotDataset(repo_id)` | episode_ranges + MIN_FRAMES (既存ロジック) | lerobot video decode | CAN bus state → trajectory |
| NVIDIA AV | SDK zip DL + unpack | egomotion 100Hz→10Hz + history/future window | `ffmpeg -ss {t} -frames:v 1` (シーク抽出) | parquet window slice |
| KIT Scenes | rosbag / parquet DL | pose timestamps + window check | 画像ファイル or rosbag image_raw | pose diff → ego |
| (future) | implement `download_episode` | implement `compute_valid_samples` | implement `extract_frame` | implement `extract_egomotion` |

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
