# Real Training Pipeline — Requirements / Design / Todo

## Objective

Replace the current mock/placeholder Flyte pipeline with real training that:
1. Downloads real data from HuggingFace
2. Pre-processes it (Issue #30 fix: pre-extract frames, Hz/resolution conversion)
3. Trains the real `AutoE2E` model with `TrajectoryImitationLoss`
4. Evaluates with real ADE/FDE metrics (trajectory integration)
5. Logs everything to MLflow in a single consolidated run per experiment

---

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌───────────┐    ┌──────────┐
│ data_ingest │───▶│ data_processing  │───▶│ train_il  │───▶│ evaluate │
│ (per dataset)│    │ (Issue #30 fix)  │    │ (AutoE2E) │    │ (MLflow) │
└─────────────┘    └──────────────────┘    └───────────┘    └──────────┘
                                                │
                                                ▼
                                          ┌──────────────┐    ┌──────────┐
                                          │train_offline_rl│──▶│ evaluate │
                                          └──────────────┘    └──────────┘
```

### Data Flow (FlyteDirectory / FlyteFile)

```
data_ingest    → FlyteDirectory: raw lerobot cache (videos + parquet)
data_processing → FlyteDirectory: WebDataset shards (JPEG frames + ego.npy + meta.json)
train_il       → TrainOutput(checkpoint=FlyteFile, metadata=FlyteFile)
evaluate       → EvalMetrics(ade, fde, gate_pass) + MLflow run
```

---

## Task Specifications

### 1. `data_ingest` (container: data-prep)

**Purpose**: Download raw dataset from HuggingFace. No transformation.

**Input** (Flyte UI params):
- `dataset: Dataset` — enum dropdown (L2D, NVIDIA_PHYSICAL_AI)
- `episodes: list[int] | None` — specific episodes or all

**Logic**:
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id=dataset.value, episodes=episodes)
# lerobot caches to ~/.cache/huggingface/lerobot/
# Copy cache dir → output FlyteDirectory
```

**Output**: `FlyteDirectory` containing raw lerobot cache (MP4 videos + parquet metadata)

**Docker deps**: `lerobot>=0.5.1`, `huggingface_hub`, `datasets`

---

### 2. `data_processing` (container: data-prep) — NEW TASK

**Purpose**: Pre-extract frames + egomotion aligned to valid sample points.
Solves Issue #30: avoids per-frame video decode during training.

**Input**:
- `raw_data: FlyteDirectory` — output of data_ingest
- `hz: int = 10` — target frequency
- `image_size: int = 256` — resize target (square)
- `cameras: list[str]` — cameras to keep (default: all 7 L2D cameras)
- `backbone_name: str = "swinv2_tiny_window8_256"` — for normalization params

**Logic** (per episode):
1. Load lerobot dataset from local cache
2. For each episode:
   a. Load `observation.state.vehicle` → derive egomotion signals (speed, accel_x, yaw_rate, curvature)
   b. Compute valid sample indices: `[64, 65, ..., T-65]` (need 64 history + 64 future)
   c. For each valid sample index:
      - Decode camera frames at that timestep (7 cameras)
      - Resize to `image_size x image_size`
      - Extract egomotion_history (64×4=256 floats) + trajectory_target (64×2=128 floats)
      - Save as WebDataset sample:
        ```
        {sample_key}.cam_0.jpg  (front_left)
        {sample_key}.cam_1.jpg  (left_forward)
        ...
        {sample_key}.cam_6.jpg  (map)
        {sample_key}.ego.npy    (384 floats: 256 history + 128 target)
        {sample_key}.meta.json  (episode_index, frame_index, dataset)
        ```
3. Pack into tar shards (1000 samples per shard)

**Output**: `FlyteDirectory` containing `train-000000.tar`, `train-000001.tar`, ...

**Key implementation**: Use `lerobot_dataset[frame_idx]` to get camera frames, then resize.
For video-based datasets: use `torchvision.io.read_video` or `av` to seek to specific frame.

**Docker deps**: `lerobot>=0.5.1`, `webdataset`, `av`, `Pillow`, `torchvision`

---

### 3. `train_il` (container: training)

**Purpose**: Train `AutoE2E` model on pre-extracted WebDataset shards.

**Input**:
- `shards: list[FlyteDirectory]` — processed shards from ALL datasets
- `dataset: Dataset` — for metadata only
- `backbone: Backbone` — swin_v2_tiny / convnext_v2_tiny / resnet_50
- `fusion_mode: FusionMode` — concat / cross_attn / bev
- `epochs: int = 3` — short for testing
- `batch_size: int = 4`
- `lr: float = 1e-4`
- `weight_decay: float = 1e-2`
- `grad_clip: float = 1.0`
- `amp: bool = True`
- `warmup_steps: int = 500`

**Logic**:
```python
from data_parsing.pre_extracted import make_pre_extracted_loader
from model_components.auto_e2e import AutoE2E
from model_components.losses import TrajectoryImitationLoss

# DataLoader from WebDataset shards (already exists in pre_extracted.py)
loader = make_pre_extracted_loader(shard_dir, batch_size=batch_size)

# Model
model = AutoE2E(backbone=backbone, fusion_mode=fusion_mode, ...)

# Training loop (from train.py logic)
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = TrajectoryImitationLoss(loss_type="smooth_l1")
for epoch in range(epochs):
    for batch in loader:
        pred, _, _ = model(batch["visual_tiles"], batch["visual_history"], batch["egomotion_history"])
        loss = loss_fn(pred, batch["trajectory_target"])
        loss.backward()
        ...
```

**Output**: `TrainOutput(checkpoint=FlyteFile, metadata=FlyteFile)`

Metadata JSON contains:
```json
{
  "data": {"datasets": ["yaak-ai/L2D", "nvidia/PhysicalAI"], "shard_dirs": [...], "total_samples": 5000},
  "model": {"backbone": "swin_v2_tiny", "fusion_mode": "concat", "embed_dim": 256, "num_views": 7},
  "training": {"epochs": 3, "batch_size": 4, "lr": 1e-4, "weight_decay": 1e-2, "grad_clip": 1.0, "amp": true, "optimizer": "AdamW", "scheduler": "cosine", "final_loss": 0.045, "losses_per_epoch": [0.12, 0.07, 0.045]},
  "context": {"flyte_execution_id": "abc123", "docker_image": "381491877296.dkr.ecr.../training:latest", "checkpoint_path": "/tmp/best.pt", "git_commit": "def456"}
}
```

**Docker deps**: `timm`, `torch`, `torchvision`, `webdataset`, `numpy`, `flytekit`, `pyyaml`

**Note**: `train_il` does NOT log to MLflow. That's `evaluate`'s job.

---

### 4. `train_offline_rl` (container: offline-rl)

**Purpose**: Refine IL checkpoint via IQL (Implicit Q-Learning).

**Input**:
- `pretrained: FlyteFile` — IL checkpoint
- `shards: list[FlyteDirectory]` — same processed shards
- `il_metadata: FlyteFile` — IL metadata (for provenance chain)
- `epochs: int = 5`
- `tau: float = 0.7`
- `beta: float = 3.0`
- `replay_buffer_size: int = 100000`
- `discount: float = 0.99`

**Output**: `TrainOutput(checkpoint=FlyteFile, metadata=FlyteFile)`

Metadata nests IL metadata inside:
```json
{
  "base_model": {"il_metadata": {...full IL metadata...}, "il_checkpoint_path": "..."},
  "rl": {"method": "IQL", "epochs": 5, "tau": 0.7, ...},
  "context": {"flyte_execution_id": "...", "docker_image": "...", ...}
}
```

---

### 5. `evaluate` (container: eval) — THE ONLY MLflow LOGGING POINT

**Purpose**: Run open-loop evaluation + log everything to MLflow.

**Input**:
- `checkpoint: FlyteFile`
- `shards: list[FlyteDirectory]` — eval split
- `train_metadata: FlyteFile` — full provenance chain
- `experiment_name: str` — "imitation-learning" or "offline-rl"

**Logic**:
```python
import mlflow
from evaluation.metrics import compute_open_loop_metrics, gate_check

mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name=...):
    # 1. Log ALL params from metadata (flattened)
    mlflow.log_params({...})  # data/*, model/*, train/*, rl/*, ctx/*

    # 2. Log training loss curves as metrics
    for i, loss in enumerate(meta["training"]["losses_per_epoch"]):
        mlflow.log_metric("train/loss", loss, step=i)

    # 3. Run evaluation
    model = AutoE2E(...)
    model.load_state_dict(torch.load(ckpt)["model_state_dict"])
    for batch in eval_loader:
        pred, _, _ = model(...)
        # integrate predicted (accel, curvature) → trajectory
        # compare vs ground truth

    metrics = compute_open_loop_metrics(...)
    mlflow.log_metrics({"eval/ade": ade, "eval/fde": fde, "eval/gate_pass": ...})

    # 4. Artifacts
    mlflow.log_artifact(config.yaml)
    mlflow.log_artifact(checkpoint, artifact_path="model")

    # 5. Model Registry
    mlflow.register_model(...)
```

**MLflow Experiment UI Row**:
| Run | backbone | fusion | dataset | epochs | lr | batch_size | eval/ade | eval/fde | gate |
|-----|----------|--------|---------|--------|-----|-----------|----------|----------|------|

---

## MLflow Design

### Experiments (2 only)
- `imitation-learning` — IL training experiments
- `offline-rl` — RL refinement experiments

### Params logged per run (all filterable in UI)
```
data/datasets              = "yaak-ai/L2D,nvidia/PhysicalAI"
data/total_samples         = 5000
data/hz                    = 10
data/image_size            = 256
model/backbone             = "swin_v2_tiny"
model/fusion_mode          = "concat"
model/embed_dim            = 256
model/num_views            = 7
train/epochs               = 3
train/batch_size           = 4
train/lr                   = 0.0001
train/weight_decay         = 0.01
train/grad_clip            = 1.0
train/amp                  = true
train/optimizer            = "AdamW"
train/scheduler            = "cosine"
train/final_loss           = 0.045
rl/method                  = "IQL"        (offline-rl only)
rl/tau                     = 0.7          (offline-rl only)
rl/beta                    = 3.0          (offline-rl only)
ctx/flyte_execution_id     = "abc123"
ctx/train_docker_image     = "381491877296.dkr.ecr.../training:latest"
ctx/eval_docker_image      = "381491877296.dkr.ecr.../eval:latest"
ctx/git_commit             = "def456"
ctx/checkpoint_s3_path     = "s3://..."
base/il_execution_id       = "xyz789"     (offline-rl only)
```

### Artifacts per run
- `config.yaml` — full nested config for exact reproduction
- `model/best.pt` — checkpoint
- Registered in Model Registry: `auto-e2e-driving-policy`

---

## Flyte Workflow Definitions

### `wf_data_ingest`
```
Params: dataset (dropdown), episodes
Output: FlyteDirectory (raw)
```

### `wf_data_processing` (NEW)
```
Params: raw_data (FlyteDirectory), hz, image_size, cameras
Output: FlyteDirectory (WebDataset shards)
```

### `wf_train_il`
```
Params: shards (FlyteDirectory), backbone, fusion_mode, epochs, batch_size, lr, ...
Output: EvalMetrics
Internals: train_il → evaluate (with experiment_name="imitation-learning")
```

### `wf_train_offline_rl`
```
Params: pretrained, shards, il_metadata, epochs, tau, beta, ...
Output: EvalMetrics
Internals: train_offline_rl → evaluate (with experiment_name="offline-rl")
```

### `wf_full_pipeline`
```
Params: dataset, backbone, fusion_mode, epochs_il, epochs_rl, batch_size, lr, tau, beta
Internals:
  shards_l2d = data_ingest(L2D) → data_processing(shards_l2d)
  shards_nvidia = data_ingest(NVIDIA) → data_processing(shards_nvidia)
  il_out = train_il([processed_l2d, processed_nvidia], ...)
  evaluate(il_out, experiment="imitation-learning")
  rl_out = train_offline_rl(il_out.checkpoint, [shards], il_out.metadata, ...)
  evaluate(rl_out, experiment="offline-rl")
```

---

## Docker Image Changes

### `data-prep` image
Add:
```
lerobot>=0.5.1
huggingface_hub
datasets
av
webdataset
Pillow
torchvision
```

### `training` image
Already has: `timm`, `torch` (CUDA), `webdataset`, `flytekit`
Add: `pyyaml` (if missing)
Verify: `torchvision` is in pytorch base image ✓

### `eval` image
Add: `mlflow-skinny`, `pyyaml`, `timm`, `torch`, `torchvision`, `webdataset`

---

## Issue #30 Resolution

The `data_processing` task fully addresses Issue #30:

| Problem | Solution |
|---------|----------|
| Full video read per __getitem__ | Pre-extract frames at valid indices only |
| SeekVideoReader overhead | Done once at processing time, not training time |
| Multi-worker memory pressure | Shards are sequential tar reads, no video in memory |
| TorchCodec serialization | Not needed — frames are pre-extracted JPEGs |

**Format**: WebDataset tar shards (1000 samples/shard)
- Compatible with existing `pre_extracted.py` DataLoader
- Sequential disk I/O (no random access)
- Worker-safe (split by shard)

---

## Pre-existing Code to Reuse

| Component | Location | Usage |
|-----------|----------|-------|
| `AutoE2E` model | `Model/model_components/auto_e2e.py` | train_il task |
| `TrajectoryImitationLoss` | `Model/model_components/losses/` | train_il task |
| `L2DDataset` | `Model/data_parsing/l2d/dataset.py` | data_processing (frame extraction logic) |
| `extract_egomotion` | `Model/data_parsing/l2d/egomotion.py` | data_processing |
| `CAMERA_NAMES` | `Model/data_parsing/l2d/camera.py` | data_processing (which cameras) |
| `make_pre_extracted_loader` | `Model/data_parsing/pre_extracted.py` | train_il DataLoader |
| `compute_open_loop_metrics` | `Model/evaluation/metrics.py` | evaluate task |
| `gate_check` | `Model/evaluation/metrics.py` | evaluate task |
| `integrate_trajectory` | `Model/evaluation/metrics.py` | evaluate task |

---

## Todo (Implementation Order)

### Phase 1: Docker images
- [ ] Update `platform/docker/data-prep/Dockerfile` — add lerobot, av, webdataset, torchvision
- [ ] Update `platform/docker/eval/Dockerfile` — add mlflow-skinny, timm, torch, webdataset, pyyaml
- [ ] Verify `training` Dockerfile has all deps
- [ ] CodeBuild: rebuild all 4 images

### Phase 2: `data_ingest` task (real)
- [ ] Implement: `LeRobotDataset(repo_id, episodes)` download
- [ ] Output: FlyteDirectory with raw lerobot cache
- [ ] Test: verify HF download works in pod (HF_TOKEN from Secrets Manager)

### Phase 3: `data_processing` task (NEW)
- [ ] Implement frame extraction loop (per episode, per valid sample)
- [ ] Camera frame decode: `lerobot_dataset[frame_idx]` → PIL Image → resize → JPEG
- [ ] Egomotion: `extract_egomotion()` → numpy → pack 384 floats
- [ ] WebDataset tar packing (1000 samples/shard)
- [ ] Output manifest: `{total_samples, episodes, cameras, hz, image_size}`
- [ ] Test: verify shards are readable by `make_pre_extracted_loader`

### Phase 4: `train_il` task (real)
- [ ] Use `make_pre_extracted_loader` for DataLoader
- [ ] Instantiate `AutoE2E(backbone, fusion_mode, ...)`
- [ ] Training loop: forward → loss → backward → step (from train.py)
- [ ] AMP support (bf16)
- [ ] Save checkpoint with `model_state_dict` + config
- [ ] Output metadata JSON with all provenance
- [ ] Test: epochs=1, batch_size=2, verify loss decreases

### Phase 5: `evaluate` task (real)
- [ ] Load model from checkpoint
- [ ] Run inference on eval shards
- [ ] `integrate_trajectory()` → ADE/FDE
- [ ] `gate_check()` → pass/fail
- [ ] MLflow logging: all params, metrics, artifacts, model registry
- [ ] Test: verify MLflow UI shows correct data

### Phase 6: `train_offline_rl` task (real)
- [ ] IQL implementation using pre-extracted shards
- [ ] Nest IL metadata in output
- [ ] Test: verify RL checkpoint improves ADE vs IL baseline

### Phase 7: Flyte register + E2E test
- [ ] Update `workflows.py` with all real tasks
- [ ] Add `wf_data_processing` workflow
- [ ] Update `wf_full_pipeline` to include processing step
- [ ] CodeBuild register
- [ ] Launch `wf_full_pipeline` (epochs_il=3, epochs_rl=3)
- [ ] Verify MLflow shows real metrics

---

## Test Run Parameters (for validation)

```
dataset = L2D
episodes = [0, 1, 2]    # just 3 episodes for speed
hz = 10
image_size = 256
backbone = swin_v2_tiny
fusion_mode = concat
epochs_il = 3
batch_size = 4
lr = 1e-4
amp = true
epochs_rl = 3
tau = 0.7
beta = 3.0
```

Expected: ~5-10 min total on g6e.4xlarge (L40S GPU).

---

## Implementation Principles

### 1. Flyte Native Only

All task logic is written **directly inside `@task` decorated functions**. No subprocess calls to `train.py`. No `argparse`. The Flyte task IS the training script.

```python
@task(container_image=TRAINING_IMAGE, requests=Resources(gpu="1"))
def train_il(shards: list[FlyteDirectory], backbone: Backbone, ...) -> TrainOutput:
    # ALL training logic inline here
    model = AutoE2E(backbone=backbone.value, ...)
    loader = make_flyte_loader(shards)
    for epoch in range(epochs):
        for batch in loader:
            ...
```

### 2. Custom DataLoader for Flyte (not reusing existing L2DDataset)

The existing `L2DDataset` downloads from HuggingFace at training time — that's `data_ingest`'s job in Flyte.

For Flyte execution:
- `data_processing` outputs WebDataset shards to S3
- `train_il` reads those shards via `make_pre_extracted_loader` (already exists in `pre_extracted.py`)
- The loader expects: `{key}.cam_X.jpg` + `{key}.ego.npy` + `{key}.meta.json` in tar files

The existing `pre_extracted.py` DataLoader is the starting point but may be modified to match the exact shard format produced by `data_processing`.

### 3. Dual-Mode: Flyte vs Local Execution

The model code (`AutoE2E`, `TrajectoryImitationLoss`, etc.) must work in both modes:

```python
def get_dataloader(shard_dirs: list[str] | None, args=None):
    """Factory: Flyte mode uses shards, local mode uses L2DDataset."""
    if shard_dirs:
        # Flyte path: pre-extracted WebDataset shards
        from data_parsing.pre_extracted import make_pre_extracted_loader
        return make_pre_extracted_loader(shard_dirs, batch_size=args.batch_size)
    else:
        # Local path: direct HF download via L2DDataset (existing behavior)
        from data_parsing.l2d import L2DDataset
        dataset = L2DDataset(repo_id=args.repo_id, episodes=args.episodes)
        return DataLoader(dataset, batch_size=args.batch_size, ...)
```

This means:
- `train.py` (local CLI) continues to work: `python train.py --backbone swin_v2_tiny --epochs 3`
- Flyte `train_il` task calls the same model/loss code but uses the WebDataset DataLoader
- The `if` branch is determined by whether `shard_dirs` is provided (Flyte) or not (local)

### 4. No Duplication of Model Logic

The Flyte task imports and uses the exact same:
- `AutoE2E` from `model_components.auto_e2e`
- `TrajectoryImitationLoss` from `model_components.losses`
- `compute_open_loop_metrics`, `gate_check` from `evaluation.metrics`

These are NOT reimplemented inside the workflow. The workflow is the orchestration layer only.

### 5. Separate Docker Image per Pipeline Stage

Each task has its own Dockerfile and container image, even if dependencies overlap:

```
platform/docker/
├── data-prep/Dockerfile       → data_ingest + data_processing tasks
├── training/Dockerfile        → train_il task
├── eval/Dockerfile            → evaluate task
├── offline-rl/Dockerfile      → train_offline_rl task
```

ECR repositories (already exist):
- `auto-e2e/data-prep`
- `auto-e2e/training`
- `auto-e2e/eval`
- `auto-e2e/offline-rl`

Rationale:
- Different GPU/CPU requirements per stage (data-prep = CPU only, training = GPU)
- Independent version pinning (e.g., training pins PyTorch CUDA, eval can be lighter)
- Smaller blast radius for dependency changes
- Clear ownership: each Dockerfile declares exactly what that stage needs

All images COPY `Model/` for shared model code access. Each image installs only the Python packages required for its specific task.
