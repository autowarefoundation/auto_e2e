# Getting Started: AutoE2E ML Platform

This guide walks you through the typical ML engineer workflow: modify the model/pipeline, run training via Flyte, and analyze results in MLflow.

## Prerequisites

- Access to Flyte Console: https://d1fk8c95f6ice9.cloudfront.net/console
- Access to MLflow UI: https://d33520viyb0smg.cloudfront.net/
- Project: `auto-e2e` / Domain: `development`

---

## Quick Start: Run Your First Training

1. Open [Flyte Console](https://d1fk8c95f6ice9.cloudfront.net/console)
2. Navigate to **auto-e2e** → **development** → **Workflows**
3. Select `wf_full_pipeline`
4. Click **Launch Workflow**
5. Leave all defaults and click **Launch**

This runs: Data Ingest → IL Training → Evaluation → Offline RL

---

## Typical Workflow

### 1. Modify Model or Pipeline

Edit `platform/pipelines/workflows.py`. Each `@task` contains the full logic:

```python
@task(container_image=TRAINING_IMAGE, requests=Resources(gpu="1"))
def train_il(shards: FlyteDirectory, backbone: Backbone, ...) -> FlyteFile:
    # Your training logic here
    ...
    return FlyteFile("/tmp/ckpt/best.pt")
```

### 2. Build & Register

```bash
# Build Docker images
aws codebuild start-build --project-name auto-e2e-platform-build-images \
  --source-type-override S3 \
  --source-location-override auto-e2e-platform-codebuild-cache-381491877296/source.zip

# Register workflows to Flyte
aws codebuild start-build --project-name auto-e2e-platform-flyte-register \
  --source-type-override S3 \
  --source-location-override auto-e2e-platform-codebuild-cache-381491877296/source.zip
```

### 3. Launch from Flyte UI

- Go to Flyte Console → `auto-e2e` → `development` → Workflows
- Select the workflow (e.g., `wf_train_il`)
- Fill parameters (dropdowns for backbone/fusion, numbers for hyperparams)
- Click **Launch**

### 4. Monitor Execution

- Flyte Console shows real-time task status (Pending → Running → Succeeded)
- Click into a task node to see logs

### 5. Analyze Results in MLflow

- Open [MLflow UI](https://d33520viyb0smg.cloudfront.net/)
- Experiments:
  - `auto-e2e/il-training` — training loss, hyperparams
  - `auto-e2e/evaluation` — ADE, FDE, gate pass/fail
  - `auto-e2e/offline-rl` — IQL losses
- Compare runs: select multiple runs → **Compare** button
- Filter by `model/backbone`, `model/fusion_mode`, or `git_commit`

---

## Workflows Reference

| Workflow | Purpose | When to Use |
|----------|---------|-------------|
| `wf_data_ingest` | Download & preprocess dataset | Once per dataset version, or when preprocessing changes |
| `wf_train_il` | Imitation Learning training | Every training experiment |
| `wf_evaluate` | Open-loop eval (ADE/FDE) | After each training to check quality |
| `wf_train_offline_rl` | Offline RL refinement | To improve IL checkpoint without simulator |
| `wf_full_pipeline` | All stages end-to-end | Quick full run with one click |

---

## Data Flow Between Tasks

```
wf_data_ingest          wf_train_il           wf_evaluate
─────────────           ───────────           ───────────
    │                       │                     │
    ▼                       ▼                     ▼
FlyteDirectory ──S3──▶ shards input       checkpoint + shards
(WebDataset shards)     │                     │
                        ▼                     ▼
                   FlyteFile ───S3───▶  evaluation metrics
                   (checkpoint)            (ADE, FDE)
```

Flyte automatically handles S3 upload/download between tasks. You never write S3 code — just return Python objects and accept them as function arguments.

---

## MLflow: What Gets Logged

Each training run records:

| Category | Examples |
|----------|---------|
| **Params** (filterable in UI) | `model/backbone`, `model/fusion_mode`, `train/lr`, `train/epochs`, `git_commit` |
| **Metrics** (plotted as charts) | `train_loss` (per epoch), `ade`, `fde` |
| **Artifacts** (downloadable) | `best.pt` (checkpoint), `config.yaml` |
| **Model Registry** | Versioned under `auto-e2e-driving-policy` |

### Comparing Experiments

1. Go to MLflow → Experiment `auto-e2e/il-training`
2. Select 2+ runs with checkboxes
3. Click **Compare**
4. See side-by-side params, metric charts, and artifacts

---

## Parameter Tuning Guide

### What to Try First

| Parameter | Range | Impact |
|-----------|-------|--------|
| `backbone` | swin_v2_tiny → convnext_v2_tiny | Architecture capacity |
| `fusion_mode` | concat → cross_attn → bev | Spatial understanding |
| `lr` | 1e-4 to 1e-3 | Convergence speed |
| `batch_size` | 4 to 16 | GPU utilization vs. generalization |
| `epochs` | 10 to 50 | Underfitting vs. overfitting |

### Offline RL Tuning

| Parameter | Range | Impact |
|-----------|-------|--------|
| `tau` | 0.5 to 0.9 | Conservative (high) vs. aggressive (low) |
| `beta` | 1.0 to 10.0 | How much to trust expert data |

---

## Adding a New Dataset

1. Add enum value in `workflows.py`:
   ```python
   class Dataset(enum.Enum):
       L2D = "yaak-ai/L2D"
       NVIDIA_PHYSICAL_AI = "nvidia/PhysicalAI"
       YOUR_DATASET = "org/your-dataset"  # add here
   ```

2. Implement download logic in `data_ingest` task

3. Re-build images and re-register:
   ```bash
   # Upload source, build images, register
   ```

4. New dataset appears as dropdown option in Flyte UI

---

## Adding a New Backbone

1. Add enum value:
   ```python
   class Backbone(enum.Enum):
       SWIN_V2_TINY = "swin_v2_tiny"
       YOUR_MODEL = "your_model_name"  # add here
   ```

2. Implement model construction in `train_il` task

3. Re-build + re-register

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Task stuck in Pending | GPU node provisioning (1-3 min). Check Flyte Console logs. |
| MLflow connection error | Verify `MLFLOW_TRACKING_URI` env in task. Should be `http://mlflow.mlflow.svc.cluster.local:5000` |
| Out of GPU memory | Reduce `batch_size` or use smaller `backbone` |
| Workflow not visible | Re-run `auto-e2e-platform-flyte-register` CodeBuild |
