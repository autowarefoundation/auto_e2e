# Flyte Workflow Parameter Reference

Launch workflows from Flyte Console: https://d1fk8c95f6ice9.cloudfront.net/console

Project: `auto-e2e` / Domain: `development`

---

## wf_data_ingest

Downloads a raw dataset and converts it into WebDataset shards for training.

| Parameter | Type | Default | UI Control | Description |
|-----------|------|---------|------------|-------------|
| `dataset` | Enum | `L2D` | Dropdown | Source dataset. `L2D` = yaak-ai/L2D (HuggingFace), `NVIDIA_PHYSICAL_AI` = nvidia/PhysicalAI |
| `version_tag` | str | `10hz-224px-v1` | Text | Processing version tag. Change when reprocessing with different settings (e.g., `20hz-256px-v2`) |
| `hz` | int | `10` | Number | Resampling frequency for egomotion/frames (Hz). Higher = more data |
| `image_size` | int | `224` | Number | Camera image resize target (square px). Must match model input size |
| `episodes` | int | `5` | Number | Number of episodes to process. Use `-1` for all, `1-5` for testing |

**Output**: `FlyteDirectory` — S3 directory containing WebDataset `.tar` shards

---

## wf_train_il

Imitation Learning (supervised training). Learns a driving policy from expert demonstrations.

| Parameter | Type | Default | UI Control | Description |
|-----------|------|---------|------------|-------------|
| `shards` | FlyteDirectory | (required) | URI input | Output URI from `wf_data_ingest`. Copy from Flyte Console → previous execution → Outputs tab |
| `backbone` | Enum | `SWIN_V2_TINY` | Dropdown | Image encoder. `SWIN_V2_TINY` (22M params, good balance), `CONVNEXT_V2_TINY` (28M, higher accuracy), `RESNET_50` (25M, fastest) |
| `fusion_mode` | Enum | `CONCAT` | Dropdown | Multi-camera feature fusion. `CONCAT` (simple concatenation, fastest), `CROSS_ATTN` (attention mechanism, higher accuracy), `BEV` (Bird's Eye View, spatial understanding) |
| `epochs` | int | `10` | Number | Training epochs. Too few = underfitting, too many = overfitting |
| `batch_size` | int | `4` | Number | Mini-batch size. GPU memory dependent (g6e.4xlarge L40S 48GB supports up to 8-16) |
| `lr` | float | `0.001` | Number | Learning rate. Too high = divergence, too low = slow convergence. Typical range: `0.0001` - `0.001` |

**Output**: `FlyteFile` — Best checkpoint (`.pt`). Automatically logged to MLflow (params, metrics, model registry).

---

## wf_evaluate

Open-loop evaluation. Compares predicted trajectories against ground truth.

| Parameter | Type | Default | UI Control | Description |
|-----------|------|---------|------------|-------------|
| `checkpoint` | FlyteFile | (required) | URI input | Output URI from `wf_train_il` (checkpoint file) |
| `shards` | FlyteDirectory | (required) | URI input | Evaluation data. Output URI from `wf_data_ingest` |

**Output**: `EvalMetrics` (NamedTuple)
- `ade` (float): Average Displacement Error (meters) — mean deviation across all timesteps. Lower is better.
- `fde` (float): Final Displacement Error (meters) — deviation at final predicted point. Lower is better.
- `gate_pass` (bool): `True` if ade < 2.0 AND fde < 4.0

Metrics automatically logged to MLflow experiment `auto-e2e/evaluation`.

---

## wf_train_offline_rl

Offline RL (IQL) to refine IL policy. No simulator needed — uses recorded data only.

| Parameter | Type | Default | UI Control | Description |
|-----------|------|---------|------------|-------------|
| `pretrained` | FlyteFile | (required) | URI input | Output URI from `wf_train_il` (IL checkpoint to refine) |
| `shards` | FlyteDirectory | (required) | URI input | Training data. Output URI from `wf_data_ingest` |
| `epochs` | int | `5` | Number | RL training epochs |
| `tau` | float | `0.7` | Number | IQL expectile parameter. Higher = more conservative (0.5=mean, 1.0=max). Typical: `0.7-0.9` |
| `beta` | float | `3.0` | Number | Advantage weight temperature. Higher = favor expert-like actions more. Typical: `1.0-10.0` |

**Output**: `FlyteFile` — RL-refined checkpoint (`.pt`). Logged to MLflow experiment `auto-e2e/offline-rl`.

---

## wf_full_pipeline

Runs all 4 stages sequentially: Ingest → IL Training → Evaluation → Offline RL.

| Parameter | Type | Default | UI Control | Description |
|-----------|------|---------|------------|-------------|
| `dataset` | Enum | `L2D` | Dropdown | (same as wf_data_ingest) |
| `version_tag` | str | `10hz-224px-v1` | Text | (same as wf_data_ingest) |
| `backbone` | Enum | `SWIN_V2_TINY` | Dropdown | (same as wf_train_il) |
| `fusion_mode` | Enum | `CONCAT` | Dropdown | (same as wf_train_il) |
| `epochs_il` | int | `10` | Number | IL training epochs |
| `epochs_rl` | int | `5` | Number | RL training epochs |
| `batch_size` | int | `4` | Number | IL batch size |
| `lr` | float | `0.001` | Number | IL learning rate |

**Output**: `FlyteFile` — Final RL-refined checkpoint

---

## How to Find FlyteDirectory / FlyteFile URIs

When a task needs the output of a previous task:

1. Go to Flyte Console → **Executions**
2. Click the completed execution
3. Click the task node → **Outputs** tab
4. Copy the URI (format: `s3://auto-e2e-platform-artifacts-381491877296/...`)
5. Paste into the new workflow's input field

---

## First Run Guide

**Easiest**: Launch `wf_full_pipeline` with all defaults. Everything runs end-to-end.

**Step by step**:
1. Launch `wf_data_ingest` (dataset=`L2D`, all defaults) → wait for completion → copy output URI
2. Launch `wf_train_il` (paste shards URI, select backbone/fusion, defaults for rest) → copy output URI
3. Launch `wf_evaluate` (paste checkpoint + shards URIs)
4. Launch `wf_train_offline_rl` (paste pretrained + shards URIs)
