# AutoE2E Flyte Pipeline — Tasks & Usage Guide

Detailed reference for every Flyte task and workflow in `Platform/pipelines/workflows.py`.

- **Flyte Console**: https://d1fk8c95f6ice9.cloudfront.net/console (Cognito auth)
- **MLflow UI**: https://d33520viyb0smg.cloudfront.net/ (no auth)
- **Project**: `auto-e2e` / **Domain**: `development`

---

## Pipeline Overview

```
data_ingest ──▶ data_processing ──▶ train_il ──▶ evaluate_il_policy
 (per dataset)   (per dataset)          │              │ (MLflow: imitation-learning)
                                        ▼              ▼
                                 train_offline_rl ──▶ evaluate_rl_policy
                                                          (MLflow: offline-rl)
```

Each dataset is **ingested and processed independently** into its own WebDataset
shard directory. Training/eval tasks receive **all** datasets' shards as a list and
**select** the one matching the `dataset` argument (single-dataset training today;
multi-dataset on one model tracked in [#77](https://github.com/autowarefoundation/auto_e2e/issues/77)).

---

## Tasks (each runs in its own container image)

### 1. `data_ingest`  — image: `auto-e2e/data-prep`
**Purpose**: Download the raw dataset from HuggingFace. No transformation.

| Aspect | Detail |
|--------|--------|
| Inputs | `dataset: Dataset`, `episodes: int` |
| Output | `FlyteDirectory` — raw dataset cache |
| HF auth | Reads `HF_TOKEN` from the `hf-token` K8s Secret via Flyte `secret_requests` (never a workflow input → not shown in UI) |
| Resources | cpu=2, mem=24Gi, ephemeral=50Gi |

**Behavior by dataset**:
- **L2D** (`yaak-ai/L2D`): `LeRobotDataset(repo_id, episodes)` downloads the lerobot cache, copied to the output dir.
- **NVIDIA** (`nvidia/PhysicalAI-Autonomous-Vehicles`): `physical_ai_av` SDK downloads clip chunk zips, then unpacks them into the parser layout (`camera/<cam>/`, `labels/egomotion/`).

---

### 2. `data_processing`  — image: `auto-e2e/data-prep`
**Purpose**: Pre-extract aligned frames + egomotion into WebDataset shards.
Solves [#30](https://github.com/autowarefoundation/auto_e2e/issues/30) (no video decode at training time).

| Aspect | Detail |
|--------|--------|
| Inputs | `raw_data: FlyteDirectory`, `dataset: Dataset`, `hz: int`, `image_size: int`, `episodes: int` |
| Output | `FlyteDirectory` — `train-*.tar` shards + `manifest.json` |
| Resources | cpu=4, mem=16Gi, ephemeral=50Gi |

**Behavior**: Builds the dataset class (`L2DDataset` or `NvidiaAVDataset`) — both emit the
same sample schema (`visual_tiles (V,3,H,W)`, `egomotion_history (256)`,
`trajectory_target (128)`). Each sample is written to a WebDataset tar:
```
s{idx}.cam_0.jpg … s{idx}.cam_{V-1}.jpg   # JPEG frames, resized to image_size
s{idx}.ego.npy                            # 384 floats (256 history + 128 target)
s{idx}.meta.json                          # {idx, dataset}
```
`manifest.json` records `dataset`, `total_samples`, `num_views`, `hz`, `image_size`.
Each dataset stays a **separately-packed** WebDataset.

---

### 3. `train_il`  — image: `auto-e2e/training` (GPU)
**Purpose**: Train the `AutoE2E` model with imitation (trajectory) loss.

| Aspect | Detail |
|--------|--------|
| Inputs | `shards: List[FlyteDirectory]` (all datasets), `dataset: Dataset` (selects which), `backbone`, `epochs`, `batch_size`, `lr`, `weight_decay`, `grad_clip`, `amp` |
| Output | `TrainOutput(checkpoint: FlyteFile, metadata: FlyteFile)` |
| Resources | cpu=4, mem=16Gi, gpu=1 |

**Behavior**:
- `_select_shard_dir` picks the shard dir whose `manifest.json` matches `dataset`.
- `num_views` is **detected from the data** (peek first batch) so the model matches the dataset's camera count.
- Builds `AutoE2E(backbone, num_views, ...)` (BEV fusion is hardcoded since PR #94; no `fusion_mode` arg), trains with `TrajectoryImitationLoss` (smooth-L1), AdamW, AMP (bf16), grad clipping. A zero `map_input` is fed since shards carry no rendered nav-map yet (#77).
- Saves checkpoint (`model_state_dict` + config) and a `metadata.json` capturing full provenance: data, model, training hyperparams, Flyte execution id, docker image.
- **Does NOT log to MLflow** — that is the eval task's job (single point of truth).

---

### 4. `train_offline_rl`  — image: `auto-e2e/offline-rl` (GPU)
**Purpose**: Refine the IL checkpoint with Offline RL (IQL).

| Aspect | Detail |
|--------|--------|
| Inputs | `pretrained: FlyteFile` (IL ckpt), `shards: List[FlyteDirectory]`, `il_metadata: FlyteFile`, `dataset`, `epochs`, `tau`, `beta` |
| Output | `TrainOutput(checkpoint, metadata)` |
| Resources | cpu=4, mem=16Gi, gpu=1 |

**Behavior**: Loads the IL model, runs IQL-style advantage-weighted regression for `epochs`.
Output metadata nests the full IL metadata under `base_model.il_metadata` so the eval run
records the entire lineage.

---

### 5. `evaluate_il_policy`  — image: `auto-e2e/eval` (GPU)
**Purpose**: Open-loop evaluation of the **IL** policy + MLflow logging.

| Aspect | Detail |
|--------|--------|
| Inputs | `checkpoint`, `shards: List[FlyteDirectory]`, `train_metadata`, `dataset` |
| Output | `EvalMetrics(ade, fde, gate_pass)` |
| MLflow experiment | **`imitation-learning`** |

### 6. `evaluate_rl_policy`  — image: `auto-e2e/eval` (GPU)
**Purpose**: Open-loop evaluation of the **Offline-RL** policy + MLflow logging.

| Aspect | Detail |
|--------|--------|
| Inputs | same as above |
| MLflow experiment | **`offline-rl`** |

**Shared behavior** (`_run_evaluation`): loads the model, predicts trajectories, integrates
`(accel, curvature)` → (x,y) via `evaluation.metrics.integrate_trajectory`, computes
**ADE / FDE**, gate = `ade<2.0 and fde<4.0`. Logs **everything to one MLflow run**:
all params (`data/*`, `model/*`, `train/*`, `rl/*`, `ctx/*`), training loss curve,
eval metrics, `config.yaml` + checkpoint artifacts, and registers the model in
`auto-e2e-driving-policy`.

> `evaluate_il_policy` and `evaluate_rl_policy` are **separate tasks** (distinct nodes in
> the Flyte UI) sharing one implementation — so IL-eval and RL-eval are easy to tell apart.

---

## Workflows (launchable from the Flyte Console)

### `wf_data_ingest`
Single task. Params: `dataset`, `episodes`. Output: raw `FlyteDirectory`.

### `wf_data_processing`
Pure deterministic packing (no teacher call). Params: `raw_data` (URI from a prior
ingest), `dataset`, `hz`, `image_size`, `episodes`, `world_model`,
`reasoning_labels` (optional URI from `wf_generate_reasoning_labels`). When
`reasoning_labels` is set, each sample's frozen label is JOINed in by `sample_id`
and embedded as a `reasoning.json` shard member. Output: WebDataset shards
`FlyteDirectory`.

### `wf_generate_reasoning_labels`
The single place the offline teacher (Cosmos) is called. Params: `raw_data`,
`dataset`, `episodes`, `split`, `teacher` (`openai_compatible` / `mock` /
`cached`), `prompt_version`. Enumerates samples from raw (same parser/episodes as
`wf_data_processing`, so `sample_id` matches), calls the teacher only on a
`LabelCache` miss (sample_id-keyed in `REASONING_LABELS_CACHE_BUCKET`, so re-packing
never re-bills the endpoint, #98/#117), and writes a versioned label artifact
(whole-record `records.jsonl` for the JOIN + flattened parquet/jsonl for
analytics). Output: label `FlyteDirectory`.

### `wf_create_dataset`
`data_ingest → [reasoning_teacher != none] generate_reasoning_labels →
data_processing (JOIN labels)`. Produces the ready-to-train shards in one shot.
Params: `dataset`, `episodes`, `image_size`, `world_model`, `reasoning_teacher`,
`prompt_version`. Output: WebDataset shards `FlyteDirectory` (the shards ARE the
dataset; the label artifact persists independently via the generate task output +
the cache, so it is not a separate return value).

### `wf_train_il`
`train_il → evaluate_il_policy`. Params: `shards` (list), `dataset`, `backbone`, `epochs`, `batch_size`, `lr`. Logs to `imitation-learning`.

### `wf_ingest_train_eval`
`data_ingest + data_processing (all datasets) → train_il → evaluate_il_policy`.
Same as `wf_full_pipeline` but stops after IL evaluation (no offline RL). Use when
you only need a supervised checkpoint + open-loop metrics, or when the RL step is
too memory-hungry at the current BEV resolution (#77). Params: `dataset`,
`episodes`, `backbone`, `epochs_il`, `batch_size`, `lr`.

### `wf_train_offline_rl`
`train_offline_rl → evaluate_rl_policy`. Params: `pretrained`, `shards`, `il_metadata`, `dataset`, `epochs`, `tau`, `beta`. Logs to `offline-rl`.

### `wf_full_pipeline`  ← main entry point
Ingests + processes **all** datasets (L2D, NVIDIA) in parallel, then runs
IL train+eval and RL train+eval on the dataset selected by `dataset`.

| Param | Default | Meaning |
|-------|---------|---------|
| `dataset` | `L2D` | which processed dataset to train on |
| `episodes` | 3 | episodes per dataset to ingest |
| `backbone` | `SWIN_V2_TINY` | image encoder |
| `epochs_il` | 3 | IL training epochs |
| `epochs_rl` | 3 | RL refinement epochs |
| `batch_size` | 4 | |
| `lr` | 1e-4 | IL learning rate |
| `tau` | 0.7 | IQL expectile |
| `beta` | 3.0 | IQL advantage temperature |

HF token is **not** a parameter — it is injected from the `hf-token` K8s Secret.

---

## How to launch (Flyte Console)

1. Open the [Flyte Console](https://d1fk8c95f6ice9.cloudfront.net/console) → log in with Cognito.
2. `auto-e2e` → `development` → **Workflows** → `wf_full_pipeline`.
3. **Launch Workflow**, pick `dataset` (dropdown) and hyperparameters, **Launch**.
4. Watch nodes go Pending → Running → Succeeded.
5. Open [MLflow](https://d33520viyb0smg.cloudfront.net/) → `imitation-learning` / `offline-rl`
   experiments to compare runs (filter by `model/backbone`, `data/dataset`, etc.).

## Enum values (UI dropdowns)

| Enum | Options (value) |
|------|-----------------|
| `Dataset` | `yaak-ai/L2D`, `nvidia/PhysicalAI-Autonomous-Vehicles` |
| `Backbone` | `swin_v2_tiny`, `conv_next_v2_tiny`, `res_net_50` |

View fusion is no longer an enum/parameter: BEV fusion is hardcoded in the model
since PR #94 (concat / cross_attn were removed).

> When launching via the raw Admin API, pass the **enum value** (e.g.
> `nvidia/PhysicalAI-Autonomous-Vehicles`), not the enum name.
