# How to Use Flyte — AutoE2E Pipeline UI Guide

A practical, screen-by-screen guide to the Flyte Console for the AutoE2E training
platform. Organized by **what you want to do**, so you can jump straight to your use case.

- **Flyte Console**: https://d1fk8c95f6ice9.cloudfront.net/console
- **Login**: Cognito-protected. Credentials are shared with **Core Contributors only** —
  ask **Ryota Yamada** for access. You will be redirected to a Cognito login page first.
- **Project**: `auto-e2e` · **Domain**: `development`

> For task internals and parameters, see `flyte_workflow_parameters.md`.
> This document focuses on **navigating the UI**.
> The trajectory-overlay production path is ops-only and uses a VPC-local
> CodeBuild launcher; see Use case G.

---

## 0. First time: logging in and finding the project

1. Open https://d1fk8c95f6ice9.cloudfront.net/console
2. You are redirected to the Cognito hosted login. Sign in with the credentials
   provided to Core Contributors (ask **Ryota Yamada** — they are never stored in git).
3. You land on the Flyte Console home. Top-left, make sure the **Project** selector
   shows **`auto-e2e`** and the **Domain** selector shows **`development`**.
   (Other Flyte sample projects were archived, so `auto-e2e` should be the only one.)
4. The left sidebar has three main sections you will use:
   - **Workflows** — the DAGs you can launch
   - **Tasks** — individual reusable steps
   - **Executions** — the history of every run

---

## Core concepts (30-second version)

| Term | What it is | Where in UI |
|------|------------|-------------|
| **Task** | One containerized step (e.g. `train_il`) | Tasks tab |
| **Workflow** | A DAG wiring tasks together (e.g. `wf_full_pipeline`) | Workflows tab |
| **Launch Plan** | A runnable, versioned binding of a workflow + default inputs | "Launch Workflow" button |
| **Execution** | One actual run of a launch plan | Executions tab |
| **Node** | One task instance inside a running execution (n0, n1, …) | Inside an execution |

---

## Use case A — "I want to train a model end-to-end"

**You are**: an ML engineer who wants a full ingest → train → evaluate run.

**Workflow to use**: `wf_full_pipeline`

### Steps
1. Sidebar → **Workflows** → click **`workflows.wf_full_pipeline`**.
2. Top-right → **Launch Workflow**.
3. A form appears with the inputs. Fill in:
   - **`dataset`** (dropdown): `yaak-ai/L2D` or `nvidia/PhysicalAI-Autonomous-Vehicles`
     — which processed dataset to actually train on.
   - **`backbone`** (dropdown): `swin_v2_tiny` / `conv_next_v2_tiny` / `res_net_50`
   - **`epochs_il`**, **`epochs_rl`**, **`batch_size`**, **`lr`**, **`tau`**, **`beta`**, **`episodes`** — numbers, defaults are fine for a smoke run.

   There is no `fusion_mode` input: BEV fusion is hardcoded in the model since
   PR #94 (concat / cross_attn were removed). To run IL without the memory-hungry
   offline-RL step, launch **`workflows.wf_ingest_train_eval`** instead.
   - **No `hf_token` field** — the HF token is injected from a Kubernetes Secret.
4. Click **Launch**. You are taken to the new **execution** page.

### What you will see while it runs
- A **graph (DAG) view** with nodes: `n0 … n7`.
- Nodes light up Pending → Running → Succeeded.
- Both datasets are ingested + processed in parallel (you will see two ingest and
  two processing nodes running side by side), then the selected dataset flows into
  training and evaluation.

### When it finishes
- The execution badge turns **Succeeded** (green).
- Jump to **MLflow** to read the metrics (see Use case D).

---

## Use case B — "I just want to (re)build the dataset shards"

**You are**: someone iterating on preprocessing, or preparing data before training.

**Workflows**: `wf_data_ingest`, then `wf_data_processing`

### Steps
1. **Workflows → `wf_data_ingest` → Launch Workflow**.
   - Set `dataset` and `episodes`. Launch.
   - When it succeeds, open the execution → **Outputs** tab → copy the
     `FlyteDirectory` URI (the raw dataset cache).
2. **Workflows → `wf_data_processing` → Launch Workflow**.
   - Paste the raw URI into **`raw_data`**.
   - Set `dataset`, `hz`, `image_size`, `episodes`. Launch.
   - Output is a `FlyteDirectory` of WebDataset `.tar` shards + `manifest.json`.

### How to read the output URI
- Open the execution → **Nodes** → click the task node → **Outputs** panel.
- Each `FlyteDirectory` / `FlyteFile` shows an `s3://…` URI you can reuse as input
  to a later workflow.

---

## Use case C — "I have shards already and just want to train / refine"

**You are**: someone who already has processed shards and wants to skip ingest.

**Workflows**: `wf_train_il`, then `wf_train_offline_rl`

### IL training
1. **Workflows → `wf_train_il` → Launch Workflow**.
2. **`shards`** is a **list of `FlyteDirectory`** — add one entry per dataset's
   processed shard dir (paste the URIs from Use case B). The task picks the one
   matching **`dataset`**.
3. Set `backbone`, `epochs`, `batch_size`, `lr`. Launch.
4. Output: a `TrainOutput` with `checkpoint` and `metadata` FlyteFiles
   (grab their URIs from the Outputs panel).

### Offline-RL refinement
1. **Workflows → `wf_train_offline_rl` → Launch Workflow**.
2. Fill:
   - **`pretrained`**: the IL `checkpoint` URI.
   - **`il_metadata`**: the IL `metadata` URI.
   - **`shards`**: same list of shard dirs.
   - **`dataset`**, `epochs`, `tau`, `beta`.
3. Launch. It refines the IL policy with IQL and runs `evaluate_rl_policy`.

---

## Use case D — "I want to see results / compare experiments"

**You are**: anyone evaluating model quality.

Flyte shows **execution status**; **MLflow** shows **metrics**. Use both.

### In Flyte (did it run? where did it fail?)
1. Sidebar → **Executions**. The list shows every run with status, start time, duration.
2. Click an execution to open the DAG. Red node = failure.
3. Click a failed node → **Logs** (Kubernetes logs) and the error message panel.

### In MLflow (how good is the model?)
1. Open https://d33520viyb0smg.cloudfront.net/
2. Pick an experiment:
   - **`imitation-learning`** — IL runs (logged by `evaluate_il_policy`)
   - **`offline-rl`** — RL-refined runs (logged by `evaluate_rl_policy`)
3. The run table shows one row per run. Key columns:
   - `model/backbone`, `model/fusion_mode`, `data/dataset`
   - `eval/ade`, `eval/fde`, `eval/gate_pass`
   - `train/lr`, `train/epochs`, etc.
4. Select multiple runs → **Compare** to overlay loss curves and compare params.
5. Each run also stores `config.yaml` + the checkpoint as **artifacts**, and the
   model is registered under **`auto-e2e-driving-policy`** in the Model Registry.

---

## Use case E — "Something failed. How do I debug?"

1. **Executions** → open the failed (red) execution.
2. In the DAG, find the red node. Note which task it is (`data_ingest`, `train_il`, …).
3. Click the node → right panel:
   - **Execution Details**: the error message (e.g. OOMKilled, ImagePullBackOff, a Python traceback).
   - **Logs**: live/last Kubernetes pod logs for that task.
   - **Inputs / Outputs**: the exact data the node received and produced.
4. Common failures and meaning:
   | Symptom | Likely cause |
   |---------|--------------|
   | `OOMKilled` (exit 137) | task needs more memory — raise `Resources(mem=…)` |
   | `ImagePullBackOff` | ECR image missing/wrong tag |
   | `exceeded quota: project-quota` | namespace ResourceQuota too small |
   | `WebIdentityErr / AssumeRoleWithWebIdentity` | Flyte S3 auth not on access-key |
   | `Bus error / shared memory` | DataLoader `num_workers` too high for `/dev/shm` |
5. Fix the root cause, then **Relaunch** (button on the execution page reuses the
   same inputs) or launch fresh from the workflow.

---

## Use case F — "I want to monitor a long-running training run"

1. Open the execution page. It **auto-refreshes**.
2. The **timeline / Gantt view** (toggle near the graph) shows how long each node
   takes and what is running now.
3. Click the running training node → **Logs** to watch epoch-by-epoch loss prints.
4. GPU nodes (`train_il`, `train_offline_rl`, eval tasks) may sit in
   **Pending / ContainerCreating** for 1–3 minutes while a GPU node is provisioned
   by EKS Auto Mode — this is normal.

---

## Use case G — "I want to publish v2.1 and precompute Console overlays"

**You are**: a platform operator publishing an immutable dataset snapshot and
precomputing one registered model's canonical trajectory overlays.

**Workflow**: `wf_create_publish_and_precompute_overlays`

The DataModelConsole never invokes this workflow. Launch it through the
VPC-local CodeBuild project so Flyte registration and every task image use ECR
digests rather than mutable tags.

### Prepare the tested source and images

Run these commands from the tested feature-branch checkout:

```bash
export AWS_PROFILE=autowarefoundation
export AWS_REGION=us-west-2
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
CACHE_BUCKET="auto-e2e-platform-codebuild-cache-${ACCOUNT_ID}"

git archive --format=zip --output=/tmp/auto-e2e-source.zip HEAD
aws s3 cp /tmp/auto-e2e-source.zip "s3://${CACHE_BUCKET}/source.zip"

aws codebuild start-build \
  --project-name auto-e2e-platform-build-images
```

Wait for the image build to reach `SUCCEEDED`. The overlay launcher resolves
the resulting `training`, `eval`, `offline-rl`, and `data-prep` images to ECR
digests. It also recomputes the preprocessing and inference source digests
inside the source bundle; Flyte tasks reject any mismatch at runtime.

### Launch the one-episode smoke

Choose an immutable numeric version from the MLflow registered model
`auto-e2e-driving-policy`; do not use a moving alias.

```bash
MODEL_VERSION=30  # Example only; replace with the version you selected.
aws codebuild start-build \
  --project-name auto-e2e-platform-overlay-launch \
  --environment-variables-override \
    "name=MODEL_VERSION,value=${MODEL_VERSION},type=PLAINTEXT"
```

The launcher defaults to `EPISODES=1`, `DATASET_VERSION=v2.1`, and
`BASE_SEEDS=[0]`. It derives a `kitscenes-smoke-<digest>` publication name from
the data contract, inference code, image, source revision, and smoke size. This
keeps the smoke snapshot separate from the write-once production coordinate.

CodeBuild prints the remote Flyte execution. In Flyte Console, open
**Executions** and inspect the newest
`wf_create_publish_and_precompute_overlays` run. The high-level order is:

```text
wf_create_dataset_sharded
  -> wf_publish_dataset_snapshot
  -> wf_precompute_overlays
  -> overlay manifest
  -> OVLSET building-to-ready gate
```

Treat the smoke as successful only when the Flyte execution succeeds and the
Console can read the published model overlay. A CodeBuild `SUCCEEDED` status
only confirms that the remote execution was submitted.

### Launch the full immutable publication

After reviewing smoke duration, GPU utilization, storage, and estimated cost:

```bash
aws codebuild start-build \
  --project-name auto-e2e-platform-overlay-launch \
  --environment-variables-override \
    "name=MODEL_VERSION,value=${MODEL_VERSION},type=PLAINTEXT" \
    "name=PUBLISHED_DATASET,value=kitscenes,type=PLAINTEXT" \
    "name=EPISODES,value=0,type=PLAINTEXT"
```

`EPISODES=0` is rejected unless `PUBLISHED_DATASET` is explicit. The production
coordinate is `kitscenes/v2.1`.

### Retry and immutability rules

- Retrying the exact same model, source, image, seeds, and contract is
  idempotent. Existing compatible S3 objects and DynamoDB records are reused.
- A conflicting body or identity at the same model/dataset/version/schema
  coordinate fails; it is never overwritten.
- Do not change `BASE_SEEDS`, the model artifact, or inference contract after a
  coordinate is ready. Publish a new dataset/schema coordinate for an
  intentional canonical-result change.
- The ready gate is written last. A failed or `building` set is not advertised
  by the Console, and retrying a ready set never moves it back to `building`.

---

## Use case H — "I want an MP4 report for an overlay shard"

**Workflow**: `wf_export_trajectory_report`

Use this after the overlay set is `ready`. It is a CPU-only optional export and
does not load the model checkpoint or run inference again.

1. Copy one published shard URI from the dataset manifest.
2. Copy its matching `overlay.bin.gz` URI from the overlay manifest or DynamoDB
   pointer.
3. In Flyte Console, launch `wf_export_trajectory_report`.
4. Set `shard` and `overlay` to those immutable S3 URIs. Optionally set
   `scene_uids`, `seed_index`, `camera_index`, and `max_frames_per_scene`.
5. Download the returned `FlyteDirectory`, which contains per-scene MP4 files,
   thumbnails, metrics, and `manifest.json`.

The task joins predictions by `sample_uid` and rejects a shard/overlay mismatch.
It is cached by immutable input URI and report schema.

---

## Reading the DAG of `wf_full_pipeline`

```
n0 data_ingest(L2D)        n2 data_ingest(NVIDIA)     ← run in parallel
        │                          │
n1 data_processing(L2D)    n3 data_processing(NVIDIA) ← run in parallel
        └───────────┬──────────────┘
                    ▼  (both shard dirs passed; dataset arg selects one)
            n4 train_il
                    ▼
            n5 evaluate_il_policy   → MLflow: imitation-learning
                    ▼
            n6 train_offline_rl
                    ▼
            n7 evaluate_rl_policy   → MLflow: offline-rl
```
(Node numbering can vary; hover a node to see its task name.)

---

## Tips

- **Launch Plan versions**: every `pyflyte register` creates a new version. The UI
  defaults to the latest. If you need an exact version, pick it from the version
  dropdown on the workflow page.
- **Inputs are immutable per execution**: to change a parameter, launch a new run.
- **Outputs are addressable**: any node's output `s3://…` URI can be fed as input to
  another workflow — this is how you chain `wf_data_processing` → `wf_train_il` manually.
- **Secrets never appear in the UI**: the HF token is injected from the `hf-token`
  Kubernetes Secret and is not a workflow input, so it will not show up in any
  Inputs panel.
- **Archived clutter**: the default Flyte sample projects (`flytesnacks`, etc.) were
  archived so only `auto-e2e` is visible.

---

## Quick reference: which workflow for which goal

| Goal | Workflow | Key inputs |
|------|----------|------------|
| Full run, one command | `wf_full_pipeline` | `dataset`, hyperparams |
| Download raw data only | `wf_data_ingest` | `dataset`, `episodes` |
| Preprocess raw → shards | `wf_data_processing` | `raw_data` URI, optional `reasoning_labels` |
| Generate reasoning labels (teacher, cached) | `wf_generate_reasoning_labels` | `raw_data` URI, `teacher` |
| Raw → ready-to-train dataset | `wf_create_dataset` | `dataset`, `episodes`, `reasoning_teacher` |
| Sharded build → v2.1 publish → overlays | `wf_create_publish_and_precompute_overlays` | ops-only CodeBuild launch, `model_version` |
| Publish existing shards → overlays | `wf_publish_and_precompute_overlays` | `shards`, immutable model/runtime identities |
| Publish existing shards only | `wf_publish_dataset_snapshot` | `shards`, `published_dataset`, `dataset_version` |
| Precompute an already identified snapshot | `wf_precompute_overlays` | `shards`, model version, dataset manifest digest |
| Export an overlay shard as MP4 | `wf_export_trajectory_report` | matching immutable `shard` and `overlay` URIs |
| Train IL from existing shards | `wf_train_il` | `shards` list, `dataset` |
| Refine with Offline RL | `wf_train_offline_rl` | `pretrained`, `il_metadata`, `shards` |
| See metrics | (MLflow, not Flyte) | experiment `imitation-learning` / `offline-rl` |
