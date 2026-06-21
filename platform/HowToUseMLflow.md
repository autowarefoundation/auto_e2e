# How to Use MLflow — AutoE2E Experiment Tracking Guide

A practical, screen-by-screen guide to the MLflow UI for the AutoE2E training
platform. Organized by **what you want to do**.

- **MLflow UI**: https://d33520viyb0smg.cloudfront.net/ (no login required)
- Metrics are written **only by the evaluation tasks** (`evaluate_il_policy`,
  `evaluate_rl_policy`), so every run already contains the full train→eval picture.

> Flyte tells you **whether a run executed**; MLflow tells you **how good the model is**.
> For launching pipelines see `HowToUseFlyte.md`.

---

## 0. Layout overview

When you open MLflow you land on the **Experiments** page. Left sidebar lists
experiments; the main panel shows the runs of the selected experiment.

The three experiments that matter:

| Experiment | Written by | Contains |
|------------|-----------|----------|
| **`imitation-learning`** | `evaluate_il_policy` | IL-trained policies + their eval metrics |
| **`offline-rl`** | `evaluate_rl_policy` | IQL-refined policies + their eval metrics |
| `Default` | — | empty / unused |

There is also a **Models** tab (top nav) for the Model Registry
(`auto-e2e-driving-policy`).

---

## Core concepts (30-second version)

| Term | What it is |
|------|------------|
| **Experiment** | A named bucket of runs (e.g. `imitation-learning`) |
| **Run** | One evaluated model: its params, metrics, artifacts |
| **Param** | A fixed input value (backbone, lr, dataset…) — filterable |
| **Metric** | A measured value, possibly per-step (train loss, ADE, FDE) |
| **Artifact** | A file saved with the run (checkpoint, `config.yaml`) |
| **Registered Model** | A versioned model in the Model Registry |

---

## What every run records

Each evaluation run logs a consistent schema, so runs are directly comparable:

**Params** (single values, filterable / sortable):
```
data/dataset              e.g. yaak-ai/L2D
model/backbone            e.g. swin_v2_tiny
model/fusion_mode         e.g. concat
train/epochs, train/batch_size, train/lr, train/weight_decay, train/amp
train/final_loss
rl/method, rl/tau, rl/beta, rl/epochs            (offline-rl runs only)
ctx/train_execution_id    Flyte execution that produced the checkpoint
ctx/train_docker_image    exact training image
ctx/eval_execution_id, ctx/eval_docker_image
```

**Metrics**:
```
train/loss        per-epoch training loss curve (step = epoch)
eval/ade          Average Displacement Error (meters, lower is better)
eval/fde          Final Displacement Error (meters, lower is better)
eval/gate_pass    1.0 if ade<2.0 AND fde<4.0 else 0.0
rl/q_loss, rl/v_loss, rl/policy_loss             (offline-rl runs only)
```

**Artifacts**:
```
config.yaml       full nested config (data+model+training+context) for exact reproduction
model/            the checkpoint (.pt), also registered in the Model Registry
```

---

## Use case A — "Which model is the best?"

**You are**: anyone choosing a model to promote.

1. Open the **`imitation-learning`** experiment.
2. In the run table, click the **`eval/ade`** column header to **sort ascending**
   (lowest displacement error = best).
3. Add/show the columns you care about (gear/columns button): `model/backbone`,
   `model/fusion_mode`, `data/dataset`, `eval/ade`, `eval/fde`, `eval/gate_pass`.
4. The top row is your best IL model. Repeat in **`offline-rl`** to see if RL
   refinement improved it.

> Lower `eval/ade` and `eval/fde` are better. `eval/gate_pass = 1` means it cleared
> the quality gate (ade<2.0m, fde<4.0m).

---

## Use case B — "Compare a few candidate runs side by side"

**You are**: comparing backbones, fusion modes, or hyperparameters.

1. In the experiment run table, tick the **checkboxes** of 2–5 runs.
2. Click **Compare**.
3. The compare view shows:
   - A **parameter table** (differing params highlighted) — instantly see what changed.
   - **Metric charts** — `train/loss` curves overlaid, plus bar charts for `eval/ade` / `eval/fde`.
4. Use this to answer "did `bev` fusion beat `concat`?" or "did a lower `lr` help?".

---

## Use case C — "Filter runs by configuration"

**You are**: looking for all runs of a specific setup.

1. In the run table, use the **search bar** with MLflow's filter syntax:
   ```
   params.`model/backbone` = "swin_v2_tiny"
   params.`data/dataset` = "yaak-ai/L2D" and metrics.`eval/ade` < 2.0
   params.`model/fusion_mode` = "bev"
   ```
   (Backtick the keys that contain `/`.)
2. Sort and column-pick as in Use case A.

---

## Use case D — "Inspect one run in depth"

**You are**: auditing or reproducing a specific result.

1. Click the run name to open the **run detail page**.
2. **Parameters** section: every input (data, model, training, rl, context).
3. **Metrics** section: click `train/loss` to see the **per-epoch curve**; `eval/*`
   show final values.
4. **Artifacts** section:
   - `config.yaml` — download to reproduce the exact run.
   - `model/` — the checkpoint.
5. **Provenance**: the `ctx/train_execution_id` param links back to the Flyte
   execution that trained this model (paste it into the Flyte Executions search).

---

## Use case E — "Did Offline RL actually improve the model?"

**You are**: validating the RL refinement step.

1. Note the IL run's `eval/ade` in the **`imitation-learning`** experiment.
2. Open the matching run in **`offline-rl`** (same backbone/fusion/dataset; the
   `base/*` or `ctx/*` params link them).
3. Compare `eval/ade` / `eval/fde`: a lower value in the offline-rl run means IQL
   refinement helped. (With only a few epochs the absolute numbers are high; what
   matters is IL-vs-RL relative improvement.)

---

## Use case F — "Manage models for deployment" (Model Registry)

**You are**: promoting a model toward production.

1. Top nav → **Models**.
2. Open **`auto-e2e-driving-policy`** — every evaluated checkpoint is registered as
   a new **version** here.
3. For a version you can:
   - View its **source run** (links back to the experiment run with all metrics).
   - Add a **description** / tags.
   - (When stages are enabled) transition it through Staging → Production.
4. To fetch a specific model programmatically:
   ```python
   import mlflow
   mlflow.set_tracking_uri("http://mlflow.mlflow.svc.cluster.local:5000")  # in-cluster
   model_uri = "models:/auto-e2e-driving-policy/<version>"
   local = mlflow.artifacts.download_artifacts(model_uri)
   ```

---

## Use case G — "Track training progress live"

**You are**: watching a run that is still training.

- MLflow runs are created by the **evaluation** task, which runs **after** training,
  so live per-epoch curves appear once evaluation starts logging.
- To watch training epoch-by-epoch **as it happens**, use the **Flyte** task logs
  (see `HowToUseFlyte.md`, Use case F). MLflow is for the consolidated post-run view.

---

## Reading the metrics

| Metric | Meaning | Good value |
|--------|---------|-----------|
| `eval/ade` | mean (x,y) error over the 6.4s horizon (m) | lower; <2.0 passes gate |
| `eval/fde` | error at the final predicted point (m) | lower; <4.0 passes gate |
| `eval/gate_pass` | 1.0 if both gates pass | 1.0 |
| `train/loss` | smooth-L1 trajectory loss per epoch | decreasing & flattening |
| `rl/q_loss`,`rl/v_loss`,`rl/policy_loss` | IQL component losses | decreasing |

> Trajectories are produced by integrating predicted `(accel, curvature)` into
> (x,y) positions and comparing to ground truth (see `evaluation/metrics.py`).

---

## Tips

- **One run = full story**: because only the eval task logs, each run already bundles
  the training params, loss curve, eval metrics, config, and checkpoint. No need to
  cross-reference a separate "training run".
- **Backtick slashed keys** in the search bar: ``params.`model/backbone` = "…"``.
- **Compare is your friend** for ablations — select runs, hit Compare.
- **Provenance is built in**: `ctx/*` params tie each run to its Flyte execution and
  Docker image, so any result is fully traceable and reproducible.
- **Stale experiments were removed**: only `Default`, `imitation-learning`, and
  `offline-rl` remain; old `auto-e2e/*` and `test-*` experiments were deleted.

---

## Quick reference: goal → where to look

| Goal | Where |
|------|-------|
| Pick the best model | experiment → sort by `eval/ade` |
| Compare configs | select runs → **Compare** |
| Find runs of a setup | search bar filter on `params.*` |
| Reproduce a run | run detail → `config.yaml` artifact |
| Verify RL improvement | compare IL run vs offline-rl run `eval/ade` |
| Deploy a model | **Models** → `auto-e2e-driving-policy` |
| Watch live training | (Flyte logs, not MLflow) |
