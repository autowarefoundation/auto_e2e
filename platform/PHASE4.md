# AutoE2E Phase 4 Platform Design: Evaluation Pipeline

Status: DRAFT — ready for review.

## Goal

Every training checkpoint is automatically evaluated with open-loop metrics.
Models that pass the gate are promoted to MLflow Model Registry (alias: `staging`).

## Model Output Recap

AutoE2E predicts a **trajectory** of 64 timesteps × 2 signals at 10Hz (6.4s horizon):
- `acceleration_x` (m/s²) — longitudinal acceleration
- `curvature` (1/m) — path curvature (= yaw_rate / speed)

Ground truth (from egomotion): same 64 × 2 format derived from vehicle states.

## Metrics

### Primary: ADE / FDE (via position integration)

Since the model predicts acceleration + curvature (not position directly),
we **integrate** the predicted signals into an (x, y) trajectory and compare
against the integrated ground truth:

```
Given initial state [x₀, y₀, v₀, θ₀] from egomotion history:

For each timestep t (dt = 0.1s):
    v(t) = v(t-1) + acceleration(t) * dt
    θ(t) = θ(t-1) + curvature(t) * v(t) * dt
    x(t) = x(t-1) + v(t) * cos(θ(t)) * dt
    y(t) = y(t-1) + v(t) * sin(θ(t)) * dt
```

| Metric | Definition | Horizon |
|--------|-----------|---------|
| ADE@1s | Mean L2 position error over 0–1s (10 steps) | 1.0s |
| ADE@2s | Mean L2 position error over 0–2s (20 steps) | 2.0s |
| ADE@3s | Mean L2 position error over 0–3s (30 steps) | 3.0s |
| ADE@6.4s | Mean L2 position error over full horizon | 6.4s |
| FDE@6.4s | L2 position error at final step (t=6.4s) | 6.4s |

### Secondary: Comfort Metrics

| Metric | Definition | Threshold |
|--------|-----------|-----------|
| Jerk | d(acceleration)/dt (smoothness) | < 2.5 m/s³ |
| Lateral acceleration | curvature × v² | < 3.0 m/s² |
| Max deceleration | min(acceleration_x) | > -4.0 m/s² |

### Direct Signal Metrics (no integration)

| Metric | Definition |
|--------|-----------|
| Accel MAE | Mean absolute error of predicted acceleration |
| Curvature MAE | Mean absolute error of predicted curvature |

## Architecture

```
Checkpoint (S3)  +  Val shards (S3)
        │                    │
        ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Flyte Evaluation Pipeline                                      │
│                                                                 │
│  1. load_checkpoint   Download from S3, load model weights      │
│  2. run_inference     Forward pass on val shards (GPU)          │
│  3. compute_metrics   Integrate → ADE/FDE + comfort (CPU)       │
│  4. gate_check        Compare vs baseline thresholds            │
│  5. promote_model     Pass → MLflow alias "staging"             │
│                                                                 │
│  Compute: GPU node for inference (g6e), CPU for metrics         │
│  Container: eval image (model + metrics code)                   │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
MLflow Model Registry: None → staging → champion
```

## Detailed Design

### 1. Position Integration (accel + curvature → x, y)

```python
def integrate_trajectory(
    accel: np.ndarray,      # (T,) predicted acceleration_x
    curvature: np.ndarray,  # (T,) predicted curvature
    v0: float,              # initial speed from egomotion history
    theta0: float,          # initial heading from egomotion history
    dt: float = 0.1,
) -> np.ndarray:
    """Integrate acceleration + curvature into (x, y) positions.
    
    Returns: (T, 2) array of [x, y] positions relative to initial pose.
    """
    T = len(accel)
    positions = np.zeros((T, 2))
    v, theta = v0, theta0
    x, y = 0.0, 0.0
    
    for t in range(T):
        v = v + accel[t] * dt
        v = max(v, 0.0)  # no reverse
        theta = theta + curvature[t] * v * dt
        x = x + v * np.cos(theta) * dt
        y = y + v * np.sin(theta) * dt
        positions[t] = [x, y]
    
    return positions
```

### 2. Metric Computation

```python
def compute_open_loop_metrics(
    pred_accel: np.ndarray,    # (B, 64)
    pred_curv: np.ndarray,     # (B, 64)
    gt_accel: np.ndarray,      # (B, 64)
    gt_curv: np.ndarray,       # (B, 64)
    initial_speed: np.ndarray, # (B,) from last frame of egomotion history
    initial_heading: np.ndarray, # (B,)
) -> dict:
    """Compute all open-loop eval metrics."""
    B = pred_accel.shape[0]
    
    ade_1s, ade_2s, ade_3s, ade_full, fde = [], [], [], [], []
    
    for i in range(B):
        pred_xy = integrate_trajectory(pred_accel[i], pred_curv[i],
                                        initial_speed[i], initial_heading[i])
        gt_xy = integrate_trajectory(gt_accel[i], gt_curv[i],
                                      initial_speed[i], initial_heading[i])
        errors = np.linalg.norm(pred_xy - gt_xy, axis=1)  # (64,)
        
        ade_1s.append(errors[:10].mean())
        ade_2s.append(errors[:20].mean())
        ade_3s.append(errors[:30].mean())
        ade_full.append(errors.mean())
        fde.append(errors[-1])
    
    return {
        "ADE@1s": np.mean(ade_1s),
        "ADE@2s": np.mean(ade_2s),
        "ADE@3s": np.mean(ade_3s),
        "ADE@6.4s": np.mean(ade_full),
        "FDE@6.4s": np.mean(fde),
        "accel_mae": np.mean(np.abs(pred_accel - gt_accel)),
        "curvature_mae": np.mean(np.abs(pred_curv - gt_curv)),
    }
```

### 3. Gate Logic

```python
# Thresholds for promotion to staging (tuned after initial training)
GATE_THRESHOLDS = {
    "ADE@3s": 2.0,    # meters (must be below)
    "FDE@6.4s": 5.0,  # meters (must be below)
}

def gate_check(metrics: dict, thresholds: dict = GATE_THRESHOLDS) -> bool:
    """Returns True if metrics pass all gate thresholds."""
    for key, max_val in thresholds.items():
        if metrics.get(key, float("inf")) > max_val:
            return False
    return True
```

Thresholds are initial baselines — tightened as models improve. A model must
also beat the current `staging` model's metrics to be promoted (ratchet).

### 4. Flyte Evaluation Workflow

```python
@task(container_image=EVAL_IMAGE, requests=Resources(cpu="4", mem="16Gi", gpu="1"),
      labels={"kueue.x-k8s.io/queue-name": "gpu-queue"})
def run_eval(checkpoint_s3: str, val_shard_dir: str) -> dict:
    """Load checkpoint, inference on val set, compute metrics."""
    ...

@task(container_image=EVAL_IMAGE, requests=Resources(cpu="1", mem="1Gi"))
def promote_if_passed(metrics: dict, checkpoint_s3: str, run_id: str) -> bool:
    """Gate check + MLflow promotion."""
    ...

@workflow
def evaluate_checkpoint(
    checkpoint_s3: str,
    val_shard_dir: str = "s3://datasets/l2d/v1.0/shards/val-*.tar",
) -> bool:
    metrics = run_eval(checkpoint_s3=checkpoint_s3, val_shard_dir=val_shard_dir)
    return promote_if_passed(metrics=metrics, checkpoint_s3=checkpoint_s3)
```

### 5. Eval Container

**No separate eval image needed.** Training image already contains:
- PyTorch + model code (inference)
- Pillow (visualization)
- numpy (metric computation)

The position integration logic is being contributed in PR #74
(trajectory rendering), which implements the same
`accel + curvature → velocity → (x, y)` integration used for eval metrics.
Once merged, the integration function lives in `Model/` and is available
in the training container without any Dockerfile changes.

Eval workflow uses the same `auto-e2e/training:latest` image as PyTorchJobs.

### 6. Integration with Training Pipeline

Two trigger modes:
1. **Post-training (Flyte)**: training workflow calls `evaluate_checkpoint` as
   the final step after saving checkpoint to S3.
2. **Manual**: user runs eval workflow from Flyte UI with a checkpoint path.

### 7. MLflow Model Registry Promotion

```
None (initial) → staging (passes gate) → champion (manual promotion after review)
```

Uses MLflow **aliases** (not deprecated stages):
- `mlflow.register_model(model_uri, "auto_e2e")`
- `client.set_registered_model_alias("auto_e2e", "staging", version)`
- `champion` alias set manually after human review of staging model.

### 8. Implementation Plan

1. **Metrics module** (`Model/evaluation/metrics.py`): `integrate_trajectory`,
   `compute_open_loop_metrics`, `gate_check`. Reuse integration logic from
   PR #74 (trajectory rendering) — same math, different output format.
2. **Flyte eval workflow** (`platform/pipelines/evaluation/workflow.py`).
3. **Val shard generation**: split L2D shards into train/val (80/20 by episode).
4. **Wire into training workflow**: add eval step after training completes.
5. **End-to-end verify**: train → checkpoint → eval → gate → MLflow promote.

No separate eval Dockerfile — training image has everything needed.

### 9. Cost

- GPU eval: ~30s for 100 val samples (single forward pass, no backprop).
- Runs on the same warm g6e node (Kueue queued after training job completes).
- Metrics computation is CPU-only (numpy), negligible cost.

### 10. Open Questions

1. **Threshold values**: Initial ADE@3s < 2.0m, FDE@6.4s < 5.0m are loose
   estimates. First real training will establish baselines.
2. **Val split strategy**: Random 80/20 by episode, or hold-out specific
   episodes (e.g., different weather/route)?
3. **Comfort metrics**: Include in gate or report-only? Starting with report-only.
4. **Initial state for integration**: Use last known speed + heading from
   egomotion history `signals[-1]` = [speed, accel, yaw_rate, curvature].
   `v0 = speed[-1]`, `θ0 = cumsum(yaw_rate * dt)[-1]`.
