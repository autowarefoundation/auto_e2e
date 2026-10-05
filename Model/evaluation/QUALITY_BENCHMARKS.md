# World-Model Quality Benchmarks

Speed benchmarks live in [`../speed_benchmark/`](../speed_benchmark/). This folder
documents **quality** metrics for the policy / World Model stack:

| Metric family | What it measures |
|---------------|------------------|
| JEPA reconstruction | Per-horizon L1 / L2 / cosine of predicted vs frozen-target feature maps |
| Null-relative improvement | How much better than predicting zeros |
| Reactive vs Combined impact | Trajectory L1/L2 delta when the World Model is enabled |
| Open-loop ADE/FDE pair | Reactive and Combined ADE@3s / FDE@3s on the same GT (when labels exist) |

## Measured result (trained Combined, 12 steps)

Source: `evaluation/results/world_model_quality_trained.json` (CPU, mock backbone, seed 0). Combined IL+JEPA loss **0.462 → 0.389**. This is **not** a KITScenes checkpoint; it is a trained (not random-init) Combined run so ADE/FDE are defined. Re-run on packed shards with `--shard-dir` / `--checkpoint`.

| | Reactive | Combined | Δ (C−R) |
|--|----------|----------|---------|
| ADE@3s | 4.160 | **3.971** | −0.189 |
| FDE@3s | 11.000 | **10.428** | −0.572 |
| JEPA L1 / cosine | — | 0.0437 / 0.157 | — |

JEPA relative improvement vs a zero predictor is **0** on random frames (model L1 0.044 > null 0.018). That is expected without real video; the ADE pair is the number that answers “does Combined move the plan.”

## Quick start

```bash
cd Model

# Train Combined a few steps, then JEPA + ADE/FDE (default)
python evaluation/world_model_quality_benchmark.py --trained --train-steps 12

# Packed KITScenes/L2D partition + optional checkpoint
python evaluation/world_model_quality_benchmark.py \
  --shard-dir /path/to/partition --checkpoint ckpt.pt --train-steps 0
```

## Library API

```python
from evaluation.world_model_quality import (
    jepa_reconstruction_metrics,
    train_world_model_quality,
    world_model_trajectory_impact,
    open_loop_pair_metrics,
)
```
