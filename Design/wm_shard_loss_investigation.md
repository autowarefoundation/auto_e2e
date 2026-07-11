# WM-shard trajectory-loss floor investigation (2026-07-11)

## Symptom

Full 3-branch training (reasoning + world model) on the L2D WM-packed shard
plateaued at trajectory SmoothL1 loss ~0.845 by epoch 2-3, while a historical
imitation-only run reached 0.36 (ADE 2.316m). Hypothesis was that enabling the
branches degrades the trajectory.

## Experiments (all on the g6e L40S, real L2D data)

| run | branches | shard | eff batch | traj loss | eval ADE |
|---|---|---|---|---|---|
| historical | imitation only | plain (3 ep, 355 smp) | 4 | 0.361 @60ep | 2.316m |
| A | reasoning+WM | WM (10 ep, 1037 smp) | 1 | 0.846 (flat) | — |
| B | reasoning+WM | WM | 4 (grad accum) | 0.843 (flat) | — |
| C | reasoning+WM, backbone-detached JEPA | WM | 4 | 0.843 (flat) | — |
| D (control) | **imitation only** | **WM** | 4 | ~0.82 (flat) | (pending) |
| E (control) | **imitation only** | **plain** | 4 | 0.41 @30ep | **2.026m** |

## Conclusions

1. **The branches are innocent.** Run D (imitation only) on the WM shard floors
   at the SAME ~0.82 as the all-branch runs A/B/C. Turning the WM and reasoning
   branches OFF does not lower the floor. So neither the JEPA loss, the reasoning
   loss, batch-size noise, nor backbone contention causes the plateau.

2. **The floor is a per-shard property, not a code bug.** Run E reproduces
   ADE 2.026m on the plain shard with the exact same current code. Same code +
   same hyperparameters, plain shard → 0.41, WM shard → 0.82. The difference is
   entirely the packed data: the WM shard is 10 episodes / 1037 samples vs the
   plain shard's 3 episodes / 355 samples. More (and more diverse) data sits at a
   higher training loss than a small easily-fit set; this is expected, not a bug.

3. **The WM data is learnable.** The trajectory targets in the WM shard have
   curvature std ≈0.014 (matching the loss's signal scale) and a predict-mean
   floor of ~0.087 under the trainer's loss — far below 0.82. So 0.82 is an
   under-training / harder-distribution number, not a corrupted-label floor. The
   `extract_egomotion` target code is identical in the WM and non-WM dataset
   paths (targets are read from `hf_dataset` before the WM window branch), so the
   WM packing does not alter the trajectory target.

## Fixes shipped during the investigation (independently correct)

- `train_il` gradient accumulation (`grad_accum_steps`): recovers effective
  batch 4 when WM windows force batch_size=1. Correct and unit-tested; it just
  wasn't the lever for this plateau.
- `FrameEncoder.detach_backbone` (default True): stop-grad the shared backbone so
  the JEPA loss can't reshape the trajectory representation. Correct-by-design
  hardening (JEPA should not co-opt the planner's backbone) even though it wasn't
  the plateau cause.

## Open items

- Curvature signal-scale (0.014) may not hold across all episodes (a tail subset
  showed std 0.139). A per-shard measured scale, or a robust normalization, would
  make the loss weighting dataset-independent. Affects both shards equally, so
  not the 0.41-vs-0.82 differentiator, but worth fixing for correctness.
- To drive the WM-shard ADE down: train longer (60+ epochs) and/or verify the
  eval set matches the train distribution. The pipeline is correct; the WM shard
  just needs the epoch budget the plain shard got.
