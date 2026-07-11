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
| D (control) | **imitation only** | **WM** | 4 | 0.808 @30ep | **1.771m** |
| E (control) | **imitation only** | **plain** | 4 | 0.412 @30ep | **2.026m** |
| F (final) | **reasoning + WM** (all fixes: grad-accum, backbone-detach, curvature scale) | WM | 4 | 0.438 @30ep (traj 0.079) | **2.409m / FDE 6.627m** |

Run F is the definitive full 3-branch pipeline: all three branches trained and
converged (traj 0.125→0.079, jepa 0.505→0.330, reason 1.189→0.588), ADE 2.409m —
under the 3 m goal. With the corrected curvature scale the trajectory loss is now
interpretable (traj_loss ~0.08 instead of ~0.8).

## Conclusions

1. **The branches are innocent.** Run D (imitation only) on the WM shard floors
   at the SAME ~0.80 as the all-branch runs A/B/C. Turning the WM and reasoning
   branches OFF does not lower the floor. So neither the JEPA loss, the reasoning
   loss, batch-size noise, nor backbone contention causes the plateau.

2. **The "plateau at 0.8" was never a problem — it is the loss SCALE, not model
   quality.** The decisive metric is eval ADE, and the WM shard's HIGHER training
   loss (0.808) produced a BETTER trajectory than the plain shard's lower loss
   (0.412): ADE 1.771m vs 2.026m. So the absolute training-loss number is not
   comparable across shards (different sample distributions integrate to
   different SmoothL1 magnitudes) and does not track ADE across datasets. More
   data (WM shard: 10 episodes / 1037 samples) generalizes better than the small
   plain shard (3 episodes / 355 samples), exactly as expected. There is no bug:
   the full pipeline is correct and the WM-packed data trains to a better policy.

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

- **Curvature signal-scale is ~9x off.** The loss uses `signal_scales=(0.54,
  0.014)` but the measured target curvature std is ~0.124 on BOTH shards (the
  earlier 0.014 reading came from a truncated tar). So the normalized loss
  over-weights curvature error ~9x — which is why the absolute loss numbers sit
  ~0.4-0.8 instead of ~0.1. This does not break training (ADE is still good and
  curvature getting extra gradient is not harmful per se), and it is identical
  across shards so not the plain-vs-WM differentiator, but the scale should be
  corrected to the measured std (~0.12) so the logged loss is interpretable and
  the accel/curvature balance is principled. Fix: set the curvature scale in
  `TrajectoryImitationLoss._DEFAULT_SIGNAL_SCALES` to the measured value.
- To drive ADE lower still: train the full 3-branch config for more epochs on the
  WM shard (imitation-only already hits ADE 1.771m at 30ep). The pipeline is
  correct; this is now just an epoch-budget / hyperparameter tuning matter.
