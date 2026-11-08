# Design: AutoBEV as the AutoE2E Camera Encoder

## Document Metadata

| Field | Value |
|---|---|
| Status | In implementation |
| Owner | riita10069 |
| Created | 2026-11-07 |
| Branch | `feat/autobev-integration` (includes PR #235, `feat/lateral-recovery-kl`) |
| Builds on | `experiment/autobev/` (BEVFormer v2 T8, DINOv2 ViT-B/14), `Design/single_flyte_pipeline.md` |

## 1. Goal

Build a camera-only AutoE2E whose BEV encoder is AutoBEV, trained in two stages
inside `wf_autoe2e`:

1. Stage 1 (`training_scope=bev_only`, `camera_architecture=autobev`): train
   AutoBEV on every published BEV dataset (nuPlan, nuScenes, Argoverse 2, ONCE,
   SimBEV) with BEV segmentation, LiDAR depth, image boxes, image-space map
   segmentation and BEV object centers.
2. Stage 2 (`training_scope=multitask`): start from the stage 1 checkpoint,
   train the navigation encoder, map fusion and planner on frozen AutoBEV, then
   unfreeze AutoBEV gradually and train end to end with the stage 1 losses kept
   as auxiliary losses.

LiDAR is a training teacher only (depth targets, label visibility). Inference
reads the six cameras, the navigation raster and the ego state.

The aim is the best driving model, not a per-component ablation. Methods that
are likely to help go in together; there are no "one method removed" runs.

## 2. Geometry contract

One contract, `AUTOBEV_GEOMETRY` (`Model/model_components/autobev/geometry.py`),
decides every BEV tensor of the new architecture:

| Field | Value |
|---|---|
| geometry_id | `autobev-220x160-0p5m-v2` |
| BEV grid | 220 rows (x, front to rear) by 160 columns (y, left to right), 0.5 m |
| x / y range | -30 .. 80 m / -40 .. 40 m |
| pillar z range | -0.5 .. 3.5 m, anchors at 0, 1, 2, 3 m |
| ego origin | on the road surface (rear axle for nuPlan); datasets whose origin is elsewhere are shifted at pack time |
| image | 1344 x 768 per camera, six cameras, T8 history at 0.5 s |

The contract is written into every AutoBEV checkpoint and every pack manifest
that carries AutoBEV members, and it is compared field by field when a
checkpoint is loaded and when a dataset is planned. A checkpoint trained with
the old z range (-5 .. 3 m) has its anchors 3 to 4 m off and is refused instead
of being read.

The navigation raster keeps its own geometry (`autoe2e-bev-450x300-0p4m-v1`,
x -60 .. 120 m, y -60 .. 60 m). It is not pooled to the BEV shape. Every AutoBEV
cell centre is mapped to its world position, transformed by the pose used to
crop the map (Section 5), and sampled from the 0.4 m raster
(`Model/navigation/resample.py`). Binary map channels are sampled bilinearly,
direction channels are sampled where valid and rotated into the new frame, and
validity channels use nearest sampling. The 0.4 m raster reaches 20 m beyond
the AutoBEV grid at the sides and 30 to 40 m ahead and behind, so simulated crop
errors are kept within 8 m and 5 degrees.

## 3. Data

### 3.1 nuPlan repack

The current nuPlan pack squeezes 1920 x 1080 into 512 x 512 and carries a
1024 x 1024 front tile, so it cannot feed AutoBEV. A new pack contract
(`nuplan` pack version bump) keeps the raw snapshot and changes the pack:

- six cameras undistorted, scaled to cover 1344 x 768 with the aspect ratio
  kept, and centre-cropped; projections follow the same pixel transform; no
  front companion;
- T8 history stored in each sample (`hist_{h}_cam_{v}.jpg`) at the same
  size, without a shared frame pool;
- BEV labels on `AUTOBEV_GEOMETRY` with the AutoBEV v2 rules (boxes no LiDAR
  return saw are unknown, not negative);
- stride-8 LiDAR depth (`depth.npz`) and 3D boxes (`boxes.npz`);
- the navigation raster, route and trajectory members as before;
- Pittsburgh logs are held out of training and published as the geographic
  validation set, as in the AutoBEV experiment.

### 3.2 Other BEV datasets

nuScenes, Argoverse 2, ONCE (with ONCE-3DLanes) and SimBEV become published
datasets with raw snapshots from `wf_acquire_raw_snapshot`, adapters in
`autoe2e_adapters.py` and packers under `Model/data_parsing/<dataset>/`, ported
from the experiment packers. They carry only the AutoBEV members (images,
projections, BEV labels, depth, boxes) and have no trajectory or navigation
members. Classes a dataset cannot provide are invalid, not negative.

As built, one table-driven adapter (`read_autobev`, `AUTOBEV_SOURCES`) serves
the four datasets, and their packs read single members of the snapshot units by
range (`SnapshotMembers`) instead of extracting whole unit tars, because a
nuScenes, ONCE or SimBEV partition needs a few files of 5 to 10 GB units. Every
partition of those three reads all units and packs every N-th scene or sequence
(N fixed per packer); an Argoverse 2 partition is one log. Camera views follow
the experiment packers: nuScenes and SimBEV map one to one, Argoverse 2 drops
one rear ring camera per log by a hash of the log id, and ONCE uses cam01,
cam06, cam07, cam05, cam09, cam08 and drops the telephoto cam03. ONCE's origin
is moved 1.78 m down to the road when packing, not when loading.

### 3.3 Splits and loading

Split groups are hashed into 100 stable buckets. nuPlan Pittsburgh groups
carry a `heldout-` prefix and are always validation; 2 % of the other groups
(`CALIBRATION_FRACTION`) form the calibration split, which sets the per-class
BEV thresholds and never trains; everything else trains. Each training rank
copies every `world`-th shard of each dataset into one local directory per
dataset (`training/autobev_data.py`). Images travel as uint8 and are
normalized on the GPU, because a float T8 sample is about 0.6 GB.

### 3.4 Mixing

Stage 1 mixes all five datasets in proportion to their sample counts; SimBEV is
used only in the first half of the steps. Stage 2 draws 15 % of its samples from
the four non-nuPlan datasets plus the nuPlan BEV-only samples and applies only
the BEV segmentation loss to them, so the encoder keeps what it learned on the
other rigs.

## 4. Stage 2 schedule

| Steps | AutoBEV | Planner side |
|---|---|---|
| first 5 % | frozen (learning rate 0, optimizer state kept empty) | learning rate x1 |
| after | DINOv2 blocks 0-5 and patch embedding frozen; blocks 6-11 at x0.05 with a 0.8 decay per block below 11; BEV encoder (FPN adapters, SCA, TSA, temporal fusion, perspective and centre heads) at x0.2 | x1 |

The freeze is a learning-rate multiplier, not `requires_grad`, so the DDP graph
stays static. When the multiplier becomes non-zero the AutoBEV optimizer state is
reset and a 1 % warm-up applies to those groups.

The stage 1 losses stay as auxiliary losses (BEV segmentation, depth, centres)
on the nuPlan samples, with a weight that falls linearly from 1.0 to 0.3 over
stage 2. The seven history frames go through AutoBEV without gradient.

## 5. Localization and map fusion

Dataset poses are post-processed and too accurate: a map cropped at them makes
dead reckoning and the map fusion trivial. Training therefore hides the true
pose and crops the map at the pose an onboard estimator would report.

### 5.1 Simulated sensors

Over the 6.4 s ego history (64 rows at 10 Hz, integrated from the logged speed
and yaw rate) each sample draws one GNSS regime:

| Regime | Share | Model |
|---|---|---|
| nominal | 60 % | white noise of 0.3 to 1.0 m plus a slowly varying bias (Gauss-Markov, 30 s) |
| degraded | 25 % | nominal plus a 3 to 10 m jump lasting 2 to 30 s, as from multipath under an overpass |
| outage | 15 % | no fixes; dead reckoning only, with the drift accumulated since the last good fix (5 to 30 s before now) |

Dead reckoning uses the logged speed with a 1 to 2 % scale error and the logged
yaw rate with a 0.1 to 0.5 deg/s bias. nuPlan has no raw GNSS (its database has
no GNSS table, only `ego_pose`), so both are synthesized from the logged motion.

### 5.2 Three pose sources

The user asked for three ways to combine dead reckoning, GNSS and AutoBEV. No
published end-to-end planner compares them, so the design follows the closest
work and uses all three together rather than as alternatives:

1. EKF only: a 3-state (x, y, yaw) EKF predicts with dead reckoning and updates
   with GNSS behind a chi-square gate. This is also the fallback.
2. EKF with map matching: the AutoBEV prediction (lane_boundary for the lateral
   offset, crosswalk and stop_line for the longitudinal offset) is correlated
   with the map raster over a translation and yaw search around the EKF pose on
   the GPU, in the spirit of semantic map matching (YabLoc) and BEV-to-map
   registration (OrienterNet, Sarlin et al., CVPR 2023, arXiv:2304.02009;
   MapLocNet, arXiv:2407.08561). The peak, with an uncertainty from its
   sharpness per axis, is one more EKF observation. Axes without a sharp peak
   are left to the EKF; no sharp peak at all falls back to source 1.
3. Learned alignment: the map raster cropped at the estimated pose goes into
   MapBEVFusion as is, and the deformable map fusion learns to sample it at
   the right place. A pose-error head on the fused BEV regresses the remaining
   offset (dx, dy, dyaw) between the cropped map and the true ego frame, which
   gives the alignment a direct signal (as BEV-Locator, arXiv:2211.14927,
   regresses the pose from BEV and map queries).

During training each sample uses source 1 or source 2 with equal probability,
and source 3 is always on. At inference source 2 runs, with source 1 as its
fallback, and the network corrects what remains.

The map and route rasters are resampled at the estimated pose. Trajectory
targets, BEV labels and the camera geometry stay in the true ego frame. The
lateral recovery shift of PR #235 moves the ego and the world together and is
applied first; the pose error moves only the map, so the two compose.

## 6. Distortion mix

Each training sample gets exactly one of:

| Choice | Share | Effect |
|---|---|---|
| lateral recovery, projection | 15 % | PR #235: shift and yaw of the ego frame through the projections, recovery target |
| lateral recovery, image_warp | 15 % | PR #235: the same shift through warped images |
| camera_augment | 15 % | zoom 0.9-1.15 and principal-point shift per camera, applied to images, projections and image-space targets |
| photometric | 15 % | brightness, contrast and saturation per camera, shared across the eight frames |
| none | 40 % | |

The feature-consistency KL of PR #235 compares the distorted sample with a clean
teacher pass under `no_grad`. The previous 5-epoch nuPlan schedule becomes 15
epochs, so the clean data is seen three times with different views.

## 7. Pipeline

- `TrainingSpec` gains `camera_architecture`, the stage 2 schedule fields, the
  distortion and localization shares, and `bev_mix` (published datasets mixed
  as BEV-only samples). Defaults keep today's results.
- Checkpoints are written about every 30 minutes of training
  (`checkpoint_interval_steps`, set from the measured step time). Ray Train
  stores them under the run prefix; resume reads the last one. Run files go
  through `pyarrow.fs`, so a local directory and S3 take the same path.
- A full-size step on one RTX PRO 6000 (g7e), timed on the third step, takes
  4.0 s per sample and 16 GB in stage 1 (batch 1) and 6.3 s per sample and
  29 GB in stage 2 (batch 2, with the clean teacher pass).
- Lightweight validation runs asynchronously: a one-GPU task starts next to the
  training task, watches the run's checkpoint prefix, and evaluates each new
  checkpoint on a fixed held-out subset (ADE/FDE on nuPlan validation, BEV IoU
  on the Pittsburgh split). Results are JSON next to the checkpoint and are not
  registered in MLflow. They pick the checkpoints that go to the formal
  evaluation and tell how many epochs to train. When the Pittsburgh drivable
  IoU falls by more than 10 % relative to the stage 2 start, the training task
  halves the AutoBEV learning-rate multipliers (at most three times).
- Formal evaluation runs after training on the whole held-out Pittsburgh split
  and is the only result recorded on the MLflow model version, together with
  how the checkpoint was chosen and its calibrated BEV thresholds. The KITScenes
  benchmark returns once KITScenes is repacked to the AutoBEV contract.

This adds pipeline inputs and a new dataset layer content, keeps one workflow,
and keeps every dataset and checkpoint addressed by digest.
