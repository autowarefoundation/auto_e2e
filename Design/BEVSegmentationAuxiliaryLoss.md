# Design: Reactive Multi-Stage Training with BEV Segmentation

<!-- markdownlint-configure-file {"MD013": {"tables": false}} -->

## Document Metadata

| Field | Value |
| --- | --- |
| Status | Baseline implemented; production-data validation pending |
| Owner | riita10069 |
| Created | 2026-08-08 |
| Last revised | 2026-08-08 |
| Stage A | nuPlan full Reactive multi-task training |
| Stage B | L2D continuation without BEV segmentation loss |
| Stage C | KITScenes benchmark only |
| Model scope | Reactive branch only |
| Related issue | [#17](https://github.com/autowarefoundation/auto_e2e/issues/17) |
| Implementation status | Code and synthetic smoke complete |

## 1. Executive Summary

This design defines one sequential training and evaluation program:

```text
Stage A: nuPlan
  camera + map + route + ego history
    -> Reactive model
    -> trajectory imitation
    -> BEV segmentation auxiliary loss
    -> route reconstruction auxiliary loss

Stage B: L2D
  camera + canonical OSM map + route waypoints + ego history
    -> continue training the Stage A checkpoint
    -> trajectory imitation
    -> route reconstruction auxiliary loss
    -> no BEV segmentation loss

Stage C: KITScenes
  frozen checkpoints
    -> benchmark only
    -> no training, fine-tuning, threshold selection, or early stopping
```

The World Model and Reasoning branches are disabled. The first experiment uses
only the 10 Hz Reactive branch with a deterministic GRU planner.

The primary objective is intentionally simple. The model still predicts the
repository's acceleration and curvature sequence for runtime and evaluator
compatibility, but the only imitation loss is uniform masked Smooth L1 between
the integrated predicted XY trajectory and the recorded future XY trajectory.
The initial experiment does not use control imitation, temporal decay, route
consistency, rollout-aligned loss, endpoint loss, collision loss, or comfort
loss.

The two auxiliary objectives have separate responsibilities:

- BEV segmentation supervises the camera-only `image_bev` representation. It
  is available only in Stage A because nuPlan provides the required map, lidar,
  boxes, calibration, and poses.
- Route reconstruction supervises the navigation representation after its
  fusion gate. It is available in both Stage A and Stage B. It asks whether
  route information survives the navigation encoder and can reach the planner;
  it does not replace trajectory imitation.

All datasets use the native AutoE2E BEV geometry:

```text
height x width: 450 x 300
resolution:     0.4 m/px
X:              [-60, 120] m, forward positive
Y:              [-60,  60] m, left positive
```

Map and route inputs use one versioned semantic contract across datasets.
Dataset-native vector maps are preferred. If a dataset has geodetic pose but
no compatible map, a pinned regional OpenStreetMap snapshot is downloaded
once during offline preprocessing and converted locally. Training and
evaluation never call a public map service.

KITScenes is not part of optimizer data. Results are reported on one immutable
KITScenes benchmark manifest and compared with VAD and UniAD only when input
modalities, navigation source, checkpoint provenance, sample set, and metric
implementation are explicitly stated. Published values from a different
KITScenes track or from nuScenes are reference values, not direct comparisons.

### 1.1 Implementation status and evidence

The repository now contains the baseline implementation described here:

| Area | Status | Evidence |
| --- | --- | --- |
| Common geometry and packed targets | Implemented | `navigation/geometry.py`, `reactive_training_artifacts.py` |
| nuPlan raw scenario packing | Implemented, synthetic scenario smoke complete | `data_parsing/nuplan/packing.py` |
| L2D pinned OSM map and route targets | Implemented, deterministic encoder smoke complete | `data_parsing/l2d/navigation.py`, `osm_graph_builder.py` |
| BEV and route heads and losses | Implemented | `auxiliary_heads.py`, `reactive_multitask.py` |
| Simple integrated-XY imitation | Implemented | `trajectory_xy_loss.py` |
| Stage A to Stage B workflow and lineage | Implemented | `reactive_stage_runner.py`, `workflows.py` |
| Frozen 2 x 2 retention matrix | Implemented | `evaluate_reactive_transfer_matrix_models` |
| KITScenes checkpoint benchmark | Implemented for the repository protocol | `evaluate_kitscenes_benchmark_checkpoint` |
| Semantic occupancy Dashboard | Implemented and browser-smoked | `semantic-occupancy-view.tsx` |
| Same-manifest UniAD and VAD adapters | Pending | External model and input-contract work |
| Real regional OSM and L2D raster audit | Pending | Requires the selected production snapshot |
| City and day/night metric strata | Pending | Requires audited timezone and location labels |
| Full nuPlan to L2D training run | Pending | Requires production datasets and GPU budget |

The synthetic smoke covers Stage A optimization, a weights-only Stage B
transition with a fresh optimizer, the four retention cells, checkpoint
content hashes, and semantic artifact encode/decode. It does not substitute
for a full-data target audit or a reported training result.

## 2. Locked Decisions

The following choices are normative for the first implementation:

1. Train sequentially on nuPlan and then L2D.
2. Do not train on KITScenes.
3. Use `450 x 300` BEV queries and `0.4 m/px` for every stage.
4. Enable only the Reactive branch.
5. Set `enable_world_model=false` and `enable_reasoning=false`.
6. Use a deterministic GRU trajectory planner, not flow matching.
7. Use one simple XY trajectory imitation loss as the primary objective.
8. Use BEV segmentation only on nuPlan.
9. Use route reconstruction on nuPlan and L2D.
10. Keep map and route as runtime model inputs.
11. Keep BEV segmentation camera-only to prevent map-label leakage.
12. Generate missing maps offline from pinned OSM data, never during training.
13. Save and evaluate both the end-of-nuPlan and end-of-L2D checkpoints.
14. Do not use KITScenes metrics for training decisions.

## 3. Goals and Non-Goals

### 3.1 Goals

1. Learn a camera-derived BEV representation from dense nuPlan supervision.
2. Train navigation conditioning with a common map and route contract.
3. Preserve route information through the navigation encoder and fusion gate.
4. Scale trajectory learning with L2D without requiring L2D BEV labels.
5. Keep the primary imitation objective easy to inspect and reproduce.
6. Quantify whether L2D continuation improves trajectory transfer or causes
   catastrophic forgetting.
7. Evaluate the frozen model on KITScenes with immutable benchmark inputs.
8. Add an occupancy-style Dashboard view for BEV predictions, teachers, and
   errors without claiming 3D occupancy.

### 3.2 Non-goals

- World Model JEPA training.
- Horizon Reasoning labels or losses.
- Proactive or deliberative planning branches.
- Route consistency or rollout-aligned control losses.
- Reinforcement learning or closed-loop policy optimization.
- 3D voxel occupancy, occupancy flow, or future occupancy.
- BEV segmentation supervision on L2D.
- Training or fine-tuning on KITScenes.
- Treating an online OSM request as a runtime model dependency.
- Claiming a fair VAD or UniAD comparison across different samples or inputs.

## 4. Model Boundary

### 4.1 Reactive-only data flow

The intended forward path is:

```text
camera images
    -> Backbone
    -> FeatureFusion
    -> image_bev -----------------------------------------------+
         |                                                      |
         +-> BEVSegmentationHead                                |
                 -> semantic logits                             |
                                                                v
map_context + route_mask -> NavigationEncoder -> navigation_bev
                                                    |
                                                    v
                                             navigation gate
                                                    |
                                                    +-> RouteReconstructionHead
                                                    |       -> route logits
                                                    v
                                    MapBEVFusion(image_bev, navigation_bev)
                                                    |
                                                    v
                                           TrajectoryPlanner
                                                    |
                                                    v
                                    acceleration + curvature
                                                    |
                                                    v
                                      differentiable integration
                                                    |
                                                    v
                                             trajectory XY
```

The existing public inputs remain separate:

```text
camera_tiles
map_context
route_mask
map_valid
route_valid
egomotion_history
```

Map and route are concatenated only inside the navigation encoder, as in
[`ReactiveE2E.forward`](../Model/model_components/reactive_e2e.py).

### 4.2 Disabled branches and modes

The first experiment pins:

```text
enable_world_model: false
enable_reasoning: false
temporal_memory_mode: no_memory
planner_mode: gru
enable_route_consistency: false
training objective: simple_xy_imitation_v1
```

No `history_frames`, `future_frames`, `reasoning.json`, JEPA targets, or
teacher-generated reasoning labels are packed for these runs. Ego-motion
history remains a Reactive input.

### 4.3 Head placement

The two auxiliary heads have deliberately different boundaries.

`BEVSegmentationHead` consumes only `image_bev`. It must not consume map,
route, navigation, future camera, or trajectory tensors. This prevents it from
copying map-derived teacher classes.

`RouteReconstructionHead` consumes the gated navigation contribution at the
fusion boundary. With the current residual fusion:

```text
navigation_contribution = alpha * navigation_bev
fused_features = image_bev + navigation_contribution
```

The fusion module should expose `navigation_contribution` explicitly. The
route head does not receive raw `route_mask` through a skip connection.
Training it from `navigation_bev` before `alpha` would not prove that route
information passed the zero-initialized fusion gate.

The planner still consumes `fused_features`. Route reconstruction proves
representation retention, not planner use. Planner use is tested separately
with route-swap and route-zero counterfactuals.

## 5. Common Spatial Contract

### 5.1 BEV geometry

The default
[`BEVViewFusion`](../Model/model_components/view_fusion/bev_fusion.py) contract
is authoritative:

| Field | Value |
| --- | ---: |
| Geometry ID | `autoe2e-bev-450x300-0p4m-v1` |
| Height | `450` |
| Width | `300` |
| Resolution | `0.4 m/px` |
| X extent | `[-60, 120] m` |
| Y extent | `[-60, 60] m` |
| Z pillar extent | `[-5, 3] m` |
| Frame | ego FLU: X forward, Y left, Z up |
| Ego anchor | `(row=299.5, col=149.5)` |

Pixel centers use:

```text
row = (x_max - x) / meters_per_pixel - 0.5
col = (y_max - y) / meters_per_pixel - 0.5
```

The current KITScenes-specific `256 x 256`, `1 m/px` navigation override is
not used for Stage A or Stage B. A new `NavigationRasterGeometry` matching the
table above is required. Changing the grid changes learned BEV query semantics
and is a new experiment, not a compatible data setting.

### 5.2 Canonical navigation channels

Map input uses the existing 14 semantic `MapChannel` values:

```text
drivable_area
lane_boundary
lane_centerline
intersection
crosswalk
stop_line
static_traffic_signal
traffic_direction_sin
traffic_direction_cos
traffic_direction_valid
known_map_area
road_level
road_level_valid
overlapping_level_ambiguity
```

Route input uses two channels:

```text
selected_corridor
destination
```

All channels are rasterized directly into the common geometry. A native RGB
map is not resized and presented as if it had the same semantic contract.

### 5.3 Validity

Every sample carries:

```text
map_valid: bool
route_valid: bool
route_channel_valid: bool[2]
```

`route_channel_valid` allows a valid corridor when a compatible destination
is unavailable or outside the raster. Invalid inputs are zero-filled only
after their validity has been recorded. Losses and metrics must use validity;
zero is not silently interpreted as a known negative.

## 6. Dataset Roles and Supervision

| Dataset | Cameras | Geometry | Map source | Route source | Future XY | BEV teacher | Role |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| nuPlan | 8 | calibrated pinhole after audited rectification | native nuPlan vector map | scenario route roadblock IDs and mission goal | future ego poses | map + lidar boxes + point cloud | Stage A full training |
| L2D | 6 | pseudo until public intrinsics exist | audited native raster if convertible; otherwise pinned OSM | `observation.state.waypoints` map-matched to the same map | vehicle GPS/heading sequence | unavailable | Stage B continuation |
| KITScenes | 6 | dataset calibration | benchmark-track dependent | benchmark-track dependent | evaluator target only | not used for training | Stage C benchmark |

### 6.1 nuPlan contract

Stage A requires nuPlan sensor data, maps, and scenario metadata. The loader
must add nuPlan to the pipeline's `Dataset` enum and produce the same packed
sample ABI as other datasets.

Required sources include:

- eight camera images, intrinsics, distortion, and extrinsics;
- per-image and reference ego poses;
- current and future ego poses;
- native vector map and map version;
- route roadblock IDs and mission goal;
- lidar boxes, categories, and merged point cloud.

nuPlan maps and lidar-derived targets are teacher or navigation data. They are
not passed into the camera-only BEV segmentation head.

Splits are log-level. Adjacent frames from one log must not cross train,
validation, or test boundaries.

### 6.2 L2D contract

L2D provides six real cameras, a rendered BEV navigation map, GPS/heading, and
ten future waypoints snapped to an OSM graph. The existing rendered map is
useful for source audit, but it may not be used as the canonical map tensor
until its metric extent, orientation, palette, and route separation are
verified.

The initial robust path is:

1. read current GPS and heading;
2. load a pinned local OSM regional snapshot;
3. build the canonical semantic map raster;
4. map-match `observation.state.waypoints` to the same graph;
5. rasterize selected corridor and destination;
6. compare the result against `observation.images.map` as an audit;
7. pack canonical map and route tensors.

The recorded future vehicle GPS is the trajectory target. It must not be used
to construct the route input. Doing so would expose the answer that the
trajectory loss is intended to predict.

L2D has no supported BEV semantic teacher. The sample therefore records
`bev_segmentation_available=false`, and Stage B does not execute the
segmentation head in the training forward pass.

Splits are episode-level. Geographic grouping should be used when stable city
or route metadata is available.

### 6.3 KITScenes contract

KITScenes is evaluation-only for this program:

- no KITScenes optimizer batches;
- no checkpoint selection on KITScenes;
- no hyperparameter tuning on KITScenes;
- no threshold calibration on KITScenes;
- no reconstruction-head training on KITScenes.

The benchmark adapter limits observation history to the protocol's four
seconds and evaluates 3-second and 5-second horizons at 10 Hz. When the
declared input track provides map and route, the adapter rerasterizes them into
the checkpoint's `450 x 300` geometry; it does not load the legacy KITScenes
`256 x 256` raster into a `450 x 300` model.

## 7. Offline OSM Map Policy

### 7.1 Source priority

For a dataset sample, choose the first compatible source:

1. dataset-native vector map;
2. dataset-native raster with an audited, lossless semantic conversion;
3. pinned regional OSM vector snapshot;
4. invalid map.

The source choice is deterministic and stored in the sample metadata. A failed
native-map parse must not silently trigger a live network request.

### 7.2 Required geospatial inputs

OSM recovery requires:

- valid latitude and longitude;
- an ego heading with a documented convention;
- a timestamp or dataset revision;
- a region identifier or bounding box.

If geodetic pose is absent, privacy-redacted, or inconsistent, the sample
cannot be assigned a correct external map and receives `map_valid=false`.

### 7.3 Acquisition and preprocessing

Public Overpass requests are acceptable only for a small development smoke
test. Full datasets use regional `.osm.pbf` extracts, downloaded once and
stored as immutable source artifacts.

The production flow is:

```text
dataset pose inventory
  -> regional bounding boxes
  -> download pinned OSM extracts
  -> SHA-256 and source-date manifest
  -> local lane graph and semantic conversion
  -> optional local Valhalla tiles
  -> per-sample ego-centric rasterization
  -> quality audit
  -> immutable training shards
```

No DataLoader, training task, evaluator, or runtime forward pass accesses
Overpass, a raster tile server, or Geofabrik.

The repository already has:

- an offline-oriented OSMnx prototype in
  [`gps_to_map.py`](../Model/data_parsing/map_rendering/gps_to_map.py);
- a local canonical map reader in
  [`OSMMapAdapter`](../Model/navigation/osm_adapter.py);
- a localhost-only route provider and OSM lane resolver in
  [`valhalla.py`](../Model/navigation/valhalla.py).

The missing production component is a deterministic `.osm.pbf` to canonical
lane-graph builder. The current Matplotlib RGB renderer is not the final
14-channel semantic rasterizer.

### 7.4 Map and route are different

OSM supplies a static road graph. It does not identify the driver's selected
route. Route generation additionally requires one of:

- route roadblock or lane IDs;
- a destination and route planner;
- dataset-provided future navigation waypoints.

For L2D, use the provided OSM-snapped waypoints. For nuPlan, use scenario route
roadblock IDs and mission goal. Never infer the route from the exact future ego
trajectory used as the imitation target.

### 7.5 Quality and provenance

Every OSM-derived sample records:

```text
map_provider
map_snapshot_date
map_source_sha256
map_version
adapter_version
projection
map_match_distance statistics
map_match_heading statistics
map_valid
route_valid
```

OSM is not an HD-map guarantee. Missing lane counts, boundaries, traffic
controls, levels, or turn restrictions are represented through validity and
confidence, not invented as exact geometry.

OpenStreetMap attribution and ODbL obligations apply. Source and derived
artifact publication require a license review and visible attribution where
appropriate.

## 8. BEV Segmentation Auxiliary Task

### 8.1 Output

For nuPlan sample `b`:

```text
logits:     float[B, 8, 450, 300]
target:     float[B, 8, 450, 300] in [0, 1]
valid_mask: bool [B, 8, 450, 300]
```

Channels are independent and may overlap:

| Index | Class | Teacher source |
| ---: | --- | --- |
| 0 | `drivable_area` | nuPlan vector map |
| 1 | `lane_area` | lane and lane-connector polygons |
| 2 | `intersection` | intersection polygons |
| 3 | `crosswalk` | crosswalk polygons |
| 4 | `stop_line` | stop-line footprint |
| 5 | `vehicle` | current lidar-box footprint |
| 6 | `vulnerable_road_user` | pedestrian and bicycle footprints |
| 7 | `other_obstacle` | cone, barrier, sign, and generic-object footprints |

Route, destination, traffic-light state, and future ego trajectory are not BEV
segmentation classes.

### 8.2 Target generation

Targets are current-frame, 2D, ego-centric semantic occupancy. Static polygons
come from the map. Dynamic footprints come from current lidar boxes. All
geometry is transformed into the reference ego FLU frame.

Polygons are clipped to the BEV extent and rasterized at `4x` linear
supersampling before averaging into `0.4 m` cells. Targets are stored as
fractional occupancy.

Static validity requires map-layer availability, known map coverage, and
camera geometric visibility. Dynamic validity requires current lidar
observability or a positive box footprint, plus camera geometric visibility.
Unknown and unobserved cells are ignored rather than labeled background.

No future annotation enters a BEV target.

### 8.3 Camera geometry

nuPlan's asynchronous camera timestamps require per-sample pose compensation.
The target frame is one reference lidar/ego pose. Each camera projection maps
that reference frame into the rectified model-input image.

Raw distorted images must not be paired with an uncorrected pinhole matrix.
The first implementation rectifies offline with a versioned policy and records
native calibration, rectified calibration, image transform, and time offsets.

### 8.4 Head

The initial head is:

```text
image_bev [B, 256, 450, 300]
  -> Conv2d(256, 64, 1, bias=False)
  -> GroupNorm(8, 64)
  -> SiLU
  -> depthwise Conv2d(64, 64, 3, padding=1, bias=False)
  -> GroupNorm(8, 64)
  -> SiLU
  -> Conv2d(64, 8, 1)
  -> logits [B, 8, 450, 300]
```

There is no sigmoid inside the head.

### 8.5 Loss

For each active class, use an equal mixture of masked class-balanced
`BCEWithLogits` and masked Soft Dice:

```text
L_bev_class[k] = 0.5 * L_bce[k] + 0.5 * L_dice[k]

L_bev =
  sum(active[k] * L_bev_class[k])
  / max(sum(active[k]), 1)
```

Class positive weights are computed once from the nuPlan training split:

```text
pos_weight[k] =
  clip(valid_negative_count[k] / valid_positive_count[k], 1, 20)
```

Validation and test labels do not affect the weights. A class with no valid
cells is inactive. A batch with no active BEV class contributes no BEV loss
but may still contribute trajectory and route losses.

## 9. Route Reconstruction Auxiliary Task

### 9.1 Purpose

Trajectory supervision alone can solve many frames without route information.
The current navigation fusion also starts with a zero-valued residual gate.
Route reconstruction provides a direct gradient proving that selected-route
information survives the navigation encoder and gate.

It does not prove that the trajectory planner uses that information. That
claim requires counterfactual evaluation.

### 9.2 Target contract

The target is the detached, versioned route raster:

```text
route_target:        float[B, 2, 450, 300]
route_channel_valid: bool [B, 2]

channel 0: selected corridor occupancy
channel 1: destination heatmap
```

The corridor is a lane polygon union when reliable lane boundaries exist.
Otherwise it is a confidence-labeled buffer around the map-matched route
centerline. The buffer policy and source are stored in metadata.

The destination target is a Gaussian heatmap centered on the route's local
goal when that goal is inside the BEV raster. If it is unavailable or outside
the raster, destination validity is false while corridor validity may remain
true.

### 9.3 Dataset construction

nuPlan:

```text
route roadblock IDs + native map + mission goal
  -> lane sequence
  -> selected corridor
  -> visible destination heatmap
```

L2D:

```text
observation.state.waypoints + pinned OSM graph
  -> waypoint map matching
  -> connected route sequence
  -> selected corridor
  -> final valid waypoint heatmap
```

The L2D actual future GPS trajectory is excluded from this construction.

### 9.4 Head and gradient boundary

The head is intentionally small:

```text
navigation_contribution [B, 256, 450, 300]
  -> Conv2d(256, 64, 1)
  -> GroupNorm(8, 64)
  -> SiLU
  -> depthwise Conv2d(64, 64, 3, padding=1)
  -> SiLU
  -> Conv2d(64, 2, 1)
  -> route logits [B, 2, 450, 300]
```

It has no access to raw route pixels. Gradients reach:

- `RouteReconstructionHead`;
- `NavigationEncoder`;
- navigation fusion gate and navigation-side fusion parameters.

They do not reach `Backbone` or `FeatureFusion` through this auxiliary term.
The trajectory and BEV losses remain responsible for camera representation.

### 9.5 Loss

Corridor occupancy uses masked weighted BCE plus Soft Dice:

```text
L_corridor = 0.5 * L_weighted_bce + 0.5 * L_dice
```

The destination heatmap uses a focal heatmap loss to avoid domination by
background pixels:

```text
L_route_reconstruction =
  L_corridor + 0.25 * L_destination_focal
```

Only valid channels and samples contribute. An all-invalid batch returns a
differentiable zero for this term.

### 9.6 Avoiding a misleading result

Because route is already an input, successful reconstruction can be a useful
information-path check while still being an easy autoencoding task. The first
version controls this risk by:

- attaching after the navigation gate;
- forbidding a raw-route skip connection;
- limiting decoder capacity;
- evaluating route swap and route zero behavior;
- reporting reconstruction IoU separately from trajectory response.

Route patch masking, denoising, contrastive route objectives, and synthetic
alternative routes are later ablations, not part of the initial run.

## 10. Simple Trajectory Imitation

### 10.1 Common target

Both training datasets produce:

```text
trajectory_xy_m:     float[B, 64, 2]
trajectory_valid:    bool [B, 64]
initial_speed_mps:   float[B]
frequency:           10 Hz
horizon:             6.4 s
frame:               current ego FLU
```

The target excludes the current point and starts at `t + 0.1 s`.

nuPlan future ego poses are transformed into the current ego frame and
resampled at 10 Hz. L2D future GPS/heading states are projected into a local
metric frame and transformed into current ego coordinates. Samples with
non-finite poses, implausible jumps, or insufficient future coverage are
rejected or masked.

### 10.2 Prediction

The initial planner keeps the current runtime output:

```text
predicted_controls: float[B, 64, 2]
  signal 0: acceleration
  signal 1: curvature
```

`integrate_controls_torch` converts these controls and current speed into:

```text
predicted_xy_m: float[B, 64, 2]
```

Keeping the control output preserves vehicle-kinematic structure and existing
runtime compatibility. The training target and loss are nevertheless only XY
trajectory.

### 10.3 Loss

Use uniform masked Smooth L1 over X and Y:

```text
L_trajectory =
  sum(valid[t] * smooth_l1(predicted_xy[t] - target_xy[t], beta=1 m))
  / max(2 * sum(valid[t]), 1)
```

Every valid timestep has equal weight. The initial objective has:

- no temporal decay;
- no acceleration or curvature target loss;
- no dataset-specific signal scaling;
- no explicit endpoint term;
- no heading term;
- no route distance term;
- no collision or drivable-area term;
- no rollout-aligned selector.

ADE, FDE, comfort, collision, and route compliance remain evaluation metrics.
They are not silently folded into the primary loss.

## 11. Combined Objectives

### 11.1 Stage A

nuPlan full training uses:

```text
L_stage_a =
  L_trajectory
  + lambda_bev   * L_bev
  + lambda_route * L_route_reconstruction
```

All Reactive core modules and both auxiliary heads are trainable:

```text
Backbone
FeatureFusion
NavigationEncoder
MapBEVFusion
TemporalMemory(no_memory implementation)
TrajectoryPlanner
BEVSegmentationHead
RouteReconstructionHead
```

Map has no separate reconstruction loss in v1. The navigation encoder receives
map-dependent gradients from trajectory imitation and route reconstruction.
The `no_memory` TemporalMemory implementation has no independent learning
objective and is not counted as an additional branch.

### 11.2 Stage B

L2D continuation uses:

```text
L_stage_b =
  L_trajectory
  + lambda_route * L_route_reconstruction

lambda_bev = 0
```

`BEVSegmentationHead` is loaded from Stage A but excluded from the Stage B
optimizer. It is not executed for training batches. Shared camera features are
allowed to change, so post-L2D segmentation quality is a retention diagnostic,
not a guaranteed invariant.

### 11.3 Auxiliary weights

Weights are frozen before full training using a fixed nuPlan mini-split. They
are selected from a small predeclared grid by shared-parameter gradient norms,
not by KITScenes performance.

Initial target ranges are:

```text
BEV-to-trajectory shared gradient norm ratio:   0.1 to 0.5
route-to-trajectory navigation gradient ratio:  0.1 to 0.5
```

If no candidate is finite and within range, the full run is blocked for a loss
or target audit. Dynamic weighting, GradNorm, PCGrad, and uncertainty weighting
are out of scope.

## 12. Sequential Training Program

### 12.1 Stage 0: freeze data contracts

Before optimizer work:

1. implement and audit the common `450 x 300` geometry;
2. implement the nuPlan parser and source manifest;
3. implement deterministic OSM regional ingest;
4. audit L2D waypoint and map alignment;
5. freeze train and validation splits;
6. freeze trajectory, BEV, map, and route schema versions.

### 12.2 Stage A: nuPlan full training

Stage A starts from the normal image-backbone initialization and trains the
complete Reactive path jointly.

Recommended baseline:

| Field | Value |
| --- | --- |
| Optimizer | AdamW |
| Precision | float32 first; bf16 only after parity audit |
| Gradient clipping | global norm `1.0` |
| Effective batch | at least 16 through accumulation |
| Planner | GRU |
| Model selection | nuPlan validation trajectory metric |
| Auxiliary guard | finite BEV and route metrics with nonzero intended gradients |

The exact learning rates and step count are experiment configuration, not
dataset defaults. The source manifest, optimizer state, scheduler state, and
best checkpoint are published.

KITScenes is not run during Stage A model selection.

### 12.3 Stage B: L2D continuation

Stage B loads Stage A model weights, starts a fresh optimizer and scheduler,
and continues training on L2D.

Resetting optimizer state is intentional: Stage B changes camera count,
projection type, map source, and objective availability. Carrying Adam moments
across that boundary would couple the datasets in a difficult-to-audit way.

All Reactive core modules remain trainable. The learning rate should be lower
than Stage A and frozen before the full run. There is no nuPlan replay in the
initial sequential baseline, because the first question is whether pure L2D
continuation adds scale or causes forgetting.

Publish:

- the final and best Stage B checkpoints;
- the exact Stage A parent checkpoint hash;
- L2D validation trajectory metrics;
- route reconstruction metrics;
- a fixed nuPlan retention evaluation of the Stage B checkpoint.

### 12.4 Stage C: KITScenes benchmark

Evaluate at least:

| Checkpoint | Purpose |
| --- | --- |
| End of Stage A | nuPlan-only transfer |
| End of Stage B | effect of L2D continuation |

These checkpoint choices are predeclared. KITScenes results do not choose
which checkpoint becomes the reported primary model. The Stage B checkpoint
is primary by training-program definition; Stage A is a diagnostic.

No checkpoint receives KITScenes gradient updates.

## 13. KITScenes Benchmark and Baselines

### 13.1 Protocol

Use the existing immutable KITScenes benchmark manifest contract:

- 10 Hz;
- four seconds of past observation;
- 3-second and 5-second horizons;
- exact sample UID list and digest;
- dataset and SDK revisions;
- declared input track;
- checkpoint SHA-256;
- evaluator version.

Report at minimum:

- ADE at 3 s and 5 s;
- FDE at 3 s and 5 s;
- drivable-surface survival when authority assets are available;
- collision-free rate when authority assets are available;
- centerline distance when authority assets are available;
- Multi-Maneuver Score when official references are available.

Do not synthesize unavailable authority metrics.

### 13.2 Current protocol limitation

KITScenes currently has a split and manifest ambiguity: the paper describes a
200-window development protocol from `val` plus `overlap-train-val`, while the
website describes 200 `test-e2e` samples and a future community leaderboard.
The released exact authority manifest and evaluator remain the source of truth
when available.

Until then, results must be labeled either:

```text
paper_protocol_approximation
official
```

The two statuses are never merged.

### 13.3 Input-track limitation

The proposed primary model consumes camera, semantic map, and route raster.
Published KITScenes UniAD results are camera-based and use a discrete
navigation command derived from the ground-truth future trajectory. The input
information is therefore different.

Every comparison table must include:

```text
camera count
history length
map input
route or command input
navigation source
training datasets
fine-tuning datasets
checkpoint source
sample manifest
```

A map-and-route-conditioned AutoE2E result may be shown next to a camera-based
UniAD result for context, but the document must not claim an architecture win
from that row alone.

The held-out `test-e2e` release withholds map and geodetic pose. OSM cannot be
recovered without pose. If the official track does not provide or permit map
and route inputs, the full model is ineligible for that track. A separately
trained no-map model is required; silently setting map and route to zero is not
a fair substitute.

### 13.4 UniAD and VAD comparison

Use two comparison levels:

1. **Published reference:** record the official KITScenes UniAD row and its
   protocol verbatim. Record VAD's published nuScenes values only as
   cross-dataset background, never as a KITScenes score.
2. **Same-manifest execution:** adapt official public UniAD and VAD
   checkpoints to the exact frozen KITScenes manifest and evaluate their XY
   trajectories with the same evaluator used for AutoE2E.

Same-manifest adapters must pin:

- upstream repository revision;
- checkpoint URL and SHA-256;
- image preprocessing and camera ordering;
- calibration conversion;
- history policy;
- navigation-command construction;
- output-frame conversion;
- any unsupported or dropped sample.

No baseline is fine-tuned on KITScenes. Any baseline that cannot consume the
declared sample without future leakage is reported as unsupported rather than
given a fabricated score.

## 14. Dashboard Visualization

### 14.1 Semantic occupancy view

Add a **Semantic occupancy** view synchronized with camera playback:

- top-down view;
- isometric occupancy-style view;
- `Prediction`, `Teacher`, and `Error` modes;
- class visibility controls with color swatches;
- confidence threshold and opacity controls;
- ego footprint and metric range markers;
- per-class confidence under the pointer.

The persistent title is **2D BEV semantic occupancy**. Isometric extrusion is
a display device, not predicted height. The UI must not call it Tesla
Occupancy, 3D occupancy, or voxel occupancy.

Teacher and Error are available for nuPlan only. L2D and KITScenes show
Prediction unless a compatible teacher artifact is explicitly present.

### 14.2 Route retention diagnostics

A separate debug overlay may show:

- input selected corridor;
- reconstructed corridor probability;
- input destination;
- reconstructed destination heatmap;
- route-swap trajectory delta.

This overlay is labeled as a representation diagnostic, not route planning
ground truth.

### 14.3 Artifact boundary

Dense semantic probabilities are not appended to the trajectory-oriented AOVL
artifact. Use a separate immutable artifact keyed by:

```text
model checkpoint SHA-256
dataset manifest SHA-256
sample UID
geometry ID
taxonomy version
head version
```

Predictions store quantized sigmoid probabilities:

```text
probability_u8: uint8[N, 8, 450, 300]
```

Teacher artifacts independently store:

```text
target_u8: uint8[N, 8, 450, 300]
valid_bits: packed bool[N, 8, 450, 300]
```

Dashboard inference is precomputed in GPU/Flyte jobs. The Dashboard API does
not run the model or fetch OSM.

## 15. Metrics

### 15.1 Trajectory

Report:

- ADE and FDE at 1, 2, 3, 5, and 6.4 seconds where target coverage permits;
- longitudinal and lateral displacement error;
- valid horizon coverage;
- non-finite prediction rate;
- comfort metrics as diagnostics;
- route and drivable compliance as diagnostics.

### 15.2 BEV segmentation

On nuPlan valid cells, report per class and macro:

- IoU;
- Dice/F1;
- pixel Average Precision;
- precision and recall;
- Brier score;
- calibration error;
- positive prevalence;
- valid-cell coverage.

Stratify at minimum by distance, city, day/night, and static/dynamic class.

### 15.3 Route reconstruction and use

Report:

- corridor IoU and Dice;
- destination localization error on valid destinations;
- valid sample and channel counts;
- fusion-gate magnitude;
- route-input gradient evidence;
- trajectory delta under route zeroing;
- trajectory delta and directional correctness under route swap.

A high route IoU with zero trajectory response is a failed route-use result,
not success.

### 15.4 Sequential transfer

Evaluate Stage A and Stage B checkpoints on frozen nuPlan and L2D validation
sets. Report a 2 x 2 matrix:

| Checkpoint | nuPlan validation | L2D validation |
| --- | --- | --- |
| Stage A | in-domain baseline | zero-shot transfer |
| Stage B | retention/forgetting | continued-training result |

This separates data-scale benefit from catastrophic forgetting before looking
at KITScenes.

## 16. Required Ablations

The first research matrix is:

| ID | Stage A losses | Stage B losses | Purpose |
| --- | --- | --- | --- |
| A0 | trajectory | trajectory | simple Reactive baseline |
| A1 | trajectory + BEV | trajectory | isolate BEV supervision |
| A2 | trajectory + route reconstruction | trajectory + route reconstruction | isolate route retention |
| A3 | trajectory + BEV + route reconstruction | trajectory + route reconstruction | proposed full program |

All runs share:

- source and split manifests;
- initialization policy;
- geometry;
- model capacity;
- optimizer steps per stage;
- batch size;
- checkpoint-selection rule;
- KITScenes manifest.

The full three-seed KITScenes evaluation is required for A0 and A3. A1 and A2
may first use one seed for diagnosis, but any reported comparison must state
the seed count.

## 17. Data Artifacts

### 17.1 Packed sample

The common sample schema contains:

```text
camera images
projection and image transform
map_context.npz
route_mask.npz
trajectory_xy.npz
egomotion history
sample metadata
```

nuPlan additionally contains:

```text
bev_segmentation.npz
```

L2D does not contain a fabricated empty BEV teacher. Availability is explicit
in metadata and manifest.

### 17.2 Required manifest fields

Each immutable dataset manifest records:

```text
dataset and source revision
ordered shard hashes
sample count and rejection counts
camera order
projection types
geometry ID
map schema and source versions
OSM snapshot hashes where used
route schema and matcher version
trajectory schema
BEV taxonomy and target-policy digest where available
split membership
license and attribution state
```

### 17.3 Checkpoint lineage

Stage B checkpoints record:

```text
stage_a_parent_checkpoint_sha256
stage_a_config_digest
stage_b_dataset_manifest_sha256
stage_b_config_digest
model_state_sha256
```

Mutable registry aliases are not sufficient provenance.

## 18. Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Missing required camera | Reject sample; do not zero-fill a view |
| Invalid camera calibration | Reject calibrated sample or use an explicitly declared pseudo track |
| Geometry mismatch | Fail before model construction |
| Live map request during train/eval | Hard error |
| Missing GPS for OSM fallback | `map_valid=false`; no guessed map |
| OSM snapshot digest mismatch | Fail artifact build |
| Route built from imitation future trajectory | Hard error |
| Route map match fails quality policy | `route_valid=false` and count |
| Missing L2D BEV teacher | Expected; skip BEV head and loss |
| Missing nuPlan BEV teacher in Stage A | Reject sample or fail above frozen threshold |
| No valid cells for one auxiliary term | Differentiable zero for that term |
| Non-finite target, loss, or gradient | Stop run and preserve sample IDs |
| Stage B parent hash mismatch | Refuse resume |
| KITScenes sample-set mismatch | Fail benchmark |
| KITScenes used for checkpoint selection | Invalidate experiment |
| Baseline input modality omitted | Do not publish comparison |

## 19. Implementation Stages

### Stage 1: Common contracts

- Add the `450 x 300` navigation geometry.
- Add packed map, route-channel validity, and XY trajectory schemas.
- Add nuPlan to dataset selection.
- Add manifest validation and checkpoint lineage.

### Stage 2: Map and route preprocessing

- Implement nuPlan map and route adapters.
- Implement deterministic OSM `.pbf` ingest and canonical graph build.
- Implement L2D waypoint map matching.
- Audit canonical rasters against L2D provided map images.
- Add attribution and source-digest metadata.

### Stage 3: Targets and heads

- Implement nuPlan BEV target builder.
- Add `BEVSegmentationHead`.
- Expose gated navigation contribution.
- Add `RouteReconstructionHead`.
- Add masked BEV and route losses.

### Stage 4: Simple trajectory objective

- Build common future XY targets.
- Integrate predicted controls with Torch.
- Add uniform masked XY Smooth L1.
- Disable legacy objective terms for this objective version.

### Stage 5: Sequential workflows

- Build and audit nuPlan shards.
- Train and evaluate Stage A.
- Build and audit L2D navigation shards.
- Continue Stage B with a fresh optimizer.
- Run frozen cross-dataset retention evaluation.

### Stage 6: KITScenes

- Freeze the benchmark manifest.
- Evaluate Stage A and Stage B checkpoints.
- Add pinned UniAD and VAD adapters.
- Publish input-aware comparison tables.

### Stage 7: Dashboard

- Precompute semantic prediction and teacher artifacts.
- Add top-down and isometric semantic views.
- Add optional route-retention diagnostics.
- Verify desktop and mobile rendering with Playwright.

## 20. Test Plan

### 20.1 Geometry and data

- Metric points round-trip through raster coordinates within tolerance.
- Ego anchor and axis directions match camera BEV.
- nuPlan per-camera pose compensation passes synthetic projection tests.
- OSM rebuilds are byte-deterministic from one pinned snapshot.
- L2D waypoints never read post-sample trajectory target rows.
- Map and route validity survive pack/decode unchanged.

### 20.2 Losses

- Perfect prediction approaches zero loss.
- Invalid cells and timesteps contribute zero.
- Empty classes and all-invalid batches remain finite.
- BEV loss has no gradient into navigation modules.
- Route loss has no gradient into camera modules.
- Trajectory loss reaches camera, navigation, fusion, and planner modules.
- Legacy route and rollout losses remain inactive.

### 20.3 Model behavior

- Segmentation logits do not change when map or route inputs change.
- Route logits do change when valid route inputs change.
- Route reconstruction receives no raw-route skip.
- Fusion gate receives nonzero route-loss gradient.
- Route swap changes the planner output on route-choice scenes.
- Camera count can change from nuPlan eight to L2D six without tensor surgery.

### 20.4 Workflows

- Stage A cannot start without BEV target availability.
- Stage B rejects a checkpoint without Stage A lineage.
- Stage B does not instantiate BEV loss.
- World Model and Reasoning parameters are absent or gradient-free.
- KITScenes tasks expose no optimizer.
- Benchmark results bind exact sample and checkpoint hashes.

### 20.5 Dashboard

- Probability quantization error is at most `1/255`.
- Teacher-unavailable state is explicit.
- Controls do not overlap at desktop or mobile sizes.
- WebGL canvas is nonblank and correctly framed.
- Isometric view is labeled as 2D semantic occupancy.

## 21. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| L2D pseudo camera geometry degrades calibrated BEV features | Keep Stage A checkpoint, use lower Stage B LR, report nuPlan retention |
| OSM is incomplete at lane level | Validity/confidence, source audit, no HD-map claim |
| Route reconstruction learns only a pixel copy | Decode after gate, limit head, require route-swap response |
| Route loss distorts camera features | Route auxiliary gradients are blocked from camera branch |
| L2D continuation forgets BEV semantics | Freeze head, retain Stage A checkpoint, evaluate nuPlan segmentation after Stage B |
| Auxiliary gradients dominate trajectory | Fixed-batch gradient calibration before full training |
| Map source differs between datasets | Canonical semantic channels and one geometry |
| Future trajectory leaks into route input | Source-specific route builders and provenance tests |
| KITScenes map/route inputs differ from published baselines | Separate input tracks and prohibit direct win claims |
| KITScenes is used repeatedly as a validation set | Predeclare checkpoints and run benchmark only after design freeze |
| OSM license obligations are missed | Snapshot manifest, attribution, legal review |

## 22. Deferred Work

- nuPlan replay during L2D continuation.
- Joint mixed-dataset batches.
- Direct XY planner output instead of control integration.
- Future occupancy and occupancy flow.
- Map reconstruction auxiliary loss.
- Route denoising or contrastive objectives.
- Alternative-route counterfactual training.
- Closed-loop simulation and policy optimization.
- World Model and Reasoning reintroduction.
- Official `test-e2e` submission when the input contract is released.

## 23. References

1. Philion, J. and Fidler, S. **Lift, Splat, Shoot: Encoding Images from
   Arbitrary Camera Rigs by Implicitly Unprojecting to 3D.** ECCV 2020.
   [Paper](https://arxiv.org/abs/2008.05711),
   [implementation](https://github.com/nv-tlabs/lift-splat-shoot).
2. Li, Z. et al. **BEVFormer: Learning Bird's-Eye-View Representation from
   Multi-Camera Images via Spatiotemporal Transformers.** ECCV 2022.
   [Paper](https://arxiv.org/abs/2203.17270).
3. Liu, Z. et al. **BEVFusion: A Simple and Robust LiDAR-Camera Fusion
   Framework.** NeurIPS 2022.
   [Paper](https://arxiv.org/abs/2205.13790).
4. Hu, A. et al. **FIERY: Future Instance Prediction in Bird's-Eye View from
   Surround Monocular Cameras.** ICCV 2021.
   [Paper](https://arxiv.org/abs/2104.10490).
5. Zhang, J. et al. **BEVerse: Unified Perception and Prediction in Birds-Eye
   View for Vision-Centric Autonomous Driving.** arXiv 2022.
   [Paper](https://arxiv.org/abs/2205.09743).
6. Hu, S. et al. **ST-P3: End-to-end Vision-based Autonomous Driving via
   Spatial-Temporal Feature Learning.** ECCV 2022.
   [Paper](https://arxiv.org/abs/2207.07601).
7. Hu, Y. et al. **Planning-oriented Autonomous Driving (UniAD).** CVPR 2023.
   [Paper](https://arxiv.org/abs/2212.10156),
   [implementation](https://github.com/OpenDriveLab/UniAD).
8. Jiang, B. et al. **VAD: Vectorized Scene Representation for Efficient
   Autonomous Driving.** ICCV 2023.
   [Paper](https://arxiv.org/abs/2303.12077),
   [implementation](https://github.com/hustvl/VAD).
9. Caesar, H. et al. **nuPlan: A Closed-loop ML-based Planning Benchmark for
   Autonomous Vehicles.** CVPR ADP3 Workshop 2021.
   [Paper](https://arxiv.org/abs/2106.11810),
   [devkit](https://github.com/motional/nuplan-devkit).
10. Yaak AI. **L2D: Learning to Drive.**
    [Dataset](https://huggingface.co/datasets/yaak-ai/L2D),
    [overview](https://huggingface.co/blog/lerobot-goes-to-driving-school).
11. KIT-MRT. **KITScenes Multimodal E2E Driving Benchmark.**
    [Benchmark](https://kitscenes.com/benchmarks/multimodal-e2e-driving),
    [dataset](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal).
12. OpenStreetMap contributors. **OpenStreetMap data and ODbL.**
    [Copyright and license](https://www.openstreetmap.org/copyright).
13. Boeing, G. **OSMnx: New Methods for Acquiring, Constructing, Analyzing,
    and Visualizing Complex Street Networks.** Computers, Environment and
    Urban Systems 2017.
    [Paper](https://doi.org/10.1016/j.compenvurbsys.2017.05.004),
    [implementation](https://github.com/gboeing/osmnx).
14. Valhalla contributors. **Valhalla routing engine.**
    [Implementation](https://github.com/valhalla/valhalla).
