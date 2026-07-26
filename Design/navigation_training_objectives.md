# Design: Navigation-Aware Training Objectives for KITScenes

## Document Metadata

| Field | Value |
|---|---|
| Status | Proposed |
| Owner | riita10069 |
| Created | 2026-07-26 |
| Related issue | [#149](https://github.com/autowarefoundation/auto_e2e/issues/149) |
| Builds on | `Docs/navigation_input_design.md` |
| Initial dataset | KITScenes navigation v3 only |
| Route information boundary | Reactive branch only |

## 1. Executive Summary

The first route-conditioned KITScenes experiment proves that route pixels reach
the Reactive planner and receive non-zero gradients. It does not yet strongly
reward the model for using route information:

- the 64th control step receives only `0.95^63 = 0.0395` of the first step's
  trajectory-loss weight;
- most samples do not contain an imminent route choice;
- the imitation loss does not directly penalize leaving the selected route;
- the camera BEV has no explicit geometric supervision before it is fused with
  the semantic map and route.

This design introduces one versioned KITScenes training objective with four
independently configurable components:

1. a camera-only BEV semantic segmentation auxiliary head;
2. mean-normalized long-horizon trajectory weights with decay `0.99`;
3. deterministic junction- and maneuver-aware training resampling;
4. a differentiable selected-route consistency loss.

The initial combined training path is:

```text
camera images
    -> shared camera backbone
    -> camera FeatureFusion
    -> image_bev ------------------------------------+
         |                                           |
         +-> train-only BEV segmentation head        |
         |      -> semantic map targets               |
         |                                            v
map_context + route_mask -> NavigationEncoder -> MapBEVFusion
                                                   |
                                                   v
                                      Reactive trajectory planner
                                                   |
                                                   +-> controls
                                                        |-- long-horizon
                                                        |   imitation loss
                                                        +-- differentiable
                                                            route loss
```

The BEV segmentation head reads `image_bev` before navigation fusion. It cannot
copy the semantic map input. Route-supervision fields are training targets, not
model inputs. Route tensors still do not enter the Reasoning branch.

All four components can be enabled together, but each has an explicit config
field, checkpoint identity, MLflow metric namespace, and focused test. This
preserves the ability to run controlled ablations after the combined result.

## 2. Context and Evidence

### 2.1 Current model path

The current Reactive path is:

```text
camera -> Backbone -> FeatureFusion -> image_bev ------------------+
                                                                    |
map_context + route_mask -> shared NavigationEncoder -> nav_bev ----+
                                                                    |
                                                zero-init residual fusion
                                                                    |
                                                     TrajectoryPlanner
```

The public model API keeps `map_context` and `route_mask` separate, then gates
and concatenates them immediately before one shared navigation encoder. This
design does not change that initial #149 architecture.

### 2.2 Current training behavior

The controlled run uses:

- 64 predicted `(acceleration, curvature)` controls at 10 Hz;
- Smooth L1 imitation loss;
- temporal decay `0.95`;
- no route-specific training loss;
- uniform training sample exposure after the frozen scene-level split;
- early stopping on validation ADE with patience 3.

The route-conditioned run reached Epoch 9 with ADE `3.7805 m` and FDE
`10.5128 m`. It continued from Epoch 10 into Epoch 11 with non-zero route-input,
navigation-encoder, and navigation-fusion gradients. This is evidence of a
working information path, not evidence that the route is used optimally.

### 2.3 Why the changes are coupled

The four changes address different failure modes:

| Change | Failure mode |
|---|---|
| Long-horizon weighting | Route choices occur after the heavily weighted near-term controls |
| Junction-aware resampling | Straight lane following dominates the optimizer |
| Route consistency | Control imitation alone does not identify the selected lane sequence as a constraint |
| BEV segmentation | Camera BEV geometry is learned only indirectly through planning |

Applying only a route loss to a weak camera BEV can produce map dependence
without better perception. Applying only segmentation does not teach the
planner which branch is selected. Applying only resampling repeats the same
weak objective. The combined objective is therefore a coherent first
performance experiment, while independent switches retain scientific
traceability.

## 3. Goals and Non-Goals

### 3.1 Goals

1. Increase supervision on the full 6.4-second control horizon.
2. Increase optimizer exposure to samples where route intent disambiguates the
   camera scene.
3. Penalize predicted trajectories that leave the selected lane corridor or
   take the wrong junction branch.
4. Train the camera BEV to represent road geometry without reading the map or
   route tensors.
5. Preserve the #149 navigation ABI and Reactive-only route boundary.
6. Keep every new objective deterministic, masked by validity, and observable.
7. Allow a combined run and controlled per-component ablations from the same
   implementation.
8. Keep validation membership and distribution unchanged.

### 3.2 Non-goals

- Route input to `HorizonReasoningHead`.
- A separate route encoder in the first combined experiment.
- Counterfactual trajectory labels for routes that were not driven.
- Synthetic wrong-route training.
- Map/route dropout or localization perturbation.
- Changing the 64-step `(acceleration, curvature)` output ABI.
- Changing runtime navigation inputs or adding a runtime segmentation output.
- Replacing the existing early-stopping policy in the first comparison.
- Applying the new objective to L2D or NVIDIA PhysicalAI-AV.

## 4. Information Boundaries

There are three distinct contracts:

```text
model input:
  camera_tiles, map_context, route_mask, map_valid, route_valid, histories

train-only target:
  semantic BEV labels, selected-route distance/direction fields,
  target controls

runtime output:
  64 x (acceleration, curvature)
```

The following rules are normative:

1. The BEV segmentation head receives only `image_bev`.
2. `map_context` is a target for the segmentation loss and remains a separate
   input to `NavigationEncoder`; it is never concatenated into the segmentation
   head.
3. Selected-route distance and direction fields are consumed only by the
   training loss.
4. Exact future ego positions are never serialized as route inputs or route
   supervision.
5. Ground-truth controls may be integrated inside the loss for masking or a
   relative destination hinge because they are already the imitation target.
   They are never passed to the forward path used for inference.
6. Route-derived tensors remain inside the Reactive training path and never
   enter the Reasoning head, its teacher labels, or its cache.

## 5. Model Architecture

### 5.1 Reactive output contract

`ReactiveE2E` exposes train-only auxiliary outputs through the existing
`aux_outputs` dictionary:

```text
trajectory: [B, 128]
aux_outputs:
  reasoning_pred: optional existing value
  future_state_pred: optional existing value
  bev_segmentation_logits: optional [B, 6, 256, 256]
```

Inference returns only the trajectory. The segmentation head is not executed
when `mode != "train"`, so it adds no vehicle latency.

### 5.2 BEV segmentation head

The initial head is intentionally small:

```text
image_bev [B, 256, 256, 256]
  -> Conv2d(256, 128, kernel=3, padding=1)
  -> GroupNorm(32, 128)
  -> GELU
  -> Conv2d(128, 6, kernel=1)
  -> logits [B, 6, 256, 256]
```

The spatial geometry is exactly `kitscenes-v3-bev-1m-v1`. No resize is
permitted when the camera BEV and navigation geometry already match. A shape
mismatch is a contract error, not an invitation to bilinear-resize labels.

The six initial target channels are:

1. `DRIVABLE_AREA`;
2. `LANE_BOUNDARY`;
3. `LANE_CENTERLINE`;
4. `INTERSECTION`;
5. `CROSSWALK`;
6. `STOP_LINE`.

The following map channels are excluded:

- static traffic-signal points are too sparse for the initial dense head;
- traffic direction is a vector-regression task, not binary segmentation;
- known-map area is a supervision mask;
- road level and overlap ambiguity are map provenance, not camera-visible
  semantic classes.

### 5.3 No segmentation shortcut

Attaching the head after `MapBEVFusion` would let the model copy the supplied
semantic map. That can drive segmentation loss to zero without improving visual
features. The head therefore branches from `image_bev` before
`NavigationEncoder` output is fused.

This is enforced by a gradient-isolation test:

- segmentation loss must produce gradients in `Backbone`, `FeatureFusion`, and
  `BEVSegmentationHead`;
- it must not produce gradients in `NavigationEncoder`, `MapBEVFusion`,
  `TrajectoryPlanner`, or the Reasoning branch when evaluated by itself.

### 5.4 Runtime footprint

The head remains in the training checkpoint so a run is exactly reconstructable
and segmentation quality can be evaluated. It is skipped in inference. A
deployment export may omit its parameters after verifying that trajectory
outputs remain byte-identical.

## 6. BEV Segmentation Loss

### 6.1 Validity

For sample `b`, pixel `p`, and target channel `c`, supervision is valid only
when:

```text
map_valid[b]
and map_layer_valid[b, c]
and map_context[b, KNOWN_MAP_AREA, p] == 1
```

`map_layer_valid` is derived from `NavigationMap.layer_availability` and packed
as training metadata. An unavailable layer is ignored, not treated as an
all-negative label.

### 6.2 Loss definition

The segmentation loss combines masked weighted BCE and soft Dice:

```text
L_bev = 0.5 * L_masked_bce + 0.5 * L_masked_dice
```

Per-channel positive weights are measured only on the frozen training split,
clipped to `[1, 20]`, frozen in a versioned JSON artifact, and recorded in the
checkpoint. Validation samples never contribute to class-weight estimation.

Dice is computed per sample and channel, then averaged only over valid
sample-channel pairs. Empty valid target channels contribute BCE but are
excluded from Dice to avoid an unstable empty-set score.

### 6.3 Metrics

Validation reports per-channel and macro:

- IoU at threshold `0.5`;
- Dice;
- precision and recall;
- valid pixel and positive pixel counts.

Metrics are computed from `image_bev` logits, never navigation-fused features.

## 7. Long-Horizon Imitation Loss

### 7.1 New KITScenes policy

The combined objective changes KITScenes only:

```text
temporal_decay = 0.99
temporal_weight_normalization = "mean_one"
```

L2D and NVIDIA policies remain unchanged.

For `T=64`:

```text
raw_weight[t] = 0.99^t
weight[t] = raw_weight[t] / mean(raw_weight)
```

The final-to-first relative weight becomes:

```text
0.99^63 = 0.5309
```

instead of:

```text
0.95^63 = 0.0395
```

Mean-one normalization keeps the average trajectory-loss scale constant when
changing the temporal distribution. Without normalization, changing `0.95` to
`0.99` also changes the trajectory loss's scale relative to JEPA, Reasoning,
and the new auxiliary losses.

### 7.2 Signal normalization

The existing KITScenes signal scales remain:

```text
acceleration_scale = 0.778
curvature_scale = 0.0350
```

Long-horizon weighting does not change the control units, Smooth L1 definition,
or output shape.

### 7.3 Configuration and compatibility

Temporal decay and normalization mode are checkpoint-defining configuration.
An objective-v1 checkpoint cannot resume from the current `0.95`,
non-normalized run. The first objective-v1 run starts from a fresh model
initialization with the same seed and frozen data split.

The implementation supports `0.99` and `1.0`, but the first combined run uses
`0.99`. Uniform `1.0` remains a later ablation.

## 8. Junction-Aware Training Resampling

### 8.1 Scope

Resampling applies only to the training iterator. Validation remains one
exposure per unique sample in the frozen scene-level split. Evaluation rejects
duplicate validation `sample_uid` values as it does today.

### 8.2 Deterministic exposure policy

The initial maximum repeat count is four:

| Condition | Repeat count |
|---|---:|
| `route_valid` and maneuver in `left`, `right`, `u_turn`, `merge`, `exit` | 4 |
| `route_valid` and `route_intersection=true` | 2 |
| all other samples | 1 |

Conditions use `max`, not multiplication. A left turn in an intersection is
repeated four times, not eight.

The route maneuver is the existing 100 m selected-route lookahead label. It is
route-derived and does not inspect the future ego trajectory.

### 8.3 WebDataset placement

The repeat transform operates on raw WebDataset samples:

```text
read tar
  -> frozen train/validation group filter
  -> parse navigation_meta.json only
  -> deterministic repeat
  -> epoch-seeded shuffle
  -> image/window decode
  -> batch
```

Repeating before image decode prevents unnecessary decode work for discarded
split members and keeps each repeated sample subject to the epoch shuffle.

The transform is a picklable generator, not a lambda. It must work with the
existing worker and bounded multi-loader lifecycle.

### 8.4 Reproducibility

The following values are logged for every epoch:

- unique training sample count;
- effective exposure count;
- exposure counts by maneuver and junction status;
- repeat-policy version and configuration;
- digest of `(sample_uid, repeat_count)` sorted by sample UID.

With the same dataset, split, policy, and epoch seed, the exposure digest must
be identical.

### 8.5 Bias controls

Resampling changes the optimization distribution, not the benchmark
distribution. Aggregate metrics therefore continue to use the original
validation distribution. Junction and maneuver slices are reported separately.

No inverse-frequency weighting is applied on top of deterministic repetition in
the initial implementation. Combining both would make the effective objective
harder to audit.

## 9. Route-Supervision Artifact

### 9.1 Why an additional target is required

Sampling a binary corridor with `grid_sample` gives useful gradients only near
mask edges. It gives no direction toward the route when a predicted point is
far outside the corridor. A selected-route distance field provides a smooth,
metric target.

Map traffic-direction channels are not sufficient for route supervision at
intersections because they describe all mapped lanes, not only the selected
lane sequence.

### 9.2 Packed training target

KITScenes preprocessing adds a loss-only member:

```text
route_supervision.npz
  distance_to_corridor_m: [256, 256] float32
  route_heading_sin:      [256, 256] float32
  route_heading_cos:      [256, 256] float32
  route_heading_valid:    [256, 256] uint8
  destination_xy_m:       [2] float32
  destination_visible:    scalar uint8
```

The artifact is produced from the canonical selected lane sequence and the
sample's final ego-local geometry. It contains no future ego trajectory.

Distance is zero inside the selected corridor and the Euclidean distance in
meters outside it, clipped at 30 m. Heading is the tangent of the selected
route centerline and is valid only where the nearest selected centerline is
unambiguous.

For `route_valid=false`, all fields are zero and all validity fields are false.

### 9.3 Production input remains unchanged

`route_supervision.npz` is decoded into the training batch but is never passed
to `AutoE2E.forward`. The runtime model continues to receive only the binary
selected corridor and destination marker through `route_mask`.

Adding this member creates a new packed training-contract version. Existing
navigation scene artifacts can be deterministically repacked without rerunning
source ingest, Lanelet2 matching, or the Cosmos teacher.

## 10. Differentiable Control Integration

The route loss converts predicted controls into ego-FLU positions with a Torch
implementation matching `evaluation.metrics.integrate_trajectory`:

```text
v[t]     = clamp_min(v[t-1] + acceleration[t] * 0.1, 0)
theta[t] = theta[t-1] + curvature[t] * v[t] * 0.1
x[t]     = x[t-1] + v[t] * cos(theta[t]) * 0.1
y[t]     = y[t-1] + v[t] * sin(theta[t]) * 0.1
```

Initial speed is the final causal speed value in `egomotion_history`. Initial
position and heading are zero in the current ego-FLU frame.

The Torch and NumPy implementations must agree within `1e-5 m` for normal
controls and explicitly tested zero-speed, braking-to-zero, and turning cases.

Positions are converted to the navigation grid using
`NavigationRasterGeometry` and sampled with `grid_sample(align_corners=False)`.
Out-of-bounds positions receive an explicit differentiable distance-to-bounds
penalty; zero padding must not make leaving the raster look like zero route
distance.

## 11. Route Consistency Loss

### 11.1 Eligibility

The route loss is active only when:

- `route_valid=true`;
- route quality passed the existing scene policy;
- the target integrated trajectory has at least 90% selected-corridor
  compliance over in-bounds points.

The last condition prevents a map-match or corridor-width error from forcing
the model away from the demonstrated behavior. It uses the target only as a
train-time loss mask and is never a model input. Eligibility and rejection
counts are logged.

### 11.2 Terms

Let `p_t` and `theta_t` be integrated predicted positions and headings.

#### Corridor distance

```text
L_corridor = weighted_mean(
    smooth_l1(sample(distance_field, p_t) / 10 m)
    + out_of_bounds_distance(p_t) / 10 m
)
```

It uses the same mean-normalized `0.99` horizon weights as trajectory
imitation.

#### Late junction branch

For `route_intersection=true`, the final 32 steps receive an additional
selected-route distance penalty:

```text
L_branch = mean_t=32..63(
    smooth_l1(sample(distance_field, p_t) / route_corridor_width_m)
)
```

This is the differentiable training surrogate for wrong-branch rate. The
discrete wrong-branch metric remains an evaluation metric.

#### Destination approach

A naive absolute terminal distance to the destination can reward unsafe
acceleration when the destination is visible but not reachable in 6.4 seconds.
The initial loss therefore uses a demonstrator-relative hinge:

```text
L_destination = relu(
    distance(predicted_terminal, destination)
    - distance(target_terminal, destination)
    - 1 m
) / 10 m
```

It is active only when `destination_visible=true`. The model is penalized for
ending materially farther from the selected destination than the demonstrator,
but is not rewarded for overshooting the demonstrated progress.

#### Route heading

```text
L_heading = mean(
    1 - (
      cos(theta_t) * sampled_route_heading_cos
      + sin(theta_t) * sampled_route_heading_sin
    )
)
```

It is active only where route heading is valid and predicted speed is at least
`1 m/s`. This avoids assigning unstable heading penalties while stationary.

### 11.3 Combined route term

The normalized initial route term is:

```text
L_route =
    1.00 * L_corridor
  + 2.00 * L_branch
  + 0.50 * L_destination
  + 0.25 * L_heading
```

Each term averages only over eligible samples and returns differentiable zero
when its eligible set is empty. Empty eligibility must not produce NaN.

## 12. Total Training Objective

The complete objective is:

```text
L_total =
    L_trajectory
  + lambda_bev       * L_bev
  + lambda_route     * L_route
  + lambda_jepa      * L_jepa
  + lambda_reasoning * L_reasoning
```

Initial weights:

| Weight | Value |
|---|---:|
| `lambda_bev` | 0.10 |
| `lambda_route` | 0.10 |
| `lambda_jepa` | 1.00 |
| `lambda_reasoning` | 0.05 |

The existing JEPA and Reasoning values remain unchanged.

### 12.1 Gradient budget gate

Numeric weights alone do not guarantee balanced gradients. Before a full run,
the training smoke test computes per-loss gradient norms on one fixed batch:

- `L_trajectory` into the planner and camera backbone;
- `L_route` into the planner;
- `L_bev` into the camera backbone and FeatureFusion.

Neither auxiliary term may exceed the relevant trajectory gradient norm by
more than `2x` on the fixed smoke batch. If it does, its lambda is reduced and
the frozen config is updated before the full run. This is a stability gate, not
an automatic per-step gradient-balancing algorithm.

### 12.2 No silent loss activation

Every auxiliary has both an enable flag and a positive weight. Invalid
combinations fail:

- enabled BEV loss with `lambda_bev <= 0`;
- enabled route loss with `lambda_route <= 0`;
- route loss on a dataset without the supervision contract;
- segmentation head channels that differ from the frozen target list.

## 13. Configuration and Checkpoint Contract

The following fields are checkpoint-defining:

```text
training_objective_version = "kitscenes_navigation_objective_v1"

trajectory:
  temporal_decay: 0.99
  temporal_weight_normalization: mean_one

bev_segmentation:
  enabled: true
  target_channels: [0, 1, 2, 3, 4, 5]
  loss: bce_dice
  weight: 0.10
  class_weight_artifact_sha256: ...

junction_sampling:
  enabled: true
  policy_version: navigation_repeat_v1
  turn_repeat: 4
  junction_repeat: 2
  max_repeat: 4

route_consistency:
  enabled: true
  artifact_version: route_supervision_v1
  weight: 0.10
  target_compliance_threshold: 0.90
  term_weights: [1.00, 2.00, 0.50, 0.25]
```

Resume validation rejects any mismatch. Old checkpoints cannot be interpreted
as objective-v1 checkpoints.

MLflow records every field above plus:

- packed dataset and route-supervision digests;
- effective exposure digest per epoch;
- each unweighted and weighted loss term;
- each auxiliary gradient probe;
- route-loss eligible sample counts;
- segmentation valid-pixel counts.

## 14. Training and Evaluation Procedure

### 14.1 Fresh training

The objective-v1 experiment starts from a fresh initialization. It uses:

- the same seed as the controlled #149 comparison;
- the same frozen train/validation scene membership;
- the same camera and navigation geometry;
- the same batch size and gradient accumulation;
- the same backbone, planner, Reasoning, and World Model settings;
- maximum 20 epochs;
- existing ADE early stopping with patience 3.

Early stopping remains unchanged in the first comparison so the training
objective is not confounded with a new checkpoint-selection rule. All epoch
checkpoints and route metrics remain available for post-training analysis.

### 14.2 Required runs

The minimum full comparison is:

| Run | BEV seg | Decay | Sampling | Route loss | Route input |
|---|---:|---:|---:|---:|---:|
| Existing controlled route run | off | 0.95 | uniform | off | on |
| Objective-v1 combined | on | 0.99 | aware | on | on |
| Objective-v1 no-route control | on | 0.99 | aware | off | off |

The no-route control keeps BEV supervision, long-horizon weighting, and the
same training exposure distribution. It isolates the contribution of route
input plus route consistency from the general training improvements.

The implementation also supports leave-one-component-out runs without code
changes. Full leave-one-out training is required only if the combined result is
ambiguous or regresses.

### 14.3 Metrics

Retain all existing metrics and add:

- ADE and FDE at 1, 2, 3, and 6.4 seconds;
- junction/non-junction ADE and FDE;
- left/right/straight maneuver ADE and FDE;
- wrong-branch rate;
- selected-route compliance and outside distance;
- destination-approach error;
- route-swap counterfactual response;
- six-channel camera-only BEV IoU/Dice;
- effective train exposure distribution;
- each route-loss term and eligibility rate.

Aggregate metrics use the original validation distribution. Oversampled
training metrics are never presented as benchmark metrics.

### 14.4 Success criteria

The combined objective is useful when:

1. aggregate ADE and FDE do not regress by more than 2% against the existing
   route-conditioned best checkpoint;
2. junction FDE or wrong-branch rate improves with a paired bootstrap 95%
   confidence interval excluding zero;
3. route-swap counterfactuals change the prediction in the selected direction;
4. camera-only BEV macro IoU is above the all-background baseline;
5. lane-follow performance remains valid when route input is disabled;
6. all auxiliary branches show non-zero intended gradients and zero forbidden
   gradients.

The exact metric values and confidence intervals are reported even when the
hypothesis is not supported.

## 15. Failure Semantics

| Condition | Behavior |
|---|---|
| `map_valid=false` | Skip BEV segmentation for that sample |
| Semantic layer unavailable | Mask that channel, do not create negative labels |
| `route_valid=false` | Skip route loss and use repeat count 1 |
| Missing route-supervision member with route loss enabled | Fail before optimizer creation |
| Target route compliance below 90% | Skip route loss for that sample and count rejection |
| No eligible route samples in a batch | Differentiable zero route loss |
| No eligible route samples in an epoch | Fail the route-enabled KITScenes run |
| Predicted point outside raster | Apply explicit distance-to-bounds penalty |
| Destination not visible | Skip destination term |
| Route heading invalid or predicted speed below 1 m/s | Skip heading term |
| Non-finite auxiliary loss or metric | Fail before checkpoint upload |

## 16. Implementation Boundaries

### 16.1 Model

- Add a train-only `BEVSegmentationHead`.
- Return its logits through the existing auxiliary-output dictionary.
- Do not modify the Reasoning input contract.
- Keep the shared map/route navigation encoder for this experiment.

### 16.2 Data

- Add layer-valid metadata.
- Add `route_supervision.npz`.
- Add deterministic raw-sample repeat transform.
- Repack from existing immutable scene navigation and reasoning artifacts.

### 16.3 Training

- Add mean-normalized temporal weighting.
- Add masked BCE/Dice loss.
- Add Torch control integration and route consistency loss.
- Record all objective settings in checkpoint and MLflow provenance.

### 16.4 Evaluation

- Add camera-only BEV metrics.
- Reuse existing route/junction and counterfactual metrics.
- Keep unique, unmodified validation samples.

## 17. Staged Implementation

1. Loss and integrator primitives:
   mean-normalized trajectory weights, differentiable control integration, and
   route-loss unit tests.
2. BEV head:
   camera-only branch, masked BCE/Dice, auxiliary-output contract, and gradient
   isolation.
3. Data contract:
   route-supervision artifact, layer validity, deterministic repack, and golden
   fixtures.
4. Sampling:
   pre-decode repeat transform, exposure audit, and worker determinism tests.
5. Flyte integration:
   config, checkpoint compatibility, MLflow metrics, and recovery workflow.
6. Smoke:
   small KITScenes subset forward/backward, gradient budget, checkpoint/resume,
   and one-epoch metric publication.
7. Full training:
   combined objective and matched no-route control.

Each stage is independently testable. The full run is not launched until the
fixed smoke batch passes the gradient budget and no-forbidden-gradient checks.

## 18. Test Plan

### 18.1 Unit tests

- `0.99` mean-normalized weights have mean one and final/first ratio
  `0.99^63`.
- `1.0` produces uniform mean-one weights.
- Torch integration matches NumPy integration.
- Raster coordinate conversion matches `NavigationRasterGeometry`.
- Out-of-bounds trajectories receive positive route loss.
- A trajectory on the selected corridor has lower loss than a wrong branch.
- The destination hinge does not reward passing the target terminal progress.
- Stationary samples do not receive route-heading loss.
- Empty valid sets return differentiable zero.
- BCE/Dice masks invalid maps, pixels, and unavailable channels.
- BEV segmentation gradients do not enter navigation or planner parameters.
- Repeat counts and exposure digests match the frozen policy.

### 18.2 Data tests

- The route-supervision artifact is deterministic and lossless.
- Distance is zero inside the corridor and positive outside.
- Direction vectors have unit norm wherever valid.
- Invalid routes contain no valid direction or destination target.
- No future ego trajectory is serialized in route supervision.
- Repack preserves the frozen sample UID and validation group inventories.

### 18.3 Integration tests

- Objective-v1 performs a complete optimizer step with all losses enabled.
- Every intended branch receives a finite non-zero gradient.
- A no-route batch skips route loss but still trains trajectory and BEV losses.
- Resume succeeds only with an identical objective config.
- Validation sample count and UID digest match the frozen manifest.
- MLflow receives all component losses and exposure metrics.

### 18.4 GPU smoke

- One epoch on a small KITScenes subset.
- No OOM, NaN, skipped optimizer steps, or DataLoader worker leaks.
- Segmentation and route losses decrease on an overfit micro-batch.
- Route-conditioned predictions change under a route swap.
- Inference output and latency are unchanged when the training-only head is
  disabled.

## 19. Rejected Alternatives

### 19.1 Segmentation after map fusion

Rejected because the model can copy `map_context`, producing a misleadingly
good auxiliary metric without improving camera features.

### 19.2 Binary-mask-only route sampling

Rejected as the sole route loss because gradients vanish when predicted points
are far from the route corridor.

### 19.3 Absolute destination terminal loss

Rejected because a visible destination may be beyond the 6.4-second reachable
horizon and can reward unsafe acceleration.

### 19.4 Validation oversampling

Rejected because it changes benchmark semantics and invalidates aggregate
metric comparison.

### 19.5 Automatic adaptive loss balancing

Deferred. It adds another stateful optimization algorithm and complicates
checkpoint reproducibility. The initial implementation uses fixed weights and a
pre-run gradient budget.

### 19.6 Route input to Reasoning

Rejected for this objective version. The initial #149 boundary remains
Reactive-only.

## 20. Acceptance Criteria

The design is implemented when:

1. all four components are independently configurable and jointly trainable;
2. BEV segmentation reads camera-only BEV features;
3. long-horizon weights are KITScenes-specific and mean-normalized;
4. training exposure is deterministic and validation remains unchanged;
5. route supervision contains no exact future ego trajectory and is never a
   model input;
6. route loss is differentiable, validity-gated, and coordinate-tested;
7. checkpoint resume rejects objective mismatches;
8. focused unit and integration tests pass;
9. a Flyte smoke completes with finite losses and intended gradients;
10. the combined and no-route-control full runs are evaluated on the frozen
    KITScenes benchmark and navigation slices.

## 21. References

- AutoE2E navigation input design:
  `Docs/navigation_input_design.md`.
- AutoE2E [#149](https://github.com/autowarefoundation/auto_e2e/issues/149).
- Bansal et al., [ChauffeurNet: Learning to Drive by Imitating the Best and
  Synthesizing the Worst](https://arxiv.org/abs/1812.03079).
- Philion and Fidler, [Lift, Splat, Shoot: Encoding Images From Arbitrary Camera
  Rigs by Implicitly Unprojecting to 3D](https://arxiv.org/abs/2008.05711).
- Li et al., [BEVFormer: Learning Bird's-Eye-View Representation from
  Multi-Camera Images via Spatiotemporal
  Transformers](https://arxiv.org/abs/2203.17270).
