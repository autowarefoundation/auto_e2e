# Design: Navigation-Aware Training Objectives for KITScenes

## Document Metadata

| Field | Value |
|---|---|
| Status | Implemented and smoke-validated; full comparison pending |
| Owner | riita10069 |
| Created | 2026-07-26 |
| Related issue | [#149](https://github.com/autowarefoundation/auto_e2e/issues/149) |
| Builds on | `Docs/navigation_input_design.md` |
| Initial dataset | KITScenes navigation v3.1 only |
| Route information boundary | Reactive branch only |

## 1. Executive Summary

The first route-conditioned KITScenes experiment proves that route pixels reach
the Reactive planner and receive non-zero gradients. It does not yet strongly
reward the model for using route information:

- the 64th control step receives only `0.95^63 = 0.0395` of the first step's
  trajectory-loss weight;
- most samples do not contain an imminent route choice;
- the imitation loss does not directly penalize leaving the selected route.

This design introduces one versioned KITScenes training objective with three
independently configurable components:

1. mean-normalized long-horizon trajectory weights with decay `0.99`;
2. deterministic junction- and maneuver-aware training resampling;
3. a differentiable selected-route consistency loss.

The initial combined training path is:

```text
camera images
    -> shared camera backbone
    -> camera FeatureFusion
    -> image_bev -------------------------------------+
                                                     |
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

Route-supervision fields are training targets, not model inputs. Route tensors
still do not enter the Reasoning branch.

All three components can be enabled together, but each has an explicit config
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

The three changes address different failure modes:

| Change | Failure mode |
|---|---|
| Long-horizon weighting | Route choices occur after the heavily weighted near-term controls |
| Junction-aware resampling | Straight lane following dominates the optimizer |
| Route consistency | Control imitation alone does not identify the selected lane sequence as a constraint |

Applying only a route loss leaves route-choice samples underrepresented.
Applying only resampling repeats the same weak objective. Applying only
long-horizon weighting does not encode selected-lane compliance. The combined
objective is therefore a coherent first performance experiment, while
independent switches retain scientific traceability.

## 3. Goals and Non-Goals

### 3.1 Goals

1. Increase supervision on the full 6.4-second control horizon.
2. Increase optimizer exposure to samples where route intent disambiguates the
   camera scene.
3. Penalize predicted trajectories that leave the selected lane corridor or
   take the wrong junction branch.
4. Preserve the #149 navigation ABI and Reactive-only route boundary.
5. Keep every new objective deterministic, masked by validity, and observable.
6. Allow a combined run and controlled per-component ablations from the same
   implementation.
7. Keep validation membership and distribution unchanged.

### 3.2 Non-goals

- Route input to `HorizonReasoningHead`.
- A separate route encoder in the first combined experiment.
- Counterfactual trajectory labels for routes that were not driven.
- Synthetic wrong-route training.
- Map/route dropout or localization perturbation.
- Changing the 64-step `(acceleration, curvature)` output ABI.
- BEV segmentation auxiliary loss or BEV encoder pretraining.
- Changing runtime navigation inputs.
- Replacing the existing early-stopping policy in the first comparison.
- Applying the new objective to L2D or NVIDIA PhysicalAI-AV.

## 4. Information Boundaries

There are three distinct contracts:

```text
model input:
  camera_tiles, map_context, route_mask, map_valid, route_valid, histories

train-only target:
  selected-route distance/direction fields, target controls

runtime output:
  64 x (acceleration, curvature)
```

The following rules are normative:

1. Selected-route distance and direction fields are consumed only by the
   training loss.
2. Exact future ego positions are never serialized as route inputs or route
   supervision.
3. Ground-truth controls may be integrated inside the loss for masking or a
   relative destination hinge because they are already the imitation target.
   They are never passed to the forward path used for inference.
4. Route-derived tensors remain inside the Reactive training path and never
   enter the Reasoning head, its teacher labels, or its cache.

## 5. Deferred BEV Pretraining

BEV segmentation is outside this experiment. The current packed KITScenes
contract has no dense dynamic-object BEV teacher. Lanelet2 provides static road
geometry, but supervising static map layers is not equivalent to learning
dynamic occupancy for vehicles, pedestrians, and other agents.

Camera-to-BEV pretraining should be specified separately in
[#17](https://github.com/autowarefoundation/auto_e2e/issues/17). PandaSet is the
proposed source dataset because it provides synchronized cameras, LiDAR,
3D cuboids, and point-cloud semantic labels from which static and dynamic BEV
targets can be constructed. That work must define its own class taxonomy,
projection contract, checkpoint transfer boundary, and dataset attribution.
It must not block or alter the KITScenes route-conditioned experiment defined
here.

## 6. Long-Horizon Imitation Loss

### 6.1 New KITScenes policy

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
and the route auxiliary loss.

### 6.2 Signal normalization

The existing KITScenes signal scales remain:

```text
acceleration_scale = 0.778
curvature_scale = 0.0350
```

Long-horizon weighting does not change the control units, Smooth L1 definition,
or output shape.

### 6.3 Configuration and compatibility

Temporal decay and normalization mode are checkpoint-defining configuration.
An objective-v1 checkpoint cannot resume from the current `0.95`,
non-normalized run. The first objective-v1 run starts from a fresh model
initialization with the same seed and frozen data split.

The implementation supports `0.99` and `1.0`, but the first combined run uses
`0.99`. Uniform `1.0` remains a later ablation.

## 7. Junction-Aware Training Resampling

### 7.1 Scope

Resampling applies only to the training iterator. Validation remains one
exposure per unique sample in the frozen scene-level split. Evaluation rejects
duplicate validation `sample_uid` values as it does today.

### 7.2 Deterministic exposure policy

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

### 7.3 WebDataset placement

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

### 7.4 Reproducibility

The following values are logged for every epoch:

- unique training sample count;
- effective exposure count;
- exposure counts by maneuver and junction status;
- repeat-policy version and configuration;
- digest of `(sample_uid, repeat_count)` sorted by sample UID.

With the same dataset, split, policy, and epoch seed, the exposure digest must
be identical.

### 7.5 Bias controls

Resampling changes the optimization distribution, not the benchmark
distribution. Aggregate metrics therefore continue to use the original
validation distribution. Junction and maneuver slices are reported separately.

No inverse-frequency weighting is applied on top of deterministic repetition in
the initial implementation. Combining both would make the effective objective
harder to audit.

## 8. Route-Supervision Artifact

### 8.1 Why an additional target is required

Sampling a binary corridor with `grid_sample` gives useful gradients only near
mask edges. It gives no direction toward the route when a predicted point is
far outside the corridor. A selected-route distance field provides a smooth,
metric target.

Map traffic-direction channels are not sufficient for route supervision at
intersections because they describe all mapped lanes, not only the selected
lane sequence.

### 8.2 Packed training target

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

### 8.3 Production input remains unchanged

`route_supervision.npz` is decoded into the training batch but is never passed
to `AutoE2E.forward`. The runtime model continues to receive only the binary
selected corridor and destination marker through `route_mask`.

Adding this member creates a new packed training-contract version. Existing
navigation scene artifacts can be deterministically repacked without rerunning
source ingest, Lanelet2 matching, or the Cosmos teacher.

## 9. Differentiable Control Integration

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

## 10. Route Consistency Loss

### 10.1 Eligibility

The route loss is active only when:

- `route_valid=true`;
- route quality passed the existing scene policy;
- the target integrated trajectory has at least 90% selected-corridor
  compliance over in-bounds points.

The last condition prevents a map-match or corridor-width error from forcing
the model away from the demonstrated behavior. It uses the target only as a
train-time loss mask and is never a model input. Eligibility and rejection
counts are logged.

### 10.2 Terms

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

### 10.3 Combined route term

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

## 11. Total Training Objective

The complete objective is:

```text
L_total =
    L_trajectory
  + lambda_route     * L_route
  + lambda_jepa      * L_jepa
  + lambda_reasoning * L_reasoning
```

Initial weights:

| Weight | Value |
|---|---:|
| `lambda_route` | 0.10 |
| `lambda_jepa` | 1.00 |
| `lambda_reasoning` | 0.05 |

The existing JEPA and Reasoning values remain unchanged.

### 11.1 Gradient budget gate

Numeric weights alone do not guarantee balanced gradients. Before a full run,
the training smoke test computes per-loss gradient norms on one fixed batch:

- `L_trajectory` into the planner and camera backbone;
- `L_route` into the planner;

`L_route` may not exceed the relevant trajectory gradient norm by more than
`2x` on the fixed smoke batch. If it does, its lambda is reduced and the frozen
config is updated before the full run. This is a stability gate, not an
automatic per-step gradient-balancing algorithm.

### 11.2 No silent loss activation

The route auxiliary has both an enable flag and a positive weight. Invalid
combinations fail:

- enabled route loss with `lambda_route <= 0`;
- route loss on a dataset without the supervision contract.

## 12. Configuration and Checkpoint Contract

The following fields are checkpoint-defining:

```text
training_objective_version = "kitscenes_navigation_objective_v1"

trajectory:
  temporal_decay: 0.99
  temporal_weight_normalization: mean_one

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
- route-loss eligible sample counts.

## 13. Training and Evaluation Procedure

### 13.1 Fresh training

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

### 13.2 Required runs

The minimum full comparison is:

| Run | Decay | Sampling | Route loss | Route input |
|---|---:|---:|---:|---:|
| Existing controlled route run | 0.95 | uniform | off | on |
| Objective-v1 combined | 0.99 | aware | on | on |
| Objective-v1 no-route control | 0.99 | aware | off | off |

The no-route control keeps long-horizon weighting and the same training
exposure distribution. It isolates the contribution of route input plus route
consistency from the general training improvements.

The implementation also supports leave-one-component-out runs without code
changes. Full leave-one-out training is required only if the combined result is
ambiguous or regresses.

### 13.3 Metrics

Retain all existing metrics and add:

- ADE and FDE at 1, 2, 3, and 6.4 seconds;
- junction/non-junction ADE and FDE;
- left/right/straight maneuver ADE and FDE;
- wrong-branch rate;
- selected-route compliance and outside distance;
- destination-approach error;
- route-swap counterfactual response;
- effective train exposure distribution;
- each route-loss term and eligibility rate.

Aggregate metrics use the original validation distribution. Oversampled
training metrics are never presented as benchmark metrics.

### 13.4 Success criteria

The combined objective is useful when:

1. aggregate ADE and FDE do not regress by more than 2% against the existing
   route-conditioned best checkpoint;
2. junction FDE or wrong-branch rate improves with a paired bootstrap 95%
   confidence interval excluding zero;
3. route-swap counterfactuals change the prediction in the selected direction;
4. lane-follow performance remains valid when route input is disabled;
5. the route auxiliary shows non-zero intended gradients and zero forbidden
   gradients.

The exact metric values and confidence intervals are reported even when the
hypothesis is not supported.

## 14. Failure Semantics

| Condition | Behavior |
|---|---|
| `route_valid=false` | Skip route loss and use repeat count 1 |
| Missing route-supervision member with route loss enabled | Fail before optimizer creation |
| Target route compliance below 90% | Skip route loss for that sample and count rejection |
| No eligible route samples in a batch | Differentiable zero route loss |
| No eligible route samples in an epoch | Fail the route-enabled KITScenes run |
| Predicted point outside raster | Apply explicit distance-to-bounds penalty |
| Destination not visible | Skip destination term |
| Route heading invalid or predicted speed below 1 m/s | Skip heading term |
| Non-finite auxiliary loss or metric | Fail before checkpoint upload |

## 15. Implementation Boundaries

### 15.1 Model

- Do not modify the Reasoning input contract.
- Keep the shared map/route navigation encoder for this experiment.

### 15.2 Data

- Add `route_supervision.npz`.
- Add deterministic raw-sample repeat transform.
- Repack from existing immutable scene navigation and reasoning artifacts.

### 15.3 Training

- Add mean-normalized temporal weighting.
- Add Torch control integration and route consistency loss.
- Record all objective settings in checkpoint and MLflow provenance.

### 15.4 Evaluation

- Reuse existing route/junction and counterfactual metrics.
- Keep unique, unmodified validation samples.

## 16. Staged Implementation

1. Loss and integrator primitives:
   mean-normalized trajectory weights, differentiable control integration, and
   route-loss unit tests.
2. Data contract:
   route-supervision artifact, deterministic repack, and golden
   fixtures.
3. Sampling:
   pre-decode repeat transform, exposure audit, and worker determinism tests.
4. Flyte integration:
   config, checkpoint compatibility, MLflow metrics, and recovery workflow.
5. Smoke:
   small KITScenes subset forward/backward, gradient budget, checkpoint/resume,
   and one-epoch metric publication.
6. Full training:
   combined objective and matched no-route control.

Each stage is independently testable. The full run is not launched until the
fixed smoke batch passes the gradient budget and no-forbidden-gradient checks.

## 17. Test Plan

### 17.1 Unit tests

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
- Repeat counts and exposure digests match the frozen policy.

### 17.2 Data tests

- The route-supervision artifact is deterministic and lossless.
- Distance is zero inside the corridor and positive outside.
- Direction vectors have unit norm wherever valid.
- Invalid routes contain no valid direction or destination target.
- No future ego trajectory is serialized in route supervision.
- Repack preserves the frozen sample UID and validation group inventories.

### 17.3 Integration tests

- Objective-v1 performs a complete optimizer step with all losses enabled.
- Every intended branch receives a finite non-zero gradient.
- A no-route batch skips route loss but still trains trajectory loss.
- Resume succeeds only with an identical objective config.
- Validation sample count and UID digest match the frozen manifest.
- MLflow receives all component losses and exposure metrics.

### 17.4 GPU smoke

- One epoch on a small KITScenes subset.
- No OOM, NaN, skipped optimizer steps, or DataLoader worker leaks.
- Route loss decreases on an overfit micro-batch.
- Route-conditioned predictions change under a route swap.

## 18. Rejected Alternatives

### 18.1 Binary-mask-only route sampling

Rejected as the sole route loss because gradients vanish when predicted points
are far from the route corridor.

### 18.2 Absolute destination terminal loss

Rejected because a visible destination may be beyond the 6.4-second reachable
horizon and can reward unsafe acceleration.

### 18.3 Validation oversampling

Rejected because it changes benchmark semantics and invalidates aggregate
metric comparison.

### 18.4 Automatic adaptive loss balancing

Deferred. It adds another stateful optimization algorithm and complicates
checkpoint reproducibility. The initial implementation uses fixed weights and a
pre-run gradient budget.

### 18.5 Route input to Reasoning

Rejected for this objective version. The initial #149 boundary remains
Reactive-only.

## 19. Implementation and Smoke Results

The implementation is on branch `feat/navigation-training-objectives-v1` at
commit `796bc642dfe84853da3c94367041ebd9638a1807`. It includes:

- KITScenes temporal decay `0.99` with mean-one normalization;
- packed schema v6 and loss-only `route_supervision.npz`;
- deterministic `navigation_repeat_v1` exposure;
- differentiable route corridor, branch, destination, and heading losses;
- objective identity and resume validation in Flyte;
- MLflow component, eligibility, gradient, and horizon metrics;
- KITScenes dataset version v3.1 and audited recovery subset support.

### 19.1 Test results

Local focused tests passed with `93 passed, 2 skipped`. The full local suite
passed with `700 passed, 22 skipped` except for one environment-only `pyproj`
import failure. Installing `pyproj` into an isolated target made the affected
benchmark group pass with `20 passed`.

On the EC2 development host, the Python 3.12 and Flytekit 1.14.9 environment
passed:

- workflow tests: `53 passed`;
- combined focused tests: `149 passed, 1 skipped`.

The L40S route-loss micro-overfit reduced route loss from `4.8912873` to
`0.0000927427`, produced finite gradients, and allocated approximately
`1.3 MB` at peak for the isolated loss.

### 19.2 Flyte GPU smoke

The immutable one-epoch smoke used:

| Field | Value |
|---|---|
| Flyte execution | `azm4tbtmlwm6z79cjq9d` |
| MLflow run | `b1ac46a839cc4cbb93e37202f95a972f` |
| Source commit | `796bc642dfe84853da3c94367041ebd9638a1807` |
| Dataset | KITScenes v3.1, audited 10-partition subset |
| Training/eval image | `sha256:4ad61aaf3fc9e25cb0b681b1fc05a2268fe27493283a9eadc6f78cf6ebd7710c` |
| Data-prep image | `sha256:43898b5773d2bde24894dd4b4ca7591c6d8f51ff48fee465dfb1eabb0322e223` |
| Training objective | `kitscenes_navigation_objective_v1` |
| Route loss weight | `0.10` |
| Result | Succeeded, train/eval restart count 0 |

The training exposure contained 462 unique samples and 1,289 effective
exposures. Its deterministic exposure digest was
`2013335b9d95dad39fc62f732ced09f6bfb7cfa5cb7067e0b76e15d373523542`.
Of 1,289 route candidates, 849 passed the target-compliance gate and 440 were
rejected.

The first fixed-batch route-to-trajectory planner gradient ratio was `1.1887`,
below the `2.0` budget. The route-input gradient became non-zero at optimizer
step 2 (`4.09e-7`), as did the navigation-encoder gradient (`8.91e-4`).

Epoch 1 metrics were:

| Metric | Value |
|---|---:|
| Total loss | 1.18164 |
| Trajectory loss | 0.51993 |
| Route loss | 0.72266 |
| Weighted route loss | 0.07227 |
| Corridor / branch / destination / heading | 0.05937 / 0.21795 / 0.42870 / 0.05218 |
| Validation ADE / FDE at 1 s | 0.08482 / 0.16200 m |
| Validation ADE / FDE at 2 s | 0.23878 / 0.75724 m |
| Validation ADE / FDE at 3 s | 0.82374 / 3.43980 m |
| Validation ADE / FDE at 6.4 s | 8.19002 / 27.53298 m |

The checkpoint SHA-256 was
`c453ecbb97d67483d43bfe444e5138feb357fc5c049bbff0d011fc8f03aecf54`.
The one-epoch benchmark quality gate did not pass, as expected for a smoke
run; the smoke acceptance condition is finite end-to-end optimization,
checkpointing, and evaluation rather than final model quality.

No OOM, NaN, skipped optimizer step, or container restart occurred. A Flyte
Propeller metrics-counter panic occurred while reconciling the repack array;
the controller replica recovered, all repack tasks completed, and the workflow
reached the terminal succeeded phase. The model and data tasks did not fail.

### 19.3 Remaining experiment

Acceptance items 1 through 8 are implemented and smoke-validated. The combined
20-epoch run, matched no-route control, and frozen full-benchmark comparison in
acceptance item 9 remain intentionally unlaunched. They are performance
experiments, not implementation or smoke blockers.

## 20. Acceptance Criteria

The design is implemented when:

1. all three components are independently configurable and jointly trainable;
2. long-horizon weights are KITScenes-specific and mean-normalized;
3. training exposure is deterministic and validation remains unchanged;
4. route supervision contains no exact future ego trajectory and is never a
   model input;
5. route loss is differentiable, validity-gated, and coordinate-tested;
6. checkpoint resume rejects objective mismatches;
7. focused unit and integration tests pass;
8. a Flyte smoke completes with finite losses and intended gradients;
9. the combined and no-route-control full runs are evaluated on the frozen
   KITScenes benchmark and navigation slices.

## 21. References

- AutoE2E navigation input design:
  `Docs/navigation_input_design.md`.
- AutoE2E [#149](https://github.com/autowarefoundation/auto_e2e/issues/149).
- AutoE2E [#17](https://github.com/autowarefoundation/auto_e2e/issues/17).
- [PandaSet](https://pandaset.org/), provided by Hesai and Scale AI.
- Bansal et al., [ChauffeurNet: Learning to Drive by Imitating the Best and
  Synthesizing the Worst](https://arxiv.org/abs/1812.03079).
