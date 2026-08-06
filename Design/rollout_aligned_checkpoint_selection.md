# Design: Rollout-Aligned Loss and Scene-Balanced Checkpoint Selection

## Document Metadata

| Field | Value |
|---|---|
| Status | Accepted for implementation with the logged-XY fallback |
| Owner | riita10069 |
| Created | 2026-07-26 |
| Initial dataset | KITScenes navigation v3.2 |
| Builds on | `Design/navigation_training_objectives.md` |
| Training path | Flyte `train_il` only |
| Model output | 64 steps of `(longitudinal acceleration, curvature)` at 10 Hz |

## 1. Executive Summary

AutoE2E is trained in control space but is evaluated primarily after those
controls are integrated into an XY trajectory. The current action loss and
ADE-based checkpoint selection do not fully represent that evaluation
contract:

1. a small control error can accumulate into a large long-horizon position
   error;
2. a checkpoint can improve average ADE while regressing safety, comfort, or
   selected-route behavior;
3. validation samples are averaged uniformly, so long scenes have more
   influence than short scenes.

This design makes two bounded changes:

1. add a differentiable XY rollout loss and target-relative comfort/map
   constraints while retaining the existing normalized action loss;
2. select checkpoints with a weighted composite score that combines natural
   and scene-balanced trajectory, comfort, map-safety, and navigation metrics.

The planner objective introduced by this design is:

```text
L_planner =
    L_action
  + 0.50 * L_rollout
  + 0.05 * L_constraint
```

The World Model and its JEPA objective remain enabled because they are a core
part of AutoE2E representation learning. The complete training objective is:

```text
L_total =
    L_planner
  + lambda_jepa * L_jepa
  + lambda_reasoning * L_reasoning_existing
```

This PR changes only `L_planner`. The matched control and treatment freeze the
same World Model, JEPA weight, Reasoning enable flag, and Reasoning weight. The
initial matched configuration keeps `enable_world_model=true` and the existing
JEPA weight. It neither adds a new Cosmos-derived loss nor changes the existing
Reasoning objective. The model architecture, route input boundary, and 64-step
control output remain unchanged.

The reconstruction audit found that integrated target controls do not meet the
initial target-quality thresholds on the full validation split. This design
therefore uses logged pose-grounded XY directly as the position target for
`L_rollout` and the target-relative map term. Target controls remain the
teacher for `L_action` and target-relative comfort. This follows the preflight
fallback instead of training against a known-inconsistent position target.

## 2. Motivation

### 2.1 Current optimization and evaluation paths

The current planner path is:

```text
camera + semantic map + selected route
                  |
                  v
      64 x (acceleration, curvature)
                  |
          normalized Smooth L1
```

Open-loop evaluation adds another operation:

```text
64 x (acceleration, curvature)
                  |
        semi-implicit unicycle rollout
                  |
                  v
             64 x (x, y)
                  |
             ADE / FDE
```

Action-space supervision remains necessary because different control sequences
can produce similar positions and because control quality matters even when
position error is small. It is not sufficient by itself because integration
amplifies systematic acceleration and curvature errors over time.

### 2.2 Scene imbalance

The frozen validation split is scene-disjoint, but the current aggregate gives
one equal vote to every sample. A scene with 1,000 valid windows therefore has
ten times the influence of a scene with 100 windows. Natural aggregation is
still useful because it describes the observed sample distribution. It should
not be the only view used to choose a checkpoint.

This design records both:

- natural metrics: every validation sample has equal weight;
- scene-balanced metrics: every `split_group_uid` has equal weight.

### 2.3 Multi-objective selection

A checkpoint should not be selected from ADE or FDE alone when training also
optimizes comfort and map/navigation behavior. The selector therefore:

1. computes every required metric from the same validation records;
2. converts metrics with different units into bounded utilities;
3. ranks checkpoints by one versioned weighted sum;
4. logs every component so the trade-off remains visible.

There are no per-metric hard gates in this research selector. Production
promotion policy is a separate concern and is outside this design.

## 3. Goals and Non-Goals

### 3.1 Goals

1. Align planner optimization with the XY rollout used for ADE and FDE.
2. Preserve the existing normalized action loss as the control-profile teacher.
3. Penalize comfort and map violations only when they exceed the demonstrated
   target behavior.
4. Prevent long scenes from dominating checkpoint selection.
5. Select checkpoints from trajectory, comfort, map-safety, and navigation
   evidence rather than ADE/FDE alone.
6. Make metric availability, score policy, and selector state reproducible
   across resume.
7. Keep the first experiment scientifically attributable to the loss and
   selection changes in this document.

### 3.2 Non-goals

- Cosmos relabeling.
- New or reweighted Cosmos-derived training supervision.
- Cosmos-derived sample weights or balancing.
- Semantic buckets.
- Weighted sampling.
- Tail, top-k, or CVaR sample mining.
- Speed or heading losses.
- A model architecture change.
- A new route encoder or route input to the Reasoning branch.
- BEV segmentation auxiliary loss.
- JEPA or Reasoning loss tuning.
- Closed-loop acceptance.

Cosmos-assisted balancing is a separate hypothesis. It should be considered
only after this loss-alignment experiment establishes the remaining
scene-balanced error distribution.

## 4. Experiment Contract

The initial experiment is a paired, three-seed comparison on one immutable
dataset snapshot:

| Arm | Planner loss | Checkpoint selection |
|---|---|---|
| A: `rollout_aligned_control_v1` | Current normalized action loss | Weighted composite score |
| B: `rollout_aligned_planner_v1` | Action + logged-XY rollout + constraint | Weighted composite score |

Both arms use:

- identical model architecture and initialization policy;
- identical dataset and frozen scene split;
- identical seeds, batch size, optimizer, and maximum epochs;
- identical camera, map, and route inputs;
- identical World Model and JEPA settings, with the World Model enabled;
- identical Reasoning enable flag, labels, and weight inherited from Arm A;
- junction-aware repeat disabled in both arms;
- no weighted sampler or hard-example mining.

The legacy `RouteConsistencyLoss` is disabled in both arms. Arm A uses the
existing normalized action objective as its planner loss. Arm B replaces the
legacy route objective with the rollout and target-relative constraint terms
defined here; it does not add them on top of the legacy objective.

Both arms save every epoch checkpoint. The current ADE/FDE selector is applied
post hoc to both histories so selector effects can be measured without extra
training. The primary paired comparison uses the same composite selector in
both arms, isolating the planner-loss change.

The auxiliary objectives are controlled variables, not treatment variables.
Any JEPA or Reasoning configuration difference between paired arms invalidates
the comparison. New Cosmos labels, a new Reasoning loss, or auxiliary-loss
weight tuning requires a separate policy version and experiment.

## 5. Preflight Reconstruction Audit

### 5.1 Status and purpose

The audit is a completed spike task and remains an immutable prerequisite, not
Step 1 of the implementation PR. It determined whether integrated target
controls were a valid proxy for logged future motion and selected the target
source used below.

The implementation PR must link an immutable audit artifact and record an
explicit Go or No-Go decision.

### 5.2 Inputs

For each audited validation sample:

- target acceleration and curvature, `[64, 2]`;
- initial speed from the final causal egomotion-history step;
- logged current pose;
- 64 logged future positions at 10 Hz;
- `sample_uid`;
- `split_group_uid`.

For KITScenes, the expected packed members are `pose.npy` and `gps.npy`.
`gps.npy` contains the current point followed by 64 future points. The audit
must first prove that these members are present and finite for every sample in
the audited snapshot. Optional support in the generic shard schema is not
evidence that a particular snapshot contains the data.

The logged GPS trajectory is converted to current ego FLU coordinates with the
same pose-grounded conversion used by the KITScenes benchmark. Rollout step
`t=0` is compared with logged future point 1, not with the current point.

### 5.3 Measurements

For each sample, integrate the target controls with the exact rollout in
Section 6 and compute:

```text
target rollout ADE@3s
target rollout FDE@3s
target rollout ADE@6.4s (noncanonical target-quality diagnostic)
target rollout FDE@6.4s (noncanonical target-quality diagnostic)
```

The audit report contains:

- sample-level records;
- natural p50, p90, p95, mean, and maximum;
- scene-level mean and p95;
- the ten scenes with the highest FDE at each horizon;
- error versus horizon plots or tables;
- missing/non-finite sample counts.

### 5.4 Provenance

The audit artifact records:

```text
dataset name and version
source revision
packed contract digest
validation split ID
validation split_group_uid digest
sorted sample_uid digest
sample count
audit code revision
rollout policy version
artifact SHA-256
```

The audit and training experiment must use the same validation snapshot
identity. A digest mismatch is a hard error.

### 5.5 Initial decision criteria

The initial reference thresholds are:

```text
p95 FDE@3s   <= 1.0 m
p95 FDE@6.4s <= 2.0 m
```

These are target-quality criteria, not model-evaluation or checkpoint gates.
The 6.4-second audit verifies the full training target and is never published
as canonical model ADE/FDE. Exceeding either threshold requires design review
and prohibits using integrated target controls as the position teacher. It
does not prohibit a versioned fallback to logged XY.

The Go decision must include a written rationale and the selected position
target source. A No-Go blocks training. A Go with `position_target=logged_xy`
requires the training task to consume the exact pose-grounded trajectory bound
to the audited validation identity.

### 5.6 Full-snapshot result and decision

The immutable full-snapshot audit used 3,820 validation samples from 40 scenes:

```text
report SHA-256: 71211cc9ff009dbd476c7a94601615e520a73d6f17785572475af8020154b983
p95 FDE@3s:     1.083 m
p95 FDE@6.4s:   2.281 m
mean FDE@6.4s:  1.269 m
missing/non-finite samples: 0 / 0
```

The mismatch was not isolated: 373 samples in 17 scenes exceeded 1.0 m at
3 seconds, and 510 samples in 26 scenes exceeded 2.0 m at 6.4 seconds. Natural
p95 error grew approximately monotonically from 0.042 m at 0.1 seconds to
2.281 m at 6.4 seconds. The current full-run model FDE is approximately
10.18 m, so the target mismatch is smaller than current model error but is not
negligible relative to the intended improvement.

Decision:

```text
integrated target controls as position target: No-Go
packed logged XY as position target:          Go for implementation and Smoke
target controls for action and comfort:       retained
```

The failed thresholds remain recorded in checkpoint metadata. Training still
requires the immutable audit, exact dataset/split digests, an explicit Go
decision, and a written rationale. It must not reinterpret `thresholds_pass`
as true.

## 6. Shared Rollout Contract

### 6.1 Source of truth

Training and evaluation use one shared PyTorch rollout implementation.
The NumPy evaluator remains only as a parity implementation and must not define
different motion semantics.

For timestep `t`:

\[
v_t=\max(0,v_{t-1}+a_t\Delta t)
\]

\[
\psi_t=\psi_{t-1}+v_t\kappa_t\Delta t
\]

\[
x_t=x_{t-1}+v_t\cos(\psi_t)\Delta t
\]

\[
y_t=y_{t-1}+v_t\sin(\psi_t)\Delta t
\]

Initial state:

```text
x[-1] = 0
y[-1] = 0
heading[-1] = 0
speed[-1] = final causal speed from egomotion_history
dt = 0.1 seconds
```

Predicted controls are integrated from this initial state. Target controls are
also integrated for target speed and heading used by comfort and footprint
orientation, but their integrated XY is diagnostic only. Position supervision
uses the packed logged future XY in the same ego FLU frame and timestep
alignment as evaluation.

### 6.2 Numerical contract

- Inputs accept `[B, T, 2]` or the existing flattened `[B, 2T]` control shape.
- Integration always runs in `torch.float32`.
- The rollout executes outside AMP autocast even when the surrounding training
  step uses AMP.
- `torch.clamp_min(speed, 0.0)` implements the non-negative-speed constraint at
  every timestep. The recurrence uses functional tensor reassignment and list
  accumulation rather than in-place state updates.
- State updates are functional; no in-place tensor updates are permitted.
- Controls, initial speed, and any supplied initial state must be finite.
  Non-finite input fails before integration.
- `dt` must be finite and positive.
- The output contains float32 positions, headings, and speeds.

The existing differentiable route rollout currently promotes float32 controls
to float64. It must be replaced or refactored to call the shared float32
implementation.

### 6.3 Known gradient limitation

When the per-step unclamped speed is below zero, `clamp_min(0)` removes the
local gradient from acceleration through speed. This is accepted for the first
implementation:

- no custom backward is introduced;
- `L_action` still supervises acceleration during stopped intervals;
- a unit test documents the zero-gradient region.

## 7. Top-Level Training Loss

For sample `i`, the planner objective is:

\[
L_i^{planner} =
L_i^{action}
+0.5L_i^{rollout}
+0.05L_i^{constraint}
\]

The planner batch loss is the ordinary sample mean:

\[
L_{planner}=\frac{1}{B}\sum_i L_i^{planner}
\]

No sample weighting, repeat weighting, or hard-example selection is applied.
Loss lambdas and all subterm definitions are checkpoint-defining policy.

The existing auxiliary objectives are then added without modification:

\[
L_{total}
=
L_{planner}
+\lambda_{jepa}L_{jepa}
+\lambda_{reasoning}L_{reasoning}^{existing}
\]

`enable_world_model=true` is required for the initial matched experiment.
`lambda_jepa`, the Reasoning enable flag, and `lambda_reasoning` are copied from
Arm A and frozen for Arm B. When existing Reasoning supervision is disabled,
its term is absent in both arms; this design does not force it off.

Keeping `lambda_jepa` fixed does not by itself guarantee that the World Model
retains the same influence after planner terms are added. Before the full run,
a fixed smoke batch records isolated gradient norms from action, rollout, and
constraint losses into planner parameters, and from weighted JEPA into the
World Model's trainable parameters. JEPA intentionally does not update the
shared image backbone in the current architecture because the World Model
frame encoder detaches that shared feature map. The treatment must show finite
non-zero JEPA gradients in World Model parameters; no automatic gradient
balancing is added.

The action term is dimensionless after signal normalization. Rollout and map
terms contain metric quantities. The fixed lambdas define their numerical
trade-off; they are not interpreted as a dimensionally homogeneous physical
sum.

## 8. Action Loss

The current normalized Smooth L1 loss and current dataset-policy temporal
weights are retained.

Let `s_a` and `s_kappa` be the existing KITScenes signal scales, and let `w_t`
be the existing temporal weights. For the navigation objective v1 base, this
means decay `0.99` with mean-one normalization. The treatment must not change
these values relative to its paired control.

\[
L_i^{action}
=
\frac{1}{T}
\sum_t w_t
\frac{1}{2}
\left[
\rho\left(\frac{\hat a_t-a_t}{s_a}\right)
+
\rho\left(\frac{\hat\kappa_t-\kappa_t}{s_\kappa}\right)
\right]
\]

where PyTorch Smooth L1 with beta 1 is:

\[
\rho(z)=
\begin{cases}
0.5z^2 & |z| < 1\\
|z|-0.5 & \text{otherwise}
\end{cases}
\]

This term is retained because:

- it directly supervises the control profile;
- multiple action sequences can produce similar XY rollouts;
- it preserves gradients in stopped intervals affected by the speed clamp;
- changing action loss in the same experiment would obscure attribution.

## 9. Rollout Position Loss

Let `p_hat[t]` be the predicted XY position from the shared rollout and
`p_logged[t]` be the packed pose-grounded future XY position. Define Euclidean
position error:

\[
e_t=\lVert\hat p_t-p_t^{logged}\rVert_2
\]

The scalar Huber function with delta `1.0 m` is:

\[
H(e;1)=
\begin{cases}
0.5e^2 & e\le1\\
e-0.5 & e>1
\end{cases}
\]

The path term is:

\[
L_i^{path}=\frac{1}{T}\sum_t H(e_t;1)
\]

The final-position term is:

\[
L_i^{final}=H(e_T;1)
\]

The rollout term is:

\[
L_i^{rollout}
=0.75L_i^{path}+0.25L_i^{final}
\]

Reduction is intentionally unweighted over time. `path` aligns with ADE and
`final` keeps an explicit FDE contribution. Multiple horizon anchors, temporal
decay, and horizon-dependent scales are not introduced in this PR.

## 10. Target-Relative Constraint Loss

The constraint contains only:

1. comfort;
2. selected-route and drivable-area compliance.

For each sample, compute all available top-level terms and take their arithmetic
mean:

```text
available = [comfort]
if route or drivable supervision is available:
    available += [map]
L_constraint = mean(available)
```

Comfort is always available for finite controls. The map term may be absent.
An unavailable optional term contributes neither a zero nor a denominator.
When no route/drivable subterm is available, `L_map` is reported as finite zero
for logging, while `L_constraint` contains comfort only.

### 10.1 Comfort quantities

Only longitudinal jerk and lateral acceleration are used:

\[
j_t=\frac{a_t-a_{t-1}}{\Delta t},\quad t=1,\ldots,T-1
\]

\[
a_t^{lat}=v_t^2\kappa_t
\]

The first jerk step does not invent an `a[-1]`; jerk is computed over the 63
adjacent action differences, matching the existing evaluator convention.

Thresholds:

```text
longitudinal jerk: 4.13 m/s^3
lateral acceleration: 4.89 m/s^2
```

For quantity `z` and threshold `c`, define:

\[
q(z;c)=\operatorname{ReLU}\left(\frac{|z|}{c}-1\right)^2
\]

Prediction and target are compared at the trajectory level:

\[
Q^{pred}=\max_t q(\hat z_t;c)
\]

\[
Q^{target}=\max_t q(z_t;c)
\]

\[
e^{comfort}
=\operatorname{ReLU}
\left(
Q^{pred}-\operatorname{stopgrad}(Q^{target})
\right)
\]

\[
L_i^{comfort}
=\frac{1}{2}
\left(e_i^{jerk}+e_i^{lat}\right)
\]

This comparison deliberately ignores timing shifts. If prediction and target
have the same peak normalized violation at different timesteps, the additional
comfort loss is zero.

### 10.2 Ego footprint

Map compliance is evaluated with four oriented footprint corners, not the ego
center. For center `(x_t, y_t)`, heading `psi_t`, length `l`, and width `w`,
the local offsets are:

```text
(+l/2, +w/2)
(+l/2, -w/2)
(-l/2, +w/2)
(-l/2, -w/2)
```

The initial policy uses a versioned `4.8 m x 2.0 m` proxy footprint for both
prediction and target. The footprint source and dimensions are part of
checkpoint metadata. If authoritative KITScenes vehicle dimensions become
available before implementation, the policy value may be corrected once,
before the audit and A/B snapshots are frozen. It may not change between arms
or during a run.

### 10.3 Outside-distance fields

Each valid region is represented by a metric outside-distance field:

```text
0 inside the valid region
Euclidean distance in metres outside the valid region
```

Required fields:

- selected-route corridor outside distance;
- drivable-area outside distance.

The existing `route_supervision.npz::distance_to_corridor_m` supplies the first
field. KITScenes preprocessing must add a loss-only drivable-area distance
field derived from the canonical semantic drivable mask. It is not a model
input and does not change inference.

Distance fields are generated offline and losslessly stored as float32. Runtime
sampling uses bilinear `grid_sample(align_corners=False)`. A point beyond the
raster receives the sampled boundary distance plus its differentiable Euclidean
distance beyond raster bounds; leaving the crop must not appear as zero
distance.

For region `r`, timestep `t`, and four transformed corners `c`:

\[
d_{t,r}=\max_c D_r(c_{t})
\]

The maximum means the entire rectangular footprint must fit inside the region.
Prediction and target use the same field, geometry, and footprint. Target
footprint positions use logged XY; target footprint headings use the
target-control rollout because KITScenes does not provide future orientation.

### 10.4 Target-relative map excess

For each region:

\[
e_{t,r}^{map}
=
\operatorname{ReLU}
\left(
d_{t,r}^{pred}
-
\operatorname{stopgrad}(d_{t,r}^{target})
-
\tau_{raster}
\right)
\]

\[
L_{i,r}^{map}
=\frac{1}{T}\sum_t e_{t,r}^{map}
\]

The tolerance is resolution-aware:

\[
\tau_{raster}=0.5\cdot meters\_per\_pixel
\]

At the initial `1.0 m/px` geometry this is `0.5 m`. It absorbs the half-pixel
quantization and bilinear-sampling floor. It does not permit a fixed absolute
map violation; it permits only a bounded difference relative to the target.

Availability:

- drivable term is active only when `map_valid=true` and its field is present;
- route term is active only when `route_valid=true` and its field is present;
- when both are active, `L_map` is their arithmetic mean;
- when only one is active, `L_map` equals that term;
- unavailable terms are masked before reduction;
- an entirely unavailable map term returns differentiable finite zero for
  logging.

Centerline distance is not used. The selected corridor, not the route
centerline, defines route compliance.

### 10.5 Relationship to Navigation Objective v1

This objective supersedes the combined `RouteConsistencyLoss` from navigation
objective v1. The following v1 terms are not added to `L_planner`:

```text
absolute corridor loss
late-junction branch loss
destination-progress loss
route-heading loss
```

Their relevant supervision is covered as follows:

- path and final rollout losses supervise the demonstrated branch and terminal
  progress in XY;
- target-relative selected-corridor distance preserves a direct route gradient;
- target-relative drivable distance adds the map-safety signal missing from v1;
- action loss continues to supervise acceleration and curvature profiles.

The replacement is intentional. Keeping both objectives would count selected
route deviation in the legacy corridor/branch terms, the new route-footprint
term, and the XY rollout term.

Implementation may extract and reuse v1's differentiable rollout, ego-to-grid,
distance sampling, and out-of-raster utilities. The legacy combined loss,
branch/destination/heading term weights, and `enable_route_consistency`
activation are removed from the objective-v2 training path. No legacy-loss
ablation is required. Objective-v1 checkpoints remain valid artifacts but
cannot resume as objective-v2 runs.

## 11. Data and Supervision Contract

### 11.1 Required sample identity

Validation requires:

```text
sample_uid
split_group_uid
```

Missing, empty, or duplicate `sample_uid` values fail validation. Missing or
empty `split_group_uid` values also fail validation; hash-bucket fallback is not
allowed for this checkpoint-selection policy. The packed `meta.json` already
contains `split_group_uid`; the decoder and collator must expose it explicitly
in each validation batch.

### 11.2 Loss-only map supervision

The packed training contract adds or versions a loss-only artifact containing:

```text
drivable_outside_distance_m: [H, W] float32
available:                    scalar uint8
geometry_id:                  metadata
```

Its geometry ID must equal the semantic map and route-supervision geometry.
The artifact is generated deterministically in Flyte preprocessing without an
external map API.

The selected-route distance field remains in `route_supervision.npz`. A
contract version bump is required because training must fail rather than
silently disable a requested constraint when the field is missing.

### 11.3 No inference contract change

The model continues to receive:

```text
camera views
map_context
route_mask
map_valid
route_valid
visual history
egomotion history
projection
```

Distance fields and logged future poses are train/evaluation-only values. They
are never passed to `AutoE2E.forward` and do not enter the Reasoning branch.
The training task converts packed `pose.npy` and `gps.npy` with the same
pose-grounded transform used by evaluation and supplies float32 `[B,64,2]`
logged XY only to the planner loss.

## 12. Per-Sample Validation Metrics

Checkpoint selection starts from per-sample records. Aggregates must not be
computed from already averaged batches.

### 12.1 Trajectory

The predicted controls are rolled out and compared with logged pose-grounded
future XY:

```text
ADE@3s
FDE@3s
```

These are the canonical `ade` and `fde` values throughout checkpoint history,
best-epoch selection, MLflow, and the model registry. Both are scene-balanced
after per-sample records are aggregated. Integrated target-control XY remains
a reconstruction diagnostic and is not substituted for logged XY.

All selector metrics in Sections 12.2 through 12.6 use the same first 30
steps. The model and planner loss retain their 64-step prediction and training
horizon; only validation and checkpoint selection use the fixed 3-second
horizon.

### 12.2 Comfort excess

Per-sample comfort excess is the same target-relative trajectory-level quantity
used by the loss:

```text
comfort_excess = 0.5 * (jerk_peak_excess + lateral_accel_peak_excess)
```

### 12.3 Off-road excess

For each timestep, mark whether any footprint corner is outside the drivable
region. Let `r_pred` and `r_target` be the resulting fractions over 30 steps:

\[
offroad\_excess=\max(0,r^{pred}-r^{target})
\]

This selector metric is dimensionless and in `[0, 1]`. The training map loss
uses metric outside distance; the selector uses an interpretable violation
fraction.

### 12.4 Route gap

Let `q_pred` and `q_target` be the fractions of timesteps for which all
footprint corners are inside the selected-route corridor:

\[
route\_gap=\max(0,q^{target}-q^{pred})
\]

It is defined only for route-valid samples.

The inside test uses the same half-pixel metric tolerance as the training map
loss. Before component availability is frozen, the selector fails if target
route compliance is at most `0.05` or target off-road rate is at least `0.95`;
these conditions indicate a saturated or inconsistent raster contract rather
than meaningful navigation or safety evidence.

### 12.5 Wrong-branch excess

For an eligible junction, `b` is a binary wrong-branch indicator under the
existing selected-route evaluation contract:

\[
wrong\_branch\_excess=\max(0,b^{pred}-b^{target})
\]

It is defined only when the target provides valid branch evidence.

### 12.6 Destination error

For a visible destination:

\[
destination\_error
=
\left|
\lVert \hat p_T-d\rVert_2
-
\lVert p_T-d\rVert_2
\right|
\]

Here, `T` is validation step 30, not the 64-step prediction endpoint.

It is reported in metres and remains target-relative.

## 13. Validation Aggregation

### 13.1 Natural aggregate

For metric `m` with eligible sample set `S_m`:

\[
M_{natural}=\frac{1}{|S_m|}\sum_{i\in S_m}m_i
\]

Record at minimum:

```text
natural ADE@3s
natural FDE@3s
natural off-road excess
natural comfort excess
natural route gap
natural wrong-branch excess
natural destination error
eligible sample count for every metric
```

### 13.2 Scene-balanced aggregate

For metric `m`, first group eligible records by `split_group_uid`. Let `S_g,m`
be the eligible samples in scene `g`:

\[
\bar m_g=\frac{1}{|S_{g,m}|}\sum_{i\in S_{g,m}}m_i
\]

\[
M_{scene}=\frac{1}{G_m}\sum_g\bar m_g
\]

Only scenes with at least one eligible record for that metric enter its
aggregate. Record:

```text
scene-balanced ADE@3s
scene-balanced FDE@3s
scene-balanced off-road excess
scene-balanced comfort excess
scene-balanced route gap
scene-balanced wrong-branch excess
scene-balanced destination error
eligible scene count for every metric
```

For each metric's scene-mean distribution also record:

```text
scene mean
scene p50
scene p90
```

`scene mean` is the scene-balanced aggregate. Percentiles use
`numpy.quantile(method="linear")` on scene means sorted by
`split_group_uid`; this fixes the quantile convention and deterministic input
order.

Adding duplicate samples to one scene without changing that scene's mean must
not change the scene-balanced aggregate.

## 14. Component Availability and Score Policy

### 14.1 Frozen availability

Component availability is computed from the immutable validation snapshot
before epoch 1 and then frozen:

- trajectory `D`: always required;
- comfort `C`: always required;
- map safety `S`: available when semantic-map supervision has finite eligible
  samples;
- route navigation: available when route-valid sample count is at least 50;
- wrong branch: available when eligible junction sample count is at least 20;
- destination: available when at least one finite destination record exists.

Coverage counts and exclusion reasons are stored in checkpoint metadata.
Coverage changes during an epoch are a contract error, not a reason to
silently reweight the score.

If route navigation is unavailable, the entire navigation component `N` is
excluded. Within an available navigation component, unavailable wrong-branch
or destination subcomponents are removed and the remaining navigation weights
are renormalized. This prevents missing evidence from being treated as perfect
behavior.

### 14.2 Versioned score policy

The selector policy stores:

```text
selector policy version
validation snapshot identity and digests
component availability and coverage counts
metric-to-utility scales
component weights after availability renormalization
min_delta
```

These values are fixed for the run and saved in every checkpoint. Changing a
utility scale or weight creates a new selector policy version.
The bounded-inverse equations in Section 16 define
`rollout_composite_selector_v3`. It consumes `rollout_validation_v2` records,
whose displacement and constraint metrics all use the first 30 steps.
Checkpoints produced by selector v2, which mixed 3-second ADE with 6.4-second
FDE and constraints, are not resume-compatible.

Before freezing the first policy, existing checkpoints are evaluated to verify
component variation and identify semantically constant components. Excess
metrics use bounded inverse utilities rather than hard clipping, so an error
above one nominal scale remains rankable instead of collapsing to zero. An
exactly zero excess may legitimately remain at utility one. A sensitivity
report recomputes rankings after independently changing each top-level weight
by `+/-20%` and renormalizing. This checks units and rank stability without
using treatment-run outcomes to tune the policy.

The workflow stores this report after every epoch. A one-epoch Smoke provides
utility-saturation evidence but not ranking evidence; two or more checkpoints
add top-1 stability and Spearman rank correlation. Before the paired three-seed
experiment, the same report is run over the available baseline checkpoint
history and retained with the experiment artifacts.

## 15. Metric Completeness and Ranking

### 15.1 Selection order

Every epoch follows this order:

1. verify validation snapshot and sample identity;
2. verify all metrics required by frozen component availability are present,
   finite, and have the expected coverage;
3. compute every available utility and the weighted composite score;
4. compare the score with the current best using `min_delta`.

Missing or non-finite required metrics fail validation immediately. The
selector does not silently remove a component after epoch 1, assign a synthetic
value, or fall back to ADE.

## 16. Composite Checkpoint Score

### 16.1 Trajectory utility

\[
D_{natural}
=
0.6\frac{1}{1+ADE^{natural}_{3s}/2.5}
+
0.4\frac{1}{1+FDE^{natural}_{3s}/3.0}
\]

\[
D_{scene}
=
0.6\frac{1}{1+ADE^{scene}_{3s}/2.5}
+
0.4\frac{1}{1+FDE^{scene}_{3s}/3.0}
\]

\[
D=0.5D_{natural}+0.5D_{scene}
\]

### 16.2 Comfort utility

\[
\bar c
=
0.5comfort^{natural}_{excess}
+
0.5comfort^{scene}_{excess}
\]

\[
C=\frac{1}{1+\bar c/0.15}
\]

### 16.3 Map-safety utility

\[
\bar r
=
0.5offroad^{natural}_{excess}
+
0.5offroad^{scene}_{excess}
\]

\[
S=\frac{1}{1+\bar r/0.10}
\]

### 16.4 Navigation utility

\[
\bar q
=
0.5route\_gap^{natural}
+
0.5route\_gap^{scene}
\]

\[
N_{route}=\frac{1}{1+\bar q/0.15}
\]

Destination error also combines natural and scene-balanced views:

\[
\bar d
=
0.5destination\_error^{natural}
+
0.5destination\_error^{scene}
\]

\[
N_{destination}=\frac{1}{1+\bar d/7.5}
\]

When wrong-branch evidence is available:

\[
\bar b
=
0.5wrong\_branch\_excess^{natural}
+
0.5wrong\_branch\_excess^{scene}
\]

\[
N_{branch}=\frac{1}{1+\bar b/1.0}
\]

\[
N
=
0.5N_{route}
+0.3N_{branch}
+0.2N_{destination}
\]

Without wrong-branch evidence:

\[
N=0.7N_{route}+0.3N_{destination}
\]

If destination evidence is unavailable, its term is removed and the remaining
navigation weights are renormalized.

### 16.5 Final score

\[
Score=0.50D+0.15C+0.15S+0.20N
\]

If `S` or `N` is unavailable, remove the component and renormalize the remaining
top-level weights to sum to one. Availability and effective weights are fixed
before epoch 1.

All component inputs, component utilities, effective weights, and final score
are logged. The checkpoint with the highest score under the fixed policy is the
best checkpoint.

## 17. Scheduler, Early Stopping, and Selection

### 17.1 Scheduler

Use:

```text
ReduceLROnPlateau(
    mode="max",
    factor=0.5,
    patience=1,
    threshold=0.0005,
    threshold_mode="abs",
)
```

For every completed validation epoch, pass the composite score to the
scheduler. Missing or non-finite required metrics fail the validation epoch
rather than producing a partial score.

Initial improvement threshold:

```text
min_delta = 0.0005
early_stopping_patience = 5
```

The scheduler threshold and selector `min_delta` intentionally match. The
three-seed experiment must report epoch-to-epoch score noise. A later change to
the score threshold requires a selector policy-version bump. Patience is an
execution policy and may be increased for a resumed run without changing the
score definition.

### 17.2 Independent best checkpoints

Maintain two independent best records:

```text
best: highest composite score
best_trajectory: highest trajectory component utility
```

The trajectory component is the existing equal combination of natural and
scene-balanced trajectory utility. It therefore reflects natural and
scene-balanced ADE@3s and FDE@3s without introducing a separate ADE-only
ranking rule.

- The first successfully validated checkpoint establishes both records.
- Composite best improves only when
  `score > best_score + min_delta`.
- Trajectory best improves only when
  `trajectory > best_trajectory + min_delta`.
- Either record may point to the same immutable checkpoint.
- The scheduler continues to consume only the composite score. The trajectory
  record affects checkpoint retention and early stopping, not learning-rate
  scheduling.

### 17.3 Bad epochs

- An epoch is improving when either composite best or trajectory best improves.
- A successfully validated epoch is a bad epoch only when neither record
  improves.
- Improvement in either record resets the bad-epoch count to zero.
- Early stopping triggers when bad epochs reach the configured patience.
- The final checkpoint is always saved.
- ADE-only fallback remains forbidden.

## 18. Checkpoint, Resume, and Registry Contract

### 18.1 Saved selector state

Every immutable epoch checkpoint stores:

```text
selector policy version
validation snapshot and sample/group digests
frozen component availability and effective weights
metric history with natural and scene-balanced aggregates
validation metric contract (`3.0 s`, 30 steps, logged XY, scene-balanced)
composite-best and trajectory-best checkpoint identities
bad epoch count
scheduler state
optimizer/scaler/RNG state
```

Resume rejects any mismatch in policy, validation identity, availability,
score weights, utility scales, or loss configuration. Restoring only model and
optimizer state is insufficient.

For an unchanged training policy, resume restores the model, optimizer,
`ReduceLROnPlateau`, gradient scaler, RNG, metric history, best checkpoint, and
bad-epoch state exactly.

An explicit continuation transition supports:

```text
early-stopping patience: increased
optionally, navigation repeat: disabled -> enabled
```

The caller must opt into this transition. No other config or dataset identity
change is allowed. The model, optimizer state, current learning rate, metric
history, and historical composite best are preserved. The historical
trajectory best is reconstructed from the stored trajectory component. The
plateau scheduler and bad-epoch count reset so a checkpoint that stopped under
the former patience can continue. A transition is rejected unless at least one
new epoch will run.

### 18.2 Workflow outputs

The training workflow records three roles:

- `final`: the last completed epoch;
- `best`: the highest composite score;
- `best_trajectory`: the highest trajectory component utility.

The task output remains the composite-best checkpoint so downstream evaluation
behavior does not change. Metadata exposes all three records.

### 18.3 Model registry

Registry roles are:

```text
final
best
best_trajectory
```

The `best` role is selected by the versioned composite score. Production
promotion gates, if later required, operate after research checkpoint
selection and are not part of this policy.

Every v3 registry version stores `validation_ade_3s_m`,
`validation_fde_3s_m`, `validation_metric_version`,
`validation_metric_horizon_seconds`, `validation_metric_horizon_steps`,
`validation_metric_target_source`, and `validation_metric_aggregation`.
Generic `validation_ade` and `validation_fde` aliases have the same canonical
3-second meaning for v3 versions.

## 19. MLflow and Auditability

Log the following namespaces:

```text
train/loss_action
train/loss_rollout_path
train/loss_rollout_final
train/loss_rollout
train/loss_comfort_jerk
train/loss_comfort_lateral
train/loss_map_route
train/loss_map_drivable
train/loss_constraint
train/loss_total

validation/natural/*
validation/scene_balanced/*
validation/scene_distribution/*/{count,mean,p50,p90}
validation/coverage/*

val/ade
val/fde
val/ade_3s_scene_balanced_logged_xy
val/fde_3s_scene_balanced_logged_xy

selection/component/*
selection/effective_weight/*
selection/score
selection/bad_epochs
selection/score_improved
selection/trajectory_improved

audit/reconstruction/*
```

Checkpoint metadata and MLflow must agree on epoch, score, component values,
effective weights, validation digests, and the validation metric contract.
`val/ade` and `val/fde` are canonical scene-balanced logged-XY 3-second
metrics. Target-control rollout diagnostics use the explicit
`val/control_rollout_*` namespace. The immutable checkpoint metadata is
authoritative if an MLflow retry occurs.

## 20. Implementation Plan

### Preflight: separate spike

1. Extract the shared rollout into a minimal auditable function.
2. Verify pose availability and sample alignment on the frozen snapshot.
3. Run target-control reconstruction against logged future pose.
4. Publish the immutable audit artifact and Go/No-Go decision.
5. If integrated controls are No-Go, select logged XY explicitly and version
   the target-source contract before implementing the planner loss.

### PR Step 1: shared rollout

- Add the float32 PyTorch source-of-truth rollout.
- Refactor training and evaluation callers to use it where practical.
- Retain a NumPy parity implementation for external/CPU reports.
- Add AMP, finite-input, gradient, and parity tests.

### PR Step 2: minimal planner loss

- Preserve normalized action loss.
- Add path and final rollout losses against packed logged XY.
- Add trajectory-level target-relative comfort.
- Add pointwise target-relative route/drivable footprint loss.
- Retire the legacy combined route-consistency objective from the v2 training
  path while reusing its geometry helpers.
- Add deterministic offline drivable distance supervision.
- Keep ordinary sample-mean reduction.

### PR Step 3: validation records and aggregation

- Emit one complete record per validation `sample_uid`.
- Require `split_group_uid`.
- Compute natural and scene-balanced aggregates.
- Log scene counts and deterministic p50/p90 distributions.

### PR Step 4: checkpoint selector

- Freeze component availability before epoch 1.
- Compute the natural/scene-balanced composite score.
- Select and persist the highest-scoring checkpoint.

### PR Step 5: Flyte lifecycle integration

- Change scheduler input and mode.
- Drive early stopping from either composite-best or trajectory-best
  improvement while keeping the scheduler on composite score.
- Update checkpoint and resume validation.
- Update MLflow logging and model-registry roles.

## 21. Test Plan

### 21.1 Reconstruction audit

1. Report target-rollout versus logged-pose p50/p90/p95 at 3 s and 6.4 s.
2. Persist dataset, validation-group, and sample UID digests.
3. Report scene-level distributions and worst scenes.
4. Compare target-control headings with centered logged-XY tangents at moving
   steps using a versioned minimum baseline, and report angular p50/p90/p95.
5. Reject missing, misaligned, or non-finite pose records.

Heading alignment is diagnostic in the first policy and does not alter the
position Go/No-Go thresholds. It must be recorded before changing the target
footprint heading source.

### 21.2 Rollout

1. PyTorch and NumPy integration agree within the documented tolerance.
2. Prediction rollout equal to logged XY produces zero rollout loss.
3. Position loss has a finite non-zero gradient to acceleration.
4. Position loss has a finite non-zero gradient to curvature.
5. Braking through zero documents the clamp's zero-gradient region.
6. AMP input still produces float32 rollout internals and outputs.
7. Non-finite controls or initial state fail before integration.
8. Flattened and structured controls produce identical output.

### 21.3 Constraint loss

1. Prediction equal to target produces zero comfort loss.
2. Equal comfort peaks at shifted timesteps produce zero additional loss.
3. A larger predicted peak increases comfort loss.
4. Prediction footprint equal to the logged target footprint produces zero map
   loss.
5. Greater predicted footprint outside distance increases map loss.
6. Corner checks detect violations missed by the ego center.
7. Route and map validity masks act independently.
8. Missing optional map supervision returns finite differentiable zero.
9. Out-of-raster motion is penalized and remains differentiable.

### 21.4 Aggregation

1. Natural aggregation equals the direct eligible-sample mean.
2. Scene-balanced aggregation equals the mean of scene means.
3. Duplicating samples within one scene without changing its mean does not
   change scene-balanced output.
4. Missing `split_group_uid` fails validation.
5. Scene p50/p90 use the fixed linear quantile method deterministically.
6. Metric-specific eligibility produces the expected scene counts.

### 21.5 Checkpoint selection

1. Missing or non-finite required metrics fail validation.
2. The first complete checkpoint becomes best.
3. Later checkpoints independently update composite-best and trajectory-best
   using the same `min_delta`.
4. Scene-balanced trajectory utility changes the score.
5. Route utility is target-relative.
6. Coverage shortage excludes optional components and renormalizes weights.
7. Component availability does not change after epoch 1.
8. Early stopping resets when either best record improves.
9. Resume reproduces the same best epoch and bad-epoch count.
10. Resume rejects policy, availability, weight, or validation drift.

### 21.6 Performance

Benchmark:

- rollout forward;
- rollout forward and backward;
- full training step;
- validation aggregation and selector overhead.

The allowed full-step throughput regression is at most 12% relative to the
paired control under the same batch and hardware configuration.

## 22. Acceptance Criteria

Run three paired seeds on the same immutable snapshot.

Primary criteria:

1. Paired median composite score for B is not worse than A.
2. At least two of three seeds have B composite score greater than or equal to
   A.
3. Paired median natural ADE@3s and FDE@3s regress by no more than 5%.
4. Paired median scene-balanced ADE@3s or scene-balanced FDE@3s improves.
5. Safety, comfort, and navigation component values and paired deltas are
   reported separately; none is hidden by the aggregate score.
6. Full training-step throughput regresses by no more than 12%.

Required diagnostic report:

- natural ADE/FDE;
- scene-balanced ADE/FDE;
- scene p50/p90;
- each rollout and constraint loss;
- reconstruction audit errors;
- coverage and effective score weights;
- best and final checkpoint identities;
- every composite-score component and its paired delta.

A fixed 5% improvement and universal absolute-threshold success across all
seeds are not required.

## 23. Risks and Mitigations

### Target controls do not reconstruct logged motion

Mitigation: implemented. The full audit rejected integrated target-control XY.
The planner now compares predicted rollout positions with packed logged XY,
while retaining target controls for action and comfort. Audit identity and the
observed reconstruction error remain immutable metadata. The audit also records
target-control heading error against moving logged-path tangents before any
change to the target footprint heading source.

### Constraint scale overwhelms action learning

Mitigation: fixed small top-level weight, per-term logging, gradient tests, and
matched throughput/training diagnostics. Any weight change creates a new loss
policy version.

### Footprint or raster artifacts create false map penalties

Mitigation: prediction is compared relative to target on the same geometry,
with a half-pixel resolution-aware tolerance. Footprint dimensions, geometry,
and effective metric tolerance are checkpoint metadata. Selector calibration
rejects saturated target route or drivable metrics before epoch 1.

### Composite score hides a regression

Mitigation: every component and paired A/B delta remains visible in MLflow and
the experiment report, and score weights/scales are versioned. The weighted
trade-off is intentional in research selection. A separate production
promotion policy may add hard operational gates later without changing how the
research checkpoint is ranked.

### Sparse navigation evidence makes ranking unstable

Mitigation: freeze coverage before epoch 1, require at least 50 route-valid and
20 wrong-branch-eligible samples, and exclude unavailable components rather
than treating missing evidence as success.

## 24. Final Decision

This proposal changes only:

1. planner training loss by retaining action supervision and adding logged-XY
   rollout, target-relative comfort, and target-relative footprint map
   constraints;
2. validation by adding natural and `split_group_uid` scene-balanced
   aggregation;
3. checkpoint selection by replacing ADE-only comparison with a versioned
   weighted composite score, with explicit coverage and resume contracts.

It does not add Cosmos supervision, sample balancing, tail mining, speed or
heading losses, new JEPA/Reasoning objectives, or architecture changes. The
existing World Model and JEPA objective remain enabled and unchanged in both
matched arms.

The primary hypothesis is:

> Rollout-aligned planner loss improves scene-balanced ADE or FDE without
> sacrificing the combined trajectory, safety, comfort, and navigation
> utility represented by the checkpoint score.

The reconstruction audit selected logged XY instead of integrated target
controls as the position teacher. The three-seed paired experiment then
determines whether loss alignment is sufficient or whether a later, separate
data-balancing proposal is justified.
