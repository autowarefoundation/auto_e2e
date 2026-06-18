# Proposal: BEV-only trajectory scoring + few-step inference upgrade for `FlowMatchingPlanner`

> Per CONTRIBUTING.md, this is posted as a Discussion before any PR, since
> it touches the trajectory planning module. References PR #40 (driving
> policy / FlowMatchingPlanner), PR #55 (map encoder / MapBEVFusion), issue
> #35 (BEV resolution / hardware target), issue #66 (open-loop eval
> pipeline), and the GoalFlow paper (arXiv:2503.05689) discussed in the
> 17/06 WG meeting.

## Context

`FlowMatchingPlanner` (merged in #40) already does conditional flow
matching with BEV cross-attention and AdaLN modulation, and the map
pipeline (#55) already fuses a rasterized nav-map into that BEV before the
planner runs. What's still missing relative to GoalFlow's design — which
we discussed in detail this week — is everything downstream of trajectory
*generation*: GoalFlow doesn't just sample one trajectory, it samples many,
scores them against a goal-point vocabulary and a learned drivable-area
classifier, and picks the best one.

We were told to get BEV working first, before investing in the full
goal-point vocabulary + learned DAC classifier machinery GoalFlow uses.
This proposal is scoped to exactly that boundary: it adds multi-sample
generation and re-ranking using **only data we already have on disk**
(the rasterized map image, the predicted trajectory's own kinematics) —
zero new training, zero new loss terms, zero dependency on a semantic BEV
head or goal-point clustering that doesn't exist yet.

## What this proposal adds (Phase 1)

**1. `TrajectoryComplianceScorer`** — a wrapper around any `BasePlanner`
that draws `K` samples per scene (vectorized via batch-repeat, one
`forward()` call) and re-ranks them by:

- **Drivable-area compliance**, read directly off the *raw rasterized*
  `map_input` pixels at each waypoint's projected BEV coordinate. This is
  a colour lookup, not a classifier — it stands in for GoalFlow's learned
  DAC score until (if) we build a real semantic BEV head.
- **Kinematic comfort**, penalizing samples whose (acceleration,
  curvature) sequence exceeds configurable comfort bounds.

Selection mirrors GoalFlow's own two modes: `"nearest"` (argmax — pick the
single best-scoring sample) or `"mean"` (softmax-weighted blend across all
K samples).

**2. Shifted inference-time timestep schedule** for `FlowMatchingPlanner`
— a small, additive change (`timestep_schedule="uniform"|"shifted"`,
default `"uniform"` preserves exact current behaviour) implementing the
same `t_shifted = (alpha·t)/(1+(alpha-1)·t)` warp GoalFlow uses at
inference, which their own ablation shows matters most at low step counts
— directly relevant to our Renesas R-Car deployment target and the
CPU/GPU latency work already in the benchmark thread.

## What this proposal deliberately does NOT add (Phase 2 — deferred)

- No goal-point vocabulary or offline-cached goal scores (GoalFlow's
  `cluster_points_8192_.npy` + `goal_point_scores.gz` equivalent).
- No learned BEV semantic segmentation head (GoalFlow's
  `_bev_semantic_head`) — Phase 1's drivable-area check is a raw colour
  lookup precisely so we don't need one yet.
- No classifier-free-guidance-style goal-conditioned/unconditioned fusion.

These are real GoalFlow ideas worth revisiting once KITScenes' HD map
gives us a reliable semantic drivable-area channel to train against — but
building the goal-point + DAC-classifier machinery before we have that
signal would mean training it against the same colour-lookup proxy this
proposal already gives us for free, which isn't worth the added
complexity yet.

## Calibration caveat (needs a reviewer who owns the map renderer)

`project_xy_to_bev_pixel` in the attached code needs `pixels_per_meter`,
`ego_row`, and `ego_col` confirmed against whatever convention the
KITScenes/L2D map renderer actually uses to produce `map_input`. The
defaults follow the 120 m front / 60 m rear / 60 m each side @ 0.4 m
geometry from issue #35, but I have not verified them against the
renderer itself — would appreciate a second pair of eyes here (cc Richard
/ Zain) before this gets merged, since a miscalibrated lookup would
silently score every trajectory as "compliant" or "non-compliant"
regardless of where it actually goes.

## Sensor scope — camera + map only, no LiDAR (deliberate, not a gap)

Worth stating explicitly rather than leaving implicit: this proposal, like
the rest of `auto_e2e` today, uses only camera tiles and the rasterized
map image — `AutoE2E.forward()` has no LiDAR input anywhere in its
signature, and nothing here changes that.

It is worth separating two things that both get called "BEV" in this
discussion, since GoalFlow conflates them in a way that could cause
confusion later. GoalFlow's BEV comes from fusing camera features with a
*live LiDAR point cloud* (their camera backbone plus a separate LiDAR
encoder) — a runtime perception sensor that gives direct range
measurement. Our map-BEV is different in kind: it is a rendering of a
*pre-surveyed HD map* (KITScenes' vectorized map data), built offline, not
measured live. Skipping LiDAR does not cost us the map signal at all —
that signal was never LiDAR-dependent in the first place, and `map_input`
already reaches the planner today through `MapBEVFusion`.

What skipping LiDAR does cost us is direct range measurement for dynamic
obstacles (other vehicles, pedestrians, cyclists). Without LiDAR, that has
to come entirely from camera-based depth and 3D understanding, which is
the harder and less metrically reliable half of the perception problem.
That trade-off is consistent with the project's existing all-camera
architecture and the Renesas R-Car embedded deployment target, so it is
not a new decision this proposal is introducing — but the WG's own scope
for the Driving Model Team lists "cameras, LIDAR/RADAR" as the intended
sensor suite, so this should be treated as a deliberate, visible
simplification for now rather than something that quietly becomes
permanent because nobody revisited it.

Genuinely open to other framings here, or to being told this trade-off
analysis is missing something — flagging it explicitly so the group can
weigh in rather than letting it default silently into "how things are."

## Files in the attached PR

| File | Status |
|------|--------|
| `Model/model_components/trajectory_planning/trajectory_scorer.py` | New |
| `Model/model_components/trajectory_planning/flow_matching_planner.py` | Modified (additive, see patch notes) |

Zero changes to `auto_e2e.py`, `base.py`, `gru_planner.py`, or any existing
public call site. `TrajectoryComplianceScorer` is opt-in — nothing wires it
into the default forward pass automatically, since that's a separate
design decision (does the scorer live inside `AutoE2E.forward()` behind a
config flag, or stay a standalone post-processing step callers opt into?)
that probably deserves its own discussion once Phase 1 lands and we have
real numbers from it.

## Testing status

**Smoke test (CPU, zero-GPU, no KITScenes data required):** Passed locally
against a fake `BasePlanner` that satisfies the `BasePlanner` contract.
Covers shape correctness, out-of-bounds waypoint handling, both selection
modes (`nearest` / `mean`), and the invalid-selection-mode error path.
This test does not require a GPU or the actual trained model and can be
reproduced on any machine with PyTorch installed.

**Quality / integration test (does it actually improve ADE?):** Not yet run.
This requires a trained `FlowMatchingPlanner` checkpoint and parsed
KITScenes map+camera data feeding real `map_input` tensors — both of
which are currently inaccessible due to hardware constraints on the
contributor's machine for the next few days (GPU unavailable). This test
is also contingent on the map-pixel calibration review flagged above, since
a miscalibrated `ego_row` / `ego_col` would make the DAC score meaningless
and render any ADE comparison uninformative.

**What this means for the PR:** the code is structurally sound and the
`BasePlanner` contract is fully satisfied, but the PR should be treated as
ready for code review and calibration review, not ready to merge until
quality validation on real data is confirmed. Will update the PR once
hardware is available.

## Open questions for the group

1. Where should `TrajectoryComplianceScorer` actually plug in —
   `AutoE2E.forward()` behind a flag, or kept as a standalone utility
   callers use explicitly (e.g. only at eval/inference time, never during
   training)?
2. Is `num_samples` (K) something we want exposed as a runtime knob for
   the Renesas target, where compute budget is tight, vs always running
   at a fixed K?
3. Does the comfort-bound scoring belong here at all, or should it instead
   become an auxiliary *training* loss term on `FlowMatchingPlanner`
   directly (closer to how GoalFlow's own ablation table treats each
   signal as a separate, addable loss)?
4. Is camera+map-only the right scope for this phase, or is there a
   lighter-weight way to bring RADAR or LiDAR range data in earlier than
   planned, given the Driving Model Team's stated longer-term sensor
   suite? Open to being told there's a better way to sequence this than
   what's proposed here.

This is a first pass, not a final design — alternative approaches,
different scoring signals, or a different Phase 1/Phase 2 split are all
welcome. Posting it now mainly to get the scoping question (what needs
goal-point/LiDAR infrastructure we don't have yet, vs what doesn't) in
front of the group before writing more code against it.
