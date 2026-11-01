# Closed-loop results

## 2026-10-22: first runs on g6e.xlarge (1x L40S)

AlpaSim `affc2eab209f`, auto_e2e `e841c45b` (PR #177 head `f1e53980` plus
this directory), `driver=autoe2e_nuplan`, `topology=1gpu`. Each rollout is
200 steps of 100 ms; the ego follows the log for the first 7 s, then AutoE2E
plans every step (130 decisions per rollout).

### mock-01d503d4 (`CHECKPOINT=MOCK`, straight at current speed)

Plumbing check: one `rollout.asl`, front-camera video and metrics in about
5 minutes. Rear-ended 8.7 s after takeover, 50.0 m of 73.8 m recorded.

### nuplan-ep5-5scenes (nuPlan Epoch 5, `ca8b43d7a777`)

All five rollouts completed in about 85 minutes. The driver handles one
request at a time (avg batch 1.0) at about 6.6 s per inference, at 50% GPU
utilisation and 18-20 GB GPU memory for all services together.

`summarize_run.py` (times in seconds after takeover):

| Scene | Collision | Off-road (first) | Wrong lane (first) | Max dist to GT traj. [m] | Driven / recorded [m] |
| :--- | :---: | :---: | :---: | ---: | ---: |
| 01d503d4 | no | yes (4.1) | no | 12.18 | 49.3 / 73.8 |
| 023b7fcc | no | yes (3.0) | no | 25.01 | 75.0 / 106.9 |
| 0245ff75 | no | yes (2.8) | yes (0.9) | 24.34 | 85.8 / 169.8 |
| 026d6a39 | no | no | no | 26.28 | 121.0 / 107.5 |
| 02eadd92 | no | yes (2.5) | no | 26.32 | 106.1 / 93.9 |

In all five videos the plan bends to the right within about 1.5 s of
takeover, on straight roads and at an intersection alike. In 01d503d4 the plan
becomes a tight right circle by 8 s after takeover and the ego leaves the
road, where the NuRec render breaks down. The same direction in every scene
points at a systematic input or frame mismatch rather than scene difficulty;
the cause is not yet identified.

### Diagnosis: ego-motion feedback, not a configuration error

Checked and ruled out:

- Camera assignment: the six frames the driver received (extracted from
  `rollout.asl` with `alpasim_utils.asl_to_frames`) show front, front-left,
  front-right, rear-left, rear and rear-right as the `nuplan_nurec` profile
  expects.
- Egomotion construction: row order (oldest first), 10 Hz grid, rad/s units
  and the nuPlan curvature rule match `data_parsing/nuplan/packing.py`.
- Dataset policy: the policy the plugin derives from the dataset name equals
  the one stored in the checkpoint (`mask_latest_history_acceleration=False`).
- Open-loop quality: the first plan of every rollout, made with the ego still
  on the log, matches the recorded drive (01d503d4: 0.1 m longitudinal and
  0.3 m lateral at 5 s).

[`replay_driver.py`](./replay_driver.py) rebuilds each decision's inputs from
the log, reproduces the logged plan exactly, and reruns it with the yaw rate
and curvature columns of the egomotion history set to zero. Heading of the
plan 3 s ahead, in degrees (left positive):

| Scene | Decision 1 | Decision 11 | Decision 21 | Decision 21, yaw/curvature = 0 |
| :--- | ---: | ---: | ---: | ---: |
| 01d503d4 | +1.63 | -2.83 | -19.09 | +1.66 |
| 023b7fcc | +3.21 | -3.52 | -39.73 | +1.25 |
| 0245ff75 | +1.87 | -5.11 | -19.87 | +1.11 |
| 026d6a39 | +0.40 | -8.26 | -90.32 | -0.03 |
| 02eadd92 | -0.30 | -8.75 | -69.40 | +0.35 |

With the yaw history removed the plan stays within about 2 degrees of straight
even when the ego has already turned 5 to 10 degrees off the lane. The turn is
therefore driven by the ego's own yaw history, not by the cameras:

1. The first plans start with a small right kink (heading -0.5 degrees at
   0.1 s in 023b7fcc) before bending left, and the controller follows it.
2. The resulting negative yaw rate enters the next egomotion history, and the
   model extrapolates it into a sharper right turn, at a gain above one
   (newest yaw rate -0.127 rad/s gives a planned -9.6 degrees in 1 s in
   01d503d4).
3. The loop diverges within 2 to 3 s and the ego leaves the road.

This is the copycat failure of imitation policies that condition on their own
past motion: the history explains the logged future well open-loop, and feeds
back closed-loop. A cheap check without retraining is a closed-loop run with
the yaw rate and curvature history zeroed at inference.

## 2026-11-01: map and route inputs, first GRPO run (r1)

AlpaSim `affc2eab209f` with `alpasim.patch`, `driver=autoe2e_nuplan`,
nuPlan Epoch 5, scene `clipgt-01d503d4`, 110 steps (7 s on the log, then
4 s and 40 decisions in closed loop), one rollout per iteration. The driver
loaded the scene's HD map and fed map context and route mask from here on.

r1 ran the first version of `rl/grpo_step.py`: phase-1 reward
`-(ADE + 0.5 FDE)`, plain SGD (lr 1e-4) on the log-likelihood, KL only in the
low-frequency subspace, no trust region. Review found the SGD step scaled
curvature about 100x over acceleration and the KL ignored high frequencies;
both are fixed in the current code, so r1 documents the failure, not the
current method.

| Iteration | Collision | Off-road (first) | Max dist to GT traj. [m] | Driven / recorded [m] |
| :--- | :--- | :--- | ---: | ---: |
| 0 (imitation) | no | yes (3.8 s) | 5.89 | 30.1 / 73.8 |
| 1 | front (2.9 s) | no | 2.33 | 28.4 / 73.8 |
| 2 | no | no | 3.81 | 30.8 / 73.8 |
| 3 | front (3.9 s) | no | 1.11 | 28.8 / 73.8 |

Times are after takeover; "driven / recorded" compares the 11 s rollout with
the whole 20 s recording.

| GRPO step | Decisions | Mean-plan reward | Group reward std | KL to imitation | Replay error [m] |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 (iteration 0 rollout) | 37 | -20.4 | 0.84 | 0 | 1.3e-4 |
| 2 (iteration 1 rollout) | 40 | -12.5 | 0.36 | 71 | 3.5e-5 |
| 3 (iteration 2 rollout) | 39 | -16.2 | 0.71 | 1852 | 9.4e-5 |

- The replay reproduces the executed plans to 0.1 mm, so the updates were
  on-policy.
- The KL grows from 71 to 1852 in two steps: with no bound on the step, the
  planner left the imitation policy far beyond the sampling scale. The
  driving changes from one iteration to the next (off-road, then front
  collisions) are therefore not evidence of learning; with one rollout per
  iteration they cannot be separated from that drift.
- The phase-1 reward never sees other vehicles; iterations 1 and 3 hit one
  front-first. The current reward (`rl/reward.py`) gates collisions.
