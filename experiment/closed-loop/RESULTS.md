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

## 2026-11-01: review rounds and the loop across iterations (r2)

Seven adversarial review rounds (Claude Code and Codex, each round on the
whole diff with the same neutral prompt) changed the reward and the update
between r1 and r3. The fixes that changed results:

- Reward: NAVSIM's collision types and TTC rule, AlpaSim's offroad rule,
  EPDMS's human filter matched by place, EP from the ego's place over the
  log's moving time, only the objects AlpaSim simulated, nuPlan's comfort
  filtering, a full 4 s horizon.
- Update: Dr. GRPO advantages, gated plans scored by their safe fraction, a
  line search that requires the surrogate to fall and bounds every state's
  plan change, a fresh seed per step, a replay check on position and heading.
- Driver input: a route corridor along the route's direction, the travelled
  path behind the ego and unique successors ahead, lanes without edges
  widened as in AlpaSim.

r2 ran the loop with the round-4 reward to check it across iterations, and
was stopped in its third step once that was shown:

| GRPO step | Decisions | Replay error [m] | Anchor | Step scale | Surrogate before / after |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 (imitation rollout) | 37 | 1.3e-4 | 0 | 1/256 | 0 / -0.0133 |
| 2 (rollout of the step-1 planner) | 37 | 3.4e-4 | 0.22 | 1/1024 | 0.0047 / -0.0015 |

- The driver ran the RL checkpoint, and the replay of that rollout matched it
  to 0.3 mm, so checkpoints pass between steps and updates stay on-policy.
- The anchor rose from 0 to 0.22 after one step and the line search shrank
  the next step to stay in the trust region; r1's unbounded steps had reached
  71 and then 1852.
- A GRPO step takes about 8 minutes on the L40S for 37 decisions (scoring
  9 plans per decision on the CPU dominates) and peaks at 4.3 GB of GPU memory.
- The replay check also refused a rollout recorded before a driver-input
  change (0.095 m at decision 5), which is the case it exists for.

## 2026-11-01: GRPO with the final reward and update (r3)

Same setup as r1 (scene `clipgt-01d503d4`, 110 steps, one rollout per
iteration), with the code after the seventh review round.

| Iteration | Collision | Off-road (first) | Max dist to GT traj. [m] | Driven [m] |
| :--- | :--- | :--- | ---: | ---: |
| 0 (imitation) | no | yes (3.8 s) | 5.92 | 30.2 |
| 1 | no | yes (3.8 s) | 5.82 | 30.0 |
| 2 | no | yes (3.8 s) | 5.90 | 30.1 |
| 3 | no | yes (3.8 s) | 5.86 | 30.1 |

| GRPO step | Decisions | Mean-plan reward | Gated decisions | Group reward std | Anchor | Step scale | Plan change | Surrogate before / after | Replay error [m] |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37 | 0.325 | 59% | 0.032 | 0 | 1/128 | 0.091 | 0 / -4.0e-5 | 1.3e-4 |
| 2 | 38 | 0.316 | 61% | 0.027 | 0.023 | 1/256 | 0.060 | 1.2e-4 / 7.7e-5 | 1.1e-4 |
| 3 | 37 | 0.326 | 59% | 0.024 | 0.003 | 1/256 | 0.021 | 3.0e-5 / 1.6e-5 | 1.2e-4 |

- The loop works end to end: every step replayed the executed plans to
  0.13 mm, lowered the surrogate, and stayed inside the trust region.
- It does not yet change the driving. The accepted steps move the mean plan
  by 0.02-0.09 in the trust-region distance, a tenth of the bound or less:
  the line search halves the step until the surrogate falls, so what limits
  the step is the direction, not the bound. Two reasons are visible in the
  numbers. The planner's gradient is approximate (its recurrent state is
  detached from the BEV lookup), and the within-group reward spread is small
  (std 0.02-0.03), so a group of 8 plans carries little signal.
- At most states the imitation plan leaves the lanes within 4 s (about 60% of
  the decisions are gated), the drift of #221, and the policy leaves the road
  3.8 s after takeover in every iteration.
- Next: larger exploration (`--sigma`) to widen the reward spread, more
  scenes and rollouts per step, and checking how much the detach in the GRU
  planner costs the update direction.
