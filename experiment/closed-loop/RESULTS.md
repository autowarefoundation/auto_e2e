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

Candidates to check, in order of cost:

1. Open-loop replay: keep the ego on the log for the whole rollout with the
   driver running (`force_gt_duration_us` 20 s,
   `skip_driver_during_force_gt=false`) and compare plans with the recorded
   future. A right bias here rules out closed-loop compounding.
2. The `nuplan_nurec` rig: camera-to-reference extrinsics against the
   rendered `extra_cameras`, and the rear-axle reference frame of the NVIDIA
   rig versus nuPlan's. Unlike the KITScenes profile, this rig has no parity
   test against packed samples.
3. Camera-only inputs: the checkpoint was trained with route conditioning and
   runs here with `map_valid = route_valid = False`.
