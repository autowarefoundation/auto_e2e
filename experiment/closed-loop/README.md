# Closed-loop experiment: AutoE2E in AlpaSim

Runs the AutoE2E AlpaSim driver plugin (`Model/plugins/alpasim_driver`, PR #177)
in closed loop on NuRec scenes, on a single GPU host with docker compose, and
keeps the `rollout.asl`, videos and metrics of every run. The scripts are a
staging ground for running the same loop on EKS later, so each step is a
separate script with pinned inputs.

## What runs where

AlpaSim is a set of gRPC services that the wizard starts with docker compose:

| Service | Image | Role in this experiment |
| :--- | :--- | :--- |
| runtime | `alpasim-base:autoe2e-<commit>` | Steps the simulation, writes `rollout.asl` |
| renderer | `nvcr.io/nvidia/nre/nre-ga:26.04` | NuRec: renders the six nuPlan cameras from the scene USDZ |
| driver | `alpasim-base:autoe2e-<commit>` | The `autoe2e` plugin running the checkpoint |
| controller, physics | `alpasim-base:autoe2e-<commit>` | Vehicle model and ground constraint |
| eval (in runtime) | | Metrics and videos from the logs |

`driver=autoe2e_nuplan` renders the nuPlan v1.1 rig on the NVIDIA camera IDs of
the public NuRec scenes, so the nuPlan Epoch 5 checkpoint sees the cameras it
was trained on. No KITScenes reconstruction exists yet, so KITScenes
checkpoints cannot be run here.

## Inputs

[`config.env`](./config.env) pins the AlpaSim commit, the checkpoint SHA-256
and the driver config; [`scenes.txt`](./scenes.txt) lists the scenes. The
scenes come from the gated Hugging Face dataset
`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` (release 26.01), about 2 GB each.

## AlpaSim patch

AlpaSim's driver API gives a model the cameras, ego poses and route
(`PredictionInput.route`), but not the HD map, and not which scene it drives.
AutoE2E was trained with a 14-channel map raster and a route mask, so without
them it runs on its camera-only track (#221). [`alpasim.patch`](./alpasim.patch)
is the whole fork: 4 lines that add `PredictionInput.scene_id` from the
session's `debug_info.scene_id`. With it the plugin opens the scene's USDZ in
the scene cache, which `driver=autoe2e*` mounts read-only at `/mnt/nre-data`,
and draws the map and route with the nuPlan training rasterizer
(`Model/plugins/alpasim_driver/alpasim_autoe2e/scene_map.py`).

`setup_alpasim.sh` applies the patch after checking out `ALPASIM_COMMIT`. On a
host that is already set up, apply it by hand; the driver mounts `src/`, so no
image rebuild is needed:

```bash
git -C ~/alpasim apply <repo>/experiment/closed-loop/alpasim.patch   # undo: git apply -R
```

Rollouts recorded before the patch ran camera-only; `replay_driver.py` now
replays them with the map and route, so their replayed plans differ from the
logged ones.

## Host requirements

NVIDIA driver >= 570, Docker with compose and buildx, the NVIDIA Container
Toolkit, and about 100 GB of free disk: the NRE image is 14 GB compressed and
the driver image is built on a CUDA devel base. `setup_alpasim.sh` installs uv
and Rust if they are missing.

## Steps

```bash
# 1. Build the AlpaSim image with the plugin, pull NRE (about 30 min).
bash experiment/closed-loop/setup_alpasim.sh

# 2. Stage the checkpoint and Model/ for the driver container. The checkpoint
#    is in the Platform account; presign it where that account is readable.
CHECKPOINT_URL=<presigned GET url> bash experiment/closed-loop/stage_driver.sh

# 3. Token of a Hugging Face account with access to the NuRec dataset.
install -m 600 /dev/stdin ~/.config/closed-loop/hf_token <<< "<token>"

# 4. Plumbing check without a model, then the checkpoint on all scenes.
CHECKPOINT=MOCK SCENES=clipgt-01d503d4-449b-46fc-8d78-9085e70d3554 \
    RUN_NAME=mock-01d503d4 bash experiment/closed-loop/run_scenes.sh
RUN_NAME=nuplan-ep5-5scenes bash experiment/closed-loop/run_scenes.sh

# 5. Pack a run and upload it with a presigned PUT URL.
RESULTS_PUT_URL=<presigned PUT url> \
    bash experiment/closed-loop/publish_results.sh nuplan-ep5-5scenes
```

Each run lands in `~/closed-loop-runs/<run name>/`:

- `experiment.json`: AlpaSim and auto_e2e commits, checkpoint, scenes
- `rollouts/<scene>/<rollout>/rollout.asl`, `metrics.parquet`, front-camera mp4
- `aggregate/metrics_results.txt`, `aggregate/videos/{all,violations/*}`

## Notes for the PR

- AlpaSim's `.dockerignore` is an allowlist. Copying the plugin into
  `plugins/` is not enough: `!plugins/alpasim_driver/**` must be added, or
  `uv sync` in the image fails with "references a workspace in
  `tool.uv.sources` ... but is not a workspace member".
- `setup_alpasim.sh` also adds `alpasim_autoe2e = { workspace = true }` to
  the root `[tool.uv.sources]`, as AlpaSim does for the transfuser plugin.
  Whether the `autoe2e` extra resolves without it was not tested.

## Closed-loop RL: GRPO on one scene

[`rl/`](./rl) is the smallest RL loop on top of the runs above. It is an
experiment outside `wf_autoe2e`: the checkpoints it writes have no Flyte
lineage and stay on the GPU host. If it proves useful, the loop is redesigned
for the pipeline in `Design/` first.

```bash
# One scene, 3 GRPO steps, 4 s of closed loop per rollout (about 40 min on an L40S).
ITERATIONS=3 bash experiment/closed-loop/rl/rl_loop.sh
python -m pytest experiment/closed-loop/rl/test_grpo.py   # CPU unit tests
```

Each iteration rolls out the current policy with `run_scenes.sh`, then
`rl/grpo_step.py` replays every decision of that rollout, samples a group of
plans per decision, scores them and takes one trust-region step on the GRU
planner. The encoders stay frozen. Each step writes `iter-<n>.pt` and
`iter-<n>.json` (per-decision reward terms, anchor, replay error) under
`$ALPASIM_ROOT/data/drivers/autoe2e/rl/<tag>/`.

Choices and why:

- Reward ([`rl/reward.py`](./rl/reward.py)): progress gated by safety,
  `NC * DAC * DDC * V * EP * TTC * LK * C` over 4 s of each plan. The terms
  are those of NAVSIM's EPDMS (arXiv:2406.15349, arXiv:2506.04218), and the
  product with progress follows CaRL (arXiv:2504.17838): a plan that stops
  scores 0, where a weighted average would still pay for its safety (#123).
  NC and TTC use the logged objects' future boxes, which the policy never
  observes, and DAC, DDC and LK use the vector map rather than the raster the
  policy reads, so every term can change the ranking of plans (#123's
  non-redundancy). V keeps plans within 2.5-3 m of the recorded path, where
  the reconstruction is reliable; as RAD (arXiv:2502.13144) does, it measures
  distance to the path, not to the recorded position at the same time, so
  braking is not penalised. The phase-1 reward of
  `Design/reinforcement_learning.md` §6.1 (distance to the log) is dropped:
  the anchor to the imitation planner already keeps the policy near the log.
  Episodes still end when the ego is more than 3 m from the recording.
- Update rule: GRPO (Shao et al., 2024, arXiv:2402.03300) with groups of 8
  plans sampled at the same closed-loop state, as ReCogDrive
  (arXiv:2506.08052) and AutoVLA (arXiv:2506.13757) score whole trajectories
  as one action. The reward of a plan depends only on the plan and the
  state, so the plans of one state can be compared without rolling each out,
  and no value network is needed. Rollout-level groups would cost one
  5-minute rollout per sample at the current 6.6 s per decision.
- Exploration (#211): the GRU planner is deterministic, so the policy adds
  Gaussian noise to 4 low-frequency DCT coefficients of each control signal,
  acceleration and curvature. Samples differ by smooth trajectory-level
  changes rather than per-step jitter, the structured exploration #211 asks
  for. It does not follow a published recipe. The same perturbation works on
  Bézier control points, so it does not decide #211's planner question.
- On-policy states: AlpaSim executes the mean plan, and the samples are drawn
  at the states that plan reached; `grpo_step.py` stops if the replayed mean
  differs from the executed plan by more than 5 cm. The drift of #221 brings
  the ego to states away from the recording, where samples that recover score
  higher, the signal DAgger (Ross et al., 2011) gets from an expert.
- Step: the natural gradient with respect to the mean plan
  (Fisher-preconditioned, so the smaller-sigma signal is not favoured), passed
  to the planner's parameters by back-propagation, plus a pull towards the
  imitation planner over all 64 steps. The step is halved until the new mean
  plan stays within a trust region of the old one on every scored state, as
  in TRPO. Without it the first run's steps diverged (see RESULTS.md).
- Gated plans score `-1 + safe_fraction`, the share of their steps before
  the first violation, so a plan that collides never ranks above one that
  merely stops, and among plans that all fail, the one that fails later ranks
  higher, as CaRL's episode ending at a violation implies. Without it, the
  imitation policy's plans left the lanes within 4 s at most states and whole
  groups scored the same.

Limits: the reward of a decision ignores what the executed plan does to later
decisions, so this is a contextual bandit on visited states, not credit
assignment over the episode. Logged objects do not react to the ego. The
renderer bound keeps the policy near the recorded path. Collisions are
checked at the 0.1 s steps, not swept between them. The route corridor covers
AlpaSim's 80 m route, where nuPlan training drew every route roadblock. Scene
maps are rasterized whole at every decision (about 0.4 s for 400 lanes).
