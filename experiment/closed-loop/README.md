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
- The root `pyproject.toml` needs `alpasim_autoe2e = { workspace = true }` in
  `[tool.uv.sources]` next to the `autoe2e` extra.
