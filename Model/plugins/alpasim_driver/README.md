# AutoE2E AlpaSim Driver Plugin

This package plugs the AutoE2E reactive policy into [NVIDIA AlpaSim](https://github.com/NVlabs/alpasim) as a driver model, for closed-loop evaluation and RL rollouts.

It drives with stage 2 AutoE2E checkpoints written by `training/trainer.py` (AutoBEV camera encoder, six cameras at 1344x768, T8 history).

## How it works

```mermaid
graph TD
    AlpaSim[AlpaSim runtime] -->|8 frames per camera, 500 ms apart + ego poses| Driver[AutoE2EDriver]
    Driver --> Parser[AlpasimStreamParser]
    Parser -->|1344x768 crops, T8 history, projections, egomotion| Model[AutoE2E]
    Model -->|64 x acceleration, curvature| Rollout[integrate_controls_torch]
    Rollout -->|ModelPrediction anchored at t0| AlpaSim
```

- **`AutoE2EDriver`** ([`plugin.py`](./alpasim_autoe2e/plugin.py)) is an AlpaSim `BaseTrajectoryModel`, registered as `autoe2e` under the `alpasim.models` entry point. It loads the checkpoint into `AutoE2E` after checking its AutoBEV geometry contract, calls `AutoE2E.forward` with named inputs, and integrates the predicted controls with the training rollout (`training/losses/control_rollout.py`). The result is returned as waypoints anchored at `t0`, the timestamp of the newest camera frame.
- **`AlpasimStreamParser`** ([`parser.py`](./alpasim_autoe2e/parser.py)) builds the tensors of an AutoBEV sample (`data_processing/autobev_members.py`):

| Input | Built from |
| :--- | :--- |
| `images` `[1,6,3,768,1344]` | Newest frame of each camera in the canonical slot order, scaled to cover 1344x768 with the aspect ratio kept, centre-cropped (`crop_params`), ImageNet-normalised. |
| `history_images` `[1,7,6,3,768,1344]` | The seven previous frames, 500 ms apart, oldest first. |
| `projection`, `history_projection` | The calibrated intrinsics scaled to the rendered frame, then the crop transform of `crop_projection`. History projections map current-ego points into each past crop, using the ego poses interpolated at the frame timestamps. |
| `egomotion_history` `[1,256]` | 64 rows of `[speed, acceleration, yaw_rate, curvature]` before `t0`, from the ego poses by `np.gradient` on a 10 Hz grid. `AutoE2E` scales speed and acceleration itself. |
| `map_context`, `route_mask` | All zeros, with `map_valid = route_valid = False`. AlpaSim's model API passes no map. |
| `map_pose`, `map_pose_covariance`, `map_matching` | Zero, because the simulator pose is exact; a fixed diagonal covariance of 1 m, 1 m and 1° (`MAP_POSE_SIGMA`, the nominal accuracy of the training EKF); `True`. |

## Rig profiles

[`configs/calibration/`](./alpasim_autoe2e/configs/calibration/) holds one JSON file per dataset rig. Each file stores native intrinsics, image size, camera-to-reference extrinsics and the model slot of every camera. The driver picks the profile whose cameras equal `driver.inference.use_cameras`.

- `kit_scenes.json` was exported with [`tools/export_kitscenes_calibration.py`](./tools/export_kitscenes_calibration.py). It matches 89 of the 117 KITScenes validation scenes. The other 28 use two later calibration revisions (25 and 3 scenes), each about 7 cm and 1° from it. For a world recorded with a later revision, export its profile and set `AUTOE2E_RIG_PROFILE=/path/to/profile.json`.
- `nuplan.json` is the nuPlan v1.1 rig of vehicle veh-35, exported from one of its log databases with [`tools/export_nuplan_calibration.py`](./tools/export_nuplan_calibration.py). The cameras keep their channel names (`CAM_F0`, ...). Training undistorts nuPlan images, so the profile holds the rectified pinhole intrinsics, and the world must render rectified cameras. All vehicles except veh-08 share the intrinsics, and their extrinsics differ by at most 9 cm and 2.8°. For a world recorded with another vehicle, export that vehicle's profile and set `AUTOE2E_RIG_PROFILE`.
- `nuplan_nurec.json` is the same rig on NVIDIA camera IDs, for NuRec scenes of the NVIDIA rig (see `driver=autoe2e_nuplan` below).
- A profile also states the dataset's curvature rule (`egomotion`): KITScenes divides by `max(speed, 0.1)`, while nuPlan zeroes curvature below 0.5 m/s and clips it to ±0.5.
- The reference frame of a profile must be the rig frame of the rendered world. For KITScenes, that is the frame of `poses.txt`; for nuPlan, the ego pose.

## Configuration

`driver=autoe2e` selects [`configs/driver/autoe2e.yaml`](./alpasim_autoe2e/configs/driver/autoe2e.yaml). Its runtime settings are in [`autoe2e_configs.yaml`](./alpasim_autoe2e/configs/driver/autoe2e_configs.yaml).

| Setting | Value | Why |
| :--- | :--- | :--- |
| `driver.model.checkpoint_path` | required | A stage 2 AutoE2E checkpoint. `MOCK` drives straight without a model, to test the simulator plumbing with any driver config; `UNTRAINED` builds AutoE2E with random weights. |
| `driver.inference.context_length`, `subsample_factor` | `8`, `5` | Every 5th of the 10 Hz frames: the T8 history at 500 ms. |
| `runtime.simulation_config.force_gt_duration_us` | `7_000_000` | The ego follows the recording for 7 s, so the 6.4 s egomotion history and the 3.5 s camera history are real before AutoE2E drives. |
| `runtime.simulation_config.cameras` | native size | The plugin scales each frame to cover 1344x768 and centre-crops it, as the AutoBEV packs do. |
| `services.driver.environments` | `AUTOE2E_MODEL_ROOT=/mnt/drivers/autoe2e/Model` | Where the driver container finds the AutoE2E model code. |

`driver=autoe2e_nuplan` drives the nuPlan v1.1 rig in NuRec scenes of the NVIDIA camera rig. AlpaSim's `runtime.extra_cameras` can only override cameras a scene already has. So the six nuPlan cameras are rendered as rectified pinhole cameras on six NVIDIA camera IDs, the way AlpaSim's Transfuser plugin renders its nuPlan-style rig, and `nuplan_nurec.json` maps them back. `nuplan.json` keeps the `CAM_*` names for worlds reconstructed from nuPlan. [`tools/profile_to_extra_cameras.py`](./tools/profile_to_extra_cameras.py) regenerates them from the profile, and a test keeps the two identical.

`driver.model` may only hold keys of AlpaSim's `ModelConfig`; the driver rejects any other key at startup.

## Running in AlpaSim

The plugin imports the AutoE2E model code from the repository's `Model/` directory. In-repo it finds it on its own; elsewhere, set `AUTOE2E_MODEL_ROOT`.

```bash
# 1. Checkpoint and model code under the wizard's defines.filesystem/drivers (/mnt/drivers in the container)
mkdir -p "$ALPASIM_FILESYSTEM/drivers/autoe2e"
cp /path/to/checkpoint.pt "$ALPASIM_FILESYSTEM/drivers/autoe2e/"
rsync -a --exclude tests Model/ "$ALPASIM_FILESYSTEM/drivers/autoe2e/Model/"

# 2. The plugin as an AlpaSim workspace member (Docker cannot follow symlinks out of its context)
rm -rf "$ALPASIM_ROOT/plugins/alpasim_driver"
cp -r Model/plugins/alpasim_driver "$ALPASIM_ROOT/plugins/"
```

Next, add `autoe2e = ["alpasim_autoe2e"]` to the plugin extras in `$ALPASIM_ROOT/pyproject.toml`, and `--extra autoe2e` to the `uv sync` line of its `Dockerfile`. Then build the image and run:

```bash
cd "$ALPASIM_ROOT" && uv run --project src/wizard alpasim_wizard \
    deploy=local topology=1gpu driver=autoe2e_nuplan \
    driver.model.checkpoint_path=/mnt/drivers/autoe2e/checkpoint.pt \
    defines.base_image=alpasim-base:latest \
    wizard.log_dir=$PWD/outputs/autoe2e_closed_loop_run \
    "hydra.searchpath=[file://$ALPASIM_ROOT/plugins/alpasim_driver/alpasim_autoe2e/configs]"
```

The wizard runs on the host, where the plugin is not installed, so `hydra.searchpath` points it at the plugin's configs; the driver itself runs inside the image.

The scene must render the rig's cameras. For KITScenes, that means a reconstruction that contains the six KITScenes cameras.

## Tests

```bash
cd Model
python -m pytest tests/test_alpasim_stream.py tests/test_rewards.py                   # unit tests
AUTOE2E_CHECKPOINT=/path/to/checkpoint.pt \
python -m pytest tests/test_alpasim_stream.py -m integration                          # real model
```

## Rewards

[`rewards.py`](./alpasim_autoe2e/rewards.py) holds the phase-1 reward agreed in [#123](https://github.com/autowarefoundation/auto_e2e/issues/123) and [#177](https://github.com/autowarefoundation/auto_e2e/pull/177): `GroundTruthDeviationReward`.

- **Tracking term:** for each AutoE2E decision, `-(w_ade·ADE + w_fde·FDE)` between the plan and the recorded drive at the same timestamps.
- **Renderer validity bound:** once the ego itself is more than 3 m from the recording, a −10 penalty is added and `truncated` is set. The reconstruction is only reliable near the recorded path, so this bound does not share the tracking weight.

[`tools/score_rollout.py`](./tools/score_rollout.py) applies the reward to an AlpaSim `rollout.asl`. The driver does not call the reward; the RL loop will (see `Design/reinforcement_learning.md`, which lists the further terms as future work).
