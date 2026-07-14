# AutoE2E Architecture

## Architecture Diagram
> **Note:** The architecture diagram is outdated and does not reflect the current implementation (separate map encoder branch, BEV fusion as default, residual map fusion). It will be updated in a follow-up PR.

<img src="../Media/auto_e2e_architecture.jpg" width="100%">

## Inputs and Predictions
**AutoE2E consumes as input:**
- 7 camera images at 256x256 resolution (providing a surround view of the vehicle)
- Rendered map tile (indicating the high level road network layout and future route of the vehicle)
- Egomotion history (speed, acceleration, yaw angle and yaw angle rate for the previous 6.4s at 10Hz sampling rate)
- Visual history (`(896,)` = 64 frames × 14-dim compressed scene memory; provides frame-to-frame visual context, distinct from the planner GRU's intra-trajectory temporal coherence)

**AutoE2E outputs a prediction of:**
- Future driving trajectory (modelled as future acceleration and curvature values over a 6.4s future horizon at 10Hz sampling rate)

**During training, and for purposes of model introspection, AutoE2E also predicts:**
- Future visual features at 1.6s intervals for a 6.4s future horizon (what does the future feature representation of the scene look like, this is used for a feature reconstruction loss similar to JEPA). Opt-in via `enable_world_model=True`.

**Forward signature:**
```python
trajectory = model(
    camera_tiles,        # (B, V, 3, H, W) — 7 cameras
    map_input,           # (B, 3, H_map, W_map) — BEV nav-map image
    visual_history,      # (B, 896) — frame-to-frame visual memory
    egomotion_history,   # (B, 256)
    projection=None,     # CameraProjectionModel operator — the geometry ABI
    geometry_type=None,  # "pinhole" | "rectified_pinhole" | "ftheta" | "pseudo"
    mode="infer",
)
```

`forward` returns the trajectory `(B, num_timesteps * num_signals)`, or the tuple
`(trajectory, future_state_pred)` when the World Model is enabled **and**
`mode="train"`. The pre-#94 3-tuple return no longer exists.

**Camera geometry is an operator, not a matrix.** There is no `camera_params`
argument: `#77`/`#107` replaced the `(B, V, 3, 4)` matrix with a projection
operator, so that fisheye (F-Theta) calibration is expressible and the
geometry travels with the data.

```python
from model_components.view_fusion.projection import PinholeProjection

trajectory = model(
    camera_tiles, map_input, visual_history, egomotion_history,
    projection=PinholeProjection(camera_params),  # (B, V, 3, 4) intrinsic @ extrinsic
    geometry_type="pinhole", mode="infer",
)
```

Passing no `projection` falls back to `geometry_type="pseudo"` — a **learned
spatial prior, not real geometry** (shape-testing and ablation only). Passing the
removed `camera_params=` argument now raises `TypeError` rather than being
absorbed by `**kwargs` and silently dropping your calibration.

**To learn the driving policy:**
- Imitation Learning is used to penalize trajectory prediction as well as World Model Simulation based Reinforcement Learning


