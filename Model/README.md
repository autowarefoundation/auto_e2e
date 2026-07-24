# AutoE2E Model

## Runtime Composition

`AutoE2E` is the public model. It composes:

- `ReactiveE2E`: the camera, map, temporal-memory, optional reasoning, and
  trajectory-planning path.
- `WorldActionModel`: an optional visual-history and JEPA branch that shares the
  reactive camera backbone.
- `RollingHistoryBuffer`: inference-time state used by the World Model when the
  caller does not supply an explicit history window.

The default model uses BEV camera fusion, residual map fusion, `NoMemory`, and
`BezierPlanner`. The World Model and reasoning branch are disabled by default.
The active planner is available through the read-only
`model.trajectory_planner` property.

## Inputs

`AutoE2E.forward` consumes:

- `camera_tiles`: `[B, V, 3, H, W]` real camera images.
- `map_input`: `[B, 3, H_map, W_map]` rasterized BEV navigation map.
- `visual_history`: `[B, 896]` or `[B, T, 896]`.
- `egomotion_history`: `[B, 256]` or `[B, T, 256]`.
- `projection`: an optional `CameraProjectionModel` implementation such as
  `PinholeProjection` or `FThetaProjection`.
- `geometry_type`: the matching geometry label. With no calibrated projection,
  use `"pseudo"` or omit the label.
- `history_frames`: optional explicit World Model history
  `[B, T, V, 3, H, W]`.

The navigation map is a separate input branch. It is not counted as a camera
view.

## Inference

Inference always returns one flattened control tensor:

```python
model.eval()
with torch.inference_mode():
    trajectory = model(
        camera_tiles=camera_tiles,
        map_input=map_input,
        visual_history=visual_history,
        egomotion_history=egomotion_history,
        projection=projection,
        geometry_type=geometry_type,
        history_frames=history_frames,
        mode="infer",
    )
```

`trajectory` has shape `[B, num_timesteps * num_signals]`, which is `[B, 128]`
with the defaults. It represents 64 future `(acceleration, curvature)` control
pairs, not Cartesian waypoints.

`mode="infer"` selects the model's return contract. It does not switch PyTorch
layers to evaluation behavior, so inference callers must also call
`model.eval()`.

When the World Model is enabled, supplying `history_frames` uses the stateless
windowed path. Omitting it updates the model's rolling FIFO; call
`model.reset_visual_history()` between independent sequences.

## Training Return Contract

With both optional branches disabled, training also returns only `trajectory`.
When the World Model or reasoning branch is enabled, training returns:

```python
trajectory, aux_outputs = model(..., mode="train")
```

`aux_outputs` contains:

- `future_state_pred`: predicted future backbone feature maps, or `None`.
- `future_frames`: the caller-provided JEPA targets, or `None`.
- `reasoning_pred`: `HorizonReasoningPrediction`, or `None`.

Training is orchestrated through the Flyte tasks in
`Platform/pipelines/workflows.py`; there is no standalone local training entry
point.
