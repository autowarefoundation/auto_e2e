# AutoE2E Architecture

`AutoE2E` (`model_components/auto_e2e.py`) wraps `ReactiveE2E`
(`model_components/reactive_e2e.py`). The design is in
`Design/autobev_integration.md`; the v1.0 model is the code tagged `v1.0`.

| Component | Module |
|-----------|--------|
| Camera BEV encoder | `AutoBEV` (`model_components/autobev/`): DINOv2 ViT-B/14, BEVFormer V2 t1 encoder and T8 temporal fusion on a 220x160 grid at 0.5 m |
| Map and route encoders | `SplitNavigationEncoder` with the `semantic_raster` map encoder and a lightweight route encoder |
| Map BEV fusion | deformable cross-attention, on the raster resampled at the estimated crop pose (`navigation/resample.py`, `navigation/map_matching.py`) |
| Trajectory planner | `GRUPlanner` |
| Auxiliary heads | AutoBEV BEV segmentation, centres, depth, boxes and image-space map segmentation; route reconstruction; crop pose error |

## Inputs and Predictions

`AutoE2E.forward` takes named tensors:

```python
controls, auxiliary = model(
    egomotion_history=egomotion_history,    # (B, 256): 64 x [speed, acceleration, yaw_rate, curvature]
    images=images,                          # (B, 6, 3, 768, 1344), ImageNet-normalised
    history_images=history_images,          # (B, 7, 6, 3, 768, 1344), 0.5 s apart, oldest first
    projection=projection,                  # (B, 6, 3, 4): current-ego points to image pixels
    history_projection=history_projection,  # (B, 7, 6, 3, 4): current-ego points to past images
    map_context=map_context,                # (B, 14, 450, 300) navigation raster at 0.4 m
    route_mask=route_mask,                  # (B, 2, 450, 300) route corridor and destination
    map_valid=map_valid, route_valid=route_valid,  # (B,) bool
    map_pose=map_pose,                      # (B, 3) crop pose (x, y, yaw) in the raster frame
    map_pose_covariance=map_pose_covariance,  # (B, 3, 3)
    map_matching=map_matching,              # (B,) bool: refine the crop pose against the map
)
```

The six cameras are scaled to cover 1344x768 with the aspect ratio kept and
centre-cropped (`data_processing/autobev_members.py`). `controls` is
`(B, 64 * 2)` acceleration and curvature over 6.4 s at 10 Hz;
`training/losses/control_rollout.py` integrates them to ego XY.

## Training

`training/trainer.py` trains one stage on Ray: stage 1 trains AutoBEV alone,
stage 2 the whole model from a stage 1 checkpoint. `wf_autoe2e`
(`Platform/pipelines/autoe2e.py`) runs it and validates every checkpoint with
`evaluation/checkpoint_validation.py`; `Platform/HowToUseFlyte.md` shows how to
launch it.
