# Data parsing recipe for [KIT Scenes Multimodal Dataset](https://kitscenes.com/multimodal/)

Parser for the KITScenes Multimodal dataset, producing tensors directly consumable by AutoE2E's `forward()`.

KITScenes ships per-frame JPEGs (not video), 6-DOF ego poses in `poses.txt`, and Lanelet2 HD maps. Cameras and ego poses share the 10 Hz reference timeline, so a single `frame_idx` indexes both. The four egomotion signals are *derived* from the pose stream by finite differencing — KITScenes does not store velocity/acceleration/curvature columns.

## Setup

The dataset is loaded through the [`kitscenes`](https://github.com/KIT-MRT/kitscenes) Python API. Constraints: Linux x86_64, Python 3.8–3.12.

### KITScenes SDK

```bash
cd Model/data_parsing/kit_scenes
git clone https://github.com/KIT-MRT/kitscenes.git && cd kitscenes
pip install --upgrade pip
pip install -e . --no-deps
```

### Lanelet2

`generate_bev_map_tile` in `map.py` calls `get_lanelets_in_roi` which relies on `lanelet2` to parse the underlying vector map data. Install `lanelet2` using `pip`:
```bash
pip install lanelet2
``` 

If you encounter errors like `No matching distribution found for lanelet2`, install the pre-compiled C++ binaries and Python bindings using the RoboStack Conda channel instead:

```bash
conda install -c robostack-staging ros-humble-lanelet2-python
```

Without Lanelet2, map tiles fall back to zero tensors and the dataset still functions.


## Model inputs produced

- `visual_tiles` `(7, 3, H, W)` — 7 camera frames
- `map_tile` `(3, H, W)` — BEV map tile (rasterization of Lanelet2 HD map)
- `egomotion_history` `(256,)` — 64 past timesteps × 4 signals at 10 Hz
- `visual_history` `(896,)` — zero-initialised placeholder; populated during sequential inference
- `trajectory_target` `(128,)` — 64 future timesteps × 2 signals (supervision target)
- `camera_params` `(7, 3, 4)` — projection matrices `P = K_scaled @ T_ref_to_cam` for the 7 camera views, computed from KITScenes calibration and scaled to match the backbone's resize/crop transform. 


The 7 camera views are the hi-res front camera plus the 6 surround ring cameras (`CAMERA_NAMES` in `camera.py`). The stereo front pair is dropped as it duplicates forward coverage.

### Egomotion history signals `(256,) = 64 × 4`

All four are derived from `poses.txt` (TUM format) by finite differencing on the real, possibly uneven, pose timestamps.

- `[0]` Speed (m/s) — `‖d/dt translation_xy‖`
- `[1]` Acceleration (m/s^2) — `d/dt speed`
- `[2]` Yaw angle (rad) — quaternion → ZYX Euler, Z component
- `[3]` Curvature (1/m) — `yaw_rate / speed`, with speed floored at 0.1 m/s

### Trajectory target signals `(128,) = 64 × 2`

- `[0]` Acceleration (m/s^2)
- `[1]` Curvature (1/m)

## Sampling

A sample is a `(scene_id, frame_idx)` pair, where `frame_idx` indexes the 10 Hz reference timeline. A `frame_idx` is valid when there are 64 frames behind it (history window) and 64 ahead (target window), within the span covered by *both* the ego poses and the camera frames. The current frame is excluded from both windows — history is `[idx-64, idx-1]`, target is `[idx+1, idx+64]`. A scene with `N` usable reference frames therefore yields `N − 128` valid samples.

All valid pairs are enumerated at construction time, and per-scene derived arrays (egomotion, scene-local positions, camera projection matrices) are cached then. `__getitem__` does I/O only.


## Usage

```python
from data_parsing.kit_scenes import KitScenesDataset
from torch.utils.data import DataLoader

# Whole split for training
dataset = KitScenesDataset(
    data_root="/path/to/kitscenes/data",   # the $KITSCENES_ROOT directory
    backbone_name="swinv2_tiny_window8_256",
    split="train",
)

# Single scene for forward pass validation
dataset = KitScenesDataset(
    data_root="/path/to/kitscenes/data",
    backbone_name="swinv2_tiny_window8_256",
    scene_ids=["c34c778f-ad8c-0aa9-7e1a-c86a73f887c7"],
)

loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# Training loop
for batch in loader:
    visual_tiles = batch["visual_tiles"].to(device)           # (B, 7, 3, H, W)
    map_tile = batch["map_tile"].to(device)                   # (B, 3, H, W)
    visual_history = batch["visual_history"].to(device)       # (B, 896)
    egomotion_history = batch["egomotion_history"].to(device) # (B, 256)
    trajectory_target = batch["trajectory_target"].to(device) # (B, 128)
    camera_params = batch["camera_params"].to(device)         # (B, 7, 3, 4)

    planner_loss, ego_hidden, future_visual_features = model(
        visual_tiles,
        map_tile,
        visual_history,
        egomotion_history,
        camera_params=camera_params,
        trajectory_target=trajectory_target,
        mode="train",
    )

# Inference
trajectory, ego_hidden, _ = model(
    visual_tiles,
    map_tile,
    visual_history,
    egomotion_history,
    camera_params=camera_params,
    mode="infer",
)
```

## Image preprocessing

Preprocessing is derived at runtime from the backbone's own config via timm, so normalisation, resize, and crop match the backbone's training:

```python
data_config = timm.data.resolve_model_data_config(backbone)
transform = timm.data.create_transform(**data_config, is_training=False)
```

## BEV map tile

### Rasterized maps (temporary, pending vectorized encoder)

`map.generate_bev_map_tile` rasterizes a semantic, ego-centric tile from the scene's Lanelet2 HD map using OpenCV: road borders, lane dividers, centerlines, stop lines, and pedestrian crossings, coloured to mirror the SDK's `lanelet2_ml_converter` conventions. The tile is rotated so the ego heading points up (forward = up, left = left).

Ego poses and the lanelet boundaries share the scene-local frame (metres from the map origin, axes aligned with UTM 32N); the tile is centred by subtracting the ego position, so the absolute offset cancels. 

**Important:** This rasterized map implementation for this dataset is **only applicable if the model will train using raster map representations**. The architecture is designed to support both vectorized (implementation pending) and rasterized map encoders. The `map.py` module serves as a reference implementation should rasterized maps be needed in the future.


### Benchmarking on-demand vs. pre-rendered maps

**Note:** The `rasterize_map_at_runtime` flag is provided for benchmarking. Currently, setting it to `False` (by running `forward_pass_test` with the `--no-rasterize-maps` flag) uses zero tensor fallback; actual pre-rendered raster map tile loading is not yet implemented.

- `rasterize_map_at_runtime=True` (default): On-demand runtime rasterization of Lanelet2 map tiles via `map.generate_bev_map_tile`
- `rasterize_map_at_runtime=False`: Zero tensor fallback, representative of scenarios without map data or pre-rendered/cached maps

## Forward pass test

Run from the repo root's `Model` directory. Use the absolute `$KITSCENES_ROOT` for `--dataset_root`:

```bash
cd Model/data_parsing/kit_scenes
python forward_pass_test.py \
    --dataset_root "$KITSCENES_ROOT" \
    --scene_id c34c778f-ad8c-0aa9-7e1a-c86a73f887c7
```

Whole split, or offline/CI without pretrained weights:

```bash
python forward_pass_test.py --dataset_root "$KITSCENES_ROOT" --split train
python forward_pass_test.py --dataset_root "$KITSCENES_ROOT" --scene_id <uuid> --no-pretrained
```

The test exercises both inference (`mode="infer"`) and training (`mode="train"`) modes with BEV fusion, including camera projection matrices.
