# Data parsing recipe for [KIT Scenes Multimodal Dataset](https://kitscenes.com/multimodal/)

Parser for the KITScenes Multimodal dataset, producing tensors directly consumable by AutoE2E's `forward()`.

KITScenes ships per-frame JPEGs (not video), 6-DOF ego poses in `poses.txt`, and Lanelet2 HD maps. Cameras and ego poses share the 10 Hz reference timeline, so a single `frame_idx` indexes both. The four egomotion signals are *derived* from the pose stream by finite differencing — KITScenes does not store velocity/acceleration/curvature columns.

## Setup

The dataset is loaded through the [`kitscenes`](https://github.com/KIT-MRT/kitscenes) Python API (alpha, `0.1.x`). It is not on PyPI — install from source. Constraints: Linux x86_64, Python 3.8–3.12, `numpy<2.0`.

```bash
git clone https://github.com/KIT-MRT/kitscenes.git && cd kitscenes
pip install --upgrade pip
pip install -e ".[map]"   # [map] adds the vendored Lanelet2 wheel needed for the BEV map tile
```

The `[map]` extra installs the SDK's vendored Lanelet2 build, which the BEV map rasteriser needs. Without it the parser still runs — slot 7 (the map tile) falls back to a blank white tile rather than rasterised geometry. Use `pip install -e ".[all]"` for the full SDK.


## Model inputs produced

- `visual_tiles` `(8, 3, H, W)` — 7 camera frames + 1 BEV map tile
- `egomotion_history` `(256,)` — 64 past timesteps × 4 signals at 10 Hz
- `visual_history` `(896,)` — zero-initialised placeholder; populated during sequential inference
- `trajectory_target` `(128,)` — 64 future timesteps × 2 signals (supervision target)

The 7 camera views are the hi-res front camera plus the 6 surround ring cameras (`CAMERA_NAMES` in `camera.py`). The stereo front pair is dropped as it duplicates forward coverage.

### Egomotion history signals `(256,) = 64 × 4`

All four are derived from `poses.txt` (TUM format, UTM frame) by finite differencing on the real, possibly uneven, pose timestamps.

- `[0]` Speed (m/s) — `‖d/dt translation_xy‖`
- `[1]` Acceleration (m/s^2) — `d/dt speed`
- `[2]` Yaw angle (rad) — quaternion → ZYX Euler, Z component
- `[3]` Curvature (1/m) — `yaw_rate / speed`, with speed floored at 0.1 m/s

### Trajectory target signals `(128,) = 64 × 2`

- `[0]` Acceleration (m/s^2)
- `[1]` Curvature (1/m)

This matches `DrivingPolicy`'s `fc3` output exactly.

## Sampling

A sample is a `(scene_id, frame_idx)` pair, where `frame_idx` indexes the 10 Hz reference timeline. A `frame_idx` is valid when there are 64 frames behind it (history window) and 64 ahead (target window), within the span covered by *both* the ego poses and the camera frames. The current frame is excluded from both windows — history is `[idx-64, idx)`, target is `(idx, idx+64]`. A scene with `N` usable reference frames therefore yields `N − 128` valid samples.

All valid pairs are enumerated at construction time, and the per-scene derived egomotion and UTM translation arrays are cached then. `__getitem__` does I/O only — no pose re-reads or index arithmetic at call time.

## Usage

```python
from data_parsing.kit_scenes import KitScenesDataset
from torch.utils.data import DataLoader

# Whole split for training
dataset = KitScenesDataset(
    data_root="/path/to/kitscenes",   # the $KITSCENES_ROOT directory
    backbone_name="swinv2_tiny_window8_256",
    split="train",
)

# Single scene for forward pass validation
dataset = KitScenesDataset(
    data_root="/path/to/kitscenes",
    backbone_name="swinv2_tiny_window8_256",
    scene_ids=["c34c778f-ad8c-0aa9-7e1a-c86a73f887c7"],
)

loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

for batch in loader:
    visual_tiles = batch["visual_tiles"].to(device)           # (B, 8, 3, H, W)
    visual_history = batch["visual_history"].to(device)       # (B, 896)
    egomotion_history = batch["egomotion_history"].to(device) # (B, 256)
    trajectory_target = batch["trajectory_target"].to(device) # (B, 128)

    trajectory, compressed, future = model(visual_tiles, visual_history, egomotion_history)
    loss = criterion(trajectory, trajectory_target)
```

## Image preprocessing

Preprocessing is derived at runtime from the backbone's own config via timm, so normalisation, resize, and crop always match the backbone's training:

```python
data_config = timm.data.resolve_model_data_config(backbone)
transform = timm.data.create_transform(**data_config, is_training=False)
```

The BEV map tile is passed through the same transform as the camera frames, so all 8 views share one shape and normalisation. Change the backbone by passing a different `backbone_name`.

## BEV map tile (slot 7)

`map.generate_bev_map_tile` rasterises a semantic, ego-centric tile from the scene's Lanelet2 HD map using OpenCV: road borders, lane dividers, centerlines, stop lines, and pedestrian crossings, coloured to mirror the SDK's `ml_converter` conventions. The tile is rotated so the ego heading points up (forward → up, left → left).

Ego poses and the lanelet boundaries share the world frame (UTM 32N, metres); the tile is centred by subtracting the ego position, so the absolute UTM offset cancels. If the map is unavailable (`[map]` extra missing, or no `map.osm`), the tile is a blank white canvas. Inspect a tile with `map.visualise_bev_tile`.

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
python forward_pass_test.py --dataset_root "$KITSCENES_ROOT" --split test_e2e
python forward_pass_test.py --dataset_root "$KITSCENES_ROOT" --scene_id <uuid> --no-pretrained
```

The test asserts input and output shapes (including that the trajectory head emits `(B, 128)`), so a dimension regression fails loudly instead of printing a wrong shape.