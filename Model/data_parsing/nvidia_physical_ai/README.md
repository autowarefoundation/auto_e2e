# Data parsing recipe for [NVIDIA Physical AI Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

Parser for the NVIDIA Physical AI dataset, producing tensors directly consumable by AutoE2E's `forward()`.

Download (a subset of) the dataset into `data`.

## Model inputs produced

- `visual_tiles` `(8, 3, H, W)` — 7 camera frames + 1 map tile placeholder
- `egomotion_history` `(256,)` — 64 past timesteps × 4 signals at 10 Hz
- `visual_history` `(896,)` — zero-initialised; populated during sequential inference
- `trajectory_target` `(128,)` — 64 future timesteps × 2 signals (supervision target)

### Egomotion history signals `(256,) = 64 × 4`

- `[0]` Speed (m/s) — `sqrt(vx^2 + vy^2)`
- `[1]` Acceleration (m/s^2) — `ax`
- `[2]` Yaw angle (rad) — quaternion → ZYX Euler via `EgomotionState.pose`
- `[3]` Curvature (rad/m) — `curvature`

### Trajectory target signals `(128,) = 64 × 2`

- `[0]` Acceleration (m/s^2) — `ax`
- `[1]` Curvature (rad/m) — `curvature`

## Sampling

A sample is defined by a `sample_idx` — an index into the 10 Hz downsampled egomotion sequence. A `sample_idx` is valid when there are at least 64 rows behind it (history window) and 64 rows ahead (target window). For a typical 20s clip (~200 rows at 10 Hz) this gives ~72 valid sample points per clip.

All valid `(clip_uuid, sample_idx)` pairs are enumerated at dataset construction time. `__getitem__` does I/O only — no index arithmetic at call time.

## Usage

```python
from data_parsing.nvidia_physical_ai import NvidiaAVDataset
from torch.utils.data import DataLoader

dataset = NvidiaAVDataset(
    data_root="/path/to/nvidia_physical_ai_dataset",
    backbone_name="swin_tiny_patch4_window7_224.ms_in22k",
)

# Single clip for forward pass validation
dataset = NvidiaAVDataset(
    data_root="/path/to/nvidia_physical_ai_dataset",
    backbone_name="swin_tiny_patch4_window7_224.ms_in22k",
    clip_uuids=["fd1d1b6b-59bf-4292-8295-5028aa6aa5e3"],
)

loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

for batch in loader:
    visual_tiles      = batch["visual_tiles"].to(device)       # (B, 8, 3, H, W)
    visual_history    = batch["visual_history"].to(device)     # (B, 896)
    egomotion_history = batch["egomotion_history"].to(device)  # (B, 256)
    trajectory_target = batch["trajectory_target"].to(device)  # (B, 128)

    trajectory, compressed, future = model(visual_tiles, visual_history, egomotion_history)
    loss = criterion(trajectory, trajectory_target)
```

## Image preprocessing

Preprocessing is derived at runtime from the backbone's own config via timm:

```python
data_config = timm.data.resolve_model_data_config(backbone)
transform = timm.data.create_transform(**data_config, is_training=False)
```

This means normalisation, resize, and crop parameters always match the backbone's training. If the backbone changes, pass a different `backbone_name` to `NvidiaAVDataset`.

## Forward pass test

```bash
cd Model/data_parsing/nvidia_physical_ai
python forward_pass_test.py \
    --dataset_root data \
    --clip_uuid fd1d1b6b-59bf-4292-8295-5028aa6aa5e3
```

## Additional dependencies

```
physical_ai_av   # NVIDIA dataset SDK (pip install git+https://github.com/NVlabs/physical_ai_av.git); requires python >= 3.11
timm             # backbone config and transforms
pandas
```