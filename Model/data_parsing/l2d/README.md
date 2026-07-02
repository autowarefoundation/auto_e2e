# Data parsing recipe for [yaak-ai/L2D](https://huggingface.co/datasets/yaak-ai/L2D)

Parser for the L2D LeRobot dataset (v3.0), producing tensors directly consumable by AutoE2E's `forward()`.

## Dataset overview

- 100,000 episodes at 10 FPS
- 7 camera views: 6 surround cameras + 1 BEV map (640×360)
- Vehicle state: float32 shape[8] — speed, heading, heading_error, lat, lon, alt, accel_x, accel_y
- Waypoints: float32 shape[10,2] — GPS lon/lat
- Actions: float32 shape[3] — gas, brake, steering

## Model inputs produced

- `visual_tiles` `(7, 3, H, W)` — 7 camera views (6 cameras + 1 BEV map)
- `egomotion_history` `(256,)` — 64 past timesteps × 4 signals at 10 Hz
- `visual_history` `(896,)` — zero-initialised; populated during sequential inference
- `trajectory_target` `(128,)` — 64 future timesteps × 2 signals (supervision target)

### Egomotion history signals `(256,) = 64 × 4`

| Index | Signal | Source |
|-------|--------|--------|
| 0 | Speed (m/s) | `observation.state.vehicle[0]` directly |
| 1 | Acceleration_x (m/s²) | `observation.state.vehicle[6]` directly |
| 2 | Yaw rate (rad/s) | `diff(heading) / dt` |
| 3 | Curvature (rad/m) | `yaw_rate / speed` (guarded) |

### Trajectory target signals `(128,) = 64 × 2`

| Index | Signal | Source |
|-------|--------|--------|
| 0 | Acceleration_x (m/s²) | From future frames |
| 1 | Curvature (rad/m) | From future frames |

## Camera views

| Key | Description |
|-----|-------------|
| `observation.images.front_left` | Front left camera |
| `observation.images.left_forward` | Left forward camera |
| `observation.images.right_forward` | Right forward camera |
| `observation.images.left_backward` | Left backward camera |
| `observation.images.rear` | Rear camera |
| `observation.images.right_backward` | Right backward camera |
| `observation.images.map` | BEV map (640×360) |

## Usage

```python
from data_parsing.l2d import L2DDataset
from torch.utils.data import DataLoader

dataset = L2DDataset(
    repo_id="yaak-ai/L2D",
    backbone_name="swinv2_tiny_window8_256",
)

# Single episode for forward pass validation
dataset = L2DDataset(
    repo_id="yaak-ai/L2D",
    episodes=[0],
    backbone_name="swinv2_tiny_window8_256",
)

loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

for batch in loader:
    visual_tiles      = batch["visual_tiles"].to(device)       # (B, 7, 3, H, W)
    visual_history    = batch["visual_history"].to(device)     # (B, 896)
    egomotion_history = batch["egomotion_history"].to(device)  # (B, 256)
    trajectory_target = batch["trajectory_target"].to(device)  # (B, 128)

    trajectory, compressed, future = model(visual_tiles, visual_history, egomotion_history)
    loss = criterion(trajectory, trajectory_target)
```

## Forward pass test

```bash
cd Model/data_parsing/l2d
python forward_pass_test.py --no-pretrained

# With real data (requires lerobot + cached dataset):
python forward_pass_test.py --live --episodes 0
```

## Additional dependencies

```
lerobot==0.5.1            # LeRobot dataset SDK (pip install lerobot==0.5.1)
```
