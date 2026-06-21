# GPS → BEV Map Tile Rendering

Offline preprocessing utility that turns raw GPS waypoints into BEV map tiles
in the style of the L2D dataset's pre-rendered map. Use it for datasets that
do not natively ship map images (e.g. KIT Scenes, NVIDIA PhysicalAI).

## Ego-centric framing

Tiles are rendered **ego-centric**, matching the L2D / NVIDIA / KIT Scenes
convention:

- Center: the ego pose `(ego_lat, ego_lon)` (defaults to the last GPS sample
  when omitted in `gps_to_tensor`).
- Orientation: rotated so the ego forward direction points **up** in the
  image (forward = +y).
- Frame: a local equirectangular projection — coordinates are converted from
  lon/lat to metres relative to the ego, then rotated by `-ego_heading`. The
  axes share a common metric scale (`ax.set_aspect("equal")`), so there is
  no lat/lon aspect distortion.
- Extent: a fixed metric window of `±radius_m` around the origin.

`ego_heading` is in radians, measured CCW from north (so `0` = north, the
ego is facing north and the tile is north-up).

## When to use

- You have GPS lat/lon traces per clip and want a model input equivalent to
  L2D's BEV map tile.
- You are building a dataset offline and can pre-render every tile.
- You do **not** want to render at training time — fetching road networks via
  Overpass takes seconds per call and requires internet access.

## Workflow

1. Build a `{clip_id: (latitudes, longitudes)}` mapping from your dataset.
2. Run `render_and_cache_tiles(...)` once to produce one PNG per clip plus a
   shared road-network pickle cache.
3. In the DataLoader, read the PNG, push it through your timm transform like
   any other camera tile.

## Module layout

| File | Purpose |
| --- | --- |
| `gps_to_map.py` | Core: fetch network, map-match, render, end-to-end tensor. |
| `cache.py`      | Pickle network graphs and batch-render dataset tiles. |
| `test_gps_to_map.py` | Offline tests; `osmnx.graph_from_point` is mocked. |

## Public API

```python
from data_parsing.map_rendering import (
    fetch_road_network,
    map_match_waypoints,
    render_map_tile,
    gps_to_tensor,
)
from data_parsing.map_rendering.cache import (
    cache_network,
    load_cached_network,
    render_and_cache_tiles,
)
```

### Single-clip example

```python
import timm
from data_parsing.map_rendering import gps_to_tensor

backbone = timm.create_model("swinv2_tiny_window8_256", pretrained=False)
data_cfg = timm.data.resolve_model_data_config(backbone)
transform = timm.data.create_transform(**data_cfg, is_training=False)

tensor = gps_to_tensor(
    latitudes=lats,
    longitudes=lons,
    transform=transform,
    ego_heading=heading_rad,  # radians CCW from north
    # ego_lat / ego_lon default to the last GPS sample
    radius_m=800,
)
# tensor.shape == (3, H, W) — drop straight into the visual_tiles slot.
```

### Batch preprocessing

```python
from data_parsing.map_rendering.cache import render_and_cache_tiles

paths = render_and_cache_tiles(
    dataset_gps_data={clip_id: (lats, lons) for clip_id, lats, lons in clips},
    output_dir="cache/map_tiles",
    network_cache_dir="cache/road_networks",
    radius_m=800,
)
```

`network_cache_dir` quantizes centroids to ~100 m so adjacent clips reuse the
same downloaded graph.

## Style

Defaults match the L2D BEV map palette and dimensions:

| Element | Default |
| --- | --- |
| Image size | 640 × 360 |
| Background | `#111111` |
| Road network | `#444444` |
| Route | `#00CCFF` |
| Raw GPS markers | `#FF3333` |
| DPI | 200 |

All of these are arguments on `render_map_tile` and `gps_to_tensor`.

## Dependencies

- `osmnx` (and its transitive `geopandas` / `shapely` chain)
- `matplotlib` (headless `Agg` backend is selected automatically)
- `Pillow`
- `networkx`
- `torch` (only for the tensor output path)

Install with:

```
pip install osmnx geopandas matplotlib pillow networkx
```

## Notes

- This is a **data preprocessing** utility. It lives in `data_parsing/`, not
  `model_components/`. Do not call it from a `Dataset.__getitem__`.
- Map matching can fail (waypoints outside the fetched bbox, disconnected
  components). When it does, the renderer falls back to drawing the network
  plus raw GPS markers.
- `render_and_cache_tiles` estimates `ego_heading` from the last segment of
  each GPS trace. If you have a more accurate heading source (IMU, GNSS
  course-over-ground), prefer calling `render_map_tile` directly with it.
- Tests must not require internet — `osmnx.graph_from_point` and
  `osmnx.distance.nearest_nodes` are mocked.
