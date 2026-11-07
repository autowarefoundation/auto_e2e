"""Resample navigation rasters onto the AutoBEV grid at a given crop pose.

The navigation raster (``AUTOE2E_NAVIGATION_GEOMETRY``, 0.4 m) and the camera
BEV (``AUTOBEV_GEOMETRY``, 0.5 m) cover different extents at different pitches.
They are aligned geometrically, not by pooling: every AutoBEV cell centre
``p_F`` in the crop frame ``F`` is moved to the raster frame by the crop pose
``T_M<-F = (x, y, yaw)`` and the raster is sampled there.

A pose of zero reads the raster in its own frame. A non-zero pose crops the
map where a localization estimate believes the ego to be; the camera BEV stays
in the true ego frame, so the network sees the map shifted by the pose error.

Channel rules, so a 0.4 m to 0.5 m resample keeps meaning:

- binary map and route channels: bilinear, so a two-cell lane boundary stays
  continuous as soft coverage;
- traffic direction (stored as ``(sin + 1) / 2`` and ``(cos + 1) / 2``) and road
  level: averaged over valid cells only, and the direction rotated by ``-yaw``
  into ``F``;
- validity channels: nearest.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F

from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    MAP_CHANNEL_COUNT,
    MapChannel,
    NavigationRasterGeometry,
)

_NEAREST_CHANNELS: Final = (
    MapChannel.TRAFFIC_DIRECTION_VALID,
    MapChannel.KNOWN_MAP_AREA,
    MapChannel.ROAD_LEVEL_VALID,
    MapChannel.OVERLAPPING_LEVEL_AMBIGUITY,
)


def compose_poses(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """SE(2) product ``first * second`` of [..., 3] (x, y, yaw) poses."""
    cos, sin = torch.cos(first[..., 2]), torch.sin(first[..., 2])
    x = first[..., 0] + cos * second[..., 0] - sin * second[..., 1]
    y = first[..., 1] + sin * second[..., 0] + cos * second[..., 1]
    return torch.stack([x, y, first[..., 2] + second[..., 2]], dim=-1)


def sampling_grid(
    pose: torch.Tensor,
    target: NavigationRasterGeometry,
    source: NavigationRasterGeometry = AUTOE2E_NAVIGATION_GEOMETRY,
    margin_cells: int = 0,
) -> torch.Tensor:
    """``grid_sample`` grid [B, H, W, 2] reading ``source`` at the cells of ``target``.

    ``pose`` [B, 3] is the crop pose ``T_M<-F`` in metres and radians.
    ``margin_cells`` grows the target grid by that many cells on every side
    at the target pitch (the map matcher searches shifts inside it).
    """
    pitch = target.meters_per_pixel
    rows = torch.arange(-margin_cells, target.height_px + margin_cells, device=pose.device, dtype=torch.float32)
    cols = torch.arange(-margin_cells, target.width_px + margin_cells, device=pose.device, dtype=torch.float32)
    row, col = torch.meshgrid(rows, cols, indexing="ij")
    x_f = target.x_max_m - (row + 0.5) * pitch
    y_f = target.y_max_m - (col + 0.5) * pitch
    pose = pose.float()
    cos = torch.cos(pose[:, 2])[:, None, None]
    sin = torch.sin(pose[:, 2])[:, None, None]
    x = pose[:, 0, None, None] + cos * x_f - sin * y_f
    y = pose[:, 1, None, None] + sin * x_f + cos * y_f
    grid_x = 2.0 * (source.y_max_m - y) / (source.y_max_m - source.y_min_m) - 1.0
    grid_y = 2.0 * (source.x_max_m - x) / (source.x_max_m - source.x_min_m) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


def _sample(raster: torch.Tensor, grid: torch.Tensor, mode: str) -> torch.Tensor:
    return F.grid_sample(raster.float(), grid, mode=mode, padding_mode="zeros", align_corners=False)


def resample_map_context(map_context: torch.Tensor, pose: torch.Tensor, target: NavigationRasterGeometry,
                         margin_cells: int = 0) -> torch.Tensor:
    """Resample the 14-channel map [B, 14, h, w] at ``pose`` onto ``target`` (see module docstring)."""
    if map_context.shape[1] != MAP_CHANNEL_COUNT:
        raise ValueError(f"map_context must have {MAP_CHANNEL_COUNT} channels")
    grid = sampling_grid(pose, target, margin_cells=margin_cells)
    out = _sample(map_context, grid, "bilinear")
    nearest = [int(channel) for channel in _NEAREST_CHANNELS]
    out[:, nearest] = _sample(map_context[:, nearest], grid, "nearest")

    def valid_mean(value_index: int, valid_index: int) -> torch.Tensor:
        valid = map_context[:, valid_index : valid_index + 1].float()
        weight = _sample(valid, grid, "bilinear")
        value = _sample(map_context[:, value_index : value_index + 1].float() * valid, grid, "bilinear")
        return value / weight.clamp_min(1e-6)

    level = int(MapChannel.ROAD_LEVEL)
    out[:, level : level + 1] = valid_mean(level, int(MapChannel.ROAD_LEVEL_VALID))
    sin_index, cos_index = int(MapChannel.TRAFFIC_DIRECTION_SIN), int(MapChannel.TRAFFIC_DIRECTION_COS)
    direction_valid = int(MapChannel.TRAFFIC_DIRECTION_VALID)
    sin = 2.0 * valid_mean(sin_index, direction_valid) - 1.0
    cos = 2.0 * valid_mean(cos_index, direction_valid) - 1.0
    norm = torch.sqrt(sin * sin + cos * cos).clamp_min(1e-6)
    sin, cos = sin / norm, cos / norm
    yaw_cos = torch.cos(pose[:, 2].float())[:, None, None, None]
    yaw_sin = torch.sin(pose[:, 2].float())[:, None, None, None]
    # A direction theta in M is theta - yaw in F.
    rotated_sin = sin * yaw_cos - cos * yaw_sin
    rotated_cos = cos * yaw_cos + sin * yaw_sin
    valid = out[:, direction_valid : direction_valid + 1] > 0.5
    out[:, sin_index : sin_index + 1] = torch.where(valid, (rotated_sin + 1.0) * 0.5, torch.zeros_like(sin))
    out[:, cos_index : cos_index + 1] = torch.where(valid, (rotated_cos + 1.0) * 0.5, torch.zeros_like(cos))
    return out.to(map_context.dtype)


def resample_route(route_mask: torch.Tensor, pose: torch.Tensor, target: NavigationRasterGeometry) -> torch.Tensor:
    """Resample route channels [B, C, h, w] bilinearly at ``pose`` onto ``target``."""
    return _sample(route_mask, sampling_grid(pose, target), "bilinear").to(route_mask.dtype)
