"""Lateral recovery augmentation for Reactive training (#221).

The augmentation runs on a batch after it has been loaded from the published
shards, so raw data, labels and packing are unchanged. A random subset of the
samples is moved to a laterally shifted, slightly rotated ego frame ``E'``:

- ``projection`` keeps the images and right-multiplies every ego-to-image
  projection by ``T_E<-E'``, so BEVFormer builds its BEV in ``E'`` (BevAD,
  arXiv:2603.15185).
- ``image_warp`` keeps the camera rig fixed to the ego and warps each image as
  if the camera had moved, treating pixels below the horizon as flat ground and
  pixels above it as infinitely far away (PilotNet, arXiv:1604.07316).

Both methods resample the ego-frame rasters into ``E'``, blend the shift into
the last seconds of the egomotion history so the car has drifted there
(ChauffeurNet, arXiv:1812.03079, Sec. 5.1), and replace the trajectory target
by a path that returns to the logged one within ``RECOVERY_HORIZON_S``
(PilotNet's recovery label).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from model_components.autobev.geometry import AUTOBEV_GEOMETRY
from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    MAP_CHANNEL_COUNT,
    MapChannel,
    NavigationRasterGeometry,
)

LATERAL_RECOVERY_METHODS = ("projection", "image_warp")
MAX_OFFSET_M = 1.0
MAX_YAW_RAD = math.radians(5.0)
RECOVERY_HORIZON_S = 2.5
HISTORY_BLEND_S = 2.0
MIN_RECOVERY_PATH_M = 5.0
MIN_HISTORY_PATH_M = 2.0
CONSISTENCY_TEMPERATURE = 1.0

_DT_S = 0.1
_HISTORY_STEPS = 64
_SIGNALS = 4
_SPEED_FLOOR_MPS = 0.5
_MAX_CURVATURE = 0.5
# Egomotion rows cover [-6.4 s, -0.1 s]; the current state is one step after
# the last row. T8 history images sit every 500 ms before the current frame.
_HISTORY_IMAGE_ROWS = tuple(
    _HISTORY_STEPS - 5 * (7 - index) for index in range(7)
)


@dataclass(frozen=True)
class LateralShift:
    """Per-sample shift of ``E'`` in ``E``: ``offset_m`` left, ``yaw_rad`` CCW."""

    active: torch.Tensor
    offset_m: torch.Tensor
    yaw_rad: torch.Tensor


def _quintic(u: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Quintic Hermite bases H0, H1, H4, H5 and their derivatives in ``u``.

    ``e(u) = e0 H0 + e0' H1 + e1' H4 + e1 H5`` with zero second derivatives at
    both ends.
    """
    u2, u3, u4, u5 = u * u, u ** 3, u ** 4, u ** 5
    h0 = 1.0 - 10.0 * u3 + 15.0 * u4 - 6.0 * u5
    h1 = u - 6.0 * u3 + 8.0 * u4 - 3.0 * u5
    h4 = -4.0 * u3 + 7.0 * u4 - 3.0 * u5
    h5 = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
    dh0 = -30.0 * u2 + 60.0 * u3 - 30.0 * u4
    dh1 = 1.0 - 18.0 * u2 + 32.0 * u3 - 15.0 * u4
    dh4 = -12.0 * u2 + 28.0 * u3 - 15.0 * u4
    dh5 = 30.0 * u2 - 60.0 * u3 + 30.0 * u4
    return h0, h1, h4, h5, dh0, dh1, dh4, dh5


def ego_shift_transform(offset_m: torch.Tensor, yaw_rad: torch.Tensor) -> torch.Tensor:
    """Return ``T_E<-E'`` with shape ``[B,4,4]``."""
    cos, sin = torch.cos(yaw_rad), torch.sin(yaw_rad)
    transform = torch.zeros(
        (*offset_m.shape, 4, 4),
        dtype=offset_m.dtype,
        device=offset_m.device,
    )
    transform[..., 0, 0] = cos
    transform[..., 0, 1] = -sin
    transform[..., 1, 0] = sin
    transform[..., 1, 1] = cos
    transform[..., 1, 3] = offset_m
    transform[..., 2, 2] = 1.0
    transform[..., 3, 3] = 1.0
    return transform


def _history_signals(egomotion_history: torch.Tensor) -> torch.Tensor:
    return egomotion_history.reshape(-1, _HISTORY_STEPS, _SIGNALS).float()


def _curvature(speed: torch.Tensor, yaw_rate: torch.Tensor) -> torch.Tensor:
    curvature = torch.where(
        speed > _SPEED_FLOOR_MPS,
        yaw_rate / speed.clamp_min(_SPEED_FLOOR_MPS),
        torch.zeros_like(speed),
    )
    return curvature.clamp(-_MAX_CURVATURE, _MAX_CURVATURE)


def _reverse_cumsum(values: torch.Tensor) -> torch.Tensor:
    """``out[k] = sum(values[k:])`` along dim 1."""
    return torch.flip(torch.cumsum(torch.flip(values, [1]), dim=1), [1])


def current_curvature(egomotion_history: torch.Tensor) -> torch.Tensor:
    """Path curvature of the newest history row, used as the current one."""
    signals = _history_signals(egomotion_history)
    return _curvature(signals[:, -1, 0], signals[:, -1, 2])


def _frenet_slope(
    offset_m: torch.Tensor,
    yaw_rad: torch.Tensor,
    curvature: torch.Tensor,
) -> torch.Tensor:
    """``de/ds`` that gives heading offset ``yaw`` at lateral offset ``e``.

    The offset path heading is ``atan2(e_s, 1 - k e)``.
    """
    return (1.0 - curvature * offset_m) * torch.tan(yaw_rad)


def _history_blend(
    signals: torch.Tensor,
    offset_m: torch.Tensor,
    yaw_rad: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Lateral offset ``e`` and its slope ``de/ds`` along the logged past path.

    The offset is zero before the blend window and reaches ``offset_m`` with
    heading ``yaw_rad`` at the current time, one step after the last row.
    """
    speed = signals[..., 0]
    to_now = _reverse_cumsum(speed * _DT_S)
    start = _HISTORY_STEPS - round(HISTORY_BLEND_S / _DT_S)
    blend_length = to_now[:, start]
    length = blend_length.clamp_min(1e-6)[:, None]
    u = (1.0 - to_now / length).clamp(0.0, 1.0)
    _, _, h4, h5, _, _, dh4, dh5 = _quintic(u)
    curvature_now = _curvature(speed[:, -1], signals[:, -1, 2])
    slope_end = length * _frenet_slope(offset_m, yaw_rad, curvature_now)[:, None]
    offset = offset_m[:, None] * h5 + slope_end * h4
    slope = (offset_m[:, None] * dh5 + slope_end * dh4) / length
    return offset, slope, blend_length


def shift_egomotion_history(
    egomotion_history: torch.Tensor,
    shift: LateralShift,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend the shift into the past path and update the four signals.

    The past path is offset by ``e(s)`` along its left normal. For a path with
    curvature ``k`` the shifted path has speed ``v * sqrt((1 - k e)^2 + e_s^2)``
    and heading ``theta + atan2(e_s, 1 - k e)``. Only the differences are added
    to the stored signals, so a zero shift returns the input unchanged.

    Returns the new history and the per-row lateral offset and heading offset.
    """
    signals = _history_signals(egomotion_history)
    speed, yaw_rate = signals[..., 0], signals[..., 2]
    offset, slope, _ = _history_blend(signals, shift.offset_m, shift.yaw_rad)
    path_curvature = _curvature(speed, yaw_rate)
    along = 1.0 - path_curvature * offset
    new_speed = speed * torch.sqrt(along * along + slope * slope)
    heading_offset = torch.atan2(slope, along)
    new_yaw_rate = yaw_rate + torch.gradient(
        heading_offset, spacing=_DT_S, dim=1
    )[0]
    new_acceleration = signals[..., 1] + torch.gradient(
        new_speed - speed, spacing=_DT_S, dim=1
    )[0]
    new_curvature = signals[..., 3] + (
        _curvature(new_speed, new_yaw_rate) - path_curvature
    )
    shifted = torch.stack(
        [new_speed, new_acceleration, new_yaw_rate, new_curvature],
        dim=-1,
    )
    active = shift.active[:, None, None]
    shifted = torch.where(active, shifted, signals)
    return (
        shifted.reshape(egomotion_history.shape).to(egomotion_history.dtype),
        offset,
        heading_offset,
    )


def _path_tangents(points: torch.Tensor) -> torch.Tensor:
    """Unit tangents of ``[B,N,2]`` points, carried over where the car stops."""
    gradient = torch.gradient(points, dim=1)[0]
    norm = gradient.norm(dim=-1, keepdim=True)
    tangent = gradient / norm.clamp_min(1e-6)
    moving = norm[..., 0] > 1e-6
    previous = torch.zeros_like(tangent[:, 0])
    previous[:, 0] = 1.0
    filled = []
    for index in range(points.shape[1]):
        previous = torch.where(
            moving[:, index, None],
            tangent[:, index],
            previous,
        )
        filled.append(previous)
    return torch.stack(filled, dim=1)


def _recovery_window(points: torch.Tensor) -> tuple[torch.Tensor, int]:
    steps = round(RECOVERY_HORIZON_S / _DT_S)
    segments = (points[:, 1:] - points[:, :-1]).norm(dim=-1)
    arc = torch.cumsum(segments, dim=1)
    return arc, steps


def recovery_trajectory(
    trajectory_xy_m: torch.Tensor,
    shift: LateralShift,
    curvature: torch.Tensor,
) -> torch.Tensor:
    """Return the recovery target in ``E'`` coordinates, shape ``[B,T,2]``.

    The logged path is offset by ``e(s)`` along its left normal, with
    ``e(0) = offset``, a start slope that gives heading offset ``yaw`` on a
    path of the current ``curvature``, and ``e = e' = e'' = 0`` at the arc
    length reached after ``RECOVERY_HORIZON_S``. The speed profile is kept.
    """
    xy = trajectory_xy_m.float()
    points = torch.cat([xy.new_zeros(xy.shape[0], 1, 2), xy], dim=1)
    arc, steps = _recovery_window(points)
    length = arc[:, steps - 1].clamp_min(1e-6)[:, None]
    u = (arc / length).clamp(0.0, 1.0)
    h0, h1, _, _, _, _, _, _ = _quintic(u)
    slope_start = _frenet_slope(shift.offset_m, shift.yaw_rad, curvature)
    offset = shift.offset_m[:, None] * h0 + (
        length * slope_start[:, None]
    ) * h1
    tangent = _path_tangents(points)[:, 1:]
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    # Outside the window the target is the logged path; this also keeps any
    # non-finite invalid tail away from the valid points.
    recovered = torch.where(
        (u < 1.0)[..., None],
        xy + normal * offset[..., None],
        xy,
    )
    relative = recovered - torch.stack(
        [torch.zeros_like(shift.offset_m), shift.offset_m], dim=-1
    )[:, None]
    cos = torch.cos(shift.yaw_rad)[:, None]
    sin = torch.sin(shift.yaw_rad)[:, None]
    in_shifted = torch.stack(
        [
            cos * relative[..., 0] + sin * relative[..., 1],
            -sin * relative[..., 0] + cos * relative[..., 1],
        ],
        dim=-1,
    )
    in_shifted = torch.where(shift.active[:, None, None], in_shifted, xy)
    return in_shifted.to(trajectory_xy_m.dtype)


def sample_lateral_shift(
    batch: Mapping[str, Any],
    chosen: torch.Tensor,
    generator: torch.Generator,
) -> LateralShift:
    """Draw a shift for the ``chosen`` [B] samples from the trainer's CPU generator.

    A chosen sample is only shifted when its future moves at least
    ``MIN_RECOVERY_PATH_M`` within a valid recovery window and its past moves
    at least ``MIN_HISTORY_PATH_M`` within the history blend window.
    """
    xy = batch["trajectory_xy_m"]
    device = xy.device
    batch_size = int(xy.shape[0])
    draws = torch.rand(2, batch_size, generator=generator)
    chosen = chosen.to(device=device, dtype=torch.bool)
    offset = ((2.0 * draws[0] - 1.0) * MAX_OFFSET_M).to(device)
    yaw = ((2.0 * draws[1] - 1.0) * MAX_YAW_RAD).to(device)
    points = torch.cat(
        [xy.new_zeros(batch_size, 1, 2), xy.float()],
        dim=1,
    )
    arc, steps = _recovery_window(points)
    valid = batch["trajectory_valid"].to(torch.bool)
    future_ok = valid[:, :steps].all(dim=1) & (
        arc[:, steps - 1] >= MIN_RECOVERY_PATH_M
    )
    signals = _history_signals(batch["egomotion_history"])
    _, _, blend_length = _history_blend(
        signals,
        torch.zeros_like(offset),
        torch.zeros_like(yaw),
    )
    past_ok = blend_length >= MIN_HISTORY_PATH_M
    active = chosen & future_ok & past_ok
    zero = torch.zeros_like(offset)
    return LateralShift(
        active=active,
        offset_m=torch.where(active, offset, zero),
        yaw_rad=torch.where(active, yaw, zero),
    )


def shifted_raster_grid(
    shift: LateralShift,
    geometry: NavigationRasterGeometry,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``grid_sample`` grid that reads ``E`` rasters at ``E'`` cell centres.

    The raster covers the metric extent of ``geometry`` at ``height x width``
    cells, so the 450x300 navigation rasters and the 300x200 planner BEV share
    one geometry. Returns the ``[B,H,W,2]`` grid and a ``[B,H,W]`` mask of
    cells whose bilinear support lies inside the source raster.
    """
    device = shift.offset_m.device
    x_range = geometry.x_max_m - geometry.x_min_m
    y_range = geometry.y_max_m - geometry.y_min_m
    rows = torch.arange(height, device=device, dtype=torch.float32)
    cols = torch.arange(width, device=device, dtype=torch.float32)
    row, col = torch.meshgrid(rows, cols, indexing="ij")
    x_shifted = geometry.x_max_m - (row + 0.5) * (x_range / height)
    y_shifted = geometry.y_max_m - (col + 0.5) * (y_range / width)
    cos = torch.cos(shift.yaw_rad.float())[:, None, None]
    sin = torch.sin(shift.yaw_rad.float())[:, None, None]
    x = cos * x_shifted - sin * y_shifted
    y = sin * x_shifted + cos * y_shifted + shift.offset_m.float()[:, None, None]
    grid_x = 2.0 * (geometry.y_max_m - y) / y_range - 1.0
    grid_y = 2.0 * (geometry.x_max_m - x) / x_range - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    # Outermost pixel centres sit half a cell inside the extent.
    inside = (grid_x.abs() <= 1.0 - 1.0 / width) & (
        grid_y.abs() <= 1.0 - 1.0 / height
    )
    return grid, inside


def _resample(
    raster: torch.Tensor,
    shift: LateralShift,
    geometry: NavigationRasterGeometry,
    mode: str,
) -> torch.Tensor:
    grid, _ = shifted_raster_grid(shift, geometry, *raster.shape[-2:])
    sampled = F.grid_sample(
        raster.float(),
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    ).to(raster.dtype)
    return torch.where(
        shift.active.reshape(-1, *([1] * (raster.ndim - 1))),
        sampled,
        raster,
    )


def _shift_map_context(
    map_context: torch.Tensor,
    shift: LateralShift,
    geometry: NavigationRasterGeometry,
) -> torch.Tensor:
    """Resample the map into ``E'`` and rotate its traffic directions.

    Nearest sampling keeps the binary and validity channels exact. Directions
    are stored as ``(sin + 1) / 2`` and ``(cos + 1) / 2`` in valid cells and
    are rotated by ``-yaw`` into ``E'``.
    """
    if map_context.shape[1] != MAP_CHANNEL_COUNT:
        raise ValueError("lateral recovery expects the 14-channel map")
    shifted = _resample(map_context, shift, geometry, "nearest")
    sin_index = int(MapChannel.TRAFFIC_DIRECTION_SIN)
    cos_index = int(MapChannel.TRAFFIC_DIRECTION_COS)
    valid = shifted[:, int(MapChannel.TRAFFIC_DIRECTION_VALID)] > 0.5
    sin = 2.0 * shifted[:, sin_index].float() - 1.0
    cos = 2.0 * shifted[:, cos_index].float() - 1.0
    yaw_cos = torch.cos(shift.yaw_rad.float())[:, None, None]
    yaw_sin = torch.sin(shift.yaw_rad.float())[:, None, None]
    rotated_sin = sin * yaw_cos - cos * yaw_sin
    rotated_cos = cos * yaw_cos + sin * yaw_sin
    shifted = shifted.clone()
    for index, value in ((sin_index, rotated_sin), (cos_index, rotated_cos)):
        shifted[:, index] = torch.where(
            valid,
            ((value + 1.0) * 0.5).to(shifted.dtype),
            shifted[:, index],
        )
    return torch.where(
        shift.active.reshape(-1, 1, 1, 1),
        shifted,
        map_context,
    )


def _warp_images(
    images: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Warp ``[N,3,H,W]`` images from camera ``source`` to camera ``target``.

    ``source`` and ``target`` are ``[N,3,4]`` ego-to-pixel matrices over the
    same ego points. Target pixels whose ray hits the ground ``z = 0`` in front
    of the target camera use the ground-plane homography. All other pixels are
    treated as infinitely far away. Pixels without a source are zero, the
    normalised mean colour.
    """
    count, _, height, width = images.shape
    device = images.device
    with torch.autocast(device_type=device.type, enabled=False):
        source = source.float()
        target = target.float()
        rows = torch.arange(height, device=device, dtype=torch.float32) + 0.5
        cols = torch.arange(width, device=device, dtype=torch.float32) + 0.5
        v, u = torch.meshgrid(rows, cols, indexing="ij")
        pixels = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(-1, 3)
        pixels = pixels.T.expand(count, 3, height * width)
        ground_columns = [0, 1, 3]
        ground = torch.linalg.solve(target[:, :, ground_columns], pixels)
        from_ground = source[:, :, ground_columns] @ ground
        from_infinity = source[:, :, :3] @ torch.linalg.solve(
            target[:, :, :3], pixels
        )
        # A positive third ground coordinate is a positive target depth.
        hits_ground = ground[:, 2:3] > 0.0
        projected = torch.where(hits_ground, from_ground, from_infinity)
        depth = projected[:, 2]
        safe_depth = torch.where(depth > 1e-6, depth, torch.ones_like(depth))
        grid = torch.stack(
            [
                2.0 * projected[:, 0] / safe_depth / width - 1.0,
                2.0 * projected[:, 1] / safe_depth / height - 1.0,
            ],
            dim=-1,
        )
        grid = torch.where(
            (depth > 1e-6)[..., None],
            grid,
            torch.full_like(grid, 2.0),
        )
        warped = F.grid_sample(
            images.float(),
            grid.reshape(count, height, width, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
    return warped.to(images.dtype)


def _warp_views(
    images: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Warp ``[B,V,3,H,W]`` views of the active samples only."""
    if not bool(active.any()):
        return images
    warped = images.clone()
    selected = images[active]
    flat = _warp_images(
        selected.reshape(-1, *selected.shape[2:]),
        source[active].reshape(-1, 3, 4),
        target[active].reshape(-1, 3, 4),
    )
    warped[active] = flat.reshape(selected.shape)
    return warped


def _past_poses(signals: torch.Tensor) -> torch.Tensor:
    """Past ego poses ``T_E<-past`` at every history row, shape ``[B,64,4,4]``.

    Integrated backwards from the current time, one step after the last row,
    with the logged speed and yaw rate.
    """
    speed, yaw_rate = signals[..., 0], signals[..., 2]
    heading = -_reverse_cumsum(yaw_rate) * _DT_S
    x = -_reverse_cumsum(speed * torch.cos(heading) * _DT_S)
    y = -_reverse_cumsum(speed * torch.sin(heading) * _DT_S)
    pose = ego_shift_transform(y, heading)
    pose[..., 0, 3] = x
    return pose


def _history_camera_targets(
    history: torch.Tensor,
    signals: torch.Tensor,
    lateral: torch.Tensor,
    heading: torch.Tensor,
) -> torch.Tensor:
    """Past projections of cameras moved by the blended past shift ``D_k``.

    ``history[:, k]`` maps current-ego points into past image ``k``. The moved
    camera sees them through ``history @ pose D_k^-1 pose^-1`` with the past
    pose ``pose = T_E<-past``. It is written as
    ``history + history @ pose (D_k^-1 - I) pose^-1`` so a zero shift is exact,
    and the integrated pose only enters the small correction.
    """
    rows = torch.as_tensor(_HISTORY_IMAGE_ROWS, device=history.device)
    pose = _past_poses(signals)[:, rows]
    displacement = ego_shift_transform(
        lateral[:, rows].float(),
        heading[:, rows].float(),
    )
    identity = torch.eye(4, device=history.device, dtype=torch.float32)
    correction = (
        pose
        @ (torch.linalg.inv(displacement) - identity)
        @ torch.linalg.inv(pose)
    )
    past = history.float()
    return (past + past @ correction[:, :, None]).to(history.dtype)


def _shift_boxes(boxes: torch.Tensor, offset_m: torch.Tensor, yaw_rad: torch.Tensor) -> torch.Tensor:
    """Move [N, 9] ego-frame boxes (x, y, z, l, w, h, yaw, vx, vy) into ``E'``."""
    cos, sin = torch.cos(yaw_rad), torch.sin(yaw_rad)
    x, y = boxes[:, 0], boxes[:, 1] - offset_m
    vx, vy = boxes[:, 7], boxes[:, 8]
    moved = boxes.clone()
    moved[:, 0], moved[:, 1] = cos * x + sin * y, -sin * x + cos * y
    moved[:, 6] = torch.remainder(boxes[:, 6] - yaw_rad + math.pi, 2.0 * math.pi) - math.pi
    moved[:, 7], moved[:, 8] = cos * vx + sin * vy, -sin * vx + cos * vy
    return moved


def apply_lateral_recovery(
    batch: Mapping[str, Any],
    shift: LateralShift,
    *,
    method: str,
) -> dict[str, Any]:
    """Return a copy of ``batch`` with the active samples moved to ``E'``.

    The navigation rasters move on their 0.4 m grid and the BEV labels and
    boxes on the AutoBEV grid. ``image_warp`` changes what each camera sees,
    so the LiDAR depth of those samples no longer matches and is dropped.
    """
    out = dict(batch)
    active = shift.active
    egomotion, lateral, heading = shift_egomotion_history(batch["egomotion_history"], shift)
    out["egomotion_history"] = egomotion
    out["initial_speed_mps"] = _history_signals(egomotion)[:, -1, 0].to(batch["initial_speed_mps"].dtype)
    out["trajectory_xy_m"] = recovery_trajectory(
        batch["trajectory_xy_m"], shift, current_curvature(batch["egomotion_history"]),
    )
    out["map_context"] = _shift_map_context(batch["map_context"], shift, AUTOE2E_NAVIGATION_GEOMETRY)
    out["route_mask"] = _resample(batch["route_mask"], shift, AUTOE2E_NAVIGATION_GEOMETRY, "nearest")
    for key in ("bev_segmentation_target", "bev_segmentation_valid"):
        out[key] = _resample(batch[key], shift, AUTOBEV_GEOMETRY, "nearest")
    out["autobev_boxes"] = [
        _shift_boxes(boxes, offset, yaw)
        for boxes, offset, yaw in zip(batch["autobev_boxes"], shift.offset_m.float(), shift.yaw_rad.float())
    ]

    to_original = ego_shift_transform(shift.offset_m.float(), shift.yaw_rad.float())
    current = batch["camera_projection_matrix"]
    history = batch["camera_history_projection_matrix"]
    if method == "projection":
        out["camera_projection_matrix"] = (current.float() @ to_original[:, None]).to(current.dtype)
        out["camera_history_projection_matrix"] = (
            history.float() @ to_original[:, None, None]
        ).to(history.dtype)
        return out

    to_shifted = torch.linalg.inv(to_original)
    out["visual_tiles"] = _warp_views(batch["visual_tiles"], current, current.float() @ to_shifted[:, None], active)
    displaced = _history_camera_targets(history, _history_signals(batch["egomotion_history"]), lateral, heading)
    history_tiles = batch["camera_history_tiles"].clone()
    for frame in range(history.shape[1]):
        row = _HISTORY_IMAGE_ROWS[frame]
        moved = active & ((lateral[:, row] != 0.0) | (heading[:, row] != 0.0))
        history_tiles[:, frame] = _warp_views(history_tiles[:, frame], history[:, frame], displaced[:, frame], moved)
    out["camera_history_tiles"] = history_tiles
    out["camera_history_projection_matrix"] = (displaced.float() @ to_original[:, None, None]).to(history.dtype)
    out["autobev_depth"] = batch["autobev_depth"] * (~active).reshape(-1, 1, 1, 1).to(batch["autobev_depth"].dtype)
    return out


def aligned_clean_planner_bev(
    clean_planner_bev: torch.Tensor,
    shift: LateralShift,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample the clean planner BEV (AutoBEV grid) into ``E'`` for the consistency loss.

    Returns the aligned feature and a ``[B,H,W]`` mask of the cells that exist
    in both frames.
    """
    grid, inside = shifted_raster_grid(shift, AUTOBEV_GEOMETRY, *clean_planner_bev.shape[-2:])
    aligned = F.grid_sample(
        clean_planner_bev.detach().float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    return aligned, inside


def feature_consistency_kl(
    shifted_planner_bev: torch.Tensor,
    aligned_clean: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean per-cell ``KL(clean || shifted)`` over channel softmax."""
    temperature = CONSISTENCY_TEMPERATURE
    log_shifted = F.log_softmax(shifted_planner_bev.float() / temperature, dim=1)
    log_clean = F.log_softmax(aligned_clean.float() / temperature, dim=1)
    per_cell = (log_clean.exp() * (log_clean - log_shifted)).sum(dim=1)
    weight = mask.to(per_cell.dtype)
    return (per_cell * weight).sum() / weight.sum().clamp_min(1.0)
