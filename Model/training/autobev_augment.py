"""BEV-space augmentation and camera visibility of AutoBEV training samples.

BEV-space augmentation (BEVDet, Huang et al. 2021), stage 1 only: each sample
gets a random rotation about the ego z axis, a scale of the ground plane and a
left-right flip ``A``. Everything in the ego frame moves with it: projections
become ``P A^-1`` (current and history, since history projections map
current-ego points into past images), BEV labels are resampled at ``A^-1`` of
every cell and boxes move by ``A``. The images are not warped, so the
image-space heads keep the original projection and boxes as
``autobev_image_projection`` and ``autobev_image_boxes``. Front-back flips are
not used because the grid is asymmetric (80 m ahead, 30 m behind).

``restrict_to_visible`` ANDs the BEV label validity with what the cameras can
see after every augmentation: road level for the map classes and 0 to 1 m for
the objects, inside the image-content window of zoomed cameras. Training and
validation both apply it.
"""

from __future__ import annotations

import math
from typing import Any, Final

import torch
import torch.nn.functional as F

from data_processing.reactive_training_artifacts import BEV_SEGMENTATION_CLASSES
from model_components.autobev.geometry import BEV_H, BEV_W, PC_RANGE

MAX_YAW: Final = math.pi / 16
SCALE_RANGE: Final = (0.95, 1.05)
# Heights above the road (m) at which a cell counts as seen. Map classes lie on
# the road, so only z = 0 counts for them; objects count up to 1 m.
VISIBILITY_HEIGHTS_M: Final = (0.0, 0.5, 1.0)
OBJECT_CLASSES: Final = ("vehicle", "vulnerable_road_user", "other_obstacle")


def _cell_centres(device) -> tuple[torch.Tensor, torch.Tensor]:
    mpp = (PC_RANGE[3] - PC_RANGE[0]) / BEV_H
    x = PC_RANGE[3] - (torch.arange(BEV_H, device=device, dtype=torch.float32) + 0.5) * mpp
    y = PC_RANGE[4] - (torch.arange(BEV_W, device=device, dtype=torch.float32) + 0.5) * mpp
    return torch.meshgrid(x, y, indexing="ij")


def _resample_bev(maps: torch.Tensor, inverse: torch.Tensor, mode: str) -> torch.Tensor:
    """Resample [B, C, H, W] ego-grid maps at ``A^-1`` of every output cell."""
    gx, gy = _cell_centres(maps.device)
    points = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
    source = torch.einsum("bij,mj->bmi", inverse[:, :2, :2], points)
    grid_x = (PC_RANGE[4] - source[..., 1]) / (PC_RANGE[4] - PC_RANGE[1]) * 2 - 1
    grid_y = (PC_RANGE[3] - source[..., 0]) / (PC_RANGE[3] - PC_RANGE[0]) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, BEV_H, BEV_W, 2)
    return F.grid_sample(maps.float(), grid, mode=mode, padding_mode="zeros", align_corners=False)


def bev_augment(batch: dict[str, Any], generator: torch.Generator) -> dict[str, Any]:
    """Apply one random ego transform per sample to projections, BEV labels and boxes."""
    projection = batch["camera_projection_matrix"]
    size = projection.shape[0]
    device = projection.device
    yaw = ((torch.rand(size, generator=generator) * 2 - 1) * MAX_YAW).to(device)
    scale = (SCALE_RANGE[0] + torch.rand(size, generator=generator) * (SCALE_RANGE[1] - SCALE_RANGE[0])).to(device)
    flip = torch.where(torch.rand(size, generator=generator) < 0.5, -1.0, 1.0).to(device)
    cos, sin = yaw.cos(), yaw.sin()
    a = torch.zeros(size, 4, 4, device=device)
    a[:, 0, 0], a[:, 0, 1] = scale * cos, -scale * sin * flip
    a[:, 1, 0], a[:, 1, 1] = scale * sin, scale * cos * flip
    a[:, 2, 2] = 1.0
    a[:, 3, 3] = 1.0
    inverse = torch.linalg.inv(a)
    out = dict(batch)
    out["autobev_image_projection"] = projection
    out["autobev_image_boxes"] = batch["autobev_boxes"]
    out["camera_projection_matrix"] = torch.einsum("bvij,bjk->bvik", projection.float(), inverse)
    out["camera_history_projection_matrix"] = torch.einsum(
        "bhvij,bjk->bhvik", batch["camera_history_projection_matrix"].float(), inverse)
    # Labels are soft coverage, so bilinear keeps thin lines continuous;
    # validity stays nearest, and cells from outside the grid are unlabeled.
    out["bev_segmentation_target"] = _resample_bev(batch["bev_segmentation_target"], inverse, "bilinear")
    out["bev_segmentation_valid"] = _resample_bev(batch["bev_segmentation_valid"].float(), inverse, "nearest") > 0.5
    moved = []
    for index, boxes in enumerate(batch["autobev_boxes"]):
        box = boxes.clone()
        box[:, :2] = torch.einsum("ij,mj->mi", a[index, :2, :2], boxes[:, :2])
        box[:, 3:5] = boxes[:, 3:5] * scale[index]
        box[:, 6] = torch.remainder(flip[index] * boxes[:, 6] + yaw[index] + math.pi, 2 * math.pi) - math.pi
        box[:, 7:9] = torch.einsum("ij,mj->mi", a[index, :2, :2], boxes[:, 7:9])
        moved.append(box)
    out["autobev_boxes"] = moved
    return out


def camera_visibility(projection: torch.Tensor, heights: tuple[float, ...], window: torch.Tensor) -> torch.Tensor:
    """[B, H, W] cells whose column at any of ``heights`` projects into the ``window`` [B, V, 4] of any camera."""
    gx, gy = _cell_centres(projection.device)
    visible = torch.zeros(projection.shape[0], BEV_H, BEV_W, dtype=torch.bool, device=projection.device)
    x0, y0, x1, y1 = (window[..., i, None] for i in range(4))
    for z in heights:
        points = torch.stack([gx, gy, torch.full_like(gx, z), torch.ones_like(gx)], -1).reshape(-1, 4)
        uvw = torch.einsum("bvij,nj->bvni", projection.float(), points)
        depth = uvw[..., 2]
        u, v = uvw[..., 0] / depth.clamp(min=1e-6), uvw[..., 1] / depth.clamp(min=1e-6)
        seen = (depth > 1e-6) & (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
        visible |= seen.any(1).view(-1, BEV_H, BEV_W)
    return visible


def restrict_to_visible(batch: dict[str, Any]) -> dict[str, Any]:
    """AND the BEV label validity with camera visibility (road level for map classes, 0-1 m for objects)."""
    projection = batch["camera_projection_matrix"]
    window = batch["autobev_image_window"]
    road = camera_visibility(projection, (0.0,), window)
    column = camera_visibility(projection, VISIBILITY_HEIGHTS_M, window)
    objects = torch.tensor([name in OBJECT_CLASSES for name in BEV_SEGMENTATION_CLASSES], device=road.device)
    visible = torch.where(objects.view(1, -1, 1, 1), column[:, None], road[:, None])
    return {**batch, "bev_segmentation_valid": batch["bev_segmentation_valid"] & visible}

