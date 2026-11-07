"""Give each training sample one view distortion, so repeated epochs see new views.

Each sample draws exactly one of ``DISTORTIONS`` with ``DISTORTION_SHARES``:

- ``projection`` and ``image_warp``: lateral recovery (``training.lateral_recovery``),
  which shifts and turns the ego, warps the views, and replaces the target by
  a path that returns to the logged one;
- ``camera_augment``: zoom and principal-point shift per camera, the same for
  all eight frames, applied to the images, the projections (P' = S P, so the
  focal length changes and the focal-normalized depth absorbs it) and the
  stride-8 depth targets; varies the focal length within each dataset as the
  random scaling of DG-BEV and Cube R-CNN does;
- ``photometric``: brightness, contrast and saturation per camera, shared by the
  eight frames of that camera;
- ``none``.

Every distortion runs on the whole batch with an identity setting for the
samples that did not draw it, so there is one code path; the clean batch is
kept for the feature-consistency teacher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping

import torch
import torch.nn.functional as F

from model_components.autobev.geometry import IMAGE_H, IMAGE_W
from training.lateral_recovery import (
    LateralShift,
    aligned_clean_planner_bev,
    apply_lateral_recovery,
    sample_lateral_shift,
)

DISTORTIONS: Final = ("projection", "image_warp", "camera_augment", "photometric", "none")
# About 60 % of the samples are distorted (the 15-epoch schedule sees the
# clean data three times with different views).
DISTORTION_SHARES: Final = (0.15, 0.15, 0.15, 0.15, 0.40)
ZOOM_RANGE: Final = (0.9, 1.15)
MAX_PRINCIPAL_SHIFT: Final = 0.04  # of the image size
PHOTOMETRIC_RANGE: Final = (0.8, 1.2)
IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
IMAGENET_STD: Final = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Distortion:
    """Which distortion each sample got [B] and the lateral shift applied to it."""

    choice: torch.Tensor
    shift: LateralShift

    @property
    def distorted(self) -> torch.Tensor:
        return self.choice != DISTORTIONS.index("none")


def _draw(generator: torch.Generator, bounds: tuple[float, float], *shape: int) -> torch.Tensor:
    return bounds[0] + torch.rand(shape, generator=generator) * (bounds[1] - bounds[0])


def camera_augment(batch: dict[str, Any], active: torch.Tensor, generator: torch.Generator) -> dict[str, Any]:
    images = batch["visual_tiles"]
    b, v = images.shape[:2]
    device = images.device
    on = active.to(device)[:, None]
    zoom = torch.where(on, _draw(generator, ZOOM_RANGE, b, v).to(device), 1.0)
    shift = torch.where(on[..., None], ((2 * torch.rand(b, v, 2, generator=generator) - 1) * MAX_PRINCIPAL_SHIFT)
                        .to(device), 0.0)
    size = torch.tensor([IMAGE_W, IMAGE_H], device=device, dtype=torch.float32)
    # u' = s u + t, the zoom centred on the image centre plus a shift.
    t = (1 - zoom)[..., None] * size / 2 + shift * size
    s_matrix = torch.zeros(b, v, 3, 3, device=device)
    s_matrix[..., 0, 0] = zoom
    s_matrix[..., 1, 1] = zoom
    s_matrix[..., :2, 2] = t
    s_matrix[..., 2, 2] = 1.0

    def grid(height: int, width: int) -> torch.Tensor:
        # Output pixel centres mapped back to input pixels; a stride-k map uses the same warp.
        stride = IMAGE_W / width
        ys = (torch.arange(height, device=device) + 0.5) * stride
        xs = (torch.arange(width, device=device) + 0.5) * stride
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        source_x = (gx[None, None] - t[..., 0, None, None]) / zoom[..., None, None]
        source_y = (gy[None, None] - t[..., 1, None, None]) / zoom[..., None, None]
        return torch.stack([source_x / IMAGE_W * 2 - 1, source_y / IMAGE_H * 2 - 1], -1).flatten(0, 1)

    def warp(maps: torch.Tensor, sampling: torch.Tensor, mode: str) -> torch.Tensor:
        return F.grid_sample(maps.float(), sampling, mode=mode, padding_mode="zeros", align_corners=False)

    full = grid(IMAGE_H, IMAGE_W)
    # Unselected samples keep their exact pixels; an identity resample would
    # still blur them by float rounding.
    keep = (~active).to(device).view(b, 1, 1, 1, 1)
    out = dict(batch)
    warped = warp(images.flatten(0, 1), full, "bilinear").view_as(images).to(images.dtype)
    out["visual_tiles"] = torch.where(keep, images, warped)
    history = batch["camera_history_tiles"]
    frames = history.shape[1]
    warped = warp(history.transpose(1, 2).flatten(0, 2), full.repeat_interleave(frames, 0), "bilinear")
    warped = warped.view(b, v, frames, *history.shape[-3:]).transpose(1, 2).to(history.dtype)
    out["camera_history_tiles"] = torch.where(keep[:, None], history, warped)
    out["camera_projection_matrix"] = s_matrix @ batch["camera_projection_matrix"]
    out["camera_history_projection_matrix"] = s_matrix[:, None] @ batch["camera_history_projection_matrix"]
    depth = batch["autobev_depth"]
    out["autobev_depth"] = warp(depth.flatten(0, 1)[:, None], grid(*depth.shape[-2:]), "nearest").view_as(depth)
    # Stride-8 cells of a zoomed-out image that hold no camera content.
    ones = torch.ones(b * v, 1, *depth.shape[-2:], device=device)
    out["autobev_image_valid"] = (warp(ones, grid(*depth.shape[-2:]), "nearest") > 0.5).view_as(depth)
    # Pixel rectangle [x0, y0, x1, y1] per camera that holds source content.
    out["autobev_image_window"] = torch.cat([t.clamp(min=0), torch.minimum(zoom[..., None] * size + t, size)], -1)
    return out


def photometric(batch: dict[str, Any], active: torch.Tensor, generator: torch.Generator) -> dict[str, Any]:
    images = batch["visual_tiles"]
    b, v = images.shape[:2]
    device = images.device
    on = active.to(device).view(b, 1, 1, 1, 1)
    factors = [torch.where(on, _draw(generator, PHOTOMETRIC_RANGE, b, v).to(device).view(b, v, 1, 1, 1), 1.0)
               for _ in range(3)]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)

    def jitter(normalized: torch.Tensor) -> torch.Tensor:
        brightness, contrast, saturation = factors
        rgb = (normalized.float() * std + mean) * brightness
        average = rgb.mean(dim=(-3, -2, -1), keepdim=True)
        rgb = (rgb - average) * contrast + average
        gray = rgb.mean(dim=-3, keepdim=True)
        rgb = ((rgb - gray) * saturation + gray).clamp(0.0, 1.0)
        return ((rgb - mean) / std).to(normalized.dtype)

    out = dict(batch)
    out["visual_tiles"] = jitter(images)
    out["camera_history_tiles"] = torch.stack(
        [jitter(frame) for frame in batch["camera_history_tiles"].unbind(1)], dim=1)
    return out


def apply_distortions(
    batch: Mapping[str, Any],
    generator: torch.Generator,
) -> tuple[dict[str, Any], Distortion]:
    """Draw one distortion per sample and apply it (see module docstring)."""
    size = int(batch["visual_tiles"].shape[0])
    choice = torch.multinomial(torch.tensor(DISTORTION_SHARES), size, replacement=True, generator=generator)
    out = dict(batch)
    # Lateral recovery needs a moving ego; samples that cannot take a shift stay clean.
    shift = sample_lateral_shift(out, choice <= DISTORTIONS.index("image_warp"), generator)
    for method in ("projection", "image_warp"):
        chosen = shift.active & (choice == DISTORTIONS.index(method)).to(shift.active.device)
        out = apply_lateral_recovery(out, LateralShift(chosen, shift.offset_m * chosen, shift.yaw_rad * chosen),
                                     method=method)
    out = camera_augment(out, choice == DISTORTIONS.index("camera_augment"), generator)
    out = photometric(out, choice == DISTORTIONS.index("photometric"), generator)
    applied = torch.where(
        (choice <= DISTORTIONS.index("image_warp")) & ~shift.active.cpu(), DISTORTIONS.index("none"), choice)
    return out, Distortion(applied.to(shift.active.device), shift)


def aligned_teacher_bev(clean_planner_bev: torch.Tensor, distortion: Distortion) -> tuple[torch.Tensor, torch.Tensor]:
    """The clean planner BEV moved into each distorted sample's frame, and the [B, H, W] cells to compare."""
    aligned, inside = aligned_clean_planner_bev(clean_planner_bev, distortion.shift)
    return aligned, inside & distortion.distorted[:, None, None]
