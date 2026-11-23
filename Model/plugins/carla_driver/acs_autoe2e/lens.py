"""Rectifying F-theta (fisheye) frames into the dataset's pinhole views.

A camera may be declared with alpasim's ``FthetaCameraParam`` instead of a
pinhole: the Cosmos3 world model (cosmos3-canvas) generates only NVIDIA's
Hyperion 8 cameras, each through its F-theta lens. The checkpoint was trained
on the dataset's rectified pinhole cameras, so such a frame is resampled into a
*virtual* pinhole camera with the dataset camera's intrinsics and mount, drawn
from the fisheye cameras that see it (:class:`VirtualView`). The projections
the model is given are the virtual camera's, so they stay exact.

F-theta, as NVIDIA's ``FThetaCamera`` and alpasim define it: a ray at angle
``theta`` from the optical axis and azimuth ``phi`` lands at the radius
``r = angle_to_pixeldist(theta)`` from the principal point, the offset
``(r cos phi, r sin phi)`` mapped to pixels by ``[[c, d], [e, 1]]``. The
centre of the top-left pixel is ``(0, 0)``. Rays past ``max_angle`` are black.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FthetaLens:
    """One declared F-theta lens, in the declared image's pixels."""

    principal_point: tuple[float, float]
    angle_to_pixeldist: tuple[float, ...]
    max_angle: float
    linear_cde: tuple[float, float, float]
    image_size_wh: tuple[int, int]

    @classmethod
    def from_proto(cls, param, image_size_wh: tuple[int, int]) -> "FthetaLens":
        if not list(param.angle_to_pixeldist_poly):
            raise ValueError("F-theta lens without angle_to_pixeldist_poly")
        cde = param.linear_cde
        return cls(
            principal_point=(float(param.principal_point_x), float(param.principal_point_y)),
            angle_to_pixeldist=tuple(float(v) for v in param.angle_to_pixeldist_poly),
            max_angle=float(param.max_angle) if param.max_angle > 0 else np.pi / 2,
            linear_cde=(float(cde.linear_c) or 1.0, float(cde.linear_d), float(cde.linear_e)),
            image_size_wh=(int(image_size_wh[0]), int(image_size_wh[1])),
        )

    def pixels(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates of camera-frame rays ``(..., 3)`` and which are seen."""
        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
        lateral = np.hypot(x, y)
        theta = np.arctan2(lateral, z)
        radius = np.polynomial.polynomial.polyval(theta, self.angle_to_pixeldist)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(lateral > 0, radius / lateral, 0.0)
        ox, oy = x * scale, y * scale
        c, d, e = self.linear_cde
        u = self.principal_point[0] + c * ox + d * oy
        v = self.principal_point[1] + e * ox + oy
        width, height = self.image_size_wh
        seen = (theta <= self.max_angle) & (u > -0.5) & (u < width - 0.5) & (v > -0.5) & (
            v < height - 0.5
        )
        return np.stack([u, v], axis=-1), seen


@dataclass(frozen=True)
class SourceCamera:
    """A declared F-theta camera a view is drawn from."""

    logical_id: str
    lens: FthetaLens
    camera_to_reference: np.ndarray
    """``(4, 4)``: the camera's optical frame in the rig frame."""


@dataclass
class VirtualView:
    """A dataset pinhole camera drawn from one or more declared F-theta cameras.

    ``intrinsic`` (at ``size_wh``) and ``camera_to_reference`` are the virtual
    camera's. Each tile pixel's ray is taken by the first source, in order, whose
    lens sees it; rays no source sees are black. The sources' small offsets
    from the virtual camera's position are ignored (scene points are far
    compared with them).
    """

    intrinsic: np.ndarray
    size_wh: tuple[int, int]
    camera_to_reference: np.ndarray
    sources: tuple[SourceCamera, ...]
    _grids: dict[tuple[int, int], tuple[list[torch.Tensor], float]] = field(default_factory=dict)

    @property
    def source_ids(self) -> list[str]:
        return [source.logical_id for source in self.sources]

    def _rays(self, tile_hw: tuple[int, int]) -> np.ndarray:
        """Rays of a tile's pixel centres in the virtual camera frame, ``(h, w, 3)``."""
        height, width = tile_hw
        scale = np.diag([width / self.size_wh[0], height / self.size_wh[1], 1.0])
        inverse = np.linalg.inv(scale @ self.intrinsic)
        us, vs = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        return np.stack([us, vs, np.ones_like(us)], axis=-1) @ inverse.T

    def coverage_by_source(self, tile_hw: tuple[int, int]) -> list[float]:
        """The fraction of a tile each source's lens sees on its own."""
        rays = self._rays(tile_hw)
        return [float(self._source_pixels(source, rays)[1].mean()) for source in self.sources]

    def _source_pixels(self, source: SourceCamera, rays: np.ndarray):
        source_from_virtual = (
            np.linalg.inv(source.camera_to_reference) @ self.camera_to_reference
        )[:3, :3]
        return source.lens.pixels(rays @ source_from_virtual.T)

    def drawing(self, tile_hw: tuple[int, int]) -> "VirtualView":
        """This view without the sources that draw none of its pixels."""
        rays = self._rays(tile_hw)
        taken = np.zeros(rays.shape[:2], dtype=bool)
        keep = []
        for source in self.sources:
            draws = self._source_pixels(source, rays)[1] & ~taken
            taken |= draws
            if draws.any():
                keep.append(source)
        return VirtualView(self.intrinsic, self.size_wh, self.camera_to_reference, tuple(keep))

    def grids(self, tile_hw: tuple[int, int]) -> tuple[list[torch.Tensor], float]:
        """One ``grid_sample`` grid per source (pixels it does not draw point
        outside), and the fraction of the tile drawn."""
        if tile_hw not in self._grids:
            rays = self._rays(tile_hw)
            taken = np.zeros(rays.shape[:2], dtype=bool)
            grids = []
            for source in self.sources:
                pixels, seen = self._source_pixels(source, rays)
                draws = seen & ~taken
                taken |= draws
                fw, fh = source.lens.image_size_wh
                grid = np.stack(
                    [(pixels[..., 0] + 0.5) / fw * 2 - 1, (pixels[..., 1] + 0.5) / fh * 2 - 1],
                    axis=-1,
                )
                grid[~draws] = 2.0  # zeros padding: this source adds nothing there
                grids.append(torch.from_numpy(grid.astype(np.float32)).unsqueeze(0))
            self._grids[tile_hw] = (grids, float(taken.mean()))
        return self._grids[tile_hw]

    def render(self, images: list[torch.Tensor], tile_hw: tuple[int, int]) -> torch.Tensor:
        """The virtual tile ``(3, h, w)`` from the sources' frames ``(3, H, W)`` in [0, 1]."""
        grids, _ = self.grids(tile_hw)
        tile = None
        for image, grid in zip(images, grids):
            part = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=False).squeeze(0)
            tile = part if tile is None else tile + part
        if tile is None:
            raise ValueError("a virtual view needs at least one source frame")
        return tile
