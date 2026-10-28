"""The map image L2D ships with each frame, registered into the BEV grid.

``observation.images.map`` is OpenStreetMap rendered in the standard Carto
style, centred on the ego with the planned route drawn as blue dots. It is
not an HD map, but it shows the road surface as the renderer drew it,
including parking areas and junction corners that a lane graph built from
way centrelines leaves out. Registering it needs no assumption about zoom or
orientation: the dots are the same ten route waypoints the dataset gives in
GPS, so a similarity fit of dots to waypoints yields scale and rotation per
frame.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

MAP_IMAGE_VERSION = "l2d_map_image_v1"

# Carto road fills, motorway to residential and service. The light greys of
# parking and living streets sit within JPEG noise of the land background
# (242, 239, 233), so they cannot be told apart and are not read.
_ROAD_COLORS = np.asarray([
    (233, 144, 160),  # motorway
    (249, 178, 156),  # trunk
    (252, 214, 164),  # primary
    (247, 250, 191),  # secondary
    (255, 255, 255),  # tertiary, unclassified, residential, service
], dtype=np.float64)
_ROAD_TOLERANCE = 10.0
_MIN_DOTS = 3
_MAX_RESIDUAL_PX = 3.0
_MIN_METERS_PER_PIXEL, _MAX_METERS_PER_PIXEL = 0.1, 5.0


@dataclasses.dataclass(frozen=True)
class MapImageRegistration:
    """Pixel = centre + scale * R(rotation) * (east, -north) in metres."""

    center_px: tuple[float, float]  # (column, row)
    pixels_per_meter: float
    rotation_rad: float
    residual_px: float
    dot_count: int

    def to_pixels(self, east_north_m: np.ndarray) -> np.ndarray:
        cos, sin = math.cos(self.rotation_rad), math.sin(self.rotation_rad)
        east, north = east_north_m[..., 0], east_north_m[..., 1]
        column = self.center_px[0] + self.pixels_per_meter * (cos * east - sin * north)
        row = self.center_px[1] - self.pixels_per_meter * (sin * east + cos * north)
        return np.stack([column, row], axis=-1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    return [np.argwhere(labels == index) for index in range(1, count + 1)]


def _route_dots(image: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Centroids (column, row) of the blue route dots and of the ego marker."""
    rgb = image.astype(np.int32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    dots = (blue > 180) & (red < 140) & (green < 160) & (blue - red > 80)
    ego = (red < 90) & (green < 90) & (blue < 140) & (blue >= red)
    centroids = [
        (float(points[:, 1].mean()), float(points[:, 0].mean()))
        for points in _components(dots)
        if 2 <= len(points) <= 200
    ]
    height, width = image.shape[:2]
    ego_points = [points for points in _components(ego) if len(points) >= 3]
    if ego_points:
        nearest = min(ego_points, key=lambda points: np.hypot(points[:, 1].mean() - width / 2, points[:, 0].mean() - height / 2))
        center = (float(nearest[:, 1].mean()), float(nearest[:, 0].mean()))
    else:
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
    return np.asarray(centroids, dtype=np.float64).reshape(-1, 2), center


def register(image: np.ndarray, waypoints_east_north_m: np.ndarray) -> MapImageRegistration | None:
    """Fit the image to ego-relative route waypoints; None when it does not fit."""
    dots, center = _route_dots(image)
    waypoints = np.asarray(waypoints_east_north_m, dtype=np.float64)[:, :2]
    waypoints = waypoints[np.hypot(waypoints[:, 0], waypoints[:, 1]) > 1.0]
    if len(dots) < _MIN_DOTS or len(waypoints) < _MIN_DOTS:
        return None
    # Pair dots and waypoints by their rank of distance from the ego.
    dots = dots[np.argsort(np.hypot(dots[:, 0] - center[0], dots[:, 1] - center[1]))]
    waypoints = waypoints[np.argsort(np.hypot(waypoints[:, 0], waypoints[:, 1]))]
    count = min(len(dots), len(waypoints))
    pixel = (dots[:count, 0] - center[0]) - 1j * (dots[:count, 1] - center[1])
    metric = waypoints[:count, 0] + 1j * waypoints[:count, 1]
    transform = complex(np.vdot(metric, pixel) / np.vdot(metric, metric))
    scale = abs(transform)
    residual = float(np.sqrt(np.mean(np.abs(pixel - transform * metric) ** 2)))
    if not (1.0 / _MAX_METERS_PER_PIXEL <= scale <= 1.0 / _MIN_METERS_PER_PIXEL) or residual > _MAX_RESIDUAL_PX:
        return None
    return MapImageRegistration(center, scale, math.atan2(transform.imag, transform.real), residual, count)


def road_mask(image: np.ndarray) -> np.ndarray:
    """Pixels drawn in a Carto road fill colour."""
    rgb = image.astype(np.float64)[..., None, :]
    distance = np.sqrt(((rgb - _ROAD_COLORS) ** 2).sum(axis=-1)).min(axis=-1)
    return distance <= _ROAD_TOLERANCE


def bev_layers(
    image: np.ndarray,
    registration: MapImageRegistration,
    *,
    yaw_rad: float,
    forward_m: np.ndarray,
    left_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(road, footprint) on the BEV grid given its cell-centre coordinates.

    ``forward_m`` and ``left_m`` are ego-frame grids; ``yaw_rad`` is the ego
    heading in ENU (counter-clockwise from east).
    """
    cos, sin = math.cos(yaw_rad), math.sin(yaw_rad)
    east = cos * forward_m - sin * left_m
    north = sin * forward_m + cos * left_m
    pixels = registration.to_pixels(np.stack([east, north], axis=-1))
    height, width = image.shape[:2]
    column = np.rint(pixels[..., 0]).astype(np.int64)
    row = np.rint(pixels[..., 1]).astype(np.int64)
    # Keep a margin so the attribution text and image border are not read.
    inside = (column >= 2) & (column < width - 2) & (row >= 2) & (row < height - 12)
    roads = road_mask(image)
    road = np.zeros(forward_m.shape, dtype=bool)
    road[inside] = roads[row[inside], column[inside]]
    return road, inside
