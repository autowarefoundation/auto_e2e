"""The map image L2D ships with each frame, registered into the BEV grid.

``observation.images.map`` is OpenStreetMap rendered in the standard Carto
style, centred on the ego with the planned route drawn as blue dots. It is
not an HD map, but it shows the road surface as the renderer drew it,
including parking areas and junction corners that a lane graph built from
way centrelines leaves out. Registering it needs no assumption about zoom or
orientation: the dots are the same ten route waypoints the dataset gives in
GPS, so a similarity fit of dots to waypoints yields scale and rotation per
frame. Real tiles are 640 by 360, north-up, at about 1.33 px/m.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np


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
# Real fits came out up to 4.6 degrees off north where the dots and the
# waypoints disagree; a north-up tile allows only GPS and rounding error.
_ROTATION_TOLERANCE_DEG = 2.0
_SCALE_GRID = np.geomspace(1.0 / _MAX_METERS_PER_PIXEL, 1.0 / _MIN_METERS_PER_PIXEL, 240)
_SEARCH_TOLERANCE_PX = 5.0
_EGO_MARKER_RADIUS_PX = 6.0
_EGO_CENTER_TOLERANCE_PX = 4.0
# Share of the waypoints that should show as dots for a fit to count.
_MIN_VISIBLE_FRACTION = 0.6


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

    labels, _ = ndimage.label(mask)
    # find_objects bounds each component, so no full-image pass per label
    # (dark Carto label text makes hundreds of them).
    components = []
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        points = np.argwhere(labels[box] == index)
        components.append(points + [box[0].start, box[1].start])
    return components


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
    # The renderer centres the ego; a dark label near the centre must not
    # move the origin, so only a marker within a few pixels refines it.
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    ego_points = [
        points for points in _components(ego)
        if len(points) >= 3 and np.hypot(points[:, 1].mean() - center[0], points[:, 0].mean() - center[1]) <= _EGO_CENTER_TOLERANCE_PX
    ]
    if ego_points:
        nearest = min(ego_points, key=lambda points: np.hypot(points[:, 1].mean() - center[0], points[:, 0].mean() - center[1]))
        center = (float(nearest[:, 1].mean()), float(nearest[:, 0].mean()))
    return np.asarray(centroids, dtype=np.float64).reshape(-1, 2), center


def _fit(pixel: np.ndarray, metric: np.ndarray) -> complex:
    """Least-squares similarity (complex scale-rotation) from metric to pixel."""
    return complex(np.vdot(metric, pixel) / np.vdot(metric, metric))


def _matches(transform: complex, metric: np.ndarray, dots: np.ndarray, tolerance_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Waypoints whose predicted pixel has a detected dot within tolerance."""
    predicted = transform * metric
    distance = np.abs(predicted[:, None] - dots[None, :])
    nearest = distance.argmin(axis=1)
    ok = distance[np.arange(len(metric)), nearest] <= tolerance_px
    return np.flatnonzero(ok), nearest[ok]


def register(
    image: np.ndarray,
    waypoints_east_north_m: np.ndarray,
    *,
    expected_rotation_rad: float = 0.0,
) -> MapImageRegistration | None:
    """Fit the image to ego-relative route waypoints; None when it does not fit.

    The tiles are north-up, so in a projected frame their rotation is known:
    ``expected_rotation_rad`` is minus the grid angle of true east at the
    ego (the meridian convergence, about one degree in Germany). Scale
    varies with the renderer's zoom, so it is searched; the hypothesis that lands the most waypoints on detected dots is refined by
    least squares and must keep the expected rotation. Ranking dots by
    distance from the ego does not work on real tiles, where the nearest
    waypoint hides under the ego marker and other blue map features add
    blobs.
    """
    dots, center = _route_dots(image)
    waypoints = np.asarray(waypoints_east_north_m, dtype=np.float64)[:, :2]
    waypoints = waypoints[np.hypot(waypoints[:, 0], waypoints[:, 1]) > 1.0]
    if len(dots) < _MIN_DOTS or len(waypoints) < _MIN_DOTS:
        return None
    pixel_dots = (dots[:, 0] - center[0]) - 1j * (dots[:, 1] - center[1])
    metric = waypoints[:, 0] + 1j * waypoints[:, 1]
    height, width = image.shape[:2]
    best: tuple[int, complex] | None = None
    for rotation in expected_rotation_rad + np.radians(np.arange(-_ROTATION_TOLERANCE_DEG, _ROTATION_TOLERANCE_DEG + 1e-9, 0.5)):
        for scale in _SCALE_GRID:
            transform = scale * complex(math.cos(rotation), math.sin(rotation))
            # Count distinct dots: evenly spaced dots also fit half the
            # scale, with two waypoints landing on every dot.
            _, nearest = _matches(transform, metric, pixel_dots, _SEARCH_TOLERANCE_PX)
            score = len(set(nearest.tolist()))
            if best is None or score > best[0]:
                best = (score, transform)
    assert best is not None
    transform = best[1]
    for tolerance in (_SEARCH_TOLERANCE_PX, _MAX_RESIDUAL_PX):
        inliers, nearest = _matches(transform, metric, pixel_dots, tolerance)
        if len(inliers) < _MIN_DOTS or len(set(nearest.tolist())) != len(nearest):
            return None
        transform = _fit(pixel_dots[nearest], metric[inliers])
    predicted = transform * metric
    column, row = center[0] + predicted.real, center[1] - predicted.imag
    # Waypoints under the ego marker or off the image cannot have a dot.
    visible = (np.abs(predicted) > _EGO_MARKER_RADIUS_PX) & (column >= 0) & (column < width) & (row >= 0) & (row < height)
    inliers, nearest = _matches(transform, metric, pixel_dots, _MAX_RESIDUAL_PX)
    if len(inliers) < max(_MIN_DOTS, math.ceil(_MIN_VISIBLE_FRACTION * int(visible.sum()))):
        return None
    residual = float(np.sqrt(np.mean(np.abs(pixel_dots[nearest] - transform * metric[inliers]) ** 2)))
    scale = abs(transform)
    rotation = math.atan2(transform.imag, transform.real)
    if (
        not (1.0 / _MAX_METERS_PER_PIXEL <= scale <= 1.0 / _MIN_METERS_PER_PIXEL)
        or residual > _MAX_RESIDUAL_PX
        # Dots that only fit rotated away from north are not this route.
        or abs(math.remainder(rotation - expected_rotation_rad, math.tau)) > math.radians(_ROTATION_TOLERANCE_DEG)
    ):
        return None
    return MapImageRegistration(center, scale, rotation, residual, len(inliers))


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
