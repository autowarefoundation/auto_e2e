"""Training-only selected-route supervision derived from canonical vectors."""

from __future__ import annotations

import dataclasses

import numpy as np

from .contracts import NavigationRoute
from .geometry import (
    DEFAULT_NAVIGATION_GEOMETRY,
    MapChannel,
    RouteChannel,
    NavigationRasterGeometry,
)
from .rasterizer import EgoPose, NavigationRaster


ROUTE_SUPERVISION_ARTIFACT_VERSION = "navigation_supervision_v3"
MAXIMUM_OUTSIDE_DISTANCE_M = 30.0


@dataclasses.dataclass(frozen=True)
class RouteSupervision:
    """Loss-only fields that cannot reveal the demonstrated future path."""

    distance_to_corridor_m: np.ndarray
    distance_to_drivable_m: np.ndarray
    drivable_available: bool
    route_heading_sin: np.ndarray
    route_heading_cos: np.ndarray
    route_heading_valid: np.ndarray
    destination_xy_m: np.ndarray
    destination_visible: bool

    def __post_init__(self) -> None:
        distance = np.ascontiguousarray(
            self.distance_to_corridor_m,
            dtype=np.float32,
        )
        drivable_distance = np.ascontiguousarray(
            self.distance_to_drivable_m,
            dtype=np.float32,
        )
        heading_sin = np.ascontiguousarray(
            self.route_heading_sin,
            dtype=np.float32,
        )
        heading_cos = np.ascontiguousarray(
            self.route_heading_cos,
            dtype=np.float32,
        )
        heading_valid = np.ascontiguousarray(
            self.route_heading_valid,
            dtype=np.uint8,
        )
        if distance.ndim != 2:
            raise ValueError("route distance field must have shape [H,W]")
        if (
            heading_sin.shape != distance.shape
            or drivable_distance.shape != distance.shape
            or heading_cos.shape != distance.shape
            or heading_valid.shape != distance.shape
        ):
            raise ValueError("route supervision raster shapes differ")
        if not np.isfinite(distance).all() or (
            distance.size
            and (
                float(distance.min()) < 0.0
                or float(distance.max()) > MAXIMUM_OUTSIDE_DISTANCE_M
            )
        ):
            raise ValueError("route distances must be finite and clipped")
        if not np.isfinite(drivable_distance).all() or (
            drivable_distance.size
            and (
                float(drivable_distance.min()) < 0.0
                or float(drivable_distance.max())
                > MAXIMUM_OUTSIDE_DISTANCE_M
            )
        ):
            raise ValueError(
                "drivable distances must be finite and clipped"
            )
        if not np.isfinite(heading_sin).all() or not np.isfinite(
            heading_cos
        ).all():
            raise ValueError("route headings must be finite")
        if not np.isin(heading_valid, (0, 1)).all():
            raise ValueError("route heading validity must be binary")
        destination = np.ascontiguousarray(
            self.destination_xy_m,
            dtype=np.float32,
        )
        if destination.shape != (2,) or not np.isfinite(destination).all():
            raise ValueError("route destination must have shape [2]")
        for value in (
            distance,
            drivable_distance,
            heading_sin,
            heading_cos,
            heading_valid,
            destination,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "distance_to_corridor_m", distance)
        object.__setattr__(
            self,
            "distance_to_drivable_m",
            drivable_distance,
        )
        object.__setattr__(
            self,
            "drivable_available",
            bool(self.drivable_available),
        )
        object.__setattr__(self, "route_heading_sin", heading_sin)
        object.__setattr__(self, "route_heading_cos", heading_cos)
        object.__setattr__(self, "route_heading_valid", heading_valid)
        object.__setattr__(self, "destination_xy_m", destination)
        object.__setattr__(
            self,
            "destination_visible",
            bool(self.destination_visible),
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "distance_to_corridor_m": self.distance_to_corridor_m,
            "distance_to_drivable_m": self.distance_to_drivable_m,
            "drivable_available": np.asarray(
                int(self.drivable_available),
                dtype=np.uint8,
            ),
            "route_heading_sin": self.route_heading_sin,
            "route_heading_cos": self.route_heading_cos,
            "route_heading_valid": self.route_heading_valid,
            "destination_xy_m": self.destination_xy_m,
            "destination_visible": np.asarray(
                int(self.destination_visible),
                dtype=np.uint8,
            ),
        }


def empty_route_supervision(
    geometry: NavigationRasterGeometry = DEFAULT_NAVIGATION_GEOMETRY,
) -> RouteSupervision:
    shape = (geometry.height_px, geometry.width_px)
    return RouteSupervision(
        distance_to_corridor_m=np.zeros(shape, dtype=np.float32),
        distance_to_drivable_m=np.zeros(shape, dtype=np.float32),
        drivable_available=False,
        route_heading_sin=np.zeros(shape, dtype=np.float32),
        route_heading_cos=np.zeros(shape, dtype=np.float32),
        route_heading_valid=np.zeros(shape, dtype=np.uint8),
        destination_xy_m=np.zeros(2, dtype=np.float32),
        destination_visible=False,
    )


def _map_to_ego(points_xy_m: np.ndarray, pose: EgoPose) -> np.ndarray:
    points = np.asarray(points_xy_m, dtype=np.float64)
    offsets = points[:, :2] - np.asarray(
        [pose.x_enu_m, pose.y_enu_m],
        dtype=np.float64,
    )
    cosine = float(np.cos(pose.yaw_rad))
    sine = float(np.sin(pose.yaw_rad))
    return np.column_stack(
        [
            cosine * offsets[:, 0] + sine * offsets[:, 1],
            -sine * offsets[:, 0] + cosine * offsets[:, 1],
        ]
    )


def _heading_source_fields(
    route: NavigationRoute,
    pose: EgoPose,
    geometry: NavigationRasterGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (geometry.height_px, geometry.width_px)
    sin_sum = np.zeros(shape, dtype=np.float64)
    cos_sum = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int32)
    sample_spacing_m = max(geometry.meters_per_pixel * 0.5, 0.1)

    for segment in route.lane_sequence:
        points = _map_to_ego(segment.centerline_enu_m, pose)
        for start, end in zip(points[:-1], points[1:], strict=True):
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length <= 1e-9:
                continue
            sample_count = max(2, int(np.ceil(length / sample_spacing_m)) + 1)
            parameters = np.linspace(
                0.0,
                1.0,
                sample_count,
                dtype=np.float64,
            )
            samples = start + parameters[:, None] * delta
            pixels = geometry.ego_to_pixel(samples)
            rows = np.rint(pixels[:, 0]).astype(np.int64)
            cols = np.rint(pixels[:, 1]).astype(np.int64)
            inside = (
                (rows >= 0)
                & (rows < geometry.height_px)
                & (cols >= 0)
                & (cols < geometry.width_px)
            )
            if not bool(inside.any()):
                continue
            angle = float(np.arctan2(delta[1], delta[0]))
            np.add.at(sin_sum, (rows[inside], cols[inside]), np.sin(angle))
            np.add.at(cos_sum, (rows[inside], cols[inside]), np.cos(angle))
            np.add.at(counts, (rows[inside], cols[inside]), 1)

    magnitude = np.hypot(sin_sum, cos_sum)
    source_valid = (counts > 0) & (
        magnitude / np.maximum(counts, 1) >= 0.5
    )
    source_sin = np.zeros(shape, dtype=np.float32)
    source_cos = np.zeros(shape, dtype=np.float32)
    source_sin[source_valid] = (
        sin_sum[source_valid] / magnitude[source_valid]
    ).astype(np.float32)
    source_cos[source_valid] = (
        cos_sum[source_valid] / magnitude[source_valid]
    ).astype(np.float32)
    return source_sin, source_cos, source_valid


def build_route_supervision(
    route: NavigationRoute,
    pose: EgoPose,
    raster: NavigationRaster,
    geometry: NavigationRasterGeometry = DEFAULT_NAVIGATION_GEOMETRY,
) -> RouteSupervision:
    """Build smooth loss targets from the selected route, never ego futures."""
    if raster.geometry_id != geometry.geometry_id:
        raise ValueError("route supervision geometry differs from raster")
    if raster.route_mask.shape[1:] != (
        geometry.height_px,
        geometry.width_px,
    ):
        raise ValueError("route supervision raster shape differs from geometry")

    from scipy.ndimage import distance_transform_edt

    drivable_available = bool(raster.map_valid)
    drivable = raster.map_context[MapChannel.DRIVABLE_AREA] > 0.0
    if drivable_available and not bool(drivable.any()):
        raise ValueError(
            "valid semantic map has no drivable pixels"
        )
    if drivable_available:
        drivable_distance = distance_transform_edt(
            ~drivable,
            sampling=geometry.meters_per_pixel,
        )
        drivable_distance = np.minimum(
            drivable_distance,
            MAXIMUM_OUTSIDE_DISTANCE_M,
        ).astype(np.float32)
    else:
        drivable_distance = np.full(
            drivable.shape,
            MAXIMUM_OUTSIDE_DISTANCE_M,
            dtype=np.float32,
        )

    if not route.valid or not raster.route_valid:
        empty = empty_route_supervision(geometry)
        return dataclasses.replace(
            empty,
            distance_to_drivable_m=drivable_distance,
            drivable_available=drivable_available,
        )

    corridor = (
        raster.route_mask[RouteChannel.SELECTED_CORRIDOR] > 0
    )
    if bool(corridor.any()):
        distance = distance_transform_edt(
            ~corridor,
            sampling=geometry.meters_per_pixel,
        )
        distance = np.minimum(
            distance,
            MAXIMUM_OUTSIDE_DISTANCE_M,
        ).astype(np.float32)
    else:
        distance = np.full(
            corridor.shape,
            MAXIMUM_OUTSIDE_DISTANCE_M,
            dtype=np.float32,
        )

    source_sin, source_cos, source_valid = _heading_source_fields(
        route,
        pose,
        geometry,
    )
    heading_sin = np.zeros(corridor.shape, dtype=np.float32)
    heading_cos = np.zeros(corridor.shape, dtype=np.float32)
    heading_valid = np.zeros(corridor.shape, dtype=np.uint8)
    if bool(source_valid.any()) and bool(corridor.any()):
        _, nearest = distance_transform_edt(
            ~source_valid,
            return_indices=True,
        )
        nearest_rows, nearest_cols = nearest
        heading_sin[corridor] = source_sin[
            nearest_rows[corridor],
            nearest_cols[corridor],
        ]
        heading_cos[corridor] = source_cos[
            nearest_rows[corridor],
            nearest_cols[corridor],
        ]
        heading_valid[corridor] = 1

    destination_xy = _map_to_ego(
        route.destination.position_enu_m.reshape(1, -1),
        pose,
    )[0]
    destination_visible = bool(
        geometry.contains_ego_points(destination_xy.reshape(1, 2))[0]
        and np.any(raster.route_mask[RouteChannel.DESTINATION] > 0)
    )
    return RouteSupervision(
        distance_to_corridor_m=distance,
        distance_to_drivable_m=drivable_distance,
        drivable_available=drivable_available,
        route_heading_sin=heading_sin,
        route_heading_cos=heading_cos,
        route_heading_valid=heading_valid,
        destination_xy_m=destination_xy.astype(np.float32),
        destination_visible=destination_visible,
    )
