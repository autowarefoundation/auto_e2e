"""Versioned geometry and channel layout for navigation rasters."""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Final

import numpy as np


class MapChannel(enum.IntEnum):
    DRIVABLE_AREA = 0
    LANE_BOUNDARY = 1
    LANE_CENTERLINE = 2
    INTERSECTION = 3
    CROSSWALK = 4
    STOP_LINE = 5
    STATIC_TRAFFIC_SIGNAL = 6
    TRAFFIC_DIRECTION_SIN = 7
    TRAFFIC_DIRECTION_COS = 8
    TRAFFIC_DIRECTION_VALID = 9
    KNOWN_MAP_AREA = 10
    ROAD_LEVEL = 11
    ROAD_LEVEL_VALID = 12
    OVERLAPPING_LEVEL_AMBIGUITY = 13


class RouteChannel(enum.IntEnum):
    SELECTED_CORRIDOR = 0
    DESTINATION = 1


MAP_CHANNEL_COUNT: Final = len(MapChannel)
ROUTE_CHANNEL_COUNT: Final = len(RouteChannel)
NAVIGATION_CHANNEL_COUNT: Final = MAP_CHANNEL_COUNT + ROUTE_CHANNEL_COUNT

BINARY_MAP_CHANNELS: Final[tuple[int, ...]] = (
    MapChannel.DRIVABLE_AREA,
    MapChannel.LANE_BOUNDARY,
    MapChannel.LANE_CENTERLINE,
    MapChannel.INTERSECTION,
    MapChannel.CROSSWALK,
    MapChannel.STOP_LINE,
    MapChannel.STATIC_TRAFFIC_SIGNAL,
    MapChannel.TRAFFIC_DIRECTION_VALID,
    MapChannel.KNOWN_MAP_AREA,
    MapChannel.ROAD_LEVEL_VALID,
    MapChannel.OVERLAPPING_LEVEL_AMBIGUITY,
)
CONTINUOUS_MAP_CHANNELS: Final[tuple[int, ...]] = (
    MapChannel.TRAFFIC_DIRECTION_SIN,
    MapChannel.TRAFFIC_DIRECTION_COS,
    MapChannel.ROAD_LEVEL,
)


@dataclasses.dataclass(frozen=True)
class NavigationRasterGeometry:
    geometry_id: str
    height_px: int
    width_px: int
    meters_per_pixel: float
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    ego_anchor_row: float
    ego_anchor_col: float
    matching_pc_range: tuple[float, float, float, float, float, float]
    matching_bev_h: int
    matching_bev_w: int
    route_corridor_width_m: float
    destination_marker_radius_m: float
    route_rear_clip_m: float
    frame: str = "ego_flu"
    pixel_convention: str = (
        "pixel_centers;row_front_to_rear;column_left_to_right"
    )

    def __post_init__(self) -> None:
        if not self.geometry_id:
            raise ValueError("geometry_id must not be empty")
        if self.height_px <= 0 or self.width_px <= 0:
            raise ValueError("raster dimensions must be positive")
        if not math.isfinite(self.meters_per_pixel):
            raise ValueError("meters_per_pixel must be finite")
        if self.meters_per_pixel <= 0.0:
            raise ValueError("meters_per_pixel must be positive")
        x_extent = self.x_max_m - self.x_min_m
        y_extent = self.y_max_m - self.y_min_m
        if not math.isclose(
            x_extent,
            self.height_px * self.meters_per_pixel,
            abs_tol=1e-9,
        ):
            raise ValueError("longitudinal extent does not match raster height")
        if not math.isclose(
            y_extent,
            self.width_px * self.meters_per_pixel,
            abs_tol=1e-9,
        ):
            raise ValueError("lateral extent does not match raster width")
        if self.matching_bev_h != self.height_px:
            raise ValueError("matching BEV height must equal raster height")
        if self.matching_bev_w != self.width_px:
            raise ValueError("matching BEV width must equal raster width")
        expected_range = (
            self.x_min_m,
            self.y_min_m,
            self.matching_pc_range[2],
            self.x_max_m,
            self.y_max_m,
            self.matching_pc_range[5],
        )
        if tuple(self.matching_pc_range) != expected_range:
            raise ValueError("matching_pc_range XY extent differs from raster")
        expected_anchor = self.ego_to_pixel(
            np.asarray([[0.0, 0.0]], dtype=np.float64)
        )[0]
        if not np.allclose(
            expected_anchor,
            [self.ego_anchor_row, self.ego_anchor_col],
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("ego anchor differs from pixel-center convention")
        positive = (
            self.route_corridor_width_m,
            self.destination_marker_radius_m,
            self.route_rear_clip_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("route dimensions must be finite and positive")
        if self.frame != "ego_flu":
            raise ValueError("navigation raster frame must be ego_flu")

    def ego_to_pixel(self, points_xy_m: np.ndarray) -> np.ndarray:
        """Convert ego FLU XY to fractional pixel-center row/column."""
        points = np.asarray(points_xy_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points_xy_m must have shape [N,2]")
        rows = (
            (self.x_max_m - points[:, 0]) / self.meters_per_pixel - 0.5
        )
        cols = (
            (self.y_max_m - points[:, 1]) / self.meters_per_pixel - 0.5
        )
        return np.column_stack([rows, cols])

    def pixel_to_ego(self, rows_cols: np.ndarray) -> np.ndarray:
        """Convert fractional pixel-center row/column to ego FLU XY."""
        pixels = np.asarray(rows_cols, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("rows_cols must have shape [N,2]")
        x = self.x_max_m - (pixels[:, 0] + 0.5) * self.meters_per_pixel
        y = self.y_max_m - (pixels[:, 1] + 0.5) * self.meters_per_pixel
        return np.column_stack([x, y])

    def pixel_center_grids(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x_forward, y_left)`` center grids with shape ``[H,W]``."""
        rows, cols = np.meshgrid(
            np.arange(self.height_px, dtype=np.float64),
            np.arange(self.width_px, dtype=np.float64),
            indexing="ij",
        )
        x = self.x_max_m - (rows + 0.5) * self.meters_per_pixel
        y = self.y_max_m - (cols + 0.5) * self.meters_per_pixel
        return x, y

    def contains_ego_points(self, points_xy_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points_xy_m must have shape [N,2]")
        return (
            (points[:, 0] >= self.x_min_m)
            & (points[:, 0] <= self.x_max_m)
            & (points[:, 1] >= self.y_min_m)
            & (points[:, 1] <= self.y_max_m)
        )


# Geometry audit over KITScenes v2.2:
# 0.5 m/px covered 89.31% of 6.4 s endpoints; 1.0 m/px covered 99.79%.
# The one-third rear / two-thirds front anchor matches the existing BEV origin.
DEFAULT_NAVIGATION_GEOMETRY: Final = NavigationRasterGeometry(
    geometry_id="kitscenes-v3-bev-1m-v1",
    height_px=256,
    width_px=256,
    meters_per_pixel=1.0,
    x_min_m=-85.5,
    x_max_m=170.5,
    y_min_m=-128.0,
    y_max_m=128.0,
    ego_anchor_row=170.0,
    ego_anchor_col=127.5,
    matching_pc_range=(-85.5, -128.0, -5.0, 170.5, 128.0, 3.0),
    matching_bev_h=256,
    matching_bev_w=256,
    route_corridor_width_m=3.5,
    destination_marker_radius_m=2.0,
    route_rear_clip_m=10.0,
)

LOCALIZATION_HZ: Final = 20
RASTER_DECIMATION: Final = 10
RASTER_HZ: Final = LOCALIZATION_HZ / RASTER_DECIMATION
MODEL_HZ: Final = 10
