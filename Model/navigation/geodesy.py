"""Coordinate transforms shared by canonical navigation consumers."""

from __future__ import annotations

import math
import re

import numpy as np

from .contracts import MapFrame


_EPSG_RE = re.compile(r"\bEPSG:(?P<code>[0-9]+)\b", re.IGNORECASE)
_EARTH_RADIUS_M = 6_371_008.8


def wgs84_to_map_xy(
    points_wgs84: np.ndarray,
    frame: MapFrame,
) -> np.ndarray:
    """Project latitude/longitude points into a canonical map frame."""
    points = np.asarray(points_wgs84, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("WGS84 points must have shape [N, >=2]")
    if not np.isfinite(points[:, :2]).all():
        raise ValueError("WGS84 points contain non-finite coordinates")

    match = _EPSG_RE.search(frame.projection)
    if match is not None:
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{match.group('code')}",
            always_xy=True,
        )
        origin_x, origin_y = transformer.transform(
            frame.origin_longitude_deg,
            frame.origin_latitude_deg,
        )
        x, y = transformer.transform(points[:, 1], points[:, 0])
        return np.ascontiguousarray(
            np.column_stack(
                [
                    np.asarray(x, dtype=np.float64) - origin_x,
                    np.asarray(y, dtype=np.float64) - origin_y,
                ]
            )
        )

    latitude = np.radians(points[:, 0])
    longitude = np.radians(points[:, 1])
    latitude_origin = math.radians(frame.origin_latitude_deg)
    longitude_origin = math.radians(frame.origin_longitude_deg)
    east = (
        _EARTH_RADIUS_M
        * (longitude - longitude_origin)
        * math.cos(latitude_origin)
    )
    north = _EARTH_RADIUS_M * (latitude - latitude_origin)
    return np.ascontiguousarray(np.column_stack([east, north]))
