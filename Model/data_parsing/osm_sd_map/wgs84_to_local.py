"""WGS84 (lat/lon) -> ego-local metric frame.

Replaces SDTagNet's dataset-specific city-coordinate conversions
(``av2_to_wgs_conversion`` / ``nusc_to_wgs_conversion``) with a generic
equirectangular projection centered + rotated on the ego pose, so any
GPS-bearing dataset can be used. Shares the small-window equirectangular
approximation with ``map_rendering.gps_to_map``.

Output convention matches the model's ego/BEV frame and the SDTagNet
``pc_range`` layout: **X = forward, Y = left, Z = up** (metres).

``ego_heading`` is radians, compass-style (0 = facing north, increasing toward
east) — the same convention produced by ``map_rendering.cache._heading_from_trace``.
"""

from __future__ import annotations

import math

import numpy as np

# WGS84 mean Earth radius (m); matches map_rendering.gps_to_map.
EARTH_RADIUS_M = 6_378_137.0


def wgs84_to_ego_local(latitudes, longitudes, ego_lat, ego_lon, ego_heading):
    """Project (lat, lon) arrays into the ego-local (forward, left) frame.

    Args:
        latitudes, longitudes: array-likes of equal shape (degrees).
        ego_lat, ego_lon: ego position (degrees).
        ego_heading: ego heading (radians, 0 = north, +toward east).

    Returns:
        (N, 2) float array of (x_forward, y_left) in metres.
    """
    lats = np.asarray(latitudes, dtype=float).reshape(-1)
    lons = np.asarray(longitudes, dtype=float).reshape(-1)
    if lats.shape != lons.shape:
        raise ValueError("latitudes and longitudes must have the same shape")

    deg_to_m = EARTH_RADIUS_M * math.pi / 180.0
    cos_lat = math.cos(math.radians(ego_lat))
    east = (lons - ego_lon) * cos_lat * deg_to_m
    north = (lats - ego_lat) * deg_to_m

    sin_h = math.sin(ego_heading)
    cos_h = math.cos(ego_heading)
    # Ego forward unit vector in (east, north) is (sin h, cos h); left is that
    # rotated +90 deg CCW = (-cos h, sin h).
    x_forward = east * sin_h + north * cos_h
    y_left = -east * cos_h + north * sin_h
    return np.stack([x_forward, y_left], axis=1)
