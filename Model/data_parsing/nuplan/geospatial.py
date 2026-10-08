"""Absolute WGS84 poses for nuPlan scenes.

nuPlan stores ego poses in the projected coordinate system of each city map.
The ``meta`` table of every map ``.gpkg`` names it as ``projectedCoordSystem``
(for example ``epsg:32619`` for Boston). Yaw is counter-clockwise from that
grid's +x axis, so headings are converted by projecting a point one metre
ahead instead of assuming grid north equals true north.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Map location -> projected CRS recorded in the nuPlan v1.x map packages.
NUPLAN_MAP_PROJECTED_CRS = {
    "sg-one-north": "EPSG:32648",
    "us-ma-boston": "EPSG:32619",
    "us-nv-las-vegas-strip": "EPSG:32611",
    "us-pa-pittsburgh-hazelwood": "EPSG:32617",
}
NUPLAN_GPS_FUTURE_STEPS = 64
NUPLAN_GPS_HORIZON_S = 6.4
_HEADING_PROBE_M = 1.0


def nuplan_map_projected_crs(map_root: str | Path, map_name: str) -> str:
    """Read the projected CRS of one city from its map package.

    The value must agree with ``NUPLAN_MAP_PROJECTED_CRS`` so a changed map
    package fails loudly instead of placing scenes in the wrong UTM zone.
    """
    root = Path(map_root)
    metadata_files = sorted(root.glob("nuplan-maps-v*.json"))
    if len(metadata_files) != 1:
        raise ValueError("nuPlan map root must contain one version metadata file")
    metadata = json.loads(metadata_files[0].read_text())
    try:
        version = str(metadata[map_name]["version"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"nuPlan map metadata lacks {map_name!r}") from exc
    package = root / map_name / version / "map.gpkg"
    connection = sqlite3.connect(f"file:{package}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'projectedCoordSystem'"
        ).fetchone()
    finally:
        connection.close()
    if row is None or not str(row[0]).strip():
        raise ValueError(f"nuPlan map {map_name!r} has no projected CRS")
    crs = str(row[0]).strip().upper()
    expected = NUPLAN_MAP_PROJECTED_CRS.get(map_name)
    if crs != expected:
        raise ValueError(
            f"nuPlan map {map_name!r} uses {crs}, expected {expected}"
        )
    return crs


class NuPlanGeoProjector:
    """Convert nuPlan map coordinates of one city to WGS84."""

    def __init__(self, map_name: str, crs: str) -> None:
        if NUPLAN_MAP_PROJECTED_CRS.get(map_name) != crs:
            raise ValueError(f"unsupported nuPlan map CRS {map_name!r} {crs}")
        from pyproj import Transformer

        self.map_name = map_name
        self.crs = crs
        self._transformer = Transformer.from_crs(
            crs,
            "EPSG:4326",
            always_xy=True,
        )

    def latlon(self, xy: Any) -> np.ndarray:
        """Return ``[N, 2]`` latitude/longitude degrees for ``[N, 2]`` x/y."""
        points = np.asarray(xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("nuPlan map points must have shape [N, 2]")
        if not np.isfinite(points).all():
            raise ValueError("nuPlan map points must be finite")
        # pyproj treats length-one arrays as scalars; lists keep one code path.
        lon, lat = self._transformer.transform(
            points[:, 0].tolist(),
            points[:, 1].tolist(),
        )
        result = np.stack(
            [np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)],
            axis=1,
        )
        if not np.isfinite(result).all():
            raise ValueError(f"{self.crs} produced non-finite WGS84 points")
        return result

    def heading_deg_cw_from_north(self, x: float, y: float, yaw: float) -> float:
        """Return the true-north bearing of a grid yaw at one map point."""
        ahead = (
            x + _HEADING_PROBE_M * math.cos(yaw),
            y + _HEADING_PROBE_M * math.sin(yaw),
        )
        (lat0, lon0), (lat1, lon1) = self.latlon([[x, y], ahead])
        east = math.radians(lon1 - lon0) * math.cos(math.radians(lat0))
        north = math.radians(lat1 - lat0)
        return math.degrees(math.atan2(east, north)) % 360.0


def _state_xy_yaw(state: Any) -> tuple[float, float, float]:
    pose = state.rear_axle
    return float(pose.x), float(pose.y), float(pose.heading)


def nuplan_frame_geospatial(
    scenario: Any,
    iteration: int,
    projector: NuPlanGeoProjector,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Return ``pose_current`` and the ``[65, 2]`` GPS window of one frame.

    The window holds the current rear-axle position followed by the logged
    future at 10 Hz, padded with the last pose when the log ends early, which
    matches the KITScenes geospatial contract.
    """
    current = scenario.get_ego_state_at_iteration(iteration)
    x, y, yaw = _state_xy_yaw(current)
    future = list(
        scenario.get_ego_future_trajectory(
            iteration,
            time_horizon=NUPLAN_GPS_HORIZON_S,
            num_samples=NUPLAN_GPS_FUTURE_STEPS,
        )
    )[:NUPLAN_GPS_FUTURE_STEPS]
    xy = np.asarray(
        [[x, y], *[_state_xy_yaw(state)[:2] for state in future]],
        dtype=np.float64,
    )
    observed = projector.latlon(xy)
    gps_future = np.empty((NUPLAN_GPS_FUTURE_STEPS + 1, 2), dtype=np.float64)
    gps_future[: len(observed)] = observed
    gps_future[len(observed):] = observed[-1]
    pose_current: dict[str, float | int] = {
        "latitude_deg": float(observed[0, 0]),
        "longitude_deg": float(observed[0, 1]),
        "heading_deg_cw_from_north": projector.heading_deg_cw_from_north(
            x, y, yaw
        ),
        "timestamp_ns": int(current.time_point.time_us) * 1000,
        "gps_accuracy_m": float("nan"),
    }
    return pose_current, gps_future


class NuPlanSceneGeoRecords:
    """Episode path and sample poses of one packed scene.

    Exposes the ``episode_indices``/``episode_path``/``sample_pose_records``
    interface consumed by ``data_processing.geospatial.write_geo_artifacts``.
    """

    def __init__(self, scene_id: str) -> None:
        self.scene_id = scene_id
        self._rows: list[dict[str, Any]] = []

    def add(
        self,
        *,
        sample_uid: str,
        frame_index: int,
        pose: dict[str, float | int],
    ) -> None:
        if frame_index != len(self._rows):
            raise ValueError("nuPlan scene poses must be added in frame order")
        self._rows.append({
            "sample_uid": sample_uid,
            "episode_id": self.scene_id,
            "frame_index": frame_index,
            "latitude_deg": float(pose["latitude_deg"]),
            "longitude_deg": float(pose["longitude_deg"]),
            "heading_deg_cw_from_north": float(
                pose["heading_deg_cw_from_north"]
            ),
            "timestamp_ns": int(pose["timestamp_ns"]),
            "gps_accuracy_m": None,
        })

    def episode_indices(self) -> Sequence[str]:
        return [self.scene_id]

    def episode_path(self, episode_id: str) -> np.ndarray:
        if episode_id != self.scene_id:
            raise KeyError(episode_id)
        return np.asarray(
            [
                [
                    row["latitude_deg"],
                    row["longitude_deg"],
                    row["heading_deg_cw_from_north"],
                    row["timestamp_ns"],
                ]
                for row in self._rows
            ],
            dtype=np.float64,
        )

    def sample_pose_records(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]
