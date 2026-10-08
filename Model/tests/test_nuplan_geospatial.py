"""WGS84 conversion of nuPlan map-frame ego poses."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data_parsing.nuplan.geospatial import (
    NUPLAN_GPS_FUTURE_STEPS,
    NuPlanGeoProjector,
    NuPlanSceneGeoRecords,
    nuplan_frame_geospatial,
    nuplan_map_projected_crs,
)

# A logged Boston ego pose (us-ma-boston, UTM 19N) and its WGS84 location.
BOSTON_XY = (331200.0, 4690870.0)
BOSTON_LATLON = (42.351785, -71.049500)


def _map_root(tmp_path: Path, crs: str) -> Path:
    root = tmp_path / "maps"
    package = root / "us-ma-boston" / "9.12.1817"
    package.mkdir(parents=True)
    (root / "nuplan-maps-v1.0.json").write_text(json.dumps({
        "us-ma-boston": {"version": "9.12.1817"},
    }))
    connection = sqlite3.connect(package / "map.gpkg")
    connection.execute("CREATE TABLE meta (key TEXT, value TEXT)")
    connection.executemany(
        "INSERT INTO meta VALUES (?, ?)",
        [
            ("geographicCoordSystem", "epsg:4326"),
            ("projectedCoordSystem", crs),
        ],
    )
    connection.commit()
    connection.close()
    return root


def test_map_package_crs_must_match_the_known_utm_zone(tmp_path: Path):
    assert nuplan_map_projected_crs(
        _map_root(tmp_path / "ok", "epsg:32619"),
        "us-ma-boston",
    ) == "EPSG:32619"
    with pytest.raises(ValueError, match="expected EPSG:32619"):
        nuplan_map_projected_crs(
            _map_root(tmp_path / "bad", "epsg:32618"),
            "us-ma-boston",
        )


def test_projector_places_boston_poses_and_true_north_headings():
    projector = NuPlanGeoProjector("us-ma-boston", "EPSG:32619")

    latlon = projector.latlon([BOSTON_XY])
    np.testing.assert_allclose(latlon[0], BOSTON_LATLON, atol=1e-5)

    north = projector.heading_deg_cw_from_north(*BOSTON_XY, math.pi / 2)
    east = projector.heading_deg_cw_from_north(*BOSTON_XY, 0.0)
    # UTM grid north differs from true north by the meridian convergence,
    # about -1.4 degrees this far west of the zone's central meridian.
    assert -2.5 < ((north + 180.0) % 360.0) - 180.0 < 0.0
    assert 87.5 < east < 90.0
    with pytest.raises(ValueError, match="unsupported"):
        NuPlanGeoProjector("us-ma-boston", "EPSG:32611")


def _state(x: float, y: float, yaw: float, time_us: int):
    return SimpleNamespace(
        rear_axle=SimpleNamespace(x=x, y=y, heading=yaw),
        time_point=SimpleNamespace(time_us=time_us),
    )


def test_frame_geospatial_pads_a_short_logged_future():
    future = [
        _state(BOSTON_XY[0], BOSTON_XY[1] + 0.5 * step, math.pi / 2, step)
        for step in range(1, 11)
    ]
    scenario = SimpleNamespace(
        get_ego_state_at_iteration=lambda _iteration: _state(
            *BOSTON_XY, math.pi / 2, 1_620_000_000_000_000
        ),
        get_ego_future_trajectory=lambda *_args, **_kwargs: iter(future),
    )

    pose, gps_future = nuplan_frame_geospatial(
        scenario,
        0,
        NuPlanGeoProjector("us-ma-boston", "EPSG:32619"),
    )

    assert gps_future.shape == (NUPLAN_GPS_FUTURE_STEPS + 1, 2)
    np.testing.assert_allclose(gps_future[0], BOSTON_LATLON, atol=1e-5)
    np.testing.assert_array_equal(gps_future[11], gps_future[-1])
    assert gps_future[10, 0] > gps_future[0, 0]
    assert pose["timestamp_ns"] == 1_620_000_000_000_000_000
    assert math.isnan(float(pose["gps_accuracy_m"]))


def test_scene_geo_records_expose_the_geo_artifact_interface():
    records = NuPlanSceneGeoRecords("398cd2903ae3a6e9")
    pose = {
        "latitude_deg": 42.35,
        "longitude_deg": -71.05,
        "heading_deg_cw_from_north": 10.0,
        "timestamp_ns": 7,
    }
    records.add(sample_uid="uid-0", frame_index=0, pose=pose)
    with pytest.raises(ValueError, match="frame order"):
        records.add(sample_uid="uid-2", frame_index=2, pose=pose)

    assert records.episode_indices() == ["398cd2903ae3a6e9"]
    np.testing.assert_allclose(
        records.episode_path("398cd2903ae3a6e9"),
        [[42.35, -71.05, 10.0, 7.0]],
    )
    assert records.sample_pose_records()[0]["episode_id"] == (
        "398cd2903ae3a6e9"
    )
