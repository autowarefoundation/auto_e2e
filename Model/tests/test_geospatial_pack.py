"""Geospatial v2.1 parser and binary-contract tests."""

from __future__ import annotations

import gzip
import json
import math

import numpy as np
import pytest

from data_parsing.l2d.dataset import L2DDataset
from data_processing.geospatial import (
    GPS_FUTURE_POINTS,
    POSE_BINARY_SIZE,
    decode_gps_future,
    decode_pose,
    encode_gps_future,
    encode_pose,
    episode_artifact_stem,
    geospatial_members,
    write_geo_artifacts,
)


def _pose() -> dict[str, float | int]:
    return {
        "latitude_deg": 49.123456,
        "longitude_deg": 11.654321,
        "heading_deg_cw_from_north": 271.25,
        "timestamp_ns": 1_679_814_535_266_668_123,
        "gps_accuracy_m": math.nan,
    }


def test_pose_binary_roundtrip_preserves_int64_timestamp():
    payload = encode_pose(_pose())
    assert len(payload) == POSE_BINARY_SIZE
    decoded = decode_pose(payload)
    assert decoded["latitude_deg"] == _pose()["latitude_deg"]
    assert decoded["longitude_deg"] == _pose()["longitude_deg"]
    assert decoded["heading_deg_cw_from_north"] == _pose()[
        "heading_deg_cw_from_north"
    ]
    assert decoded["timestamp_ns"] == _pose()["timestamp_ns"]
    assert math.isnan(decoded["gps_accuracy_m"])


def test_gps_binary_roundtrip_is_float64_and_fixed_shape():
    points = np.arange(GPS_FUTURE_POINTS * 2, dtype=np.float64).reshape(-1, 2)
    payload = encode_gps_future(points)
    decoded = decode_gps_future(payload)
    assert decoded.dtype == np.float64
    np.testing.assert_array_equal(decoded, points)

    with pytest.raises(ValueError, match="shape"):
        encode_gps_future(points[:-1])
    with pytest.raises(ValueError, match="bytes"):
        decode_gps_future(payload[:-1])


def test_geospatial_members_are_atomic():
    points = np.zeros((GPS_FUTURE_POINTS, 2), dtype=np.float64)
    members = geospatial_members({
        "pose_current": _pose(),
        "gps_future": points,
    })
    assert set(members) == {"pose.npy", "gps.npy"}

    with pytest.raises(ValueError, match="present together"):
        geospatial_members({"pose_current": _pose()})


def test_episode_artifact_stem_preserves_numeric_and_scene_ids():
    assert episode_artifact_stem(12) == "000012"
    assert episode_artifact_stem(
        "fd1d1b6b-59bf-4292-8295-5028aa6aa5e3"
    ) == "fd1d1b6b-59bf-4292-8295-5028aa6aa5e3"
    with pytest.raises(ValueError, match="unsafe"):
        episode_artifact_stem("../escape")


def test_l2d_geospatial_alignment_uses_current_plus_64_future():
    states = np.zeros((150, 8), dtype=np.float32)
    states[:, 1] = np.arange(150, dtype=np.float32)  # heading
    states[:, 3] = 48.0 + np.arange(150, dtype=np.float32) / 10000
    states[:, 4] = 10.0 + np.arange(150, dtype=np.float32) / 10000
    timestamps = 1_670_000_000_000_000_000 + np.arange(
        150, dtype=np.int64
    ) * 100_000_000

    pose, future = L2DDataset._extract_geospatial(states, timestamps, 64)

    assert future.shape == (65, 2)
    np.testing.assert_array_equal(future[0], states[64, 3:5])
    np.testing.assert_array_equal(future[-1], states[128, 3:5])
    assert pose["heading_deg_cw_from_north"] == float(states[64, 1])
    assert pose["timestamp_ns"] == int(timestamps[64])


class _GeoDataset:
    def episode_indices(self):
        return list(range(5))

    def episode_path(self, episode_index):
        rows = np.zeros((30, 4), dtype=np.float64)
        rows[:, 0] = 49.0 + episode_index * 0.00001
        rows[:, 1] = 11.0 + np.arange(30) * 0.00001
        rows[:, 2] = 90.0
        rows[:, 3] = 1_670_000_000_000_000_000 + np.arange(30) * 100_000_000
        return rows

    def sample_pose_records(self):
        for episode_index in self.episode_indices():
            yield {
                "sample_uid": f"l2d-v1-e{episode_index:06d}-f000064",
                "episode_index": episode_index,
                "frame_index": 64,
                "latitude_deg": 49.0,
                "longitude_deg": 11.0,
                "heading_deg_cw_from_north": 90.0,
                "timestamp_ns": 1_670_000_000_000_000_000,
                "gps_accuracy_m": None,
            }


def test_geo_artifacts_include_paths_summary_parquet_and_private_heatmap(tmp_path):
    pytest.importorskip("pyarrow")
    summary = write_geo_artifacts(
        _GeoDataset(),
        tmp_path,
        dataset_name="yaak-ai/L2D",
        dataset_version="v2.1",
        k_anonymity=5,
    )

    assert summary["source_coordinate_dtype"] == "float32"
    assert summary["stored_coordinate_dtype"] == "float64"
    assert len(list((tmp_path / "geo" / "episode_paths").glob("*.f64"))) == 5
    assert (tmp_path / "geo" / "episode_paths" / "000000.f64").stat().st_size == 10 * 4 * 8
    assert (tmp_path / "geo" / "sample_pose.parquet").is_file()

    saved = json.loads((tmp_path / "geo" / "summary.json").read_text())
    assert saved["version"] == "v2.1"
    with gzip.open(tmp_path / "geo" / "heatmap.geojson.gz", "rt") as stream:
        heatmap = json.load(stream)
    assert heatmap["type"] == "FeatureCollection"
    assert heatmap["features"]
    coordinates = [f["geometry"]["coordinates"] for f in heatmap["features"]]
    assert saved["bbox"][0] <= min(point[0] for point in coordinates)
    assert saved["bbox"][2] >= max(point[0] for point in coordinates)
