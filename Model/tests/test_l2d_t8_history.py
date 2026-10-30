"""L2D T8 camera history: rows, frame ids, and current-ego-aligned projections."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from data_parsing.l2d.dataset import L2DDataset  # noqa: E402
from reactive_training_contracts import REACTIVE_BEVFORMER_FRAME_OFFSETS  # noqa: E402


def _dataset(states: np.ndarray) -> L2DDataset:
    dataset = L2DDataset.__new__(L2DDataset)
    dataset.repo_id = "yaak-ai/L2D"
    dataset._source_hz = 10.0
    dataset._episode_ranges = {3: (100, 100 + len(states))}
    dataset._samples = [(3, 100 + 64)]
    dataset._get_vehicle_states_window = lambda start, end: states  # type: ignore[method-assign,assignment]
    timestamps = 1_700_000_000_000_000_000 + np.arange(len(states), dtype=np.int64) * 100_000_000
    dataset._get_timestamps_window = lambda start, end: timestamps  # type: ignore[method-assign,assignment]
    return dataset


def _driving_north(count: int = 200, speed_mps: float = 10.0) -> np.ndarray:
    states = np.zeros((count, 8), dtype=np.float32)
    states[:, 0] = speed_mps * 3.6  # km/h
    states[:, 1] = 0.0  # heading: true north
    states[:, 3] = 49.0 + np.arange(count) * speed_mps * 0.1 / 111_320.0
    states[:, 4] = 8.4
    return states


def test_history_rows_are_seven_half_second_steps_before_the_sample():
    dataset = _dataset(_driving_north())
    rows = dataset.bevformer_history_rows(0)
    assert rows == [(3, 64 + 5 * offset) for offset in REACTIVE_BEVFORMER_FRAME_OFFSETS[:-1]]
    assert dataset.bevformer_history_frame_ids(0)[0][5].endswith("-e000003-r000029-c5")


def test_history_projection_sees_a_current_point_from_the_earlier_pose():
    pytest.importorskip("yaml")
    dataset = _dataset(_driving_north())
    calibration = dataset.reactive_calibration_for(0, image_size=512)
    current = np.asarray(calibration["projection"]["matrix"])
    history = np.asarray(calibration["history_projection"]["matrix"])
    assert history.shape == (7, 6, 3, 4) and calibration["temporal_frame_offsets"] == list(REACTIVE_BEVFORMER_FRAME_OFFSETS)
    # 3.5 s earlier at 10 m/s the car stood 35 m behind: a point 50 m ahead
    # of the current ego appears where 85 m ahead appeared to the current camera.
    ahead_now = np.asarray([50.0, 0.0, 0.0, 1.0])
    ahead_then = np.asarray([85.0, 0.0, 0.0, 1.0])
    seen_then = history[0, 0] @ ahead_now
    expected = current[0] @ ahead_then
    assert seen_then[:2] / seen_then[2] == pytest.approx(expected[:2] / expected[2], abs=0.5)
    assert np.asarray(calibration["front_projection"]["matrix"]).shape == (1, 3, 4)


def test_history_rows_off_the_half_second_grid_are_refused():
    dataset = _dataset(_driving_north())
    # A dropped frame shifts every later row by 100 ms.
    timestamps = 1_700_000_000_000_000_000 + np.arange(200, dtype=np.int64) * 100_000_000
    timestamps[40:] += 100_000_000
    dataset._get_timestamps_window = lambda start, end: timestamps  # type: ignore[method-assign,assignment]
    with pytest.raises(ValueError, match="0.5 s grid"):
        dataset.bevformer_history_rows(0)


def test_dead_reckoning_ignores_float32_gps_steps():
    from data_parsing.l2d.dataset import _dead_reckoned_poses

    states = _driving_north()
    timestamps = np.arange(len(states), dtype=np.int64) * 100_000_000
    poses = _dead_reckoned_poses(states, timestamps, 29, 64)
    # 35 frames at 10 m/s: the oldest pose is 35 m behind, on the axis.
    assert poses[0][:2, 3] == pytest.approx([-35.0, 0.0], abs=1e-6)
    assert np.allclose(poses[-1], np.eye(4))


def test_track_points_read_the_real_l2d_parquet_schema(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from data_parsing.l2d.dataset import track_points

    vehicle = [[36.0, 0.0, 0.0, 49.0, 8.4, 0.0, 0.0, 0.0], [36.0, 0.0, 0.0, 49.0001, 8.4, 0.0, 0.0, 0.0]]
    waypoints = [[[8.4, 49.0002]] * 10, [[8.4, 49.0003]] * 10]
    table = pa.table({
        "episode_index": pa.array([7, 7], pa.int64()),
        "observation.state.vehicle": pa.array(vehicle, pa.list_(pa.float32(), 8)),
        "observation.state.waypoints": pa.array(waypoints, pa.large_list(pa.large_list(pa.float32()))),
    })
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(table, tmp_path / "data" / "chunk-000" / "file-000.parquet")
    points = track_points(tmp_path, [7])
    assert points.shape == (4, 2)
    assert points[:, 0] == pytest.approx([8.4] * 4)


def test_track_points_split_by_episode_in_one_read(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from data_parsing.l2d.dataset import track_points_by_episode

    table = pa.table({
        "episode_index": pa.array([4, 4, 9], pa.int64()),
        "observation.state.vehicle": pa.array(
            [[0, 0, 0, 49.0, 8.4, 0, 0, 0], [0, 0, 0, 49.0, 8.4, 0, 0, 0], [0, 0, 0, 52.5, 13.4, 0, 0, 0]],
            pa.list_(pa.float32(), 8),
        ),
        "observation.state.waypoints": pa.array(
            [[[8.4, 49.0]] * 10, [[8.4, 49.0]] * 10, [[13.4, 52.5]] * 10],
            pa.large_list(pa.large_list(pa.float32())),
        ),
    })
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(table, tmp_path / "data" / "chunk-000" / "file-000.parquet")
    points = track_points_by_episode(tmp_path, [4, 9])
    # Each episode keeps only its own places, deduplicated.
    assert points[4].shape == (1, 2) and points[9][0, 1] == pytest.approx(52.5)
    with pytest.raises(ValueError, match="lack episodes"):
        track_points_by_episode(tmp_path, [4, 11])
