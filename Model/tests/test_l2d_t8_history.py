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
    return dataset


def _driving_north(count: int = 200, speed_mps: float = 10.0) -> np.ndarray:
    states = np.zeros((count, 8), dtype=np.float32)
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
