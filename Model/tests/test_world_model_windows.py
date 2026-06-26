"""Tests for the World Model 1 Hz sequential windows (#16, enables JEPA #13).

The windowing logic is dataset-agnostic (takes a frame-loader callable), so it
is fully tested here without the real lerobot/L2D dataset.
"""

import pytest
import torch

from data_parsing.l2d.world_model_windows import (
    build_windows,
    required_margins,
    stride_for_hz,
    window_offsets,
)


def _fake_loader(num_views=2):
    """load_frame(row) -> [V, 3, 2, 2] filled with the row value (so a window's
    loaded rows are recoverable from the tensor contents)."""
    def load(row: int) -> torch.Tensor:
        return torch.full((num_views, 3, 2, 2), float(row))
    return load


# --- stride_for_hz -------------------------------------------------------------

def test_stride_10hz_to_1hz_is_10():
    assert stride_for_hz(10.0, 1.0) == 10

def test_stride_30hz_to_1hz_is_30():
    assert stride_for_hz(30.0, 1.0) == 30

def test_stride_rounds_and_floors_at_one():
    assert stride_for_hz(10.0, 2.0) == 5
    assert stride_for_hz(1.0, 10.0) == 1   # never below 1

def test_stride_invalid_raises():
    for bad in [(0, 1), (10, 0), (-1, 1)]:
        with pytest.raises(ValueError):
            stride_for_hz(*bad)


# --- window_offsets / required_margins ----------------------------------------

def test_window_offsets_default():
    hist, fut = window_offsets(num_frames=4, stride=10)
    assert hist == [-30, -20, -10, 0]   # oldest -> newest, current last
    assert fut == [10, 20, 30, 40]      # next N frames

def test_window_offsets_single_frame():
    hist, fut = window_offsets(num_frames=1, stride=10)
    assert hist == [0] and fut == [10]

def test_required_margins():
    assert required_margins(4, 10) == (30, 40)   # (N-1)*s before, N*s after

def test_offsets_invalid_raises():
    with pytest.raises(ValueError):
        window_offsets(0, 10)
    with pytest.raises(ValueError):
        window_offsets(4, 0)


# --- build_windows ------------------------------------------------------------

def test_build_windows_shapes():
    hist, fut = build_windows(_fake_loader(num_views=7), row=64,
                              ep_start=0, ep_end=200, num_frames=4, stride=10)
    assert hist.shape == (4, 7, 3, 2, 2)
    assert fut.shape == (4, 7, 3, 2, 2)

def test_build_windows_loads_correct_rows_oldest_to_newest():
    hist, fut = build_windows(_fake_loader(), row=50,
                              ep_start=0, ep_end=100, num_frames=4, stride=10)
    # history rows: 20,30,40,50 (current last); future: 60,70,80,90
    assert [hist[i, 0, 0, 0, 0].item() for i in range(4)] == [20, 30, 40, 50]
    assert [fut[i, 0, 0, 0, 0].item() for i in range(4)] == [60, 70, 80, 90]

def test_build_windows_history_ends_at_current():
    hist, _ = build_windows(_fake_loader(), row=33, ep_start=0, ep_end=100,
                            num_frames=4, stride=10)
    assert hist[-1, 0, 0, 0, 0].item() == 33   # newest history frame == current

def test_build_windows_raises_when_past_exceeds_episode():
    with pytest.raises(IndexError):
        build_windows(_fake_loader(), row=20, ep_start=0, ep_end=100,
                      num_frames=4, stride=10)   # needs row-30 = -10 < 0

def test_build_windows_raises_when_future_exceeds_episode():
    with pytest.raises(IndexError):
        build_windows(_fake_loader(), row=95, ep_start=0, ep_end=100,
                      num_frames=4, stride=10)   # needs row+40 = 135 >= 100

def test_build_windows_respects_episode_start_offset():
    # episode is rows [100, 200); current at 150 must read within it.
    hist, fut = build_windows(_fake_loader(), row=150, ep_start=100, ep_end=200,
                              num_frames=4, stride=10)
    assert [hist[i, 0, 0, 0, 0].item() for i in range(4)] == [120, 130, 140, 150]
    assert [fut[i, 0, 0, 0, 0].item() for i in range(4)] == [160, 170, 180, 190]
