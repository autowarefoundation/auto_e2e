"""Tests for the temporal front-camera clip builder (#98).

The teacher is shown one front-camera frame per horizon (0/+1/+2/+3/+4 s) taken
from the 1 Hz World-Model window. Verify the clip is the right length, front-only,
in temporal order, and that malformed windows raise rather than mislabel.
"""

from __future__ import annotations

import pytest
import torch

from data_processing.reasoning_label_generation.clip_builder import (
    FRONT_VIEW_INDEX,
    build_temporal_front_clip,
)
from data_processing.reasoning_label_generation.schema import NUM_HORIZONS


def _windows(n_hist, n_fut, V=6, H=8, W=8):
    # Encode each frame's (row, view) into pixel values so we can assert identity.
    hist = torch.zeros(n_hist, V, 3, H, W)
    fut = torch.zeros(n_fut, V, 3, H, W)
    for t in range(n_hist):
        for v in range(V):
            hist[t, v] = t * 100 + v
    for t in range(n_fut):
        for v in range(V):
            fut[t, v] = 1000 + t * 100 + v
    return hist, fut


def test_clip_has_num_horizons_frames_front_only_in_order():
    hist, fut = _windows(4, 4)
    clip = build_temporal_front_clip(hist, fut)
    assert len(clip) == NUM_HORIZONS
    # horizon 0 = current = last history row, front view
    assert torch.equal(clip[0], hist[-1, FRONT_VIEW_INDEX])
    # horizons 1..4 = future rows 0..3, front view, in order
    for h in range(NUM_HORIZONS - 1):
        assert torch.equal(clip[h + 1], fut[h, FRONT_VIEW_INDEX])
    # each frame is a single view [3, H, W], not multi-camera
    assert clip[0].shape == (3, 8, 8)


def test_front_view_is_index_zero():
    hist, fut = _windows(4, 4)
    clip = build_temporal_front_clip(hist, fut)
    # front view (index 0) has value = row*100 + 0; a non-front view would carry +v
    assert float(clip[0].flatten()[0]) == float(3 * 100 + FRONT_VIEW_INDEX)


def test_missing_windows_raise():
    with pytest.raises(ValueError, match="both history_frames and future_frames"):
        build_temporal_front_clip(None, None)


def test_too_few_future_frames_raise():
    hist, fut = _windows(4, 2)  # only 2 future, need NUM_HORIZONS-1 = 4
    with pytest.raises(ValueError, match="future frames"):
        build_temporal_front_clip(hist, fut)


def test_wrong_ndim_raises():
    with pytest.raises(ValueError, match=r"\[N, V, 3, H, W\]"):
        build_temporal_front_clip(torch.zeros(4, 3, 8, 8), torch.zeros(4, 3, 8, 8))
