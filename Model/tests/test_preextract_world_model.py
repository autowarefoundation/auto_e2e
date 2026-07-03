"""Tests for the offline World Model window pre-extraction (#16 / #30).

Pure planning + assembly; no lerobot, no video decode — a synthetic in-memory
frame store stands in for the JPEG cache.
"""

import pytest
import torch

from data_parsing.preextract_world_model import (
    assemble_window,
    extract_episode,
    plan_episode_windows,
)
from data_parsing.l2d.world_model_windows import build_windows, window_offsets

V, H, W = 7, 3, 8  # tiny multi-view frame for tests


def _fake_frame(idx: int) -> torch.Tensor:
    """A [V,3,H,W] frame whose contents encode its own index (for assertions)."""
    return torch.full((V, 3, H, W), float(idx))


# ---- planning --------------------------------------------------------------

def test_plan_valid_range_and_window_indices():
    valid, per_sample, unique = plan_episode_windows(100, num_frames=4, stride=10)
    # back=(N-1)*s=30, fwd=N*s=40 → valid = [30, 60)
    assert valid[0] == 30 and valid[-1] == 59 and len(valid) == 30
    # current frame is the LAST of the history; future starts at +stride
    h, f = per_sample[30]
    assert h == [0, 10, 20, 30] and f == [40, 50, 60, 70]
    h2, f2 = per_sample[59]
    assert h2 == [29, 39, 49, 59] and f2 == [69, 79, 89, 99]
    # every index stays inside the episode
    assert min(unique) >= 0 and max(unique) <= 99


def test_plan_is_content_addressed_not_blown_up():
    valid, _, unique = plan_episode_windows(100, num_frames=4, stride=10)
    naive = 2 * 4 * len(valid)          # what storing 2*N frames per sample would cost
    assert len(unique) <= 100            # bounded by episode length...
    assert len(unique) < naive           # ...and far below the naive per-sample count


def test_plan_too_short_episode_has_no_samples():
    valid, per_sample, unique = plan_episode_windows(40, num_frames=4, stride=10)
    assert valid == [] and per_sample == {} and unique == []


@pytest.mark.parametrize("num_frames,stride", [(4, 10), (4, 1), (6, 5), (2, 3)])
def test_plan_indices_consistent_with_window_offsets(num_frames, stride):
    hist_off, fut_off = window_offsets(num_frames, stride)
    _, per_sample, _ = plan_episode_windows(200, num_frames=num_frames, stride=stride)
    s = next(iter(per_sample))
    h, f = per_sample[s]
    assert h == [s + o for o in hist_off]
    assert f == [s + o for o in fut_off]


# ---- assembly --------------------------------------------------------------

def test_assemble_shapes_and_order_from_mapping():
    _, per_sample, _ = plan_episode_windows(100, num_frames=4, stride=10)
    h_idx, f_idx = per_sample[30]
    store = {i: _fake_frame(i) for i in (h_idx + f_idx)}
    history, future = assemble_window(store, h_idx, f_idx)
    assert history.shape == (4, V, 3, H, W) and future.shape == (4, V, 3, H, W)
    # frame k must equal the frame stored at h_idx[k] (order preserved)
    for k, idx in enumerate(h_idx):
        assert torch.equal(history[k], _fake_frame(idx))
    for k, idx in enumerate(f_idx):
        assert torch.equal(future[k], _fake_frame(idx))


def test_assemble_accepts_callable_store():
    _, per_sample, _ = plan_episode_windows(100, num_frames=4, stride=10)
    h_idx, f_idx = per_sample[45]
    history, future = assemble_window(_fake_frame, h_idx, f_idx)  # lazy loader
    assert history.shape == (4, V, 3, H, W)
    assert torch.equal(history[-1], _fake_frame(45))  # current frame last in history


# ---- end-to-end: extract once → assemble matches the online build_windows --

def test_extract_then_assemble_matches_online_build_windows():
    episode_len = 100
    store: dict[int, torch.Tensor] = {}
    valid, per_sample = extract_episode(
        _fake_frame, episode_len, store.__setitem__, num_frames=4, stride=10)

    # each needed frame was decoded exactly once
    assert len(store) == len(set(store))

    for s in (valid[0], valid[len(valid) // 2], valid[-1]):
        h_idx, f_idx = per_sample[s]
        pre_h, pre_f = assemble_window(store, h_idx, f_idx)

        # the online path (decode every frame on the fly) must give the same thing
        on_h, on_f = build_windows(
            load_frame=lambda row: _fake_frame(row),
            row=s, ep_start=0, ep_end=episode_len, num_frames=4, stride=10)
        assert torch.equal(pre_h, on_h) and torch.equal(pre_f, on_f)
