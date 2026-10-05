"""Tests for world-model quality evaluation helpers.

Pure-tensor tests need no GPU. Optional model-level impact test uses the
shared ``build_mock_model`` fixture (same pattern as faithfulness tests).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.world_model_quality import (
    jepa_reconstruction_metrics,
    null_predictor_metrics,
    open_loop_pair_metrics,
    relative_jepa_improvement,
    summarize_world_model_quality,
    train_world_model_quality,
    trajectory_impact_metrics,
    world_model_trajectory_impact,
)

B, C, H, W = 2, 8, 4, 4
NUM_VIEWS = 7


def _features(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(B, C, H, W, generator=g) for _ in range(4))


def test_jepa_zero_when_equal():
    target = _features(1)
    pred = tuple(t.clone() for t in target)
    m = jepa_reconstruction_metrics(pred, target)
    assert m["l1"] == pytest.approx(0.0, abs=1e-6)
    assert m["l2"] == pytest.approx(0.0, abs=1e-6)
    assert m["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert m["num_horizons"] == 4.0


def test_jepa_positive_when_different():
    pred = _features(0)
    target = _features(1)
    m = jepa_reconstruction_metrics(pred, target)
    assert m["l1"] > 0.0
    assert m["l2"] > 0.0
    assert "l1@h1" in m and "cosine@h4" in m


def test_jepa_shape_mismatch_raises():
    pred = _features(0)
    target = list(_features(1))
    target[0] = torch.randn(B, C, H, W + 1)
    with pytest.raises(ValueError, match="shape mismatch"):
        jepa_reconstruction_metrics(pred, target)


def test_null_predictor_worse_than_perfect():
    target = _features(2)
    perfect = jepa_reconstruction_metrics(tuple(t.clone() for t in target), target)
    null = null_predictor_metrics(target)
    assert null["null_l1"] > perfect["l1"]
    rel = relative_jepa_improvement(perfect, null)
    assert rel["rel_improvement_l1"] == pytest.approx(1.0, abs=1e-5)


def test_trajectory_impact_identical_is_zero():
    traj = torch.randn(B, 128)
    m = trajectory_impact_metrics(traj, traj.clone())
    assert m["trajectory_l2"] == pytest.approx(0.0, abs=1e-6)
    assert m["trajectory_l1"] == pytest.approx(0.0, abs=1e-6)


def test_trajectory_impact_detects_difference():
    a = torch.zeros(B, 128)
    b = torch.ones(B, 128)
    m = trajectory_impact_metrics(a, b)
    assert m["trajectory_l1"] == pytest.approx(1.0, abs=1e-5)
    assert m["trajectory_max_abs"] == pytest.approx(1.0, abs=1e-5)


def test_open_loop_pair_metrics_keys():
    # Perfect predictions → zero ADE for both branches.
    gt_a = np.zeros((B, 64), dtype=np.float64)
    gt_c = np.zeros((B, 64), dtype=np.float64)
    speed = np.full(B, 5.0)
    traj = torch.zeros(B, 128)  # accel block + curv block
    m = open_loop_pair_metrics(traj, traj, gt_a, gt_c, speed)
    assert m["reactive_ADE@3s"] == pytest.approx(0.0, abs=1e-6)
    assert m["combined_ADE@3s"] == pytest.approx(0.0, abs=1e-6)
    assert m["ade3s_delta_combined_minus_reactive"] == pytest.approx(0.0, abs=1e-6)


def test_summarize_merges_namespaces():
    s = summarize_world_model_quality(
        jepa={"l1": 0.5},
        impact={"trajectory_l2": 0.1},
        open_loop={"reactive_ADE@3s": 1.2},
    )
    assert s["jepa_l1"] == 0.5
    assert s["impact_trajectory_l2"] == 0.1
    assert s["ol_reactive_ADE@3s"] == 1.2


def test_world_model_impact_requires_wm(build_mock_model, device):
    model = build_mock_model(num_views=NUM_VIEWS, device=device, enable_world_model=False)
    inputs = (
        torch.randn(2, NUM_VIEWS, 3, 256, 256, device=device),
        torch.randn(2, 3, 256, 256, device=device),
        torch.randn(2, 896, device=device),
        torch.randn(2, 256, device=device),
    )
    with pytest.raises(ValueError, match="enable_world_model=True"):
        world_model_trajectory_impact(model, *inputs)


def test_world_model_impact_runs(build_mock_model, device):
    model = build_mock_model(num_views=NUM_VIEWS, device=device, enable_world_model=True)
    inputs = tuple(
        t.to(device)
        for t in (
            torch.randn(2, NUM_VIEWS, 3, 256, 256),
            torch.randn(2, 3, 256, 256),
            torch.randn(2, 896),
            torch.randn(2, 256),
        )
    )
    m = world_model_trajectory_impact(model, *inputs)
    assert "trajectory_l2" in m
    assert np.isfinite(m["trajectory_l2"])


def test_trained_quality_reports_ade_and_jepa(build_mock_model, device):
    torch.manual_seed(0)
    views = 6
    model = build_mock_model(
        num_views=views, device=device, enable_world_model=True,
    )
    b = 2
    batch = {
        "camera_tiles": torch.randn(b, views, 3, 256, 256, device=device),
        "map_context": torch.randn(b, 3, 256, 256, device=device),
        "visual_history": torch.zeros(b, 896, device=device),
        "egomotion_history": torch.randn(b, 256, device=device),
        "trajectory_target": torch.randn(b, 128, device=device),
        "history_frames": torch.randn(b, 4, views, 3, 256, 256, device=device),
        "future_frames": torch.randn(b, 4, views, 3, 256, 256, device=device),
    }
    metrics = train_world_model_quality(model, batch, steps=6, lr=1e-3)
    assert metrics["train_loss_last"] < metrics["train_loss_first"]
    assert np.isfinite(metrics["jepa_l1"])
    assert np.isfinite(metrics["ol_combined_ADE@3s"])
    assert np.isfinite(metrics["ol_reactive_ADE@3s"])
    assert "ol_ade3s_delta_combined_minus_reactive" in metrics
