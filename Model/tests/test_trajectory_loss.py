import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_components.losses import TrajectoryImitationLoss


class TestTrajectoryImitationLoss:
    def test_output_is_scalar(self):
        loss_fn = TrajectoryImitationLoss()
        pred = torch.randn(4, 128)
        target = torch.randn(4, 128)
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_gradient_flows_to_input(self):
        loss_fn = TrajectoryImitationLoss()
        pred = torch.randn(4, 128, requires_grad=True)
        target = torch.randn(4, 128)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == (4, 128)

    def test_temporal_weighting_changes_loss(self):
        pred = torch.randn(4, 128)
        target = torch.randn(4, 128)

        uniform_loss = TrajectoryImitationLoss(temporal_decay=1.0)(pred, target)
        decayed_loss = TrajectoryImitationLoss(temporal_decay=0.9)(pred, target)

        assert uniform_loss.item() != decayed_loss.item()

    def test_zero_input_produces_zero_loss(self):
        loss_fn = TrajectoryImitationLoss()
        pred = torch.zeros(4, 128)
        target = torch.zeros(4, 128)
        loss = loss_fn(pred, target)
        assert loss.item() == 0.0

    def test_smooth_l1_vs_mse_differ(self):
        pred = torch.randn(4, 128)
        target = torch.randn(4, 128)

        l1_loss = TrajectoryImitationLoss(loss_type="smooth_l1")(pred, target)
        mse_loss = TrajectoryImitationLoss(loss_type="mse")(pred, target)

        assert l1_loss.item() != mse_loss.item()

    def test_invalid_loss_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported loss_type"):
            TrajectoryImitationLoss(loss_type="l1")

    def test_per_signal_normalization_gives_curvature_gradient(self):
        # Regression for the flat-ADE bug: curvature (signal 1) is ~40x smaller
        # than accel (signal 0). Without per-signal normalization, a small
        # curvature error sits in SmoothL1's quadratic regime and produces a
        # near-zero gradient, so the planner never learns curvature. After
        # normalization, the per-element curvature gradient must be comparable
        # in magnitude to the accel gradient for equal-in-std errors.
        loss_fn = TrajectoryImitationLoss(signal_scales=(0.54, 0.014))
        # A pred that is off by ~1 std on BOTH signals (accel +0.54, curv +0.014).
        pred = torch.zeros(1, 128, requires_grad=True)
        target = torch.zeros(1, 64, 2)
        target[..., 0] = 0.54    # accel target
        target[..., 1] = 0.014   # curvature target
        loss = loss_fn(pred, target.view(1, 128))
        loss.backward()
        g = pred.grad.view(64, 2)
        accel_g = g[:, 0].abs().mean().item()
        curv_g = g[:, 1].abs().mean().item()
        # Comparable within 2x (would be ~40x apart without normalization).
        assert curv_g > 0.3 * accel_g, (accel_g, curv_g)

    def test_signal_scales_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="signal_scales must have"):
            TrajectoryImitationLoss(num_signals=2, signal_scales=(1.0,))
