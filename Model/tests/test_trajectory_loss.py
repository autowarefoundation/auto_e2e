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
