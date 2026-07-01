"""Unit tests for the JEPA feature-reconstruction loss and the optional
action conditioning of FutureState (Issue #13).

The loss compares FutureState's predicted future BEV features against
targets that, in production, come from a frozen backbone applied to the
future frames (+1.6/3.2/4.8/6.4s). Here we use random tensors of the
correct shapes — the contract under test is shapes/reduction/gradients,
not the data pipeline.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_components.future_state import FutureState
from model_components.losses import FeatureReconstructionLoss

EMBED_DIM = 32  # small for fast tests; production uses 256
B, H, W = 2, 8, 8


def _make_features(requires_grad=False, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return tuple(
        torch.randn(B, EMBED_DIM, H, W, requires_grad=requires_grad)
        for _ in range(4)
    )


class TestLossType:
    """loss_type l1/l2/smooth_l1 (V-JEPA uses L1; #13)."""

    def test_invalid_loss_type_raises(self):
        with pytest.raises(ValueError, match="loss_type"):
            FeatureReconstructionLoss(loss_type="hinge")

    def test_l1_matches_mean_abs(self):
        torch.manual_seed(0)
        target = _make_features(seed=1)
        pred = _make_features(seed=2)
        loss = FeatureReconstructionLoss(loss_type="l1")(pred, target)
        expected = torch.stack([(p - t).abs().mean()
                                for p, t in zip(pred, target)]).mean()
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_all_loss_types_zero_when_equal(self):
        target = _make_features()
        pred = tuple(t.clone() for t in target)
        for lt in ("l1", "l2", "smooth_l1"):
            loss = FeatureReconstructionLoss(loss_type=lt)(pred, target)
            assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestFeatureReconstructionLoss:
    def test_zero_when_pred_equals_target(self):
        loss_fn = FeatureReconstructionLoss()
        target = _make_features()
        pred = tuple(t.clone() for t in target)
        loss = loss_fn(pred, target)
        assert loss.dim() == 0
        assert loss.item() == pytest.approx(0.0, abs=1e-8)

    def test_positive_when_pred_differs(self):
        loss_fn = FeatureReconstructionLoss()
        pred = _make_features(seed=0)
        target = _make_features(seed=1)
        loss = loss_fn(pred, target)
        assert loss.item() > 0.0

    def test_gradient_flows_to_predictions(self):
        loss_fn = FeatureReconstructionLoss()
        pred = _make_features(requires_grad=True)
        target = _make_features()
        loss = loss_fn(pred, target)
        loss.backward()
        for t in pred:
            assert t.grad is not None
            assert torch.isfinite(t.grad).all()

    def test_temporal_weighting_changes_loss(self):
        pred = _make_features(seed=0)
        target = _make_features(seed=1)
        uniform = FeatureReconstructionLoss()(pred, target)
        weighted = FeatureReconstructionLoss(
            temporal_weights=[8.0, 1.0, 1.0, 1.0]
        )(pred, target)
        assert not torch.isclose(uniform, weighted)

    def test_reduction_none_returns_per_step(self):
        loss_fn = FeatureReconstructionLoss(reduction="none")
        pred = _make_features(seed=0)
        target = _make_features(seed=1)
        per_step = loss_fn(pred, target)
        assert per_step.shape == (4,)
        assert (per_step > 0).all()

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError):
            FeatureReconstructionLoss(reduction="sum")

    def test_wrong_number_of_steps_raises(self):
        loss_fn = FeatureReconstructionLoss()
        feats = _make_features()
        with pytest.raises(ValueError):
            loss_fn(feats[:3], feats)
        with pytest.raises(ValueError):
            loss_fn(feats, feats[:3])

    def test_shape_mismatch_raises(self):
        loss_fn = FeatureReconstructionLoss()
        pred = _make_features()
        target = list(_make_features())
        target[2] = torch.randn(B, EMBED_DIM, H + 1, W)
        with pytest.raises(ValueError):
            loss_fn(pred, target)

    def test_zero_sum_temporal_weights_do_not_produce_nan(self):
        """A zero-sum weight vector must not poison the loss with NaN
        (normalisation would otherwise divide by zero)."""
        loss_fn = FeatureReconstructionLoss(
            temporal_weights=[0.0, 0.0, 0.0, 0.0]
        )
        assert torch.isfinite(loss_fn.temporal_weights).all()
        pred = _make_features(requires_grad=True, seed=0)
        target = _make_features(seed=1)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss).all()
        loss.backward()
        for t in pred:
            assert torch.isfinite(t.grad).all()

    def test_wrong_temporal_weights_length_raises(self):
        with pytest.raises(ValueError):
            FeatureReconstructionLoss(temporal_weights=[1.0, 2.0])

    def test_end_to_end_with_future_state(self):
        """Loss must propagate gradients through FutureState parameters."""
        model = FutureState(embed_dim=EMBED_DIM, ego_hidden_dim=EMBED_DIM)
        fused = torch.randn(B, EMBED_DIM, H, W)
        ego_hidden = torch.randn(B, EMBED_DIM)
        pred = model(fused, ego_hidden)
        target = tuple(t.detach() + 0.1 for t in pred)
        loss = FeatureReconstructionLoss()(pred, target)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestFutureStateActionConditioning:
    ACTION_DIM = 128  # 64 timesteps x 2 signals

    def test_default_behavior_unchanged_shapes(self):
        """action_dim=None (default) keeps the original contract intact."""
        model = FutureState(embed_dim=EMBED_DIM, ego_hidden_dim=EMBED_DIM)
        assert model.action_proj is None
        fused = torch.randn(B, EMBED_DIM, H, W)
        ego_hidden = torch.randn(B, EMBED_DIM)
        out = model(fused, ego_hidden)
        assert len(out) == 4
        for t in out:
            assert t.shape == (B, EMBED_DIM, H, W)

    def test_trajectory_changes_output_when_action_dim_set(self):
        torch.manual_seed(0)
        model = FutureState(
            embed_dim=EMBED_DIM, ego_hidden_dim=EMBED_DIM,
            action_dim=self.ACTION_DIM,
        )
        fused = torch.randn(B, EMBED_DIM, H, W)
        ego_hidden = torch.randn(B, EMBED_DIM)
        traj_a = torch.randn(B, self.ACTION_DIM)
        traj_b = torch.randn(B, self.ACTION_DIM)
        out_a = model(fused, ego_hidden, trajectory=traj_a)
        out_b = model(fused, ego_hidden, trajectory=traj_b)
        assert not torch.allclose(out_a[0], out_b[0]), (
            "Counterfactual trajectories must produce different rollouts"
        )

    def test_action_dim_set_but_no_trajectory_still_works(self):
        model = FutureState(
            embed_dim=EMBED_DIM, ego_hidden_dim=EMBED_DIM,
            action_dim=self.ACTION_DIM,
        )
        fused = torch.randn(B, EMBED_DIM, H, W)
        ego_hidden = torch.randn(B, EMBED_DIM)
        out = model(fused, ego_hidden)
        assert len(out) == 4

    def test_trajectory_without_action_dim_raises(self):
        model = FutureState(embed_dim=EMBED_DIM, ego_hidden_dim=EMBED_DIM)
        fused = torch.randn(B, EMBED_DIM, H, W)
        ego_hidden = torch.randn(B, EMBED_DIM)
        traj = torch.randn(B, self.ACTION_DIM)
        with pytest.raises(ValueError):
            model(fused, ego_hidden, trajectory=traj)

    def test_gradient_flows_through_trajectory(self):
        """Gradients must reach the trajectory — required for training the
        planner through the world-model loss."""
        model = FutureState(
            embed_dim=EMBED_DIM, ego_hidden_dim=EMBED_DIM,
            action_dim=self.ACTION_DIM,
        )
        fused = torch.randn(B, EMBED_DIM, H, W)
        ego_hidden = torch.randn(B, EMBED_DIM)
        traj = torch.randn(B, self.ACTION_DIM, requires_grad=True)
        out = model(fused, ego_hidden, trajectory=traj)
        sum(t.pow(2).mean() for t in out).backward()
        assert traj.grad is not None
        assert torch.isfinite(traj.grad).all()
        assert model.action_proj.weight.grad is not None
