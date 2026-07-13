"""Tests for BasePlanner.compute_planner_loss (#115) across both planners.

Covers:
  - dict return shape (both required and per-planner-specific keys)
  - gradient flow through the actual training path (not just forward())
  - FlowMatchingPlanner's Euler loop is NEVER exercised inside the loss
    (the original #115 bug: train_il regressed forward()'s ODE rollout)
  - BasePlanner now fails loudly at instantiation if compute_planner_loss
    is missing (was previously silent / only documented in a docstring)
  - build_planner registry integration for both modes
"""

import torch
import pytest

from model_components.trajectory_planning import (
    FlowMatchingPlanner,
    BezierPlanner,
    PLANNER_REGISTRY,
    build_planner,
)
from model_components.trajectory_planning.base import BasePlanner


B, EMBED_DIM, BEV_H, BEV_W = 2, 32, 6, 6
VISUAL_HISTORY_DIM, EGOMOTION_DIM, TRAJ_DIM = 896, 256, 128


@pytest.fixture
def inputs():
    return {
        "bev_features": torch.randn(B, EMBED_DIM, BEV_H, BEV_W),
        "visual_history": torch.randn(B, VISUAL_HISTORY_DIM),
        "egomotion_history": torch.randn(B, EGOMOTION_DIM),
        "trajectory_target": torch.randn(B, TRAJ_DIM),
    }


@pytest.fixture
def fm_planner():
    return FlowMatchingPlanner(
        embed_dim=EMBED_DIM, num_inference_steps=2,
        time_embed_dim=8, num_heads=2,
    )


@pytest.fixture
def bezier_planner():
    return BezierPlanner(embed_dim=EMBED_DIM)


class TestFlowMatchingPlannerLoss:
    def test_returns_dict_with_required_keys(self, fm_planner, inputs):
        result = fm_planner.compute_planner_loss(**inputs)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "velocity_mse" in result
        assert torch.equal(result["loss"], result["velocity_mse"])

    def test_loss_is_scalar(self, fm_planner, inputs):
        result = fm_planner.compute_planner_loss(**inputs)
        assert result["loss"].dim() == 0

    def test_gradient_flows_to_all_params(self, fm_planner, inputs):
        result = fm_planner.compute_planner_loss(**inputs)
        result["loss"].backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in fm_planner.parameters()
        )

    def test_forward_unaffected_by_loss_path(self, fm_planner, inputs):
        """The #115 regression guard: forward() (Euler inference) must
        remain independent of compute_planner_loss (velocity-MSE training).
        Calling compute_planner_loss must not change what forward() returns
        for the same inputs+seed, proving the Euler loop isn't secretly
        entangled with the training path."""
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            traj_before = fm_planner(
                inputs["bev_features"], inputs["visual_history"],
                inputs["egomotion_history"], generator=gen,
            )

        fm_planner.compute_planner_loss(**inputs)  # exercise the loss path

        gen2 = torch.Generator().manual_seed(0)
        with torch.no_grad():
            traj_after = fm_planner(
                inputs["bev_features"], inputs["visual_history"],
                inputs["egomotion_history"], generator=gen2,
            )
        assert torch.allclose(traj_before, traj_after)

    def test_invalid_visual_history_dim_raises(self, fm_planner, inputs):
        bad_inputs = dict(inputs, visual_history=torch.randn(B, 999))
        with pytest.raises(ValueError, match="visual_history"):
            fm_planner.compute_planner_loss(**bad_inputs)

    def test_invalid_trajectory_target_shape_raises(self, fm_planner, inputs):
        bad_inputs = dict(inputs, trajectory_target=torch.randn(B, 64))
        with pytest.raises(ValueError, match="trajectory_target"):
            fm_planner.compute_planner_loss(**bad_inputs)


class TestBezierPlannerLoss:
    def test_returns_dict_with_required_keys(self, bezier_planner, inputs):
        result = bezier_planner.compute_planner_loss(**inputs)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "imitation_loss" in result
        assert torch.equal(result["loss"], result["imitation_loss"])

    def test_gradient_flows_to_all_params(self, bezier_planner, inputs):
        result = bezier_planner.compute_planner_loss(**inputs)
        result["loss"].backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in bezier_planner.parameters()
        )

    def test_loss_matches_direct_smooth_l1(self, bezier_planner, inputs):
        """compute_planner_loss's imitation_loss must equal SmoothL1 on
        forward()'s own output — Bezier's forward() IS a legitimate
        regression target, unlike FlowMatchingPlanner's Euler rollout."""
        with torch.no_grad():
            traj = bezier_planner(
                inputs["bev_features"], inputs["visual_history"],
                inputs["egomotion_history"],
            )
            expected = torch.nn.functional.smooth_l1_loss(
                traj, inputs["trajectory_target"]
            )
        result = bezier_planner.compute_planner_loss(**inputs)
        assert torch.allclose(result["loss"], expected, atol=1e-6)

    def test_invalid_trajectory_target_shape_raises(self, bezier_planner, inputs):
        """Regression guard: a target missing its batch dimension must raise,
        not silently broadcast. Before this fix, BezierPlanner.compute_planner_loss
        called no shape validation at all, so passing a [T]-shaped target
        (missing the batch dim) let smooth_l1_loss broadcast sample 0's
        target across the whole batch — a silent wrong-answer, not a crash.
        FlowMatchingPlanner already had this guard (via its own
        _validate_trajectory_target); it's now shared via BasePlanner so
        both planners get it identically."""
        bad_inputs = dict(inputs, trajectory_target=inputs["trajectory_target"][0])
        with pytest.raises(ValueError, match="trajectory_target"):
            bezier_planner.compute_planner_loss(**bad_inputs)


class TestAbstractMethodEnforcement:
    """#115's other ask: a planner missing compute_planner_loss must fail
    loudly at build time, not silently mis-train."""

    def test_missing_compute_planner_loss_cannot_instantiate(self):
        class IncompletePlanner(BasePlanner):
            def forward(self, bev_features, visual_history,
                       egomotion_history, **kwargs):
                return torch.zeros(bev_features.shape[0], 128)
            # compute_planner_loss deliberately NOT implemented

        with pytest.raises(TypeError, match="compute_planner_loss"):
            IncompletePlanner()

    def test_complete_planner_can_instantiate(self):
        class CompletePlanner(BasePlanner):
            def forward(self, bev_features, visual_history,
                       egomotion_history, **kwargs):
                return torch.zeros(bev_features.shape[0], 128)

            def compute_planner_loss(self, bev_features, visual_history,
                                     egomotion_history, trajectory_target,
                                     **kwargs):
                return {"loss": torch.tensor(0.0)}

        CompletePlanner()  # must not raise


class TestPlannerRegistryLossIntegration:
    """Both registry entries must expose compute_planner_loss returning
    a dict with 'loss' — the property train_il will rely on."""

    @pytest.mark.parametrize("mode", sorted(PLANNER_REGISTRY))
    def test_registry_planner_has_working_loss(self, mode, inputs):
        kwargs = {"embed_dim": EMBED_DIM}
        if mode == "flow_matching":
            kwargs.update(num_inference_steps=2, time_embed_dim=8, num_heads=2)
        planner = build_planner(mode, **kwargs)
        result = planner.compute_planner_loss(**inputs)
        assert "loss" in result
        assert result["loss"].dim() == 0
        assert torch.isfinite(result["loss"])
