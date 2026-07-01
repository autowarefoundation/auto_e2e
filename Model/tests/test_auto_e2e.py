"""End-to-end tests for AutoE2E after the #86 refactor.

Post-refactor contract (#88):
- ``AutoE2E`` wraps ``Reactive_E2E``; the backbone / view fusion / map encoder /
  planner all live under ``Reactive_E2E.*``.
- ``forward(...)`` returns ONLY ``trajectory`` ``[B, num_timesteps*num_signals]``
  in every mode. The old 3-tuple ``(loss/traj, ego_hidden, future_features)`` and
  the BEV ``FutureState`` output were removed (the planner loss is computed
  separately via ``compute_planner_loss``; see ``test_trajectory_planning.py``).
- Only ``bev`` view fusion exists (``concat``/``cross_attn`` were removed).
"""

import sys

import pytest
import torch

sys.path.append('..')


def make_inputs(batch_size, num_views, device, include_camera_params=False):
    visual = torch.randn(batch_size, num_views, 3, 256, 256, device=device)
    map_input = torch.randn(batch_size, 3, 256, 256, device=device)
    visual_history = torch.randn(batch_size, 896, device=device)
    egomotion = torch.randn(batch_size, 256, device=device)
    if include_camera_params:
        camera_params = torch.randn(batch_size, num_views, 3, 4, device=device)
        return visual, map_input, visual_history, egomotion, camera_params
    return visual, map_input, visual_history, egomotion


# Submodule param-name prefixes that are actually exercised by the reactive
# forward pass. FutureState is instantiated but unused in this path, and the
# MapEncoder is gated by ResidualMapFusion's alpha=0 init, so neither receives
# gradient at init — both are excluded from the gradient-coverage checks.
USED_GROUPS = ["Reactive_E2E.Backbone", "Reactive_E2E.FeatureFusion",
               "Reactive_E2E.TrajectoryPlanner"]
NO_GRAD_OK = ("Reactive_E2E.MapEncoder.", "Reactive_E2E.FutureState.")


# ---------------------------------------------------------------------------
# 1. Output shape correctness
# ---------------------------------------------------------------------------

class TestOutputShapes:
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_trajectory_shape(self, model, device, batch_size):
        visual, map_input, vis_hist, ego = make_inputs(batch_size, 7, device)
        traj = model(visual, map_input, vis_hist, ego, mode="infer")
        assert traj.shape == (batch_size, 128)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_train_mode_also_returns_trajectory(self, model, device, batch_size):
        """forward returns the trajectory in train mode too (the loss is a
        separate planner concern)."""
        visual, map_input, vis_hist, ego = make_inputs(batch_size, 7, device)
        target = torch.randn(batch_size, 128, device=device)
        traj = model(visual, map_input, vis_hist, ego, mode="train",
                     trajectory_target=target)
        assert traj.shape == (batch_size, 128)


# ---------------------------------------------------------------------------
# 2. Batch independence — changing one sample must not affect others
# ---------------------------------------------------------------------------

class TestBatchIndependence:
    def test_samples_do_not_interfere(self, model, device):
        model.eval()
        torch.manual_seed(42)
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)

        traj_both = model(visual, map_input, vis_hist, ego, mode="infer")
        traj_single = model(visual[0:1], map_input[0:1], vis_hist[0:1], ego[0:1],
                            mode="infer")

        assert torch.allclose(traj_both[0], traj_single[0], atol=1e-5), \
            "Batch samples are interfering with each other"

    def test_different_batch_neighbor_no_effect(self, model, device):
        model.eval()
        torch.manual_seed(42)
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)

        traj_a = model(visual, map_input, vis_hist, ego, mode="infer")

        visual_modified = visual.clone()
        visual_modified[1] = torch.randn_like(visual_modified[1])
        traj_b = model(visual_modified, map_input, vis_hist, ego, mode="infer")

        assert torch.allclose(traj_a[0], traj_b[0], atol=1e-5), \
            "Modifying another sample in the batch affected this sample's output"


# ---------------------------------------------------------------------------
# 4. Gradient flow — the exercised path receives gradients
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_backward_succeeds(self, model, device):
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        traj = model(visual, map_input, vis_hist, ego, mode="infer")
        traj.pow(2).mean().backward()

    def test_used_path_parameters_have_gradients(self, model, device):
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        traj = model(visual, map_input, vis_hist, ego, mode="infer")
        traj.pow(2).mean().backward()

        missing = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
            and not name.startswith(NO_GRAD_OK)
        ]
        assert not missing, f"Parameters with no gradient: {missing}"

    def test_no_vanishing_gradients(self, model, device):
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        traj = model(visual, map_input, vis_hist, ego, mode="infer")
        traj.pow(2).mean().backward()

        zero_grad = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
            and p.grad.abs().max() == 0
            and not name.startswith(NO_GRAD_OK)
        ]
        assert not zero_grad, f"Parameters with all-zero gradients: {zero_grad}"


# ---------------------------------------------------------------------------
# 5. num_views flexibility — model works with different view counts (BEV only)
# ---------------------------------------------------------------------------

class TestNumViewsFlexibility:
    @pytest.mark.parametrize("num_views", [1, 4, 8, 12])
    def test_various_num_views(self, build_mock_model, device, num_views):
        model = build_mock_model(num_views, device=device)
        visual, map_input, vis_hist, ego = make_inputs(2, num_views, device)
        traj = model(visual, map_input, vis_hist, ego, mode="infer")
        assert traj.shape == (2, 128)


# ---------------------------------------------------------------------------
# 6. Numerical stability — no NaN or Inf
# ---------------------------------------------------------------------------

class TestNumericalStability:
    def test_no_nan_in_outputs(self, model, device):
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        traj = model(visual, map_input, vis_hist, ego, mode="infer")
        assert not torch.isnan(traj).any(), "NaN in trajectory"

    def test_no_inf_in_outputs(self, model, device):
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        traj = model(visual, map_input, vis_hist, ego, mode="infer")
        assert not torch.isinf(traj).any(), "Inf in trajectory"

    def test_large_input_values(self, model, device):
        """Model should not produce NaN/Inf even with large inputs."""
        visual = torch.randn(1, 7, 3, 256, 256, device=device) * 100
        map_input = torch.randn(1, 3, 256, 256, device=device) * 100
        vis_hist = torch.randn(1, 896, device=device) * 100
        ego = torch.randn(1, 256, device=device) * 100
        traj = model(visual, map_input, vis_hist, ego, mode="infer")

        assert not torch.isnan(traj).any(), "NaN with large inputs"
        assert not torch.isinf(traj).any(), "Inf with large inputs"


# ---------------------------------------------------------------------------
# Training loop integration — optimizer.step + loss on the trajectory
# ---------------------------------------------------------------------------

class TestTrainingLoop:
    def test_optimizer_step_updates_parameters(self, build_mock_model, device):
        """forward → MSE on trajectory → backward → optimizer.step() must move
        parameters in every exercised submodule (Backbone, FeatureFusion,
        TrajectoryPlanner), not just the last layer."""
        model = build_mock_model(num_views=7, device=device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

        before = {n: p.detach().clone() for n, p in model.named_parameters()
                  if p.requires_grad}

        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        target = torch.randn(2, 128, device=device)
        traj = model(visual, map_input, vis_hist, ego, mode="train",
                     trajectory_target=target)
        loss = (traj - target).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        changed = {g: False for g in USED_GROUPS}
        for name, p in model.named_parameters():
            if not p.requires_grad or torch.equal(p.detach(), before[name]):
                continue
            for prefix in USED_GROUPS:
                if name.startswith(prefix + "."):
                    changed[prefix] = True

        unchanged = [g for g, ok in changed.items() if not ok]
        assert not unchanged, \
            f"optimizer.step() did not update any parameter in: {unchanged}"

    def test_model_to_loss_backward_integration(self, build_mock_model, device):
        """Pipe the trajectory output into an MSE loss and run backward; the
        upstream Backbone and the downstream TrajectoryPlanner must each see a
        nonzero gradient (full network depth, not just the last layer)."""
        model = build_mock_model(num_views=7, device=device)
        model.train()

        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        target = torch.randn(2, 128, device=device)
        traj = model(visual, map_input, vis_hist, ego, mode="train",
                     trajectory_target=target)
        loss = (traj - target).pow(2).mean()

        assert loss.dim() == 0 and loss.requires_grad
        assert torch.isfinite(loss), "Loss is non-finite"
        loss.backward()

        groups = {"Reactive_E2E.Backbone": False,
                  "Reactive_E2E.TrajectoryPlanner": False}
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None or p.grad.abs().max() == 0:
                continue
            for prefix in groups:
                if name.startswith(prefix + "."):
                    groups[prefix] = True
        for prefix, has_grad in groups.items():
            assert has_grad, f"No parameter in {prefix} received nonzero gradient"


# ---------------------------------------------------------------------------
# Full-pipeline robustness
# ---------------------------------------------------------------------------

class TestFullPipelineRobustness:
    def test_all_zero_inputs_produce_finite_outputs(self, build_mock_model, device):
        """Zero inputs across all paths must not cause NaN/Inf downstream."""
        model = build_mock_model(num_views=7, device=device)
        model.eval()

        visual = torch.zeros(2, 7, 3, 256, 256, device=device)
        map_input = torch.zeros(2, 3, 256, 256, device=device)
        vis_hist = torch.zeros(2, 896, device=device)
        ego = torch.zeros(2, 256, device=device)

        traj = model(visual, map_input, vis_hist, ego, mode="infer")
        assert torch.isfinite(traj).all(), "NaN/Inf in trajectory with zero inputs"

    def test_camera_params_none_then_valid_switching(self, build_mock_model, device):
        """A BEV-fusion model must accept both None and valid camera_params on the
        same instance, producing finite and distinct outputs."""
        model = build_mock_model(num_views=7, device=device)
        model.eval()

        visual, map_input, vis_hist, ego = make_inputs(1, 7, device)

        traj_none = model(visual, map_input, vis_hist, ego, mode="infer",
                          camera_params=None)
        cam_params = torch.randn(1, 7, 3, 4, device=device)
        traj_cam = model(visual, map_input, vis_hist, ego, mode="infer",
                         camera_params=cam_params)

        assert torch.isfinite(traj_none).all(), "NaN/Inf with camera_params=None"
        assert torch.isfinite(traj_cam).all(), "NaN/Inf with valid camera_params"
        assert not torch.allclose(traj_none, traj_cam, atol=1e-5), \
            "camera_params None vs valid produced identical outputs"

    def test_batch_size_one_smoke(self, build_mock_model, device):
        """End-to-end forward must work at batch_size=1 with correct shape."""
        model = build_mock_model(num_views=7, device=device)
        model.eval()
        visual, map_input, vis_hist, ego = make_inputs(1, 7, device)
        traj = model(visual, map_input, vis_hist, ego, mode="infer")

        assert traj.shape == (1, 128)
        assert torch.isfinite(traj).all()
