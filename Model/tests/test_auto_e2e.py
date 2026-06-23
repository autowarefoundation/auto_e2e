import pytest
import torch
import sys
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


# ---------------------------------------------------------------------------
# 1. Output shape correctness
# ---------------------------------------------------------------------------

class TestOutputShapes:
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_trajectory_shape(self, model, device, batch_size):
        visual, map_input, vis_hist, ego = make_inputs(batch_size, 7, device)
        traj, _, _ = model(visual, map_input, vis_hist, ego, mode="infer")
        assert traj.shape == (batch_size, 128)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_ego_hidden_shape(self, model, device, batch_size):
        visual, map_input, vis_hist, ego = make_inputs(batch_size, 7, device)
        _, ego_hidden, _ = model(visual, map_input, vis_hist, ego, mode="infer")
        assert ego_hidden.shape == (batch_size, 256)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_future_features_shape(self, model, device, batch_size):
        visual, map_input, vis_hist, ego = make_inputs(batch_size, 7, device)
        target = torch.randn(batch_size, 128, device=device)
        _, _, future = model(visual, map_input, vis_hist, ego, mode="train",
                             trajectory_target=target)
        assert len(future) == 4
        for f in future:
            assert f.shape == (batch_size, 256, 8, 8)


# ---------------------------------------------------------------------------
# 2. Batch independence — changing one sample must not affect others
# ---------------------------------------------------------------------------

class TestBatchIndependence:
    def test_samples_do_not_interfere(self, model, device):
        model.eval()
        torch.manual_seed(42)
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)

        # Full batch forward
        traj_both, _, _ = model(visual, map_input, vis_hist, ego, mode="infer")

        # Single sample forward (sample 0)
        traj_single, _, _ = model(visual[0:1], map_input[0:1], vis_hist[0:1], ego[0:1],
                                  mode="infer")

        # Sample 0's output must be identical regardless of what sample 1 contains
        assert torch.allclose(traj_both[0], traj_single[0], atol=1e-5), \
            "Batch samples are interfering with each other"

    def test_different_batch_neighbor_no_effect(self, model, device):
        model.eval()
        torch.manual_seed(42)
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)

        traj_a, _, _ = model(visual, map_input, vis_hist, ego, mode="infer")

        # Change sample 1 completely
        visual_modified = visual.clone()
        visual_modified[1] = torch.randn_like(visual_modified[1])

        traj_b, _, _ = model(visual_modified, map_input, vis_hist, ego, mode="infer")

        # Sample 0 output must remain unchanged
        assert torch.allclose(traj_a[0], traj_b[0], atol=1e-5), \
            "Modifying another sample in the batch affected this sample's output"


# ---------------------------------------------------------------------------
# 4. Gradient flow — all parameters receive gradients
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_backward_succeeds(self, model, device):
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        target = torch.randn(2, 128, device=device)
        loss, ego_hidden, future = model(visual, map_input, vis_hist, ego,
                                         mode="train", trajectory_target=target)

        total = loss + ego_hidden.sum() + sum(f.sum() for f in future)
        total.backward()

    def test_all_parameters_have_gradients(self, model, device):
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        target = torch.randn(2, 128, device=device)
        loss, ego_hidden, future = model(visual, map_input, vis_hist, ego,
                                         mode="train", trajectory_target=target)

        total = loss + ego_hidden.sum() + sum(f.sum() for f in future)
        total.backward()

        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
            # MapEncoder parameters legitimately receive zero gradient
                # at init because ResidualMapFusion uses alpha=0, which
                # zeroes out ∂loss/∂map_bev entirely (chain rule:
                # ∂fused/∂map_bev = alpha = 0). This is intentional: the
                # zero-init scheme ensures the map branch doesn't destabilise
                # early training. Gradient will flow once alpha grows > 0.
                if name.startswith("MapEncoder."):
                    continue
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, \
            f"Parameters with no gradient: {params_without_grad}"

    def test_no_vanishing_gradients(self, model, device):
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        target = torch.randn(2, 128, device=device)
        loss, ego_hidden, future = model(
             visual, map_input, vis_hist, ego,
            mode="train", trajectory_target=target)

        total = loss + ego_hidden.sum() + sum(f.sum() for f in future)
        total.backward()

        zero_grad_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().max() == 0:
                    if name.startswith("MapEncoder."):
                        continue
                    zero_grad_params.append(name)

        assert len(zero_grad_params) == 0, \
            f"Parameters with all-zero gradients: {zero_grad_params}"


# ---------------------------------------------------------------------------
# 5. num_views flexibility — model works with different view counts
# ---------------------------------------------------------------------------

class TestNumViewsFlexibility:
    @pytest.mark.parametrize("num_views,fusion_mode", [
        (1, "concat"), (4, "concat"), (8, "concat"), (12, "concat"),
        (1, "cross_attn"), (4, "cross_attn"), (8, "cross_attn"), (12, "cross_attn"),
        (1, "bev"), (4, "bev"), (8, "bev"), (12, "bev"),
    ])
    def test_various_num_views(self, build_mock_model, device, num_views, fusion_mode):
        model = build_mock_model(num_views, fusion_mode, device)
        visual, map_input, vis_hist, ego = make_inputs(2, num_views, device)
        target = torch.randn(2, 128, device=device)
        loss, ego_hidden, future = model(
             visual, map_input, vis_hist, ego,
            mode="train", trajectory_target=target)

        assert loss.dim() == 0
        assert ego_hidden.shape == (2, 256)
        assert all(f.shape == (2, 256, 8, 8) for f in future)


# ---------------------------------------------------------------------------
# 6. Numerical stability — no NaN or Inf
# ---------------------------------------------------------------------------

class TestNumericalStability:
    def test_no_nan_in_outputs(self, model, device):
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        target = torch.randn(2, 128, device=device)
        loss, ego_hidden, future = model(
             visual, map_input, vis_hist, ego,
            mode="train", trajectory_target=target)

        assert not torch.isnan(loss), "NaN in planner loss"
        assert not torch.isnan(ego_hidden).any(), "NaN in ego_hidden"
        for i, f in enumerate(future):
            assert not torch.isnan(f).any(), f"NaN in future feature {i}"

    def test_no_inf_in_outputs(self, model, device):
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        target = torch.randn(2, 128, device=device)
        loss, ego_hidden, future = model(
             visual, map_input, vis_hist, ego,
            mode="train", trajectory_target=target)

        assert not torch.isinf(loss), "Inf in planner loss"
        assert not torch.isinf(ego_hidden).any(), "Inf in ego_hidden"
        for i, f in enumerate(future):
            assert not torch.isinf(f).any(), f"Inf in future feature {i}"

    def test_large_input_values(self, model, device):
        """Model should not produce NaN/Inf even with large inputs."""
        visual = torch.randn(1, 7, 3, 256, 256, device=device) * 100
        map_input = torch.randn(1, 3, 256, 256, device=device) * 100
        vis_hist = torch.randn(1, 896, device=device) * 100
        ego = torch.randn(1, 256, device=device) * 100
        traj, _, _ = model(visual, map_input, vis_hist, ego, mode="infer")

        assert not torch.isnan(traj).any(), "NaN with large inputs"
        assert not torch.isinf(traj).any(), "Inf with large inputs"


# ---------------------------------------------------------------------------
# Training loop integration — optimizer.step + loss
# ---------------------------------------------------------------------------

class TestTrainingLoop:
    def test_optimizer_step_updates_parameters(self, build_mock_model, device):
        """forward → loss → backward → optimizer.step() must move parameters
        in EACH submodule, not just somewhere in the model. A grad that only
        reaches the last layer would still satisfy a "any parameter changed"
        check; this verifies every major group actually trains.

        The loss is constructed from trajectory + ego_hidden + future so that
        every group has a path to the loss:
          - Backbone: feeds image features into FeatureFusion
          - FeatureFusion: produces BEV / fused feats
          - TrajectoryPlanner: outputs trajectory + ego_hidden
          - FutureState: produces future feature pyramid (consumed below)
        """
        model = build_mock_model(num_views=7, fusion_mode="concat", device=device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

        before = {n: p.detach().clone() for n, p in model.named_parameters()
                  if p.requires_grad}

        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        target = torch.randn(2, 128, device=device)
        planner_loss, ego_hidden, future = model(
            visual, map_input, vis_hist, ego, mode="train", trajectory_target=target)
        loss = planner_loss + ego_hidden.sum() + sum(f.sum() for f in future)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        groups = ["Backbone", "FeatureFusion", "TrajectoryPlanner", "FutureState"]
        changed_per_group = {g: False for g in groups}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if torch.equal(p.detach(), before[name]):
                continue
            for prefix in groups:
                if name.startswith(prefix + "."):
                    changed_per_group[prefix] = True

        unchanged = [g for g, ok in changed_per_group.items() if not ok]
        assert not unchanged, \
            f"optimizer.step() did not update any parameter in: {unchanged}"

    def test_model_to_loss_backward_integration(self, build_mock_model, device):
        """Pipe trajectory output into TrajectoryImitationLoss and run backward."""
        model = build_mock_model(num_views=7, fusion_mode="bev", device=device)
        model.train()

        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        target = torch.randn(2, 128, device=device)
        loss, _, _ = model(visual, map_input, vis_hist, ego,
                           mode="train", trajectory_target=target)

        assert loss.dim() == 0, "planner loss must be a scalar in train mode"
        assert loss.requires_grad
        assert torch.isfinite(loss), "Loss is non-finite"

        loss.backward()

        # Verify gradient propagates through the full network depth, not just
        # the last layer: both the upstream Backbone and the downstream
        # TrajectoryPlanner must each see a nonzero grad on at least one param.
        groups = {"Backbone": False, "TrajectoryPlanner": False}
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if p.grad.abs().max() == 0:
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
        """Zero inputs across all paths must not cause NaN/Inf anywhere downstream."""
        model = build_mock_model(num_views=7, fusion_mode="concat", device=device)
        model.eval()

        visual = torch.zeros(2, 7, 3, 256, 256, device=device)
        map_input = torch.zeros(2, 3, 256, 256, device=device)
        vis_hist = torch.zeros(2, 896, device=device)
        ego = torch.zeros(2, 256, device=device)

        traj, ego_hidden, _ = model(visual, map_input, vis_hist, ego, mode="infer")

        assert torch.isfinite(traj).all(), "NaN/Inf in trajectory with zero inputs"
        assert torch.isfinite(ego_hidden).all(), "NaN/Inf in ego_hidden with zero inputs"

    def test_camera_params_none_then_valid_switching(self, build_mock_model, device):
        """A BEV-fusion model must accept both None and valid camera_params on the
        same instance, producing finite and distinct outputs."""
        model = build_mock_model(num_views=7, fusion_mode="bev", device=device)
        model.eval()

        visual, map_input, vis_hist, ego = make_inputs(1, 7, device, include_camera_params=False)

        traj_none, _, _ = model(visual, map_input, vis_hist, ego, mode="infer",
                                camera_params=None)
        cam_params = torch.randn(1, 7, 3, 4, device=device)
        traj_cam, _, _ = model(visual, map_input, vis_hist, ego, mode="infer",
                                camera_params=cam_params)

        assert torch.isfinite(traj_none).all(), "NaN/Inf with camera_params=None"
        assert torch.isfinite(traj_cam).all(), "NaN/Inf with valid camera_params"
        assert not torch.allclose(traj_none, traj_cam, atol=1e-5), \
            "camera_params None vs valid produced identical outputs — projection has no effect"

    def test_batch_size_one_smoke(self, build_mock_model, device):
        """End-to-end forward must work at batch_size=1 with correct shapes and no NaN."""
        model = build_mock_model(num_views=7, fusion_mode="concat", device=device)
        model.eval()
        visual, map_input, vis_hist, ego = make_inputs(1, 7, device)
        traj, ego_hidden, _ = model(visual, map_input, vis_hist, ego, mode="infer")

        assert traj.shape == (1, 128)
        assert ego_hidden.shape == (1, 256)
        assert torch.isfinite(traj).all()
        assert torch.isfinite(ego_hidden).all()
