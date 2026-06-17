import pytest
import torch
import sys
sys.path.append('..')

from model_components.trajectory_planning import (
    GRUPlanner,
    FlowMatchingPlanner,
    build_planner,
    PLANNER_REGISTRY,
)


def make_inputs(batch_size, num_views, device, include_camera_params=False):
    visual = torch.randn(batch_size, num_views, 3, 256, 256, device=device)
    map_input = torch.randn(batch_size, 3, 256, 256, device=device)
    visual_history = torch.randn(batch_size, 896, device=device)
    egomotion = torch.randn(batch_size, 256, device=device)
    if include_camera_params:
        camera_params = torch.randn(batch_size, num_views, 3, 4, device=device)
        return visual, map_input, visual_history, egomotion, camera_params
    return visual, map_input, visual_history, egomotion


class TestGRUPlannerComponent:
    def test_output_shapes(self, device):
        planner = GRUPlanner(embed_dim=256).to(device)
        bev = torch.randn(4, 256, 8, 8, device=device)
        vis_hist = torch.randn(4, 896, device=device)
        ego = torch.randn(4, 256, device=device)
        traj, ego_hidden = planner(bev, vis_hist, ego)

        assert traj.shape == (4, 128), "Expected 64 timesteps × 2 signals"
        assert ego_hidden.shape == (4, 256), "ego_hidden must be 256-dim"

    def test_works_with_arbitrary_bev_resolution(self, device):
        """Deformable cross-attention via grid_sample should be size-agnostic."""
        planner = GRUPlanner(embed_dim=256).to(device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)
        for h, w in [(8, 8), (16, 32), (45, 30)]:
            bev = torch.randn(2, 256, h, w, device=device)
            traj, ego_hidden = planner(bev, vis_hist, ego)
            assert traj.shape == (2, 128)
            assert ego_hidden.shape == (2, 256)

    def test_bev_features_influence_trajectory(self, device):
        planner = GRUPlanner(embed_dim=256).to(device)
        planner.eval()
        vis_hist = torch.randn(1, 896, device=device)
        ego = torch.randn(1, 256, device=device)

        bev_a = torch.randn(1, 256, 8, 8, device=device)
        bev_b = torch.randn(1, 256, 8, 8, device=device)

        traj_a, _ = planner(bev_a, vis_hist, ego)
        traj_b, _ = planner(bev_b, vis_hist, ego)

        assert not torch.allclose(traj_a, traj_b, atol=1e-5), \
            "Trajectory should depend on BEV features"

    def test_egomotion_influences_trajectory(self, device):
        planner = GRUPlanner(embed_dim=256).to(device)
        planner.eval()
        bev = torch.randn(1, 256, 8, 8, device=device)
        vis_hist = torch.randn(1, 896, device=device)

        traj_a, _ = planner(bev, vis_hist, torch.randn(1, 256, device=device))
        traj_b, _ = planner(bev, vis_hist, torch.randn(1, 256, device=device))

        assert not torch.allclose(traj_a, traj_b, atol=1e-5), \
            "Trajectory should depend on egomotion history"

    def test_visual_history_influences_trajectory(self, device):
        planner = GRUPlanner(embed_dim=256).to(device)
        planner.eval()
        bev = torch.randn(1, 256, 8, 8, device=device)
        ego = torch.randn(1, 256, device=device)

        traj_a, _ = planner(bev, torch.randn(1, 896, device=device), ego)
        traj_b, _ = planner(bev, torch.randn(1, 896, device=device), ego)

        assert not torch.allclose(traj_a, traj_b, atol=1e-5), \
            "Trajectory should depend on visual history"

    def test_configurable_horizon(self, device):
        planner = GRUPlanner(embed_dim=256, num_timesteps=32, num_signals=3).to(device)
        bev = torch.randn(2, 256, 8, 8, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)
        traj, _ = planner(bev, vis_hist, ego)
        assert traj.shape == (2, 32 * 3)

    def test_gradients_flow(self, device):
        planner = GRUPlanner(embed_dim=256, num_timesteps=4).to(device)
        bev = torch.randn(1, 256, 8, 8, device=device, requires_grad=True)
        vis_hist = torch.randn(1, 896, device=device, requires_grad=True)
        ego = torch.randn(1, 256, device=device, requires_grad=True)
        traj, ego_hidden = planner(bev, vis_hist, ego)
        (traj.sum() + ego_hidden.sum()).backward()
        assert bev.grad is not None and bev.grad.abs().max() > 0
        assert vis_hist.grad is not None and vis_hist.grad.abs().max() > 0
        assert ego.grad is not None and ego.grad.abs().max() > 0

    def test_wrong_visual_history_dim_raises(self, device):
        planner = GRUPlanner(embed_dim=256, visual_history_dim=896).to(device)
        bev = torch.randn(1, 256, 8, 8, device=device)
        bad_vis_hist = torch.randn(1, 1024, device=device)  # wrong last dim
        ego = torch.randn(1, 256, device=device)
        with pytest.raises(ValueError, match="visual_history last dim must be 896"):
            planner(bev, bad_vis_hist, ego)

    def test_wrong_egomotion_dim_raises(self, device):
        planner = GRUPlanner(embed_dim=256, egomotion_dim=256).to(device)
        bev = torch.randn(1, 256, 8, 8, device=device)
        vis_hist = torch.randn(1, 896, device=device)
        bad_ego = torch.randn(1, 128, device=device)  # wrong last dim
        with pytest.raises(ValueError, match="egomotion_history last dim must be 256"):
            planner(bev, vis_hist, bad_ego)

    def test_offset_scale_negative_raises(self):
        with pytest.raises(ValueError, match="offset_scale"):
            GRUPlanner(embed_dim=256, offset_scale=-1.0)

    def test_offset_scale_nan_raises(self):
        with pytest.raises(ValueError, match="offset_scale"):
            GRUPlanner(embed_dim=256, offset_scale=float("nan"))

    def test_offset_scale_inf_raises(self):
        with pytest.raises(ValueError, match="offset_scale"):
            GRUPlanner(embed_dim=256, offset_scale=float("inf"))

    def test_offset_scale_bool_raises(self):
        # bool is an int subclass; the validator must reject it explicitly.
        with pytest.raises(ValueError, match="offset_scale"):
            GRUPlanner(embed_dim=256, offset_scale=True)

    def test_offset_scale_non_numeric_raises(self):
        with pytest.raises(ValueError, match="offset_scale"):
            GRUPlanner(embed_dim=256, offset_scale="0.1")

    def test_offset_scale_zero_vs_nonzero_differ(self, device):
        """offset_scale=0 makes deformable attention sample only at the
        reference point; output must still be valid but differ from the
        nonzero default."""
        torch.manual_seed(0)
        planner_zero = GRUPlanner(embed_dim=256, offset_scale=0.0).to(device)
        torch.manual_seed(0)
        planner_pos = GRUPlanner(embed_dim=256, offset_scale=0.1).to(device)
        planner_zero.eval()
        planner_pos.eval()

        bev = torch.randn(1, 256, 8, 8, device=device)
        vis_hist = torch.randn(1, 896, device=device)
        ego = torch.randn(1, 256, device=device)

        traj_zero, _ = planner_zero(bev, vis_hist, ego)
        traj_pos, _ = planner_pos(bev, vis_hist, ego)

        assert torch.isfinite(traj_zero).all(), \
            "offset_scale=0 (reference-point-only) must still produce finite output"
        assert not torch.allclose(traj_zero, traj_pos, atol=1e-5), \
            "offset_scale=0 and offset_scale=0.1 should produce different trajectories"


# ---------------------------------------------------------------------------
# Planner registry / Flow Matching planner / backcompat
# ---------------------------------------------------------------------------


class TestPlannerRegistry:
    def test_all_modes_registered(self):
        assert "gru" in PLANNER_REGISTRY
        assert "flow_matching" in PLANNER_REGISTRY

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown planner_mode"):
            build_planner("nonexistent", embed_dim=256)

    def test_build_returns_correct_type(self):
        gru = build_planner("gru", embed_dim=256)
        assert isinstance(gru, GRUPlanner)
        fm = build_planner("flow_matching", embed_dim=256)
        assert isinstance(fm, FlowMatchingPlanner)


class TestFlowMatchingPlanner:
    def test_construct_training_data_shapes(self, device):
        planner = FlowMatchingPlanner(embed_dim=256).to(device)
        target = torch.randn(4, 128, device=device)
        u_t, t, target_velocity = planner.construct_training_data(target)
        assert u_t.shape == (4, 128)
        assert t.shape == (4,)
        assert target_velocity.shape == (4, 128)
        assert (t >= 0).all() and (t <= 1).all()

    def test_compute_planner_loss_end_to_end(self, device):
        """The canonical training-loop pattern must work:
        compute_planner_loss returns a scalar loss + ego_hidden, and
        backprop reaches the BEV input."""
        planner = FlowMatchingPlanner(embed_dim=256).to(device)
        planner.train()
        bev = torch.randn(2, 256, 8, 8, device=device, requires_grad=True)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)
        target = torch.randn(2, 128, device=device)

        loss, ego_hidden = planner.compute_planner_loss(
            bev, vis_hist, ego, target,
        )
        assert loss.dim() == 0
        assert ego_hidden.shape == (2, 256)
        assert torch.isfinite(loss)

        loss.backward()
        assert bev.grad is not None and bev.grad.abs().max() > 0

    def test_inference_forward_returns_trajectory_shape(self, device):
        planner = FlowMatchingPlanner(embed_dim=256).to(device)
        planner.eval()
        bev = torch.randn(2, 256, 8, 8, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)
        traj, ego_hidden = planner(bev, vis_hist, ego)
        assert traj.shape == (2, 128)
        assert ego_hidden.shape == (2, 256)

    def test_inference_output_is_finite(self, device):
        planner = FlowMatchingPlanner(embed_dim=256, num_inference_steps=10).to(device)
        planner.eval()
        bev = torch.randn(1, 256, 8, 8, device=device)
        vis_hist = torch.randn(1, 896, device=device)
        ego = torch.randn(1, 256, device=device)
        traj, _ = planner(bev, vis_hist, ego)
        assert torch.isfinite(traj).all()

    def _v_theta(self, planner, bev, vis_hist, ego, u_t, t):
        """Run the velocity network at fixed (u_t, t) — bypasses the
        public API so tests can pin all three inputs."""
        mod_cond = planner._modulation_conditioning(vis_hist, ego)
        bev_seq = planner._project_bev(bev)
        return planner._v_theta(u_t, t, bev_seq, mod_cond)

    def test_velocity_depends_on_bev(self, device):
        torch.manual_seed(0)
        planner = FlowMatchingPlanner(embed_dim=256).to(device)
        planner.eval()
        vis_hist = torch.randn(1, 896, device=device)
        ego = torch.randn(1, 256, device=device)
        u_t = torch.randn(1, 128, device=device)
        t = torch.tensor([0.5], device=device)

        bev_a = torch.randn(1, 256, 8, 8, device=device)
        bev_b = torch.randn(1, 256, 8, 8, device=device)

        v_a = self._v_theta(planner, bev_a, vis_hist, ego, u_t, t)
        v_b = self._v_theta(planner, bev_b, vis_hist, ego, u_t, t)
        assert not torch.allclose(v_a, v_b, atol=1e-5), \
            "v_theta is not sensitive to BEV features"

    def test_velocity_depends_on_timestep(self, device):
        torch.manual_seed(0)
        planner = FlowMatchingPlanner(embed_dim=256).to(device)
        planner.eval()
        bev = torch.randn(1, 256, 8, 8, device=device)
        vis_hist = torch.randn(1, 896, device=device)
        ego = torch.randn(1, 256, device=device)
        u_t = torch.randn(1, 128, device=device)

        v_t1 = self._v_theta(planner, bev, vis_hist, ego, u_t,
                             torch.tensor([0.1], device=device))
        v_t2 = self._v_theta(planner, bev, vis_hist, ego, u_t,
                             torch.tensor([0.9], device=device))
        assert not torch.allclose(v_t1, v_t2, atol=1e-5), \
            "v_theta is not sensitive to flow timestep"

    def test_velocity_depends_on_conditioning(self, device):
        torch.manual_seed(0)
        planner = FlowMatchingPlanner(embed_dim=256).to(device)
        planner.eval()
        bev = torch.randn(1, 256, 8, 8, device=device)
        u_t = torch.randn(1, 128, device=device)
        t = torch.tensor([0.5], device=device)

        v_a = self._v_theta(
            planner, bev,
            torch.randn(1, 896, device=device),
            torch.randn(1, 256, device=device),
            u_t, t,
        )
        v_b = self._v_theta(
            planner, bev,
            torch.randn(1, 896, device=device),
            torch.randn(1, 256, device=device),
            u_t, t,
        )
        assert not torch.allclose(v_a, v_b, atol=1e-5), \
            "v_theta is not sensitive to ego/visual_history conditioning"

    def test_inference_differs_from_noise(self, device):
        """Euler integration must actually transform the noise — output
        cannot match the input noise sample."""
        torch.manual_seed(0)
        planner = FlowMatchingPlanner(embed_dim=256, num_inference_steps=10).to(device)
        planner.eval()
        bev = torch.randn(1, 256, 8, 8, device=device)
        vis_hist = torch.randn(1, 896, device=device)
        ego = torch.randn(1, 256, device=device)

        torch.manual_seed(123)
        x0 = torch.randn(1, 128, device=device)
        torch.manual_seed(123)  # same seed — planner draws an identical x0 inside
        traj, _ = planner(bev, vis_hist, ego)
        assert not torch.allclose(traj, x0, atol=1e-3), \
            "Inference output equals the input noise — ODE did not advance"

    def test_gradient_flows(self, device):
        planner = FlowMatchingPlanner(embed_dim=256).to(device)
        bev = torch.randn(1, 256, 8, 8, device=device, requires_grad=True)
        vis_hist = torch.randn(1, 896, device=device, requires_grad=True)
        ego = torch.randn(1, 256, device=device, requires_grad=True)
        target = torch.randn(1, 128, device=device)
        loss, ego_hidden = planner.compute_planner_loss(
            bev, vis_hist, ego, target,
        )
        (loss + ego_hidden.sum()).backward()
        assert bev.grad is not None and bev.grad.abs().max() > 0
        assert vis_hist.grad is not None and vis_hist.grad.abs().max() > 0
        assert ego.grad is not None and ego.grad.abs().max() > 0

    def test_inference_generator_is_reproducible(self, device):
        """A shared torch.Generator must make inference deterministic across
        runs, and different seeds must produce different trajectories."""
        planner = FlowMatchingPlanner(embed_dim=256, num_inference_steps=10).to(device)
        planner.eval()
        bev = torch.randn(2, 256, 8, 8, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)

        gen_a = torch.Generator(device=device).manual_seed(42)
        gen_b = torch.Generator(device=device).manual_seed(42)
        gen_c = torch.Generator(device=device).manual_seed(7)

        traj_a, _ = planner(bev, vis_hist, ego, generator=gen_a)
        traj_b, _ = planner(bev, vis_hist, ego, generator=gen_b)
        traj_c, _ = planner(bev, vis_hist, ego, generator=gen_c)

        assert torch.equal(traj_a, traj_b), \
            "same generator seed must produce identical inference trajectories"
        assert not torch.allclose(traj_a, traj_c), \
            "different generator seeds must produce different trajectories"

    def test_compute_planner_loss_wrong_target_shape_raises(self, device):
        planner = FlowMatchingPlanner(embed_dim=256).to(device)
        bev = torch.randn(2, 256, 8, 8, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)
        # Wrong batch dim
        bad_target = torch.randn(3, 128, device=device)
        with pytest.raises(ValueError, match="trajectory_target must have shape"):
            planner.compute_planner_loss(bev, vis_hist, ego, bad_target)
        # Wrong feature dim
        bad_target2 = torch.randn(2, 64, device=device)
        with pytest.raises(ValueError, match="trajectory_target must have shape"):
            planner.compute_planner_loss(bev, vis_hist, ego, bad_target2)

    def test_construct_training_data_wrong_target_shape_propagates(self, device):
        """The internal _validate_flow_inputs guard must catch shape regressions
        even when the user only calls construct_training_data."""
        planner = FlowMatchingPlanner(
            embed_dim=256, num_timesteps=4, num_signals=2,
        ).to(device)
        # construct_training_data uses target's shape verbatim; if that shape
        # disagrees with planner.trajectory_dim, the internal validator fires.
        bad_target = torch.randn(2, 16, device=device)  # 16 != 4*2
        with pytest.raises(ValueError, match="noisy_trajectory must have shape"):
            planner.construct_training_data(bad_target)

    def test_validate_flow_inputs_shape_dtype(self, device):
        planner = FlowMatchingPlanner(embed_dim=256).to(device)
        good_u = torch.randn(2, 128, device=device)
        good_t = torch.rand(2, device=device)
        # Wrong u_t shape
        with pytest.raises(ValueError, match="noisy_trajectory must have shape"):
            planner._validate_flow_inputs(torch.randn(2, 64, device=device),
                                          good_t, batch_size=2)
        # Wrong t shape
        with pytest.raises(ValueError, match="flow_timestep must have shape"):
            planner._validate_flow_inputs(good_u,
                                          torch.rand(3, device=device),
                                          batch_size=2)
        # Dtype mismatch
        good_u_f64 = good_u.to(torch.float64)
        with pytest.raises(ValueError, match="must share dtype"):
            planner._validate_flow_inputs(good_u_f64, good_t, batch_size=2)

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="device-mismatch case requires CUDA")
    def test_validate_flow_inputs_device_mismatch(self):
        planner = FlowMatchingPlanner(embed_dim=256)
        u_cpu = torch.randn(1, 128)
        t_cuda = torch.rand(1, device="cuda")
        with pytest.raises(ValueError, match="must be on the same device"):
            planner._validate_flow_inputs(u_cpu, t_cuda, batch_size=1)

    def test_timestep_sampler_beta_in_range(self, device):
        """Default shifted-Beta sampler stays inside [0, beta_scale] and
        matches the trajectory_target dtype."""
        torch.manual_seed(0)
        planner = FlowMatchingPlanner(embed_dim=256).to(device)
        target = torch.randn(256, 128, device=device)
        _, t, _ = planner.construct_training_data(target)
        assert t.shape == (256,)
        assert t.dtype == target.dtype
        assert (t >= 0).all()
        assert (t <= planner.beta_scale + 1e-6).all()

    def test_timestep_sampler_beta_biased_toward_low_t(self, device):
        """Documents the noisy-end bias: shifted Beta(1.5, 1) puts mass
        below 0.5 in expectation. Use a loose bound so the assertion is
        not flaky."""
        torch.manual_seed(0)
        planner = FlowMatchingPlanner(embed_dim=256).to(device)
        target = torch.randn(2048, 128, device=device)
        _, t, _ = planner.construct_training_data(target)
        assert t.mean().item() < 0.5

    def test_timestep_sampler_uniform_option(self, device):
        """timestep_sampler='uniform' recovers U(0, 1)."""
        planner = FlowMatchingPlanner(
            embed_dim=256, timestep_sampler="uniform",
        ).to(device)
        target = torch.randn(256, 128, device=device)
        _, t, _ = planner.construct_training_data(target)
        assert t.shape == (256,)
        assert t.dtype == target.dtype
        assert (t >= 0).all()
        assert (t < 1).all()

    def test_invalid_timestep_sampler_raises(self):
        with pytest.raises(ValueError, match="timestep_sampler"):
            FlowMatchingPlanner(timestep_sampler="gaussian")
        with pytest.raises(ValueError, match="beta_alpha"):
            FlowMatchingPlanner(beta_alpha=0)
        with pytest.raises(ValueError, match="beta_scale"):
            FlowMatchingPlanner(beta_scale=1.5)


class TestGRUPlannerBackcompat:
    """The GRU planner moved into the trajectory_planning subpackage —
    its public behavior must be unchanged."""

    def test_shape_unchanged(self, device):
        planner = GRUPlanner(embed_dim=256).to(device)
        bev = torch.randn(2, 256, 8, 8, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)
        traj, ego_hidden = planner(bev, vis_hist, ego)
        assert traj.shape == (2, 128)
        assert ego_hidden.shape == (2, 256)

    def test_mode_argument_accepted(self, device):
        """forward must accept mode without changing GRU output."""
        torch.manual_seed(0)
        planner = GRUPlanner(embed_dim=256).to(device)
        planner.eval()
        bev = torch.randn(1, 256, 8, 8, device=device)
        vis_hist = torch.randn(1, 896, device=device)
        ego = torch.randn(1, 256, device=device)
        traj_train, _ = planner(bev, vis_hist, ego, mode="train")
        traj_infer, _ = planner(bev, vis_hist, ego, mode="infer")
        assert torch.allclose(traj_train, traj_infer)

    def test_extra_kwargs_ignored(self, device):
        """forward must silently swallow flow-matching-only kwargs."""
        planner = GRUPlanner(embed_dim=256).to(device)
        planner.eval()
        bev = torch.randn(1, 256, 8, 8, device=device)
        vis_hist = torch.randn(1, 896, device=device)
        ego = torch.randn(1, 256, device=device)
        traj, _ = planner(
            bev, vis_hist, ego,
            trajectory_target=torch.randn(1, 128, device=device),
            noisy_trajectory=torch.randn(1, 128, device=device),
            flow_timestep=torch.tensor([0.5], device=device),
        )
        assert traj.shape == (1, 128)


class TestAutoE2EWithFlowMatching:
    @staticmethod
    def _fm_model(build_mock_model, device):
        return build_mock_model(
            num_views=8, fusion_mode="bev", device=device,
            planner_mode="flow_matching",
            planner_kwargs={"num_inference_steps": 4},
        )

    def test_train_mode_returns_scalar_loss(self, build_mock_model, device):
        """Under the uniform Option-B contract, train mode must return a
        scalar planner loss — NOT the raw flow-matching velocity tensor.
        This documents that the velocity-vs-target footgun is gone."""
        model = self._fm_model(build_mock_model, device)
        model.train()
        visual, map_input, vis_hist, ego = make_inputs(2, 8, device)
        target = torch.randn(2, 128, device=device)
        loss, ego_hidden, future = model(
            visual, map_input, vis_hist, ego, mode="train", trajectory_target=target,
        )
        assert loss.dim() == 0, \
            "FM train mode must expose a scalar loss, not a [B, T*S] velocity"
        assert loss.requires_grad
        assert ego_hidden.shape == (2, 256)
        assert future is not None and len(future) == 4

    def test_infer_mode_returns_trajectory(self, build_mock_model, device):
        model = self._fm_model(build_mock_model, device)
        model.eval()
        visual, map_input, vis_hist, ego = make_inputs(1, 8, device)
        traj, ego_hidden, future = model(visual, map_input, vis_hist, ego, mode="infer")
        assert traj.shape == (1, 128)
        assert ego_hidden.shape == (1, 256)
        assert future is None
        assert torch.isfinite(traj).all()

    def test_backward_flows_through_planner_loss(self, build_mock_model, device):
        model = self._fm_model(build_mock_model, device)
        model.train()
        visual, map_input, vis_hist, ego = make_inputs(2, 8, device)
        target = torch.randn(2, 128, device=device)
        loss, ego_hidden, future = model(
            visual, map_input, vis_hist, ego, mode="train", trajectory_target=target,
        )
        total = loss + ego_hidden.sum() + sum(f.sum() for f in future)
        total.backward()
        # At least one Backbone and one TrajectoryPlanner param must see grad.
        backbone_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for n, p in model.named_parameters() if n.startswith("Backbone.")
        )
        planner_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for n, p in model.named_parameters()
            if n.startswith("TrajectoryPlanner.")
        )
        assert backbone_grad and planner_grad

    def test_train_mode_requires_target(self, build_mock_model, device):
        model = self._fm_model(build_mock_model, device)
        visual, map_input, vis_hist, ego = make_inputs(1, 8, device)
        with pytest.raises(ValueError, match="trajectory_target"):
            model(visual, map_input, vis_hist, ego, mode="train")

    def test_gru_and_fm_interchangeable_at_inference(self, build_mock_model, device):
        """GRU and FM both produce a [B, T*S] trajectory in inference mode —
        callers can swap planners with no other code changes."""
        gru_model = build_mock_model(num_views=8, fusion_mode="concat",
                                     device=device)
        fm_model = self._fm_model(build_mock_model, device)
        gru_model.eval()
        fm_model.eval()

        visual, map_input, vis_hist, ego = make_inputs(2, 8, device)
        gru_traj, _, _ = gru_model(visual, map_input, vis_hist, ego, mode="infer")
        fm_traj, _, _ = fm_model(visual, map_input, vis_hist, ego, mode="infer")
        assert gru_traj.shape == fm_traj.shape == (2, 128)
