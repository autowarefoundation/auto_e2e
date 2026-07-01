import pytest
import torch
import sys
sys.path.append('..')

from model_components.trajectory_planning import (
    BezierPlanner,
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

# ---------------------------------------------------------------------------
# Planner registry / Flow Matching planner / backcompat
# ---------------------------------------------------------------------------


class TestPlannerRegistry:
    def test_all_modes_registered(self):
        assert "bezier" in PLANNER_REGISTRY
        assert "flow_matching" in PLANNER_REGISTRY

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown planner_mode"):
            build_planner("nonexistent", embed_dim=256)

    def test_build_returns_correct_type(self):
        bezier = build_planner("bezier", embed_dim=256)
        assert isinstance(bezier, BezierPlanner)
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

    def test_inference_forward_returns_trajectory_shape(self, device):
        planner = FlowMatchingPlanner(embed_dim=256).to(device)
        planner.eval()
        bev = torch.randn(2, 256, 8, 8, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)
        traj = planner(bev, vis_hist, ego)
        assert traj.shape == (2, 128)

    def test_inference_output_is_finite(self, device):
        planner = FlowMatchingPlanner(embed_dim=256, num_inference_steps=10).to(device)
        planner.eval()
        bev = torch.randn(1, 256, 8, 8, device=device)
        vis_hist = torch.randn(1, 896, device=device)
        ego = torch.randn(1, 256, device=device)
        traj = planner(bev, vis_hist, ego)
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
        traj = planner(bev, vis_hist, ego)
        assert not torch.allclose(traj, x0, atol=1e-3), \
            "Inference output equals the input noise — ODE did not advance"

    def test_gradient_flows(self, device):
        """Gradients from the integrated trajectory must reach the BEV input
        and the conditioning (visual_history, egomotion)."""
        planner = FlowMatchingPlanner(embed_dim=256).to(device)
        bev = torch.randn(1, 256, 8, 8, device=device, requires_grad=True)
        vis_hist = torch.randn(1, 896, device=device, requires_grad=True)
        ego = torch.randn(1, 256, device=device, requires_grad=True)
        traj = planner(bev, vis_hist, ego)
        traj.pow(2).mean().backward()
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

        traj_a = planner(bev, vis_hist, ego, generator=gen_a)
        traj_b = planner(bev, vis_hist, ego, generator=gen_b)
        traj_c = planner(bev, vis_hist, ego, generator=gen_c)

        assert torch.equal(traj_a, traj_b), \
            "same generator seed must produce identical inference trajectories"
        assert not torch.allclose(traj_a, traj_c), \
            "different generator seeds must produce different trajectories"

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



class TestAutoE2EWithFlowMatching:
    @staticmethod
    def _fm_model(build_mock_model, device):
        return build_mock_model(
            num_views=8, device=device,
            planner_mode="flow_matching",
            planner_kwargs={"num_inference_steps": 4},
        )

    def test_infer_mode_returns_trajectory(self, build_mock_model, device):
        model = self._fm_model(build_mock_model, device)
        model.eval()
        visual, map_input, vis_hist, ego = make_inputs(1, 8, device)
        traj = model(visual, map_input, vis_hist, ego, mode="infer")
        assert traj.shape == (1, 128)
        assert torch.isfinite(traj).all()

    def test_train_mode_also_returns_trajectory(self, build_mock_model, device):
        """Post-refactor the planner forward always returns the trajectory; the
        loss is the training loop's concern, not AutoE2E.forward's."""
        model = self._fm_model(build_mock_model, device)
        model.train()
        visual, map_input, vis_hist, ego = make_inputs(2, 8, device)
        traj = model(visual, map_input, vis_hist, ego, mode="train")
        assert traj.shape == (2, 128)

    def test_backward_flows_through_model(self, build_mock_model, device):
        model = self._fm_model(build_mock_model, device)
        model.train()
        visual, map_input, vis_hist, ego = make_inputs(2, 8, device)
        target = torch.randn(2, 128, device=device)
        traj = model(visual, map_input, vis_hist, ego, mode="train")
        (traj - target).pow(2).mean().backward()
        # At least one Backbone and one TrajectoryPlanner param must see grad.
        backbone_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for n, p in model.named_parameters()
            if n.startswith("Reactive_E2E.Backbone.")
        )
        planner_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for n, p in model.named_parameters()
            if n.startswith("Reactive_E2E.TrajectoryPlanner.")
        )
        assert backbone_grad and planner_grad

    def test_bezier_and_fm_interchangeable_at_inference(self, build_mock_model, device):
        """Bezier and Flow Matching both produce a [B, T*S] trajectory in
        inference mode — callers can swap planners with no other code changes."""
        bezier_model = build_mock_model(num_views=8, device=device,
                                        planner_mode="bezier")
        fm_model = self._fm_model(build_mock_model, device)
        bezier_model.eval()
        fm_model.eval()

        visual, map_input, vis_hist, ego = make_inputs(2, 8, device)
        bezier_traj = bezier_model(visual, map_input, vis_hist, ego, mode="infer")
        fm_traj = fm_model(visual, map_input, vis_hist, ego, mode="infer")
        assert bezier_traj.shape == fm_traj.shape == (2, 128)
