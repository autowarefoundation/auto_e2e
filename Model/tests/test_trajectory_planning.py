import ast
from pathlib import Path
import warnings

import pytest
import torch
import torch.nn.functional as F

from model_components.trajectory_planning import (
    BasePlanner,
    BezierPlanner,
    FlowMatchingPlanner,
    GRUPlanner,
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

    def test_initial_noise_is_independent_of_rng_state(self, device):
        planner = FlowMatchingPlanner(embed_dim=256, num_inference_steps=4).to(device)
        planner.eval()
        bev = torch.randn(2, 256, 8, 8, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)
        initial_noise = torch.randn(2, 128, device=device)

        torch.manual_seed(1)
        traj_a = planner(
            bev, vis_hist, ego, initial_noise=initial_noise,
        )
        torch.manual_seed(999)
        traj_b = planner(
            bev, vis_hist, ego, initial_noise=initial_noise,
        )

        assert torch.equal(traj_a, traj_b)

    def test_initial_noise_is_batch_order_invariant(self, device):
        planner = FlowMatchingPlanner(embed_dim=256, num_inference_steps=4).to(device)
        planner.eval()
        bev = torch.randn(3, 256, 8, 8, device=device)
        vis_hist = torch.randn(3, 896, device=device)
        ego = torch.randn(3, 256, device=device)
        initial_noise = torch.randn(3, 128, device=device)
        order = torch.tensor([2, 0, 1], device=device)

        original = planner(
            bev, vis_hist, ego, initial_noise=initial_noise,
        )
        reordered = planner(
            bev[order],
            vis_hist[order],
            ego[order],
            initial_noise=initial_noise[order],
        )

        assert torch.allclose(original[order], reordered, atol=1e-6, rtol=1e-6)

    def test_initial_noise_validation(self, device):
        planner = FlowMatchingPlanner(embed_dim=256).to(device)
        bev = torch.randn(2, 256, 8, 8, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)

        with pytest.raises(ValueError, match="initial_noise must have shape"):
            planner(
                bev, vis_hist, ego,
                initial_noise=torch.randn(2, 64, device=device),
            )
        with pytest.raises(ValueError, match="same dtype"):
            planner(
                bev, vis_hist, ego,
                initial_noise=torch.randn(
                    2, 128, device=device, dtype=torch.float64,
                ),
            )

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
        """The planner forward always returns the trajectory; the
        loss is the training loop's concern, not AutoE2E.forward's."""
        model = self._fm_model(build_mock_model, device)
        model.train()
        visual, map_input, vis_hist, ego = make_inputs(2, 8, device)
        traj = model(visual, map_input, vis_hist, ego, mode="train")
        assert traj.shape == (2, 128)

    def test_initial_noise_threads_through_model(self, build_mock_model, device):
        model = self._fm_model(build_mock_model, device)
        model.eval()
        visual, map_input, vis_hist, ego = make_inputs(1, 8, device)
        initial_noise = torch.randn(1, 128, device=device)

        torch.manual_seed(1)
        traj_a = model(
            visual, map_input, vis_hist, ego,
            mode="infer", initial_noise=initial_noise,
        )
        torch.manual_seed(999)
        traj_b = model(
            visual, map_input, vis_hist, ego,
            mode="infer", initial_noise=initial_noise,
        )

        assert torch.equal(traj_a, traj_b)

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


class TestPlannerLossContract:
    def test_base_planner_requires_compute_planner_loss(self):
        class _IncompletePlanner(BasePlanner):
            def forward(
                self, bev_features, visual_history, egomotion_history, **kwargs,
            ):
                return torch.zeros(bev_features.shape[0], 128)

        with pytest.raises(TypeError, match="compute_planner_loss"):
            _IncompletePlanner()

    def test_build_flow_matching_emits_no_runtime_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            planner = build_planner("flow_matching", embed_dim=256)
        assert isinstance(planner, FlowMatchingPlanner)
        runtime_warnings = [
            w for w in caught if issubclass(w.category, RuntimeWarning)
        ]
        assert not runtime_warnings

    def test_flow_matching_compute_planner_loss_skips_euler_and_matches_velocity_mse(
        self, device, monkeypatch,
    ):
        torch.manual_seed(0)
        planner = FlowMatchingPlanner(embed_dim=256, num_inference_steps=4).to(device)
        bev = torch.randn(2, 256, 8, 8, device=device, requires_grad=True)
        vis_hist = torch.randn(2, 896, device=device, requires_grad=True)
        ego = torch.randn(2, 256, device=device, requires_grad=True)
        target = torch.randn(2, 128, device=device)

        def _forbid_forward(*args, **kwargs):
            raise AssertionError("compute_planner_loss must not invoke Euler forward()")

        monkeypatch.setattr(planner, "forward", _forbid_forward)

        u_t = torch.randn(2, 128, device=device)
        t = torch.tensor([0.25, 0.75], device=device)
        target_velocity = torch.randn(2, 128, device=device)
        monkeypatch.setattr(
            planner,
            "construct_training_data",
            lambda _target: (u_t, t, target_velocity),
        )

        out = planner.compute_planner_loss(bev, vis_hist, ego, target)
        assert set(out.keys()) == {"loss", "velocity_mse"}
        assert torch.equal(out["loss"], out["velocity_mse"])

        bev_seq = planner._project_bev(bev)
        mod_cond = planner._modulation_conditioning(vis_hist, ego)
        expected_v = planner._v_theta(u_t, t, bev_seq, mod_cond)
        expected_mse = F.mse_loss(expected_v, target_velocity)
        assert torch.allclose(out["loss"], expected_mse)

        out["loss"].backward()
        assert bev.grad is not None and bev.grad.abs().max() > 0
        assert vis_hist.grad is not None and vis_hist.grad.abs().max() > 0
        assert ego.grad is not None and ego.grad.abs().max() > 0

    def test_bezier_and_gru_compute_planner_loss(self, device):
        torch.manual_seed(0)
        bezier = BezierPlanner(embed_dim=256).to(device)
        bev_pooled = torch.randn(2, 75 * 50, device=device, requires_grad=True)
        vis_hist = torch.randn(2, 896, device=device, requires_grad=True)
        ego = torch.randn(2, 256, device=device, requires_grad=True)
        target = torch.randn(2, 128, device=device)

        expected_traj = bezier(bev_pooled, vis_hist, ego)
        expected_loss = F.smooth_l1_loss(expected_traj, target)
        bezier_out = bezier.compute_planner_loss(bev_pooled, vis_hist, ego, target)
        assert set(bezier_out.keys()) == {"loss"}
        assert torch.allclose(bezier_out["loss"], expected_loss)
        bezier_out["loss"].backward()
        assert bev_pooled.grad is not None and bev_pooled.grad.abs().max() > 0

        gru = GRUPlanner(embed_dim=256).to(device)
        bev_spatial = torch.randn(2, 256, 8, 8, device=device, requires_grad=True)
        vis_hist_g = torch.randn(2, 896, device=device, requires_grad=True)
        ego_g = torch.randn(2, 256, device=device, requires_grad=True)
        custom_loss = torch.nn.MSELoss()
        expected_gru_loss = custom_loss(gru(bev_spatial, vis_hist_g, ego_g), target)
        gru_out = gru.compute_planner_loss(
            bev_spatial, vis_hist_g, ego_g, target, loss_fn=custom_loss,
        )
        assert set(gru_out.keys()) == {"loss"}
        assert torch.allclose(gru_out["loss"], expected_gru_loss)
        gru_out["loss"].backward()
        assert bev_spatial.grad is not None and bev_spatial.grad.abs().max() > 0

    def test_autoe2e_return_planner_loss_for_flow_matching(
        self, build_mock_model, device, monkeypatch,
    ):
        model = build_mock_model(
            num_views=8,
            device=device,
            planner_mode="flow_matching",
            planner_kwargs={"num_inference_steps": 4},
        )
        model.train()
        visual, map_input, vis_hist, ego = make_inputs(2, 8, device)
        target = torch.randn(2, 128, device=device)

        def _forbid_forward(*args, **kwargs):
            raise AssertionError(
                "return_planner_loss=True must not run Euler forward()"
            )

        monkeypatch.setattr(
            model.Reactive_E2E.TrajectoryPlanner, "forward", _forbid_forward,
        )

        _, aux = model(
            visual,
            map_input,
            vis_hist,
            ego,
            mode="train",
            trajectory_target=target,
            return_planner_loss=True,
        )
        assert "planner_loss" in aux
        assert "velocity_mse" in aux["planner_loss"]
        loss = aux["planner_loss"]["loss"]
        assert torch.isfinite(loss)
        loss.backward()

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

    def test_train_il_wires_planner_mode_and_compute_planner_loss(self):
        workflows_path = (
            Path(__file__).resolve().parents[2]
            / "Platform"
            / "pipelines"
            / "workflows.py"
        )
        tree = ast.parse(workflows_path.read_text(encoding="utf-8"))
        train_il_fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_il"
        )
        arg_names = [arg.arg for arg in train_il_fn.args.args]
        assert "planner_mode" in arg_names

        autoe2e_call = next(
            node
            for node in ast.walk(train_il_fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AutoE2E"
        )
        assert "planner_mode" in {kw.arg for kw in autoe2e_call.keywords}

        model_calls = [
            node
            for node in ast.walk(train_il_fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "model"
        ]
        train_call = next(
            call
            for call in model_calls
            if any(kw.arg == "trajectory_target" for kw in call.keywords)
        )
        train_kw = {kw.arg for kw in train_call.keywords}
        assert "return_planner_loss" in train_kw
        assert "planner_loss_fn" in train_kw

    def test_autoe2e_return_planner_loss_for_bezier_reuses_single_rollout(
        self, build_mock_model, device,
    ):
        from model_components.losses import TrajectoryImitationLoss

        model = build_mock_model(
            num_views=8,
            device=device,
            planner_mode="bezier",
        )
        model.train()
        visual, map_input, vis_hist, ego = make_inputs(2, 8, device)
        target = torch.randn(2, 128, device=device)
        loss_fn = TrajectoryImitationLoss(loss_type="smooth_l1").to(device)

        traj, aux = model(
            visual,
            map_input,
            vis_hist,
            ego,
            mode="train",
            trajectory_target=target,
            return_planner_loss=True,
            planner_loss_fn=loss_fn,
        )
        assert traj.shape == (2, 128)
        assert torch.allclose(aux["planner_loss"]["loss"], loss_fn(traj, target))

    @pytest.mark.parametrize("mode", ["pooled_latent", "horizon_cross_attention"])
    def test_flow_matching_compute_planner_loss_threads_reasoning(self, device, mode):
        torch.manual_seed(0)
        planner = FlowMatchingPlanner(
            embed_dim=256, num_inference_steps=3, reasoning_mode=mode,
        ).to(device)
        planner.reasoning_coupling.alpha.data.fill_(1.0)
        planner.reasoning_coupling.reason_proj[-1].weight.data.normal_()

        bev = torch.randn(2, 256, 8, 8, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)
        target = torch.randn(2, 128, device=device)
        latent = torch.randn(2, 256, device=device, requires_grad=True)
        tokens = torch.randn(2, 5, 256, device=device, requires_grad=True)

        out = planner.compute_planner_loss(
            bev,
            vis_hist,
            ego,
            target,
            reasoning_latent=latent,
            reasoning_horizon_tokens=tokens,
        )
        out["loss"].backward()
        if mode == "pooled_latent":
            assert latent.grad is not None and latent.grad.abs().max() > 0
        else:
            assert tokens.grad is not None and tokens.grad.abs().max() > 0
