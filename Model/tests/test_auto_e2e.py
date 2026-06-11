import pytest
import torch
import sys
sys.path.append('..')

from model_components.backbone import Backbone
from model_components.feature_fusion import FeatureFusion
from model_components.trajectory_planner import TrajectoryPlanner
from model_components.future_state import FutureState
from model_components.view_fusion import build_view_fusion, FUSION_REGISTRY
from model_components.view_fusion.cross_attention_fusion import CrossAttentionViewFusion
from model_components.view_fusion.bev_fusion import BEVViewFusion
from model_components.losses import TrajectoryImitationLoss


def make_inputs(batch_size, num_views, device, include_camera_params=False):
    visual = torch.randn(batch_size, num_views, 3, 256, 256, device=device)
    visual_history = torch.randn(batch_size, 896, device=device)
    egomotion = torch.randn(batch_size, 256, device=device)
    if include_camera_params:
        camera_params = torch.randn(batch_size, num_views, 3, 4, device=device)
        return visual, visual_history, egomotion, camera_params
    return visual, visual_history, egomotion


# ---------------------------------------------------------------------------
# 1. Output shape correctness
# ---------------------------------------------------------------------------

class TestOutputShapes:
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_trajectory_shape(self, model, device, batch_size):
        visual, vis_hist, ego = make_inputs(batch_size, 8, device)
        traj, _, _ = model(visual, vis_hist, ego)
        assert traj.shape == (batch_size, 128)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_ego_hidden_shape(self, model, device, batch_size):
        visual, vis_hist, ego = make_inputs(batch_size, 8, device)
        _, ego_hidden, _ = model(visual, vis_hist, ego)
        assert ego_hidden.shape == (batch_size, 256)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_future_features_shape(self, model, device, batch_size):
        visual, vis_hist, ego = make_inputs(batch_size, 8, device)
        _, _, future = model(visual, vis_hist, ego)
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
        visual, vis_hist, ego = make_inputs(2, 8, device)

        # Full batch forward
        traj_both, _, _ = model(visual, vis_hist, ego)

        # Single sample forward (sample 0)
        traj_single, _, _ = model(visual[0:1], vis_hist[0:1], ego[0:1])

        # Sample 0's output must be identical regardless of what sample 1 contains
        assert torch.allclose(traj_both[0], traj_single[0], atol=1e-5), \
            "Batch samples are interfering with each other"

    def test_different_batch_neighbor_no_effect(self, model, device):
        model.eval()
        torch.manual_seed(42)
        visual, vis_hist, ego = make_inputs(2, 8, device)

        traj_a, _, _ = model(visual, vis_hist, ego)

        # Change sample 1 completely
        visual_modified = visual.clone()
        visual_modified[1] = torch.randn_like(visual_modified[1])

        traj_b, _, _ = model(visual_modified, vis_hist, ego)

        # Sample 0 output must remain unchanged
        assert torch.allclose(traj_a[0], traj_b[0], atol=1e-5), \
            "Modifying another sample in the batch affected this sample's output"


# ---------------------------------------------------------------------------
# 3. View fusion effectiveness — different views must influence output
# ---------------------------------------------------------------------------

class TestViewFusion:
    def test_different_views_produce_different_output(self, model, device):
        """Zeroing one camera view must shift the FUSED feature map.

        Asserts at the fused-feature level rather than the trajectory because
        in BEV mode the planner's deformable cross-attention samples only a
        sparse subset of BEV cells — a zeroed view that touches only un-sampled
        cells could leave the trajectory unchanged in a randomly-initialized
        model. View fusion's contract is over the fused feature map; that is
        what this test asserts.
        """
        model.eval()
        torch.manual_seed(42)
        visual, vis_hist, ego = make_inputs(1, 8, device)
        B, V, C, H, W = visual.shape

        def fused_features(x):
            features = model.Backbone(x.reshape(B * V, C, H, W))
            return model.FeatureFusion(features, B, V)

        fused_base = fused_features(visual)

        visual_zeroed = visual.clone()
        visual_zeroed[0, 3] = 0.0
        fused_zeroed = fused_features(visual_zeroed)

        assert not torch.allclose(fused_base, fused_zeroed, atol=1e-5), \
            "Zeroing a camera view had no effect on the fused feature map — fusion is broken"

    def test_all_views_contribute(self, model, device):
        """Each view must influence the FUSED feature map when perturbed.

        View fusion guarantees that every camera view contributes to the
        fused feature map produced by FeatureFusion. Whether the downstream
        planner subsequently samples every fused cell is a separate concern
        — in BEV mode the planner's deformable cross-attention samples only
        a sparse subset of BEV cells, so a view that touches only un-sampled
        cells can legitimately have no measurable trajectory influence in a
        randomly-initialized model. Asserting at the fused-feature level
        directly tests what view fusion is responsible for and works for
        all fusion modes.
        """
        model.eval()
        torch.manual_seed(42)
        visual, vis_hist, ego = make_inputs(1, 8, device)
        B, V, C, H, W = visual.shape

        def fused_features(x):
            features = model.Backbone(x.reshape(B * V, C, H, W))
            return model.FeatureFusion(features, B, V)

        fused_base = fused_features(visual)

        for view_idx in range(V):
            visual_mod = visual.clone()
            visual_mod[0, view_idx] = 5.0
            fused_mod = fused_features(visual_mod)
            assert not torch.allclose(fused_base, fused_mod, atol=1e-5), \
                f"View {view_idx} has no influence on the fused feature map"


# ---------------------------------------------------------------------------
# 4. Gradient flow — all parameters receive gradients
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_backward_succeeds(self, model, device):
        visual, vis_hist, ego = make_inputs(2, 8, device)
        traj, ego_hidden, future = model(visual, vis_hist, ego)

        loss = traj.sum() + ego_hidden.sum() + sum(f.sum() for f in future)
        loss.backward()

    def test_all_parameters_have_gradients(self, model, device):
        visual, vis_hist, ego = make_inputs(2, 8, device)
        traj, ego_hidden, future = model(visual, vis_hist, ego)

        loss = traj.sum() + ego_hidden.sum() + sum(f.sum() for f in future)
        loss.backward()

        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, \
            f"Parameters with no gradient: {params_without_grad}"

    def test_no_vanishing_gradients(self, model, device):
        visual, vis_hist, ego = make_inputs(2, 8, device)
        traj, ego_hidden, future = model(visual, vis_hist, ego)

        loss = traj.sum() + ego_hidden.sum() + sum(f.sum() for f in future)
        loss.backward()

        zero_grad_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().max() == 0:
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
        visual, vis_hist, ego = make_inputs(2, num_views, device)
        traj, ego_hidden, future = model(visual, vis_hist, ego)

        assert traj.shape == (2, 128)
        assert ego_hidden.shape == (2, 256)
        assert all(f.shape == (2, 256, 8, 8) for f in future)


# ---------------------------------------------------------------------------
# 6. Numerical stability — no NaN or Inf
# ---------------------------------------------------------------------------

class TestNumericalStability:
    def test_no_nan_in_outputs(self, model, device):
        visual, vis_hist, ego = make_inputs(2, 8, device)
        traj, ego_hidden, future = model(visual, vis_hist, ego)

        assert not torch.isnan(traj).any(), "NaN in trajectory output"
        assert not torch.isnan(ego_hidden).any(), "NaN in ego_hidden"
        for i, f in enumerate(future):
            assert not torch.isnan(f).any(), f"NaN in future feature {i}"

    def test_no_inf_in_outputs(self, model, device):
        visual, vis_hist, ego = make_inputs(2, 8, device)
        traj, ego_hidden, future = model(visual, vis_hist, ego)

        assert not torch.isinf(traj).any(), "Inf in trajectory output"
        assert not torch.isinf(ego_hidden).any(), "Inf in ego_hidden"
        for i, f in enumerate(future):
            assert not torch.isinf(f).any(), f"Inf in future feature {i}"

    def test_large_input_values(self, model, device):
        """Model should not produce NaN/Inf even with large inputs."""
        visual = torch.randn(1, 8, 3, 256, 256, device=device) * 100
        vis_hist = torch.randn(1, 896, device=device) * 100
        ego = torch.randn(1, 256, device=device) * 100
        traj, ego_hidden, future = model(visual, vis_hist, ego)

        assert not torch.isnan(traj).any(), "NaN with large inputs"
        assert not torch.isinf(traj).any(), "Inf with large inputs"


# ---------------------------------------------------------------------------
# Component-level tests
# ---------------------------------------------------------------------------

class TestFeatureFusionComponent:
    def test_output_shape(self, device):
        fusion = FeatureFusion(num_views=8, fusion_mode="concat").to(device)
        features = [
            torch.randn(16, 96, 64, 64, device=device),
            torch.randn(16, 192, 32, 32, device=device),
            torch.randn(16, 384, 16, 16, device=device),
            torch.randn(16, 768, 8, 8, device=device),
        ]
        out = fusion(features, B=2, V=8)
        assert out.shape == (2, 256, 8, 8)

    def test_view_reduction_changes_output(self, device):
        """Verify that view_reduce is not identity (actually mixes views)."""
        fusion = FeatureFusion(num_views=8, fusion_mode="concat").to(device)
        fusion.eval()

        features_a = [
            torch.randn(8, 96, 64, 64, device=device),
            torch.randn(8, 192, 32, 32, device=device),
            torch.randn(8, 384, 16, 16, device=device),
            torch.randn(8, 768, 8, 8, device=device),
        ]
        out_a = fusion(features_a, B=1, V=8)

        features_b = [f.clone() for f in features_a]
        features_b[0][3] = torch.randn_like(features_b[0][3])
        out_b = fusion(features_b, B=1, V=8)

        assert not torch.allclose(out_a, out_b, atol=1e-5)


class TestTrajectoryPlannerComponent:
    def test_output_shapes(self, device):
        planner = TrajectoryPlanner(embed_dim=256).to(device)
        bev = torch.randn(4, 256, 8, 8, device=device)
        vis_hist = torch.randn(4, 896, device=device)
        ego = torch.randn(4, 256, device=device)
        traj, ego_hidden = planner(bev, vis_hist, ego)

        assert traj.shape == (4, 128), "Expected 64 timesteps × 2 signals"
        assert ego_hidden.shape == (4, 256), "ego_hidden must be 256-dim"

    def test_works_with_arbitrary_bev_resolution(self, device):
        """Deformable cross-attention via grid_sample should be size-agnostic."""
        planner = TrajectoryPlanner(embed_dim=256).to(device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)
        for h, w in [(8, 8), (16, 32), (45, 30)]:
            bev = torch.randn(2, 256, h, w, device=device)
            traj, ego_hidden = planner(bev, vis_hist, ego)
            assert traj.shape == (2, 128)
            assert ego_hidden.shape == (2, 256)

    def test_bev_features_influence_trajectory(self, device):
        planner = TrajectoryPlanner(embed_dim=256).to(device)
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
        planner = TrajectoryPlanner(embed_dim=256).to(device)
        planner.eval()
        bev = torch.randn(1, 256, 8, 8, device=device)
        vis_hist = torch.randn(1, 896, device=device)

        traj_a, _ = planner(bev, vis_hist, torch.randn(1, 256, device=device))
        traj_b, _ = planner(bev, vis_hist, torch.randn(1, 256, device=device))

        assert not torch.allclose(traj_a, traj_b, atol=1e-5), \
            "Trajectory should depend on egomotion history"

    def test_visual_history_influences_trajectory(self, device):
        planner = TrajectoryPlanner(embed_dim=256).to(device)
        planner.eval()
        bev = torch.randn(1, 256, 8, 8, device=device)
        ego = torch.randn(1, 256, device=device)

        traj_a, _ = planner(bev, torch.randn(1, 896, device=device), ego)
        traj_b, _ = planner(bev, torch.randn(1, 896, device=device), ego)

        assert not torch.allclose(traj_a, traj_b, atol=1e-5), \
            "Trajectory should depend on visual history"

    def test_configurable_horizon(self, device):
        planner = TrajectoryPlanner(embed_dim=256, num_timesteps=32, num_signals=3).to(device)
        bev = torch.randn(2, 256, 8, 8, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)
        traj, _ = planner(bev, vis_hist, ego)
        assert traj.shape == (2, 32 * 3)

    def test_gradients_flow(self, device):
        planner = TrajectoryPlanner(embed_dim=256, num_timesteps=4).to(device)
        bev = torch.randn(1, 256, 8, 8, device=device, requires_grad=True)
        vis_hist = torch.randn(1, 896, device=device, requires_grad=True)
        ego = torch.randn(1, 256, device=device, requires_grad=True)
        traj, ego_hidden = planner(bev, vis_hist, ego)
        (traj.sum() + ego_hidden.sum()).backward()
        assert bev.grad is not None and bev.grad.abs().max() > 0
        assert vis_hist.grad is not None and vis_hist.grad.abs().max() > 0
        assert ego.grad is not None and ego.grad.abs().max() > 0

    def test_wrong_visual_history_dim_raises(self, device):
        planner = TrajectoryPlanner(embed_dim=256, visual_history_dim=896).to(device)
        bev = torch.randn(1, 256, 8, 8, device=device)
        bad_vis_hist = torch.randn(1, 1024, device=device)  # wrong last dim
        ego = torch.randn(1, 256, device=device)
        with pytest.raises(ValueError, match="visual_history last dim must be 896"):
            planner(bev, bad_vis_hist, ego)

    def test_wrong_egomotion_dim_raises(self, device):
        planner = TrajectoryPlanner(embed_dim=256, egomotion_dim=256).to(device)
        bev = torch.randn(1, 256, 8, 8, device=device)
        vis_hist = torch.randn(1, 896, device=device)
        bad_ego = torch.randn(1, 128, device=device)  # wrong last dim
        with pytest.raises(ValueError, match="egomotion_history last dim must be 256"):
            planner(bev, vis_hist, bad_ego)

    def test_offset_scale_negative_raises(self):
        with pytest.raises(ValueError, match="offset_scale"):
            TrajectoryPlanner(embed_dim=256, offset_scale=-1.0)

    def test_offset_scale_nan_raises(self):
        with pytest.raises(ValueError, match="offset_scale"):
            TrajectoryPlanner(embed_dim=256, offset_scale=float("nan"))

    def test_offset_scale_inf_raises(self):
        with pytest.raises(ValueError, match="offset_scale"):
            TrajectoryPlanner(embed_dim=256, offset_scale=float("inf"))

    def test_offset_scale_zero_vs_nonzero_differ(self, device):
        """offset_scale=0 makes deformable attention sample only at the
        reference point; output must still be valid but differ from the
        nonzero default."""
        torch.manual_seed(0)
        planner_zero = TrajectoryPlanner(embed_dim=256, offset_scale=0.0).to(device)
        torch.manual_seed(0)
        planner_pos = TrajectoryPlanner(embed_dim=256, offset_scale=0.1).to(device)
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


class TestFutureStateComponent:
    def test_accepts_ego_hidden(self, device):
        future = FutureState(embed_dim=256, ego_hidden_dim=256).to(device)
        feats = torch.randn(2, 256, 8, 8, device=device)
        ego_hidden = torch.randn(2, 256, device=device)
        out = future(feats, ego_hidden)
        assert len(out) == 4
        for f in out:
            assert f.shape == (2, 256, 8, 8)

    def test_ego_hidden_influences_output(self, device):
        future = FutureState(embed_dim=256, ego_hidden_dim=256).to(device)
        future.eval()
        feats = torch.randn(1, 256, 8, 8, device=device)

        out_a = future(feats, torch.randn(1, 256, device=device))
        out_b = future(feats, torch.randn(1, 256, device=device))

        assert not torch.allclose(out_a[0], out_b[0], atol=1e-5), \
            "ego_hidden should influence future predictions"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestFusionRegistry:
    def test_all_modes_registered(self):
        assert "concat" in FUSION_REGISTRY
        assert "cross_attn" in FUSION_REGISTRY
        assert "bev" in FUSION_REGISTRY

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown fusion_mode"):
            build_view_fusion("nonexistent", num_views=8)

    @pytest.mark.parametrize("fusion_mode", list(FUSION_REGISTRY.keys()))
    def test_all_modes_produce_correct_shape(self, device, fusion_mode):
        view_fusion_kwargs = {"bev_h": 8, "bev_w": 8} if fusion_mode == "bev" else {}
        fusion = FeatureFusion(
            num_views=8, fusion_mode=fusion_mode,
            view_fusion_kwargs=view_fusion_kwargs,
        ).to(device)
        features = [
            torch.randn(16, 96, 64, 64, device=device),
            torch.randn(16, 192, 32, 32, device=device),
            torch.randn(16, 384, 16, 16, device=device),
            torch.randn(16, 768, 8, 8, device=device),
        ]
        out = fusion(features, B=2, V=8)
        assert out.shape == (2, 256, 8, 8)


# ---------------------------------------------------------------------------
# Cross-Attention specific tests
# ---------------------------------------------------------------------------

class TestCrossAttentionFusion:
    def test_output_shape(self, device):
        fusion = CrossAttentionViewFusion(num_views=8, embed_dim=256).to(device)
        x = torch.randn(16, 256, 7, 7, device=device)
        out = fusion(x, B=2, V=8)
        assert out.shape == (2, 256, 7, 7)

    def test_view_embeddings_are_learned(self, device):
        """View embeddings should receive gradients during training."""
        fusion = CrossAttentionViewFusion(num_views=8, embed_dim=256).to(device)
        x = torch.randn(8, 256, 7, 7, device=device)
        out = fusion(x, B=1, V=8)
        out.sum().backward()
        assert fusion.view_embed.grad is not None
        assert fusion.view_embed.grad.abs().max() > 0

    def test_attention_mixes_views(self, device):
        """Attention should produce different output than simple mean pooling."""
        fusion = CrossAttentionViewFusion(num_views=4, embed_dim=256).to(device)
        fusion.eval()
        x = torch.randn(4, 256, 7, 7, device=device)

        attn_out = fusion(x, B=1, V=4)
        mean_out = x.reshape(1, 4, 256, 7, 7).mean(dim=1)

        assert not torch.allclose(attn_out, mean_out, atol=1e-3), \
            "Cross-attention output is identical to naive mean — attention has no effect"

    def test_different_view_orders_produce_different_output(self, device):
        """Attention with positional embeddings should be order-sensitive."""
        fusion = CrossAttentionViewFusion(num_views=4, embed_dim=256).to(device)
        fusion.eval()

        x = torch.randn(4, 256, 7, 7, device=device)
        out_original = fusion(x, B=1, V=4)

        x_permuted = x[[2, 0, 3, 1]]
        out_permuted = fusion(x_permuted, B=1, V=4)

        assert not torch.allclose(out_original, out_permuted, atol=1e-5), \
            "View position embeddings have no effect — output is order-invariant"


# ---------------------------------------------------------------------------
# BEV Fusion specific tests
# ---------------------------------------------------------------------------

class TestBEVFusion:
    def test_output_shape(self, device):
        fusion = BEVViewFusion(num_views=8, embed_dim=256, bev_h=8, bev_w=8).to(device)
        x = torch.randn(16, 256, 8, 8, device=device)
        out = fusion(x, B=2, V=8)
        assert out.shape == (2, 256, 8, 8)

    def test_default_resolution_is_450x300(self):
        """Production target: 450x300 BEV grid with front-biased pc_range."""
        fusion = BEVViewFusion(num_views=8, embed_dim=256)
        assert fusion.bev_h == 450
        assert fusion.bev_w == 300
        assert fusion.pc_range == (-60.0, -60.0, -5.0, 120.0, 60.0, 3.0)

    def test_asymmetric_resolution(self, device):
        """Configurable bev_h != bev_w yields a non-square BEV grid."""
        fusion = BEVViewFusion(num_views=4, embed_dim=256, bev_h=12, bev_w=20).to(device)
        x = torch.randn(4, 256, 8, 8, device=device)
        out = fusion(x, B=1, V=4)
        assert out.shape == (1, 256, 12, 20)

    def test_output_shape_with_camera_params(self, device):
        """BEV fusion should work with explicit camera projection matrices."""
        fusion = BEVViewFusion(num_views=8, embed_dim=256, bev_h=8, bev_w=8).to(device)
        x = torch.randn(16, 256, 8, 8, device=device)
        cam_params = torch.randn(2, 8, 3, 4, device=device)
        out = fusion(x, B=2, V=8, camera_params=cam_params)
        assert out.shape == (2, 256, 8, 8)

    def test_pseudo_projection_is_learned(self, device):
        """Without camera_params, pseudo_projection should receive gradients."""
        fusion = BEVViewFusion(num_views=4, embed_dim=256, bev_h=8, bev_w=8).to(device)
        x = torch.randn(4, 256, 8, 8, device=device)
        out = fusion(x, B=1, V=4)
        out.sum().backward()
        assert fusion.pseudo_projection.grad is not None
        assert fusion.pseudo_projection.grad.abs().max() > 0

    def test_bev_queries_are_learned(self, device):
        """BEV queries should receive gradients during training."""
        fusion = BEVViewFusion(num_views=4, embed_dim=256, bev_h=8, bev_w=8).to(device)
        x = torch.randn(4, 256, 8, 8, device=device)
        out = fusion(x, B=1, V=4)
        out.sum().backward()
        assert fusion.bev_queries.weight.grad is not None
        assert fusion.bev_queries.weight.grad.abs().max() > 0

    def test_camera_params_influence_output(self, device):
        """Different camera parameters should produce different BEV features."""
        fusion = BEVViewFusion(num_views=4, embed_dim=256, bev_h=8, bev_w=8).to(device)
        fusion.eval()
        x = torch.randn(4, 256, 8, 8, device=device)

        cam_a = torch.randn(1, 4, 3, 4, device=device)
        cam_b = torch.randn(1, 4, 3, 4, device=device)

        out_a = fusion(x, B=1, V=4, camera_params=cam_a)
        out_b = fusion(x, B=1, V=4, camera_params=cam_b)

        assert not torch.allclose(out_a, out_b, atol=1e-5), \
            "Different camera params produced identical output — projection has no effect"

    def test_reference_points_shape(self, device):
        """3D reference points should have expected shape."""
        fusion = BEVViewFusion(num_views=8, embed_dim=256, bev_h=7, bev_w=7,
                               num_points_in_pillar=4).to(device)
        assert fusion.reference_points_3d.shape == (49, 4, 3)

    def test_no_nan_without_camera_params(self, device):
        """BEV fusion with pseudo-projection should not produce NaN."""
        fusion = BEVViewFusion(num_views=8, embed_dim=256, bev_h=8, bev_w=8).to(device)
        x = torch.randn(16, 256, 8, 8, device=device)
        out = fusion(x, B=2, V=8)
        assert not torch.isnan(out).any(), "NaN in BEV output with pseudo-projection"

    def test_points_behind_camera_are_masked(self, device):
        """Points with negative depth should not contribute to output."""
        fusion = BEVViewFusion(num_views=1, embed_dim=256, bev_h=8, bev_w=8,
                               image_size=224,
                               pc_range=(-10, -10, -5, 10, 10, 3)).to(device)

        # Camera matrix that makes all projected depths negative:
        # z_proj = row2 @ [x, y, z, 1]^T
        # Set row2 = [0, 0, -1, -100] so z_proj = -z_world - 100 (always negative
        # since z_world ranges from -5 to 3 in this pc_range)
        cam = torch.zeros(1, 1, 3, 4, device=device)
        cam[0, 0, 0, 0] = 224.0   # fx (irrelevant since depth is negative)
        cam[0, 0, 1, 1] = 224.0   # fy
        cam[0, 0, 2, 2] = -1.0    # negate z
        cam[0, 0, 2, 3] = -100.0  # large negative offset ensures all depths < 0

        ref_2d, mask = fusion._project_to_2d(fusion.reference_points_3d, cam)

        # All points should be masked (behind camera)
        assert not mask.any(), \
            "Points behind camera (negative depth) should all be masked"

    def test_projected_center_maps_near_image_center(self, device):
        """A simple projection should map BEV center to image center."""
        fusion = BEVViewFusion(num_views=1, embed_dim=256, bev_h=7, bev_w=7,
                               image_size=224, pc_range=(-1, -1, 0.5, 1, 1, 2)).to(device)

        # Camera: fx=fy=112, cx=cy=112 (image center), z passthrough
        # BEV center (x=0, y=0) at any z > 0 projects to:
        #   u = fx*0/z + cx = 112, v = fy*0/z + cy = 112
        #   normalized: u/224 = 0.5, v/224 = 0.5
        cam = torch.zeros(1, 1, 3, 4, device=device)
        cam[0, 0, 0, 0] = 112.0   # fx
        cam[0, 0, 0, 2] = 112.0   # cx
        cam[0, 0, 1, 1] = 112.0   # fy
        cam[0, 0, 1, 2] = 112.0   # cy
        cam[0, 0, 2, 2] = 1.0     # z passthrough

        ref_2d, mask = fusion._project_to_2d(fusion.reference_points_3d, cam)
        # ref_2d: [1, 1, 49, num_z, 2]

        # BEV center is query index 24 (7×7 grid, row 3 col 3)
        center_2d = ref_2d[0, 0, 24, :, :]  # [num_z, 2]
        center_mask = mask[0, 0, 24, :]      # [num_z]

        # At least some pillar points should be valid
        assert center_mask.any(), "Center point should have valid projections"

        # Valid points should project exactly to (0.5, 0.5) since x=y=0
        valid_points = center_2d[center_mask]  # [num_valid, 2]
        expected = torch.tensor([0.5, 0.5], device=device)
        assert torch.allclose(valid_points[0], expected, atol=0.01), \
            f"BEV center should project to image center (0.5, 0.5), got {valid_points[0]}"

    def test_out_of_bounds_points_not_counted_visible(self, device):
        """When all reference points project out of image bounds, output should be zero."""
        fusion = BEVViewFusion(num_views=1, embed_dim=256, bev_h=8, bev_w=8,
                               image_size=224,
                               pc_range=(-10, -10, -5, 10, 10, 3)).to(device)
        fusion.eval()

        # Camera that projects everything to far-right of image (u >> image_size)
        # u = fx * x / z + cx, with fx=1000 and cx=5000, u/224 >> 1 for all points
        cam = torch.zeros(1, 1, 3, 4, device=device)
        cam[0, 0, 0, 0] = 1000.0  # fx (very large)
        cam[0, 0, 0, 2] = 5000.0  # cx (way off image)
        cam[0, 0, 1, 1] = 1000.0  # fy
        cam[0, 0, 1, 2] = 5000.0  # cy (way off image)
        cam[0, 0, 2, 2] = 1.0     # z passthrough (positive depth)

        x = torch.ones(1, 256, 8, 8, device=device)
        out = fusion(x, B=1, V=1, camera_params=cam)

        # ref_2d normalized = (fx*x/z + cx) / 224 >> 1, so all out of bounds
        # → mask = False everywhere → visible_count = 0 → has_observation = 0
        assert out.abs().max() < 1e-6, \
            "Out-of-bounds projections should produce zero output"

    def test_offset_scale_zero_vs_nonzero_differ(self, device):
        """offset_scale=0 disables fan-out; output must differ from a nonzero
        scale at the same seed."""
        torch.manual_seed(0)
        fusion_zero = BEVViewFusion(num_views=4, embed_dim=256, bev_h=8, bev_w=8,
                                    offset_scale=0.0).to(device)
        torch.manual_seed(0)
        fusion_pos = BEVViewFusion(num_views=4, embed_dim=256, bev_h=8, bev_w=8,
                                   offset_scale=0.1).to(device)
        fusion_zero.eval()
        fusion_pos.eval()
        x = torch.randn(4, 256, 8, 8, device=device)
        out_zero = fusion_zero(x, B=1, V=4)
        out_pos = fusion_pos(x, B=1, V=4)
        assert not torch.allclose(out_zero, out_pos, atol=1e-5), \
            "offset_scale=0 and offset_scale=0.1 produced identical BEV output"

    def test_offset_scale_negative_raises(self):
        with pytest.raises(ValueError, match="offset_scale"):
            BEVViewFusion(num_views=4, embed_dim=256, bev_h=8, bev_w=8,
                          offset_scale=-1.0)

    def test_offset_scale_nan_raises(self):
        with pytest.raises(ValueError, match="offset_scale"):
            BEVViewFusion(num_views=4, embed_dim=256, bev_h=8, bev_w=8,
                          offset_scale=float("nan"))

    def test_offset_scale_inf_raises(self):
        with pytest.raises(ValueError, match="offset_scale"):
            BEVViewFusion(num_views=4, embed_dim=256, bev_h=8, bev_w=8,
                          offset_scale=float("inf"))

    def test_no_visible_camera_produces_zero_output(self, device):
        """If no camera can see any BEV cell, output should be exactly zero."""
        fusion = BEVViewFusion(num_views=1, embed_dim=256, bev_h=8, bev_w=8,
                               image_size=224,
                               pc_range=(-10, -10, -5, 10, 10, 3)).to(device)
        fusion.eval()

        x = torch.ones(1, 256, 8, 8, device=device)

        # Camera that places everything behind (negative depth)
        cam_behind = torch.zeros(1, 1, 3, 4, device=device)
        cam_behind[0, 0, 2, 2] = -1.0
        cam_behind[0, 0, 2, 3] = -100.0
        out = fusion(x, B=1, V=1, camera_params=cam_behind)

        # has_observation mask zeroes output after FFN
        assert out.abs().max() < 1e-6, \
            "No visible camera should produce zero BEV features"


# ---------------------------------------------------------------------------
# Integration tests — full backbone (slow, marked for separate CI tier)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFullBackboneIntegration:
    """End-to-end tests with the real pretrained backbone.

    These verify that the full pipeline (backbone → fusion → planner → future)
    produces correct shapes and numerically stable outputs. Run separately
    from unit tests via: pytest -m integration
    """

    def test_full_forward_pass(self, full_model, device):
        """Smoke test: full model forward produces expected output shapes."""
        visual, vis_hist, ego = make_inputs(1, 8, device)
        traj, ego_hidden, future = full_model(visual, vis_hist, ego)

        assert traj.shape == (1, 128)
        assert ego_hidden.shape == (1, 256)
        assert len(future) == 4
        for f in future:
            assert f.shape == (1, 256, 8, 8)

    def test_full_forward_no_nan(self, full_model, device):
        """Full pipeline must not produce NaN with real backbone weights."""
        visual, vis_hist, ego = make_inputs(2, 8, device)
        traj, ego_hidden, future = full_model(visual, vis_hist, ego)

        assert not torch.isnan(traj).any()
        assert not torch.isnan(ego_hidden).any()
        for f in future:
            assert not torch.isnan(f).any()


@pytest.mark.integration
class TestResNet50Backbone:
    """Exercises the dynamic backbone_channels computation on a backbone
    whose feature_info shape differs from Swin (5 stages of channels
    64/256/512/1024/2048 vs Swin's 4 stages of 96/192/384/768)."""

    def test_resnet50_forward_pass(self, device):
        from model_components.auto_e2e import AutoE2E
        try:
            model = AutoE2E(
                backbone="res_net_50", num_views=8, fusion_mode="concat",
                is_pretrained=False,
            ).to(device)
        except (FileNotFoundError, OSError) as e:
            pytest.skip(f"Backbone construction failed: {e}")

        # Dynamic backbone_channels = sum of all 5 ResNet50 stages = 3904
        assert model.Backbone.backbone_channels == 64 + 256 + 512 + 1024 + 2048

        visual, vis_hist, ego = make_inputs(1, 8, device)
        traj, ego_hidden, future = model(visual, vis_hist, ego)

        assert traj.shape == (1, 128)
        assert ego_hidden.shape == (1, 256)
        assert len(future) == 4
        for f in future:
            assert f.shape == (1, 256, 8, 8)
        assert torch.isfinite(traj).all()
        assert torch.isfinite(ego_hidden).all()


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
        model = build_mock_model(num_views=8, fusion_mode="concat", device=device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

        before = {n: p.detach().clone() for n, p in model.named_parameters()
                  if p.requires_grad}

        visual, vis_hist, ego = make_inputs(2, 8, device)
        traj, ego_hidden, future = model(visual, vis_hist, ego)
        loss = traj.sum() + ego_hidden.sum() + sum(f.sum() for f in future)

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
        model = build_mock_model(num_views=8, fusion_mode="concat", device=device)
        model.train()
        loss_fn = TrajectoryImitationLoss(num_timesteps=64, num_signals=2).to(device)

        visual, vis_hist, ego = make_inputs(2, 8, device)
        traj, _, _ = model(visual, vis_hist, ego)

        target = torch.randn_like(traj)
        loss = loss_fn(traj, target)
        loss.backward()

        assert torch.isfinite(loss), "Loss is non-finite"
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
# Behavioral tests on planner / future-state internals
# ---------------------------------------------------------------------------

class TestTrajectoryDynamics:
    def test_trajectory_does_not_saturate(self, device):
        """The GRU should produce time-varying outputs, not saturate to a constant.

        Reshape the 64×2 trajectory and check that the second half of the
        first signal channel is NOT a single repeated value.
        """
        torch.manual_seed(0)
        planner = TrajectoryPlanner(embed_dim=256).to(device)
        planner.eval()
        bev = torch.randn(1, 256, 8, 8, device=device)
        vis_hist = torch.randn(1, 896, device=device)
        ego = torch.randn(1, 256, device=device)

        traj, _ = planner(bev, vis_hist, ego)
        # traj: [1, 128] = [1, 64*2]; reshape so dim 1 = timesteps
        late = traj.view(1, 64, 2)[0, 32:, 0]
        # Require substantial variation across the 32 late timesteps. A bare
        # >1 threshold would pass even when 31 of 32 values collapse to the
        # same constant — near-saturation we still want to catch.
        assert late.unique().numel() >= 8, \
            "Trajectory saturates to a constant value over the last 32 timesteps"

    def test_deformable_clamp_handles_extreme_query(self, device):
        """Extreme query magnitudes push sampling locations outside [0, 1];
        the clamp inside _deformable_cross_attn must keep output finite AND
        actively engage. Verified by:
          1. With offset_scale=0.1, an extreme query produces sample_locs
             that fall outside [0, 1] BEFORE the clamp — so the clamp is
             the only thing keeping the result finite.
          2. Output is finite at offset_scale=0.1 with the extreme query.
          3. With offset_scale=0 (offsets disabled), the same extreme query
             still produces finite output via the reference point alone.
        """
        torch.manual_seed(0)
        planner_pos = TrajectoryPlanner(embed_dim=256, offset_scale=0.1).to(device)
        torch.manual_seed(0)
        planner_zero = TrajectoryPlanner(embed_dim=256, offset_scale=0.0).to(device)
        planner_pos.eval()
        planner_zero.eval()

        # Build extreme query and value tensors directly to exercise the clamp.
        B, C, H, W = 1, 256, 8, 8
        query = torch.full((B, C), 1e6, device=device)
        values = torch.randn(B, C, H, W, device=device)

        # Sanity: with offsets active, sample_locs WITHOUT clamp would land
        # outside [0, 1] for this query, so the clamp is the only reason
        # the output remains finite.
        ref = planner_pos.reference_point(query).sigmoid()
        offs = (planner_pos.sampling_offsets(query)
                .reshape(B, planner_pos.num_points, 2) * planner_pos.offset_scale)
        unclamped = ref.unsqueeze(1) + offs
        outside = ((unclamped < 0) | (unclamped > 1)).any()
        assert outside, \
            "Test setup failed: extreme query did not push sample_locs out of bounds"

        out_pos = planner_pos._deformable_cross_attn(query, values)
        out_zero = planner_zero._deformable_cross_attn(query, values)

        assert torch.isfinite(out_pos).all(), \
            "Clamp failed: output contains NaN/Inf for extreme query (offset_scale=0.1)"
        assert torch.isfinite(out_zero).all(), \
            "Reference-point-only path produced NaN/Inf for extreme query"

    def test_offset_scale_zero_vs_nonzero_differ_in_attn(self, device):
        """With normal-magnitude queries, offset_scale=0 vs 0.1 must produce
        different deformable attention outputs — proving offsets actually
        contribute (and don't just get squashed by the clamp). Uses the same
        seed for the two planners so the only meaningful difference is whether
        offsets fan out around the reference point."""
        torch.manual_seed(0)
        planner_pos = TrajectoryPlanner(embed_dim=256, offset_scale=0.1).to(device)
        torch.manual_seed(0)
        planner_zero = TrajectoryPlanner(embed_dim=256, offset_scale=0.0).to(device)
        planner_pos.eval()
        planner_zero.eval()

        B, C, H, W = 1, 256, 8, 8
        query = torch.randn(B, C, device=device)
        values = torch.randn(B, C, H, W, device=device)

        out_pos = planner_pos._deformable_cross_attn(query, values)
        out_zero = planner_zero._deformable_cross_attn(query, values)

        assert torch.isfinite(out_pos).all() and torch.isfinite(out_zero).all()
        assert not torch.allclose(out_pos, out_zero, atol=1e-5), \
            "offset_scale=0 vs 0.1 produced identical attn output — offsets inactive"


class TestFutureStateChunkSplit:
    def test_four_outputs_are_distinct(self, device):
        """torch.chunk must split along channels, not return 4 views of the same data."""
        torch.manual_seed(0)
        future = FutureState(embed_dim=256, ego_hidden_dim=256).to(device)
        future.eval()
        feats = torch.randn(2, 256, 8, 8, device=device)
        ego_hidden = torch.randn(2, 256, device=device)

        out = future(feats, ego_hidden)
        assert len(out) == 4
        for i in range(4):
            for j in range(i + 1, 4):
                assert not torch.allclose(out[i], out[j], atol=1e-5), \
                    f"FutureState outputs {i} and {j} are identical — chunk is broken"

    def test_ego_hidden_changes_all_four_outputs(self, device):
        """Different ego_hidden must shift every one of the 4 future predictions."""
        torch.manual_seed(0)
        future = FutureState(embed_dim=256, ego_hidden_dim=256).to(device)
        future.eval()
        feats = torch.randn(1, 256, 8, 8, device=device)

        ego_a = torch.randn(1, 256, device=device)
        ego_b = torch.randn(1, 256, device=device)

        out_a = future(feats, ego_a)
        out_b = future(feats, ego_b)

        for i in range(4):
            assert not torch.allclose(out_a[i], out_b[i], atol=1e-5), \
                f"Future output {i} did not change when ego_hidden changed"


class TestVisualHistoryNonZeroDifference:
    def test_two_nonzero_visual_histories_differ(self, device):
        """Both visual_history inputs are non-zero and distinct — outputs must differ."""
        torch.manual_seed(0)
        planner = TrajectoryPlanner(embed_dim=256).to(device)
        planner.eval()
        bev = torch.randn(1, 256, 8, 8, device=device)
        ego = torch.randn(1, 256, device=device)

        vh_a = torch.randn(1, 896, device=device)
        vh_b = torch.randn(1, 896, device=device)
        # Sanity: both are non-zero
        assert vh_a.abs().max() > 0 and vh_b.abs().max() > 0
        assert not torch.allclose(vh_a, vh_b)

        traj_a, _ = planner(bev, vh_a, ego)
        traj_b, _ = planner(bev, vh_b, ego)

        assert not torch.allclose(traj_a, traj_b, atol=1e-5), \
            "Two distinct non-zero visual_history inputs produced the same trajectory"


# ---------------------------------------------------------------------------
# Full-pipeline robustness
# ---------------------------------------------------------------------------

class TestFullPipelineRobustness:
    def test_all_zero_inputs_produce_finite_outputs(self, build_mock_model, device):
        """Zero inputs across all paths must not cause NaN/Inf anywhere downstream."""
        model = build_mock_model(num_views=8, fusion_mode="concat", device=device)
        model.eval()

        visual = torch.zeros(2, 8, 3, 256, 256, device=device)
        vis_hist = torch.zeros(2, 896, device=device)
        ego = torch.zeros(2, 256, device=device)

        traj, ego_hidden, future = model(visual, vis_hist, ego)

        assert torch.isfinite(traj).all(), "NaN/Inf in trajectory with zero inputs"
        assert torch.isfinite(ego_hidden).all(), "NaN/Inf in ego_hidden with zero inputs"
        for i, f in enumerate(future):
            assert torch.isfinite(f).all(), f"NaN/Inf in future feature {i} with zero inputs"

    def test_camera_params_none_then_valid_switching(self, build_mock_model, device):
        """A BEV-fusion model must accept both None and valid camera_params on the
        same instance, producing finite and distinct outputs."""
        model = build_mock_model(num_views=8, fusion_mode="bev", device=device)
        model.eval()

        visual, vis_hist, ego = make_inputs(1, 8, device, include_camera_params=False)

        traj_none, _, _ = model(visual, vis_hist, ego, camera_params=None)
        cam_params = torch.randn(1, 8, 3, 4, device=device)
        traj_cam, _, _ = model(visual, vis_hist, ego, camera_params=cam_params)

        assert torch.isfinite(traj_none).all(), "NaN/Inf with camera_params=None"
        assert torch.isfinite(traj_cam).all(), "NaN/Inf with valid camera_params"
        assert not torch.allclose(traj_none, traj_cam, atol=1e-5), \
            "camera_params None vs valid produced identical outputs — projection has no effect"

    def test_batch_size_one_smoke(self, build_mock_model, device):
        """End-to-end forward must work at batch_size=1 with correct shapes and no NaN."""
        model = build_mock_model(num_views=8, fusion_mode="concat", device=device)
        model.eval()
        visual, vis_hist, ego = make_inputs(1, 8, device)
        traj, ego_hidden, future = model(visual, vis_hist, ego)

        assert traj.shape == (1, 128)
        assert ego_hidden.shape == (1, 256)
        assert len(future) == 4
        for f in future:
            assert f.shape == (1, 256, 8, 8)
        assert torch.isfinite(traj).all()
        assert torch.isfinite(ego_hidden).all()
        for f in future:
            assert torch.isfinite(f).all()


class _StubBackboneWithFeatureInfo(torch.nn.Module):
    """Channels-first backbone exposing timm-style feature_info."""

    def __init__(self):
        super().__init__()
        self.stage0 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.stage1 = torch.nn.Conv2d(32, 48, 3, stride=2, padding=1)
        self.stage2 = torch.nn.Conv2d(48, 64, 3, stride=2, padding=1)
        self.feature_info = [{"num_chs": 32}, {"num_chs": 48}, {"num_chs": 64}]

    def forward(self, x):
        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        return [s0, s1, s2]


class _StubBackboneNoFeatureInfo(torch.nn.Module):
    """Channels-first backbone with NO feature_info (probe fallback path)."""

    def __init__(self):
        super().__init__()
        self.stage0 = torch.nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.stage1 = torch.nn.Conv2d(24, 56, 3, stride=2, padding=1)
        self.stage2 = torch.nn.Conv2d(56, 112, 3, stride=2, padding=1)

    def forward(self, x):
        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        return [s0, s1, s2]


class _StubBackboneSwinLike(torch.nn.Module):
    """Channels-last backbone (B, H, W, C) — exercises permute branch."""

    def __init__(self):
        super().__init__()
        self.stage0 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.stage1 = torch.nn.Conv2d(32, 48, 3, stride=2, padding=1)
        self.feature_info = [{"num_chs": 32}, {"num_chs": 48}]

    def forward(self, x):
        s0_cf = self.stage0(x)                                  # [B, 32, H, W]
        s1_cf = self.stage1(s0_cf)                              # [B, 48, H, W]
        s0 = s0_cf.permute(0, 2, 3, 1).contiguous()             # [B, H, W, 32]
        s1 = s1_cf.permute(0, 2, 3, 1).contiguous()             # [B, H, W, 48]
        return [s0, s1]


class TestBackboneChannelDiscovery:
    """Cover the backbone_channels discovery + layout-detection in Backbone."""

    def _make_backbone(self, monkeypatch, stub_module):
        # Patch the registry call so build_backbone returns our stub.
        monkeypatch.setattr(
            "model_components.backbone.build_backbone",
            lambda *a, **kw: stub_module,
        )
        return Backbone(backbone="stub", is_pretrained=False)

    def test_feature_info_path_sums_channels(self, monkeypatch):
        bb = self._make_backbone(monkeypatch, _StubBackboneWithFeatureInfo())
        assert bb.backbone_channels == 32 + 48 + 64

    def test_probe_fallback_when_feature_info_missing(self, monkeypatch):
        bb = self._make_backbone(monkeypatch, _StubBackboneNoFeatureInfo())
        # No feature_info — channels recovered via probing.
        assert bb.backbone_channels == 24 + 56 + 112

    def test_feature_info_channels_match_forward_output(self, monkeypatch, device):
        """sum(feature_info channels) must equal the actual concat-channel dim
        of the forward output."""
        bb = self._make_backbone(monkeypatch, _StubBackboneWithFeatureInfo()).to(device)
        x = torch.randn(2, 3, 32, 32, device=device)
        feats = bb(x)
        total_c = sum(f.shape[1] for f in feats)
        assert total_c == bb.backbone_channels

    def test_probe_channels_match_forward_output(self, monkeypatch, device):
        bb = self._make_backbone(monkeypatch, _StubBackboneNoFeatureInfo()).to(device)
        x = torch.randn(2, 3, 32, 32, device=device)
        feats = bb(x)
        total_c = sum(f.shape[1] for f in feats)
        assert total_c == bb.backbone_channels

    def test_channels_last_backbone_is_permuted(self, monkeypatch, device):
        """Channels-last (B, H, W, C) output must be permuted to (B, C, H, W)
        based on tensor shape, NOT on the backbone name."""
        bb = self._make_backbone(monkeypatch, _StubBackboneSwinLike()).to(device)
        x = torch.randn(2, 3, 32, 32, device=device)
        feats = bb(x)
        # After Backbone.forward, every feature must be channels-first with the
        # expected channel count at dim 1.
        assert feats[0].shape[1] == 32
        assert feats[1].shape[1] == 48

    def test_channels_first_backbone_not_permuted(self, monkeypatch, device):
        bb = self._make_backbone(monkeypatch, _StubBackboneWithFeatureInfo()).to(device)
        x = torch.randn(2, 3, 32, 32, device=device)
        feats = bb(x)
        for f, expected in zip(feats, [32, 48, 64]):
            assert f.shape[1] == expected


class TestFeatureFusionWithSwinChannels:
    def test_dynamic_backbone_channels_with_swin_sizes(self, device):
        """FeatureFusion should accept Swin's per-stage channels (96, 192, 384, 768)
        at their natural spatial resolutions and produce the expected fused shape."""
        backbone_channels = 96 + 192 + 384 + 768  # 1440
        fusion = FeatureFusion(
            num_views=8, backbone_channels=backbone_channels, fusion_mode="concat",
        ).to(device)

        # Per-stage Swin spatial dims for a 256x256 input
        features = [
            torch.randn(16, 96, 64, 64, device=device),
            torch.randn(16, 192, 32, 32, device=device),
            torch.randn(16, 384, 16, 16, device=device),
            torch.randn(16, 768, 8, 8, device=device),
        ]
        out = fusion(features, B=2, V=8)
        assert out.shape == (2, 256, 8, 8)
        assert torch.isfinite(out).all()
