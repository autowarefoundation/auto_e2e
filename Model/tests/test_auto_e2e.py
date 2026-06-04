import pytest
import torch
import sys
sys.path.append('..')

from model_components.auto_e2e import AutoE2E
from model_components.backbone import Backbone
from model_components.feature_fusion import FeatureFusion
from model_components.driving_policy import DrivingPolicy
from model_components.future_state import FutureState
from model_components.view_fusion import build_view_fusion, FUSION_REGISTRY
from model_components.view_fusion.cross_attention_fusion import CrossAttentionViewFusion
from model_components.view_fusion.bev_fusion import BEVViewFusion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(params=["concat", "cross_attn", "bev"])
def model(request, device):
    m = AutoE2E(num_views=8, fusion_mode=request.param)
    return m.to(device)


def make_inputs(batch_size, num_views, device, include_camera_params=False):
    visual = torch.randn(batch_size, num_views, 3, 224, 224, device=device)
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
        visual, vh, ego = make_inputs(batch_size, 8, device)
        traj, _, _ = model(visual, vh, ego)
        assert traj.shape == (batch_size, 128)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_compressed_visual_shape(self, model, device, batch_size):
        visual, vh, ego = make_inputs(batch_size, 8, device)
        _, compressed, _ = model(visual, vh, ego)
        assert compressed.shape == (batch_size, 14)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_future_features_shape(self, model, device, batch_size):
        visual, vh, ego = make_inputs(batch_size, 8, device)
        _, _, future = model(visual, vh, ego)
        assert len(future) == 4
        for f in future:
            assert f.shape == (batch_size, 1440, 7, 7)


# ---------------------------------------------------------------------------
# 2. Batch independence — changing one sample must not affect others
# ---------------------------------------------------------------------------

class TestBatchIndependence:
    def test_samples_do_not_interfere(self, model, device):
        model.eval()
        torch.manual_seed(42)
        visual, vh, ego = make_inputs(2, 8, device)

        # Full batch forward
        traj_both, _, _ = model(visual, vh, ego)

        # Single sample forward (sample 0)
        traj_single, _, _ = model(
            visual[0:1], vh[0:1], ego[0:1]
        )

        # Sample 0's output must be identical regardless of what sample 1 contains
        assert torch.allclose(traj_both[0], traj_single[0], atol=1e-5), \
            "Batch samples are interfering with each other"

    def test_different_batch_neighbor_no_effect(self, model, device):
        model.eval()
        torch.manual_seed(42)
        visual, vh, ego = make_inputs(2, 8, device)

        traj_a, _, _ = model(visual, vh, ego)

        # Change sample 1 completely
        visual_modified = visual.clone()
        visual_modified[1] = torch.randn_like(visual_modified[1])
        vh_modified = vh.clone()
        vh_modified[1] = torch.randn_like(vh_modified[1])

        traj_b, _, _ = model(visual_modified, vh_modified, ego)

        # Sample 0 output must remain unchanged
        assert torch.allclose(traj_a[0], traj_b[0], atol=1e-5), \
            "Modifying another sample in the batch affected this sample's output"


# ---------------------------------------------------------------------------
# 3. View fusion effectiveness — different views must influence output
# ---------------------------------------------------------------------------

class TestViewFusion:
    def test_different_views_produce_different_output(self, model, device):
        model.eval()
        torch.manual_seed(42)
        visual, vh, ego = make_inputs(1, 8, device)

        traj_a, _, _ = model(visual, vh, ego)

        # Replace one camera view with zeros
        visual_zeroed = visual.clone()
        visual_zeroed[0, 3] = 0.0

        traj_b, _, _ = model(visual_zeroed, vh, ego)

        # Output should differ — proving that the zeroed view had influence
        assert not torch.allclose(traj_a, traj_b, atol=1e-5), \
            "Changing a camera view had no effect on output — fusion is broken"

    def test_all_views_contribute(self, model, device):
        """Each view should influence the output when zeroed out."""
        model.eval()
        torch.manual_seed(42)
        visual, vh, ego = make_inputs(1, 8, device)

        traj_base, _, _ = model(visual, vh, ego)

        for view_idx in range(8):
            visual_mod = visual.clone()
            visual_mod[0, view_idx] = 0.0
            traj_mod, _, _ = model(visual_mod, vh, ego)
            assert not torch.allclose(traj_base, traj_mod, atol=1e-5), \
                f"View {view_idx} has no influence on the output"


# ---------------------------------------------------------------------------
# 4. Gradient flow — all parameters receive gradients
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_backward_succeeds(self, model, device):
        visual, vh, ego = make_inputs(2, 8, device)
        traj, compressed, future = model(visual, vh, ego)

        loss = traj.sum() + compressed.sum() + sum(f.sum() for f in future)
        loss.backward()

    def test_all_parameters_have_gradients(self, model, device):
        visual, vh, ego = make_inputs(2, 8, device)
        traj, compressed, future = model(visual, vh, ego)

        loss = traj.sum() + compressed.sum() + sum(f.sum() for f in future)
        loss.backward()

        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, \
            f"Parameters with no gradient: {params_without_grad}"

    def test_no_vanishing_gradients(self, model, device):
        visual, vh, ego = make_inputs(2, 8, device)
        traj, compressed, future = model(visual, vh, ego)

        loss = traj.sum() + compressed.sum() + sum(f.sum() for f in future)
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
    def test_various_num_views(self, device, num_views, fusion_mode):
        model = AutoE2E(num_views=num_views, fusion_mode=fusion_mode).to(device)
        visual, vh, ego = make_inputs(2, num_views, device)
        traj, compressed, future = model(visual, vh, ego)

        assert traj.shape == (2, 128)
        assert compressed.shape == (2, 14)
        assert all(f.shape == (2, 1440, 7, 7) for f in future)


# ---------------------------------------------------------------------------
# 6. Numerical stability — no NaN or Inf
# ---------------------------------------------------------------------------

class TestNumericalStability:
    def test_no_nan_in_outputs(self, model, device):
        visual, vh, ego = make_inputs(2, 8, device)
        traj, compressed, future = model(visual, vh, ego)

        assert not torch.isnan(traj).any(), "NaN in trajectory output"
        assert not torch.isnan(compressed).any(), "NaN in compressed visual"
        for i, f in enumerate(future):
            assert not torch.isnan(f).any(), f"NaN in future feature {i}"

    def test_no_inf_in_outputs(self, model, device):
        visual, vh, ego = make_inputs(2, 8, device)
        traj, compressed, future = model(visual, vh, ego)

        assert not torch.isinf(traj).any(), "Inf in trajectory output"
        assert not torch.isinf(compressed).any(), "Inf in compressed visual"
        for i, f in enumerate(future):
            assert not torch.isinf(f).any(), f"Inf in future feature {i}"

    def test_large_input_values(self, model, device):
        """Model should not produce NaN/Inf even with large inputs."""
        visual = torch.randn(1, 8, 3, 224, 224, device=device) * 100
        vh = torch.randn(1, 896, device=device) * 100
        ego = torch.randn(1, 256, device=device) * 100
        traj, compressed, future = model(visual, vh, ego)

        assert not torch.isnan(traj).any(), "NaN with large inputs"
        assert not torch.isinf(traj).any(), "Inf with large inputs"


# ---------------------------------------------------------------------------
# Component-level tests
# ---------------------------------------------------------------------------

class TestFeatureFusionComponent:
    def test_output_shape(self, device):
        fusion = FeatureFusion(num_views=8, fusion_mode="concat").to(device)
        features = [
            torch.randn(16, 56, 56, 96, device=device),
            torch.randn(16, 28, 28, 192, device=device),
            torch.randn(16, 14, 14, 384, device=device),
            torch.randn(16, 7, 7, 768, device=device),
        ]
        out = fusion(features, B=2, V=8)
        assert out.shape == (2, 1440, 7, 7)

    def test_view_reduction_changes_output(self, device):
        """Verify that view_reduce is not identity (actually mixes views)."""
        fusion = FeatureFusion(num_views=8, fusion_mode="concat").to(device)
        fusion.eval()

        features_a = [
            torch.randn(8, 56, 56, 96, device=device),
            torch.randn(8, 28, 28, 192, device=device),
            torch.randn(8, 14, 14, 384, device=device),
            torch.randn(8, 7, 7, 768, device=device),
        ]
        out_a = fusion(features_a, B=1, V=8)

        features_b = [f.clone() for f in features_a]
        features_b[0][3] = torch.randn_like(features_b[0][3])
        out_b = fusion(features_b, B=1, V=8)

        assert not torch.allclose(out_a, out_b, atol=1e-5)


class TestDrivingPolicyComponent:
    def test_flatten_preserves_batch(self, device):
        """The critical fix: flatten must NOT collapse batch dimension."""
        policy = DrivingPolicy().to(device)
        fused = torch.randn(4, 1440, 7, 7, device=device)
        vh = torch.randn(4, 896, device=device)
        ego = torch.randn(4, 256, device=device)
        traj, compressed = policy(fused, vh, ego)

        assert traj.shape[0] == 4, "Batch dimension lost in DrivingPolicy"
        assert compressed.shape[0] == 4, "Batch dimension lost in compression"


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
        fusion = FeatureFusion(num_views=8, fusion_mode=fusion_mode).to(device)
        features = [
            torch.randn(16, 56, 56, 96, device=device),
            torch.randn(16, 28, 28, 192, device=device),
            torch.randn(16, 14, 14, 384, device=device),
            torch.randn(16, 7, 7, 768, device=device),
        ]
        out = fusion(features, B=2, V=8)
        assert out.shape == (2, 1440, 7, 7)


# ---------------------------------------------------------------------------
# Cross-Attention specific tests
# ---------------------------------------------------------------------------

class TestCrossAttentionFusion:
    def test_output_shape(self, device):
        fusion = CrossAttentionViewFusion(num_views=8, embed_dim=1440).to(device)
        x = torch.randn(16, 1440, 7, 7, device=device)
        out = fusion(x, B=2, V=8)
        assert out.shape == (2, 1440, 7, 7)

    def test_view_embeddings_are_learned(self, device):
        """View embeddings should receive gradients during training."""
        fusion = CrossAttentionViewFusion(num_views=8, embed_dim=1440).to(device)
        x = torch.randn(8, 1440, 7, 7, device=device)
        out = fusion(x, B=1, V=8)
        out.sum().backward()
        assert fusion.view_embed.grad is not None
        assert fusion.view_embed.grad.abs().max() > 0

    def test_attention_mixes_views(self, device):
        """Attention should produce different output than simple mean pooling."""
        fusion = CrossAttentionViewFusion(num_views=4, embed_dim=1440).to(device)
        fusion.eval()
        x = torch.randn(4, 1440, 7, 7, device=device)

        attn_out = fusion(x, B=1, V=4)
        mean_out = x.reshape(1, 4, 1440, 7, 7).mean(dim=1)

        assert not torch.allclose(attn_out, mean_out, atol=1e-3), \
            "Cross-attention output is identical to naive mean — attention has no effect"

    def test_different_view_orders_produce_different_output(self, device):
        """Attention with positional embeddings should be order-sensitive."""
        fusion = CrossAttentionViewFusion(num_views=4, embed_dim=1440).to(device)
        fusion.eval()

        x = torch.randn(4, 1440, 7, 7, device=device)
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
        fusion = BEVViewFusion(num_views=8, embed_dim=1440).to(device)
        x = torch.randn(16, 1440, 7, 7, device=device)
        out = fusion(x, B=2, V=8)
        assert out.shape == (2, 1440, 7, 7)

    def test_output_shape_with_camera_params(self, device):
        """BEV fusion should work with explicit camera projection matrices."""
        fusion = BEVViewFusion(num_views=8, embed_dim=1440).to(device)
        x = torch.randn(16, 1440, 7, 7, device=device)
        cam_params = torch.randn(2, 8, 3, 4, device=device)
        out = fusion(x, B=2, V=8, camera_params=cam_params)
        assert out.shape == (2, 1440, 7, 7)

    def test_pseudo_projection_is_learned(self, device):
        """Without camera_params, pseudo_projection should receive gradients."""
        fusion = BEVViewFusion(num_views=4, embed_dim=1440).to(device)
        x = torch.randn(4, 1440, 7, 7, device=device)
        out = fusion(x, B=1, V=4)
        out.sum().backward()
        assert fusion.pseudo_projection.grad is not None
        assert fusion.pseudo_projection.grad.abs().max() > 0

    def test_bev_queries_are_learned(self, device):
        """BEV queries should receive gradients during training."""
        fusion = BEVViewFusion(num_views=4, embed_dim=1440).to(device)
        x = torch.randn(4, 1440, 7, 7, device=device)
        out = fusion(x, B=1, V=4)
        out.sum().backward()
        assert fusion.bev_queries.weight.grad is not None
        assert fusion.bev_queries.weight.grad.abs().max() > 0

    def test_camera_params_influence_output(self, device):
        """Different camera parameters should produce different BEV features."""
        fusion = BEVViewFusion(num_views=4, embed_dim=1440).to(device)
        fusion.eval()
        x = torch.randn(4, 1440, 7, 7, device=device)

        cam_a = torch.randn(1, 4, 3, 4, device=device)
        cam_b = torch.randn(1, 4, 3, 4, device=device)

        out_a = fusion(x, B=1, V=4, camera_params=cam_a)
        out_b = fusion(x, B=1, V=4, camera_params=cam_b)

        assert not torch.allclose(out_a, out_b, atol=1e-5), \
            "Different camera params produced identical output — projection has no effect"

    def test_reference_points_shape(self, device):
        """3D reference points should have expected shape."""
        fusion = BEVViewFusion(num_views=8, embed_dim=1440, bev_h=7, bev_w=7,
                               num_points_in_pillar=4).to(device)
        assert fusion.reference_points_3d.shape == (49, 4, 3)

    def test_no_nan_without_camera_params(self, device):
        """BEV fusion with pseudo-projection should not produce NaN."""
        fusion = BEVViewFusion(num_views=8, embed_dim=1440).to(device)
        x = torch.randn(16, 1440, 7, 7, device=device)
        out = fusion(x, B=2, V=8)
        assert not torch.isnan(out).any(), "NaN in BEV output with pseudo-projection"

    def test_points_behind_camera_are_masked(self, device):
        """Points with negative depth should not contribute to output."""
        fusion = BEVViewFusion(num_views=1, embed_dim=1440, image_size=224,
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
        fusion = BEVViewFusion(num_views=1, embed_dim=1440, bev_h=7, bev_w=7,
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
        fusion = BEVViewFusion(num_views=1, embed_dim=1440, image_size=224,
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

        x = torch.ones(1, 1440, 7, 7, device=device)
        out = fusion(x, B=1, V=1, camera_params=cam)

        # ref_2d normalized = (fx*x/z + cx) / 224 >> 1, so all out of bounds
        # → mask = False everywhere → visible_count = 0 → has_observation = 0
        assert out.abs().max() < 1e-6, \
            "Out-of-bounds projections should produce zero output"

    def test_no_visible_camera_produces_zero_output(self, device):
        """If no camera can see any BEV cell, output should be exactly zero."""
        fusion = BEVViewFusion(num_views=1, embed_dim=1440, image_size=224,
                               pc_range=(-10, -10, -5, 10, 10, 3)).to(device)
        fusion.eval()

        x = torch.ones(1, 1440, 7, 7, device=device)

        # Camera that places everything behind (negative depth)
        cam_behind = torch.zeros(1, 1, 3, 4, device=device)
        cam_behind[0, 0, 2, 2] = -1.0
        cam_behind[0, 0, 2, 3] = -100.0
        out = fusion(x, B=1, V=1, camera_params=cam_behind)

        # has_observation mask zeroes output after FFN
        assert out.abs().max() < 1e-6, \
            "No visible camera should produce zero BEV features"
