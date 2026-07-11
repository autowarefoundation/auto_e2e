import pytest
import torch
import sys
sys.path.append('..')

from model_components.feature_fusion import FeatureFusion
from model_components.view_fusion import build_view_fusion, FUSION_REGISTRY
from model_components.view_fusion.bev_fusion import BEVViewFusion
from model_components.view_fusion.projection import (
    FThetaProjection,
    PinholeProjection,
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
# View fusion effectiveness — different views must influence output
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
        visual, map_input, vis_hist, ego = make_inputs(1, 7, device)
        B, V, C, H, W = visual.shape

        def fused_features(x):
            features = model.Reactive_E2E.Backbone(x.reshape(B * V, C, H, W))
            return model.Reactive_E2E.FeatureFusion(features, B, V)

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
        visual, map_input, vis_hist, ego = make_inputs(1, 7, device)
        B, V, C, H, W = visual.shape

        def fused_features(x):
            features = model.Reactive_E2E.Backbone(x.reshape(B * V, C, H, W))
            return model.Reactive_E2E.FeatureFusion(features, B, V)

        fused_base = fused_features(visual)

        for view_idx in range(V):
            visual_mod = visual.clone()
            visual_mod[0, view_idx] = 5.0
            fused_mod = fused_features(visual_mod)
            assert not torch.allclose(fused_base, fused_mod, atol=1e-5), \
                f"View {view_idx} has no influence on the fused feature map"

    def test_views_contribute_to_fused_with_real_projection(self, build_mock_model, device):
        """Every view must influence the fused feature map when a REAL pinhole
        projection operator drives the BEV projection path.

        The other view-contribution tests run only the pseudo path, which uses
        the learnable prior rather than geometry-driven projection. Real
        deployments pass a projection operator; this test strengthens coverage by
        feeding a PinholeProjection through FeatureFusion and verifying each view
        still contributes.
        """
        model = build_mock_model(num_views=4, fusion_mode="bev", device=device)
        model.eval()
        torch.manual_seed(42)
        visual, map_input, vis_hist, ego = make_inputs(1, 4, device)
        B, V, C, H, W = visual.shape

        # Identity-like ego-to-pixel projection that places every BEV reference
        # point inside the image with positive depth on every view:
        #   u_pix = x_world + 128, v_pix = y_world + 128, depth = 1.
        # With image_size=256 and the default pc_range
        # (x in [-60, 120], y in [-60, 60]), normalized image coords land in
        # [0.27, 0.97] × [0.27, 0.73] — fully visible.
        cam_params = torch.zeros(B, V, 3, 4, device=device)
        cam_params[..., 0, 0] = 1.0
        cam_params[..., 0, 3] = 128.0
        cam_params[..., 1, 1] = 1.0
        cam_params[..., 1, 3] = 128.0
        cam_params[..., 2, 3] = 1.0
        projection = PinholeProjection(cam_params)

        def fused_features(x):
            features = model.Reactive_E2E.Backbone(x.reshape(B * V, C, H, W))
            return model.Reactive_E2E.FeatureFusion(features, B, V, projection=projection)

        fused_base = fused_features(visual)

        for view_idx in range(V):
            visual_mod = visual.clone()
            visual_mod[0, view_idx] = 5.0
            fused_mod = fused_features(visual_mod)
            assert not torch.allclose(fused_base, fused_mod, atol=1e-5), \
                f"View {view_idx} has no influence on the fused feature map " \
                f"under real projection"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestFusionRegistry:
    def test_all_modes_registered(self):
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

    def test_output_shape_with_pinhole_projection(self, device):
        """BEV fusion should work with an explicit pinhole projection operator."""
        fusion = BEVViewFusion(num_views=8, embed_dim=256, bev_h=8, bev_w=8).to(device)
        x = torch.randn(16, 256, 8, 8, device=device)
        proj = PinholeProjection(torch.randn(2, 8, 3, 4, device=device))
        out = fusion(x, B=2, V=8, projection=proj)
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

        cam_a = PinholeProjection(torch.randn(1, 4, 3, 4, device=device))
        cam_b = PinholeProjection(torch.randn(1, 4, 3, 4, device=device))

        out_a = fusion(x, B=1, V=4, projection=cam_a)
        out_b = fusion(x, B=1, V=4, projection=cam_b)

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
        out = fusion(x, B=1, V=1, projection=PinholeProjection(cam))

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

    def test_offset_scale_bool_raises(self):
        # bool is an int subclass; the validator must reject it explicitly.
        with pytest.raises(ValueError, match="offset_scale"):
            BEVViewFusion(num_views=4, embed_dim=256, bev_h=8, bev_w=8,
                          offset_scale=True)

    def test_offset_scale_non_numeric_raises(self):
        with pytest.raises(ValueError, match="offset_scale"):
            BEVViewFusion(num_views=4, embed_dim=256, bev_h=8, bev_w=8,
                          offset_scale="0.1")

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
        out = fusion(x, B=1, V=1, projection=PinholeProjection(cam_behind))

        # has_observation mask zeroes output after FFN
        assert out.abs().max() < 1e-6, \
            "No visible camera should produce zero BEV features"


# ---------------------------------------------------------------------------
# Runtime-V-dynamic tests (Issue #77): one instance, any camera count
# ---------------------------------------------------------------------------

class TestBEVFusionVDynamic:
    """A single BEVViewFusion instance must consume batches of any view count.

    This is the core Issue #77 acceptance: one model, alternating V (e.g. L2D's
    6 real cams and NVIDIA's 7), no re-instantiation, output always
    [B, embed_dim, bev_h, bev_w], gradients flowing both times.
    """

    def test_alternating_view_counts_same_instance(self, device):
        fusion = BEVViewFusion(num_views=6, embed_dim=256, bev_h=8, bev_w=8).to(device)
        for V in (6, 7, 6, 8):
            x = torch.randn(2 * V, 256, 8, 8, device=device)
            out = fusion(x, B=2, V=V)
            assert out.shape == (2, 256, 8, 8), f"wrong shape at V={V}"

    def test_gradients_flow_for_both_view_counts(self, device):
        """bev_queries / value_proj / output_proj must receive gradients at
        every view count (the #77 gradient-flow criterion)."""
        fusion = BEVViewFusion(num_views=6, embed_dim=256, bev_h=8, bev_w=8).to(device)
        cam7 = torch.zeros(1, 7, 3, 4, device=device)
        cam7[..., 0, 0] = 1.0
        cam7[..., 0, 3] = 128.0
        cam7[..., 1, 1] = 1.0
        cam7[..., 1, 3] = 128.0
        cam7[..., 2, 3] = 1.0
        for V, cam in ((7, cam7), (6, cam7[:, :6])):
            fusion.zero_grad(set_to_none=True)
            x = torch.randn(V, 256, 8, 8, device=device)
            out = fusion(x, B=1, V=V, projection=PinholeProjection(cam))
            out.sum().backward()
            for name, p in (("bev_queries", fusion.bev_queries.weight),
                            ("value_proj", fusion.value_proj.weight),
                            ("output_proj", fusion.output_proj.weight)):
                assert p.grad is not None and p.grad.abs().max() > 0, \
                    f"{name} got no gradient at V={V}"

    def test_pseudo_path_view_count_agnostic(self, device):
        """The calibration-free pseudo prior also runs any V on one instance."""
        fusion = BEVViewFusion(num_views=8, embed_dim=256, bev_h=8, bev_w=8).to(device)
        for V in (5, 6, 7, 8):
            out = fusion(torch.randn(V, 256, 8, 8, device=device), B=1, V=V)
            assert out.shape == (1, 256, 8, 8)
            assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Honest-geometry contract (Issue #77): no silent pseudo, no false real claim
# ---------------------------------------------------------------------------

class TestGeometryDeclaration:
    def test_pseudo_label_contradicts_real_operator(self, device):
        """geometry_type='pseudo' must not label a real (pinhole) operator."""
        fusion = BEVViewFusion(num_views=4, embed_dim=256, bev_h=8, bev_w=8).to(device)
        x = torch.randn(4, 256, 8, 8, device=device)
        proj = PinholeProjection(torch.randn(1, 4, 3, 4, device=device))
        with pytest.raises(ValueError, match="contradicts"):
            fusion(x, B=1, V=4, projection=proj, geometry_type="pseudo")

    def test_real_label_without_operator_raises(self, device):
        """Claiming real geometry without a projection operator is rejected — the
        pseudo path is never entered on behalf of a caller that meant real."""
        fusion = BEVViewFusion(num_views=4, embed_dim=256, bev_h=8, bev_w=8).to(device)
        x = torch.randn(4, 256, 8, 8, device=device)
        with pytest.raises(ValueError, match="requires a projection operator"):
            fusion(x, B=1, V=4, geometry_type="pinhole")

    def test_projection_view_count_must_match_v(self, device):
        fusion = BEVViewFusion(num_views=4, embed_dim=256, bev_h=8, bev_w=8).to(device)
        x = torch.randn(4, 256, 8, 8, device=device)
        proj = PinholeProjection(torch.randn(1, 5, 3, 4, device=device))  # 5 != 4
        with pytest.raises(ValueError, match="num_views"):
            fusion(x, B=1, V=4, projection=proj)

    def test_unknown_geometry_type_raises(self, device):
        fusion = BEVViewFusion(num_views=4, embed_dim=256, bev_h=8, bev_w=8).to(device)
        x = torch.randn(4, 256, 8, 8, device=device)
        with pytest.raises(ValueError, match="geometry_type"):
            fusion(x, B=1, V=4, geometry_type="bogus")


# ---------------------------------------------------------------------------
# Native f-theta projection through the fusion module (no rectification)
# ---------------------------------------------------------------------------

class TestFThetaFusion:
    def test_ftheta_projection_runs_end_to_end(self, device):
        fusion = BEVViewFusion(num_views=3, embed_dim=256, bev_h=8, bev_w=8,
                               image_size=256).to(device)
        T = torch.eye(4, device=device).reshape(1, 1, 4, 4).expand(1, 3, 4, 4).contiguous()
        fw_poly = torch.tensor([0.0, 200.0], device=device)
        proj = FThetaProjection(T, fw_poly, cx=128.0, cy=128.0)
        x = torch.randn(3, 256, 8, 8, device=device)
        out = fusion(x, B=1, V=3, projection=proj, geometry_type="ftheta")
        assert out.shape == (1, 256, 8, 8)
        assert not torch.isnan(out).any()
