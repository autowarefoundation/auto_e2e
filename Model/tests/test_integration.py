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
        """Smoke test: full model forward produces the expected trajectory shape."""
        visual, map_input, vis_hist, ego = make_inputs(1, 7, device)

        traj = full_model(visual, map_input, vis_hist, ego, mode="train")
        assert traj.shape == (1, 128)

        traj2 = full_model(visual, map_input, vis_hist, ego, mode="infer")
        assert traj2.shape == (1, 128)

    def test_full_forward_no_nan(self, full_model, device):
        """Full pipeline must not produce NaN with real backbone weights."""
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        traj = full_model(visual, map_input, vis_hist, ego, mode="train")
        assert not torch.isnan(traj).any()


@pytest.mark.integration
class TestResNet50Backbone:
    """Exercises the dynamic backbone_channels computation on a backbone
    whose feature_info shape differs from Swin (5 stages of channels
    64/256/512/1024/2048 vs Swin's 4 stages of 96/192/384/768)."""

    def test_resnet50_forward_pass(self, device):
        from model_components.auto_e2e import AutoE2E
        try:
            model = AutoE2E(
                backbone="res_net_50", num_views=7,
                view_fusion_kwargs={"bev_h": 8, "bev_w": 8},
                is_pretrained=False,
            ).to(device)
        except (FileNotFoundError, OSError) as e:
            pytest.skip(f"Backbone construction failed: {e}")

        # Dynamic backbone_channels = sum of all 5 ResNet50 stages = 3904
        assert model.Reactive_E2E.Backbone.backbone_channels == \
            64 + 256 + 512 + 1024 + 2048

        visual, map_input, vis_hist, ego = make_inputs(1, 7, device)

        traj = model(visual, map_input, vis_hist, ego, mode="train")
        assert traj.shape == (1, 128) and torch.isfinite(traj).all()

        traj2 = model(visual, map_input, vis_hist, ego, mode="infer")
        assert traj2.shape == (1, 128) and torch.isfinite(traj2).all()


class TestRealArchitectureSmoke:
    """Exercises the real timm backbone architectures with random weights
    (is_pretrained=False, so no pretrained download), covering per-stage
    channel discovery the mock backbone hardcodes — including ResNet50's
    5 stages — and forward-signature regressions."""

    EXPECTED_CHANNELS = {
        "swin_v2_tiny": 96 + 192 + 384 + 768,
        "conv_next_v2_tiny": 96 + 192 + 384 + 768,
        "res_net_50": 64 + 256 + 512 + 1024 + 2048,
    }

    @pytest.mark.parametrize(
        "backbone", ["swin_v2_tiny", "conv_next_v2_tiny", "res_net_50"]
    )
    def test_real_backbone_forward(self, backbone, device):
        from model_components.auto_e2e import AutoE2E
        model = AutoE2E(
            backbone=backbone, num_views=7,
            view_fusion_kwargs={"bev_h": 8, "bev_w": 8},
            is_pretrained=False,
        ).to(device)

        assert model.Reactive_E2E.Backbone.backbone_channels == \
            self.EXPECTED_CHANNELS[backbone]

        visual, map_input, vis_hist, ego = make_inputs(1, 7, device)

        traj = model(visual, map_input, vis_hist, ego, mode="train")
        assert traj.shape == (1, 128) and torch.isfinite(traj).all()

        traj2 = model(visual, map_input, vis_hist, ego, mode="infer")
        assert traj2.shape == (1, 128) and torch.isfinite(traj2).all()
