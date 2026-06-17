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
        """Smoke test: full model forward produces expected output shapes."""
        visual, map_input, vis_hist, ego = make_inputs(1, 7, device)
        target = torch.randn(1, 128, device=device)
        loss, ego_hidden, future = full_model(
             visual, map_input, vis_hist, ego,
            mode="train", trajectory_target=target)

        assert loss.dim() == 0
        assert ego_hidden.shape == (1, 256)
        assert len(future) == 4
        for f in future:
            assert f.shape == (1, 256, 8, 8)

        traj, _, _ = full_model(visual, vis_hist, ego, mode="infer")
        assert traj.shape == (1, 128)

    def test_full_forward_no_nan(self, full_model, device):
        """Full pipeline must not produce NaN with real backbone weights."""
        visual, map_input, vis_hist, ego = make_inputs(2, 7, device)
        target = torch.randn(2, 128, device=device)
        loss, ego_hidden, future = full_model(visual, map_input, vis_hist, ego,
                                              mode="train", trajectory_target=target)

        assert not torch.isnan(loss)
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
                backbone="res_net_50", num_views=7, fusion_mode="concat",
                is_pretrained=False,
            ).to(device)
        except (FileNotFoundError, OSError) as e:
            pytest.skip(f"Backbone construction failed: {e}")

        # Dynamic backbone_channels = sum of all 5 ResNet50 stages = 3904
        assert model.Backbone.backbone_channels == 64 + 256 + 512 + 1024 + 2048

        visual, map_input, vis_hist, ego = make_inputs(1, 7, device)
        target = torch.randn(1, 128, device=device)
        loss, ego_hidden, future = model(
            visual, map_input, vis_hist, ego, mode="train", trajectory_target=target)

        assert loss.dim() == 0
        assert ego_hidden.shape == (1, 256)
        assert len(future) == 4
        for f in future:
            assert f.shape == (1, 256, 8, 8)
        assert torch.isfinite(loss)
        assert torch.isfinite(ego_hidden).all()

        traj, _, _ = model(visual, vis_hist, ego, mode="infer")
        assert traj.shape == (1, 128)
        assert torch.isfinite(traj).all()
