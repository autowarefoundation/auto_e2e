import torch
import sys
sys.path.append('..')

from model_components.feature_fusion import FeatureFusion


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
