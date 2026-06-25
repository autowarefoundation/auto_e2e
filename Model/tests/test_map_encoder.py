"""Tests for the map encoder and map BEV fusion modules.

Covers:
  - RasterizedMapEncoder: output shape, channel layout, gradient flow
  - ResidualMapFusion: zero-alpha init, output shape, alpha learns, gradient flow
  - MapCrossAttentionFusion: output shape, map influences output, gradient flow
  - MAP_ENCODER_REGISTRY and MAP_FUSION_REGISTRY
  - AutoE2E map integration: zero map produces same output as alpha=0 baseline,
    map_input influences trajectory once alpha is non-zero, MapEncoder and
    MapBEVFusion parameters receive gradients
"""

import pytest
import torch
import sys
sys.path.append('..')

from model_components.map_encoder import (
    MAP_ENCODER_REGISTRY,
    MAP_FUSION_REGISTRY,
    build_map_encoder,
    build_map_bev_fusion,
    RasterizedMapEncoder,
)
from model_components.map_encoder.map_bev_fusion.residual_fusion import ResidualMapFusion
from model_components.map_encoder.map_bev_fusion.cross_attention_fusion import MapCrossAttentionFusion


_MAP_H = 256
_MAP_W = 256


def _make_bev_pair(batch_size, embed_dim, spatial, device):
    image_bev = torch.randn(batch_size, embed_dim, spatial, spatial, device=device)
    map_bev = torch.randn(batch_size, embed_dim, spatial, spatial, device=device)
    return image_bev, map_bev


@pytest.fixture(scope="session")
def map_encoder(device):
    return RasterizedMapEncoder(embed_dim=256, output_h=8, output_w=8).to(device)


class TestRasterizedMapEncoder:
    def test_output_shape(self, map_encoder, device):
        x = torch.randn(2, 3, _MAP_H, _MAP_W, device=device)
        out = map_encoder(x)
        assert out.shape == (2, 256, 8, 8), f"Expected (2, 256, 8, 8), got {tuple(out.shape)}"

    def test_output_is_channels_first(self, device):
        enc = RasterizedMapEncoder(embed_dim=64, output_h=4, output_w=4).to(device)
        x = torch.randn(1, 3, _MAP_H, _MAP_W, device=device)
        out = enc(x)
        assert out.dim() == 4
        assert out.shape[1] == 64, "Channel dim should be at position 1"

    def test_output_size_is_honoured(self, device):
        for h, w in [(8, 8), (8, 16), (16, 8)]:
            enc = RasterizedMapEncoder(embed_dim=32, output_h=h, output_w=w).to(device)
            x = torch.randn(1, 3, _MAP_H, _MAP_W, device=device)
            out = enc(x)
            assert out.shape[-2:] == (h, w), \
                f"Expected spatial ({h}, {w}), got {tuple(out.shape[-2:])}"

    def test_no_nan_on_zero_input(self, map_encoder, device):
        x = torch.zeros(2, 3, _MAP_H, _MAP_W, device=device)
        out = map_encoder(x)
        assert torch.isfinite(out).all(), "NaN/Inf with zero map input"

    def test_different_inputs_produce_different_outputs(self, map_encoder, device):
        map_encoder.eval()
        a = torch.randn(1, 3, _MAP_H, _MAP_W, device=device)
        b = torch.randn(1, 3, _MAP_H, _MAP_W, device=device)
        assert not torch.allclose(map_encoder(a), map_encoder(b), atol=1e-5), \
            "Different map inputs produced identical features"

    def test_gradient_flows_through_encoder(self, device):
        enc = RasterizedMapEncoder(embed_dim=64, output_h=4, output_w=4).to(device)
        x = torch.randn(1, 3, _MAP_H, _MAP_W, device=device, requires_grad=True)
        enc(x).sum().backward()
        assert x.grad is not None and x.grad.abs().max() > 0

    def test_all_parameters_receive_gradients(self, device):
        enc = RasterizedMapEncoder(embed_dim=64, output_h=4, output_w=4).to(device)
        x = torch.randn(2, 3, _MAP_H, _MAP_W, device=device)
        enc(x).sum().backward()
        no_grad = [n for n, p in enc.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert not no_grad, f"Parameters without grad: {no_grad}"

    def test_is_randomly_initialized(self):
        """Two independently constructed encoders must have different weights."""
        enc_a = RasterizedMapEncoder(embed_dim=64, output_h=4, output_w=4)
        enc_b = RasterizedMapEncoder(embed_dim=64, output_h=4, output_w=4)
        w_a = next(enc_a._backbone.parameters())
        w_b = next(enc_b._backbone.parameters())
        assert not torch.equal(w_a, w_b), \
            "Two encoders have identical weights — random init may not be working"


class TestResidualMapFusion:
    def test_output_shape(self, device):
        fusion = ResidualMapFusion(embed_dim=256).to(device)
        image_bev, map_bev = _make_bev_pair(2, 256, 8, device)
        out = fusion(image_bev, map_bev)
        assert out.shape == (2, 256, 8, 8)

    def test_alpha_initialized_to_zero(self):
        fusion = ResidualMapFusion(embed_dim=256)
        assert torch.all(fusion.alpha == 0), \
            "alpha must be zero-initialized so map has no effect at training start"

    def test_zero_alpha_returns_image_bev_unchanged(self, device):
        fusion = ResidualMapFusion(embed_dim=256).to(device)
        image_bev, map_bev = _make_bev_pair(1, 256, 8, device)
        out = fusion(image_bev, map_bev)
        assert torch.allclose(out, image_bev), \
            "With alpha=0 the output should equal image_bev exactly"

    def test_nonzero_alpha_changes_output(self, device):
        fusion = ResidualMapFusion(embed_dim=256).to(device)
        with torch.no_grad():
            fusion.alpha.fill_(1.0)
        image_bev, map_bev = _make_bev_pair(1, 256, 8, device)
        out = fusion(image_bev, map_bev)
        assert not torch.allclose(out, image_bev, atol=1e-5), \
            "With alpha=1 the output should differ from image_bev"

    def test_alpha_is_per_channel(self, device):
        embed_dim = 64
        fusion = ResidualMapFusion(embed_dim=embed_dim).to(device)
        assert fusion.alpha.shape == (embed_dim,), \
            f"alpha should have shape ({embed_dim},), got {tuple(fusion.alpha.shape)}"

    def test_alpha_receives_gradient(self, device):
        fusion = ResidualMapFusion(embed_dim=256).to(device)
        image_bev, map_bev = _make_bev_pair(2, 256, 8, device)
        with torch.no_grad():
            fusion.alpha.fill_(0.1)
        fusion(image_bev, map_bev).sum().backward()
        assert fusion.alpha.grad is not None, "alpha has no gradient"
        assert fusion.alpha.grad.abs().max() > 0, "alpha gradient is all-zero"

    def test_no_nan_with_zero_inputs(self, device):
        fusion = ResidualMapFusion(embed_dim=256).to(device)
        with torch.no_grad():
            fusion.alpha.fill_(1.0)
        out = fusion(
            torch.zeros(1, 256, 8, 8, device=device),
            torch.zeros(1, 256, 8, 8, device=device),
        )
        assert torch.isfinite(out).all()


class TestMapCrossAttentionFusion:
    def test_output_shape(self, device):
        fusion = MapCrossAttentionFusion(embed_dim=256).to(device)
        image_bev, map_bev = _make_bev_pair(2, 256, 8, device)
        out = fusion(image_bev, map_bev)
        assert out.shape == (2, 256, 8, 8)

    def test_map_influences_output(self, device):
        fusion = MapCrossAttentionFusion(embed_dim=256).to(device)
        fusion.eval()
        image_bev = torch.randn(1, 256, 8, 8, device=device)
        map_a = torch.randn(1, 256, 8, 8, device=device)
        map_b = torch.randn(1, 256, 8, 8, device=device)
        out_a = fusion(image_bev, map_a)
        out_b = fusion(image_bev, map_b)
        assert not torch.allclose(out_a, out_b, atol=1e-5), \
            "Different map inputs produced identical output — cross-attention has no effect"

    def test_gradient_flows_through_fusion(self, device):
        fusion = MapCrossAttentionFusion(embed_dim=64).to(device)
        image_bev = torch.randn(1, 64, 4, 4, device=device, requires_grad=True)
        map_bev = torch.randn(1, 64, 4, 4, device=device, requires_grad=True)
        fusion(image_bev, map_bev).sum().backward()
        assert image_bev.grad is not None and image_bev.grad.abs().max() > 0
        assert map_bev.grad is not None and map_bev.grad.abs().max() > 0

    def test_no_nan_with_zero_inputs(self, device):
        fusion = MapCrossAttentionFusion(embed_dim=64).to(device)
        out = fusion(
            torch.zeros(1, 64, 4, 4, device=device),
            torch.zeros(1, 64, 4, 4, device=device),
        )
        assert torch.isfinite(out).all()

    def test_output_matches_input_spatial_size(self, device):
        for h, w in [(4, 4), (8, 8), (7, 7)]:
            fusion = MapCrossAttentionFusion(embed_dim=32).to(device)
            image_bev = torch.randn(1, 32, h, w, device=device)
            map_bev = torch.randn(1, 32, h, w, device=device)
            out = fusion(image_bev, map_bev)
            assert out.shape == (1, 32, h, w), \
                f"Expected (1, 32, {h}, {w}), got {tuple(out.shape)}"



class TestMapRegistries:
    def test_rasterized_in_encoder_registry(self):
        assert "rasterized" in MAP_ENCODER_REGISTRY

    def test_unknown_encoder_type_raises(self):
        with pytest.raises(ValueError, match="Unknown map_type"):
            build_map_encoder("vectorized")

    def test_residual_and_cross_attn_in_fusion_registry(self):
        assert "residual" in MAP_FUSION_REGISTRY
        assert "cross_attn" in MAP_FUSION_REGISTRY

    def test_unknown_fusion_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown map_fusion_mode"):
            build_map_bev_fusion("nonexistent")

    def test_build_map_encoder_returns_correct_type(self):
        enc = build_map_encoder("rasterized", embed_dim=32, output_h=4, output_w=4)
        assert isinstance(enc, RasterizedMapEncoder)

    def test_build_map_bev_fusion_residual(self, device):
        fusion = build_map_bev_fusion("residual", embed_dim=64).to(device)
        assert isinstance(fusion, ResidualMapFusion)

    def test_build_map_bev_fusion_cross_attn(self, device):
        fusion = build_map_bev_fusion("cross_attn", embed_dim=64).to(device)
        assert isinstance(fusion, MapCrossAttentionFusion)


class TestAutoE2EMapIntegration:
    """End-to-end tests verifying the map branch is correctly wired into AutoE2E.

    Uses the build_mock_model fixture from conftest — it already handles patching
    Backbone with MockBackbone internally, so there's no need to import conftest
    directly (pytest makes conftest a plugin, not an importable module).
    """

    def _make_model(self, build_mock_model, device, map_fusion_mode="residual"):
        return build_mock_model(
            num_views=7,
            fusion_mode="bev",
            device=device,
            map_fusion_mode=map_fusion_mode,
        )

    def test_zero_map_with_zero_alpha_equals_no_map(self, build_mock_model, device):
        """Only for residual fusion: With alpha=0 (init) and zero map input, output must equal a forward pass
        using the same camera inputs but a random map — proving the gate is closed."""
        model = self._make_model(build_mock_model, device, map_fusion_mode="residual")
        model.eval()

        visual = torch.randn(2, 7, 3, 256, 256, device=device)
        map_zero = torch.zeros(2, 3, _MAP_H, _MAP_W, device=device)
        map_rand = torch.randn(2, 3, _MAP_H, _MAP_W, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)

        traj_zero = model(visual, map_zero, vis_hist, ego, mode="infer")
        traj_rand = model(visual, map_rand, vis_hist, ego, mode="infer")

        assert torch.allclose(traj_zero, traj_rand, atol=1e-5), \
            "With alpha=0, different map inputs should produce identical trajectories"

    def test_nonzero_alpha_makes_map_influence_trajectory(self, build_mock_model, device):
        """Only for residual fusion: Once alpha is non-zero, different map inputs must produce different trajectories."""
        model = self._make_model(build_mock_model, device, map_fusion_mode="residual")
        model.eval()
        with torch.no_grad():
            model.Reactive_E2E.MapBEVFusion.alpha.fill_(1.0)

        visual = torch.randn(2, 7, 3, 256, 256, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)

        traj_a = model(visual, torch.randn(2, 3, _MAP_H, _MAP_W, device=device), vis_hist, ego, mode="infer")
        traj_b = model(visual, torch.randn(2, 3, _MAP_H, _MAP_W, device=device), vis_hist, ego, mode="infer")

        assert not torch.allclose(traj_a, traj_b, atol=1e-5), \
            "With alpha=1, different map inputs should produce different trajectories"

    def test_cross_attn_map_encoder_parameters_receive_gradients(self, build_mock_model, device):
        model = self._make_model(build_mock_model, device, map_fusion_mode="cross_attn")
        model.train()

        visual = torch.randn(2, 7, 3, 256, 256, device=device)
        map_input = torch.randn(2, 3, _MAP_H, _MAP_W, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)

        traj = model(visual, map_input, vis_hist, ego, mode="train")
        traj.pow(2).mean().backward()

        no_grad = [n for n, p in model.Reactive_E2E.MapEncoder.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert not no_grad, f"MapEncoder params without grad: {no_grad}"

    def test_alpha_receives_gradient(self, build_mock_model, device):
        """Only for residual fusion: alpha must receive a non-zero gradient when map influences trajectory."""
        model = self._make_model(build_mock_model, device, map_fusion_mode="residual")
        model.train()
        with torch.no_grad():
            model.Reactive_E2E.MapBEVFusion.alpha.fill_(0.1)

        visual = torch.randn(2, 7, 3, 256, 256, device=device)
        map_input = torch.randn(2, 3, _MAP_H, _MAP_W, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)

        traj = model(visual, map_input, vis_hist, ego, mode="train")
        traj.pow(2).mean().backward()

        alpha = model.Reactive_E2E.MapBEVFusion.alpha
        assert alpha.grad is not None, "alpha has no gradient"
        assert alpha.grad.abs().max() > 0, "alpha gradient is all-zero"

    def test_map_encoder_receives_gradients_when_alpha_nonzero(self, build_mock_model, device):
        """Only for residual fusion: MapEncoder parameters must receive gradients once alpha is non-zero."""
        model = self._make_model(build_mock_model, device, map_fusion_mode="residual")
        model.train()
        with torch.no_grad():
            model.Reactive_E2E.MapBEVFusion.alpha.fill_(0.1)

        visual = torch.randn(2, 7, 3, 256, 256, device=device)
        map_input = torch.randn(2, 3, _MAP_H, _MAP_W, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)

        traj = model(visual, map_input, vis_hist, ego, mode="train")
        traj.pow(2).mean().backward()

        no_grad = [n for n, p in model.Reactive_E2E.MapEncoder.named_parameters()
                if p.requires_grad and p.grad is None]
        assert not no_grad, f"MapEncoder params without grad after alpha=0.1: {no_grad}"

    def test_cross_attn_fusion_mode_forward_succeeds(self, build_mock_model, device):
        model = self._make_model(build_mock_model, device, map_fusion_mode="cross_attn")
        visual = torch.randn(2, 7, 3, 256, 256, device=device)
        map_input = torch.randn(2, 3, _MAP_H, _MAP_W, device=device)
        vis_hist = torch.randn(2, 896, device=device)
        ego = torch.randn(2, 256, device=device)
        traj = model(visual, map_input, vis_hist, ego, mode="infer")
        assert traj.shape == (2, 128)
        assert torch.isfinite(traj).all()

    def test_map_encoder_attribute_exists(self, build_mock_model, device):
        model = self._make_model(build_mock_model, device)
        assert hasattr(model.Reactive_E2E, "MapEncoder"), "Reactive_E2E missing MapEncoder attribute"
        assert hasattr(model.Reactive_E2E, "MapBEVFusion"), "Reactive_E2E missing MapBEVFusion attribute"

    def test_invalid_map_fusion_mode_raises(self, build_mock_model, device):
        with pytest.raises(ValueError, match="Unknown map_fusion_mode"):
            build_mock_model(
                num_views=7,
                fusion_mode="bev",
                device=device,
                map_fusion_mode="nonexistent",
            )