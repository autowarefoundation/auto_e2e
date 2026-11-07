"""Tests for the map encoder registry and the deformable map BEV fusion.

Covers:
  - MapDeformableCrossAttentionFusion: output shape, map influences output,
    gradient flow, configurable sample points
  - MAP_ENCODER_REGISTRY and MAP_FUSION_REGISTRY
"""

import pytest
import torch

from model_components.map_encoder import (
    MAP_ENCODER_REGISTRY,
    MAP_FUSION_REGISTRY,
    SemanticRasterEncoder,
    build_map_encoder,
    build_map_bev_fusion,
)
from model_components.map_encoder.map_bev_fusion.deformable_cross_attention_fusion import (
    MapDeformableCrossAttentionFusion,
)


def _make_bev_pair(batch_size, embed_dim, spatial, device):
    image_bev = torch.randn(batch_size, embed_dim, spatial, spatial, device=device)
    map_bev = torch.randn(batch_size, embed_dim, spatial, spatial, device=device)
    return image_bev, map_bev


def _open_deformable_navigation_path(fusion):
    with torch.no_grad():
        fusion.out_proj.weight.copy_(
            torch.eye(
                fusion.embed_dim,
                device=fusion.out_proj.weight.device,
                dtype=fusion.out_proj.weight.dtype,
            )
        )
        fusion.out_proj.bias.zero_()


class TestMapDeformableCrossAttentionFusion:
    def test_output_shape(self, device):
        fusion = MapDeformableCrossAttentionFusion(embed_dim=256).to(device)
        image_bev, map_bev = _make_bev_pair(2, 256, 8, device)
        out = fusion(image_bev, map_bev)
        assert out.shape == (2, 256, 8, 8)

    def test_initialization_preserves_camera_bev_exactly(self, device):
        fusion = MapDeformableCrossAttentionFusion(embed_dim=64).to(device)
        fusion.eval()
        image_bev, map_bev = _make_bev_pair(2, 64, 8, device)

        out = fusion(image_bev, map_bev)

        assert torch.equal(out, image_bev)

    def test_map_influences_output(self, device):
        fusion = MapDeformableCrossAttentionFusion(embed_dim=256).to(device)
        _open_deformable_navigation_path(fusion)
        fusion.eval()
        image_bev = torch.randn(1, 256, 8, 8, device=device)
        map_a = torch.randn(1, 256, 8, 8, device=device)
        map_b = torch.randn(1, 256, 8, 8, device=device)
        out_a = fusion(image_bev, map_a)
        out_b = fusion(image_bev, map_b)
        assert not torch.allclose(out_a, out_b, atol=1e-5), \
            "Different map inputs produced identical output — deformable attention has no effect"

    def test_gradient_flows_through_fusion(self, device):
        fusion = MapDeformableCrossAttentionFusion(embed_dim=64).to(device)
        _open_deformable_navigation_path(fusion)
        image_bev = torch.randn(1, 64, 4, 4, device=device, requires_grad=True)
        map_bev = torch.randn(1, 64, 4, 4, device=device, requires_grad=True)
        fusion(image_bev, map_bev).sum().backward()
        assert image_bev.grad is not None and image_bev.grad.abs().max() > 0
        assert map_bev.grad is not None and map_bev.grad.abs().max() > 0

    def test_all_parameters_receive_gradients(self, device):
        fusion = MapDeformableCrossAttentionFusion(embed_dim=64).to(device)
        _open_deformable_navigation_path(fusion)
        image_bev = torch.randn(1, 64, 4, 4, device=device)
        map_bev = torch.randn(1, 64, 4, 4, device=device)
        fusion(image_bev, map_bev).sum().backward()
        no_grad = [n for n, p in fusion.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert not no_grad, f"Parameters without grad: {no_grad}"

    def test_no_nan_with_zero_inputs(self, device):
        fusion = MapDeformableCrossAttentionFusion(embed_dim=64).to(device)
        out = fusion(
            torch.zeros(1, 64, 4, 4, device=device),
            torch.zeros(1, 64, 4, 4, device=device),
        )
        assert torch.isfinite(out).all()

    def test_output_matches_input_spatial_size(self, device):
        for h, w in [(4, 4), (8, 8), (7, 7), (4, 8)]:
            fusion = MapDeformableCrossAttentionFusion(embed_dim=32).to(device)
            image_bev = torch.randn(1, 32, h, w, device=device)
            map_bev = torch.randn(1, 32, h, w, device=device)
            out = fusion(image_bev, map_bev)
            assert out.shape == (1, 32, h, w), \
                f"Expected (1, 32, {h}, {w}), got {tuple(out.shape)}"

    def test_num_sample_points_is_configurable(self, device):
        image_bev, map_bev = _make_bev_pair(1, 32, 8, device)
        for k in (1, 4, 8):
            fusion = MapDeformableCrossAttentionFusion(
                embed_dim=32, num_sample_points=k
            ).to(device)
            out = fusion(image_bev, map_bev)
            assert out.shape == (1, 32, 8, 8), \
                f"K={k} produced {tuple(out.shape)}"



class TestMapRegistries:
    def test_semantic_raster_is_the_encoder(self):
        assert set(MAP_ENCODER_REGISTRY) == {"semantic_raster"}

    def test_unknown_encoder_type_raises(self):
        with pytest.raises(ValueError, match="Unknown map_type"):
            build_map_encoder("vectorized")

    def test_deformable_is_the_fusion(self):
        assert set(MAP_FUSION_REGISTRY) == {"deformable"}

    def test_unknown_fusion_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown map_fusion_mode"):
            build_map_bev_fusion("residual")

    def test_build_map_encoder_returns_correct_type(self):
        enc = build_map_encoder("semantic_raster", in_channels=14, embed_dim=32, output_h=4, output_w=4)
        assert isinstance(enc, SemanticRasterEncoder)

    def test_build_map_bev_fusion_deformable(self, device):
        fusion = build_map_bev_fusion("deformable", embed_dim=64).to(device)
        assert isinstance(fusion, MapDeformableCrossAttentionFusion)
