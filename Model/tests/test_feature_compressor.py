"""Unit tests for FeatureCompressor (memory-bounded JEPA forecast space)."""

import pytest
import torch

from model_components.feature_compressor import FeatureCompressor

EMBED = 256


def test_full_mode_is_channel_identity(device):
    c = FeatureCompressor(EMBED, mode="full").to(device)
    assert c.out_channels == EMBED
    x = torch.randn(2, EMBED, 8, 8, device=device)
    out = c(x)
    assert out.shape == (2, EMBED, 8, 8)
    assert torch.equal(out, x)  # no projection, no pooling


def test_occupancy_mode_collapses_to_one_channel(device):
    c = FeatureCompressor(EMBED, mode="occupancy").to(device)
    assert c.out_channels == 1
    x = torch.randn(2, EMBED, 8, 8, device=device)
    out = c(x)
    assert out.shape == (2, 1, 8, 8)
    assert torch.allclose(out, x.mean(dim=1, keepdim=True))


def test_projected_mode_reduces_channels_with_learned_proj(device):
    c = FeatureCompressor(EMBED, mode="projected", compressed_dim=16).to(device)
    assert c.out_channels == 16
    x = torch.randn(2, EMBED, 8, 8, device=device)
    out = c(x)
    assert out.shape == (2, 16, 8, 8)
    assert any(p.requires_grad for p in c.parameters())  # learnable projection


def test_spatial_stride_downsamples(device):
    c = FeatureCompressor(EMBED, mode="projected", compressed_dim=16,
                          spatial_stride=4).to(device)
    x = torch.randn(2, EMBED, 32, 24, device=device)
    out = c(x)
    assert out.shape == (2, 16, 8, 6)  # 32/4, 24/4


def test_memory_saving_matches_channel_mean_with_more_info(device):
    """projected(16) + stride/4 reaches the same element count as the channel
    mean (occupancy) but keeps 16 learned channels instead of 1."""
    x = torch.randn(1, EMBED, 32, 32, device=device)
    occ = FeatureCompressor(EMBED, mode="occupancy").to(device)(x)        # 1*32*32
    proj = FeatureCompressor(EMBED, mode="projected", compressed_dim=16,
                             spatial_stride=4).to(device)(x)               # 16*8*8
    assert occ.numel() == proj.numel()  # same memory budget
    assert proj.shape[1] == 16          # but 16 semantic channels, not 1


def test_invalid_mode_and_stride():
    with pytest.raises(ValueError, match="mode must be"):
        FeatureCompressor(EMBED, mode="nope")
    with pytest.raises(ValueError, match="spatial_stride"):
        FeatureCompressor(EMBED, spatial_stride=0)


def test_gradients_flow_through_projection(device):
    c = FeatureCompressor(EMBED, mode="projected", compressed_dim=8).to(device)
    x = torch.randn(2, EMBED, 8, 8, device=device)
    c(x).pow(2).mean().backward()
    assert all(p.grad is not None for p in c.parameters())
