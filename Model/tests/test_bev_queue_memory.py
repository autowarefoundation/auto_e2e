"""Unit tests for the BEV-queue temporal-attention memory candidate (#20)."""

import pytest
import torch
import torch.nn as nn

from model_components.temporal_memory import (
    BaseTemporalMemory,
    BevQueueMemory,
    build_temporal_memory,
)

VIS, EGO = 896, 256


def _hist(B, T, device):
    return (torch.randn(B, T, VIS, device=device),
            torch.randn(B, T, EGO, device=device))


def test_is_base_temporal_memory(device):
    m = BevQueueMemory(visual_dim=VIS, egomotion_dim=EGO).to(device)
    assert isinstance(m, BaseTemporalMemory)


def test_output_contract_shapes(device):
    m = BevQueueMemory(visual_dim=VIS, egomotion_dim=EGO).to(device)
    v, e = _hist(2, 6, device)
    v_ctx, e_ctx = m(v, e)
    assert v_ctx.shape == (2, VIS)
    assert e_ctx.shape == (2, EGO)
    assert torch.isfinite(v_ctx).all() and torch.isfinite(e_ctx).all()


def test_registry_builds_bev_queue(device):
    m = build_temporal_memory("bev_queue", visual_dim=VIS, egomotion_dim=EGO,
                              num_heads=4, num_layers=1).to(device)
    assert isinstance(m, BevQueueMemory)
    v, e = _hist(2, 5, device)
    v_ctx, e_ctx = m(v, e)
    assert v_ctx.shape == (2, VIS) and e_ctx.shape == (2, EGO)


def test_fallback_passthrough_when_no_time_dim(device):
    m = BevQueueMemory(visual_dim=VIS, egomotion_dim=EGO).to(device)
    v = torch.randn(2, VIS, device=device)
    e = torch.randn(2, EGO, device=device)
    v_ctx, e_ctx = m(v, e)
    assert torch.equal(v_ctx, v) and torch.equal(e_ctx, e)


def test_variable_sequence_length(device):
    m = BevQueueMemory(visual_dim=VIS, egomotion_dim=EGO).to(device)
    for T in (1, 3, 16):
        v, e = _hist(2, T, device)
        v_ctx, e_ctx = m(v, e)
        assert v_ctx.shape == (2, VIS)


def test_queue_len_keeps_most_recent(device):
    """With queue_len=K, only the last K steps drive the output: changing the
    older (trimmed) steps must NOT change the context."""
    torch.manual_seed(0)
    m = BevQueueMemory(visual_dim=VIS, egomotion_dim=EGO, queue_len=3).to(device)
    m.eval()
    v, e = _hist(1, 8, device)
    v_ctx_a, _ = m(v, e)
    # Mutate the OLDEST step (index 0), which is trimmed away by queue_len=3.
    v2 = v.clone()
    v2[:, 0, :] += 100.0
    v_ctx_b, _ = m(v2, e)
    assert torch.allclose(v_ctx_a, v_ctx_b, atol=1e-5)


def test_temporal_order_matters(device):
    """Positional encoding makes the module order-aware: reversing the queue
    must change the output (unlike a permutation-invariant pool)."""
    torch.manual_seed(0)
    m = BevQueueMemory(visual_dim=VIS, egomotion_dim=EGO).to(device)
    m.eval()
    v, e = _hist(1, 6, device)
    fwd, _ = m(v, e)
    rev, _ = m(v.flip(1), e.flip(1))
    assert not torch.allclose(fwd, rev, atol=1e-5)


def test_gradients_flow(device):
    m = BevQueueMemory(visual_dim=VIS, egomotion_dim=EGO, num_layers=1).to(device)
    v, e = _hist(2, 5, device)
    v_ctx, e_ctx = m(v, e)
    (v_ctx.pow(2).mean() + e_ctx.pow(2).mean()).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


def test_invalid_args():
    with pytest.raises(ValueError, match="even"):
        BevQueueMemory(d_model=255)
    with pytest.raises(ValueError, match="divisible"):
        BevQueueMemory(d_model=512, num_heads=7)
    with pytest.raises(ValueError, match="queue_len"):
        BevQueueMemory(queue_len=0)


def test_reactive_e2e_with_bev_queue_memory(device):
    """ReactiveE2E (post-refactor home of the temporal memory) must accept
    temporal_memory_mode='bev_queue' with a [B, T, feat] history and run
    end-to-end (BEV fusion; planner='bezier' since GRU was removed)."""
    from unittest.mock import patch

    from model_components.reactive_e2e import ReactiveE2E

    class _MockBackbone(nn.Module):
        def __init__(self, backbone="swin_v2_tiny", is_pretrained=True, **kwargs):
            super().__init__()
            self.backbone_channels = 1440
            self._stages = nn.ModuleList([
                nn.Sequential(nn.Conv2d(3, 96, 3, 1, 1), nn.AdaptiveAvgPool2d(64)),
                nn.Sequential(nn.Conv2d(96, 192, 3, 1, 1), nn.AdaptiveAvgPool2d(32)),
                nn.Sequential(nn.Conv2d(192, 384, 3, 1, 1), nn.AdaptiveAvgPool2d(16)),
                nn.Sequential(nn.Conv2d(384, 768, 3, 1, 1), nn.AdaptiveAvgPool2d(8)),
            ])

        def forward(self, image):
            outs, x = [], image
            for stage in self._stages:
                x = stage(x)
                outs.append(x)
            return outs

    with patch("model_components.reactive_e2e.Backbone", _MockBackbone):
        model = ReactiveE2E(num_views=8,
                            view_fusion_kwargs={"bev_h": 8, "bev_w": 8},
                            planner_mode="bezier",
                            temporal_memory_mode="bev_queue",
                            temporal_memory_kwargs={"num_layers": 1}).to(device)
    assert isinstance(model.TemporalMemory, BevQueueMemory)

    x = torch.randn(2, 8, 3, 256, 256, device=device)
    map_input = torch.randn(2, 3, 256, 256, device=device)
    vis = torch.randn(2, 4, 896, device=device)   # [B, T, feat] history
    ego = torch.randn(2, 4, 256, device=device)
    out = model(x, map_input, vis, ego)
    traj = out[0] if isinstance(out, (tuple, list)) else out
    assert traj.shape == (2, 128)
    assert torch.isfinite(traj).all()
