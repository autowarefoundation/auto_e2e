"""Unit tests for the optional 1 Hz history encoder (Issue #20).

HistoryEncoder compresses the 10 Hz past sequence to ~1 Hz
(coarser-in-time, richer-in-feature) and summarises it into a context
vector. It encodes the PAST — it is not a planner — and does not touch
AutoE2E's forward contract.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch.nn as nn

from model_components.temporal_memory.one_hz_encoder import HistoryEncoder, OneHzHistoryEncoder


class _MockBackbone(nn.Module):
    """Minimal stand-in for Backbone (4 channels-first feature maps), so the
    AutoE2E integration tests below don't load pretrained weights."""

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


B = 4
INPUT_DIM = 64
HIDDEN_DIM = 96


def test_output_shape_default_config():
    encoder = HistoryEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    history = torch.randn(B, 64, INPUT_DIM)  # 64 steps @ 10 Hz = 6.4 s
    context = encoder(history)
    assert context.shape == (B, HIDDEN_DIM)
    assert torch.isfinite(context).all()


def test_temporal_compression_ratio():
    """T=64 at 10 Hz with ratio 10 must yield 6 compressed (~1 Hz) steps;
    the trailing partial window is dropped."""
    encoder = HistoryEncoder(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, subsample_ratio=10,
    )
    history = torch.randn(B, 64, INPUT_DIM)
    compressed = encoder.compress(history)
    assert compressed.shape == (B, 6, HIDDEN_DIM)
    assert encoder.compressed_length(64) == 6
    assert encoder.output_hz == pytest.approx(1.0)


def test_configurable_subsample_ratio():
    for ratio, T, expected in [(4, 64, 16), (8, 64, 8), (10, 70, 7), (1, 5, 5)]:
        encoder = HistoryEncoder(
            input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, subsample_ratio=ratio,
        )
        compressed = encoder.compress(torch.randn(B, T, INPUT_DIM))
        assert compressed.shape == (B, expected, HIDDEN_DIM), (
            f"ratio={ratio}, T={T}"
        )


def test_richer_in_feature():
    """Compressed steps must carry a larger feature dim than the input when
    hidden_dim > input_dim (coarser-in-time, richer-in-feature)."""
    encoder = HistoryEncoder(input_dim=32, hidden_dim=128, subsample_ratio=10)
    compressed = encoder.compress(torch.randn(B, 64, 32))
    assert compressed.shape[-1] == 128


def test_handles_various_history_lengths():
    encoder = HistoryEncoder(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, subsample_ratio=10,
    )
    for T in (10, 32, 64, 100, 127):
        context = encoder(torch.randn(2, T, INPUT_DIM))
        assert context.shape == (2, HIDDEN_DIM), f"T={T}"


def test_too_short_history_raises():
    encoder = HistoryEncoder(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, subsample_ratio=10,
    )
    with pytest.raises(ValueError):
        encoder(torch.randn(B, 9, INPUT_DIM))


def test_invalid_subsample_ratio_raises():
    with pytest.raises(ValueError):
        HistoryEncoder(subsample_ratio=0)


def test_gradients_flow_to_all_parameters_and_input():
    encoder = HistoryEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    history = torch.randn(B, 64, INPUT_DIM, requires_grad=True)
    context = encoder(history)
    context.pow(2).mean().backward()

    assert history.grad is not None
    assert torch.isfinite(history.grad).all()
    for name, p in encoder.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"


def test_keeps_most_recent_frames_when_length_not_multiple_of_ratio():
    """With T not a multiple of the ratio, the OLDEST ``T % ratio`` frames
    must be dropped (zero gradient) and every most-recent frame must receive
    gradient. History is ordered oldest -> most recent, so this guarantees
    the most informative (recent) 0.x s of the past are never discarded.

    Regression test: the previous implementation let the strided Conv1d drop
    the trailing window, i.e. with T=64/ratio=10 the most RECENT frames
    60-63 had exactly zero gradient.
    """
    encoder = HistoryEncoder(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, subsample_ratio=10,
    )
    T = 64  # 64 % 10 = 4 leftover frames
    remainder = T % encoder.subsample_ratio
    history = torch.randn(B, T, INPUT_DIM, requires_grad=True)
    encoder(history).pow(2).mean().backward()

    per_frame_grad = history.grad.abs().sum(dim=(0, 2))  # [T]
    # The oldest ``remainder`` frames are discarded (left-trim) ...
    assert torch.all(per_frame_grad[:remainder] == 0), (
        "Oldest leftover frames should not influence the output"
    )
    # ... and ALL remaining frames — most-recent ones included — contribute.
    assert torch.all(per_frame_grad[remainder:] > 0), (
        "Most recent frames must receive gradient (they were dropped "
        "before the fix)"
    )
    # Output length contract is unchanged: T' = T // ratio.
    assert encoder.compress(history.detach()).shape[1] == T // 10


def test_context_depends_on_history():
    torch.manual_seed(0)
    encoder = HistoryEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    encoder.eval()
    with torch.no_grad():
        ctx_a = encoder(torch.randn(B, 64, INPUT_DIM))
        ctx_b = encoder(torch.randn(B, 64, INPUT_DIM))
    assert not torch.allclose(ctx_a, ctx_b)


def test_one_hz_history_encoder_pipeline():
    encoder = OneHzHistoryEncoder(visual_dim=896, egomotion_dim=256, subsample_ratio=10)
    visual = torch.randn(B, 64, 896)
    ego = torch.randn(B, 64, 256)
    
    v_ctx, e_ctx = encoder(visual, ego)
    
    assert v_ctx.shape == (B, 896)
    assert e_ctx.shape == (B, 256)
    
    # Fallback flat input test
    v_flat = torch.randn(B, 896)
    e_flat = torch.randn(B, 256)
    v_out, e_out = encoder(v_flat, e_flat)
    assert torch.allclose(v_flat, v_out)
    assert torch.allclose(e_flat, e_out)


def test_autoe2e_consumes_one_hz_temporal_memory():
    """End-to-end data flow: AutoE2E(temporal_memory_mode='one_hz') feeds the
    [B, T, feat] history through TemporalMemory and the COMPRESSED context into
    the planner, in both train and infer. Gradient must reach the memory."""
    from unittest.mock import patch

    from model_components.auto_e2e import AutoE2E

    with patch("model_components.reactive_e2e.Backbone", _MockBackbone):
        model = AutoE2E(num_views=8, view_fusion_kwargs={"bev_h": 8, "bev_w": 8},
                        temporal_memory_mode="one_hz")

    x = torch.randn(2, 8, 3, 256, 256)
    map_input = torch.randn(2, 3, 256, 256)
    vis = torch.randn(2, 20, 896)   # [B, T, visual_dim] — sequence form
    ego = torch.randn(2, 20, 256)   # [B, T, egomotion_dim]

    # Train: forward returns the trajectory; an MSE on it must backprop into
    # TemporalMemory (its compressed context is consumed by the planner).
    traj = model(x, map_input, vis, ego, mode="train")
    assert traj.shape == (2, 128) and torch.isfinite(traj).all()
    traj.pow(2).mean().backward()
    assert any(p.grad is not None
               for p in model.Reactive_E2E.TemporalMemory.parameters()), \
        "compressed temporal context must be consumed by the planner (grad must reach TemporalMemory)"

    # Infer: trajectory only.
    traj2 = model(x, map_input, vis, ego, mode="infer")
    assert traj2.shape == (2, 128)


def test_autoe2e_default_no_memory_passthrough():
    """Default temporal_memory_mode='no_memory' keeps the flat [B, feat] path
    working — default AutoE2E behaviour is unchanged."""
    from unittest.mock import patch

    from model_components.auto_e2e import AutoE2E
    from model_components.temporal_memory import NoMemory

    with patch("model_components.reactive_e2e.Backbone", _MockBackbone):
        model = AutoE2E(num_views=8,
                        view_fusion_kwargs={"bev_h": 8, "bev_w": 8})  # default no_memory
    assert isinstance(model.Reactive_E2E.TemporalMemory, NoMemory)

    x = torch.randn(2, 8, 3, 256, 256)
    map_input = torch.randn(2, 3, 256, 256)
    vis = torch.randn(2, 896)       # flat history (default contract)
    ego = torch.randn(2, 256)
    traj = model(x, map_input, vis, ego, mode="infer")
    assert traj.shape == (2, 128)
