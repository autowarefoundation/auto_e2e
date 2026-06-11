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

from model_components.history_encoder import HistoryEncoder

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


def test_context_depends_on_history():
    torch.manual_seed(0)
    encoder = HistoryEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    encoder.eval()
    with torch.no_grad():
        ctx_a = encoder(torch.randn(B, 64, INPUT_DIM))
        ctx_b = encoder(torch.randn(B, 64, INPUT_DIM))
    assert not torch.allclose(ctx_a, ctx_b)
