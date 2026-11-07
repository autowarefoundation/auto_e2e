"""``normalize_egomotion`` must scale the right columns without touching its input.

The failure this guards against is silent. Scaling in place mutates the caller's
batch, and the scaling is not idempotent — a second forward over the same batch
divides speed by 33 again. Nothing raises; the run produces a plausible number
from an input that is two orders of magnitude too small. The stage 2 trainer
runs the distorted and the clean teacher pass on the same batch and would hit
exactly this.
"""

from __future__ import annotations

import pytest
import torch

from model_components.auto_e2e import EGOMOTION_SCALE, normalize_egomotion

_SPEED_SCALE = 1.0 / EGOMOTION_SCALE[0]
_ACCELERATION_SCALE = 1.0 / EGOMOTION_SCALE[1]

# 64 timesteps x [speed, acceleration, yaw_rate, curvature] — the packed
# contract every dataset parser emits.
_WIDTH = 256


def _history(batch_size: int = 4) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch_size, _WIDTH)


def test_does_not_mutate_its_input():
    history = _history()
    before = history.clone()

    normalize_egomotion(history)

    assert torch.equal(history, before)


def test_is_idempotent_in_effect():
    """Normalizing an already-normalized *copy* must not compound.

    Purity is what buys this: because the function returns a new tensor, the
    caller's history can be normalized any number of times and each call sees
    the same raw input.
    """
    history = _history()

    once = normalize_egomotion(history)
    twice = normalize_egomotion(history)

    assert torch.equal(once, twice)


def test_scales_speed_and_acceleration_only():
    history = _history()

    scaled = normalize_egomotion(history)

    raw = history.reshape(-1, 64, 4)
    out = scaled.reshape(-1, 64, 4)
    assert torch.allclose(out[..., 0], raw[..., 0] / _SPEED_SCALE)
    assert torch.allclose(out[..., 1], raw[..., 1] / _ACCELERATION_SCALE)
    # yaw_rate and curvature are bounded already and must pass through.
    assert torch.equal(out[..., 2], raw[..., 2])
    assert torch.equal(out[..., 3], raw[..., 3])


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_scales_every_sample_in_the_batch(batch_size):
    history = _history(batch_size)

    scaled = normalize_egomotion(history).reshape(batch_size, 64, 4)

    expected = history.reshape(batch_size, 64, 4)[..., 0] / _SPEED_SCALE
    assert torch.allclose(scaled[..., 0], expected)


def test_preserves_shape_dtype_and_device():
    history = _history().to(torch.float64)

    scaled = normalize_egomotion(history)

    assert scaled.shape == history.shape
    assert scaled.dtype == history.dtype
    assert scaled.device == history.device


def test_gradients_flow_through():
    history = _history().requires_grad_(True)

    normalize_egomotion(history).sum().backward()

    assert history.grad is not None
    # d/dx of x/33 summed over the 64 speed entries of each row.
    assert torch.allclose(
        history.grad.reshape(-1, 64, 4)[..., 0],
        torch.full((history.shape[0], 64), 1.0 / _SPEED_SCALE),
    )
