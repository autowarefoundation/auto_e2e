"""Unit tests for the JEPA target encoder (frozen / EMA, stop-gradient).

Verifies the collapse-safety guarantees of the target branch and that it
plugs into the already-merged ``FutureState`` + ``FeatureReconstructionLoss``:
  - the target encoder never carries gradient (frozen params, detached output),
  - "frozen" mode is a stable deep copy independent of the online encoder,
  - "ema" mode moves the target toward the online encoder on ``update()``,
  - ``compute_jepa_loss`` is differentiable w.r.t. the prediction ONLY,
  - end-to-end with the real FutureState world-model head.
"""

import pytest
import torch
import torch.nn as nn

from model_components.future_state import FutureState
from model_components.jepa_target_encoder import (
    JepaTargetEncoder,
    compute_jepa_loss,
)
from model_components.losses.feature_reconstruction_loss import (
    FeatureReconstructionLoss,
)

EMBED_DIM = 256
NUM_FUTURE = 4


class _TinyEncoder(nn.Module):
    """Maps a feature map [B, C, H, W] -> [B, C, H, W] (stand-in for backbone)."""

    def __init__(self, c=EMBED_DIM):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        return self.bn(self.conv(x))


def _future_obs(batch, device, h=8, w=8):
    return [torch.randn(batch, EMBED_DIM, h, w, device=device)
            for _ in range(NUM_FUTURE)]


def test_invalid_mode_and_decay():
    enc = _TinyEncoder()
    with pytest.raises(ValueError, match="mode must be"):
        JepaTargetEncoder(enc, mode="bogus")
    with pytest.raises(ValueError, match="ema_decay"):
        JepaTargetEncoder(enc, mode="ema", ema_decay=1.5)


def test_target_params_are_frozen(device):
    target = JepaTargetEncoder(_TinyEncoder(), mode="frozen").to(device)
    assert all(not p.requires_grad for p in target.parameters())


def test_forward_output_is_detached(device):
    target = JepaTargetEncoder(_TinyEncoder(), mode="ema").to(device)
    outs = target(_future_obs(2, device))
    assert len(outs) == NUM_FUTURE
    for o in outs:
        assert not o.requires_grad
        assert o.grad_fn is None
        assert o.shape == (2, EMBED_DIM, 8, 8)


def test_frozen_is_independent_deep_copy(device):
    online = _TinyEncoder().to(device)
    target = JepaTargetEncoder(online, mode="frozen").to(device)
    before = [p.clone() for p in target.encoder.parameters()]

    # Mutate the ONLINE encoder; the frozen target must not change.
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
    target.update(online)  # no-op in frozen mode

    for b, p in zip(before, target.encoder.parameters()):
        assert torch.equal(b, p)


def test_ema_update_moves_target_toward_online(device):
    torch.manual_seed(0)
    online = _TinyEncoder().to(device)
    target = JepaTargetEncoder(online, mode="ema", ema_decay=0.9).to(device)

    # Push the online encoder far away, then EMA-update once.
    with torch.no_grad():
        for p in online.parameters():
            p.add_(10.0)

    t_before = [p.clone() for p in target.encoder.parameters()]
    target.update(online)

    for tb, t_after, o_p in zip(t_before, target.encoder.parameters(),
                                online.parameters()):
        moved = (t_after - tb).abs().sum().item()
        assert moved > 0.0, "ema target did not move"
        # decay=0.9 -> target should move ~10% of the way, not all the way.
        assert not torch.allclose(t_after, o_p), "target jumped fully to online"
        expected = 0.9 * tb + 0.1 * o_p
        assert torch.allclose(t_after, expected, atol=1e-5)


def test_compute_jepa_loss_zero_when_prediction_matches_target(device):
    target = JepaTargetEncoder(_TinyEncoder(), mode="frozen").to(device)
    loss_fn = FeatureReconstructionLoss(num_future_steps=NUM_FUTURE).to(device)
    obs = _future_obs(2, device)
    targets = target(obs)
    predicted = [t.clone().requires_grad_(True) for t in targets]
    loss = compute_jepa_loss(predicted, obs, target, loss_fn, weight=1.0)
    assert loss.ndim == 0
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_compute_jepa_loss_weight_scales_linearly(device):
    target = JepaTargetEncoder(_TinyEncoder(), mode="frozen").to(device)
    loss_fn = FeatureReconstructionLoss(num_future_steps=NUM_FUTURE).to(device)
    obs = _future_obs(3, device)
    predicted = [torch.randn(3, EMBED_DIM, 8, 8, device=device)
                 for _ in range(NUM_FUTURE)]
    l1 = compute_jepa_loss(predicted, obs, target, loss_fn, weight=1.0)
    l2 = compute_jepa_loss(predicted, obs, target, loss_fn, weight=0.25)
    assert l2.item() == pytest.approx(0.25 * l1.item(), rel=1e-5)


def test_gradient_flows_to_prediction_only(device):
    target = JepaTargetEncoder(_TinyEncoder(), mode="ema").to(device)
    loss_fn = FeatureReconstructionLoss(num_future_steps=NUM_FUTURE).to(device)
    obs = _future_obs(2, device)
    predicted = [torch.randn(2, EMBED_DIM, 8, 8, device=device,
                             requires_grad=True) for _ in range(NUM_FUTURE)]

    loss = compute_jepa_loss(predicted, obs, target, loss_fn, weight=0.5)
    loss.backward()

    assert all(p.grad is not None for p in predicted), \
        "prediction must receive gradient"
    assert all(p.grad is None for p in target.parameters()), \
        "target encoder must NOT receive gradient (stop-gradient)"


def test_end_to_end_with_future_state(device):
    """FutureState (predictor) + JepaTargetEncoder (targets) + recon loss.

    Mirrors the intended training step: the world-model head predicts future
    features, targets come from the frozen/EMA encoder on future observations,
    and the JEPA loss back-props into FutureState but never into the target."""
    torch.manual_seed(0)
    future_state = FutureState(embed_dim=EMBED_DIM,
                               ego_hidden_dim=EMBED_DIM).to(device)
    target = JepaTargetEncoder(_TinyEncoder(), mode="ema").to(device)
    loss_fn = FeatureReconstructionLoss(num_future_steps=NUM_FUTURE).to(device)

    fused = torch.randn(2, EMBED_DIM, 8, 8, device=device)
    ego_hidden = torch.randn(2, EMBED_DIM, device=device)
    predicted = future_state(fused, ego_hidden)
    assert len(predicted) == NUM_FUTURE

    future_obs = _future_obs(2, device)
    loss = compute_jepa_loss(predicted, future_obs, target, loss_fn, weight=0.1)
    assert loss.ndim == 0 and torch.isfinite(loss)

    loss.backward()
    assert any(p.grad is not None for p in future_state.parameters()), \
        "FutureState must receive gradient from the JEPA loss"
    assert all(p.grad is None for p in target.parameters()), \
        "target encoder must stay gradient-free"
