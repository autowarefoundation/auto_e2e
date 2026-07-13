"""The BEV as a spatial context source for HorizonReasoningHead (#121).

What these pin down, and why each one matters:

1. Without ``bev_context_dim`` the head is byte-identical to today. The BEV path
   must be strictly additive — a run that does not ask for it must not change.
2. The head actually USES location. This is the whole hypothesis: the planner
   destroys *where* with ``bev_features.mean(dim=(2, 3))``, so a head fed the
   unpooled grid must distinguish two BEVs that share a mean but differ in layout.
   If it cannot, the extra tokens are decoration and the ablation is pointless.
3. The two switches are independent: ``bev_detach`` controls whether the reasoning
   loss reshapes the SHARED trunk, separately from whether the head sees the BEV.
   The repo has twice found auxiliary gradient into the trunk hurts at low data,
   so a result must be attributable to one or the other.
"""

import pytest
import torch

from model_components.reasoning.horizon_reasoning_head import (
    BevContextTokenizer,
    HorizonReasoningHead,
)

B, C, H, W = 2, 256, 45, 30
HID = 256


def _head(**kw):
    return HorizonReasoningHead(hidden_dim=HID, **kw)


def test_tokenizer_shape_and_grid_is_the_ablation_knob():
    for grid in [(8, 8), (16, 16)]:
        tok = BevContextTokenizer(C, HID, grid=grid)
        out = tok(torch.randn(B, C, H, W))
        assert out.shape == (B, grid[0] * grid[1], HID)


def test_head_without_bev_ignores_the_bev_argument():
    """Strictly additive: a head built without bev_context_dim is unchanged."""
    head = _head().eval()
    vh, ego = torch.randn(B, 896), torch.randn(B, 256)
    bev = torch.randn(B, C, H, W)

    assert head.bev_tokenizer is None
    with torch.no_grad():
        a = head(vh, ego).reasoning_latent
        b = head(vh, ego, bev_context=bev).reasoning_latent
    torch.testing.assert_close(a, b)


def test_context_token_count_grows_by_the_grid():
    head = _head(bev_context_dim=C, bev_grid=(16, 16))
    vh, ego = torch.randn(B, 896), torch.randn(B, 256)
    bev = torch.randn(B, C, H, W)

    plain = head._context_tokens(vh, ego, None, None, None)
    with_bev = head._context_tokens(vh, ego, None, None, bev)
    assert plain.shape[1] == 2                      # visual + ego
    assert with_bev.shape[1] == 2 + 16 * 16         # + the spatial grid


def test_head_sees_WHERE_not_just_what():
    """The hypothesis, as a test.

    Two BEVs with the SAME per-channel mean but a different spatial layout: a
    mean-pooled context cannot tell them apart (that is exactly what the bezier
    planner throws away). The head must.
    """
    head = _head(bev_context_dim=C, bev_grid=(16, 16)).eval()
    vh, ego = torch.randn(1, 896), torch.randn(1, 256)

    left = torch.zeros(1, C, H, W)
    left[:, :, :, : W // 2] = 1.0        # mass on the left half
    right = torch.zeros(1, C, H, W)
    right[:, :, :, W // 2:] = 1.0        # mass on the right half

    # Precondition: mean-pooling really is blind to the difference.
    torch.testing.assert_close(left.mean(dim=(2, 3)), right.mean(dim=(2, 3)))

    with torch.no_grad():
        a = head(vh, ego, bev_context=left).reasoning_latent
        b = head(vh, ego, bev_context=right).reasoning_latent
    assert not torch.allclose(a, b, atol=1e-6), (
        "the head produced the same latent for two spatially different BEVs — "
        "the spatial tokens are not carrying location"
    )


def test_gradient_reaches_the_bev_when_not_detached():
    head = _head(bev_context_dim=C, bev_grid=(8, 8))
    vh, ego = torch.randn(B, 896), torch.randn(B, 256)
    bev = torch.randn(B, C, H, W, requires_grad=True)

    head(vh, ego, bev_context=bev).reasoning_latent.sum().backward()
    assert bev.grad is not None and bev.grad.abs().sum() > 0


@pytest.mark.parametrize("detach", [True, False])
def test_detach_switch_controls_the_shared_trunk_gradient(detach):
    """The second switch: does the reasoning loss RESHAPE the shared BEV?"""
    head = _head(bev_context_dim=C, bev_grid=(8, 8))
    vh, ego = torch.randn(B, 896), torch.randn(B, 256)

    trunk = torch.randn(B, C, H, W, requires_grad=True)
    fed = trunk.detach() if detach else trunk

    head(vh, ego, bev_context=fed).reasoning_latent.sum().backward()
    if detach:
        assert trunk.grad is None, "see-only arm leaked gradient into the trunk"
    else:
        assert trunk.grad is not None and trunk.grad.abs().sum() > 0
