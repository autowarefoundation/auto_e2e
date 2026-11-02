"""Shared test fixtures with mock backbone for fast unit tests.

The real backbone (SwinV2/ConvNeXt) dominates test time but is never the
subject under test, so it is replaced with a lightweight stub that produces
tensors of the correct shape while still exercising view fusion and the
trajectory planner end-to-end.

Tests use a small BEV grid (8x8) for the ``bev`` fusion mode; the production
default (450x300) is verified separately via configuration tests.

Full-backbone integration tests are available via the 'integration' marker.
"""

import pytest
import torch
import torch.nn as nn


class MockBackboneModel(nn.Module):
    """Minimal Conv backbone producing 4 feature maps in channels-first format.

    Matches the output of Backbone.forward() which returns channels-first:
      Stage 0: [B*V, 96, 64, 64]
      Stage 1: [B*V, 192, 32, 32]
      Stage 2: [B*V, 384, 16, 16]
      Stage 3: [B*V, 768,  8,  8]

    Each stage pools to its spatial size first and then applies a 1x1 conv, so
    the dims hold for any input resolution and gradients still flow. Pooling
    before the conv keeps the mock cheap: a 3x3 conv on the full 256x256 input
    made the mock the main cost of the train-step tests on CI.

    Only the 4-stage Swin/ConvNeXt layout is covered; timm's ResNet50 exposes
    5 stages and needs a real backbone or a separate mock.
    """

    def __init__(self):
        super().__init__()
        self.stage0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(64),
            nn.Conv2d(3, 96, kernel_size=1),
        )
        self.stage1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(32),
            nn.Conv2d(96, 192, kernel_size=1),
        )
        self.stage2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(192, 384, kernel_size=1),
        )
        self.stage3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(384, 768, kernel_size=1),
        )

    def forward(self, x):
        s0 = self.stage0(x)   # [B*V, 96, 64, 64]
        s1 = self.stage1(s0)  # [B*V, 192, 32, 32]
        s2 = self.stage2(s1)  # [B*V, 384, 16, 16]
        s3 = self.stage3(s2)  # [B*V, 768, 8, 8]

        return [s0, s1, s2, s3]


class MockBackbone(nn.Module):
    """Drop-in replacement for model_components.backbone.Backbone."""

    def __init__(self, backbone="swin_v2_tiny", is_pretrained=True, **kwargs):
        super().__init__()
        self.backbone = MockBackboneModel()
        # Per-stage channels (matches MockBackboneModel's 4 stages). The World
        # Model derives its JEPA feature_channels from feature_channels[-1], so
        # the mock must expose it just like the real Backbone wrapper.
        self.feature_channels = [96, 192, 384, 768]
        self.backbone_channels = sum(self.feature_channels)

    def forward(self, image):
        return self.backbone(image)


def _build_model_with_mock_backbone(num_views, fusion_mode="bev", device=None,
                                    num_timesteps=64, map_fusion_mode="residual",
                                    planner_mode="bezier", planner_kwargs=None,
                                    **model_kwargs):
    """Construct AutoE2E with the mock backbone injected.

    The backbone is patched in ``reactive_e2e``. Fusion is always BEV at a small
    8x8 grid; ``fusion_mode`` is accepted for call-site compatibility but
    ignored. Extra ``model_kwargs`` are forwarded to ``AutoE2E``.
    """
    from unittest.mock import patch
    from model_components.auto_e2e import AutoE2E

    view_fusion_kwargs = {"bev_h": 8, "bev_w": 8}
    view_fusion_kwargs.update(
        model_kwargs.pop("view_fusion_kwargs", {})
    )
    with patch('model_components.reactive_e2e.Backbone', MockBackbone):
        model = AutoE2E(
            num_views=num_views,
            view_fusion_kwargs=view_fusion_kwargs,
            num_timesteps=num_timesteps,
            planner_mode=planner_mode,
            planner_kwargs=planner_kwargs,
            map_fusion_mode=map_fusion_mode,
            **model_kwargs,
        )
    return model.to(device)


@pytest.fixture(scope="session")
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def build_mock_model():
    """Factory fixture for building models with mock backbone."""
    return _build_model_with_mock_backbone


@pytest.fixture(scope="session", params=["bev"])
def model(request, device):
    """Session-scoped model with mock backbone, shared across all tests.

    Gradient and train-mode state is reset after each test by the autouse
    fixture below.
    """
    return _build_model_with_mock_backbone(
        num_views=7, fusion_mode=request.param, device=device
    )


@pytest.fixture(autouse=True)
def _reset_model_state(request):
    """Reset session-scoped model state between tests."""
    yield
    if "model" in request.fixturenames:
        model = request.getfixturevalue("model")
        model.zero_grad(set_to_none=True)
        model.train()


@pytest.fixture(params=["bev"])
def full_model(request, device):
    """Full model with real backbone — use only for integration tests."""
    from model_components.auto_e2e import AutoE2E

    try:
        model = AutoE2E(
            num_views=7, view_fusion_kwargs={"bev_h": 8, "bev_w": 8},
        )
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"Pretrained weights unavailable: {e}")
    return model.to(device)
