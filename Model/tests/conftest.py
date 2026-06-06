"""Shared test fixtures with mock backbone for fast unit tests.

The real backbone (SwinV2/ConvNeXt) dominates test time (~80% per forward pass)
but is never the subject under test — it is pretrained and frozen. We replace it
with a lightweight stub that produces tensors of the correct shape, reducing
per-forward cost from ~50ms to <1ms while still exercising View Fusion,
DrivingPolicy, and FutureState end-to-end.

Full-backbone integration tests are available via the 'integration' marker.
"""

import pytest
import torch
import torch.nn as nn


class MockBackboneModel(nn.Module):
    """Minimal Conv backbone producing 4 feature maps matching SwinV2 output shapes.

    SwinV2 Tiny at 256x256 input produces (channels-last):
      Stage 0: [B*V, 64, 64, 96]
      Stage 1: [B*V, 32, 32, 192]
      Stage 2: [B*V, 16, 16, 384]
      Stage 3: [B*V,  8,  8, 768]

    Uses adaptive pooling after each conv to guarantee correct spatial dims
    regardless of input resolution, keeping gradients flowing for
    gradient-flow tests.
    """

    def __init__(self):
        super().__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(64),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(32),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(16),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(384, 768, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(8),
        )

    def forward(self, x):
        s0 = self.stage0(x)   # [B*V, 96, 64, 64]
        s1 = self.stage1(s0)  # [B*V, 192, 32, 32]
        s2 = self.stage2(s1)  # [B*V, 384, 16, 16]
        s3 = self.stage3(s2)  # [B*V, 768, 8, 8]

        # Convert to channels-last to match SwinV2 output format
        return [
            s0.permute(0, 2, 3, 1),  # [B*V, 64, 64, 96]
            s1.permute(0, 2, 3, 1),  # [B*V, 32, 32, 192]
            s2.permute(0, 2, 3, 1),  # [B*V, 16, 16, 384]
            s3.permute(0, 2, 3, 1),  # [B*V,  8,  8, 768]
        ]


class MockBackbone(nn.Module):
    """Drop-in replacement for model_components.backbone.Backbone."""

    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = MockBackboneModel()

    def forward(self, image):
        return self.backbone(image)


def _build_model_with_mock_backbone(num_views, fusion_mode, device):
    """Construct AutoE2E with the mock backbone injected.

    Patches Backbone at the module level during construction to avoid
    loading pretrained weights entirely.
    """
    from unittest.mock import patch
    from model_components.auto_e2e import AutoE2E

    with patch('model_components.auto_e2e.Backbone', MockBackbone):
        model = AutoE2E(num_views=num_views, fusion_mode=fusion_mode)
    return model.to(device)


@pytest.fixture(scope="session")
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def build_mock_model():
    """Factory fixture for building models with mock backbone."""
    return _build_model_with_mock_backbone


@pytest.fixture(scope="session", params=["concat", "cross_attn", "bev"])
def model(request, device):
    """Session-scoped model with mock backbone — shared across all tests.

    Built once per fusion mode to avoid redundant 1.5s construction overhead
    per test. Gradient state is reset before each test via the autouse
    _reset_model_grads fixture below.
    """
    return _build_model_with_mock_backbone(
        num_views=8, fusion_mode=request.param, device=device
    )


@pytest.fixture(autouse=True)
def _reset_model_state(request):
    """Reset session-scoped model state between tests."""
    yield
    if "model" in request.fixturenames:
        model = request.getfixturevalue("model")
        model.zero_grad(set_to_none=True)
        model.train()


@pytest.fixture(params=["concat", "cross_attn", "bev"])
def full_model(request, device):
    """Full model with real backbone — use only for integration tests."""
    from model_components.auto_e2e import AutoE2E

    try:
        model = AutoE2E(num_views=8, fusion_mode=request.param)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"Pretrained weights unavailable: {e}")
    return model.to(device)
