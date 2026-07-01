"""Shared test fixtures with mock backbone for fast unit tests.

The real backbone (SwinV2/ConvNeXt) dominates test time (~80% per forward pass)
but is never the subject under test — it is pretrained and frozen. We replace it
with a lightweight stub that produces tensors of the correct shape, reducing
per-forward cost from ~50ms to <1ms while still exercising View Fusion,
GRUPlanner, and FutureState end-to-end.

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

    Uses adaptive pooling after each conv to guarantee correct spatial dims
    regardless of input resolution, keeping gradients flowing for
    gradient-flow tests.

    NOTE: produces 4 feature maps, matching Swin/ConvNeXt backbones.
    ResNet50 in timm exposes 5 feature stages, so this mock does NOT
    cover that backbone — tests exercising ResNet50 must use a real
    backbone or a separate mock.
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

        return [s0, s1, s2, s3]


class MockBackbone(nn.Module):
    """Drop-in replacement for model_components.backbone.Backbone."""

    def __init__(self, backbone="swin_v2_tiny", is_pretrained=True, **kwargs):
        super().__init__()
        self.backbone = MockBackboneModel()
        self.backbone_channels = 1440

    def forward(self, image):
        return self.backbone(image)


def _build_model_with_mock_backbone(num_views, fusion_mode="bev", device=None,
                                    num_timesteps=64, map_fusion_mode="residual",
                                    planner_mode="bezier", planner_kwargs=None,
                                    **model_kwargs):
    """Construct AutoE2E with the mock backbone injected.

    Post-refactor (#86): the model is ``AutoE2E`` -> ``Reactive_E2E`` and the
    image backbone now lives in ``reactive_e2e``; fusion is always BEV
    (concat/cross_attn were removed) and GRU was dropped. ``fusion_mode`` is
    accepted for backward-compatibility with existing call sites but ignored
    (always BEV at a small 8x8 grid). Extra ``model_kwargs`` are forwarded.
    """
    from unittest.mock import patch
    from model_components.auto_e2e import AutoE2E

    with patch('model_components.reactive_e2e.Backbone', MockBackbone):
        model = AutoE2E(
            num_views=num_views,
            view_fusion_kwargs={"bev_h": 8, "bev_w": 8},
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
    """Session-scoped model with mock backbone — shared across all tests.

    Post-refactor only BEV fusion exists. Built once to avoid redundant
    construction overhead; gradient state is reset before each test via the
    autouse fixture below.
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
