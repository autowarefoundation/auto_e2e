import torch
import sys
sys.path.append('..')

from model_components.backbone import Backbone


class _StubBackboneWithFeatureInfo(torch.nn.Module):
    """Channels-first backbone exposing timm-style feature_info."""

    def __init__(self):
        super().__init__()
        self.stage0 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.stage1 = torch.nn.Conv2d(32, 48, 3, stride=2, padding=1)
        self.stage2 = torch.nn.Conv2d(48, 64, 3, stride=2, padding=1)
        self.feature_info = [{"num_chs": 32}, {"num_chs": 48}, {"num_chs": 64}]

    def forward(self, x):
        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        return [s0, s1, s2]


class _StubBackboneNoFeatureInfo(torch.nn.Module):
    """Channels-first backbone with NO feature_info (probe fallback path)."""

    def __init__(self):
        super().__init__()
        self.stage0 = torch.nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.stage1 = torch.nn.Conv2d(24, 56, 3, stride=2, padding=1)
        self.stage2 = torch.nn.Conv2d(56, 112, 3, stride=2, padding=1)

    def forward(self, x):
        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        return [s0, s1, s2]


class _StubBackboneSwinLike(torch.nn.Module):
    """Channels-last backbone (B, H, W, C) — exercises permute branch."""

    def __init__(self):
        super().__init__()
        self.stage0 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.stage1 = torch.nn.Conv2d(32, 48, 3, stride=2, padding=1)
        self.feature_info = [{"num_chs": 32}, {"num_chs": 48}]

    def forward(self, x):
        s0_cf = self.stage0(x)                                  # [B, 32, H, W]
        s1_cf = self.stage1(s0_cf)                              # [B, 48, H, W]
        s0 = s0_cf.permute(0, 2, 3, 1).contiguous()             # [B, H, W, 32]
        s1 = s1_cf.permute(0, 2, 3, 1).contiguous()             # [B, H, W, 48]
        return [s0, s1]


class TestBackboneChannelDiscovery:
    """Cover the backbone_channels discovery + layout-detection in Backbone."""

    def _make_backbone(self, monkeypatch, stub_module):
        # Patch the registry call so build_backbone returns our stub.
        monkeypatch.setattr(
            "model_components.backbone.build_backbone",
            lambda *a, **kw: stub_module,
        )
        return Backbone(backbone="stub", is_pretrained=False)

    def test_feature_info_path_sums_channels(self, monkeypatch):
        bb = self._make_backbone(monkeypatch, _StubBackboneWithFeatureInfo())
        assert bb.backbone_channels == 32 + 48 + 64

    def test_probe_fallback_when_feature_info_missing(self, monkeypatch):
        bb = self._make_backbone(monkeypatch, _StubBackboneNoFeatureInfo())
        # No feature_info — channels recovered via probing.
        assert bb.backbone_channels == 24 + 56 + 112

    def test_feature_info_channels_match_forward_output(self, monkeypatch, device):
        """sum(feature_info channels) must equal the actual concat-channel dim
        of the forward output."""
        bb = self._make_backbone(monkeypatch, _StubBackboneWithFeatureInfo()).to(device)
        x = torch.randn(2, 3, 32, 32, device=device)
        feats = bb(x)
        total_c = sum(f.shape[1] for f in feats)
        assert total_c == bb.backbone_channels

    def test_probe_channels_match_forward_output(self, monkeypatch, device):
        bb = self._make_backbone(monkeypatch, _StubBackboneNoFeatureInfo()).to(device)
        x = torch.randn(2, 3, 32, 32, device=device)
        feats = bb(x)
        total_c = sum(f.shape[1] for f in feats)
        assert total_c == bb.backbone_channels

    def test_channels_last_backbone_is_permuted(self, monkeypatch, device):
        """Channels-last (B, H, W, C) output must be permuted to (B, C, H, W)
        based on tensor shape, NOT on the backbone name."""
        bb = self._make_backbone(monkeypatch, _StubBackboneSwinLike()).to(device)
        x = torch.randn(2, 3, 32, 32, device=device)
        feats = bb(x)
        # After Backbone.forward, every feature must be channels-first with the
        # expected channel count at dim 1.
        assert feats[0].shape[1] == 32
        assert feats[1].shape[1] == 48

    def test_channels_first_backbone_not_permuted(self, monkeypatch, device):
        bb = self._make_backbone(monkeypatch, _StubBackboneWithFeatureInfo()).to(device)
        x = torch.randn(2, 3, 32, 32, device=device)
        feats = bb(x)
        for f, expected in zip(feats, [32, 48, 64]):
            assert f.shape[1] == expected
