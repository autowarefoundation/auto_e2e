import torch
import torch.nn as nn
from .view_fusion import build_view_fusion


class FeatureFusion(nn.Module):
    """Multi-scale feature fusion + cross-view unification.

    Two-stage process:
      1. Pool and concatenate multi-scale backbone features (per-view)
      2. Unify across camera views using the selected fusion strategy
    """

    def __init__(self, num_views=8, backbone_channels=1440, embed_dim=256, fusion_mode="concat"):
        super(FeatureFusion, self).__init__()

        # Adaptive pooling to achieve 8x8 resolution
        self.pool = nn.AdaptiveMaxPool2d(8)

        # Channel reduction to achieve correct embedding dimension
        self.channel_proj = nn.Sequential(
            nn.Conv2d(backbone_channels, embed_dim, kernel_size=1),
            nn.GELU()
        )

        # View fusion strategy (pluggable)
        self.view_fusion = build_view_fusion(fusion_mode, num_views, embed_dim)

    def forward(self, features, B, V, camera_params=None):
        # features: list of 4 multi-scale feature maps from backbone
        # Each has shape [B*V, H, W, C] (SwinV2 output format)

        for i in range(0, len(features)):
            features[i] = self.pool(features[i])

        # Concatenate scales along channels: [B*V, 1440, 8, 8]
        fused_per_view = torch.cat(features, dim=1)    # [B*V, backbone_channels, 8, 8]
        fused_per_view = self.channel_proj(fused_per_view)     # [B*V, 256, 8, 8]
        
        # Unify across views: [B*V, 256, 8, 8] → [B, 256, 8, 8]
        # camera_params is passed through for BEV fusion; ignored by other modes
        fused = self.view_fusion(fused_per_view, B, V, camera_params=camera_params)

        return fused
