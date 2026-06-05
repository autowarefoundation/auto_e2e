import torch
import torch.nn as nn
from .view_fusion import build_view_fusion


class FeatureFusion(nn.Module):
    """Multi-scale feature fusion + cross-view unification.

    Two-stage process:
      1. Pool and concatenate multi-scale backbone features (per-view)
      2. Unify across camera views using the selected fusion strategy
    """

    def __init__(self, num_views=8, fusion_mode="concat"):
        super(FeatureFusion, self).__init__()

        # Adaptive pooling to achieve 8x8 resolution
        self.pool = nn.AdaptiveMaxPool2d(8)

        # Channel count after concatenating all 4 SwinV2 stages:
        # 96 + 192 + 384 + 768 = 1440
        embed_dim = 1440

        # View fusion strategy (pluggable)
        self.view_fusion = build_view_fusion(fusion_mode, num_views, embed_dim)

    def forward(self, features, B, V, backbone="swin_v2_tiny", camera_params=None):
        # features: list of 4 multi-scale feature maps from backbone
        # Each has shape [B*V, H, W, C] (SwinV2 output format)

        f0 = self.pool(features[0])
        f1 = self.pool(features[1])
        f2 = self.pool(features[2])
        f3 = features[3]

        # Permuting features to B*V, C, H, W for SwinV2
        # with pooling
        if(backbone == "swin_v2_tiny"):
            f0 = self.pool(features[0].permute(0, 3, 1, 2))
            f1 = self.pool(features[1].permute(0, 3, 1, 2))
            f2 = self.pool(features[2].permute(0, 3, 1, 2))
            f3 = features[3].permute(0, 3, 1, 2)
        
        # Pooling features from ConvNextV2 without permutation
        if(backbone == "conv_next_v2_tiny"):
            f0 = self.pool(features[0])
            f1 = self.pool(features[1])
            f2 = self.pool(features[2])
            f3 = features[3]


        # Concatenate scales along channels: [B*V, 1440, 8, 8]
        fused_per_view = torch.cat((f0, f1, f2, f3), dim=1)

        # Unify across views: [B*V, 1440, 8, 8] → [B, 1440, 8, 8]
        # camera_params is passed through for BEV fusion; ignored by other modes
        fused = self.view_fusion(fused_per_view, B, V, camera_params=camera_params)

        return fused
