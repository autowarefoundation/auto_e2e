import torch
import torch.nn as nn
from .view_fusion import build_view_fusion


class FeatureFusion(nn.Module):
    """Multi-scale feature fusion + cross-view unification.

    Two-stage process:
      1. Pool and concatenate multi-scale backbone features (per-view)
      2. Unify across camera views using the selected fusion strategy
    """

    def __init__(self, num_views=8, backbone_channels=1440, embed_dim=256,
                 fusion_mode="concat", image_feature_size=8, view_fusion_kwargs=None):
        super(FeatureFusion, self).__init__()

        # Per-view pooled image-feature resolution (used as values for view fusion).
        # This is independent of the BEV grid size; BEV fusion reprojects these
        # values onto its own (bev_h, bev_w) grid.
        self.image_feature_size = image_feature_size
        self.pool = nn.AdaptiveMaxPool2d(image_feature_size)

        # Channel reduction to achieve correct embedding dimension
        self.channel_proj = nn.Sequential(
            nn.Conv2d(backbone_channels, embed_dim, kernel_size=1),
            nn.GELU()
        )

        # View fusion strategy (pluggable). Extra kwargs (bev_h, bev_w, pc_range,
        # image_size, ...) are forwarded to the selected fusion module.
        self.view_fusion = build_view_fusion(
            fusion_mode, num_views, embed_dim, **(view_fusion_kwargs or {})
        )

    def forward(self, features, B, V, camera_params=None,
                projection=None, geometry_type=None):
        # features: list of 4 multi-scale feature maps from backbone (channels-first)
        for i in range(0, len(features)):
            features[i] = self.pool(features[i])

        # Concatenate scales along channels at image_feature_size resolution.
        fused_per_view = torch.cat(features, dim=1)
        fused_per_view = self.channel_proj(fused_per_view)

        # Unify across views. BEV fusion output spatial size is (bev_h, bev_w).
        # Geometry (camera_params / projection operator / geometry_type) is
        # passed straight through — FeatureFusion does not interpret it.
        fused = self.view_fusion(
            fused_per_view, B, V,
            camera_params=camera_params,
            projection=projection,
            geometry_type=geometry_type,
        )

        return fused
