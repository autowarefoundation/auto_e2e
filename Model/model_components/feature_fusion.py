from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from .feature_pyramid import BEVFormerFeaturePyramid
from .view_fusion import build_view_fusion
from .view_fusion.bevformer_v2_t1 import BEVFormerV2T1ViewFusion


class FeatureFusion(nn.Module):
    """Multi-scale feature fusion + cross-view unification.

    Two-stage process:
      1. Pool and concatenate multi-scale backbone features (per-view)
      2. Unify across camera views using the selected fusion strategy
    """

    def __init__(self, num_views=8,
                 backbone_channels: Sequence[int] = (96, 192, 384, 768),
                 embed_dim=256,
                 fusion_mode="bev", image_feature_size=8, view_fusion_kwargs=None):
        super(FeatureFusion, self).__init__()

        channels = tuple(int(value) for value in backbone_channels)
        if not channels or any(value <= 0 for value in channels):
            raise ValueError(
                "backbone_channels must contain positive stage widths"
            )
        if image_feature_size <= 0:
            raise ValueError("image_feature_size must be positive")
        self.backbone_channels = channels
        self.image_feature_size = int(image_feature_size)
        view_kwargs = dict(view_fusion_kwargs or {})
        self.architecture = str(
            view_kwargs.pop("architecture", "legacy")
        )
        if self.architecture not in {"legacy", "bevformer_v2_t1"}:
            raise ValueError(
                f"unknown camera BEV architecture {self.architecture!r}"
            )
        self.feature_pyramid: BEVFormerFeaturePyramid | None
        self.refine: nn.Module
        self.view_fusion: nn.Module
        if self.architecture == "bevformer_v2_t1":
            self.feature_pyramid = BEVFormerFeaturePyramid(
                channels,
                embed_dim=embed_dim,
            )
            self.lateral_projections = nn.ModuleList()
            self.refine = nn.Identity()
            self.view_fusion = BEVFormerV2T1ViewFusion(
                num_views=num_views,
                embed_dim=embed_dim,
                **view_kwargs,
            )
            return
        self.feature_pyramid = None
        self.lateral_projections = nn.ModuleList(
            nn.Conv2d(channel_count, embed_dim, kernel_size=1)
            for channel_count in channels
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(channels)))
        self.refine = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1,
                groups=embed_dim,
                bias=False,
            ),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )

        # View fusion strategy (pluggable). Extra kwargs (bev_h, bev_w, pc_range,
        # image_size, ...) are forwarded to the selected fusion module.
        self.view_fusion = build_view_fusion(
            fusion_mode, num_views, embed_dim, **view_kwargs
        )

    def forward(self, features, B, V, projection=None, geometry_type=None,
                image_transform=None):
        # features: list of 4 multi-scale feature maps from backbone (channels-first)
        if self.architecture == "bevformer_v2_t1":
            if self.feature_pyramid is None:
                raise RuntimeError("BEVFormer feature pyramid is missing")
            pyramid = self.feature_pyramid(features)
            return self.view_fusion(
                pyramid,
                B,
                V,
                projection=projection,
                geometry_type=geometry_type,
                image_transform=image_transform,
            )
        if len(features) != len(self.lateral_projections):
            raise ValueError(
                "backbone feature count differs from configured stages"
            )
        target_size = (self.image_feature_size, self.image_feature_size)
        projected = []
        for index, (feature, projection_layer) in enumerate(zip(
            features,
            self.lateral_projections,
        )):
            if (
                feature.ndim != 4
                or feature.shape[1] != self.backbone_channels[index]
            ):
                raise ValueError(
                    f"backbone stage {index} shape differs from contract"
                )
            lateral = projection_layer(feature)
            if lateral.shape[-2:] == target_size:
                resized = lateral
            elif (
                lateral.shape[-2] >= self.image_feature_size
                and lateral.shape[-1] >= self.image_feature_size
            ):
                resized = F.adaptive_max_pool2d(lateral, target_size)
            else:
                resized = F.interpolate(
                    lateral,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(resized)
        scale_weights = self.scale_logits.softmax(dim=0)
        fused_per_view = torch.stack(
            [
                weight * feature
                for weight, feature in zip(scale_weights, projected)
            ],
            dim=0,
        ).sum(dim=0)
        fused_per_view = self.refine(fused_per_view)

        # Unify across views. BEV fusion output spatial size is (bev_h, bev_w).
        # Geometry (projection operator / geometry_type / image_transform) is
        # passed straight through — FeatureFusion does not interpret it.
        fused = self.view_fusion(
            fused_per_view, B, V,
            projection=projection,
            geometry_type=geometry_type,
            image_transform=image_transform,
        )

        return fused
