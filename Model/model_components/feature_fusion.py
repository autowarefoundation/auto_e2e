from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from .feature_pyramid import BEVFormerFeaturePyramid
from .view_fusion import build_view_fusion
from .view_fusion.bevformer_v2_t1 import BEVFormerV2T1ViewFusion
from .view_fusion.bevformer_v2_t8 import BEVFormerV2T8TemporalFusion


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
        if self.architecture not in {
            "legacy",
            "bevformer_v2_t1",
            "bevformer_v2_t8",
        }:
            raise ValueError(
                f"unknown camera BEV architecture {self.architecture!r}"
            )
        self.feature_pyramid: BEVFormerFeaturePyramid | None
        self.refine: nn.Module
        self.view_fusion: nn.Module
        self.temporal_fusion: BEVFormerV2T8TemporalFusion | None = None
        if self.architecture in {"bevformer_v2_t1", "bevformer_v2_t8"}:
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
            if self.architecture == "bevformer_v2_t8":
                self.temporal_fusion = BEVFormerV2T8TemporalFusion(
                    embed_dim=embed_dim,
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

    def build_bevformer_pyramid(
        self,
        features,
    ) -> tuple[torch.Tensor, ...]:
        """Build the cacheable FPN representation for BEVFormer cameras."""
        if self.architecture not in {"bevformer_v2_t1", "bevformer_v2_t8"}:
            raise ValueError("camera FPN cache requires BEVFormer fusion")
        if self.feature_pyramid is None:
            raise RuntimeError("BEVFormer feature pyramid is missing")
        if features is None:
            raise ValueError(
                "BEVFormer fusion requires base-resolution camera features"
            )
        return tuple(self.feature_pyramid(features))

    def fuse_bevformer_pyramid(
        self,
        pyramid,
        B,
        V,
        *,
        projection=None,
        geometry_type=None,
        image_transform=None,
    ):
        """Project one cached multi-view FPN into the current BEV frame."""
        if self.architecture not in {"bevformer_v2_t1", "bevformer_v2_t8"}:
            raise ValueError("camera FPN cache requires BEVFormer fusion")
        return self.view_fusion(
            pyramid,
            B,
            V,
            projection=projection,
            geometry_type=geometry_type,
            image_transform=image_transform,
        )

    def fuse_front_bevformer_pyramid(
        self,
        image_bev,
        front_pyramid,
        *,
        projection=None,
        geometry_type=None,
        image_transform=None,
    ):
        """Fuse the current high-resolution front FPN into an image BEV."""
        if self.architecture not in {"bevformer_v2_t1", "bevformer_v2_t8"}:
            raise ValueError("front camera FPN requires BEVFormer fusion")
        return self.view_fusion.fuse_front_camera(
            image_bev,
            front_pyramid,
            projection=projection,
            geometry_type=geometry_type,
            image_transform=image_transform,
        )

    def forward(
        self,
        features,
        B,
        V,
        projection=None,
        geometry_type=None,
        image_transform=None,
        front_features=None,
        front_projection=None,
        front_image_transform=None,
        return_base_pyramid=False,
    ):
        # features: list of 4 multi-scale feature maps from backbone (channels-first)
        if self.architecture in {"bevformer_v2_t1", "bevformer_v2_t8"}:
            pyramid = self.build_bevformer_pyramid(features)
            image_bev = self.fuse_bevformer_pyramid(
                pyramid,
                B,
                V,
                projection=projection,
                geometry_type=geometry_type,
                image_transform=image_transform,
            )
            if front_features is None:
                output = image_bev
            else:
                native_front_pyramid = self.build_bevformer_pyramid(
                    front_features
                )
                output = self.fuse_front_bevformer_pyramid(
                    image_bev,
                    native_front_pyramid,
                    projection=front_projection,
                    geometry_type=geometry_type,
                    image_transform=front_image_transform,
                )
            if return_base_pyramid:
                return output, pyramid
            return output
        if return_base_pyramid:
            raise ValueError(
                "base camera FPN output requires BEVFormer fusion"
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

    def freeze_structurally_unused_parameters(self) -> None:
        freeze = getattr(
            self.view_fusion,
            "freeze_structurally_unused_parameters",
            None,
        )
        if callable(freeze):
            freeze()

    def fuse_temporal_bevs(
        self,
        frame_bevs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if self.temporal_fusion is None:
            if len(frame_bevs) != 1:
                raise ValueError(
                    "non-T8 camera fusion accepts only the current BEV"
                )
            return frame_bevs[0]
        return self.temporal_fusion(frame_bevs)
