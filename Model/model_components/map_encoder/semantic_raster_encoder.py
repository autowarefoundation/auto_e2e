"""Fully convolutional encoder for metric semantic navigation rasters."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SemanticRasterEncoder(nn.Module):
    """Encode arbitrary-size semantic map/route rasters into camera BEV space."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        output_h: int = 450,
        output_w: int = 300,
    ) -> None:
        super().__init__()
        if min(in_channels, embed_dim, output_h, output_w) <= 0:
            raise ValueError("semantic raster dimensions must be positive")
        self.output_h = int(output_h)
        self.output_w = int(output_w)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                96,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(_group_count(96), 96),
            nn.SiLU(),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(
                96,
                96,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=96,
                bias=False,
            ),
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(192), 192),
            nn.SiLU(),
            nn.Conv2d(
                192,
                192,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=192,
                bias=False,
            ),
            nn.Conv2d(192, 192, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(192), 192),
            nn.SiLU(),
        )
        self.local_projection = nn.Conv2d(
            96,
            embed_dim,
            kernel_size=1,
            bias=False,
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(192, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(embed_dim), embed_dim),
        )
        self.output_norm = nn.Sequential(
            nn.GroupNorm(_group_count(embed_dim), embed_dim),
            nn.SiLU(),
        )

    def forward(self, navigation_raster: torch.Tensor) -> torch.Tensor:
        if navigation_raster.ndim != 4:
            raise ValueError(
                "navigation_raster must have shape [B,C,H,W]"
            )
        if navigation_raster.shape[-2:] != (self.output_h, self.output_w):
            navigation_raster = F.interpolate(
                navigation_raster,
                size=(self.output_h, self.output_w),
                mode="bilinear",
                align_corners=False,
            )
        local = self.stem(navigation_raster)
        context = self.output_projection(self.downsample(local))
        context = F.interpolate(
            context,
            size=local.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.output_norm(self.local_projection(local) + context)
