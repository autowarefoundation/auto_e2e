"""Lightweight encoder for sparse route and destination rasters."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .semantic_raster_encoder import _group_count


class LightweightRouteEncoder(nn.Module):
    """Preserve thin route features before projecting them into BEV channels."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        output_h: int = 256,
        output_w: int = 256,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        if min(
            in_channels,
            embed_dim,
            output_h,
            output_w,
            hidden_channels,
        ) <= 0:
            raise ValueError("route encoder dimensions must be positive")
        self.in_channels = int(in_channels)
        self.output_h = int(output_h)
        self.output_w = int(output_w)
        stem_channels = max(16, hidden_channels // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                stem_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(stem_channels), stem_channels),
            nn.SiLU(),
            nn.Conv2d(
                stem_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(hidden_channels, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(embed_dim), embed_dim),
            nn.SiLU(),
        )

    def forward(self, route_raster: torch.Tensor) -> torch.Tensor:
        if (
            route_raster.ndim != 4
            or route_raster.shape[1] != self.in_channels
        ):
            raise ValueError(
                "route_raster must have shape "
                f"[B,{self.in_channels},H,W]"
            )
        if route_raster.shape[-2:] != (self.output_h, self.output_w):
            route_raster = F.interpolate(
                route_raster,
                size=(self.output_h, self.output_w),
                mode="bilinear",
                align_corners=False,
            )
        local = self.stem(route_raster)
        context = F.interpolate(
            self.context(local),
            size=local.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        output = self.output_projection(local + context)
        return output
