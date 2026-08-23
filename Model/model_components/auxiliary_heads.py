"""Auxiliary prediction heads for the Reactive BEV representation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_channels(
    embed_dim: int,
    hidden_channels: int,
    output_channels: int,
    num_groups: int,
) -> None:
    if embed_dim <= 0 or hidden_channels <= 0 or output_channels <= 0:
        raise ValueError("head channel counts must be positive")
    if num_groups <= 0 or hidden_channels % num_groups:
        raise ValueError(
            "hidden_channels must be divisible by the GroupNorm group count"
        )


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class BEVSegmentationHead(nn.Module):
    """Decode independent semantic occupancy logits from camera-only BEV."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_channels: int = 64,
        num_classes: int = 8,
        num_groups: int = 8,
        output_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        _validate_channels(
            embed_dim,
            hidden_channels,
            num_classes,
            num_groups,
        )
        if output_size is not None and min(output_size) <= 0:
            raise ValueError("output_size dimensions must be positive")
        self.output_size = (
            tuple(int(value) for value in output_size)
            if output_size is not None
            else None
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, hidden_channels),
            nn.SiLU(),
            _ResidualConvBlock(hidden_channels, num_groups),
            _ResidualConvBlock(hidden_channels, num_groups),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, image_bev: torch.Tensor) -> torch.Tensor:
        if image_bev.ndim != 4:
            raise ValueError("image_bev must have shape [B,C,H,W]")
        logits = self.decoder(image_bev)
        if self.output_size is not None and logits.shape[-2:] != self.output_size:
            logits = F.interpolate(
                logits,
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits


class RouteReconstructionHead(nn.Module):
    """Decode route logits from the gated navigation contribution."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_channels: int = 64,
        route_channels: int = 2,
        num_groups: int = 8,
        output_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        _validate_channels(
            embed_dim,
            hidden_channels,
            route_channels,
            num_groups,
        )
        if output_size is not None and min(output_size) <= 0:
            raise ValueError("output_size dimensions must be positive")
        self.output_size = (
            tuple(int(value) for value in output_size)
            if output_size is not None
            else None
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, route_channels, kernel_size=1),
        )

    def forward(
        self,
        navigation_contribution: torch.Tensor,
    ) -> torch.Tensor:
        if navigation_contribution.ndim != 4:
            raise ValueError(
                "navigation_contribution must have shape [B,C,H,W]"
            )
        logits = self.decoder(navigation_contribution)
        if self.output_size is not None and logits.shape[-2:] != self.output_size:
            logits = F.interpolate(
                logits,
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits
