"""Separate map and route encoding with late route-gated fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from .lightweight_route_encoder import LightweightRouteEncoder


class SplitNavigationEncoder(nn.Module):
    """Fuse independently encoded static map and selected route features."""

    def __init__(
        self,
        map_encoder: nn.Module,
        *,
        route_channels: int,
        embed_dim: int,
        output_h: int,
        output_w: int,
        route_hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        if min(route_channels, embed_dim, output_h, output_w) <= 0:
            raise ValueError("navigation encoder dimensions must be positive")
        self.MapEncoder = map_encoder
        self.RouteEncoder = LightweightRouteEncoder(
            in_channels=route_channels,
            embed_dim=embed_dim,
            output_h=output_h,
            output_w=output_w,
            hidden_channels=route_hidden_channels,
        )
        self.route_channels = int(route_channels)
        self.embed_dim = int(embed_dim)
        self.route_gate = nn.Parameter(torch.zeros(embed_dim))

    def route_gate_values(self) -> torch.Tensor:
        return torch.sigmoid(self.route_gate)

    def forward(
        self,
        map_raster: torch.Tensor,
        route_raster: torch.Tensor,
        *,
        return_route_contribution: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if map_raster.ndim != 4:
            raise ValueError("map_raster must have shape [B,C,H,W]")
        if (
            route_raster.ndim != 4
            or route_raster.shape[:2]
            != (map_raster.shape[0], self.route_channels)
            or route_raster.shape[-2:] != map_raster.shape[-2:]
        ):
            raise ValueError(
                "route_raster must share map spatial dimensions and have "
                f"{self.route_channels} channels"
            )
        map_bev = self.MapEncoder(map_raster)
        route_bev = self.RouteEncoder(route_raster)
        if map_bev.shape != route_bev.shape:
            raise ValueError("map and route encoders produced different shapes")
        route_gate = self.route_gate_values().view(1, -1, 1, 1)
        route_contribution = route_gate * route_bev
        navigation_bev = map_bev + route_contribution
        if return_route_contribution:
            return navigation_bev, route_contribution
        return navigation_bev
