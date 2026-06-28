"""Rasterized navigation map encoder.

Takes a BEV-space RGB map image and produces a spatial feature map at the same resolution as the image BEV
features, so the two can be fused directly.
"""

from typing import Any

import torch
import torch.nn as nn
import timm


class RasterizedMapEncoder(nn.Module):
    """Encode a BEV nav-map image into a spatial feature map.
 
    Args:
        in_channels: Number of input channels. Default 3 (RGB map).
        embed_dim: Output channel dimension. Must match the image BEV
            embed_dim so MapBEVFusion can combine them directly.
        output_h: Height of the output feature map.
        output_w: Width of the output feature map.
    """
 
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        output_h: int = 8,
        output_w: int = 8,
    ) -> None:
        super().__init__()
 
        self.output_h = output_h
        self.output_w = output_w
 
        # SwinV2-Tiny backbone, randomly initialized.
        self._backbone = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=False,
            features_only=True,
            in_chans=in_channels,
        )
 
        # timm's FeatureInfo is exposed via __getattr__ (typed Tensor | Module),
        # so bind through Any to iterate it without a mypy union-attr error.
        backbone: Any = self._backbone
        self._feature_channels = [
            stage["num_chs"] for stage in backbone.feature_info
        ]
        backbone_channels = sum(self._feature_channels)
 
        self.pool = nn.AdaptiveMaxPool2d((output_h, output_w))
 
        self.channel_proj = nn.Sequential(
            nn.Conv2d(backbone_channels, embed_dim, kernel_size=1),
            nn.GELU(),
        )
 
    def forward(self, map_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            map_image: (B, 3, H, W) BEV nav-map image. H and W must be 256
                (required by swinv2_tiny_window8_256's patch embedding).
 
        Returns:
            (B, embed_dim, output_h, output_w)
        """
        features = self._backbone(map_image)
 
        # Permute to channels-first. 
        permuted = []
        for f, expected_c in zip(features, self._feature_channels):
            if f.dim() == 4 and f.shape[1] != expected_c and f.shape[-1] == expected_c:
                permuted.append(f.permute(0, 3, 1, 2).contiguous())
            else:
                permuted.append(f)
 
        pooled = [self.pool(f) for f in permuted]
        x = torch.cat(pooled, dim=1)   # (B, backbone_channels, output_h, output_w)
        return self.channel_proj(x)    # (B, embed_dim, output_h, output_w)
 