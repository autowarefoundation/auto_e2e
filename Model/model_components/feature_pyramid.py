"""BEVFormer V2-compatible ResNet feature pyramid."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVFormerFeaturePyramid(nn.Module):
    """Build four 256-channel levels from the final three backbone stages."""

    def __init__(
        self,
        backbone_channels: Sequence[int],
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in backbone_channels)
        if len(channels) < 3 or min(*channels, embed_dim) <= 0:
            raise ValueError(
                "BEVFormer FPN requires at least three positive backbone stages"
            )
        self.input_channels = channels[-3:]
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(channel_count, embed_dim, kernel_size=1)
            for channel_count in self.input_channels
        )
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
            for _ in range(3)
        ] + [
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        ])

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        selected = tuple(features[-3:])
        if len(selected) != 3:
            raise ValueError("BEVFormer FPN requires three feature maps")
        laterals = []
        for index, (feature, expected_channels, projection) in enumerate(zip(
            selected,
            self.input_channels,
            self.lateral_convs,
        )):
            if (
                feature.ndim != 4
                or feature.shape[1] != expected_channels
            ):
                raise ValueError(
                    f"BEVFormer FPN stage {index} shape differs from contract"
                )
            laterals.append(projection(feature))
        for index in range(2, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index],
                size=laterals[index - 1].shape[-2:],
                mode="nearest",
            )
        outputs = [
            convolution(lateral)
            for convolution, lateral in zip(
                self.fpn_convs[:3],
                laterals,
            )
        ]
        outputs.append(self.fpn_convs[3](F.relu(outputs[-1])))
        return outputs
