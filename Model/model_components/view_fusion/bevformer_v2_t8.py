"""Official-checkpoint-compatible temporal fusion for BEVFormer V2 T8."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class BEVFormerV2T8BasicBlock(nn.Module):
    """ResNet BasicBlock matching the official V2 temporal fusion state."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            if downsample
            else None
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        output = self.relu(self.bn1(self.conv1(value)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            identity = self.downsample(value)
        return self.relu(output + identity)


class BEVFormerV2T8TemporalFusion(nn.Module):
    """Fuse seven detached history BEVs and one current BEV."""

    def __init__(
        self,
        *,
        embed_dim: int = 256,
        frame_count: int = 8,
        inter_channels: int = 512,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if min(embed_dim, frame_count, inter_channels, num_layers) <= 0:
            raise ValueError("T8 fusion dimensions must be positive")
        if frame_count != 8:
            raise ValueError("BEVFormer V2 T8 requires exactly eight frames")
        self.embed_dim = int(embed_dim)
        self.frame_count = int(frame_count)
        self.inter_channels = int(inter_channels)
        # Official BEVFormer V2 leaves ResNetFusion.with_cp disabled. Its
        # trainable BatchNorm state must update exactly once per micro-step.
        self.activation_checkpointing = False
        in_channels = self.frame_count * self.embed_dim
        blocks = []
        for index in range(num_layers):
            block_in = in_channels if index == 0 else self.inter_channels
            blocks.append(BEVFormerV2T8BasicBlock(
                block_in,
                self.inter_channels,
                downsample=block_in != self.inter_channels,
            ))
        self.layers = nn.Sequential(*blocks)
        self.layer_norm = nn.Sequential(
            nn.Linear(self.inter_channels, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

    def forward(
        self,
        frame_bevs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(frame_bevs) != self.frame_count:
            raise ValueError(
                f"T8 fusion requires {self.frame_count} ordered BEVs"
            )
        first = frame_bevs[0]
        if first.ndim != 4 or first.shape[1] != self.embed_dim:
            raise ValueError("T8 BEVs must have shape [B,C,H,W]")
        if any(
            value.shape != first.shape
            or value.device != first.device
            or value.dtype != first.dtype
            for value in frame_bevs[1:]
        ):
            raise ValueError("all T8 BEVs must share shape, device, and dtype")
        # Upstream BEVFormer@66b65f3 transformerV2.py:308-324 passes
        # frames=(-7,-6,-5,-4,-3,-2,-1,0) directly to ResNetFusion.
        output = torch.cat(tuple(frame_bevs), dim=1).contiguous()
        output = self.layers(output)
        output = output.flatten(2).transpose(1, 2)
        output = self.layer_norm(output)
        return output.transpose(1, 2).reshape(
            first.shape[0],
            self.embed_dim,
            first.shape[2],
            first.shape[3],
        ).contiguous()
