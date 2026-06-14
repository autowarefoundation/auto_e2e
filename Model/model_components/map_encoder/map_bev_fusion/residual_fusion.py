"""Residual map BEV fusion.

Fuses image BEV features with map BEV features via a learned residual gate:

    output = image_bev + alpha * map_bev

where ``alpha`` is a per-channel learnable parameter vector (shape: ``embed_dim``),
initialized to zero and reshaped to ``(1, embed_dim, 1, 1)`` for broadcasting.
At the start of training the map branch contributes nothing, so the model behaves
identically to training without a map encoder. As training progresses, ``alpha``
grows to weight whichever map channels are useful.

"""

import torch
import torch.nn as nn


class ResidualMapFusion(nn.Module):
    """Add map BEV features to image BEV features via a per-channel gate.

    Args:
        embed_dim: Channel dimension of both input feature maps. Must match.
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()

        # Per-channel gate initialized to zero.
        # Shape (embed_dim,) so each channel of map_bev is weighted
        # independently.
        self.alpha = nn.Parameter(torch.zeros(embed_dim))

    def forward(
        self,
        image_bev: torch.Tensor,
        map_bev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_bev: (B, embed_dim, H, W) image BEV features.
            map_bev:   (B, embed_dim, H, W) map BEV features.
                       Must have the same spatial size as image_bev.

        Returns:
            (B, embed_dim, H, W) fused BEV features.
        """
        # Reshape alpha for broadcast: (1, embed_dim, 1, 1)
        gate = self.alpha.view(1, -1, 1, 1)
        return image_bev + gate * map_bev