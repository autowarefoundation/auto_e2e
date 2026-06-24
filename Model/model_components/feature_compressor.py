"""Memory-bounded compression of BEV feature maps for the JEPA forecast.

Why this module exists
----------------------
Predicting the *full* future BEV feature grid (``450 x 300 x 256``) over
several horizons is prohibitively memory-heavy. @m-zain-khawaja flagged this in
issue #13 (2026-06-23) and proposed averaging the channels to a single
``(450 x 300 x 1)`` map.

Averaging hits the memory budget but collapses the target to an
occupancy-like scalar, which throws away the **semantic feature space** that
makes the JEPA reconstruction objective worthwhile in the first place: I-JEPA
shows that predicting *abstract features* (not pixels or scalars) is precisely
what lets the encoder drop unpredictable low-level detail and keep meaningful
structure (Assran et al. 2023, "Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture", arXiv:2301.08243; V-JEPA, Bardes
et al. 2024, arXiv:2404.08471). A rank-1 channel mean is the most damaging
possible reduction for that objective.

So instead of a fixed mean, compression is exposed as a **tunable knob** along
two axes that reach the same memory budget while preserving a real
representation:

* channel compression (``mode``):
    - ``"full"``       — keep all ``embed_dim`` channels (no channel reduction).
    - ``"projected"``  — learned ``1x1`` conv ``embed_dim -> compressed_dim``
                         (DEFAULT). A rank-``compressed_dim`` projection keeps
                         far more information than the rank-1 mean for the same
                         channel count.
    - ``"occupancy"``  — mean over channels -> ``1`` (the cheapest baseline,
                         ≈ occupancy forecasting; this is @m-zain-khawaja's
                         proposal, kept as an explicit option).
* spatial compression (``spatial_stride``): average-pool the BEV grid. Since
  cost scales with ``H * W`` this is the single largest memory lever.

Example: ``mode="projected", compressed_dim=16, spatial_stride=4`` gives the
same ``256x`` memory saving as the channel mean, but with 16 learned semantic
channels instead of one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureCompressor(nn.Module):
    """Compress ``[B, embed_dim, H, W]`` BEV features for the JEPA forecast.

    Args:
        embed_dim: input channel count.
        mode: ``"full"`` | ``"projected"`` | ``"occupancy"`` (see module doc).
        compressed_dim: output channels for ``"projected"`` mode.
        spatial_stride: average-pool factor on the BEV grid (``1`` = no pool).

    Attributes:
        out_channels: channels produced by :meth:`forward` (useful to size the
            predictor and the target encoder so both live in the same space).
    """

    def __init__(self, embed_dim: int, mode: str = "projected",
                 compressed_dim: int = 64, spatial_stride: int = 1):
        super().__init__()
        if mode not in ("full", "projected", "occupancy"):
            raise ValueError(
                f"mode must be 'full', 'projected' or 'occupancy', got {mode!r}"
            )
        if spatial_stride < 1:
            raise ValueError(f"spatial_stride must be >= 1, got {spatial_stride}")
        self.mode = mode
        self.spatial_stride = spatial_stride

        if mode == "projected":
            self.proj: nn.Module | None = nn.Conv2d(embed_dim, compressed_dim, 1)
            self.out_channels = compressed_dim
        elif mode == "occupancy":
            self.proj = None
            self.out_channels = 1
        else:  # full
            self.proj = None
            self.out_channels = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, embed_dim, H, W] -> [B, out_channels, H//s, W//s]``."""
        if self.mode == "occupancy":
            x = x.mean(dim=1, keepdim=True)
        elif self.mode == "projected":
            assert self.proj is not None  # for mypy; set in __init__
            x = self.proj(x)
        if self.spatial_stride > 1:
            x = F.avg_pool2d(x, kernel_size=self.spatial_stride,
                             stride=self.spatial_stride)
        return x
