import torch
import torch.nn as nn


class ConcatViewFusion(nn.Module):
    """Fuse multi-view features by concatenating along channels and reducing with Conv2d."""

    def __init__(self, num_views, embed_dim=1440):
        super().__init__()
        self.view_reduce = nn.Sequential(
            nn.Conv2d(num_views * embed_dim, embed_dim, kernel_size=1),
            nn.GELU()
        )

    def forward(self, fused_per_view, B, V, camera_params=None):
        # fused_per_view: [B*V, C, H, W]
        C, H, W = fused_per_view.shape[1], fused_per_view.shape[2], fused_per_view.shape[3]

        # Separate batch and views: [B, V, C, H, W]
        x = fused_per_view.reshape(B, V, C, H, W)

        # Merge views into channel dim: [B, V*C, H, W]
        x = x.reshape(B, V * C, H, W)

        # Reduce to [B, C, H, W]
        return self.view_reduce(x)
