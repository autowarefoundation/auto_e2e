import torch
import torch.nn as nn


class CrossAttentionViewFusion(nn.Module):
    """Fuse multi-view features using cross-camera attention.

    Each spatial position attends across all camera views to learn
    which views are relevant, using learnable camera position embeddings
    to encode view identity.

    Reference:
        - PETR (Liu et al., ECCV 2022): position embedding transformation
        - UniAD (Hu et al., CVPR 2023): unified query-based cross-attention
    """

    def __init__(self, num_views, embed_dim=1440, num_heads=8, dropout=0.1):
        super().__init__()

        self.num_views = num_views
        self.embed_dim = embed_dim

        # Learnable camera/view position embeddings
        self.view_embed = nn.Parameter(torch.randn(1, num_views, embed_dim) * 0.02)

        # Layer norm before attention
        self.norm = nn.LayerNorm(embed_dim)

        # Multi-head cross-view attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, fused_per_view, B, V, camera_params=None):
        # fused_per_view: [B*V, C, H, W]
        C, H, W = fused_per_view.shape[1], fused_per_view.shape[2], fused_per_view.shape[3]

        # Reshape to [B, V, C, H, W]
        x = fused_per_view.reshape(B, V, C, H, W)

        # Reshape to [B, V, C, H*W] then transpose to [B, H*W, V, C]
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, V, C)

        # Add learnable view position embeddings
        x = x + self.view_embed

        # Pre-norm
        x_norm = self.norm(x)

        # Cross-view attention: each spatial position attends across all views
        attn_out, _ = self.cross_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Feed-forward with residual
        x = x + self.ffn(self.norm_ffn(x))

        # Pool across views: [B*H*W, V, C] → [B*H*W, C]
        x = x.mean(dim=1)

        # Reshape back to spatial: [B, C, H, W]
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x
