"""Token cross-attention fusion of OSM vector features into the image BEV.

Each BEV cell attends to the SD-map token sequence produced by
``OSMVectorMapEncoder``. Unlike ``MapCrossAttentionFusion`` (which fuses two
spatial grids), the keys/values here are a *variable-length, padded token set*,
so this module:

  * uses ``F.scaled_dot_product_attention`` (flash / mem-efficient backend) so
    the full BEV grid (e.g. 450x300 = 135k queries) never materialises a
    ``Q x K`` weight matrix,
  * honours the encoder's ``key_padding_mask`` (padded tokens are ignored),
  * prepends a learnable null token so a sample with no OSM elements (all keys
    masked) still has one valid key and cannot produce NaNs,
  * gates the map contribution with a per-channel parameter initialised to zero,
    so at the start of training the output equals ``image_bev`` exactly (same
    convention as ``ResidualMapFusion``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OSMCrossAttnFusion(nn.Module):
    """Fuse OSM map tokens into image BEV via masked cross-attention.

    Args:
        embed_dim: Channel dimension of the BEV features and map tokens.
        num_heads: Number of attention heads. Must divide ``embed_dim``.
        dropout: Dropout on attention weights (train only) and in the FFN.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

        # Learnable "no map information" key so attention always has a valid key.
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.null_token, std=0.02)

        # Per-channel gate, zero-initialised: map has no effect at training start.
        self.alpha = nn.Parameter(torch.zeros(embed_dim))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, C) -> (B, num_heads, L, head_dim)
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        image_bev: torch.Tensor,
        osm_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            image_bev: (B, embed_dim, H, W) — queries.
            osm_tokens: (B, N, embed_dim) — SD-map tokens (keys/values).
            key_padding_mask: (B, N) bool, ``True`` at padded tokens. ``None``
                treats all tokens as valid.

        Returns:
            (B, embed_dim, H, W) image BEV updated with SD-map context.
        """
        B, C, H, W = image_bev.shape

        q = image_bev.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Prepend the always-valid null token to keys/values.
        null = self.null_token.expand(B, -1, -1)
        kv = torch.cat([null, osm_tokens], dim=1)
        kv = self.norm_kv(kv)

        if key_padding_mask is not None:
            null_valid = torch.zeros((B, 1), dtype=torch.bool, device=kv.device)
            padding = torch.cat([null_valid, key_padding_mask], dim=1)  # (B, N+1)
            # SDPA bool mask: True = participate. Invert padding, broadcast over
            # heads and queries -> (B, 1, 1, N+1).
            attn_mask = (~padding)[:, None, None, :]
        else:
            attn_mask = None

        qh = self._heads(self.q_proj(self.norm_q(q)))
        kh = self._heads(self.k_proj(kv))
        vh = self._heads(self.v_proj(kv))

        attn = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, num_heads, H*W, head_dim)
        attn = attn.transpose(1, 2).reshape(B, H * W, C)
        attn = self.out_proj(attn)

        # Map-derived context (attention + FFN), gated to zero at init.
        h = attn + self.ffn(self.norm_ffn(attn))
        out = q + self.alpha * h

        return out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
