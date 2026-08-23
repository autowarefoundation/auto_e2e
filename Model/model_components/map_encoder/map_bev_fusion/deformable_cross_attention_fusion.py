"""Deformable cross-attention map BEV fusion.

Each spatial query in the image BEV attends to K learned-offset sample points
in the map BEV instead of all H*W tokens. Sampling is done with
``F.grid_sample`` (bilinear interpolation), so no custom CUDA kernels are
needed. This drops the cost of fusion from O(N^2) to O(N*K), making it viable
at production BEV grids (450x300 = 135K tokens) where dense cross-attention
would OOM.

The design follows the deformable spatial cross-attention introduced by
BEVFormer: a query predicts K sampling offsets relative to its own reference
position, samples features there, and aggregates them with per-head softmax
weights. Unlike BEVFormer, the reference plane here is the map BEV itself, so
the reference point of each query is simply its own pixel position.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MapDeformableCrossAttentionFusion(nn.Module):
    """Fuse image BEV and map BEV via deformable spatial cross-attention.

    Instead of attending to all ``H*W`` map tokens, each query pixel predicts
    K 2D offsets from its own position, samples the map BEV at those locations
    (bilinearly), and aggregates the K samples with per-head softmax weights.

    Args:
        embed_dim: Channel dimension of both input feature maps.
        num_sample_points: K -- number of offset sample points per query.
        num_heads: Number of attention heads. Heads share the K sampling
            locations but learn independent attention weights.
        dropout: Dropout applied inside the FFN.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_sample_points: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        query_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if (
            embed_dim % num_heads
            or min(
                embed_dim,
                num_sample_points,
                num_heads,
                query_chunk_size,
            ) <= 0
        ):
            raise ValueError("deformable map fusion dimensions are invalid")

        self.embed_dim = embed_dim
        self.num_points = num_sample_points
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query_chunk_size = int(query_chunk_size)

        # Predict K 2D offsets (pixel displacements) from the query feature.
        self.offset_proj = nn.Linear(embed_dim, num_sample_points * 2)

        # Predict per-head attention weights over the K sample points.
        self.attn_proj = nn.Linear(embed_dim, num_heads * num_sample_points)

        # Pre-norm on queries, then output projection with residual.
        self.norm_query = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # FFN with residual, matching the other fusion modes.
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.reset_residual_parameters()

    def reset_residual_parameters(self) -> None:
        """Keep the pretrained camera BEV path unchanged at initialization."""
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        final_linear = self.ffn[3]
        if not isinstance(final_linear, nn.Linear):
            raise RuntimeError("deformable fusion FFN layout changed")
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(
        self,
        image_bev: torch.Tensor,
        map_bev: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse ``map_bev`` into ``image_bev``.

        Args:
            image_bev: (B, embed_dim, H, W) image BEV features -- queries.
            map_bev: (B, embed_dim, H, W) map BEV features -- sampled values.
                Must have the same spatial size as image_bev.

        Returns:
            (B, embed_dim, H, W) image BEV updated with map context.
        """
        B, C, H, W = image_bev.shape
        if map_bev.shape != image_bev.shape:
            raise ValueError("image_bev and map_bev must have identical shapes")
        if C != self.embed_dim or min(H, W) <= 1:
            raise ValueError("map fusion input differs from configured shape")
        N = H * W
        K = self.num_points
        nH = self.num_heads

        # Reference grid: each query's own pixel position in grid_sample
        # coordinates [-1, 1] (align_corners=True).
        ys = torch.linspace(-1.0, 1.0, H, device=image_bev.device,
                            dtype=image_bev.dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=image_bev.device,
                            dtype=image_bev.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        ref_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        ref_grid = ref_grid.reshape(1, N, 1, 2)  # (1, N, 1, 2)

        # Flatten spatial dims: (B, N, C)
        q = image_bev.permute(0, 2, 3, 1).reshape(B, N, C)
        q_norm = self.norm_query(q)

        # Offsets are pixel displacements; rescale to grid coordinates
        # (a pixel step is 2/(size-1) with align_corners=True).
        scale = torch.tensor(
            [2.0 / (W - 1), 2.0 / (H - 1)],
            device=image_bev.device,
            dtype=image_bev.dtype,
        )
        attended_chunks = []
        for start in range(0, N, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, N)
            count = stop - start
            chunk_query = q_norm[:, start:stop]
            offsets = self.offset_proj(chunk_query).reshape(
                B,
                count,
                K,
                2,
            )
            offsets = offsets * scale.view(1, 1, 1, 2)
            attn_logits = self.attn_proj(chunk_query).reshape(
                B,
                count,
                nH,
                K,
            )
            sample_grid = (
                ref_grid[:, start:stop] + offsets
            ).reshape(B, count, K, 2)
            sampled = F.grid_sample(
                map_bev,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).permute(0, 2, 3, 1)
            sampled = sampled.reshape(
                B,
                count,
                K,
                nH,
                self.head_dim,
            ).permute(0, 1, 3, 2, 4)
            attn_weights = F.softmax(attn_logits, dim=-1)
            attended_chunks.append(
                (
                    sampled * attn_weights.unsqueeze(-1)
                ).sum(dim=3).reshape(B, count, C)
            )
        attn_out = torch.cat(attended_chunks, dim=1)

        # Output projection with residual, then FFN with residual.
        q = q + self.out_proj(attn_out)
        q = q + self.ffn(self.norm_ffn(q))

        # Reshape back to spatial: (B, C, H, W)
        return q.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
