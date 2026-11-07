"""Portable BEVFormer V2 t1 encoder with multi-scale deformable attention."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .projection import ImageTransform, PinholeProjection


def _radial_offset_bias(
    num_heads: int,
    num_groups: int,
    num_points: int,
) -> torch.Tensor:
    angles = (
        torch.arange(num_heads, dtype=torch.float32)
        * (2.0 * math.pi / num_heads)
    )
    directions = torch.stack([angles.cos(), angles.sin()], dim=-1)
    directions = directions / directions.abs().amax(
        dim=-1,
        keepdim=True,
    )
    bias = directions.reshape(num_heads, 1, 1, 2).repeat(
        1,
        num_groups,
        num_points,
        1,
    )
    radii = torch.arange(
        1,
        num_points + 1,
        dtype=torch.float32,
    ).reshape(1, 1, num_points, 1)
    return (bias * radii).reshape(-1)


def _checkpoint_encoder_layer(
    layer: nn.Module,
    bev_h: int,
    bev_w: int,
    num_views: int,
    query: torch.Tensor,
    position: torch.Tensor,
    reference_2d: torch.Tensor,
    reference_mask: torch.Tensor,
    level_embeddings: torch.Tensor,
    camera_embeddings: torch.Tensor,
    reference_depth_weight: torch.Tensor | None,
    *features: torch.Tensor,
) -> torch.Tensor:
    return layer(
        query,
        position,
        features,
        reference_2d,
        reference_mask,
        bev_h=bev_h,
        bev_w=bev_w,
        num_views=num_views,
        level_embeddings=level_embeddings,
        camera_embeddings=camera_embeddings,
        reference_depth_weight=reference_depth_weight,
    )


class T1DeformableSelfAttention(nn.Module):
    """BEVFormer temporal self-attention with current BEV duplicated as t1."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.1,
        query_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if min(num_points, query_chunk_size) <= 0:
            raise ValueError("attention dimensions must be positive")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = 2
        self.head_dim = embed_dim // num_heads
        self.query_chunk_size = query_chunk_size
        self.sampling_offsets = nn.Linear(
            embed_dim * self.num_bev_queue,
            self.num_bev_queue * num_heads * num_points * 2,
        )
        self.attention_weights = nn.Linear(
            embed_dim * self.num_bev_queue,
            self.num_bev_queue * num_heads * num_points,
        )
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.sampling_offsets.weight)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(
                _radial_offset_bias(
                    self.num_heads,
                    self.num_bev_queue,
                    self.num_points,
                )
            )
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        *,
        bev_h: int,
        bev_w: int,
    ) -> torch.Tensor:
        batch_size, query_count, channels = query.shape
        if (
            channels != self.embed_dim
            or query_count != bev_h * bev_w
            or query_pos.shape != query.shape
        ):
            raise ValueError("t1 self-attention input shape differs from contract")
        value = self.value_proj(query)
        value_spatial = value.reshape(
            batch_size,
            bev_h,
            bev_w,
            self.num_heads,
            self.head_dim,
        ).permute(0, 3, 4, 1, 2).reshape(
            batch_size * self.num_heads,
            self.head_dim,
            bev_h,
            bev_w,
        )
        rows = (
            torch.arange(bev_h, device=query.device, dtype=torch.float32)
            + 0.5
        ) / bev_h
        cols = (
            torch.arange(bev_w, device=query.device, dtype=torch.float32)
            + 0.5
        ) / bev_w
        grid_row, grid_col = torch.meshgrid(rows, cols, indexing="ij")
        reference = torch.stack(
            [grid_col, grid_row],
            dim=-1,
        ).reshape(1, query_count, 1, 1, 2)
        output_chunks = []
        normalizer = torch.tensor(
            [bev_w, bev_h],
            device=query.device,
            dtype=torch.float32,
        )
        positioned_query = query + query_pos
        predictor_input = torch.cat([query, positioned_query], dim=-1)
        for start in range(0, query_count, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, query_count)
            chunk_size = stop - start
            offsets = self.sampling_offsets(
                predictor_input[:, start:stop]
            ).reshape(
                batch_size,
                chunk_size,
                self.num_heads,
                self.num_bev_queue,
                self.num_points,
                2,
            )
            weights = self.attention_weights(
                predictor_input[:, start:stop]
            ).reshape(
                batch_size,
                chunk_size,
                self.num_heads,
                self.num_bev_queue,
                self.num_points,
            ).softmax(dim=-1)
            queue_outputs = []
            for queue_index in range(self.num_bev_queue):
                locations = (
                    reference[:, start:stop]
                    + offsets[:, :, :, queue_index].float() / normalizer
                )
                sample_grid = (
                    locations * 2.0 - 1.0
                ).permute(0, 2, 1, 3, 4).reshape(
                    batch_size * self.num_heads,
                    chunk_size,
                    self.num_points,
                    2,
                )
                with torch.autocast(
                    device_type=query.device.type,
                    enabled=False,
                ):
                    sampled = F.grid_sample(
                        value_spatial.float(),
                        sample_grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).reshape(
                        batch_size,
                        self.num_heads,
                        self.head_dim,
                        chunk_size,
                        self.num_points,
                    ).permute(0, 3, 1, 4, 2)
                queue_weight = weights[:, :, :, queue_index]
                queue_outputs.append(
                    (
                        sampled * queue_weight.float().unsqueeze(-1)
                    ).sum(dim=3)
                )
            output_chunks.append(
                torch.stack(queue_outputs, dim=0).mean(dim=0).reshape(
                    batch_size,
                    chunk_size,
                    self.embed_dim,
                )
            )
        attended = torch.cat(output_chunks, dim=1).to(query.dtype)
        return query + self.dropout(self.output_proj(attended))


class MultiScaleSpatialCrossAttention(nn.Module):
    """Sample calibrated camera FPN levels with independent attention heads."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
        dropout: float = 0.1,
        query_chunk_size: int = 4096,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if min(num_levels, num_points, query_chunk_size) <= 0:
            raise ValueError("attention dimensions must be positive")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        self.query_chunk_size = query_chunk_size
        self.activation_checkpointing = bool(activation_checkpointing)
        self.sampling_offsets = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_points * 2,
        )
        self.attention_weights = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_points,
        )
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.sampling_offsets.weight)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(
                _radial_offset_bias(
                    self.num_heads,
                    self.num_levels,
                    self.num_points,
                )
            )
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _sample_level(
        self,
        value_per_head: torch.Tensor,
        sample_grid: torch.Tensor,
        level_weight: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, chunk_size = level_weight.shape[:2]
        output_dtype = value_per_head.dtype
        with torch.autocast(
            device_type=value_per_head.device.type,
            enabled=False,
        ):
            sampled = F.grid_sample(
                value_per_head.float(),
                sample_grid.float(),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).reshape(
                batch_size,
                self.num_heads,
                self.head_dim,
                chunk_size,
                self.num_points,
            ).permute(0, 3, 1, 4, 2)
            attended = (
                sampled * level_weight.float().unsqueeze(-1)
            ).sum(dim=3)
        return attended.to(output_dtype)

    def _project_values(
        self,
        features: Sequence[torch.Tensor],
        *,
        batch_size: int,
        num_views: int,
        level_embeddings: torch.Tensor,
        camera_embeddings: torch.Tensor,
    ) -> list[torch.Tensor]:
        projected = []
        for level, feature in enumerate(features):
            if (
                feature.ndim != 4
                or feature.shape[0] != batch_size * num_views
                or feature.shape[1] != self.embed_dim
            ):
                raise ValueError(
                    f"camera feature level {level} differs from contract"
                )
            height, width = feature.shape[-2:]
            values = feature.reshape(
                batch_size,
                num_views,
                self.embed_dim,
                height,
                width,
            )
            values = values.permute(0, 1, 3, 4, 2)
            values = values + level_embeddings[level].reshape(1, 1, 1, 1, -1)
            # Per sample and view: the embedding follows the calibration.
            values = values + camera_embeddings.reshape(batch_size, num_views, 1, 1, -1)
            values = self.value_proj(values)
            values = values.permute(0, 1, 4, 2, 3).contiguous()
            projected.append(values)
        return projected

    def forward(
        self,
        query: torch.Tensor,
        features: Sequence[torch.Tensor],
        reference_points_2d: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        num_views: int,
        level_embeddings: torch.Tensor,
        camera_embeddings: torch.Tensor,
        reference_depth_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample camera features at projected pillar points.

        ``reference_depth_weight`` ([B, V, N, Z], optional) scales the
        attention weight of every sampling point by how likely its anchor
        pillar point's depth is under the camera's predicted depth
        distribution (depth-aware lifting as in DFA3D), then renormalizes
        per query and head. Sampling points are offset from their anchor in
        the image plane, so the anchor depth is an approximation for them.
        None keeps the original behavior.
        """
        batch_size, query_count, channels = query.shape
        if channels != self.embed_dim or len(features) != self.num_levels:
            raise ValueError("spatial cross-attention input differs from contract")
        values = self._project_values(
            features,
            batch_size=batch_size,
            num_views=num_views,
            level_embeddings=level_embeddings,
            camera_embeddings=camera_embeddings,
        )
        projection_batch = reference_points_2d.shape[0]
        if projection_batch not in (1, batch_size):
            raise ValueError("projection batch cannot broadcast to query batch")
        if projection_batch == 1 and batch_size > 1:
            reference_points_2d = reference_points_2d.expand(
                batch_size,
                -1,
                -1,
                -1,
                -1,
            )
            reference_mask = reference_mask.expand(
                batch_size,
                -1,
                -1,
                -1,
            )
        pillar_points = reference_points_2d.shape[3]
        anchor_indices = (
            torch.arange(self.num_points, device=query.device)
            % pillar_points
        )
        camera_outputs = query.new_zeros(
            batch_size,
            query_count,
            self.embed_dim,
        )
        camera_counts = query.new_zeros(
            batch_size,
            query_count,
            1,
        )
        for view_index in range(num_views):
            output_chunks = []
            visible_chunks = []
            for start in range(0, query_count, self.query_chunk_size):
                stop = min(start + self.query_chunk_size, query_count)
                chunk_size = stop - start
                chunk_query = query[:, start:stop]
                offsets = self.sampling_offsets(chunk_query).reshape(
                    batch_size,
                    chunk_size,
                    self.num_heads,
                    self.num_levels,
                    self.num_points,
                    2,
                )
                weights = self.attention_weights(chunk_query).reshape(
                    batch_size,
                    chunk_size,
                    self.num_heads,
                    self.num_levels,
                    self.num_points,
                )
                weights = weights.flatten(-2).softmax(dim=-1).reshape_as(
                    weights
                )
                if reference_depth_weight is not None:
                    depth_weight = reference_depth_weight[
                        :,
                        view_index,
                        start:stop,
                    ][..., anchor_indices]
                    weights = weights * depth_weight[:, :, None, None, :].to(
                        weights.dtype
                    )
                    # Renormalize over levels and points so the depth weight
                    # redistributes attention within a view instead of
                    # shrinking the view's contribution to the camera mean.
                    weights = weights / weights.sum(
                        dim=(-2, -1),
                        keepdim=True,
                    ).clamp_min(1e-6)
                attended = chunk_query.new_zeros(
                    batch_size,
                    chunk_size,
                    self.num_heads,
                    self.head_dim,
                )
                visible = reference_mask[
                    :,
                    view_index,
                    start:stop,
                ].any(dim=-1)
                reference = reference_points_2d[
                    :,
                    view_index,
                    start:stop,
                    anchor_indices,
                ].float()
                for level, value in enumerate(values):
                    height, width = value.shape[-2:]
                    normalizer = torch.tensor(
                        [width, height],
                        device=query.device,
                        dtype=torch.float32,
                    )
                    locations = (
                        reference.unsqueeze(2)
                        + offsets[:, :, :, level].float() / normalizer
                    )
                    sample_grid = (
                        locations * 2.0 - 1.0
                    ).permute(0, 2, 1, 3, 4).reshape(
                        batch_size * self.num_heads,
                        chunk_size,
                        self.num_points,
                        2,
                    )
                    value_per_head = value[:, view_index].reshape(
                        batch_size,
                        self.num_heads,
                        self.head_dim,
                        height,
                        width,
                    ).reshape(
                        batch_size * self.num_heads,
                        self.head_dim,
                        height,
                        width,
                    )
                    level_weight = weights[:, :, :, level]
                    sample_inputs = (
                        value_per_head,
                        sample_grid,
                        level_weight,
                    )
                    if (
                        self.activation_checkpointing
                        and self.training
                        and torch.is_grad_enabled()
                    ):
                        level_attended = checkpoint(
                            self._sample_level,
                            *sample_inputs,
                            use_reentrant=False,
                        )
                    else:
                        level_attended = self._sample_level(*sample_inputs)
                    attended = attended + level_attended
                output_chunks.append(
                    attended.reshape(
                        batch_size,
                        chunk_size,
                        self.embed_dim,
                    )
                )
                visible_chunks.append(
                    visible.unsqueeze(-1).to(query.dtype)
                )
            view_output = torch.cat(output_chunks, dim=1)
            view_visible = torch.cat(visible_chunks, dim=1)
            camera_outputs = camera_outputs + view_output * view_visible
            camera_counts = camera_counts + view_visible
        attended = camera_outputs / camera_counts.clamp_min(1.0)
        return query + self.dropout(self.output_proj(attended))


class BEVFormerV2T1EncoderLayer(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        feedforward_channels: int,
        dropout: float,
        query_chunk_size: int,
    ) -> None:
        super().__init__()
        self.self_attention = T1DeformableSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_points=4,
            dropout=dropout,
            query_chunk_size=query_chunk_size,
        )
        self.cross_attention = MultiScaleSpatialCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
            query_chunk_size=query_chunk_size,
            activation_checkpointing=False,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_channels, embed_dim),
            nn.Dropout(dropout),
        )
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LayerNorm(embed_dim),
        ])

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        features: Sequence[torch.Tensor],
        reference_points_2d: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        bev_h: int,
        bev_w: int,
        num_views: int,
        level_embeddings: torch.Tensor,
        camera_embeddings: torch.Tensor,
        reference_depth_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.norms[0](
            self.self_attention(
                query,
                query_pos,
                bev_h=bev_h,
                bev_w=bev_w,
            )
        )
        query = self.norms[1](
            self.cross_attention(
                query,
                features,
                reference_points_2d,
                reference_mask,
                num_views=num_views,
                level_embeddings=level_embeddings,
                camera_embeddings=camera_embeddings,
                reference_depth_weight=reference_depth_weight,
            )
        )
        return self.norms[2](query + self.ffn(query))


CALIBRATION_FEATURES = 7
# Camera centres are divided by this so a few-metre rig maps to unit scale.
CALIBRATION_CENTRE_SCALE_M = 2.0


def calibration_features(matrix: torch.Tensor, image_transform: ImageTransform) -> torch.Tensor:
    """[B, V, 7] camera centre (ego, scaled), unit optical axis and focal length of P = K [R | t].

    The focal length is relative to the model-input width, so the embedding
    sees the field of view rather than the pixel count.
    """
    m = matrix[..., :3].float()
    centre = -torch.linalg.solve(m, matrix[..., 3:].float())[..., 0]
    axis = m[..., 2, :] / m[..., 2, :].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    focal = (m[..., 0, :] - (m[..., 0, :] * axis).sum(-1, keepdim=True) * axis).norm(dim=-1, keepdim=True)
    width = float(image_transform.model_input_size[0])
    return torch.cat([centre / CALIBRATION_CENTRE_SCALE_M, axis, focal / width], dim=-1)


class BEVFormerV2T1ViewFusion(nn.Module):
    """Six-layer t1 BEVFormer encoder that lifts calibrated camera features to one BEV frame.

    Every call passes the real projection and model-input image transform.
    The camera embedding is computed from each camera's calibration (its centre
    and optical axis in the ego frame and its focal length) instead of being
    indexed by camera slot, so a rig with other mounting positions gets the
    embedding of where its cameras actually look.
    """

    def __init__(
        self,
        *,
        num_views: int,
        embed_dim: int,
        bev_h: int,
        bev_w: int,
        pc_range: Sequence[float],
        num_points_in_pillar: int = 4,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
        num_encoder_layers: int = 6,
        feedforward_channels: int = 512,
        dropout: float = 0.1,
        query_chunk_size: int = 4096,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if len(pc_range) != 6:
            raise ValueError("pc_range must contain six values")
        self.num_views = num_views
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_points_in_pillar = num_points_in_pillar
        self.num_levels = num_levels
        self.activation_checkpointing = bool(activation_checkpointing)
        self.pc_range = tuple(float(value) for value in pc_range)
        self.bev_queries = nn.Embedding(bev_h * bev_w, embed_dim)
        self.row_embed = nn.Embedding(bev_h, embed_dim // 2)
        self.col_embed = nn.Embedding(bev_w, embed_dim // 2)
        self.level_embeddings = nn.Parameter(torch.empty(num_levels, embed_dim))
        self.camera_embedding = nn.Sequential(
            nn.Linear(CALIBRATION_FEATURES, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim),
        )
        self.reference_points_3d: torch.Tensor
        self.layers = nn.ModuleList([
            BEVFormerV2T1EncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                feedforward_channels=feedforward_channels,
                dropout=dropout,
                query_chunk_size=query_chunk_size,
            )
            for _ in range(num_encoder_layers)
        ])
        nn.init.normal_(self.level_embeddings)
        self._init_reference_points()

    def _init_reference_points(self) -> None:
        rows = (torch.arange(self.bev_h, dtype=torch.float32) + 0.5) / self.bev_h
        cols = (torch.arange(self.bev_w, dtype=torch.float32) + 0.5) / self.bev_w
        heights = (torch.arange(self.num_points_in_pillar, dtype=torch.float32) + 0.5) / self.num_points_in_pillar
        grid_row, grid_col, grid_height = torch.meshgrid(rows, cols, heights, indexing="ij")
        reference = torch.stack([1.0 - grid_row, 1.0 - grid_col, grid_height], dim=-1).reshape(
            self.bev_h * self.bev_w, self.num_points_in_pillar, 3,
        )
        self.register_buffer("reference_points_3d", reference)

    def _ego_reference_homo(self, reference_points_3d: torch.Tensor) -> torch.Tensor:
        reference = reference_points_3d.clone()
        pc_range = self.pc_range
        reference[..., 0] = reference[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference[..., 1] = reference[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference[..., 2] = reference[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
        ones = torch.ones(*reference.shape[:-1], 1, device=reference.device, dtype=reference.dtype)
        return torch.cat([reference, ones], dim=-1).reshape(-1, 4)

    def _query_position(self, batch_size: int) -> torch.Tensor:
        rows = self.row_embed.weight[:, None, :].expand(self.bev_h, self.bev_w, -1)
        cols = self.col_embed.weight[None, :, :].expand(self.bev_h, self.bev_w, -1)
        # Rows are ego X and columns ego Y: channels are [X embedding, Y embedding].
        position = torch.cat([rows, cols], dim=-1).reshape(1, self.bev_h * self.bev_w, self.embed_dim)
        return position.expand(batch_size, -1, -1)

    def forward(
        self,
        multi_scale_features: Sequence[torch.Tensor],
        batch_size: int,
        num_views: int,
        *,
        projection: PinholeProjection,
        image_transform: ImageTransform,
        reference_depth_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(multi_scale_features) != self.num_levels or num_views != self.num_views:
            raise ValueError("camera features differ from the encoder contract")
        result = projection.project_ego_to_image(
            self._ego_reference_homo(self.reference_points_3d), image_transform,
        )
        query_count = self.bev_h * self.bev_w
        reference_2d = result.uv_norm.reshape(
            batch_size, num_views, query_count, self.num_points_in_pillar, 2)
        reference_mask = result.valid_mask.reshape(
            batch_size, num_views, query_count, self.num_points_in_pillar)
        camera_embeddings = self.camera_embedding(
            calibration_features(projection.matrix, image_transform).to(self.level_embeddings.dtype))
        query = self.bev_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        position = self._query_position(batch_size).to(query.dtype)
        for layer in self.layers:
            arguments = (
                layer, self.bev_h, self.bev_w, num_views, query, position, reference_2d, reference_mask,
                self.level_embeddings, camera_embeddings, reference_depth_weight, *multi_scale_features,
            )
            query = (
                checkpoint(_checkpoint_encoder_layer, *arguments, use_reentrant=True)
                if self.activation_checkpointing and self.training and torch.is_grad_enabled()
                else _checkpoint_encoder_layer(*arguments)
            )
        observed = reference_mask.any(dim=3).any(dim=1)
        query = query * observed.unsqueeze(-1).to(query.dtype)
        return query.reshape(batch_size, self.bev_h, self.bev_w, self.embed_dim).permute(0, 3, 1, 2).contiguous()
