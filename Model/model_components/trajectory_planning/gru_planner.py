"""Deterministic GRU planner with deformable BEV feature lookup."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BasePlanner
from .reasoning_coupling import ReasoningCoupling


class GRUPlanner(BasePlanner):
    """Decode acceleration and curvature with a recurrent ego query."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_timesteps: int = 64,
        num_signals: int = 2,
        num_points: int = 8,
        egomotion_dim: int = 256,
        visual_history_dim: int = 896,
        offset_scale: float = 0.1,
        reasoning_mode: str = "none",
    ) -> None:
        super().__init__()
        if (
            not isinstance(offset_scale, (int, float))
            or isinstance(offset_scale, bool)
            or not math.isfinite(offset_scale)
            or offset_scale < 0.0
        ):
            raise ValueError(
                "offset_scale must be a finite non-negative number"
            )
        if num_timesteps <= 0 or num_signals <= 0 or num_points <= 0:
            raise ValueError("planner dimensions must be positive")
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps
        self.num_signals = num_signals
        self.num_points = num_points
        self.egomotion_dim = egomotion_dim
        self.visual_history_dim = visual_history_dim
        self.offset_scale = float(offset_scale)

        self.ego_query = nn.Embedding(1, embed_dim)
        self.ego_state_proj = nn.Linear(egomotion_dim, embed_dim)
        self.visual_history_proj = nn.Linear(visual_history_dim, embed_dim)
        self.reasoning_coupling = ReasoningCoupling(
            embed_dim,
            mode=reasoning_mode,
        )
        self.reference_point = nn.Linear(embed_dim, 2)
        self.sampling_offsets = nn.Linear(embed_dim, num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim)
        self.control_head = nn.Linear(embed_dim, num_signals)

    def _cross_attend(
        self,
        query: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        reference = self.reference_point(query).sigmoid()
        offsets = self.sampling_offsets(query).reshape(
            batch_size,
            self.num_points,
            2,
        )
        locations = (
            reference.unsqueeze(1) + offsets * self.offset_scale
        ).clamp(0.0, 1.0)
        grid = (locations * 2.0 - 1.0).unsqueeze(2)
        sampled = F.grid_sample(
            values,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        weights = self.attention_weights(query).softmax(dim=-1)
        return self.output_proj(
            (sampled * weights.unsqueeze(-1)).sum(dim=1)
        )

    def forward(
        self,
        bev_features: torch.Tensor,
        visual_history: torch.Tensor,
        egomotion_history: torch.Tensor,
        reasoning_latent: torch.Tensor | None = None,
        reasoning_horizon_tokens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if visual_history.shape[-1] != self.visual_history_dim:
            raise ValueError(
                "visual_history last dimension differs from planner contract"
            )
        if egomotion_history.shape[-1] != self.egomotion_dim:
            raise ValueError(
                "egomotion_history last dimension differs from planner contract"
            )
        context = (
            self.ego_state_proj(egomotion_history)
            + self.visual_history_proj(visual_history)
        )
        context = self.reasoning_coupling(
            context,
            reasoning_latent=reasoning_latent,
            horizon_tokens=reasoning_horizon_tokens,
        )
        hidden = context.unsqueeze(0)
        values = self.value_proj(
            bev_features.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2).contiguous()
        ego_query = self.ego_query.weight
        controls = []
        for _ in range(self.num_timesteps):
            attended = self._cross_attend(
                hidden.squeeze(0) + ego_query,
                values,
            )
            _, hidden = self.gru(attended.unsqueeze(0), hidden)
            controls.append(self.control_head(hidden.squeeze(0)))
        return torch.cat(controls, dim=1)
