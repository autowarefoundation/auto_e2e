"""Deterministic GRU planner with deformable BEV feature lookup."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sampling offsets are scaled into a tenth of the BEV per unit, so the first
# steps look near the predicted reference point.
OFFSET_SCALE: Final = 0.1


class GRUPlanner(nn.Module):
    """Decode ``num_timesteps`` controls with a recurrent ego query over the BEV.

    The ego state initializes the hidden state; at every step the query reads
    ``num_points`` deformable samples of the BEV and the GRU emits one control.
    """

    def __init__(
        self,
        embed_dim: int,
        num_timesteps: int,
        num_signals: int,
        num_points: int,
        egomotion_dim: int,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_points = num_points
        self.ego_query = nn.Embedding(1, embed_dim)
        self.ego_state_proj = nn.Linear(egomotion_dim, embed_dim)
        self.reference_point = nn.Linear(embed_dim, 2)
        self.sampling_offsets = nn.Linear(embed_dim, num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim)
        self.control_head = nn.Linear(embed_dim, num_signals)

    def _cross_attend(self, query: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        batch_size = query.shape[0]
        reference = self.reference_point(query).sigmoid()
        offsets = self.sampling_offsets(query).reshape(batch_size, self.num_points, 2)
        locations = (reference.unsqueeze(1) + offsets * OFFSET_SCALE).clamp(0.0, 1.0)
        sampled = F.grid_sample(
            values, (locations * 2.0 - 1.0).unsqueeze(2), mode="bilinear", padding_mode="zeros",
            align_corners=False,
        ).squeeze(-1).permute(0, 2, 1)
        weights = self.attention_weights(query).softmax(dim=-1)
        return self.output_proj((sampled * weights.unsqueeze(-1)).sum(dim=1))

    def forward(self, bev_features: torch.Tensor, egomotion_history: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` BEV and ``[B, egomotion_dim]`` ego state to ``[B, num_timesteps * num_signals]``."""
        hidden = self.ego_state_proj(egomotion_history).unsqueeze(0)
        values = self.value_proj(bev_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        controls = []
        for _ in range(self.num_timesteps):
            # Keep deformable lookup gradients out of the recurrent state history.
            query = hidden.squeeze(0).detach() + self.ego_query.weight
            _, hidden = self.gru(self._cross_attend(query, values).unsqueeze(0), hidden)
            controls.append(self.control_head(hidden.squeeze(0)))
        return torch.cat(controls, dim=1)
