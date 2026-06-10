import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BasePlanner


def _validate_offset_scale(offset_scale):
    if not isinstance(offset_scale, (int, float)) or isinstance(offset_scale, bool):
        raise ValueError(
            f"offset_scale must be a finite non-negative number, "
            f"got {offset_scale!r}."
        )
    if not math.isfinite(offset_scale) or offset_scale < 0:
        raise ValueError(
            f"offset_scale must be a finite non-negative number, "
            f"got {offset_scale!r}."
        )


class GRUPlanner(BasePlanner):
    """Autoregressive trajectory decoder with deformable cross-attention to BEV.

    Replaces the flatten-then-MLP DrivingPolicy. A single learnable ego query
    travels through time via a GRU; at every timestep it cross-attends to the
    BEV feature map by predicting a reference point and sampling offsets and
    looking up features with bilinear ``F.grid_sample`` (the same pattern used
    in BEVViewFusion). This keeps the planner shape-agnostic with respect to
    the BEV grid, so the BEV resolution can change without retraining the head.

    Outputs at each of ``num_timesteps`` steps the ``num_signals`` waypoint
    components (default 2: acceleration and curvature). The final GRU hidden
    state (``ego_hidden``, 256-dim) replaces the legacy 14-dim compressed
    visual feature vector and is consumed downstream by FutureState.
    """

    def __init__(self, embed_dim=256, num_timesteps=64, num_signals=2,
                 num_points=8, egomotion_dim=256, visual_history_dim=896,
                 offset_scale=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps
        self.num_signals = num_signals
        self.num_points = num_points
        self.egomotion_dim = egomotion_dim
        self.visual_history_dim = visual_history_dim
        # offset_scale bounds the per-point sampling offset around the predicted
        # reference point in normalized BEV coordinates. The default 0.1 means
        # offsets reach up to 10% of the BEV grid extent in each direction. The
        # reference point itself is sigmoid-bounded to [0, 1], so the head can
        # still attend anywhere on the grid — offset_scale only constrains the
        # local fan-out around that anchor.
        _validate_offset_scale(offset_scale)
        self.offset_scale = offset_scale

        self.ego_query = nn.Embedding(1, embed_dim)
        self.ego_state_proj = nn.Linear(egomotion_dim, embed_dim)
        # visual_history carries frame-to-frame visual memory (default 896 =
        # 64 frames × 14-dim compressed per frame), distinct from the GRU's
        # intra-trajectory temporal coherence. Both signals are summed into
        # the initial hidden state so the planner conditions on past dynamics
        # AND past scene context from step 0.
        self.visual_history_proj = nn.Linear(visual_history_dim, embed_dim)

        # Deformable cross-attention parameters predicted from the query state
        self.reference_point = nn.Linear(embed_dim, 2)
        self.sampling_offsets = nn.Linear(embed_dim, num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.gru = nn.GRU(embed_dim, embed_dim)
        self.waypoint_head = nn.Linear(embed_dim, num_signals)

    def _deformable_cross_attn(self, query, values):
        """Deformable cross-attention to BEV via grid_sample.

        Args:
            query: [B, C] — current query state (GRU hidden + ego query bias).
            values: [B, C, H, W] — value-projected BEV features.

        Returns:
            attended: [B, C]
        """
        B = query.shape[0]

        ref_point = self.reference_point(query).sigmoid()                  # [B, 2]
        offsets = self.sampling_offsets(query).reshape(B, self.num_points, 2)
        offsets = offsets * self.offset_scale
        attn_w = self.attention_weights(query).softmax(dim=-1)             # [B, P]

        sample_locs = ref_point.unsqueeze(1) + offsets                     # [B, P, 2]
        sample_locs = sample_locs.clamp(0, 1)
        sample_grid = (sample_locs * 2 - 1).unsqueeze(2)                   # [B, P, 1, 2]

        sampled = F.grid_sample(
            values, sample_grid, mode='bilinear',
            padding_mode='zeros', align_corners=False,
        )                                                                  # [B, C, P, 1]
        sampled = sampled.squeeze(-1).permute(0, 2, 1)                     # [B, P, C]

        attended = (sampled * attn_w.unsqueeze(-1)).sum(dim=1)             # [B, C]
        return self.output_proj(attended)

    def forward(self, bev_features, visual_history, egomotion_history,
                mode="train", **kwargs):
        """
        Args:
            bev_features: [B, embed_dim, H, W] — any spatial resolution.
            visual_history: [B, visual_history_dim].
            egomotion_history: [B, egomotion_dim].
            mode: "train" or "infer". GRU planner is mode-agnostic; the
                argument is accepted for interface compatibility with
                stochastic decoders (Flow Matching, diffusion).
            **kwargs: ignored. Accepts extra inputs (e.g. trajectory_target,
                noisy_trajectory, flow_timestep) that other planners require
                so callers can stay planner-agnostic.

        Returns:
            trajectory: [B, num_timesteps * num_signals]
            ego_hidden: [B, embed_dim] — final GRU hidden state.
        """
        if visual_history.shape[-1] != self.visual_history_dim:
            raise ValueError(
                f"visual_history last dim must be {self.visual_history_dim}, "
                f"got tensor of shape {tuple(visual_history.shape)}."
            )
        if egomotion_history.shape[-1] != self.egomotion_dim:
            raise ValueError(
                f"egomotion_history last dim must be {self.egomotion_dim}, "
                f"got tensor of shape {tuple(egomotion_history.shape)}."
            )

        # Initialize GRU hidden state from ego state + visual history: [1, B, C]
        h = (self.ego_state_proj(egomotion_history)
             + self.visual_history_proj(visual_history)).unsqueeze(0)

        # Pre-project BEV features for value lookup, preserving spatial layout
        # required by grid_sample: [B, C, H, W]
        bev_perm = bev_features.permute(0, 2, 3, 1)                        # [B, H, W, C]
        values = self.value_proj(bev_perm).permute(0, 3, 1, 2).contiguous()

        ego_q = self.ego_query.weight                                      # [1, C]

        waypoints = []
        # NOTE: this loop is inherently sequential: each step's query depends on
        # the previous GRU hidden state, so the 64 iterations cannot be batched
        # along the time axis. This is a known latency concern; future work may
        # replace it with non-autoregressive decoding (parallel waypoint heads
        # over a fixed set of learned time queries) to amortize cost.
        for _ in range(self.num_timesteps):
            query = h.squeeze(0) + ego_q                                   # [B, C]
            attended = self._deformable_cross_attn(query, values)          # [B, C]

            _, h = self.gru(attended.unsqueeze(0), h)                      # h: [1, B, C]
            waypoints.append(self.waypoint_head(h.squeeze(0)))             # [B, num_signals]

        trajectory = torch.cat(waypoints, dim=1)                           # [B, T*S]
        ego_hidden = h.squeeze(0)                                          # [B, C]
        return trajectory, ego_hidden
