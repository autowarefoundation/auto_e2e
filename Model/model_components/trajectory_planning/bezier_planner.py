import math

import torch
import torch.nn as nn

from .base import BasePlanner
from .reasoning_coupling import ReasoningCoupling


class BezierPlanner(BasePlanner):
    """Bezier-smoothed trajectory planner (optional, swappable).

    Instead of decoding waypoints autoregressively, this head predicts a small
    set of Bernstein control points and expands them, through a fixed
    precomputed Bernstein basis, into the full ``num_timesteps`` x
    ``num_signals`` trajectory. Because the trajectory is a linear combination
    of only ``num_controls`` control points, the resulting control-signal
    profiles are smooth **by construction** (low jerk) with no post-processing
    filter and no trainable smoothness penalty.

    IMPORTANT — output semantics. The trajectory follows the repository's
    official unicycle-dynamics contract: each of the ``num_signals`` channels
    is a control signal, by default ``(acceleration, curvature)`` — NOT raw
    Cartesian ``(x, y)`` waypoints. The Bezier smoothing is therefore applied
    to the acceleration/curvature *profiles*, which is exactly what reduces
    jerk and yields controller-friendly, physically feasible commands.

    """

    def __init__(self, embed_dim=256, num_timesteps=64, num_signals=2,
                 num_controls=5, egomotion_dim=256, visual_history_dim=896,
                 reasoning_mode="none"):
        super().__init__()
        if num_controls < 2:
            raise ValueError(
                f"num_controls must be >= 2 to define a Bezier curve, "
                f"got {num_controls}."
            )
        if num_controls > num_timesteps:
            raise ValueError(
                f"num_controls ({num_controls}) cannot exceed num_timesteps "
                f"({num_timesteps})."
            )
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps
        self.num_signals = num_signals
        self.num_controls = num_controls
        self.egomotion_dim = egomotion_dim
        self.visual_history_dim = visual_history_dim

        # Context aggregation: ego state + visual history + global BEV summary.
        self.ego_state_proj = nn.Linear(egomotion_dim, embed_dim)
        self.visual_history_proj = nn.Linear(visual_history_dim, embed_dim)
        self.bev_proj = nn.Linear(embed_dim, embed_dim)
        # Zero-init the visual-history projection so the World-Model-derived
        # visual_history starts as a STRICT no-op and the planner learns to open
        # it only as the WM matures — mirroring the reasoning branch's zero-init
        # coupling (alpha=0). Rationale (#13): with the WM on, visual_history is a
        # non-stationary WM output; a default-init projection makes the planner
        # depend on that moving, partly-eval-unavailable signal from step 0, which
        # empirically made the full 3-branch model slightly WORSE than the
        # imitation baseline (visual_history=zeros). Zero-init makes the WM branch
        # strictly additive: the planner is identical to the imitation baseline at
        # init and can only improve as it learns to use a trained visual_history.
        nn.init.zeros_(self.visual_history_proj.weight)
        nn.init.zeros_(self.visual_history_proj.bias)
        self.context_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Reasoning coupling (zero-init; no-op at init). Injects the reasoning
        # branch into the planner context before the control head.
        self.reasoning_coupling = ReasoningCoupling(embed_dim, mode=reasoning_mode)

        # Predict (num_controls x num_signals) Bezier control points.
        self.control_head = nn.Linear(embed_dim, num_controls * num_signals)


        # Fixed Bernstein basis [num_timesteps, num_controls] (no scipy).
        self.register_buffer(
            "bernstein_basis",
            self._bernstein_basis(num_timesteps, num_controls),
            persistent=False,
        )

    @staticmethod
    def _bernstein_basis(num_points, num_controls):
        """Bernstein polynomial basis B_{i,n}(t), n = num_controls - 1.

        Returns a [num_points, num_controls] matrix evaluated on a uniform
        grid t in [0, 1]. Uses ``math.comb`` from the standard library, so
        there is no ``scipy`` dependency.
        """
        n = num_controls - 1
        t = torch.linspace(0.0, 1.0, num_points).unsqueeze(1)          # [P, 1]
        i = torch.arange(num_controls, dtype=torch.float32).unsqueeze(0)  # [1, C]
        comb = torch.tensor(
            [math.comb(n, k) for k in range(num_controls)],
            dtype=torch.float32,
        ).unsqueeze(0)                                                  # [1, C]
        # B_{i,n}(t) = C(n, i) * t^i * (1 - t)^(n - i)
        basis = comb * (t ** i) * ((1.0 - t) ** (n - i))               # [P, C]
        return basis

    def forward(self, bev_features, visual_history, egomotion_history,
                reasoning_latent=None, reasoning_horizon_tokens=None,
                **kwargs):
        """
        Args:
            bev_features: [B, embed_dim, H, W] — any spatial resolution.
            visual_history: [B, visual_history_dim].
            egomotion_history: [B, egomotion_dim].
            reasoning_latent: optional [B, embed_dim] pooled reasoning latent
                (used by reasoning_mode="pooled_latent").
            reasoning_horizon_tokens: optional [B, 5, embed_dim] per-horizon
                reasoning tokens (used by reasoning_mode="horizon_cross_attention").

        Returns:
            trajectory: [B, num_timesteps * num_signals]
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

        B = bev_features.shape[0]

        # Global BEV summary via spatial mean: [B, embed_dim].
        bev_context = bev_features.mean(dim=(2, 3))

        context = (
            self.ego_state_proj(egomotion_history)
            + self.visual_history_proj(visual_history)
            + self.bev_proj(bev_context)
        )
        # Zero-init reasoning residual (no-op at init; see ReasoningCoupling).
        context = self.reasoning_coupling(
            context,
            reasoning_latent=reasoning_latent,
            horizon_tokens=reasoning_horizon_tokens,
        )
        bezier_feature = self.context_mlp(context)                          # [B, C]

        control_points = self.control_head(bezier_feature).view(
            B, self.num_controls, self.num_signals
        )                                                               # [B, C, S]

        # Expand control points through the fixed Bernstein basis:
        # [P, C] x [B, C, S] -> [B, P, S]
        trajectory = torch.einsum(
            "pc,bcs->bps", self.bernstein_basis, control_points
        )
        trajectory = trajectory.reshape(
            B, self.num_timesteps * self.num_signals
        )
        return trajectory

