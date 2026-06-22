import math

import torch
import torch.nn as nn

from .base import BasePlanner
from ..losses.trajectory_loss import TrajectoryImitationLoss


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

    Output contract (matches ``TrajectoryPlanner``):
        trajectory: [B, num_timesteps * num_signals]
        ego_hidden: [B, embed_dim]   (consumed by FutureState)
    """

    def __init__(self, embed_dim=256, num_timesteps=64, num_signals=2,
                 num_controls=5, egomotion_dim=256, visual_history_dim=896):
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
        self.context_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Predict (num_controls x num_signals) Bezier control points.
        self.control_head = nn.Linear(embed_dim, num_controls * num_signals)

        # Imitation loss is owned by the shared losses/ module (smooth_l1 +
        # temporal decay), not reimplemented here. The planner only invokes it.
        self.trajectory_loss = TrajectoryImitationLoss(
            num_timesteps=num_timesteps, num_signals=num_signals,
        )

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
                **kwargs):
        """
        Args:
            bev_features: [B, embed_dim, H, W] — any spatial resolution.
            visual_history: [B, visual_history_dim].
            egomotion_history: [B, egomotion_dim].

        Returns:
            trajectory: [B, num_timesteps * num_signals]
            ego_hidden: [B, embed_dim]
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
        ego_hidden = self.context_mlp(context)                          # [B, C]

        control_points = self.control_head(ego_hidden).view(
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
        return trajectory, ego_hidden

    def compute_planner_loss(self, bev_features, visual_history,
                             egomotion_history, trajectory_target):
        """Return ``(loss, ego_hidden)`` as required by ``BasePlanner``.

        The loss is NOT defined here: it delegates to the shared
        ``TrajectoryImitationLoss`` in ``Model/model_components/losses`` so the
        planner owns no loss logic of its own. ``ego_hidden`` is the same
        context vector ``forward()`` produces, so ``AutoE2E`` can feed
        ``FutureState`` in train mode without a second pass.
        """
        trajectory, ego_hidden = self(
            bev_features, visual_history, egomotion_history
        )
        loss = self.trajectory_loss(trajectory, trajectory_target)
        return loss, ego_hidden
