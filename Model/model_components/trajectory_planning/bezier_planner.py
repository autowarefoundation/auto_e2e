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
    profiles are smooth by construction (low jerk) with no post-processing
    filter and no trainable smoothness penalty.

    Output semantics follow the unicycle-dynamics contract: each of the
    ``num_signals`` channels is a control signal, by default
    ``(acceleration, curvature)``, not Cartesian ``(x, y)`` waypoints, so the
    smoothing applies to the control profiles.

    ``bev_features`` is the pooled fused-BEV feature vector
    ``[B, feature_dim]`` produced by ``ReactiveE2E.FusedFeaturePooling``
    (``75 * 50 = 3750`` by default), not a spatial ``[B, C, H, W]`` map. The
    pooling module owns the resolution invariance; the planner concatenates the
    pooled BEV vector with the ego-motion context and the zero-init projected
    visual history.
    """

    def __init__(self, feature_dim = 3750, embed_dim = 256, num_timesteps=64, num_signals=2,
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
      
        self.feature_dim = feature_dim
        self.num_timesteps = num_timesteps
        self.num_signals = num_signals
        self.num_controls = num_controls
        self.egomotion_dim = egomotion_dim
        self.visual_history_dim = visual_history_dim
        self.combined_feature_dim = feature_dim + egomotion_dim + visual_history_dim
        self.reduced_combined_feature_dim = int(self.combined_feature_dim/2)

        self.visual_history_proj = nn.Linear(visual_history_dim, visual_history_dim)
        # Zero-init so the world-model visual_history starts as a no-op: it is a
        # non-stationary signal, and the planner matches the imitation baseline
        # at init until it learns to use it.
        nn.init.zeros_(self.visual_history_proj.weight)
        nn.init.zeros_(self.visual_history_proj.bias)
        self.context_mlp = nn.Sequential(
            nn.Linear(self.combined_feature_dim, self.reduced_combined_feature_dim),
            nn.GELU(),
            nn.Linear(self.reduced_combined_feature_dim, self.reduced_combined_feature_dim),
        )

        # The concatenated context is wider than the reasoning width, so the
        # coupling projects its residual into it.
        self.reasoning_coupling = ReasoningCoupling(
            embed_dim,
            mode=reasoning_mode,
            context_dim=self.combined_feature_dim,
        )

        self.control_head = nn.Linear(self.reduced_combined_feature_dim, num_controls * num_signals)

        # Fixed Bernstein basis [num_timesteps, num_controls].
        self.register_buffer(
            "bernstein_basis",
            self._bernstein_basis(num_timesteps, num_controls),
            persistent=False,
        )

    @staticmethod
    def _bernstein_basis(num_points, num_controls):
        """Bernstein polynomial basis B_{i,n}(t), n = num_controls - 1.

        Returns a [num_points, num_controls] matrix evaluated on a uniform
        grid t in [0, 1].
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
            bev_features: [B, feature_dim] pooled fused-BEV vector (see
                ``FusedFeaturePooling``).
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

        if bev_features.dim() != 2 or bev_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"bev_features must be a pooled [B, {self.feature_dim}] vector "
                f"(see FusedFeaturePooling), got tensor of shape "
                f"{tuple(bev_features.shape)}."
            )

        B = bev_features.shape[0]

        context = torch.cat((
            bev_features,
            egomotion_history,
            self.visual_history_proj(visual_history)), dim=1)

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

    def _compute_loss_and_trajectory(
        self,
        bev_features,
        visual_history,
        egomotion_history,
        trajectory_target,
        *,
        loss_fn=None,
        reasoning_latent=None,
        reasoning_horizon_tokens=None,
        **kwargs,
    ):
        return self._compute_imitation_loss_and_trajectory(
            bev_features,
            visual_history,
            egomotion_history,
            trajectory_target,
            loss_fn=loss_fn,
            reasoning_latent=reasoning_latent,
            reasoning_horizon_tokens=reasoning_horizon_tokens,
            **kwargs,
        )

    def compute_planner_loss(
        self,
        bev_features,
        visual_history,
        egomotion_history,
        trajectory_target,
        *,
        loss_fn=None,
        reasoning_latent=None,
        reasoning_horizon_tokens=None,
        **kwargs,
    ):
        """Compute trajectory imitation loss on the Bezier rollout."""
        loss_dict, _ = self._compute_loss_and_trajectory(
            bev_features,
            visual_history,
            egomotion_history,
            trajectory_target,
            loss_fn=loss_fn,
            reasoning_latent=reasoning_latent,
            reasoning_horizon_tokens=reasoning_horizon_tokens,
            **kwargs,
        )
        return loss_dict

