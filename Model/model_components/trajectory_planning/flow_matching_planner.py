import math

import torch
import torch.nn as nn

from .base import BasePlanner


class FlowMatchingPlanner(BasePlanner):
    """Flow Matching trajectory decoder.

    Replaces the autoregressive GRU loop with a conditional vector field
    v_theta(u_t, t, c) trained to map a noise prior x_0 ~ N(0, I) to the
    target trajectory x_1 along the linear path
    ``u_t = (1 - t) * x_0 + t * x_1``. Following Lipman et al. (2023), the
    target velocity at u_t is simply ``x_1 - x_0``, so training reduces to
    a per-sample MSE between v_theta and that constant velocity.

    At inference, we sample a fresh noise tensor and integrate
    ``dx/dt = v_theta(x, t, c)`` from t=0 to t=1 with a fixed-step Euler
    solver (``num_inference_steps`` steps). The full conditioning context
    c is built once per sample from BEV mean-pooling, visual history, and
    egomotion history; it is reused across all integration steps so the
    ODE call is cheap.

    Outputs match the GRU planner contract: ``(trajectory, ego_hidden)``
    where ``ego_hidden`` is a learned projection of the conditioning and
    is consumed downstream by FutureState. In training mode the first
    return is the *predicted velocity* at the sampled (u_t, t), not a
    trajectory — the caller pairs it with the matching target velocity
    when computing the flow-matching loss.
    """

    def __init__(self, embed_dim=256, num_timesteps=64, num_signals=2,
                 egomotion_dim=256, visual_history_dim=896,
                 num_inference_steps=10, hidden_dim=512, time_embed_dim=128):
        super().__init__()

        if num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1, got {num_inference_steps}."
            )
        if time_embed_dim % 2 != 0:
            raise ValueError(
                f"time_embed_dim must be even, got {time_embed_dim}."
            )

        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps
        self.num_signals = num_signals
        self.trajectory_dim = num_timesteps * num_signals
        self.egomotion_dim = egomotion_dim
        self.visual_history_dim = visual_history_dim
        self.num_inference_steps = num_inference_steps
        self.time_embed_dim = time_embed_dim

        # Conditioning encoders mirror GRUPlanner so swapping planners is
        # weight-shape-comparable for the ego/visual_history paths.
        self.ego_state_proj = nn.Linear(egomotion_dim, embed_dim)
        self.visual_history_proj = nn.Linear(visual_history_dim, embed_dim)
        self.bev_pool_proj = nn.Linear(embed_dim, embed_dim)
        self.cond_to_ego_hidden = nn.Linear(embed_dim, embed_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        v_in = self.trajectory_dim + 2 * embed_dim
        self.v_net = nn.Sequential(
            nn.Linear(v_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.trajectory_dim),
        )

    def _validate_inputs(self, visual_history, egomotion_history):
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

    def _sinusoidal_time_embedding(self, t):
        """Map t in [0, 1] to a sinusoidal embedding of size time_embed_dim.

        Args:
            t: [B] — flow timesteps.

        Returns:
            [B, time_embed_dim] embedding.
        """
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def _build_conditioning(self, bev_features, visual_history, egomotion_history):
        # Spatial mean-pool keeps the velocity head BEV-resolution-agnostic.
        bev_pool = bev_features.mean(dim=(2, 3))
        return (
            self.bev_pool_proj(bev_pool)
            + self.visual_history_proj(visual_history)
            + self.ego_state_proj(egomotion_history)
        )

    def _construct_training_data(self, trajectory_target):
        """Sample (u_t, t, target_velocity) for one flow-matching training step.

        Returns:
            u_t: [B, trajectory_dim] — the noisy interpolated state.
            t: [B] — flow timesteps in [0, 1].
            target_velocity: [B, trajectory_dim] — the true velocity x_1 - x_0
                that v_theta should predict at (u_t, t).
        """
        B = trajectory_target.shape[0]
        x_0 = torch.randn_like(trajectory_target)
        t = torch.rand(B, device=trajectory_target.device,
                       dtype=trajectory_target.dtype)
        u_t = (1.0 - t).unsqueeze(-1) * x_0 + t.unsqueeze(-1) * trajectory_target
        target_velocity = trajectory_target - x_0
        return u_t, t, target_velocity

    def _v_theta(self, u_t, t, cond):
        """Conditional velocity network.

        Args:
            u_t: [B, trajectory_dim]
            t: [B]
            cond: [B, embed_dim]

        Returns:
            velocity: [B, trajectory_dim]
        """
        t_emb = self.time_mlp(self._sinusoidal_time_embedding(t))
        return self.v_net(torch.cat([u_t, t_emb, cond], dim=-1))

    def forward(self, bev_features, visual_history, egomotion_history,
                mode="train", trajectory_target=None,
                noisy_trajectory=None, flow_timestep=None, **kwargs):
        """
        Args:
            bev_features: [B, embed_dim, H, W].
            visual_history: [B, visual_history_dim].
            egomotion_history: [B, egomotion_dim].
            mode: "train" returns predicted velocity at a sampled (u_t, t);
                anything else (e.g. "infer") integrates the ODE from noise
                to trajectory.
            trajectory_target: [B, trajectory_dim], required in train mode
                unless ``noisy_trajectory`` and ``flow_timestep`` are both
                supplied. Used to sample (u_t, t, target_velocity).
            noisy_trajectory: optional pre-sampled u_t for train mode, lets
                the caller share the same (u_t, t) across loss components
                without re-sampling.
            flow_timestep: optional pre-sampled t paired with
                ``noisy_trajectory``.
            **kwargs: ignored.

        Returns:
            train mode: (velocity_pred [B, trajectory_dim], ego_hidden [B, embed_dim])
            infer mode: (trajectory [B, trajectory_dim], ego_hidden [B, embed_dim])
        """
        self._validate_inputs(visual_history, egomotion_history)
        cond = self._build_conditioning(
            bev_features, visual_history, egomotion_history
        )
        ego_hidden = self.cond_to_ego_hidden(cond)

        if mode == "train":
            if noisy_trajectory is not None and flow_timestep is not None:
                u_t, t = noisy_trajectory, flow_timestep
            elif trajectory_target is not None:
                u_t, t, _ = self._construct_training_data(trajectory_target)
            else:
                raise ValueError(
                    "FlowMatchingPlanner train mode requires either "
                    "trajectory_target, or both noisy_trajectory and "
                    "flow_timestep."
                )
            velocity_pred = self._v_theta(u_t, t, cond)
            return velocity_pred, ego_hidden

        # Inference: Euler-integrate dx/dt = v_theta(x, t, cond) over [0, 1].
        B = bev_features.shape[0]
        x = torch.randn(B, self.trajectory_dim,
                        device=bev_features.device, dtype=bev_features.dtype)
        dt = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            t_val = step * dt
            t = torch.full((B,), t_val,
                           device=bev_features.device, dtype=bev_features.dtype)
            v = self._v_theta(x, t, cond)
            x = x + dt * v
        return x, ego_hidden
