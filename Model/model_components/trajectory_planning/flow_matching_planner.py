import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BasePlanner


class FlowMatchingPlanner(BasePlanner):
    """Flow Matching trajectory decoder with BEV cross-attention.

    Replaces the autoregressive GRU loop with a conditional vector field
    v_theta(u_t, t, c) trained to map a noise prior x_0 ~ N(0, I) to the
    target trajectory x_1 along the linear path
    ``u_t = (1 - t) * x_0 + t * x_1``. Following Lipman et al. (2023), the
    target velocity at u_t is simply ``x_1 - x_0``, so training reduces to
    a per-sample MSE between v_theta and that constant velocity. Training
    timesteps are drawn by default from a shifted Beta(1.5, 1) schedule
    (``t = 0.999 - 0.999 * Beta(1.5, 1).sample()``) following pi-0.5 /
    Alpamayo, biasing samples toward the noisy (low-t) end where the
    velocity field is hardest to learn.

    The velocity network preserves the BEV grid's spatial structure: the
    noisy trajectory is treated as a sequence of ``num_timesteps`` action
    tokens (each ``num_signals``-dimensional) which act as queries over a
    flattened BEV spatial map (``H*W`` keys/values) via multi-head
    cross-attention. Time and the ego/visual_history conditioning are
    injected on the attention output through AdaLN-style affine
    modulation (gamma, beta) — the DiT pattern adapted to flow matching.
    A per-token velocity head maps each attended action token back to
    ``num_signals`` and the output is reshaped to ``(B, T*num_signals)``.

    At inference, we sample a fresh noise tensor and integrate
    ``dx/dt = v_theta(x, t, c)`` from t=0 to t=1 with a fixed-step Euler
    solver (``num_inference_steps`` steps). The BEV map and the
    modulation conditioning are computed once per sample and reused
    across all integration steps so the ODE call is cheap.

    Outputs match the GRU planner contract: ``(trajectory, ego_hidden)``
    where ``ego_hidden`` is a learned projection of pooled BEV plus
    visual_history and ego state, and is consumed downstream by
    FutureState. ``forward()`` always returns the integrated trajectory;
    the flow-matching loss lives in ``compute_planner_loss`` so the raw
    velocity tensor never escapes the planner — the caller cannot pair it
    with the wrong target.
    """

    def __init__(self, embed_dim=256, num_timesteps=64, num_signals=2,
                 egomotion_dim=256, visual_history_dim=896,
                 num_inference_steps=10, time_embed_dim=128, num_heads=4,
                 timestep_sampler="beta", beta_alpha=1.5, beta_scale=0.999):
        super().__init__()

        if num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1, got {num_inference_steps}."
            )
        if time_embed_dim % 2 != 0:
            raise ValueError(
                f"time_embed_dim must be even, got {time_embed_dim}."
            )
        if timestep_sampler not in ("beta", "uniform"):
            raise ValueError(
                f"timestep_sampler must be 'beta' or 'uniform', "
                f"got {timestep_sampler!r}."
            )
        if not (isinstance(beta_alpha, (int, float))
                and not isinstance(beta_alpha, bool)
                and math.isfinite(beta_alpha) and beta_alpha > 0):
            raise ValueError(
                f"beta_alpha must be a finite positive number, got {beta_alpha!r}."
            )
        if not (isinstance(beta_scale, (int, float))
                and not isinstance(beta_scale, bool)
                and math.isfinite(beta_scale)
                and 0 < beta_scale <= 1):
            raise ValueError(
                f"beta_scale must satisfy 0 < beta_scale <= 1, got {beta_scale!r}."
            )

        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps
        self.num_signals = num_signals
        self.trajectory_dim = num_timesteps * num_signals
        self.egomotion_dim = egomotion_dim
        self.visual_history_dim = visual_history_dim
        self.num_inference_steps = num_inference_steps
        self.time_embed_dim = time_embed_dim
        self.num_heads = num_heads

        # Flow timestep schedule. The default is the pi-0.5 / Alpamayo
        # shifted Beta(beta_alpha, 1) sampler, which biases t toward the
        # noisy (low-t) end and empirically improves flow-matching policy
        # training; "uniform" recovers the textbook U(0, 1).
        self.timestep_sampler = timestep_sampler
        self.beta_alpha = float(beta_alpha)
        self.beta_scale = float(beta_scale)
        self._beta_dist = torch.distributions.Beta(
            torch.tensor(self.beta_alpha), torch.tensor(1.0),
        )

        # Conditioning encoders for the AdaLN modulation path. BEV is NOT
        # pooled into this conditioning — it enters the velocity field via
        # cross-attention to preserve spatial detail.
        self.ego_state_proj = nn.Linear(egomotion_dim, embed_dim)
        self.visual_history_proj = nn.Linear(visual_history_dim, embed_dim)

        # ego_hidden is a single summary vector consumed by FutureState, so
        # it can still pool BEV — its job is "scene gist", not waypoint
        # placement.
        self.bev_pool_proj = nn.Linear(embed_dim, embed_dim)
        self.cond_to_ego_hidden = nn.Linear(embed_dim, embed_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Per-timestep action token projection: each (acc, curv) pair becomes
        # an embed_dim query.
        self.action_proj = nn.Linear(num_signals, embed_dim)
        self.bev_kv_proj = nn.Linear(embed_dim, embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )

        # AdaLN: produce (gamma, beta) from (time + visual_history + ego).
        # The LayerNorm has no affine — gamma/beta supply the scale and shift.
        self.attn_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim),
        )

        self.velocity_head = nn.Linear(embed_dim, num_signals)

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

    def _validate_flow_inputs(self, u_t, t, batch_size):
        expected_u = (batch_size, self.trajectory_dim)
        if tuple(u_t.shape) != expected_u:
            raise ValueError(
                f"noisy_trajectory must have shape {expected_u} "
                f"(batch_size, num_timesteps * num_signals), got {tuple(u_t.shape)}."
            )
        if tuple(t.shape) != (batch_size,):
            raise ValueError(
                f"flow_timestep must have shape ({batch_size},), got {tuple(t.shape)}."
            )
        if u_t.device != t.device:
            raise ValueError(
                f"noisy_trajectory and flow_timestep must be on the same device, "
                f"got {u_t.device} and {t.device}."
            )
        if u_t.dtype != t.dtype:
            raise ValueError(
                f"noisy_trajectory and flow_timestep must share dtype, "
                f"got {u_t.dtype} and {t.dtype}."
            )

    def _validate_trajectory_target(self, trajectory_target, batch_size, device):
        expected = (batch_size, self.trajectory_dim)
        if tuple(trajectory_target.shape) != expected:
            raise ValueError(
                f"trajectory_target must have shape {expected} "
                f"(batch_size, num_timesteps * num_signals), got "
                f"{tuple(trajectory_target.shape)}."
            )
        if trajectory_target.device != device:
            raise ValueError(
                f"trajectory_target must be on the same device as bev_features, "
                f"got {trajectory_target.device} and {device}."
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

    def _modulation_conditioning(self, visual_history, egomotion_history):
        """Conditioning vector fed into AdaLN — excludes BEV (cross-attn) and
        time (added per-step in the Euler loop)."""
        return (
            self.visual_history_proj(visual_history)
            + self.ego_state_proj(egomotion_history)
        )

    def _ego_hidden(self, bev_features, mod_cond):
        bev_pool = bev_features.mean(dim=(2, 3))
        return self.cond_to_ego_hidden(self.bev_pool_proj(bev_pool) + mod_cond)

    def _sample_timesteps(self, batch_size, device, dtype):
        """Sample flow timesteps t in [0, 1).

        Default follows Alpamayo / pi-0.5: a shifted Beta(beta_alpha, 1.0)
        biased toward the noisy (low-t) end, which improves flow-matching
        policy training. ``timestep_sampler='uniform'`` falls back to U(0, 1).
        """
        if self.timestep_sampler == "uniform":
            return torch.rand(batch_size, device=device, dtype=dtype)
        # beta: t = beta_scale - beta_scale * Beta(alpha, 1).sample()
        b = self._beta_dist.sample((batch_size,)).to(device=device, dtype=dtype)
        return self.beta_scale - self.beta_scale * b

    def construct_training_data(self, trajectory_target):
        """Sample (u_t, t, target_velocity) for one flow-matching training step.

        Used internally by ``compute_planner_loss``. Kept public so that
        advanced callers can share the same (u_t, t) across multiple loss
        terms without re-sampling — the canonical path is
        ``compute_planner_loss``, which never exposes the raw velocity.

        Returns:
            u_t: [B, trajectory_dim] — the noisy interpolated state.
            t: [B] — flow timesteps in [0, 1].
            target_velocity: [B, trajectory_dim] — the true velocity x_1 - x_0
                that v_theta should predict at (u_t, t).
        """
        B = trajectory_target.shape[0]
        x_0 = torch.randn_like(trajectory_target)
        t = self._sample_timesteps(
            B, trajectory_target.device, trajectory_target.dtype,
        )
        u_t = (1.0 - t).unsqueeze(-1) * x_0 + t.unsqueeze(-1) * trajectory_target
        target_velocity = trajectory_target - x_0
        # Cheap defense-in-depth: catch shape regressions even on the
        # internal sampling path.
        self._validate_flow_inputs(u_t, t, B)
        return u_t, t, target_velocity

    def _project_bev(self, bev_features):
        """Flatten BEV to a sequence of projected key/value tokens.

        ``[B, embed_dim, H, W]`` → ``[B, H*W, embed_dim]``. The projection
        is independent of u_t and t, so callers compute it once per
        forward() and reuse it across all Euler steps in inference.
        """
        bev_seq = bev_features.flatten(2).transpose(1, 2)
        return self.bev_kv_proj(bev_seq)

    def _v_theta(self, u_t, t, bev_seq, mod_cond):
        """Conditional velocity network with BEV cross-attention + AdaLN.

        Args:
            u_t: [B, trajectory_dim]
            t: [B]
            bev_seq: [B, H*W, embed_dim] — BEV keys/values already produced
                by ``_project_bev``. Precomputed once per forward() to avoid
                re-flattening and re-projecting on every Euler step.
            mod_cond: [B, embed_dim] — visual_history + egomotion conditioning.

        Returns:
            velocity: [B, trajectory_dim]
        """
        B = u_t.shape[0]

        # Action queries: one token per future timestep.
        u_t_seq = u_t.reshape(B, self.num_timesteps, self.num_signals)
        queries = self.action_proj(u_t_seq)                      # [B, T, C]

        attended, _ = self.cross_attn(queries, bev_seq, bev_seq) # [B, T, C]

        # AdaLN: time + (visual_history + egomotion) → (gamma, beta).
        t_emb = self.time_mlp(self._sinusoidal_time_embedding(t))
        gamma, beta = self.adaln_modulation(mod_cond + t_emb).chunk(2, dim=-1)
        normed = self.attn_norm(attended)
        modulated = normed * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        velocity_seq = self.velocity_head(modulated)             # [B, T, S]
        return velocity_seq.reshape(B, self.trajectory_dim)

    def forward(self, bev_features, visual_history, egomotion_history,
                generator=None, **kwargs):
        """Inference: Euler-integrate ``dx/dt = v_theta(x, t, ...)`` over [0, 1].

        Args:
            bev_features: [B, embed_dim, H, W].
            visual_history: [B, visual_history_dim].
            egomotion_history: [B, egomotion_dim].
            generator: optional ``torch.Generator`` used to seed the noise
                prior so evaluation runs are reproducible.
            **kwargs: ignored. Accepts extra inputs other planners or
                callers might pass so call sites can stay planner-agnostic.

        Returns:
            trajectory: [B, trajectory_dim] — integrated from a noise sample.
            ego_hidden: [B, embed_dim] — context vector consumed downstream
                by FutureState.
        """
        self._validate_inputs(visual_history, egomotion_history)
        mod_cond = self._modulation_conditioning(visual_history, egomotion_history)
        ego_hidden = self._ego_hidden(bev_features, mod_cond)
        # bev_seq is computed once and reused across every Euler step.
        bev_seq = self._project_bev(bev_features)

        B = bev_features.shape[0]
        x = torch.randn(B, self.trajectory_dim,
                        device=bev_features.device, dtype=bev_features.dtype,
                        generator=generator)
        dt = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            t_val = step * dt
            t = torch.full((B,), t_val,
                           device=bev_features.device, dtype=bev_features.dtype)
            v = self._v_theta(x, t, bev_seq, mod_cond)
            x = x + dt * v
        return x, ego_hidden

    def compute_planner_loss(self, bev_features, visual_history,
                             egomotion_history, trajectory_target):
        """Flow-matching MSE between predicted and target conditional velocity.

        Samples (u_t, t, target_velocity) from ``construct_training_data``
        and computes ``F.mse_loss(v_theta(u_t, t, c), target_velocity)``.
        The raw predicted velocity never leaves this method, so the caller
        cannot accidentally MSE it against a trajectory target.

        Returns ``(loss, ego_hidden)`` as required by ``BasePlanner``.
        """
        self._validate_inputs(visual_history, egomotion_history)
        B = bev_features.shape[0]
        self._validate_trajectory_target(
            trajectory_target, B, bev_features.device,
        )

        u_t, t, target_velocity = self.construct_training_data(trajectory_target)

        mod_cond = self._modulation_conditioning(visual_history, egomotion_history)
        ego_hidden = self._ego_hidden(bev_features, mod_cond)
        bev_seq = self._project_bev(bev_features)

        velocity_pred = self._v_theta(u_t, t, bev_seq, mod_cond)
        loss = F.mse_loss(velocity_pred, target_velocity)
        return loss, ego_hidden
