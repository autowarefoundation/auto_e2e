"""Lightweight, training-free multi-sample trajectory scorer.

Implements the "Phase 1" BEV-only re-ranking proposed in the diffusion /
flow-matching driving-policy discussion (see PROPOSAL_diffusion_driving_policy.md).
Given K trajectory samples drawn from a stochastic planner (e.g.
FlowMatchingPlanner sampled with different noise seeds), this module scores
each sample by:

  1. Drivable-area compliance, read directly off the *rasterized* BEV map
     image that RasterizedMapEncoder consumes as input (no new training
     required — this is a deterministic geometric + colour lookup, not a
     learned classifier).
  2. Kinematic comfort, i.e. how much each sample's (acceleration,
     curvature) sequence violates configurable comfort bounds.

...and returns either the single best-scoring sample per batch element, or
a softmax-weighted blend across samples (mirroring GoalFlow's "nearest" vs
"mean" trajectory-selection modes, arXiv:2503.05689).

Deliberately excluded (tracked as Phase 2 in the proposal):
  - Learned BEV semantic segmentation head (GoalFlow's `_bev_semantic_head`)
  - Goal-point vocabulary + image/DAC scorer trained offline
  - Classifier-free-guidance-style goal-conditioned/unconditioned fusion

Phase 1 intentionally has zero new trainable parameters and zero new loss
terms, so it can be merged and evaluated without retraining the perception
or planner stack — it only changes *how many* samples are drawn from an
already-trained stochastic planner and *how* the best one is picked.

Calibration note: `pixels_per_meter`, `ego_row`, and `ego_col` below MUST be
set to match whatever convention the KITScenes / L2D map renderer actually
uses to produce `map_input`. The defaults here follow the BEV geometry
discussed for this fork (120 m front / 60 m rear / 60 m each side at 0.4 m
resolution -> 450 x 300 px, issue #35) but have NOT been verified against
the renderer itself — confirm with whoever owns that code before relying
on the compliance score in any reported metric.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ScorerConfig:
    # --- BEV pixel-space calibration (see calibration note above) ---
    pixels_per_meter: float = 2.5          # 1 / 0.4 m
    bev_h: int = 450                       # rows: forward axis
    bev_w: int = 300                       # cols: lateral axis
    ego_row: int = 300                     # row index where ego (x=0) sits
                                            # (150m*2.5=375 if symmetric; the
                                            # 120m-front/60m-rear split used
                                            # in issue #35 gives 60*2.5=150 from
                                            # the *bottom*, i.e. row 450-150=300
                                            # from the top if rendered front-up).
    ego_col: int = 150                     # col index where ego (y=0) sits
                                            # (lateral center: 300 / 2)
    forward_is_negative_row: bool = True   # True if increasing x (forward)
                                            # moves to smaller row indices
                                            # (image rendered nose-up).

    # --- drivable-area colour lookup ---
    # RGB tuple(s) considered "drivable" in the rendered map_input image.
    # Confirm against the actual renderer palette before use.
    drivable_rgb: tuple = (255, 255, 255)
    drivable_rgb_tolerance: int = 10       # per-channel L1 tolerance

    # --- kinematic comfort bounds ---
    max_comfortable_accel: float = 3.0     # m/s^2
    max_comfortable_lateral_accel: float = 2.0  # m/s^2, = curvature * speed^2
    dt: float = 0.1                        # seconds between model timesteps
    initial_speed: float = 5.0             # m/s, matches existing decode convention

    # --- scoring weights ---
    dac_weight: float = 1.0
    comfort_weight: float = 0.5

    # --- selection mode ---
    selection: str = "nearest"             # "nearest" (argmax) or "mean" (softmax blend)
    softmax_temperature: float = 1.0


def decode_trajectory_to_xy(trajectory: torch.Tensor, num_timesteps: int,
                             dt: float = 0.1,
                             initial_speed: float = 5.0) -> torch.Tensor:
    """Decode (acceleration, curvature) pairs into (x, y) waypoints.

    Mirrors the bicycle-model integration already used for offline
    evaluation against Waymo ground truth. If/when a canonical decode
    utility lands (tracked under the open-loop eval pipeline, issue #66),
    this function should be replaced by an import from that module instead
    of duplicating the integration here.

    Args:
        trajectory: [..., num_timesteps * 2] — flat (accel, curvature) pairs.
        num_timesteps: number of (accel, curvature) pairs encoded.
        dt: seconds between timesteps.
        initial_speed: assumed starting speed in m/s.

    Returns:
        xy: [..., num_timesteps, 2] waypoints in ego-relative meters.
    """
    *batch_shape, _ = trajectory.shape
    pairs = trajectory.reshape(*batch_shape, num_timesteps, 2)
    accels, curvatures = pairs[..., 0], pairs[..., 1]

    device, dtype = trajectory.device, trajectory.dtype
    flat = accels.reshape(-1, num_timesteps)
    flat_curv = curvatures.reshape(-1, num_timesteps)
    n = flat.shape[0]

    x = torch.zeros(n, device=device, dtype=dtype)
    y = torch.zeros(n, device=device, dtype=dtype)
    heading = torch.zeros(n, device=device, dtype=dtype)
    speed = torch.full((n,), initial_speed, device=device, dtype=dtype)

    xs, ys = [], []
    for t in range(num_timesteps):
        speed = torch.clamp(speed + flat[:, t] * dt, min=0.0)
        heading = heading + flat_curv[:, t] * speed * dt
        x = x + torch.cos(heading) * speed * dt
        y = y + torch.sin(heading) * speed * dt
        xs.append(x.clone())
        ys.append(y.clone())

    xy = torch.stack([torch.stack(xs, dim=-1), torch.stack(ys, dim=-1)], dim=-1)
    return xy.reshape(*batch_shape, num_timesteps, 2)


def project_xy_to_bev_pixel(xy: torch.Tensor, config: ScorerConfig) -> torch.Tensor:
    """Project ego-relative (x, y) meters into BEV pixel (row, col) indices.

    Args:
        xy: [..., 2] ego-relative coordinates in meters (x=forward, y=left).
        config: calibration parameters — see module docstring.

    Returns:
        pixel: [..., 2] integer (row, col) indices, NOT clamped to
            [0, bev_h) / [0, bev_w) — caller must mask out-of-bounds points
            (see `drivable_area_compliance`).
    """
    x, y = xy[..., 0], xy[..., 1]
    row_offset = -x if config.forward_is_negative_row else x
    row = config.ego_row + row_offset * config.pixels_per_meter
    col = config.ego_col - y * config.pixels_per_meter  # +y = left = smaller col
    return torch.stack([row, col], dim=-1).round().long()


def drivable_area_compliance(xy: torch.Tensor, map_input: torch.Tensor,
                             config: ScorerConfig) -> torch.Tensor:
    """Fraction of trajectory waypoints landing on a "drivable" map pixel.

    Args:
        xy: [B, num_timesteps, 2] ego-relative waypoints in meters.
        map_input: [B, 3, bev_h, bev_w] rasterized BEV map image, the same
            tensor fed to RasterizedMapEncoder (channel order assumed RGB,
            values in [0, 255] or normalized — see note below).
        config: calibration parameters.

    Returns:
        compliance: [B] fraction in [0, 1] of waypoints inside the
            drivable-area colour band and within image bounds.

    Note: if `map_input` has already been ImageNet-normalized upstream of
    this call, the colour lookup must run on a separate un-normalized copy
    of the map image — wire this scorer to whichever stage in the data
    pipeline still has raw pixel values.
    """
    B, _, H, W = map_input.shape
    T = xy.shape[1]
    pixels = project_xy_to_bev_pixel(xy, config)  # [B, T, 2]

    rows, cols = pixels[..., 0], pixels[..., 1]
    in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)

    rows_c = rows.clamp(0, H - 1)
    cols_c = cols.clamp(0, W - 1)

    target = torch.tensor(config.drivable_rgb, device=map_input.device,
                          dtype=map_input.dtype).view(1, 1, 3)

    compliant = torch.zeros(B, T, dtype=torch.bool, device=map_input.device)
    for b in range(B):
        sampled = map_input[b, :, rows_c[b], cols_c[b]].transpose(0, 1)  # [T, 3]
        diff = (sampled - target[0]).abs().sum(dim=-1)
        compliant[b] = diff <= (3 * config.drivable_rgb_tolerance)

    compliant = compliant & in_bounds
    return compliant.float().mean(dim=1)  # [B]


def kinematic_comfort_score(trajectory: torch.Tensor, num_timesteps: int,
                            config: ScorerConfig) -> torch.Tensor:
    """Penalize (acceleration, curvature) samples that exceed comfort bounds.

    Returns a score in [0, 1] where 1.0 means no bound violations at any
    timestep and 0.0 means every timestep violates at least one bound.

    Args:
        trajectory: [B, num_timesteps * 2] flat (accel, curvature) pairs.
        num_timesteps: number of pairs encoded.
        config: comfort bound parameters.

    Returns:
        score: [B].
    """
    pairs = trajectory.reshape(trajectory.shape[0], num_timesteps, 2)
    accels, curvatures = pairs[..., 0], pairs[..., 1]

    # Approximate speed via cumulative integration of accel (matches decode).
    speed = torch.clamp(
        config.initial_speed + torch.cumsum(accels * config.dt, dim=1),
        min=0.0,
    )
    lateral_accel = curvatures * speed.pow(2)

    accel_violation = (accels.abs() > config.max_comfortable_accel).float()
    lateral_violation = (
        lateral_accel.abs() > config.max_comfortable_lateral_accel
    ).float()

    violation_rate = torch.maximum(accel_violation, lateral_violation).mean(dim=1)
    return 1.0 - violation_rate


class TrajectoryComplianceScorer(nn.Module):
    """Wraps any `BasePlanner` to draw K samples and re-rank them.

    Has zero trainable parameters by design (Phase 1 — see module
    docstring). Works with any planner whose `forward()` accepts a batch
    dimension that can be safely repeated (true for FlowMatchingPlanner via
    its `generator` kwarg for reproducible re-sampling; a deterministic
    planner such as GRUPlanner will simply produce K identical samples and
    this module degenerates to a no-op pass-through with K=1 behaviour).
    """

    def __init__(self, planner: nn.Module, num_timesteps: int,
                config: Optional[ScorerConfig] = None):
        super().__init__()
        self.planner = planner
        self.num_timesteps = num_timesteps
        self.config = config or ScorerConfig()

    @torch.no_grad()
    def sample_and_score(self, bev_features: torch.Tensor,
                         visual_history: torch.Tensor,
                         egomotion_history: torch.Tensor,
                         map_input: torch.Tensor,
                         num_samples: int = 8,
                         seed: Optional[int] = None):
        """Draw `num_samples` trajectories per batch element and re-rank.

        Args:
            bev_features: [B, embed_dim, H, W] — fused image+map BEV, as
                already produced by AutoE2E before the planner call.
            visual_history: [B, visual_history_dim].
            egomotion_history: [B, egomotion_dim].
            map_input: [B, 3, bev_h, bev_w] raw (un-normalized) rasterized
                map image — see `drivable_area_compliance` note on
                normalization.
            num_samples: K, number of stochastic samples per batch element.
            seed: optional base seed for reproducible re-sampling.

        Returns:
            trajectory: [B, num_timesteps * num_signals] — best (or
                softmax-blended) trajectory per batch element.
            ego_hidden: [B, embed_dim] — from the FIRST sample only,
                consistent with how FutureState is meant to receive a
                single scene-gist vector, not a per-candidate one.
            scores: [B, num_samples] — combined score per candidate, for
                logging / debugging.
        """
        B = bev_features.shape[0]
        device = bev_features.device

        bev_rep = bev_features.repeat_interleave(num_samples, dim=0)
        vh_rep = visual_history.repeat_interleave(num_samples, dim=0)
        eh_rep = egomotion_history.repeat_interleave(num_samples, dim=0)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        trajectories, ego_hidden_all = self.planner(
            bev_rep, vh_rep, eh_rep, generator=generator,
        )
        # trajectories: [B*K, trajectory_dim]
        trajectory_dim = trajectories.shape[-1]
        trajectories = trajectories.view(B, num_samples, trajectory_dim)
        ego_hidden_all = ego_hidden_all.view(B, num_samples, -1)

        xy = decode_trajectory_to_xy(
            trajectories, self.num_timesteps,
            dt=self.config.dt, initial_speed=self.config.initial_speed,
        )  # [B, K, T, 2]

        dac_scores = torch.stack([
            drivable_area_compliance(xy[:, k], map_input, self.config)
            for k in range(num_samples)
        ], dim=1)  # [B, K]

        comfort_scores = torch.stack([
            kinematic_comfort_score(
                trajectories[:, k], self.num_timesteps, self.config,
            )
            for k in range(num_samples)
        ], dim=1)  # [B, K]

        combined = (
            self.config.dac_weight * dac_scores
            + self.config.comfort_weight * comfort_scores
        )

        if self.config.selection == "nearest":
            best_idx = combined.argmax(dim=1)
            trajectory = trajectories[torch.arange(B, device=device), best_idx]
        elif self.config.selection == "mean":
            weights = torch.softmax(
                combined / self.config.softmax_temperature, dim=1,
            )
            trajectory = (trajectories * weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(
                f"config.selection must be 'nearest' or 'mean', "
                f"got {self.config.selection!r}."
            )

        ego_hidden = ego_hidden_all[:, 0]
        return trajectory, ego_hidden, combined
