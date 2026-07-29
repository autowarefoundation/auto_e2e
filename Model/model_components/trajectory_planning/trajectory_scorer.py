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

Calibration: `meters_per_pixel`, `ego_row`, `ego_col` below come directly
from `Model.navigation.geometry.DEFAULT_NAVIGATION_GEOMETRY` — the actual
geometry `map_context` is rasterized with (#161, merged, resolved #148/#149).
This is no longer a guess to verify; it's imported from the same source of
truth the renderer itself uses. Drivable-area compliance reads
`map_context`'s `MapChannel.DRIVABLE_AREA` channel directly — a real
binary semantic mask, not a pixel-colour heuristic on a human-viewable
image (map_context is 14 semantic channels now, not a rendered RGB image).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from Model.evaluation.metrics import integrate_trajectory
from Model.navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY, MapChannel


@dataclass
class ScorerConfig:
    # --- BEV pixel-space calibration ---
    # Sourced directly from Model.navigation.geometry.DEFAULT_NAVIGATION_GEOMETRY
    # (geometry_id="kitscenes-v3-bev-1m-v1") — the same geometry map_context
    # is actually rasterized with, not a guessed/reverse-engineered value.
    # This resolves the #148/#149 calibration risk this module used to warn
    # about: those issues are merged (see #161), and this is their real
    # output geometry, not a placeholder.
    meters_per_pixel: float = DEFAULT_NAVIGATION_GEOMETRY.meters_per_pixel
    bev_h: int = DEFAULT_NAVIGATION_GEOMETRY.height_px
    bev_w: int = DEFAULT_NAVIGATION_GEOMETRY.width_px
    x_max_m: float = DEFAULT_NAVIGATION_GEOMETRY.x_max_m
    y_max_m: float = DEFAULT_NAVIGATION_GEOMETRY.y_max_m
    ego_row: float = DEFAULT_NAVIGATION_GEOMETRY.ego_anchor_row
    ego_col: float = DEFAULT_NAVIGATION_GEOMETRY.ego_anchor_col

    # --- drivable-area lookup ---
    # map_context's DRIVABLE_AREA channel (Model.navigation.geometry.MapChannel)
    # is one of BINARY_MAP_CHANNELS — a real semantic 0/1 mask, not a pixel
    # colour to fuzzy-match. Replaces the old RGB-tolerance heuristic
    # entirely: map_context is 14 semantic channels now (#161), not a
    # human-viewable rasterized image, so there is no colour to match
    # against any more.
    drivable_area_channel: int = int(MapChannel.DRIVABLE_AREA)
    drivable_threshold: float = 0.5

    # --- kinematic comfort bounds ---
    max_comfortable_accel: float = 3.0     # m/s^2
    max_comfortable_lateral_accel: float = 2.0  # m/s^2, = curvature * speed^2
    dt: float = 0.1                        # seconds between model timesteps
    # NOTE: no fixed initial_speed field — real per-scene speed is read
    # out of egomotion_history via extract_initial_speed(), not guessed.

    # --- scoring weights ---
    dac_weight: float = 1.0
    comfort_weight: float = 0.5

    # --- selection mode ---
    selection: str = "nearest"             # "nearest" (argmax) or "mean" (softmax blend)
    softmax_temperature: float = 1.0


def extract_initial_speed(egomotion_history: torch.Tensor) -> torch.Tensor:
    """Read the real per-scene starting speed out of egomotion_history.

    egomotion_history is (256,) = 64 history timesteps x 4 signals
    [speed, acceleration, yaw_rate, curvature] (see
    Model/data_parsing/kit_scenes/egomotion.py). The most recent history
    row (index -1) is "now" — its speed channel (index 0) is exactly the
    v0 that the prediction horizon starts from.

    Args:
        egomotion_history: [..., 256].

    Returns:
        initial_speed: [...] real starting speed in m/s, one per row.
    """
    *batch_shape, dim = egomotion_history.shape
    if dim != 256:
        raise ValueError(
            f"egomotion_history last dim must be 256 (64 timesteps x 4 "
            f"signals), got {dim}."
        )
    history = egomotion_history.reshape(*batch_shape, 64, 4)
    return history[..., -1, 0]


def decode_trajectory_to_xy(trajectory: torch.Tensor, num_timesteps: int,
                             initial_speed: torch.Tensor,
                             dt: float = 0.1) -> torch.Tensor:
    """Decode (acceleration, curvature) pairs into (x, y) waypoints.

    Thin torch<->numpy wrapper around the canonical
    ``Model.evaluation.metrics.integrate_trajectory`` bicycle-model
    integrator, so this scorer and offline open-loop eval (ADE/FDE against
    Waymo/KITScenes ground truth) share one integration implementation
    instead of two that can silently drift apart.

    Runs at @torch.no_grad() call sites only (TrajectoryComplianceScorer
    has zero trainable parameters by design — see module docstring), so
    the per-row Python loop and numpy round-trip cost nothing that
    matters: K is small (default 8) and this never sits in a training step.

    Args:
        trajectory: [..., num_timesteps * 2] — flat (accel, curvature) pairs.
        num_timesteps: number of (accel, curvature) pairs encoded.
        initial_speed: [...] real starting speed in m/s, one per row —
            see ``extract_initial_speed``. Broadcasts against
            ``trajectory``'s leading dims; every row MUST carry its own
            real value, not a fixed placeholder — a fixed default here
            silently overrides every sample's actual starting speed with
            the same guess, regardless of how fast the ego really was
            moving.
        dt: seconds between timesteps.

    Returns:
        xy: [..., num_timesteps, 2] waypoints in ego-relative meters.
    """
    *batch_shape, _ = trajectory.shape
    pairs = trajectory.reshape(*batch_shape, num_timesteps, 2)
    accels = pairs[..., 0].reshape(-1, num_timesteps)
    curvatures = pairs[..., 1].reshape(-1, num_timesteps)
    speeds = initial_speed.reshape(-1)

    if speeds.shape[0] != accels.shape[0]:
        raise ValueError(
            f"initial_speed must broadcast to trajectory's leading dims: "
            f"got {speeds.shape[0]} speed rows for {accels.shape[0]} "
            f"trajectory rows."
        )

    device, dtype = trajectory.device, trajectory.dtype
    accels_np = accels.detach().cpu().numpy()
    curv_np = curvatures.detach().cpu().numpy()
    speeds_np = speeds.detach().cpu().numpy()

    n = accels_np.shape[0]
    xy_np = np.empty((n, num_timesteps, 2), dtype=np.float64)
    for i in range(n):
        xy_np[i] = integrate_trajectory(
            accels_np[i], curv_np[i], float(speeds_np[i]), dt=dt,
        )

    xy = torch.from_numpy(xy_np).to(device=device, dtype=dtype)
    return xy.reshape(*batch_shape, num_timesteps, 2)


def project_xy_to_bev_pixel(xy: torch.Tensor, config: ScorerConfig) -> torch.Tensor:
    """Project ego-relative (x, y) meters into BEV pixel (row, col) indices.

    Mirrors NavigationRasterGeometry.ego_to_pixel's formula exactly
    (Model/navigation/geometry.py) — same sign conventions, same -0.5
    pixel-center offset — so a trajectory scored here and the same
    trajectory rendered through the real geometry object land on the same
    pixel. Reimplemented in torch (not called directly) only to stay
    differentiable/GPU-resident for the batched xy tensor this receives;
    the arithmetic is identical, not independently derived.

    Args:
        xy: [..., 2] ego-relative coordinates in meters (x=forward, y=left).
        config: calibration parameters — see module docstring.

    Returns:
        pixel: [..., 2] integer (row, col) indices, NOT clamped to
            [0, bev_h) / [0, bev_w) — caller must mask out-of-bounds points
            (see `drivable_area_compliance`).
    """
    x, y = xy[..., 0], xy[..., 1]
    row = (config.x_max_m - x) / config.meters_per_pixel - 0.5
    col = (config.y_max_m - y) / config.meters_per_pixel - 0.5
    return torch.stack([row, col], dim=-1).round().long()


def drivable_area_compliance(xy: torch.Tensor, map_context: torch.Tensor,
                             config: ScorerConfig) -> torch.Tensor:
    """Fraction of trajectory waypoints landing on the drivable-area mask.

    Args:
        xy: [B, num_timesteps, 2] ego-relative waypoints in meters.
        map_context: [B, 14, bev_h, bev_w] semantic navigation raster (#161)
            — the same tensor ReactiveE2E.forward() takes as map_context.
            Channel config.drivable_area_channel (MapChannel.DRIVABLE_AREA)
            is a real binary 0/1 mask (Model.navigation.geometry.
            BINARY_MAP_CHANNELS), not a pixel colour — no ImageNet
            normalization or colour-space concern applies here the way it
            did for the old rendered-RGB map_input.
        config: calibration parameters.

    Returns:
        compliance: [B] fraction in [0, 1] of waypoints on a drivable
            pixel and within the raster's bounds.
    """
    B, _, H, W = map_context.shape
    T = xy.shape[1]
    pixels = project_xy_to_bev_pixel(xy, config)  # [B, T, 2]

    rows, cols = pixels[..., 0], pixels[..., 1]
    in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)

    rows_c = rows.clamp(0, H - 1)
    cols_c = cols.clamp(0, W - 1)

    drivable = map_context[:, config.drivable_area_channel]  # [B, H, W]

    compliant = torch.zeros(B, T, dtype=torch.bool, device=map_context.device)
    for b in range(B):
        compliant[b] = drivable[b, rows_c[b], cols_c[b]] > config.drivable_threshold

    compliant = compliant & in_bounds
    return compliant.float().mean(dim=1)  # [B]


def kinematic_comfort_score(trajectory: torch.Tensor, num_timesteps: int,
                            initial_speed: torch.Tensor,
                            config: ScorerConfig) -> torch.Tensor:
    """Penalize (acceleration, curvature) samples that exceed comfort bounds.

    Returns a score in [0, 1] where 1.0 means no bound violations at any
    timestep and 0.0 means every timestep violates at least one bound.

    Args:
        trajectory: [B, num_timesteps * 2] flat (accel, curvature) pairs.
        num_timesteps: number of pairs encoded.
        initial_speed: [B] real starting speed in m/s — see
            ``extract_initial_speed``. Same reasoning as
            ``decode_trajectory_to_xy``: a fixed guess here would silently
            score every sample's lateral-accel comfort against the wrong
            speed profile.
        config: comfort bound parameters.

    Returns:
        score: [B].
    """
    pairs = trajectory.reshape(trajectory.shape[0], num_timesteps, 2)
    accels, curvatures = pairs[..., 0], pairs[..., 1]

    # Approximate speed via cumulative integration of accel (matches decode).
    speed = torch.clamp(
        initial_speed.reshape(-1, 1) + torch.cumsum(accels * config.dt, dim=1),
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
                         map_context: torch.Tensor,
                         num_samples: int = 8,
                         seed: Optional[int] = None):
        """Draw `num_samples` trajectories per batch element and re-rank.

        Args:
            bev_features: [B, embed_dim, H, W] — fused image+map BEV, as
                already produced by AutoE2E before the planner call.
            visual_history: [B, visual_history_dim].
            egomotion_history: [B, egomotion_dim].
            map_context: [B, 14, bev_h, bev_w] semantic navigation raster
                (#161) — see `drivable_area_compliance`. Renamed from
                map_input to match ReactiveE2E.forward()'s own naming
                after #161; this is no longer a human-viewable rendered
                image, so the old name (implying a raw image to feed
                a CNN) was misleading.
            num_samples: K, number of stochastic samples per batch element.
            seed: optional base seed for reproducible re-sampling.

        Returns:
            trajectory: [B, num_timesteps * num_signals] — best (or
                softmax-blended) trajectory per batch element.
            scores: [B, num_samples] — combined score per candidate, for
                logging / debugging.

        Note: this used to also return an `ego_hidden` second element,
        unpacked from `self.planner(...)` as if forward() returned a
        2-tuple. It never did — BasePlanner.forward() has always returned
        a single trajectory tensor (see base.py's own docstring), so that
        unpack would raise the moment this ran against a real planner
        instead of a test double shaped to match the wrong contract.
        FutureState (the only place ego_hidden was ever consumed) isn't
        wired into AutoE2E.forward() any more — the World Model path
        (WorldActionModel.predict_future) superseded it — so there's
        nothing left downstream expecting a second return value.
        """
        B = bev_features.shape[0]
        device = bev_features.device

        bev_rep = bev_features.repeat_interleave(num_samples, dim=0)
        vh_rep = visual_history.repeat_interleave(num_samples, dim=0)
        eh_rep = egomotion_history.repeat_interleave(num_samples, dim=0)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        trajectories = self.planner(
            bev_rep, vh_rep, eh_rep, generator=generator,
        )
        # trajectories: [B*K, trajectory_dim]
        trajectory_dim = trajectories.shape[-1]
        trajectories = trajectories.view(B, num_samples, trajectory_dim)

        initial_speed = extract_initial_speed(eh_rep).view(B, num_samples)

        xy = decode_trajectory_to_xy(
            trajectories, self.num_timesteps,
            initial_speed=initial_speed, dt=self.config.dt,
        )  # [B, K, T, 2]

        dac_scores = torch.stack([
            drivable_area_compliance(xy[:, k], map_context, self.config)
            for k in range(num_samples)
        ], dim=1)  # [B, K]

        comfort_scores = torch.stack([
            kinematic_comfort_score(
                trajectories[:, k], self.num_timesteps,
                initial_speed[:, k], self.config,
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

        return trajectory, combined
