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

DO NOT calibrate against Model/data_parsing/kit_scenes/map.py as it stands
today without checking #148 and #149 first: #149 proposes replacing the
current 640x360 non-square render with a 256x256 square tile at 120 m in
*all four* directions (symmetric), and #148 is reworking what the map
actually encodes (route direction from GPS traces, not just a static
drivable-area raster). Both are open and assigned to riita10069 as of
2026-07-21 — the geometry below may need to change again once those land,
not just be verified against today's renderer.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from Model.evaluation.metrics import integrate_trajectory


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
            drivable_area_compliance(xy[:, k], map_input, self.config)
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
