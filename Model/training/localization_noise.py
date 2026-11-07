"""Simulate the onboard localization error the map raster is cropped with.

Dataset poses are post-processed and too accurate: a map cropped at them leaves
the map fusion and the dead reckoning nothing to do. During training the true
pose is hidden. Over the 6.4 s egomotion history each sample synthesizes dead
reckoning (DR) and GNSS observations from the logged motion, runs a 3-state EKF
(x, y, yaw) on them, and returns the pose the EKF reports at the current frame,
expressed in the true ego frame. nuPlan has no raw GNSS (its database holds
only post-processed ``ego_pose``), so GNSS is synthesized too.

GNSS regimes, one per sample:

- nominal (``NOMINAL_SHARE``): white noise of ``NOMINAL_SIGMA_M`` plus a slowly
  varying bias (first-order Gauss-Markov, ``BIAS_TAU_S``);
- degraded (``DEGRADED_SHARE``): nominal plus a ``JUMP_M`` jump lasting
  ``JUMP_DURATION_S``, as from multipath under an overpass;
- outage (the rest): no fixes in the window; the EKF starts from the drift DR
  accumulated since the last good fix ``OUTAGE_S`` ago.

The EKF state is (x, y, yaw, yaw-rate bias), so a constant gyro bias is
estimated from the GNSS track instead of being treated as white noise. The
final crop error is kept within the raster margin (``MAX_CROP_OFFSET_M``).

DR uses the logged speed with a ``SPEED_SCALE_ERROR`` scale error and the
logged yaw rate with a ``YAW_RATE_BIAS_DEG_S`` bias.

The trajectory targets and the camera geometry stay in the true ego frame; only
the map crop moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch

DT_S: Final = 0.1
HISTORY_STEPS: Final = 64
SIGNALS: Final = 4
NOMINAL_SHARE: Final = 0.60
DEGRADED_SHARE: Final = 0.25
NOMINAL_SIGMA_M: Final = (0.3, 1.0)
BIAS_SIGMA_M: Final = 0.5
BIAS_TAU_S: Final = 30.0
JUMP_M: Final = (3.0, 10.0)
JUMP_DURATION_S: Final = (2.0, 30.0)
OUTAGE_S: Final = (5.0, 30.0)
SPEED_SCALE_ERROR: Final = (0.01, 0.02)
YAW_RATE_BIAS_DEG_S: Final = (0.1, 0.5)
# What the EKF assumes: the receiver's nominal accuracy and DR noise bounds.
EKF_GNSS_SIGMA_M: Final = 1.0
EKF_SPEED_SIGMA: Final = 0.02
# White gyro noise; the yaw-rate bias is a state of the EKF, starting at zero
# with the largest simulated bias as its standard deviation.
EKF_GYRO_NOISE_DEG_S: Final = 0.05
EKF_BIAS_WALK_DEG_S2: Final = 0.002
EKF_INITIAL_SIGMA_M: Final = 0.5
EKF_INITIAL_YAW_SIGMA_DEG: Final = 1.0
# Innovations beyond the 99 % chi-square bound with two degrees of freedom
# are rejected, so a jump is ignored until the covariance has grown to it.
GATE_CHI2: Final = 9.21
# The 0.4 m navigation raster extends 30 to 40 m beyond the AutoBEV grid, so
# crops are kept within 10 m and 10 degrees of the truth; a larger drift would
# leave the grid without map, which no onboard system would hand on as a map.
MAX_CROP_OFFSET_M: Final = 10.0
MAX_CROP_YAW_DEG: Final = 10.0
# Share of samples whose crop is then refined by map matching.
MAP_MATCHING_SHARE: Final = 0.5


@dataclass(frozen=True)
class LocalizationEstimate:
    """Crop pose ``T_E<-Ê`` [B, 3], its covariance [B, 3, 3] and which samples use map matching [B]."""

    pose: torch.Tensor
    covariance: torch.Tensor
    map_matching: torch.Tensor


def _uniform(generator: torch.Generator, bounds: tuple[float, float], size: int) -> torch.Tensor:
    return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(size, generator=generator)


def _signed(generator: torch.Generator, size: int) -> torch.Tensor:
    return torch.where(torch.rand(size, generator=generator) < 0.5, -1.0, 1.0)


def _integrate(speed: torch.Tensor, yaw_rate: torch.Tensor) -> torch.Tensor:
    """World poses [B, T+1, 3] from the window start (origin) by unicycle integration."""
    heading = torch.cat([torch.zeros_like(yaw_rate[:, :1]), torch.cumsum(yaw_rate * DT_S, dim=1)], dim=1)
    step_x = speed * torch.cos(heading[:, :-1]) * DT_S
    step_y = speed * torch.sin(heading[:, :-1]) * DT_S
    zero = torch.zeros_like(speed[:, :1])
    x = torch.cat([zero, torch.cumsum(step_x, dim=1)], dim=1)
    y = torch.cat([zero, torch.cumsum(step_y, dim=1)], dim=1)
    return torch.stack([x, y, heading], dim=-1)


def simulate_localization(egomotion_history: torch.Tensor, generator: torch.Generator) -> LocalizationEstimate:
    """EKF crop pose for every sample of a [B, 256] egomotion history (see module docstring)."""
    device = egomotion_history.device
    signals = egomotion_history.reshape(-1, HISTORY_STEPS, SIGNALS).float().cpu()
    batch = signals.shape[0]
    speed, yaw_rate = signals[..., 0].clamp_min(0.0), signals[..., 2]
    truth = _integrate(speed, yaw_rate)  # [B, 65, 3]; row 64 is the current frame

    regime = torch.rand(batch, generator=generator)
    nominal = regime < NOMINAL_SHARE
    degraded = (regime >= NOMINAL_SHARE) & (regime < NOMINAL_SHARE + DEGRADED_SHARE)
    outage = ~(nominal | degraded)

    # Dead reckoning with a scale error and a yaw-rate bias.
    scale = 1.0 + _signed(generator, batch) * _uniform(generator, SPEED_SCALE_ERROR, batch)
    bias = _signed(generator, batch) * torch.deg2rad(_uniform(generator, YAW_RATE_BIAS_DEG_S, batch))
    dr_speed = speed * scale[:, None]
    dr_yaw_rate = yaw_rate + bias[:, None]

    # GNSS fixes at every row, including the current frame.
    times = torch.arange(HISTORY_STEPS + 1, dtype=torch.float32) * DT_S - HISTORY_STEPS * DT_S
    white = torch.randn(batch, HISTORY_STEPS + 1, 2, generator=generator) * _uniform(
        generator, NOMINAL_SIGMA_M, batch)[:, None, None]
    decay = math.exp(-DT_S / BIAS_TAU_S)
    drift = [torch.randn(batch, 2, generator=generator) * BIAS_SIGMA_M]
    for _ in range(HISTORY_STEPS):
        innovation = torch.randn(batch, 2, generator=generator) * BIAS_SIGMA_M * math.sqrt(1.0 - decay * decay)
        drift.append(decay * drift[-1] + innovation)
    jump_size = _uniform(generator, JUMP_M, batch)
    jump_angle = torch.rand(batch, generator=generator) * 2.0 * math.pi
    jump = torch.stack([jump_size * torch.cos(jump_angle), jump_size * torch.sin(jump_angle)], dim=-1)
    duration = _uniform(generator, JUMP_DURATION_S, batch)
    # The jump overlaps the window: it starts up to `duration` before it.
    start = times[0] - duration + torch.rand(batch, generator=generator) * (duration - times[0])
    jumping = degraded[:, None] & (times[None] >= start[:, None]) & (times[None] < (start + duration)[:, None])
    fixes = truth[..., :2] + white + torch.stack(drift, dim=1) + jumping[..., None] * jump[:, None]
    available = ~outage[:, None].expand(-1, HISTORY_STEPS + 1)

    # EKF start: a nominal fix error, or the DR drift since the last good fix.
    since_fix = (_uniform(generator, OUTAGE_S, batch) + times[0]).clamp_min(0.0) * outage
    start_speed = speed[:, 0]
    along = (scale - 1.0) * start_speed * since_fix
    across = 0.5 * bias * start_speed * since_fix.square()
    heading_error = bias * since_fix
    initial_sigma = torch.full((batch,), EKF_INITIAL_SIGMA_M)
    initial = torch.stack([
        truth[:, 0, 0] + along + torch.randn(batch, generator=generator) * initial_sigma,
        truth[:, 0, 1] + across + torch.randn(batch, generator=generator) * initial_sigma,
        truth[:, 0, 2] + heading_error
        + torch.randn(batch, generator=generator) * math.radians(EKF_INITIAL_YAW_SIGMA_DEG),
    ], dim=-1)
    bias_sigma = math.radians(YAW_RATE_BIAS_DEG_S[1])
    lon_sigma = EKF_SPEED_SIGMA * start_speed * since_fix + EKF_INITIAL_SIGMA_M
    lat_sigma = 0.5 * bias_sigma * start_speed * since_fix.square() + EKF_INITIAL_SIGMA_M
    yaw_sigma0 = bias_sigma * since_fix + math.radians(EKF_INITIAL_YAW_SIGMA_DEG)
    # State (x, y, yaw, yaw-rate bias).
    state = torch.cat([initial, torch.zeros(batch, 1)], dim=-1)
    covariance = torch.diag_embed(torch.stack([
        lon_sigma.square(), lat_sigma.square(), yaw_sigma0.square(), torch.full((batch,), bias_sigma ** 2),
    ], dim=-1))

    observation = torch.eye(2, 4).expand(batch, 2, 4)
    gnss_noise = torch.eye(2) * EKF_GNSS_SIGMA_M ** 2
    gyro = math.radians(EKF_GYRO_NOISE_DEG_S) * DT_S
    walk = math.radians(EKF_BIAS_WALK_DEG_S2) * DT_S

    def update(state, covariance, row):
        residual = fixes[:, row] - state[:, :2]
        inverse = torch.linalg.inv(covariance[:, :2, :2] + gnss_noise)
        mahalanobis = torch.einsum("bi,bij,bj->b", residual, inverse, residual)
        accept = (available[:, row] & (mahalanobis <= GATE_CHI2)).float()[:, None, None]
        gain = covariance @ observation.transpose(1, 2) @ inverse * accept
        return state + (gain @ residual[..., None])[..., 0], (torch.eye(4) - gain @ observation) @ covariance

    state, covariance = update(state, covariance, 0)
    for row in range(1, HISTORY_STEPS + 1):
        v, theta = dr_speed[:, row - 1], state[:, 2]
        w = dr_yaw_rate[:, row - 1] - state[:, 3]
        state = state + torch.stack([v * torch.cos(theta), v * torch.sin(theta), w, torch.zeros_like(v)], -1) * DT_S
        jacobian = torch.eye(4).repeat(batch, 1, 1)
        jacobian[:, 0, 2] = -v * torch.sin(theta) * DT_S
        jacobian[:, 1, 2] = v * torch.cos(theta) * DT_S
        jacobian[:, 2, 3] = -DT_S
        position_variance = (EKF_SPEED_SIGMA * v * DT_S + 1e-3) ** 2
        process = torch.diag_embed(torch.stack([
            position_variance, position_variance, torch.full_like(v, gyro ** 2), torch.full_like(v, walk ** 2),
        ], dim=-1))
        covariance = jacobian @ covariance @ jacobian.transpose(1, 2) + process
        state, covariance = update(state, covariance, row)
    covariance = covariance[:, :3, :3]

    # Express the estimate in the true current ego frame: T_E<-Ê = T_W<-E^-1 T_W<-Ê.
    current = truth[:, -1]
    cos, sin = torch.cos(current[:, 2]), torch.sin(current[:, 2])
    dx, dy = state[:, 0] - current[:, 0], state[:, 1] - current[:, 1]
    pose = torch.stack([
        cos * dx + sin * dy,
        -sin * dx + cos * dy,
        torch.remainder(state[:, 2] - current[:, 2] + math.pi, 2.0 * math.pi) - math.pi,
    ], dim=-1)
    rotation = torch.zeros(batch, 3, 3)
    rotation[:, 0, 0], rotation[:, 0, 1] = cos, sin
    rotation[:, 1, 0], rotation[:, 1, 1] = -sin, cos
    rotation[:, 2, 2] = 1.0
    covariance = rotation @ covariance @ rotation.transpose(1, 2)
    offset = pose[:, :2].norm(dim=-1, keepdim=True)
    pose = torch.cat([
        pose[:, :2] * (MAX_CROP_OFFSET_M / offset.clamp_min(MAX_CROP_OFFSET_M)),
        pose[:, 2:].clamp(-math.radians(MAX_CROP_YAW_DEG), math.radians(MAX_CROP_YAW_DEG)),
    ], dim=-1)
    map_matching = torch.rand(batch, generator=generator) < MAP_MATCHING_SHARE
    return LocalizationEstimate(pose.to(device), covariance.to(device), map_matching.to(device))
