"""Fidelity-aware reward for stage-3 closed-loop RL (issue #123).

v1 (the issue's concrete formula, minus an imitation KL term)::

    R = w_safe R_safety + w_prog R_progress + w_comf R_comfort
      + g * R_wm

Handcrafted terms are **not** imitation:

* ``R_safety`` is mean lane-departure in the intended-path Frenet frame
  (cross-track vs the path heading along the trajectory), not ``|y|`` in
  the t=0 ego frame. Turning in a ~3.5 m lane is free; leaving it is not.
* ``R_progress`` is along-track displacement of the predicted trajectory
  along that same path, normalized by the coast horizon. Matching the
  expert's positions is not the objective; going forward along the path is.
* ``R_comfort`` is jerk / lateral-accel excess vs the physical thresholds
  already used by ``RolloutAlignedLoss``, not vs the expert log.

``g = g_sat * exp(-mse / T)`` gates **only** the WM consequence term — a
broken WM must not zero collision/comfort. ``T`` and ``g_sat`` are derived
below from the offline experiment's WM residual scale; ``g_sat < 1`` so a
"good enough" reconstruction does not fully trust a (possibly misleading)
consequence.

This is the tensor #177 (AlpaSim closed-loop) should call later. It does
not import that PR.

The reasoning-band is intentionally absent: #123's DPI trap.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from training.losses.control_rollout import integrate_controls_torch


FIDELITY_AWARE_REWARD_VERSION = "v1_safety_comfort_progress_gated_wm"

V1_WEIGHTS = {
    "safety": 1.0,
    "progress": 1.0,
    "comfort": 0.5,
}

# Physical comfort limits — same numbers as RolloutAlignedLoss.
JERK_THRESHOLD_MPS3 = 4.13
LATERAL_ACCEL_THRESHOLD_MPS2 = 4.89

# --- fidelity temperature / saturation --------------------------------------
# Offline experiment WM residual is an (B, 8) tensor.
#   faithful arm: mse = 0
#   noise arm:    pred ~ N(0, σ_noise²) with σ_noise = 10  →  E[mse] = 100
# A barely-faithful WM is defined as 10× smaller residual than that noise
# (σ_good = 1). T is its expected per-element MSE, so:
#   g(barely-faithful) = g_sat / e
#   g(noise)           = g_sat * exp(-100) ≈ 0
# and the exponential still moves in the operating range instead of sitting
# at the ceiling for every "pretty good" WM.
EXPERIMENT_WM_DIM = 8
EXPERIMENT_NOISE_WM_SIGMA = 10.0
FAITHFUL_NOISE_RATIO = 10.0
FIDELITY_TEMPERATURE = (
    EXPERIMENT_NOISE_WM_SIGMA / FAITHFUL_NOISE_RATIO
) ** 2  # 1.0

# Even mse = 0 does not fully trust R_wm: reconstruction fidelity is not
# consequence-target fidelity (the experiment's misleading preference).
# Rule of succession on the experiment's 8-d residual: after D exact
# matches, P(trust) = (D+1)/(D+2) = 0.9, not 1.
FIDELITY_SATURATION = (EXPERIMENT_WM_DIM + 1) / (EXPERIMENT_WM_DIM + 2)

# Prefix/suffix used to treat the intended path as an infinite lane so
# along-track is not capped at the expert's last point (that cap would
# re-introduce imitation).
_LANE_EXTENSION_M = 1.0e3


@dataclass(frozen=True)
class FidelityAwareRewardResult:
    """Per-sample reward and diagnostic scalars."""

    reward: torch.Tensor
    fidelity: torch.Tensor
    base_reward: torch.Tensor
    consequence_reward: torch.Tensor | None
    metadata: dict[str, float]


def world_model_fidelity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = FIDELITY_TEMPERATURE,
    saturation: float = FIDELITY_SATURATION,
    reduce_dims: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Map WM prediction error to a ``(0, g_sat]`` fidelity weight.

    ``fidelity = saturation * exp(-mse / temperature)``. High fidelity
    means the world model is a trustworthy source of consequence
    information for this sample; low fidelity down-weights any reward
    that depends on those predictions. ``saturation < 1`` keeps a
    "good enough" WM from fully deferring to a misspecified consequence.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {tuple(prediction.shape)} vs "
            f"{tuple(target.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if saturation <= 0 or saturation > 1:
        raise ValueError(
            f"saturation must be in (0, 1], got {saturation}"
        )

    err = (prediction.float() - target.float()).pow(2)
    if reduce_dims is None:
        # Reduce all but batch dim when present.
        if err.ndim == 0:
            mse = err
        else:
            mse = err.flatten(1).mean(dim=1)
    else:
        mse = err.mean(dim=reduce_dims)

    return float(saturation) * torch.exp(-mse / temperature)


def consequence_alignment_reward(
    predicted_future: torch.Tensor,
    preferred_future: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward from WM-predicted futures vs an external preference target.

    ``preferred_future`` is intended to carry information outside the
    reactive policy's inputs (e.g. HD-map / LiDAR conflict geometry, or a
    human preference embedding) — not a re-encoding of BEV/ego already fed
    to the planner.
    """
    if predicted_future.shape != preferred_future.shape:
        raise ValueError(
            "predicted_future/preferred_future shape mismatch: "
            f"{tuple(predicted_future.shape)} vs {tuple(preferred_future.shape)}"
        )
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    # Negative MSE so better alignment → higher reward. Keep per-batch.
    if predicted_future.ndim == 0:
        mse = (predicted_future.float() - preferred_future.float()).pow(2)
    else:
        mse = (
            (predicted_future.float() - preferred_future.float())
            .pow(2)
            .flatten(1)
            .mean(dim=1)
        )
    return -scale * mse


def _polyline_along_cross(
    pos: torch.Tensor,
    ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-point Frenet coords of ``pos`` vs polyline ``ref``.

    Both ``(B, T, 2)``. Along-track is arc length from ``ref[:, 0]``;
    cross-track is signed left-positive distance to the closest segment.
    """
    seg = ref[:, 1:, :] - ref[:, :-1, :]
    seg_len = torch.linalg.norm(seg, dim=-1)
    seg_len_sq = seg_len.square().clamp(min=1e-12)
    zeros = torch.zeros(
        ref.shape[0], 1, dtype=ref.dtype, device=ref.device,
    )
    arc_start = torch.cat(
        [zeros, torch.cumsum(seg_len, dim=-1)[:, :-1]], dim=-1,
    )

    start = ref[:, None, :-1, :]
    rel = pos[:, :, None, :] - start
    t_proj = (rel * seg[:, None, :, :]).sum(-1) / seg_len_sq[:, None, :]
    t_clamped = t_proj.clamp(0.0, 1.0)
    closest = start + t_clamped.unsqueeze(-1) * seg[:, None, :, :]
    offset = pos[:, :, None, :] - closest
    dist2 = (offset * offset).sum(-1)
    idx = dist2.argmin(dim=-1)

    batch = torch.arange(pos.shape[0], device=pos.device)[:, None]
    time = torch.arange(pos.shape[1], device=pos.device)[None, :]
    t_star = t_clamped[batch, time, idx]
    along = arc_start[batch, idx] + t_star * seg_len[batch, idx]
    seg_n = seg[batch, idx]
    off_n = offset[batch, time, idx]
    denom = seg_len[batch, idx].clamp(min=1e-6)
    cross = (
        seg_n[..., 0] * off_n[..., 1] - seg_n[..., 1] * off_n[..., 0]
    ) / denom
    return along, cross


def path_frame_along_cross(
    pos: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_heading: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Along-track / cross-track of ``pos`` in the intended-path frame.

    The path is the polyline ``ref_pos``, extended along ``ref_heading``
    at both ends so progress is not capped at the last logged point (that
    cap would reward matching the expert's horizon, i.e. imitation).

    This is **not** ``|y|`` in the t=0 ego frame: a curved in-lane path
    has large ego-y and near-zero cross-track.
    """
    u0 = torch.stack(
        [ref_heading[:, 0].cos(), ref_heading[:, 0].sin()], dim=-1,
    )
    u1 = torch.stack(
        [ref_heading[:, -1].cos(), ref_heading[:, -1].sin()], dim=-1,
    )
    origin = torch.zeros(
        ref_pos.shape[0], 1, 2, dtype=ref_pos.dtype, device=ref_pos.device,
    )
    extend = pos.new_tensor(_LANE_EXTENSION_M)
    prefix = origin - extend * u0[:, None, :]
    suffix = ref_pos[:, -1:, :] + extend * u1[:, None, :]
    ref_ext = torch.cat([prefix, origin, ref_pos, suffix], dim=1)
    along_ext, cross = _polyline_along_cross(pos, ref_ext)
    return along_ext - _LANE_EXTENSION_M, cross


def _absolute_comfort_penalty(
    controls: torch.Tensor,
    speeds: torch.Tensor,
    *,
    dt: float,
) -> torch.Tensor:
    """Mean jerk / lateral-accel excess vs physical thresholds, not the log.

    ``comfort_excess_per_sample`` charges ``relu(peak_pred - peak_target)``,
    which is zero when the expert is equally jerky. v1 must not do that.
    """
    accel = controls[:, :, 0]
    curvature = controls[:, :, 1]
    jerk = (accel[:, 1:] - accel[:, :-1]) / dt
    lateral = speeds.square() * curvature
    jerk_excess = torch.relu(jerk.abs() - JERK_THRESHOLD_MPS3).mean(dim=1)
    lateral_excess = torch.relu(
        lateral.abs() - LATERAL_ACCEL_THRESHOLD_MPS2,
    ).mean(dim=1)
    return 0.5 * (jerk_excess + lateral_excess)


def v1_handcrafted_reward(
    controls: torch.Tensor,
    intended_controls: torch.Tensor,
    initial_speed: torch.Tensor,
    *,
    dt: float = 0.1,
    lane_half_width_m: float = 1.75,
) -> dict[str, torch.Tensor]:
    """#123 handcrafted terms in the intended-path frame.

    ``intended_controls`` is the logged plan used as **path geometry**
    (centerline heading along the trajectory), not an imitation target.
    Returns per-sample ``safety``, ``progress``, ``comfort``, and ``base``
    (weighted sum). Higher is better.
    """
    pos, _, speeds = integrate_controls_torch(controls, initial_speed, dt=dt)
    ref_pos, ref_heading, _ = integrate_controls_torch(
        intended_controls, initial_speed, dt=dt,
    )
    along, cross = path_frame_along_cross(pos, ref_pos, ref_heading)
    horizon_m = initial_speed.clamp(min=1e-3) * dt * pos.shape[1]
    r_progress = along[:, -1] / horizon_m
    r_safety = -torch.relu(cross.abs() - lane_half_width_m).mean(dim=1)
    r_comfort = -_absolute_comfort_penalty(controls, speeds, dt=dt)
    base = (
        V1_WEIGHTS["safety"] * r_safety
        + V1_WEIGHTS["progress"] * r_progress
        + V1_WEIGHTS["comfort"] * r_comfort
    )
    return {
        "safety": r_safety,
        "progress": r_progress,
        "comfort": r_comfort,
        "base": base,
        "along_track_m": along[:, -1],
        "cross_track_m": cross.abs().amax(dim=1),
    }


def fidelity_aware_reward(
    base_reward: torch.Tensor,
    *,
    wm_prediction: torch.Tensor,
    wm_target: torch.Tensor,
    preferred_future: torch.Tensor | None = None,
    predicted_future: torch.Tensor | None = None,
    fidelity_temperature: float = FIDELITY_TEMPERATURE,
    fidelity_saturation: float = FIDELITY_SATURATION,
    consequence_scale: float = 1.0,
    consequence_weight: float = 1.0,
    min_fidelity: float = 0.0,
) -> FidelityAwareRewardResult:
    """#123 v1: ``R = base + g * R_wm``. ``g`` does not multiply safety/comfort."""
    if base_reward.ndim > 1:
        raise ValueError(
            f"base_reward must be scalar or (B,), got shape {tuple(base_reward.shape)}"
        )

    fidelity = world_model_fidelity(
        wm_prediction,
        wm_target,
        temperature=fidelity_temperature,
        saturation=fidelity_saturation,
    )
    if fidelity.shape != base_reward.shape and not (
        fidelity.ndim == 1 and base_reward.ndim == 0
    ):
        # Allow broadcasting scalar base over batch fidelity.
        if base_reward.ndim == 0 and fidelity.ndim == 1:
            base_reward = base_reward.expand_as(fidelity)
        elif fidelity.ndim == 0 and base_reward.ndim == 1:
            fidelity = fidelity.expand_as(base_reward)
        elif fidelity.shape != base_reward.shape:
            raise ValueError(
                "fidelity/base_reward shape mismatch after reduction: "
                f"{tuple(fidelity.shape)} vs {tuple(base_reward.shape)}"
            )

    fidelity = fidelity.clamp(min=float(min_fidelity), max=1.0)

    consequence: torch.Tensor | None = None
    total = base_reward.float()
    if preferred_future is not None:
        if predicted_future is None:
            raise ValueError(
                "predicted_future is required when preferred_future is provided"
            )
        consequence = consequence_alignment_reward(
            predicted_future,
            preferred_future,
            scale=consequence_scale,
        )
        if consequence.shape != total.shape:
            raise ValueError(
                "consequence/base_reward shape mismatch: "
                f"{tuple(consequence.shape)} vs {tuple(total.shape)}"
            )
        total = total + float(consequence_weight) * fidelity * consequence

    reward = total
    metadata = {
        "fidelity_mean": float(fidelity.detach().mean().cpu()),
        "base_reward_mean": float(base_reward.detach().float().mean().cpu()),
        "reward_mean": float(reward.detach().mean().cpu()),
        "fidelity_temperature": float(fidelity_temperature),
        "fidelity_saturation": float(fidelity_saturation),
        "consequence_weight": float(consequence_weight),
        "min_fidelity": float(min_fidelity),
    }
    if consequence is not None:
        metadata["consequence_reward_mean"] = float(
            consequence.detach().mean().cpu()
        )

    return FidelityAwareRewardResult(
        reward=reward,
        fidelity=fidelity,
        base_reward=base_reward.float(),
        consequence_reward=consequence,
        metadata=metadata,
    )


def soft_advantage_from_reward(
    reward: torch.Tensor,
    *,
    baseline: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Optional REINFORCE-style advantage: ``(r - baseline) / temperature``.

    Kept separate from :func:`fidelity_aware_reward` so callers can plug the
    gated reward into whatever policy-gradient estimator they choose once
    ``compute_planner_loss`` (#115) is restored.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if baseline is None:
        baseline = reward.detach().mean()
    return (reward - baseline) / temperature
