"""Unit tests for the fidelity-aware reward prototype (#123)."""

from __future__ import annotations

import math

import torch

from training.losses.control_rollout import integrate_controls_torch
from training.losses.fidelity_aware_reward import (
    EXPERIMENT_NOISE_WM_SIGMA,
    EXPERIMENT_WM_DIM,
    FAITHFUL_NOISE_RATIO,
    FIDELITY_AWARE_REWARD_VERSION,
    FIDELITY_SATURATION,
    FIDELITY_TEMPERATURE,
    consequence_alignment_reward,
    fidelity_aware_reward,
    soft_advantage_from_reward,
    v1_handcrafted_reward,
    world_model_fidelity,
)


def test_version_pin() -> None:
    assert FIDELITY_AWARE_REWARD_VERSION == "v1_safety_comfort_progress_gated_wm"


def test_fidelity_temperature_is_derived_from_experiment_mse() -> None:
    assert FAITHFUL_NOISE_RATIO == 10.0
    assert EXPERIMENT_NOISE_WM_SIGMA == 10.0
    assert FIDELITY_TEMPERATURE == (
        EXPERIMENT_NOISE_WM_SIGMA / FAITHFUL_NOISE_RATIO
    ) ** 2
    assert FIDELITY_SATURATION == (
        EXPERIMENT_WM_DIM + 1
    ) / (EXPERIMENT_WM_DIM + 2)
    assert FIDELITY_SATURATION < 1.0


def test_perfect_wm_fidelity_saturates_below_one() -> None:
    x = torch.randn(4, 8, 8)
    fid = world_model_fidelity(x, x.clone())
    assert torch.allclose(fid, torch.full_like(fid, FIDELITY_SATURATION))
    assert torch.all(fid < 1.0)


def test_error_lowers_fidelity() -> None:
    pred = torch.zeros(2, 4)
    target = torch.ones(2, 4)
    fid = world_model_fidelity(pred, target, temperature=FIDELITY_TEMPERATURE)
    assert torch.all(fid < FIDELITY_SATURATION)
    assert torch.all(fid > 0.0)


def test_good_enough_wm_does_not_hit_saturation_ceiling() -> None:
    """mse = T (barely-faithful) must sit at g_sat/e, not g_sat."""
    pred = torch.zeros(2, EXPERIMENT_WM_DIM)
    target = torch.ones(2, EXPERIMENT_WM_DIM)
    fid = world_model_fidelity(pred, target)
    expected = FIDELITY_SATURATION * math.exp(-1.0 / FIDELITY_TEMPERATURE)
    assert torch.allclose(fid, torch.full_like(fid, expected), atol=1e-5)
    perfect = world_model_fidelity(pred, pred)
    assert float(perfect[0]) > float(fid[0])
    assert float(perfect[0]) < 1.0


def test_fidelity_does_not_zero_handcrafted_base() -> None:
    """#123: a broken WM must not wipe safety/comfort/progress."""
    base = torch.tensor([1.0, 1.0])
    wm_pred = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    wm_tgt = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    out = fidelity_aware_reward(
        base,
        wm_prediction=wm_pred,
        wm_target=wm_tgt,
    )
    assert torch.allclose(out.reward, base)
    assert abs(out.fidelity[0].item() - FIDELITY_SATURATION) < 1e-6
    assert out.fidelity[1].item() < out.fidelity[0].item()


def test_consequence_term_requires_external_preference() -> None:
    base = torch.zeros(2)
    wm = torch.zeros(2, 3)
    pred_fut = torch.zeros(2, 3)
    pref = torch.ones(2, 3)
    out = fidelity_aware_reward(
        base,
        wm_prediction=wm,
        wm_target=wm.clone(),
        predicted_future=pred_fut,
        preferred_future=pref,
        consequence_weight=1.0,
        consequence_scale=1.0,
    )
    # Saturated WM, consequence = -mse = -1 → reward -g_sat.
    assert torch.allclose(
        out.reward,
        torch.full((2,), -FIDELITY_SATURATION),
    )
    assert out.consequence_reward is not None


def test_consequence_alignment_reward_shapes() -> None:
    r = consequence_alignment_reward(torch.zeros(3, 2), torch.ones(3, 2), scale=2.0)
    assert r.shape == (3,)
    assert torch.all(r < 0)


def test_soft_advantage_zero_mean_without_baseline() -> None:
    reward = torch.tensor([1.0, 3.0])
    adv = soft_advantage_from_reward(reward)
    assert abs(adv.mean().item()) < 1e-6


def test_safety_ignores_in_lane_turn() -> None:
    """Curvature 0.05 at 32 steps / 5 m/s has |y| > 1.75 and used to cost ~1.07.

    In the intended-path frame the same curve is the lane centerline, so
    r_safety must be ~0. This fails on the old ego-y gate.
    """
    timesteps = 32
    dt = 0.1
    curve = torch.zeros(1, timesteps, 2)
    curve[:, :, 1] = 0.05
    v0 = torch.tensor([5.0])
    terms = v1_handcrafted_reward(curve, curve, v0, dt=dt)
    pos, _, _ = integrate_controls_torch(curve, v0, dt=dt)
    ego_y_penalty = float(torch.relu(pos[..., 1].abs() - 1.75).mean())
    assert pos[0, :, 1].abs().max().item() > 1.75
    assert ego_y_penalty > 1.0
    assert abs(terms["safety"].item()) < 1e-4


def test_safety_penalises_leaving_lane() -> None:
    timesteps = 32
    dt = 0.1
    straight = torch.zeros(1, timesteps, 2)
    leave = torch.zeros(1, timesteps, 2)
    leave[:, :, 1] = 0.05
    v0 = torch.tensor([5.0])
    terms = v1_handcrafted_reward(leave, straight, v0, dt=dt)
    assert terms["safety"].item() < -1.0
    assert terms["cross_track_m"].item() > 1.75


def test_progress_rewards_forward_motion_not_imitation() -> None:
    """Stopped-vs-stopped is the old imitation optimum; v1 must prefer coasting."""
    timesteps = 32
    brake = torch.zeros(2, timesteps, 2)
    brake[:, :, 0] = -100.0
    coast = torch.zeros(2, timesteps, 2)
    controls = torch.stack([brake[0], coast[0]])
    v0 = torch.tensor([5.0, 5.0])
    terms = v1_handcrafted_reward(controls, brake, v0)
    assert terms["progress"][1].item() > terms["progress"][0].item()


def test_comfort_uses_physical_threshold_not_expert_log() -> None:
    """Matching a jerky log used to yield comfort 0; v1 must still penalise."""
    timesteps = 16
    t = torch.arange(timesteps, dtype=torch.float32)
    jerky = torch.stack(
        (
            8.0 * torch.sin(math.pi * t + 0.5),
            torch.zeros(timesteps),
        ),
        dim=-1,
    ).unsqueeze(0)
    v0 = torch.tensor([5.0])
    terms = v1_handcrafted_reward(jerky, jerky, v0)
    assert terms["comfort"].item() < 0.0


def test_jerky_experiment_signal_has_nonzero_accel_at_samples() -> None:
    from evaluation.fidelity_reward_experiment import _pair

    controls, _, _ = _pair()
    jerky_accel = controls[1, :, 0]
    assert jerky_accel.abs().max().item() > 1.0


def test_expert_outranks_jerky_on_v1_base() -> None:
    from evaluation.fidelity_reward_experiment import run_fidelity_reward_experiment

    report = run_fidelity_reward_experiment()
    assert report["base"]["expert_minus_jerky"] > 0
    assert abs(report["base"]["progress_expert"] - 1.0) < 0.02
    assert report["base"]["comfort_jerky"] < report["base"]["comfort_expert"]
    assert report["jerky_max_abs_accel"] > 1.0
    assert report["noise_wm"]["ranking_matches_base"] is True
    assert report["noise_wm"]["fidelity_mean"] < 0.05
    assert abs(report["faithful_wm"]["fidelity_mean"] - FIDELITY_SATURATION) < 1e-6
    assert report["faithful_wm"]["winner"] == "jerky"
    assert report["fidelity_temperature"] == FIDELITY_TEMPERATURE
    assert report["fidelity_saturation"] == FIDELITY_SATURATION
    assert report["faithful_wm"]["fidelity_mean"] < 1.0
