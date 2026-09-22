"""Offline #123 experiment: expert vs jerky ranking under a WM fidelity gate.

Handcrafted safety/comfort/progress should prefer the expert plan. A
*misleading* WM consequence term would prefer the jerky plan if left ungated.
When the WM is noise, ``g ≈ 0`` so the ranking must match the handcrafted
base — the gate AlpaSim (#177) should call, without importing that PR.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from training.losses.control_rollout import integrate_controls_torch
from training.losses.fidelity_aware_reward import (
    EXPERIMENT_NOISE_WM_SIGMA,
    EXPERIMENT_WM_DIM,
    FIDELITY_AWARE_REWARD_VERSION,
    FIDELITY_SATURATION,
    FIDELITY_TEMPERATURE,
    fidelity_aware_reward,
    v1_handcrafted_reward,
)


def _pair(
    timesteps: int = 32,
    dt: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.arange(timesteps, dtype=torch.float32)
    time_s = t * dt
    expert = torch.zeros(1, timesteps, 2)
    # Original ``8.0 * sin(2π t / 2)`` with t = 0,1,2,... is sin(π t) and
    # every integer sample is a zero crossing (max |a| ~ 4e-5). Use time
    # in seconds (2 s period, as written) plus π/4 so sample instants are
    # off the zeros. Curvature already samples off-zero at period 3.
    jerky = torch.stack(
        (
            8.0 * torch.sin(2.0 * torch.pi * time_s / 2.0 + 0.25 * torch.pi),
            0.25 * torch.sin(2.0 * torch.pi * t / 3.0),
        ),
        dim=-1,
    ).unsqueeze(0)
    controls = torch.cat([expert, jerky], dim=0)
    expert_batch = expert.expand(2, -1, -1).contiguous()
    v0 = torch.full((2,), 5.0)
    return controls, expert_batch, v0


def run_fidelity_reward_experiment(*, seed: int = 123) -> dict[str, Any]:
    torch.manual_seed(seed)
    controls, expert, v0 = _pair()
    terms = v1_handcrafted_reward(controls, expert, v0)
    base = terms["base"]
    expert_minus_jerky = float(base[0] - base[1])

    pos, _, _ = integrate_controls_torch(controls, v0)
    max_abs_accel = float(controls[1, :, 0].abs().max())
    max_abs_ego_y = float(pos[1, :, 1].abs().max())
    max_abs_cross_track = float(terms["cross_track_m"][1])

    # Misleading WM preference: consequence wants the jerky sample.
    # Magnitude is 2× the handcrafted gap / g_sat so a saturated faithful
    # WM still flips, while g ≈ 0 cannot. Reconstruction fidelity is not
    # preference fidelity — that is the point of the gate.
    preferred = torch.ones(2, 4)
    gap = max(expert_minus_jerky, 0.05)
    mse_needed = 2.0 * gap / FIDELITY_SATURATION
    delta = math.sqrt(mse_needed)
    predicted = torch.stack(
        [torch.full((4,), 1.0 + delta), torch.ones(4)],
    )

    faithful = fidelity_aware_reward(
        base,
        wm_prediction=torch.zeros(2, EXPERIMENT_WM_DIM),
        wm_target=torch.zeros(2, EXPERIMENT_WM_DIM),
        predicted_future=predicted,
        preferred_future=preferred,
        consequence_weight=1.0,
        consequence_scale=1.0,
    )
    noise_pred = EXPERIMENT_NOISE_WM_SIGMA * torch.randn(2, EXPERIMENT_WM_DIM)
    noisy = fidelity_aware_reward(
        base,
        wm_prediction=noise_pred,
        wm_target=torch.zeros(2, EXPERIMENT_WM_DIM),
        predicted_future=predicted,
        preferred_future=preferred,
        consequence_weight=1.0,
        consequence_scale=1.0,
    )

    def _rank(reward: torch.Tensor) -> str:
        return "expert" if float(reward[0] - reward[1]) > 0 else "jerky"

    base_winner = _rank(base)
    return {
        "version": FIDELITY_AWARE_REWARD_VERSION,
        "source": "constructed_expert_vs_jerky",
        "fidelity_temperature": FIDELITY_TEMPERATURE,
        "fidelity_saturation": FIDELITY_SATURATION,
        "jerky_max_abs_accel": max_abs_accel,
        "jerky_max_abs_ego_y": max_abs_ego_y,
        "jerky_max_abs_cross_track": max_abs_cross_track,
        "base": {
            "expert": float(base[0]),
            "jerky": float(base[1]),
            "expert_minus_jerky": expert_minus_jerky,
            "winner": base_winner,
            "safety_expert": float(terms["safety"][0]),
            "safety_jerky": float(terms["safety"][1]),
            "progress_expert": float(terms["progress"][0]),
            "progress_jerky": float(terms["progress"][1]),
            "comfort_expert": float(terms["comfort"][0]),
            "comfort_jerky": float(terms["comfort"][1]),
        },
        "faithful_wm": {
            "expert": float(faithful.reward[0]),
            "jerky": float(faithful.reward[1]),
            "winner": _rank(faithful.reward),
            "fidelity_mean": faithful.metadata["fidelity_mean"],
        },
        "noise_wm": {
            "expert": float(noisy.reward[0]),
            "jerky": float(noisy.reward[1]),
            "winner": _rank(noisy.reward),
            "fidelity_mean": noisy.metadata["fidelity_mean"],
            "ranking_matches_base": _rank(noisy.reward) == base_winner,
        },
    }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    report = run_fidelity_reward_experiment(seed=args.seed)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
