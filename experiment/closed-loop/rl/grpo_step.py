"""One GRPO update of AutoE2E's planner from the closed-loop rollouts of one run.

For every AutoE2E decision of every ``rollout.asl`` under ``run_dir``, until
the ego first leaves the renderer's valid region (3 m from the recorded path),
this rebuilds the driver's inputs from the log (``replay_driver.py``), runs
the frozen encoders once and keeps what the planner received. It then samples
a group of plans around the planner's mean (``grpo.py``) and scores each with
the gated driving reward (``reward.py``). A plan a safety gate rejects scores
``-1 + safe_fraction`` instead of 0: it ranks below every plan that passes,
however little that plan progresses, and among failing plans the one that
stays safe longer ranks higher, so a group in which every plan fails still
has a direction. The states are those the current
policy visited in closed loop, so the update is on-policy even though AlpaSim
executed the mean plan; the replayed mean must reproduce the executed plan.

The step follows the planner-parameter gradient of the natural-gradient
surrogate (natural in the mean plan, not in the parameters). The GRU planner
detaches its recurrent state from the deformable lookup's query, so this is
the gradient of the planner as trained, not the exact gradient of the
surrogate. The step is therefore halved until it both lowers the surrogate
(advantage term plus anchor) on the scored states and keeps the new mean plan
within ``--trust-region`` of the old one on every state: TRPO's line search,
with the largest distance in place of the mean. If no length qualifies, the
planner is left unchanged.

Run inside the AlpaSim image with a GPU (``rl_loop.sh`` does):

    python experiment/closed-loop/rl/grpo_step.py <run_dir> <checkpoint> \\
        --reference <imitation checkpoint> --output <new checkpoint>
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from grpo import group_advantages, grpo_loss, low_frequency_basis, plan_distance, sample_plans  # noqa: E402
from replay_driver import NUPLAN_CAMERAS, prediction_input, read_log  # noqa: E402
from reward import HORIZON_STEPS, scene_scorer  # noqa: E402
from score_rollout import read_rollout, se3, to_trajectory  # noqa: E402

from alpasim_autoe2e.plugin import AutoE2EDriver  # noqa: E402
from alpasim_autoe2e.scene_map import SCENE_ROOT, find_artifact  # noqa: E402
from training.losses.control_rollout import integrate_controls_torch  # noqa: E402

RENDERER_BOUND_M = 3.0
MAX_REPLAY_HEADING_ERROR_RAD = 0.01
WAYPOINT_US = 100_000
MAX_REPLAY_ERROR_M = 0.05
STEPS = 64


class PlannerInputs:
    """Captures the planner's inputs during the driver's no-grad forward."""

    def __init__(self, planner: torch.nn.Module) -> None:
        self.args: tuple = ()
        self.kwargs: dict = {}
        planner.register_forward_pre_hook(self._capture, with_kwargs=True)

    def _capture(self, module, args, kwargs) -> None:
        self.args, self.kwargs = args, kwargs


def logged_plan_in_rig(poses: list) -> tuple[np.ndarray, np.ndarray]:
    """XY and heading of the plan the driver returned, in the rig frame at its anchor."""
    plan = to_trajectory(poses)
    rig_from_local = np.linalg.inv(se3(plan, int(poses[0].timestamp_us)))
    in_rig = [rig_from_local @ se3(plan, int(p.timestamp_us)) for p in poses[1:]]
    return (np.array([m[:2, 3] for m in in_rig]),
            np.array([math.atan2(m[1, 0], m[0, 0]) for m in in_rig]))


def to_local(positions: np.ndarray, headings: np.ndarray, local_from_rig: np.ndarray) -> np.ndarray:
    """``[G, T, 3]`` local x, y, heading of plans given in the rig frame at t0."""
    xy = positions @ local_from_rig[:2, :2].T + local_from_rig[:2, 3]
    yaw0 = np.arctan2(local_from_rig[1, 0], local_from_rig[0, 0])
    return np.concatenate([xy, (headings + yaw0)[..., None]], axis=-1)


def _to(value, device: str):
    """``value`` with every tensor inside tuples and dicts moved to ``device``."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to(v, device) for v in value)
    if isinstance(value, dict):
        return {k: _to(v, device) for k, v in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("checkpoint", type=Path, help="policy that drove the rollouts")
    parser.add_argument("--reference", type=Path, required=True, help="imitation checkpoint the anchor pulls to")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--basis", type=int, default=4, help="low-frequency directions per signal")
    parser.add_argument(
        "--sigma", type=float, nargs=2, default=(0.5, 0.005), metavar=("ACCEL", "CURVATURE"),
        help="std of each orthonormal-basis coefficient; per step about 0.25x this",
    )
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1.0, help="largest step tried along the gradient")
    parser.add_argument("--trust-region", type=float, default=1.0,
                        help="largest plan_distance between new and old mean plans on any scored state")
    parser.add_argument("--seed", type=int, default=0, help="exploration noise; vary it between steps")
    opts = parser.parse_args()

    driver = AutoE2EDriver(str(opts.checkpoint), NUPLAN_CAMERAS, device="cuda")
    model = driver.model
    planner = model.Reactive_E2E.TrajectoryPlanner
    captured = PlannerInputs(planner)
    reference = AutoE2EDriver(
        str(opts.reference), NUPLAN_CAMERAS, device="cuda"
    ).model.Reactive_E2E.TrajectoryPlanner
    reference.requires_grad_(False)
    model.requires_grad_(False)
    planner.requires_grad_(True)
    # cuDNN's GRU backward needs training mode; the planner has no dropout or
    # normalisation, so its output is the same as in eval mode.
    planner.train()

    basis = low_frequency_basis(STEPS, opts.basis)
    sigma = torch.tensor(opts.sigma)
    generator = torch.Generator().manual_seed(opts.seed)
    states: list[tuple[tuple, dict, torch.Tensor]] = []
    rows: list[dict] = []
    for asl in sorted(opts.run_dir.glob("rollouts/*/*/rollout.asl")):
        log = asyncio.run(read_log(asl))
        recorded, logged_plans = asyncio.run(read_rollout(str(asl)))
        artifact = find_artifact(SCENE_ROOT, log.scene_id)
        if artifact is None:
            raise SystemExit(f"no USDZ for scene {log.scene_id!r} under {SCENE_ROOT}")
        # The rollout's last ego pose: AlpaSim clips the objects to the rollout.
        try:
            scorer = scene_scorer(artifact, recorded, int(log.poses[-1].timestamp_us))
        except ValueError as error:  # e.g. a scene without an HD map: no reward to compute
            print(f"skipping {asl}: {error}", file=sys.stderr)
            continue
        for index, now_us in enumerate(log.requests):
            if index >= len(logged_plans):
                break  # the rollout ended before this request was answered
            parsed = driver.parser.parse(prediction_input(log, driver.camera_ids, now_us))
            local_from_rig = np.asarray(parsed.pose_local_to_rig_t0.as_se3(), dtype=np.float64)
            deviation = scorer.path_distance(local_from_rig[:2, 3])
            horizon_end_us = parsed.t0_us + HORIZON_STEPS * WAYPOINT_US
            if deviation > RENDERER_BOUND_M or horizon_end_us >= recorded.time_range_us.stop:
                break  # the renderer is no longer reliable, or the 4 s horizon passes the clip's end
            driver._run_model(parsed)  # no grad; the hook keeps the planner inputs
            with torch.enable_grad():
                mu = planner(*captured.args, **captured.kwargs).reshape(STEPS, 2).float()
            with torch.no_grad():
                mu_ref = reference(*captured.args, **captured.kwargs).reshape(STEPS, 2).float()
            plans, z = sample_plans(mu, basis, sigma, opts.group_size, generator)
            controls = torch.cat([mu.detach()[None], plans]).cpu()
            positions, headings, speeds = integrate_controls_torch(
                controls, torch.full((opts.group_size + 1,), parsed.initial_speed_mps)
            )
            logged_xy, logged_heading = logged_plan_in_rig(logged_plans[index])
            replay_error = float(np.abs(positions[0].numpy() - logged_xy).max())
            heading_error = float(np.abs(np.angle(np.exp(1j * (headings[0].numpy() - logged_heading)))).max())
            if replay_error > MAX_REPLAY_ERROR_M or heading_error > MAX_REPLAY_HEADING_ERROR_RAD:
                raise SystemExit(
                    f"{asl} decision {index + 1}: the replayed plan is {replay_error:.3f} m and "
                    f"{heading_error:.4f} rad from the executed one, so the update would be off-policy"
                )
            local = to_local(positions.numpy(), headings.numpy(), local_from_rig)
            start = to_local(np.zeros((1, 1, 2)), np.zeros((1, 1)), local_from_rig)[0, 0]
            scores = [scorer.score(local[g], speeds[g].numpy(), controls[g].numpy(), start,
                                   parsed.initial_speed_mps, parsed.t0_us)
                      for g in range(opts.group_size + 1)]
            gated = torch.tensor([s["gated"] for s in scores[1:]], dtype=torch.bool)
            rewards = torch.tensor([s["reward"] - 1.0 + s["safe_fraction"] if s["gated"] else s["reward"]
                                    for s in scores[1:]])
            advantages = group_advantages(rewards)
            with torch.enable_grad():
                loss, anchor = grpo_loss(mu, mu_ref, z, advantages, basis, sigma, opts.anchor_weight)
                loss.backward()
            # On the CPU: one state's BEV features can take over 100 MB of GPU memory.
            states.append((_to(captured.args, "cpu"), _to(captured.kwargs, "cpu"), mu.detach().cpu(),
                           mu_ref.cpu(), z.cpu(), advantages))
            rows.append({
                "rollout": str(asl.relative_to(opts.run_dir)),
                "decision": index + 1,
                "ego_deviation_m": deviation,
                "replay_error_m": replay_error,
                "mean_plan": scores[0],
                "group_reward_mean": float(rewards.mean()),
                "group_reward_std": float(rewards.std(unbiased=False)),
                "group_gated": int(gated.sum()),
                "anchor": float(anchor),
            })
    if not rows:
        raise SystemExit(f"no decision within the valid region under {opts.run_dir}")

    # The gradient averaged over the decisions, then TRPO's backtracking line search.
    parameters = [p for p in planner.parameters()]
    start_params = [p.detach().clone() for p in parameters]
    direction = [-(p.grad / len(rows)) if p.grad is not None else torch.zeros_like(p) for p in parameters]

    def evaluate() -> tuple[float, float]:
        """Mean surrogate and largest plan change of the current planner on the scored states."""
        losses, moves = [], []
        for args, kwargs, mu_old, mu_ref, z, advantages in states:
            mu_new = planner(*_to(args, "cuda"), **_to(kwargs, "cuda")).reshape(STEPS, 2).float().cpu()
            policy = -(advantages * (z * (basis.T @ (mu_new - mu_old))).sum(dim=(1, 2))).mean()
            losses.append(float(policy + opts.anchor_weight * ((mu_new - mu_ref) ** 2).sum() / 2.0))
            moves.append(float(plan_distance(mu_new, mu_old, sigma)))
        return float(np.mean(losses)), max(moves)

    with torch.no_grad():
        surrogate_before, _ = evaluate()
        scale, moved, surrogate_after = opts.lr, 0.0, surrogate_before
        for _ in range(12):
            for p, p0, d in zip(parameters, start_params, direction):
                p.copy_(p0 + scale * d)
            surrogate_after, moved = evaluate()
            if moved <= opts.trust_region and surrogate_after < surrogate_before:
                break
            scale /= 2.0
        else:
            for p, p0 in zip(parameters, start_params):
                p.copy_(p0)
            scale, moved, surrogate_after = 0.0, 0.0, surrogate_before

    summary = {
        "decisions": len(rows),
        "mean_plan_reward": float(np.mean([r["mean_plan"]["reward"] for r in rows])),
        "mean_plan_gated": float(np.mean([r["mean_plan"]["gated"] for r in rows])),
        "group_reward_std": float(np.mean([r["group_reward_std"] for r in rows])),
        "anchor": float(np.mean([r["anchor"] for r in rows])),
        "step_scale": scale,
        "step_plan_distance": moved,
        "surrogate_before": surrogate_before,
        "surrogate_after": surrogate_after,
        "max_replay_error_m": float(max(r["replay_error_m"] for r in rows)),
    }
    checkpoint = torch.load(opts.checkpoint, map_location="cpu")
    checkpoint["model_state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
    checkpoint["closed_loop_grpo"] = {
        "parent_sha256": hashlib.sha256(opts.checkpoint.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(opts.reference.read_bytes()).hexdigest(),
        "run_dir": str(opts.run_dir),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(opts).items()},
        "summary": summary,
    }
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, opts.output)
    opts.output.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "decisions": rows}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
