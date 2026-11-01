"""One GRPO update of AutoE2E's planner from the closed-loop rollouts of one run.

For every AutoE2E decision of every ``rollout.asl`` under ``run_dir``, until
the ego first leaves the renderer's valid region (the phase-1 bound), this
rebuilds the driver's inputs from the log (``replay_driver.py``), runs the
frozen encoders once and keeps what the planner received. It then samples a
group of plans around the planner's mean (``grpo.py``), scores each against
the recorded drive with the phase-1 reward and takes one SGD step on the
planner. The states are those the current policy visited in closed loop, so
the update is on-policy even though AlpaSim executed the mean plan.

Run inside the AlpaSim image with a GPU (``rl_loop.sh`` does):

    python experiment/closed-loop/rl/grpo_step.py <run_dir> <checkpoint> \\
        --reference <imitation checkpoint> --output <new checkpoint>
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from grpo import group_advantages, grpo_loss, low_frequency_basis, sample_plans  # noqa: E402
from replay_driver import NUPLAN_CAMERAS, prediction_input, read_log  # noqa: E402
from score_rollout import read_rollout, se3, to_trajectory  # noqa: E402

from alpasim_autoe2e.plugin import AutoE2EDriver  # noqa: E402
from alpasim_autoe2e.rewards import GroundTruthDeviationReward  # noqa: E402
from training.losses.control_rollout import integrate_controls_torch  # noqa: E402

WAYPOINT_STEP_US = 100_000


class PlannerInputs:
    """Captures the planner's inputs during the driver's no-grad forward."""

    def __init__(self, planner: torch.nn.Module) -> None:
        self.args: tuple = ()
        self.kwargs: dict = {}
        planner.register_forward_pre_hook(self._capture, with_kwargs=True)

    def _capture(self, module, args, kwargs) -> None:
        self.args, self.kwargs = args, kwargs


def recorded_in_rig(recorded, pose_t0_se3: np.ndarray, t0_us: int, steps: int) -> tuple[np.ndarray, float]:
    """Recorded positions at the plan's timestamps and the ego's offset from the recording at t0."""
    rig_from_local = np.linalg.inv(pose_t0_se3)
    times = [t0_us + WAYPOINT_STEP_US * k for k in range(1, steps + 1)]
    xy = [(rig_from_local @ se3(recorded, t))[:2, 3] for t in times if t in recorded.time_range_us]
    deviation = float(np.linalg.norm((rig_from_local @ se3(recorded, t0_us))[:2, 3]))
    return np.array(xy).reshape(-1, 2), deviation


def logged_plan_in_rig(poses: list) -> np.ndarray:
    """The plan the driver returned, in the rig frame at its anchor (as ``score_rollout``)."""
    plan = to_trajectory(poses)
    rig_from_local = np.linalg.inv(se3(plan, int(poses[0].timestamp_us)))
    return np.array([(rig_from_local @ se3(plan, int(p.timestamp_us)))[:2, 3] for p in poses[1:]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("checkpoint", type=Path, help="policy that drove the rollouts")
    parser.add_argument("--reference", type=Path, required=True, help="imitation checkpoint the KL anchors to")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--basis", type=int, default=4, help="low-frequency directions per signal")
    parser.add_argument(
        "--sigma", type=float, nargs=2, default=(0.5, 0.005), metavar=("ACCEL", "CURVATURE"),
        help="std of each orthonormal-basis coefficient; per step about 0.15x this",
    )
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
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
    optimizer = torch.optim.SGD(planner.parameters(), lr=opts.lr)

    reward = GroundTruthDeviationReward()
    basis = low_frequency_basis(64, opts.basis)
    sigma = torch.tensor(opts.sigma)
    generator = torch.Generator().manual_seed(opts.seed)
    rows: list[dict] = []
    optimizer.zero_grad()
    for asl in sorted(opts.run_dir.glob("rollouts/*/*/rollout.asl")):
        log = asyncio.run(read_log(asl))
        recorded, logged_plans = asyncio.run(read_rollout(str(asl)))
        for index, now_us in enumerate(log.requests):
            parsed = driver.parser.parse(prediction_input(log, driver.camera_ids, now_us))
            recorded_xy, deviation = recorded_in_rig(
                recorded, np.asarray(parsed.pose_local_to_rig_t0.as_se3()), parsed.t0_us, 64
            )
            if deviation > reward.max_deviation_m or not len(recorded_xy):
                break  # renderer no longer reliable (or the clip ended): truncated here
            driver._run_model(parsed)  # no grad; the hook keeps the planner inputs
            with torch.enable_grad():
                mu = planner(*captured.args, **captured.kwargs).reshape(64, 2).float()
                with torch.no_grad():
                    mu_ref = reference(*captured.args, **captured.kwargs).reshape(64, 2).float()
                plans, z = sample_plans(mu, basis, sigma, opts.group_size, generator)
                positions, _, _ = integrate_controls_torch(
                    torch.cat([mu.detach()[None], plans]).cpu(),
                    torch.full((opts.group_size + 1,), parsed.initial_speed_mps),
                )
                scores = [reward.compute(xy, recorded_xy, deviation) for xy in positions.numpy()]
                rewards = torch.tensor([s.reward for s in scores[1:]])
                loss, kl = grpo_loss(
                    mu, mu_ref, z, group_advantages(rewards), basis, sigma, opts.kl_weight
                )
                loss.backward()
            logged = logged_plan_in_rig(logged_plans[index])
            rows.append({
                "rollout": str(asl.relative_to(opts.run_dir)),
                "decision": index + 1,
                "ego_deviation_m": deviation,
                "mean_plan_reward": scores[0].reward,
                "group_reward_mean": float(rewards.mean()),
                "group_reward_std": float(rewards.std(unbiased=False)),
                "kl": float(kl),
                # The replayed mean plan should be the plan the driver executed.
                "replay_error_m": float(np.abs(positions[0].numpy() - logged).max()),
            })
    if not rows:
        raise SystemExit(f"no decision within the valid region under {opts.run_dir}")
    for parameter in planner.parameters():
        if parameter.grad is not None:
            parameter.grad /= len(rows)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(planner.parameters(), float("inf")))
    optimizer.step()

    checkpoint = torch.load(opts.checkpoint, map_location="cpu")
    checkpoint["model_state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
    checkpoint["closed_loop_grpo"] = {
        "parent_sha256": hashlib.sha256(opts.checkpoint.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(opts.reference.read_bytes()).hexdigest(),
        "run_dir": str(opts.run_dir),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(opts).items()},
        "decisions": len(rows),
        "planner_grad_norm": grad_norm,
    }
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, opts.output)
    summary = {
        "decisions": len(rows),
        "mean_plan_reward": float(np.mean([r["mean_plan_reward"] for r in rows])),
        "group_reward_std": float(np.mean([r["group_reward_std"] for r in rows])),
        "kl": float(np.mean([r["kl"] for r in rows])),
        "planner_grad_norm": grad_norm,
        "max_replay_error_m": float(max(r["replay_error_m"] for r in rows)),
    }
    (opts.output.with_suffix(".json")).write_text(
        json.dumps({"summary": summary, "decisions": rows}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
