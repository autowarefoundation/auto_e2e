"""Score an AlpaSim rollout with the phase-1 reward, one row per AutoE2E decision.

Reads the rollout log (``rollout.asl``) that AlpaSim writes for every rollout.
From it, it takes each driver response (the plan), the recorded ego trajectory
and the ego poses. Run it where ``alpasim_utils`` is installed, e.g. inside the
AlpaSim image:

    python Model/plugins/alpasim_driver/tools/score_rollout.py \
        <log_dir>/rollouts/<scene>/<rollout>/rollout.asl [--json scores.json]
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from alpasim_utils.geometry import Trajectory
from alpasim_utils.logs import async_read_pb_log

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpasim_autoe2e.rewards import GroundTruthDeviationReward  # noqa: E402


def to_trajectory(poses) -> Trajectory:
    poses = sorted(poses, key=lambda pose: pose.timestamp_us)
    quaternions = np.array(
        [[p.pose.quat.x, p.pose.quat.y, p.pose.quat.z, p.pose.quat.w] for p in poses]
    )
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return Trajectory(
        np.array([p.timestamp_us for p in poses], dtype=np.uint64),
        np.array([[p.pose.vec.x, p.pose.vec.y, p.pose.vec.z] for p in poses], dtype=np.float32),
        quaternions.astype(np.float32),
    )


def se3(trajectory: Trajectory, timestamp_us: int) -> np.ndarray:
    return np.asarray(trajectory.interpolate_pose(int(timestamp_us)).as_se3(), dtype=np.float64)


async def read_rollout(path: str):
    recorded, plans = None, []
    async for message in async_read_pb_log(path):
        kind = message.WhichOneof("log_entry")
        if kind == "rollout_metadata":
            recorded = to_trajectory(
                message.rollout_metadata.ego_rig_recorded_ground_truth_trajectory.poses
            )
        elif kind == "driver_return":
            plans.append(list(message.driver_return.trajectory.poses))
    if recorded is None:
        raise ValueError(f"{path} has no rollout metadata with the recorded trajectory")
    return recorded, plans


def score(path: str, reward: GroundTruthDeviationReward) -> list[dict]:
    recorded, plans = asyncio.run(read_rollout(path))
    recorded_range = recorded.time_range_us
    rows = []
    for index, poses in enumerate(plans, start=1):
        # The driver anchors each plan at t0 with the ego pose it planned from.
        anchor, waypoints = poses[0], poses[1:]
        t0_us = int(anchor.timestamp_us)
        plan = to_trajectory(poses)
        rig_from_local = np.linalg.inv(se3(plan, t0_us))
        plan_xy, recorded_xy = [], []
        for waypoint in waypoints:
            t_us = int(waypoint.timestamp_us)
            plan_xy.append((rig_from_local @ se3(plan, t_us))[:2, 3])
            if t_us in recorded_range:
                recorded_xy.append((rig_from_local @ se3(recorded, t_us))[:2, 3])
        ego_deviation = float(np.linalg.norm((rig_from_local @ se3(recorded, t0_us))[:2, 3]))
        result = reward.compute(np.array(plan_xy), np.array(recorded_xy), ego_deviation)
        rows.append({"decision": index, "t0_us": t0_us, **asdict(result)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_asl")
    parser.add_argument("--json", type=Path, help="also write the rows to this file")
    args = parser.parse_args()
    rows = score(args.rollout_asl, GroundTruthDeviationReward())
    for row in rows:
        print(
            f"decision {row['decision']}: reward {row['reward']:7.3f}  "
            f"ADE {row['ade_m']:.2f} m  FDE {row['fde_m']:.2f} m  "
            f"ego off recording {row['ego_deviation_m']:.3f} m  truncated {row['truncated']}"
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
