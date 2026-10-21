"""Tests for the phase-1 closed-loop reward and the rollout scorer.

The reward is the negative ADE/FDE between AutoE2E's plan and the recorded
drive. Its renderer validity bound acts on the ego's own distance from the
recording, independently of the tracking weight.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins" / "alpasim_driver"
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from alpasim_autoe2e.rewards import GroundTruthDeviationReward  # noqa: E402

STEPS = np.arange(1, 65)


def _straight(speed_mps: float, lateral_m: float = 0.0) -> np.ndarray:
    return np.stack([speed_mps * 0.1 * STEPS, np.full(64, lateral_m)], axis=1)


def test_matching_the_recording_scores_zero() -> None:
    result = GroundTruthDeviationReward().compute(_straight(8.0), _straight(8.0), 0.0)
    assert result.reward == 0.0
    assert result.ade_m == 0.0 and result.fde_m == 0.0
    assert not result.truncated


def test_constant_lateral_offset() -> None:
    result = GroundTruthDeviationReward().compute(_straight(8.0, 0.5), _straight(8.0), 0.0)
    assert result.ade_m == pytest.approx(0.5)
    assert result.fde_m == pytest.approx(0.5)
    assert result.reward == pytest.approx(-(1.0 * 0.5 + 0.5 * 0.5))


def test_weights_scale_the_tracking_term() -> None:
    reward = GroundTruthDeviationReward(ade_weight=2.0, fde_weight=1.0, tracking_weight=0.5)
    result = reward.compute(_straight(8.0, 1.0), _straight(8.0), 0.0)
    assert result.tracking == pytest.approx(-0.5 * (2.0 * 1.0 + 1.0 * 1.0))


def test_slower_plan_grows_linearly_with_the_horizon() -> None:
    # 8 m/s against a recorded 10 m/s: the gap is 0.2 m per step.
    result = GroundTruthDeviationReward().compute(_straight(8.0), _straight(10.0), 0.0)
    assert result.ade_m == pytest.approx(0.2 * STEPS.mean())
    assert result.fde_m == pytest.approx(0.2 * 64)


def test_only_steps_with_a_recorded_position_count() -> None:
    plan, recorded = _straight(8.0), _straight(8.0)
    plan[30:, 1] = 5.0  # diverges only where the recording has ended
    result = GroundTruthDeviationReward().compute(plan, recorded[:30], 0.0)
    assert result.ade_m == 0.0 and result.fde_m == 0.0


def test_ego_leaving_the_valid_region_truncates() -> None:
    result = GroundTruthDeviationReward().compute(_straight(8.0), _straight(8.0), 3.5)
    assert result.truncated
    assert result.boundary == -10.0
    assert result.reward == -10.0


def test_boundary_is_strictly_beyond_the_threshold() -> None:
    result = GroundTruthDeviationReward().compute(_straight(8.0), _straight(8.0), 3.0)
    assert not result.truncated and result.boundary == 0.0


def test_a_diverging_plan_alone_does_not_truncate() -> None:
    # The decision-3 case of the first AlpaSim rollout: the plan brakes and
    # ends 13 m short of the recording while the ego is 2 cm off it.
    result = GroundTruthDeviationReward().compute(_straight(1.6), _straight(3.6), 0.02)
    assert result.fde_m == pytest.approx(12.8)
    assert not result.truncated


def test_boundary_does_not_share_the_tracking_weight() -> None:
    reward = GroundTruthDeviationReward(tracking_weight=0.1, terminal_penalty=10.0)
    result = reward.compute(_straight(8.0, 1.0), _straight(8.0), 4.0)
    assert result.boundary == -10.0
    assert result.tracking == pytest.approx(-0.1 * 1.5)


def test_a_stopped_ego_is_penalised_and_eventually_truncated() -> None:
    # Standing still while the recording drives on: tracking grows with the
    # recorded speed, and the ego soon falls more than 3 m behind.
    result = GroundTruthDeviationReward().compute(np.zeros((64, 2)), _straight(5.0), 3.2)
    assert result.ade_m == pytest.approx(0.5 * STEPS.mean())
    assert result.truncated


def test_torch_inputs_match_numpy() -> None:
    plan = torch.tensor(_straight(8.0, 0.3), requires_grad=True)
    from_torch = GroundTruthDeviationReward().compute(plan, torch.tensor(_straight(8.0)), 0.0)
    from_numpy = GroundTruthDeviationReward().compute(_straight(8.0, 0.3), _straight(8.0), 0.0)
    assert from_torch.reward == pytest.approx(from_numpy.reward)


def test_invalid_inputs_are_rejected() -> None:
    reward = GroundTruthDeviationReward()
    with pytest.raises(ValueError, match="overlap"):
        reward.compute(_straight(8.0), np.zeros((0, 2)), 0.0)
    with pytest.raises(ValueError, match="positions"):
        reward.compute(np.zeros(64), _straight(8.0), 0.0)
    with pytest.raises(ValueError):
        GroundTruthDeviationReward(max_deviation_m=0.0)


def test_score_rollout_reads_an_alpasim_log(tmp_path: Path) -> None:
    """Write a rollout.asl with AlpaSim's LogWriter and score it end to end."""
    pytest.importorskip("alpasim_utils")
    pytest.importorskip("aiofiles")
    from alpasim_grpc.v0 import common_pb2, egodriver_pb2, logging_pb2
    from alpasim_utils.logs import LogWriter

    sys.path.insert(0, str(_PLUGIN_DIR / "tools"))
    from score_rollout import score

    def pose(t_us: int, x: float, y: float = 0.0) -> common_pb2.PoseAtTime:
        message = common_pb2.PoseAtTime(timestamp_us=t_us)
        message.pose.vec.x, message.pose.vec.y = x, y
        message.pose.quat.w = 1.0
        return message

    t0 = 10_000_000
    metadata = logging_pb2.LogEntry()
    for k in range(0, 201):  # recorded: 10 m/s along x for 20 s
        metadata.rollout_metadata.ego_rig_recorded_ground_truth_trajectory.poses.append(
            pose(k * 100_000, 10.0 * k * 0.1)
        )
    on_path = logging_pb2.LogEntry(driver_return=egodriver_pb2.DriveResponse())
    off_path = logging_pb2.LogEntry(driver_return=egodriver_pb2.DriveResponse())
    for k in range(0, 65):  # plans at 8 m/s; the second one from 4 m to the left
        on_path.driver_return.trajectory.poses.append(pose(t0 + k * 100_000, 100.0 + 0.8 * k))
        off_path.driver_return.trajectory.poses.append(
            pose(t0 + k * 100_000, 100.0 + 0.8 * k, 4.0)
        )

    log = tmp_path / "rollout.asl"

    async def write() -> None:
        async with LogWriter(str(log)) as writer:
            for message in (metadata, on_path, off_path):
                await writer.on_message(message)

    asyncio.run(write())
    rows = score(str(log), GroundTruthDeviationReward())

    assert [row["decision"] for row in rows] == [1, 2]
    assert rows[0]["ade_m"] == pytest.approx(0.2 * STEPS.mean(), abs=1e-3)
    assert rows[0]["fde_m"] == pytest.approx(12.8, abs=1e-3)
    assert rows[0]["reward"] == pytest.approx(-(6.5 + 0.5 * 12.8), abs=1e-3)
    assert not rows[0]["truncated"]
    assert rows[1]["ego_deviation_m"] == pytest.approx(4.0, abs=1e-3)
    assert rows[1]["truncated"] and rows[1]["boundary"] == -10.0
