"""Tests for the pose-grounded target rollout reconstruction audit."""

import numpy as np
import pytest

from evaluation.reconstruction_audit import (
    AUDIT_SCHEMA_VERSION,
    audit_target_rollout_reconstruction,
)


def _fixture():
    batch_size = 3
    timesteps = 64
    controls = np.zeros((batch_size, timesteps, 2), dtype=np.float32)
    speeds = np.asarray([2.0, 4.0, 6.0], dtype=np.float32)
    time = np.arange(1, timesteps + 1, dtype=np.float64) * 0.1
    logged = np.zeros_like(controls, dtype=np.float64)
    logged[:, :, 0] = speeds[:, None] * time[None, :]
    return (
        controls,
        logged,
        speeds,
        ["sample-c", "sample-a", "sample-b"],
        ["scene-2", "scene-1", "scene-1"],
    )


def test_reconstruction_audit_accepts_exact_straight_rollout():
    report = audit_target_rollout_reconstruction(*_fixture())

    assert report["schema_version"] == AUDIT_SCHEMA_VERSION
    assert report["go"] is True
    assert report["sample_count"] == 3
    assert report["scene_count"] == 2
    assert report["metrics"]["fde_full_m"]["natural"]["p95"] < 2e-5
    assert [scene["split_group_uid"] for scene in report["scenes"]] == [
        "scene-1",
        "scene-2",
    ]


def test_reconstruction_audit_rejects_large_pose_error():
    controls, logged, speeds, sample_uids, group_uids = _fixture()
    logged[:, :, 1] = 3.0

    report = audit_target_rollout_reconstruction(
        controls,
        logged,
        speeds,
        sample_uids,
        group_uids,
    )

    assert report["go"] is False
    assert report["metrics"]["fde_3s_m"]["natural"]["p95"] == pytest.approx(3.0)
    assert report["metrics"]["fde_full_m"]["natural"]["p95"] == pytest.approx(3.0)


def test_reconstruction_audit_identity_is_order_independent():
    fixture = _fixture()
    first = audit_target_rollout_reconstruction(*fixture)
    order = [2, 0, 1]
    reordered = (
        fixture[0][order],
        fixture[1][order],
        fixture[2][order],
        [fixture[3][index] for index in order],
        [fixture[4][index] for index in order],
    )
    second = audit_target_rollout_reconstruction(*reordered)

    assert first["sample_uid_digest"] == second["sample_uid_digest"]
    assert first["split_group_uid_digest"] == second["split_group_uid_digest"]
    assert first["metrics"] == second["metrics"]


def test_reconstruction_audit_rejects_duplicate_sample_uid():
    controls, logged, speeds, _, group_uids = _fixture()
    with pytest.raises(ValueError, match="non-empty and unique"):
        audit_target_rollout_reconstruction(
            controls,
            logged,
            speeds,
            ["duplicate", "duplicate", "sample"],
            group_uids,
        )
