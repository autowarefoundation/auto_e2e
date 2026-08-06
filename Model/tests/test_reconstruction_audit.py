"""Tests for the pose-grounded target rollout reconstruction audit."""

import io
import json
import tarfile

import numpy as np
import pytest

from data_processing.geospatial import encode_gps_future, encode_pose
from evaluation.reconstruction_audit import (
    AUDIT_SCHEMA_VERSION,
    audit_packed_target_rollout_reconstruction,
    audit_target_rollout_reconstruction,
    load_packed_reconstruction_inputs,
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
    assert report["thresholds_pass"] is True
    assert report["decision"]["status"] == "pending_review"
    assert report["sample_count"] == 3
    assert report["scene_count"] == 2
    assert len(report["error_by_step"]) == 64
    assert report["metrics"]["fde_full_m"]["natural"]["p95"] < 2e-5
    heading = report["heading_alignment"]
    assert heading["valid_step_count"] == 3 * 64
    assert heading["valid_sample_count"] == 3
    assert heading["full_horizon"]["p95"] < 1e-5
    assert heading["at_3s"]["p95"] < 1e-5
    assert heading["at_full_horizon"]["p95"] < 1e-5
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

    assert report["thresholds_pass"] is False
    assert (
        report["decision"]["automatic_recommendation"]
        == "review_required"
    )
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


def _add_tar_member(
    archive: tarfile.TarFile,
    name: str,
    payload: bytes,
) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _write_packed_sample(
    archive: tarfile.TarFile,
    *,
    sample_uid: str,
    group_uid: str,
    speed_mps: float,
    include_gps: bool = True,
    include_metadata: bool = True,
) -> None:
    history = np.zeros((64, 4), dtype="<f4")
    history[-1, 0] = speed_mps
    controls = np.zeros((64, 2), dtype="<f4")
    controls[:, 0] = 0.25
    ego = np.concatenate([history.ravel(), controls.ravel()]).tobytes()
    latitude = 49.0 + speed_mps * 1e-5
    longitude = 8.4
    gps = np.tile(
        np.asarray([[latitude, longitude]], dtype=np.float64),
        (65, 1),
    )
    metadata = json.dumps(
        {
            "sample_uid": sample_uid,
            "split_group_uid": group_uid,
        }
    ).encode("ascii")
    pose = encode_pose(
        {
            "latitude_deg": latitude,
            "longitude_deg": longitude,
            "heading_deg_cw_from_north": 90.0,
            "timestamp_ns": 123,
            "gps_accuracy_m": 0.5,
        }
    )
    members = {
        "cam_0.jpg": b"not-decoded",
        "ego.npy": ego,
        "pose.npy": pose,
    }
    if include_metadata:
        members["meta.json"] = metadata
    if include_gps:
        members["gps.npy"] = encode_gps_future(gps)
    for suffix, payload in members.items():
        _add_tar_member(archive, f"{sample_uid}.{suffix}", payload)


def test_packed_loader_reads_only_selected_validation_groups(
    tmp_path,
    monkeypatch,
):
    tar_path = tmp_path / "samples.tar"
    with tarfile.open(tar_path, "w") as archive:
        _write_packed_sample(
            archive,
            sample_uid="sample-z",
            group_uid="scene-train",
            speed_mps=9.0,
        )
        _write_packed_sample(
            archive,
            sample_uid="sample-b",
            group_uid="scene-val",
            speed_mps=4.0,
        )
        _write_packed_sample(
            archive,
            sample_uid="sample-a",
            group_uid="scene-val",
            speed_mps=2.0,
        )

    inputs = load_packed_reconstruction_inputs(
        [tmp_path],
        validation_group_uids=["scene-val"],
    )

    assert inputs.sample_uids == ("sample-a", "sample-b")
    assert inputs.split_group_uids == ("scene-val", "scene-val")
    np.testing.assert_array_equal(
        inputs.initial_speeds_mps,
        np.asarray([2.0, 4.0], dtype=np.float32),
    )
    assert inputs.target_controls.shape == (2, 64, 2)
    assert inputs.logged_gps.shape == (2, 65, 2)
    assert inputs.current_poses.shape == (2, 3)

    def fake_wgs84_to_ego_xy(gps, poses):
        assert gps.shape == (2, 65, 2)
        assert poses.shape == (2, 3)
        return np.zeros((2, 64, 2), dtype=np.float64)

    monkeypatch.setattr(
        "evaluation.kitscenes_benchmark.wgs84_trajectory_to_ego_xy",
        fake_wgs84_to_ego_xy,
    )
    report = audit_packed_target_rollout_reconstruction(inputs)
    assert report["sample_uid_digest"]
    assert report["sample_count"] == 2
    assert report["heading_alignment"]["valid_step_count"] == 0
    assert report["heading_alignment"]["full_horizon"] is None


def test_packed_loader_rejects_missing_selected_member(tmp_path):
    tar_path = tmp_path / "samples.tar"
    with tarfile.open(tar_path, "w") as archive:
        _write_packed_sample(
            archive,
            sample_uid="sample-a",
            group_uid="scene-val",
            speed_mps=2.0,
            include_gps=False,
        )

    with pytest.raises(ValueError, match="audit members"):
        load_packed_reconstruction_inputs(
            [tmp_path],
            validation_group_uids=["scene-val"],
        )


def test_packed_loader_rejects_missing_metadata(tmp_path):
    tar_path = tmp_path / "samples.tar"
    with tarfile.open(tar_path, "w") as archive:
        _write_packed_sample(
            archive,
            sample_uid="sample-a",
            group_uid="scene-val",
            speed_mps=2.0,
            include_metadata=False,
        )

    with pytest.raises(ValueError, match="missing audit metadata"):
        load_packed_reconstruction_inputs([tmp_path])
