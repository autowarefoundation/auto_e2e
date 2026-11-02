# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the UniAD adapter; never download weights/data or import MMCV."""

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import struct
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accuracy_benchmark.uniad import (
    audit_results,
    common,
    geometry,
    inputs,
    metrics,
    prepare_kitscenes,
    protocol,
    shared_metrics,
    eval_kitscenes,
)


@pytest.fixture
def synthetic_scene(tmp_path):
    n = 100
    poses = np.zeros((n, 8))
    poses[:, 0] = np.arange(n) / 10 + 1000
    poses[:, 1] = np.arange(n) * 0.4
    poses[:, 2] = 0.002 * np.arange(n) ** 2
    poses[:, 4:8] = Rotation.from_euler("ZYX", [-0.2, 0.03, -0.01]).as_quat()
    payload = {"reference": "base_frame"}
    for i, camera in enumerate(protocol.CAMERAS):
        extrinsic = np.eye(4)
        extrinsic[:3, 3] = [i * 0.1, i * 0.2, 1.0]
        payload[camera] = dict(
            camera_model="pinhole",
            intrinsics=dict(
                focal_length=3.0, principal_point_u=2.0, principal_point_v=1.0
            ),
            T_to_reference=extrinsic.tolist(),
            resolution=dict(width=4, height=2 + i % 2),
        )
    raw = tmp_path / "raw"
    raw.mkdir()
    for camera in protocol.CAMERAS:
        (raw / camera).mkdir()
        for frame in range(n):
            Image.new(
                "RGB", (4, payload[camera]["resolution"]["height"]), (10, 20, frame)
            ).save(raw / camera / f"{frame:010d}.jpg")
    sid = "d7b1c9c6-e178-e094-81ab-00aee41b6524"
    transform = np.eye(4)
    transform[:3, :3] = protocol.UNIAD_TO_KIT
    projections = []
    for camera in protocol.CAMERAS:
        c = geometry.camera_calibration(payload, camera)
        projection = geometry.raw_projection(
            c, c.image_size, (512, 512)
        ) @ np.linalg.inv(transform)
        projections.append(projection[:3])
    packed = np.asarray(projections)[[0, 2, 1, 4, 3, 5]]
    anchors = {}
    for index in range(40, 46):
        pose = dict(
            latitude=49.0,
            longitude=8.4,
            heading_deg=90 - np.degrees(-0.2),
            timestamp_ns=round(poses[index, 0] * 1e9),
        )
        target = geometry.world_delta_to_published(
            poses[index + 1 : index + 31, 1:3] - poses[index, 1:3], pose
        )
        anchors[index] = dict(
            sample_uid=f"kitscenes-v1-{sid}-f{index:06d}",
            evaluation_pose=pose,
            projection_512=packed.tolist(),
            target_xy_10hz=target.tolist(),
        )
    return sid, poses, payload, raw, anchors


def test_native_image_extents_and_bgr_padding():
    rgb = np.full((6, 3, 5, 3), [10, 20, 30], dtype=np.uint8)
    shapes = np.array([[2, 4]] * 5 + [[3, 5]])
    padded, original = protocol.normalize_and_pad(rgb, shapes)
    assert padded.shape == (6, 32, 32, 3)
    np.testing.assert_array_equal(
        padded[0, 0, 0],
        np.array([30, 20, 10], np.float32)
        - np.array([103.53, 116.28, 123.675], np.float32),
    )
    assert not padded[0, 2:].any() and not padded[0, :, 4:].any()
    assert original[0] == (2, 4, 3) and original[5] == (3, 5, 3)


def test_temporal_phase_coverage_and_keyframe_reset(synthetic_scene):
    sid, poses, payload, raw, anchors = synthetic_scene
    tracks = prepare_kitscenes.scene_streams(sid, anchors, poses, raw, payload)
    assert common.coverage(tracks["dense"]) == dict(
        samples=6, scenes=1, streams=5, model_frames=41
    )
    assert common.coverage(tracks["keyframes"]) == dict(
        samples=1, scenes=1, streams=1, model_frames=9
    )
    assert tracks["dense"][0]["frames"][0]["frame"] == 5
    assert tracks["keyframes"][0]["frames"][0]["frame"] == 0
    assert [f["frame"] for f in tracks["keyframes"][0]["frames"] if f["scored"]] == [40]
    scene = dict(scene_id=sid, tracks=tracks)
    assert audit_results.audit_inputs(scene, poses, "dense", full=True) < 1e-6
    assert audit_results.audit_inputs(scene, poses, "keyframes", full=True) < 1e-6
    for stream in tracks["dense"]:
        np.testing.assert_array_equal(stream["frames"][0]["can_bus"][:3], np.zeros(3))
        assert 0 <= stream["frames"][0]["can_bus"][-2] < 2 * np.pi
    bad = copy.deepcopy(scene)
    bad["tracks"]["dense"][0]["frames"][0]["can_bus"][-1] -= 360
    with pytest.raises(ValueError, match="audit failed"):
        audit_results.audit_inputs(bad, poses, "dense")


def test_loader_checksums_and_dimensions(synthetic_scene):
    sid, poses, payload, raw, anchors = synthetic_scene
    frame = prepare_kitscenes.scene_streams(
        sid, anchors, poses, raw, payload, smoke=True
    )["dense"][0]["frames"][0]
    arrays = inputs.load_raw_kitscenes_frame(frame)
    assert arrays["images_rgb"].shape == (6, 3, 4, 3)
    assert not arrays["images_rgb"][0, 2].any()
    frame["raw_kitscenes_images"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        inputs.load_raw_kitscenes_frame(frame)


def test_canonical_yaw_target_and_body_mapping(synthetic_scene):
    _, poses, _, _, _ = synthetic_scene
    target, transform, velocity, world_velocity = metrics.canonical_reference(poses, 40)
    theta = -0.2
    expected_rotation = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    np.testing.assert_allclose(
        target, (poses[41:71, 1:3] - poses[40, 1:3]) @ expected_rotation.T
    )
    np.testing.assert_allclose(
        transform,
        expected_rotation @ Rotation.from_quat(poses[40, 4:8]).as_matrix()[:2, :2],
        atol=1e-15,
    )
    np.testing.assert_allclose(velocity, world_velocity @ expected_rotation.T)
    native = np.tile([1.0, 2.0], (6, 1))
    np.testing.assert_array_equal(
        protocol.to_kit_xy(native), np.tile([2.0, -1.0], (6, 1))
    )


def test_interpolation_is_not_future_extrapolation():
    native = protocol.NATIVE_TIMES[:, None] * [2.0, -1.0]
    np.testing.assert_allclose(
        protocol.dense_prediction(native), protocol.DENSE_TIMES[:, None] * [2.0, -1.0]
    )
    with pytest.raises(ValueError):
        protocol.dense_prediction(np.zeros((10, 2)))
    scored = metrics.independent_metrics(
        np.tile([[[3.0, 4.0]]], (2, 30, 1)), np.zeros((2, 30, 2))
    )
    assert set(scored) == {"1", "2", "3"}
    assert all(x == dict(ade_m=5.0, fde_m=5.0, count=2) for x in scored.values())


@pytest.mark.parametrize(
    "actual,expected",
    [(["a", "a"], ["a", "b"]), (["a"], ["a", "b"]), (["a", "x"], ["a", "b"])],
)
def test_coverage_rejects_duplicate_missing_and_unexpected(actual, expected):
    with pytest.raises(ValueError):
        common.membership(actual, expected)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/outside", "a\\b"])
def test_untrusted_archive_paths(tmp_path, name):
    with pytest.raises(ValueError):
        common.safe_child(tmp_path, name)


def test_extract_rejects_symlink_and_preserves_existing(tmp_path):
    archive = tmp_path / "scene.tar"
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo("poses.txt")
        member.type = tarfile.SYMTYPE
        member.linkname = "../other"
        tar.addfile(member)
    with pytest.raises(ValueError, match="archive member"):
        prepare_kitscenes.extract_selected(archive, tmp_path / "out", "scene", [], True)
    with tarfile.open(archive, "w") as tar:
        for name in prepare_kitscenes.METADATA:
            member = tarfile.TarInfo(name)
            member.size = 3
            tar.addfile(member, io.BytesIO(b"new"))
    (tmp_path / "out").mkdir()
    (tmp_path / "out/poses.txt").write_bytes(b"old")
    with pytest.raises(ValueError, match="refusing overwrite"):
        prepare_kitscenes.extract_selected(archive, tmp_path / "out", "scene", [], True)
    assert (tmp_path / "out/poses.txt").read_bytes() == b"old"


def test_extract_skips_archive_root_entry(tmp_path):
    archive = tmp_path / "scene.tar"
    with tarfile.open(archive, "w") as tar:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        tar.addfile(root)
        for name in prepare_kitscenes.METADATA:
            member = tarfile.TarInfo("./" + name)
            member.size = 3
            tar.addfile(member, io.BytesIO(b"abc"))
    out = tmp_path / "out"
    out.mkdir()
    prepare_kitscenes.extract_selected(archive, out, "scene", [], True)
    assert (out / "poses.txt").read_bytes() == b"abc"


def test_swapped_mirror_camera_slots_are_rejected(synthetic_scene):
    sid, poses, payload, raw, anchors = synthetic_scene
    for anchor in anchors.values():
        packed = anchor["projection_512"]
        packed[1], packed[2] = packed[2], packed[1]
    with pytest.raises(ValueError, match="camera slots"):
        prepare_kitscenes.scene_streams(sid, anchors, poses, raw, payload)


def test_smoke_extracts_only_smoke_frames(synthetic_scene):
    sid, poses, payload, raw, anchors = synthetic_scene
    smoke = prepare_kitscenes.needed_frames(
        *prepare_kitscenes.schedule(anchors, len(poses), smoke=True)
    )
    full = prepare_kitscenes.needed_frames(
        *prepare_kitscenes.schedule(anchors, len(poses))
    )
    assert smoke < full
    tracks = prepare_kitscenes.scene_streams(sid, anchors, poses, raw, payload, True)
    used = {f["frame"] for t in tracks.values() for s in t for f in s["frames"]}
    assert used == smoke


def test_can_bus_position_is_relative_to_stream_start(synthetic_scene):
    _, poses, *_ = synthetic_scene
    can = protocol.past_can_bus(poses, 50, 40)
    np.testing.assert_allclose(can[:3], poses[50, 1:4] - poses[40, 1:4])


def test_shared_metric_source_hash_is_mandatory(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("raise RuntimeError('must never execute')")
    with pytest.raises(ValueError, match="checksum"):
        shared_metrics.load_metric_code(source)


def test_reference_comparison_preserves_failed_gate(tmp_path):
    row = dict(
        sample_uid="anchor",
        frame=40,
        prediction_frame="body_FLU",
        predictions_body_xy_2hz={"fixed_forward": np.zeros((6, 2)).tolist()},
    )
    reference = tmp_path / "reference.jsonl"
    actual = tmp_path / "actual.jsonl"
    reference.write_text(json.dumps(row) + "\n")
    row["predictions_body_xy_2hz"]["fixed_forward"][0][0] = 0.051
    actual.write_text(json.dumps(row) + "\n")
    report = audit_results.compare_samples(actual, reference)
    assert report["status"] == "failed" and report["tolerance_m"] == 0.05


def test_constant_velocity_uses_matching_canonical_and_legacy_targets(synthetic_scene):
    sid, poses, payload, raw, anchors = synthetic_scene
    frame = [
        f
        for f in prepare_kitscenes.scene_streams(sid, anchors, poses, raw, payload)[
            "dense"
        ][0]["frames"]
        if f["scored"]
    ][0]
    row = dict(predictions_body_xy_2hz={"fixed_forward": np.zeros((6, 2)).tolist()})
    prediction, target = metrics.row_trajectories(row, frame, poses, "dense")
    assert set(target) == {"canonical", "legacy_wgs84"}
    np.testing.assert_array_equal(
        target["legacy_wgs84"], anchors[frame["frame"]]["target_xy_10hz"]
    )
    assert prediction["canonical"]["constant_velocity"].shape == (30, 2)
    assert not np.array_equal(target["canonical"], target["legacy_wgs84"])


def test_manifest_selection_excludes_overlap():
    parts = [
        dict(
            partition_id=f"part-{i:016x}",
            source_uri=f"s3://x/kitscenes-v35-val-a/{i}",
            sample_count=94 + (i < 37),
            shard_count=1,
        )
        for i in range(117)
    ]
    parts.append(
        dict(
            partition_id="part-ffffffffffffffff",
            source_uri="s3://x/kitscenes-v35-overlap-a/1",
            sample_count=2490,
            shard_count=1,
        )
    )
    shards = [
        dict(
            name=f"{p['partition_id']}-train-000000.tar",
            key=f"kitscenes-val/v3.5/shards/{p['partition_id']}-train-000000.tar",
        )
        for p in parts
    ]
    release = dict(
        dataset="kitscenes-val",
        version="v3.5",
        status="ready",
        partitions=parts,
        shard_entries=shards,
    )
    selected = prepare_kitscenes.select_shards(release)
    assert len(selected) == 117 and all("ffffffffffffffff" not in x for x in selected)


def test_raw_shard_parser_rejects_duplicate_members(tmp_path):
    archive = tmp_path / "shard.tar"
    with tarfile.open(archive, "w") as tar:
        for _ in range(2):
            member = tarfile.TarInfo("sample.meta.json")
            member.size = 2
            tar.addfile(member, io.BytesIO(b"{}"))
    with pytest.raises(ValueError, match="mirror member"):
        prepare_kitscenes.read_shard(archive)


def test_read_real_format_mirror_metadata(synthetic_scene, tmp_path):
    sid, poses, _, _, anchors = synthetic_scene
    uid = anchors[40]["sample_uid"]
    pose = anchors[40]["evaluation_pose"]
    gps = np.tile([pose["latitude"], pose["longitude"]], (65, 1))
    gps[:, 0] += np.arange(65) * 1e-6
    target = geometry.wgs84_xy(gps[1:], pose)
    trajectory = io.BytesIO()
    np.savez(
        trajectory, trajectory_xy_m=target, trajectory_valid=np.ones(64, dtype=bool)
    )
    calib = dict(
        temporal_frame_offsets=list(range(-7, 1)),
        temporal_frame_interval_us=500000,
        projection=dict(matrix=anchors[40]["projection_512"]),
    )
    history = [
        [f"kitscenes-v1-{sid}-r{40 + d:06d}-c{c}" for c in range(6)]
        for d in range(-35, 0, 5)
    ]
    blobs = {
        ".meta.json": json.dumps(
            dict(sample_uid=uid, frame_idx=40, split_group_uid="kitscenes-" + sid)
        ).encode(),
        ".calib.json": json.dumps(calib).encode(),
        ".bev_history_index.json": json.dumps(history).encode(),
        ".trajectory_xy.npz": trajectory.getvalue(),
        ".pose.npy": struct.pack(
            "<dddqf",
            pose["latitude"],
            pose["longitude"],
            pose["heading_deg"],
            round(poses[40, 0] * 1e9),
            0.0,
        ),
        ".gps.npy": gps.astype("<f8").tobytes(),
    }
    archive = tmp_path / "valid_mirror.tar"
    with tarfile.open(archive, "w") as tar:
        for suffix, blob in blobs.items():
            member = tarfile.TarInfo(uid + suffix)
            member.size = len(blob)
            tar.addfile(member, io.BytesIO(blob))
    actual_sid, decoded = prepare_kitscenes.read_shard(archive)
    assert actual_sid == sid and list(decoded) == [40]
    np.testing.assert_array_equal(decoded[40]["target_xy_10hz"], target[:30])


def test_gpu_unavailable_preserves_failure_report(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = SimpleNamespace(
        out=tmp_path / "failed_run",
        scene="test",
        track="dense",
        oracle_diagnostic=False,
    )
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        eval_kitscenes.run_worker(args)
    report = common.read_json(args.out / "run.json")
    assert report["status"] == "failed" and "requires a CUDA GPU" in report["error"]
    assert report["adapter_source_sha256"] == common.adapter_hashes()


def test_native_frame_uses_no_future_fields(monkeypatch):
    """The model-input boundary cannot read targets or oracle commands."""
    import torch

    arrays = dict(
        images_rgb=np.zeros((6, 2, 4, 3), dtype=np.uint8),
        image_shapes=np.tile([2, 4], (6, 1)),
        can_bus=np.zeros(18),
        lidar2img=np.tile(np.eye(4), (6, 1, 1)),
        l2g_r_mat=np.eye(3),
        l2g_t=np.zeros(3),
    )
    monkeypatch.setattr(inputs, "load_raw_kitscenes_frame", lambda frame: arrays)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)

    def cpu_constructor(original):
        def construct(*args, **kwargs):
            kwargs.pop("device", None)
            return original(*args, **kwargs)

        return construct

    for name in ("tensor", "as_tensor", "zeros", "ones"):
        monkeypatch.setattr(torch, name, cpu_constructor(getattr(torch, name)))
    result = inputs.input_kwargs(dict(frame=40, timestamp=1004.0), "stream", object)
    assert result["command"].tolist() == [2]
    assert result["img"][0].shape == (1, 6, 3, 32, 32)
    assert not result["gt_segmentation"][0].any()
    assert result["img_metas"][0][0]["img_shape"] == [(32, 32, 3)] * 6


def test_reference_result_scope_and_counts():
    reference = common.read_json(
        Path(common.__file__).with_name("reference_results.json")
    )
    assert (
        reference["coverage"]["dense"]["sample_uid_sha256"] == common.DENSE_UID_SHA256
    )
    assert reference["coverage"]["keyframes"]["samples"] == 202
    assert reference["packaged_gpu_replay_verified"] is False
    assert reference["metrics"]["horizons_4_5"] == "N/A"


@pytest.fixture
def metric_source(tmp_path, monkeypatch):
    # The numerical statements from the pinned evaluator, in a minimal synthetic
    # module. Its top-level exception must never execute during AST extraction.
    source = """raise RuntimeError("whole module must not execute")
def evaluate_reactive_multitask():
    displacement = predicted_xy - target_xy
    errors = torch.linalg.vector_norm(displacement, dim=-1)
    for name, step_count in horizon_steps.items():
        usable = min(step_count, errors.shape[1])
        horizon_valid = valid[:, :usable]
        per_sample_count = horizon_valid.sum(dim=1)
        eligible_ade = per_sample_count > 0
        if bool(eligible_ade.any()):
            per_sample_error = ((errors[:, :usable] * horizon_valid).sum(dim=1)
                                / per_sample_count.clamp_min(1))
            trajectory_ade_sum[name] += float(per_sample_error[eligible_ade].sum().item())
            trajectory_ade_count[name] += int(eligible_ade.sum().item())
        if step_count <= errors.shape[1]:
            eligible_fde = valid[:, step_count - 1]
            if bool(eligible_fde.any()):
                trajectory_fde_sum[name] += float(errors[eligible_fde, step_count-1].sum().item())
                trajectory_fde_count[name] += int(eligible_fde.sum().item())
"""
    path = tmp_path / "metric_fixture.py"
    path.write_bytes(source.encode())
    monkeypatch.setattr(
        shared_metrics, "SOURCE_SHA256", hashlib.sha256(source.encode()).hexdigest()
    )
    return path


def test_pinned_statement_extraction_and_mask_counts(metric_source):
    code = shared_metrics.load_metric_code(metric_source)
    shared_metrics.self_test(code)
    prediction = np.tile([[[3.0, 4.0]]], (2, 30, 1))
    mask = np.ones((2, 30), dtype=bool)
    mask[1, 20:] = False
    values, counts = shared_metrics.evaluate(
        code, prediction, np.zeros_like(prediction), mask
    )
    assert values["3"] == dict(ade_m=5.0, fde_m=5.0)
    assert counts["3"] == dict(ade=2, fde=1)


def test_complete_raw_audit_and_tamper_detection(
    synthetic_scene, metric_source, tmp_path
):
    sid, poses, payload, raw, anchors = synthetic_scene
    tracks = prepare_kitscenes.scene_streams(sid, anchors, poses, raw, payload)
    prepared, run = tmp_path / "prepared", tmp_path / "run"
    directory = prepared / "scenes" / sid
    metadata = directory / "source_metadata"
    metadata.mkdir(parents=True)
    np.savetxt(metadata / "poses.txt", poses)
    np.savetxt(metadata / "timestamp.reference.txt", poses[:, 0])
    scene = dict(
        scene_id=sid,
        tracks=tracks,
        metadata_sha256={
            name: common.sha256(metadata / name)
            for name in ("poses.txt", "timestamp.reference.txt")
        },
    )
    common.write_json(directory / "inputs.json", scene)
    record = dict(
        scene_id=sid,
        path=f"scenes/{sid}/inputs.json",
        sha256=common.sha256(directory / "inputs.json"),
    )
    counts = {t: common.coverage(s) for t, s in tracks.items()}
    hashes = {
        t: common.uid_hash(
            [f["sample_uid"] for s in streams for f in s["frames"] if f["scored"]]
        )
        for t, streams in tracks.items()
    }
    common.write_json(
        prepared / "manifest.json",
        dict(
            status="ready",
            full_evaluation=False,
            scenes=[record],
            coverage=counts,
            sample_uid_sha256=hashes,
        ),
    )
    run.mkdir()
    common.write_json(
        run / "status.json",
        dict(
            tracks=list(tracks),
            prepared_manifest_sha256=common.sha256(prepared / "manifest.json"),
        ),
    )
    for track, streams in tracks.items():
        destination = run / track / sid
        destination.mkdir(parents=True)
        rows = [
            dict(
                sample_uid=f["sample_uid"],
                scene_id=s["scene_id"],
                physical_scene_id=sid,
                frame=f["frame"],
                timestamp=f["timestamp"],
                fixed_command=2,
                prediction_frame="body_FLU",
                predictions_body_xy_2hz={"fixed_forward": np.zeros((6, 2)).tolist()},
            )
            for s in streams
            for f in s["frames"]
            if f["scored"]
        ]
        (destination / "samples.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        common.write_json(
            destination / "run.json",
            dict(
                status="predictions_complete",
                coverage=counts[track],
                input_sha256=record["sha256"],
                checkpoint_loaded_strict=True,
                samples_sha256=common.sha256(destination / "samples.jsonl"),
            ),
        )
    result = audit_results.audit(prepared, run, metric_source, run / "summary.json")
    assert result["status"] == "passed_independent_raw_rescore"
    assert result["tracks"]["dense"]["coverage"]["samples"] == 6
    assert set(result["tracks"]["dense"]["metrics"]) == {"canonical", "legacy_wgs84"}
    assert set(result["tracks"]["keyframes"]["metrics"]) == {"canonical"}
    assert result["tracks"]["dense"]["metrics"]["canonical"]["fixed_forward"][
        "shared_counts"
    ]["3"] == dict(ade=6, fde=6)
    path = run / "dense" / sid / "samples.jsonl"
    original = path.read_text()
    path.write_text(original + original.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="hash changed"):
        audit_results.audit(prepared, run, metric_source, run / "failed_audit.json")
    assert common.read_json(run / "failed_audit.json")["status"] == "failed"
    assert path.read_text().startswith(original)
