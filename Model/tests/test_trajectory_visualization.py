"""Offline trajectory report tests adapted from trajectory-rendering PR #74."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evaluation.metrics import integrate_trajectory
from Platform.pipelines.overlay import write_overlay
from Tools.trajectory_visualization.artifacts import read_shard_samples
from Tools.trajectory_visualization.kinematics import (
    AOVL_V1_CONTROL_CONTRACT,
    integrate_controls,
)
from Tools.trajectory_visualization.rendering import render_frame
from Tools.trajectory_visualization.report import (
    generate_report,
    load_scene_selections,
)


def _tar_member(
    archive: tarfile.TarFile,
    name: str,
    payload: bytes,
) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _jpeg(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(output, format="JPEG")
    return output.getvalue()


def _write_shard(
    path: Path,
    sample_uids: list[str],
    *,
    frame_indices: list[int] | None = None,
    calibration: dict | None = None,
    camera_color: str = "#334155",
) -> None:
    if frame_indices is None:
        frame_indices = [64 + index for index in range(len(sample_uids))]
    if len(frame_indices) != len(sample_uids):
        raise ValueError("frame_indices must align with sample_uids")
    calibration_bytes = json.dumps(calibration or {
        "dataset": "yaak-ai/L2D",
        "geometry_type": "pseudo",
        "projection": None,
    }).encode()
    with tarfile.open(path, mode="w") as archive:
        for index, sample_uid in enumerate(sample_uids):
            metadata = json.dumps({
                "dataset": "yaak-ai/L2D",
                "sample_uid": sample_uid,
                "split_group_uid": "l2d-v1-e000001",
                "frame_idx": frame_indices[index],
            }).encode()
            ego = np.zeros(64 * 4 + 64 * 2, dtype="<f4")
            ego[63 * 4] = 8.0 + index
            ego[64 * 4 :: 2] = 0.1
            _tar_member(archive, f"{sample_uid}.meta.json", metadata)
            _tar_member(archive, f"{sample_uid}.ego.npy", ego.tobytes())
            _tar_member(
                archive,
                f"{sample_uid}.cam_0.jpg",
                _jpeg(camera_color),
            )
            _tar_member(
                archive,
                f"{sample_uid}.calib.json",
                calibration_bytes,
            )


def _write_publication_manifests(
    root: Path,
    *,
    shard: Path,
    overlay: Path,
    sample_count: int,
    seeds: list[int] | None = None,
) -> tuple[Path, Path]:
    seeds = seeds or [0]
    dataset_manifest = {
        "schema_version": "v1",
        "status": "ready",
        "dataset": "yaak-ai/L2D",
        "version": "v2.1",
        "total_samples": sample_count,
        "shard_entries": [{
            "name": shard.name,
            "key": f"l2d/v2.1/shards/{shard.name}",
            "byte_size": shard.stat().st_size,
            "content_identity": "1" * 64,
        }],
    }
    dataset_path = root / "dataset-manifest.json"
    dataset_path.write_text(json.dumps(
        dataset_manifest,
        indent=2,
        sort_keys=True,
    ))
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    overlay_payload = overlay.read_bytes()
    overlay_manifest = {
        "schema_version": "v1",
        "status": "ready",
        "registered_model_name": "AutoE2E",
        "model_version": 7,
        "run_id": "run-123",
        "model_artifact_sha256": "2" * 64,
        "dataset": "yaak-ai/L2D",
        "version": "v2.1",
        "dataset_manifest_sha256": dataset_sha256,
        "request_identity": "3" * 64,
        "cache_identity": "4" * 64,
        "seeds": seeds,
        "sampler": "model-default",
        "num_inference_steps": 1,
        "inference_contract_version": "v1",
        "noise_policy_version": "v1",
        "overlay_binary_schema": "v1",
        "shards": [{
            "shard": shard.name,
            "s3_key": "overlays/example/overlay.bin.gz",
            "sha256": hashlib.sha256(overlay_payload).hexdigest(),
            "byte_size": len(overlay_payload),
            "sample_count": sample_count,
            "seeds": seeds,
        }],
    }
    overlay_path = root / "overlay-manifest.json"
    overlay_path.write_text(json.dumps(
        overlay_manifest,
        indent=2,
        sort_keys=True,
    ))
    return dataset_path, overlay_path


def test_report_integrator_matches_evaluation_reference():
    controls = np.zeros((64, 2), dtype=np.float32)
    controls[:, 0] = 0.25
    controls[:, 1] = 0.01

    actual = integrate_controls(controls, 7.5)
    expected = integrate_trajectory(
        controls[:, 0],
        controls[:, 1],
        7.5,
    )

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    mirrored = integrate_controls(controls, 7.5, curvature_sign=-1)
    np.testing.assert_allclose(mirrored[:, 0], expected[:, 0])
    np.testing.assert_allclose(mirrored[:, 1], -expected[:, 1])


def test_report_joins_aovl_by_uid_and_writes_scene_artifacts(tmp_path):
    sample_uids = [
        "l2d-v1-e000001-f000064",
        "l2d-v1-e000001-f000065",
    ]
    shard = tmp_path / "train-000000.tar"
    _write_shard(
        shard,
        list(reversed(sample_uids)),
        frame_indices=[65, 64],
    )
    overlay = tmp_path / "overlay.bin.gz"
    controls = np.zeros((2, 1, 64, 2), dtype=np.float32)
    controls[1, 0, :, 1] = 0.02
    write_overlay(
        overlay,
        list(reversed(sample_uids)),
        controls,
        np.array([8.0, 9.0], dtype=np.float32),
    )
    dataset_manifest, overlay_manifest = _write_publication_manifests(
        tmp_path,
        shard=shard,
        overlay=overlay,
        sample_count=2,
    )

    rendered_sizes = []

    def fake_video_writer(path, frames, fps):
        assert fps == 10.0
        for frame in frames:
            rendered_sizes.append(frame.size)
        path.write_bytes(b"synthetic-mp4")

    output = tmp_path / "report"
    manifest = generate_report(
        shard_path=shard,
        overlay_path=overlay,
        output_dir=output,
        dataset_manifest_path=dataset_manifest,
        overlay_manifest_path=overlay_manifest,
        video_writer=fake_video_writer,
    )

    assert rendered_sizes == [(1280, 720), (1280, 720)]
    assert manifest["dataset"] == "yaak-ai/L2D"
    assert manifest["render"]["curvature_sign"] == -1
    assert manifest["render"]["base_seed"] == 0
    assert manifest["render"]["control_contract"] == (
        AOVL_V1_CONTROL_CONTRACT.manifest()
    )
    assert manifest["render"]["panel_order"] == ["camera", "metric_bev"]
    assert (
        manifest["render"]["camera_projection_status"]
        == "unsupported_pseudo_geometry"
    )
    assert manifest["publication"]["model"]["artifact_sha256"] == "2" * 64
    assert manifest["publication"]["dataset"]["version"] == "v2.1"
    assert manifest["publication"]["overlay"]["request_identity"] == "3" * 64
    assert manifest["scene_count"] == 1
    assert manifest["frame_count"] == 2
    scene = manifest["scenes"][0]
    assert scene["sample_uids"] == sample_uids
    assert scene["start_frame"] == 64
    assert scene["end_frame"] == 65
    assert scene["metrics"]["max_error_m"] > 0
    assert (output / scene["video"]).read_bytes() == b"synthetic-mp4"
    assert (output / scene["thumbnail"]).stat().st_size > 0
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_report_uses_explicit_scene_frame_selection(tmp_path):
    sample_uids = [
        "l2d-v1-e000001-f000064",
        "l2d-v1-e000001-f000065",
        "l2d-v1-e000001-f000066",
    ]
    shard = tmp_path / "train-000000.tar"
    _write_shard(shard, sample_uids)
    overlay = tmp_path / "overlay.bin.gz"
    write_overlay(
        overlay,
        sample_uids,
        np.zeros((3, 1, 64, 2), dtype=np.float32),
        np.array([8.0, 9.0, 10.0], dtype=np.float32),
    )
    dataset_manifest, overlay_manifest = _write_publication_manifests(
        tmp_path,
        shard=shard,
        overlay=overlay,
        sample_count=3,
    )
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "schema_version": 1,
        "scenes": [{
            "scene_uid": "l2d-v1-e000001",
            "start_frame": 64,
            "end_frame": 65,
        }],
    }))

    rendered = []

    def fake_video_writer(path, frames, fps):
        rendered.extend(frames)
        path.write_bytes(b"synthetic-mp4")

    output = tmp_path / "report"
    manifest = generate_report(
        shard_path=shard,
        overlay_path=overlay,
        output_dir=output,
        dataset_manifest_path=dataset_manifest,
        overlay_manifest_path=overlay_manifest,
        scene_selections=load_scene_selections(selection),
        max_frames_per_scene=1,
        video_writer=fake_video_writer,
    )

    assert len(rendered) == 2
    assert manifest["scenes"][0]["sample_uids"] == sample_uids[:2]
    assert manifest["render"]["scene_selection"] == [{
        "scene_uid": "l2d-v1-e000001",
        "start_frame": 64,
        "end_frame": 65,
    }]


def test_selection_manifest_rejects_legacy_episode_identity(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "schema_version": 1,
        "scenes": [{"episode_id": "legacy-episode"}],
    }))

    with pytest.raises(ValueError, match="unknown fields"):
        load_scene_selections(selection)


def test_report_rejects_overlay_rows_outside_the_shard(tmp_path):
    sample_uid = "l2d-v1-e000001-f000064"
    shard = tmp_path / "train-000000.tar"
    _write_shard(shard, [sample_uid])
    overlay = tmp_path / "overlay.bin.gz"
    write_overlay(
        overlay,
        [sample_uid, "l2d-v1-e000001-f000065"],
        np.zeros((2, 1, 64, 2), dtype=np.float32),
        np.array([8.0, 9.0], dtype=np.float32),
    )
    dataset_manifest, overlay_manifest = _write_publication_manifests(
        tmp_path,
        shard=shard,
        overlay=overlay,
        sample_count=2,
    )

    with pytest.raises(ValueError, match="exactly match"):
        generate_report(
            shard_path=shard,
            overlay_path=overlay,
            output_dir=tmp_path / "report",
            dataset_manifest_path=dataset_manifest,
            overlay_manifest_path=overlay_manifest,
        )


def test_report_rejects_overlay_speed_mismatched_with_shard(tmp_path):
    sample_uid = "l2d-v1-e000001-f000064"
    shard = tmp_path / "train-000000.tar"
    _write_shard(shard, [sample_uid])
    overlay = tmp_path / "overlay.bin.gz"
    write_overlay(
        overlay,
        [sample_uid],
        np.zeros((1, 1, 64, 2), dtype=np.float32),
        np.array([99.0], dtype=np.float32),
    )
    dataset_manifest, overlay_manifest = _write_publication_manifests(
        tmp_path,
        shard=shard,
        overlay=overlay,
        sample_count=1,
    )

    with pytest.raises(ValueError, match="disagrees with shard history"):
        generate_report(
            shard_path=shard,
            overlay_path=overlay,
            output_dir=tmp_path / "report",
            dataset_manifest_path=dataset_manifest,
            overlay_manifest_path=overlay_manifest,
        )


def test_report_rejects_changed_dataset_manifest(tmp_path):
    sample_uid = "l2d-v1-e000001-f000064"
    shard = tmp_path / "train-000000.tar"
    _write_shard(shard, [sample_uid])
    overlay = tmp_path / "overlay.bin.gz"
    write_overlay(
        overlay,
        [sample_uid],
        np.zeros((1, 1, 64, 2), dtype=np.float32),
        np.array([8.0], dtype=np.float32),
    )
    dataset_manifest, overlay_manifest = _write_publication_manifests(
        tmp_path,
        shard=shard,
        overlay=overlay,
        sample_count=1,
    )
    document = json.loads(dataset_manifest.read_text())
    document["total_samples"] = 2
    dataset_manifest.write_text(json.dumps(
        document,
        indent=2,
        sort_keys=True,
    ))

    with pytest.raises(ValueError, match="manifest digests differ"):
        generate_report(
            shard_path=shard,
            overlay_path=overlay,
            output_dir=tmp_path / "report",
            dataset_manifest_path=dataset_manifest,
            overlay_manifest_path=overlay_manifest,
        )


def test_report_rejects_overlay_body_digest_mismatch(tmp_path):
    sample_uid = "l2d-v1-e000001-f000064"
    shard = tmp_path / "train-000000.tar"
    _write_shard(shard, [sample_uid])
    overlay = tmp_path / "overlay.bin.gz"
    write_overlay(
        overlay,
        [sample_uid],
        np.zeros((1, 1, 64, 2), dtype=np.float32),
        np.array([8.0], dtype=np.float32),
    )
    dataset_manifest, overlay_manifest = _write_publication_manifests(
        tmp_path,
        shard=shard,
        overlay=overlay,
        sample_count=1,
    )
    document = json.loads(overlay_manifest.read_text())
    document["shards"][0]["sha256"] = "f" * 64
    overlay_manifest.write_text(json.dumps(
        document,
        indent=2,
        sort_keys=True,
    ))

    with pytest.raises(ValueError, match="local overlay SHA-256"):
        generate_report(
            shard_path=shard,
            overlay_path=overlay,
            output_dir=tmp_path / "report",
            dataset_manifest_path=dataset_manifest,
            overlay_manifest_path=overlay_manifest,
        )


def test_shard_reader_rejects_non_integer_frame_index(tmp_path):
    shard = tmp_path / "train-000000.tar"
    _write_shard(
        shard,
        ["l2d-v1-e000001-f000064"],
        frame_indices=[64.5],
    )

    with pytest.raises(ValueError, match="invalid frame_idx"):
        read_shard_samples(shard)


def test_shard_reader_rejects_missing_selected_camera(tmp_path):
    shard = tmp_path / "train-000000.tar"
    _write_shard(shard, ["l2d-v1-e000001-f000064"])

    try:
        read_shard_samples(shard, camera_index=1)
    except ValueError as exc:
        assert "missing report members" in str(exc)
    else:
        raise AssertionError("missing camera must fail report generation")


def test_render_frame_keeps_camera_and_bev_panels_in_declared_order(tmp_path):
    sample_uid = "l2d-v1-e000001-f000064"
    shard = tmp_path / "train-000000.tar"
    _write_shard(
        shard,
        [sample_uid],
        camera_color="#ff0000",
        calibration={
            "dataset": "yaak-ai/L2D",
            "geometry_type": "pinhole",
            "projection": {
                "type": "pinhole",
                "matrix": [[
                    [32.0, -50.0, 0.0, 0.0],
                    [32.0, 0.0, 0.0, 64.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]],
            },
        },
    )
    sample = read_shard_samples(shard)[0]
    controls = np.zeros((64, 2), dtype=np.float32)
    target = integrate_controls(controls, 8.0, curvature_sign=-1)
    prediction = target.copy()
    prediction[:, 1] += np.linspace(0, 5, 64)

    rendered = render_frame(
        sample,
        prediction=prediction,
        target=target,
        v0=8.0,
        base_seed=0,
        extent=30.0,
        camera_index=0,
    )

    assert rendered.size == (1280, 720)
    pixels = np.asarray(rendered)
    camera_panel = pixels[136:696, 24:584]
    bev_panel = pixels[136:696, 696:1256]
    assert camera_panel[..., 0].mean() > 200
    assert camera_panel[..., 1].mean() < 30
    assert bev_panel[..., 0].mean() < 80
    assert np.any(np.all(pixels == (52, 211, 153), axis=2))
