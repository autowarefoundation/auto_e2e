"""Offline trajectory report tests adapted from trajectory-rendering PR #74."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.metrics import integrate_trajectory
from Platform.pipelines.overlay import write_overlay
from Tools.trajectory_visualization.artifacts import read_shard_samples
from Tools.trajectory_visualization.rendering import (
    integrate_controls,
    render_frame,
)
from Tools.trajectory_visualization.report import generate_report


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


def _write_shard(path: Path, sample_uids: list[str]) -> None:
    calibration = json.dumps({
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
                "frame_idx": 64 + index,
            }).encode()
            ego = np.zeros(64 * 4 + 64 * 2, dtype="<f4")
            ego[63 * 4] = 8.0 + index
            ego[64 * 4 :: 2] = 0.1
            _tar_member(archive, f"{sample_uid}.meta.json", metadata)
            _tar_member(archive, f"{sample_uid}.ego.npy", ego.tobytes())
            _tar_member(
                archive,
                f"{sample_uid}.cam_0.jpg",
                _jpeg("#334155"),
            )
            _tar_member(
                archive,
                f"{sample_uid}.calib.json",
                calibration,
            )


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
    _write_shard(shard, sample_uids)
    overlay = tmp_path / "overlay.bin.gz"
    controls = np.zeros((2, 1, 64, 2), dtype=np.float32)
    controls[1, 0, :, 1] = 0.02
    write_overlay(
        overlay,
        list(reversed(sample_uids)),
        controls,
        np.array([9.0, 8.0], dtype=np.float32),
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
        video_writer=fake_video_writer,
    )

    assert rendered_sizes == [(1280, 720), (1280, 720)]
    assert manifest["dataset"] == "yaak-ai/L2D"
    assert manifest["render"]["curvature_sign"] == -1
    assert manifest["render"]["base_seed"] == 0
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


def test_shard_reader_rejects_missing_selected_camera(tmp_path):
    shard = tmp_path / "train-000000.tar"
    _write_shard(shard, ["l2d-v1-e000001-f000064"])

    try:
        read_shard_samples(shard, camera_index=1)
    except ValueError as exc:
        assert "missing report members" in str(exc)
    else:
        raise AssertionError("missing camera must fail report generation")


def test_render_frame_changes_camera_and_preserves_fixed_dimensions(tmp_path):
    sample_uid = "l2d-v1-e000001-f000064"
    shard = tmp_path / "train-000000.tar"
    _write_shard(shard, [sample_uid])
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
    assert pixels.mean() > 2
    assert np.any(np.all(pixels == (52, 211, 153), axis=2))
