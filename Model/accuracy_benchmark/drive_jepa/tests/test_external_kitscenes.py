"""Verify population identity, causality, prediction support and scoring."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from accuracy_benchmark.drive_jepa.external_kitscenes import (
    digest_ids,
    published_digest_ids,
    resample_prediction,
    scene_samples,
    score_predictions,
)
from accuracy_benchmark.drive_jepa.trajectory_scoring import displacement_horizon_totals


def test_interpolation_includes_origin_and_never_extrapolates():
    times = np.arange(1, 9) * 0.5
    xy = np.column_stack([times * 2, times * 0])
    interpolated = resample_prediction(xy, times)
    assert interpolated.shape == (40, 2)
    np.testing.assert_allclose(interpolated[:, 0], np.arange(1, 41) * 0.2)
    totals = score_predictions(
        interpolated[None], np.zeros((1, 50, 2), dtype=np.float32)
    )
    assert "5s" not in totals
    assert totals["4s"] == pytest.approx((4.1, 1, 8.0, 1))


@pytest.mark.parametrize(
    "times,xy",
    [
        ([0.5, 0.5], [[1, 0], [2, 0]]),
        ([0, 1], [[1, 0], [2, 0]]),
        ([0.5, 1], [[np.nan, 0], [2, 0]]),
        ([0.5, np.inf], [[1, 0], [2, 0]]),
    ],
)
def test_invalid_predictions_fail(times, xy):
    with pytest.raises(ValueError):
        resample_prediction(xy, times)


def test_masked_scoring_weights_samples_equally_and_requires_fde_endpoint():
    errors = torch.tensor([[1.0, 3.0, 100.0], [2.0, 4.0, 6.0]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    totals = displacement_horizon_totals(errors, valid, {"h": 3})
    assert totals["h"] == (6.0, 2, 6.0, 1)


def test_shared_scorer_preserves_reactive_partial_horizon_policy():
    errors = torch.arange(1, 129, dtype=torch.float32).reshape(2, 64)
    valid = torch.zeros_like(errors, dtype=torch.bool)
    valid[0, :50] = True
    valid[1, :30] = True
    totals = displacement_horizon_totals(
        errors, valid, {"3s": 30, "5s": 50, "6p4s": 64}
    )
    assert totals["3s"] == (95.0, 2, 124.0, 2)
    assert totals["5s"] == (105.0, 2, 50.0, 1)
    assert totals["6p4s"] == (105.0, 2, 0.0, 0)


def fake_scene(positions, camera_length=None):
    count = len(positions)
    stamps = np.arange(count, dtype=np.int64) * 100_000_000
    poses = [
        SimpleNamespace(
            timestamp_ns=int(t),
            translation=np.r_[xy, 0.0],
            rotation=np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),
        )
        for t, xy in zip(stamps, positions)
    ]
    loader = SimpleNamespace(
        get_reference_timestamps=lambda: stamps,
        get_frame_indices=lambda camera: list(range(camera_length or count)),
    )
    return SimpleNamespace(
        get_sensor_loader=lambda scene: loader,
        get_scene=lambda scene: SimpleNamespace(ego_poses=poses),
    )


def test_population_uses_40_history_50_future_and_all_six_camera_spans():
    # Vehicle drives north: local forward must be +x, local lateral near zero.
    positions = np.column_stack([np.zeros(200), np.arange(200) * 0.1])
    samples = list(scene_samples(fake_scene(positions, camera_length=180), "scene"))
    assert len(samples) == 90
    assert samples[0]["frame_idx"] == 40
    assert samples[-1]["frame_idx"] == 129
    np.testing.assert_allclose(samples[0]["target_xy"][:, 0], np.arange(1, 51) * 0.1)
    np.testing.assert_allclose(samples[0]["target_xy"][:, 1], 0, atol=1e-6)
    np.testing.assert_allclose(samples[0]["velocity_xy"], [1, 0], atol=1e-6)


def test_status_does_not_depend_on_future_poses():
    positions = np.column_stack([np.arange(100) * 0.1, np.zeros(100)])
    before = next(scene_samples(fake_scene(positions), "scene"))
    positions[41:] *= 100
    after = next(scene_samples(fake_scene(positions), "scene"))
    np.testing.assert_array_equal(before["velocity_xy"], after["velocity_xy"])
    np.testing.assert_array_equal(before["acceleration_xy"], after["acceleration_xy"])
    assert not np.array_equal(before["target_xy"], after["target_xy"])


def test_digest_matches_reactive_order_independent_no_trailing_newline():
    import hashlib

    assert digest_ids(["b", "a"]) == hashlib.sha256(b"a\nb").hexdigest()


def test_overlay_digest_includes_terminal_newline():
    import hashlib

    assert published_digest_ids(["b", "a"]) == hashlib.sha256(b"a\nb\n").hexdigest()
    assert published_digest_ids(["b", "a"]) != digest_ids(["b", "a"])


def test_scene_outputs_resume_without_redownload_or_inference(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import accuracy_benchmark.drive_jepa.external_kitscenes as benchmark
    import accuracy_benchmark.drive_jepa.drive_jepa_adapter as drive_jepa

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint identity")
    positions = np.column_stack([np.zeros(100), np.arange(100) * 0.1])
    calls = []

    class Adapter:
        prediction_times_s = np.arange(1, 9) * 0.5

        def predict(self, sample):
            calls.append(sample["sample_uid"])
            return np.column_stack([self.prediction_times_s, np.zeros(8)])

    monkeypatch.setattr(benchmark, "check_sdk", lambda root: None)
    monkeypatch.setattr(benchmark, "git_revision", lambda root: "a" * 40)
    monkeypatch.setattr(
        benchmark, "sdk_split_scene_ids", lambda split: ["scene-a", "scene-b"]
    )
    monkeypatch.setattr(benchmark, "EXPECTED_VAL_SAMPLES", 20)
    monkeypatch.setattr(
        benchmark,
        "EXPECTED_VAL_SAMPLE_UID_SHA256",
        published_digest_ids(
            [
                f"kitscenes-v1-{scene}-f{frame:06d}"
                for scene in ("scene-a", "scene-b")
                for frame in range(40, 50)
            ]
        ),
    )
    monkeypatch.setattr(
        benchmark, "raw_scene", lambda args, scene: nullcontext(fake_scene(positions))
    )
    monkeypatch.setattr(drive_jepa, "create_adapter", lambda args: Adapter())
    args = SimpleNamespace(
        sdk_root=tmp_path,
        scene=None,
        stream=True,
        raw_root=None,
        commands={},
        command_file=None,
        command=[0.0, 0.0, 0.0, 0.0],
        dataset_revision="b" * 40,
        drive_jepa_root=tmp_path,
        checkpoint=checkpoint,
        adapter="accuracy_benchmark.drive_jepa.drive_jepa_adapter:create_adapter",
        camera="camera_base_front_center",
        precision="bf16",
        episode_ids=None,
        output=tmp_path / "results",
        reference_report=None,
    )
    first = benchmark.evaluate(args)
    second = benchmark.evaluate(args)
    assert first == second
    assert len(calls) == 20
    assert second["completed_scene_count"] == 2
    assert second["sample_count"] == 20
    assert second["metrics"]["trajectory"]["ade_4s_m"] == pytest.approx(0, abs=1e-6)
    assert second["metrics"]["trajectory"]["ade_5s_m"] is None


def test_pinned_scene_identity_matches_published_val_digest():
    from accuracy_benchmark.drive_jepa.source import (
        KITSCENES_STANDARD_VAL_SCENE_UID_SHA256,
        sdk_split_scene_ids,
    )

    scenes = sdk_split_scene_ids("val")
    assert len(scenes) == 117
    assert (
        digest_ids([f"kitscenes-{scene}" for scene in scenes])
        == KITSCENES_STANDARD_VAL_SCENE_UID_SHA256
    )
