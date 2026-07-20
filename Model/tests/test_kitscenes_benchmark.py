"""KITScenes fixed-window benchmark contract tests."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from evaluation.kitscenes_benchmark import (
    EVALUATOR_VERSION,
    MANIFEST_SCHEMA_VERSION,
    PAPER_PROTOCOL_SOURCE,
    PROTOCOL_ID,
    compute_displacement_metrics,
    limit_egomotion_history,
    load_benchmark_manifest,
    parse_benchmark_manifest,
    sample_uid_digest,
    wgs84_trajectory_to_ego_xy,
)


def _uid(frame: int = 64) -> str:
    return (
        "kitscenes-v1-01234567-89ab-cdef-0123-456789abcdef-"
        f"f{frame:06d}"
    )


def _manifest(**overrides):
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "benchmark_id": "auto-e2e-development-v1",
        "protocol_status": "development",
        "protocol_source": PAPER_PROTOCOL_SOURCE,
        "authority": "auto-e2e",
        "release_id": "pinned-alpha-v1",
        "dataset_revision": "a" * 40,
        "sdk_revision": "b" * 40,
        "source_splits": ["train"],
        "sample_count": 1,
        "sample_uids": [_uid()],
        "frequency_hz": 10,
        "past_seconds": 4,
        "horizons_seconds": [3, 5],
        "input_track": "camera-map",
        "history_adapter": "left_zero_pad_to_64",
    }
    payload.update(overrides)
    return payload


def test_manifest_parses_fixed_protocol_and_byte_digest(tmp_path):
    payload = _manifest()
    raw = json.dumps(payload, indent=2).encode("ascii")
    path = tmp_path / "benchmark.json"
    path.write_bytes(raw)

    manifest, digest = load_benchmark_manifest(path)

    assert manifest.sample_uids == (_uid(),)
    assert manifest.observation_steps == 40
    assert manifest.horizon_steps == (30, 50)
    assert digest == hashlib.sha256(raw).hexdigest()
    assert EVALUATOR_VERSION == "kitscenes_pose_displacement_v1"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_revision", "short", "40-character commit SHA"),
        ("sdk_revision", "A" * 40, "40-character commit SHA"),
        ("authority", "bad authority", "unsafe authority"),
        ("release_id", "../release", "unsafe release_id"),
        ("frequency_hz", 20, "requires 10 Hz"),
        ("past_seconds", 6, "requires 10 Hz"),
        ("horizons_seconds", [3], "requires 10 Hz"),
        ("history_adapter", "truncate", "left_zero_pad_to_64"),
    ],
)
def test_manifest_rejects_provenance_or_protocol_drift(
    field, value, message
):
    with pytest.raises(ValueError, match=message):
        parse_benchmark_manifest(_manifest(**{field: value}))


def test_manifest_rejects_duplicate_or_malformed_sample_uids():
    with pytest.raises(ValueError, match="duplicates"):
        parse_benchmark_manifest(
            _manifest(
                sample_count=2,
                sample_uids=[_uid(), _uid()],
            )
        )

    with pytest.raises(ValueError, match="unsafe sample UIDs"):
        parse_benchmark_manifest(
            _manifest(sample_uids=["kitscenes-v1-not-a-scene-f000064"])
        )


def test_paper_approximation_requires_exact_200_sample_contract():
    sample_uids = [_uid(frame) for frame in range(200)]
    manifest = parse_benchmark_manifest(
        _manifest(
            protocol_status="paper_protocol_approximation",
            authority="auto-e2e",
            release_id="local-paper-approx-v1",
            source_splits=["val", "overlap-train-val"],
            sample_count=200,
            sample_uids=sample_uids,
        )
    )
    assert manifest.protocol_status == "paper_protocol_approximation"

    with pytest.raises(ValueError, match="must cite"):
        parse_benchmark_manifest(
            _manifest(
                protocol_status="paper_protocol_approximation",
                protocol_source="https://example.com/protocol",
                source_splits=["val", "overlap-train-val"],
                sample_count=200,
                sample_uids=sample_uids,
            )
        )


def test_official_status_requires_authority_and_200_samples():
    sample_uids = [_uid(frame) for frame in range(200)]
    manifest = parse_benchmark_manifest(
        _manifest(
            protocol_status="official",
            authority="KIT-MRT",
            release_id="challenge-v1",
            source_splits=["test-e2e"],
            sample_count=200,
            sample_uids=sample_uids,
        )
    )
    assert manifest.authority == "KIT-MRT"

    with pytest.raises(ValueError, match="issued by KIT-MRT"):
        parse_benchmark_manifest(
            _manifest(
                protocol_status="official",
                source_splits=["test-e2e"],
                sample_count=200,
                sample_uids=sample_uids,
            )
        )


def test_history_adapter_masks_only_context_older_than_four_seconds():
    history = torch.arange(2 * 64 * 4).reshape(2, 64 * 4)
    original = history.clone()

    limited = limit_egomotion_history(
        history, observation_steps=40
    ).reshape(2, 64, 4)

    assert torch.count_nonzero(limited[:, :24]) == 0
    assert torch.equal(limited[:, 24:], original.reshape(2, 64, 4)[:, 24:])
    assert torch.equal(history, original)


def test_wgs84_trajectory_is_rotated_into_current_ego_frame():
    pyproj = pytest.importorskip("pyproj")
    to_wgs84 = pyproj.Transformer.from_crs(
        "EPSG:32632", "EPSG:4326", always_xy=True
    )
    steps = np.arange(65, dtype=np.float64)
    utm_paths = np.stack([
        np.column_stack([
            np.full(65, 450_000.0),
            5_420_000.0 + steps,
        ]),
        np.column_stack([
            450_000.0 + steps,
            np.full(65, 5_420_000.0),
        ]),
    ])
    gps = []
    for path in utm_paths:
        longitude, latitude = to_wgs84.transform(path[:, 0], path[:, 1])
        gps.append(np.column_stack([latitude, longitude]))
    gps = np.asarray(gps)
    current_pose = np.column_stack([
        gps[:, 0, 0],
        gps[:, 0, 1],
        np.array([0.0, 90.0]),
    ])

    xy = wgs84_trajectory_to_ego_xy(gps, current_pose)

    assert xy.shape == (2, 64, 2)
    assert xy[:, :, 0] == pytest.approx(
        np.tile(np.arange(1, 65), (2, 1)), abs=1e-5
    )
    assert xy[:, :, 1] == pytest.approx(
        np.zeros((2, 64)), abs=1e-5
    )


def test_wgs84_trajectory_rejects_mismatched_current_pose():
    gps = np.zeros((1, 65, 2), dtype=np.float64)
    pose = np.zeros((1, 3), dtype=np.float64)
    pose[0, 0] = 1.0

    with pytest.raises(ValueError, match="does not match"):
        wgs84_trajectory_to_ego_xy(gps, pose)


def test_displacement_metrics_pin_three_and_five_second_horizons():
    predicted = np.zeros((1, 64, 2), dtype=np.float64)
    predicted[:, :, 0] = 1.0
    target = np.zeros_like(predicted)

    metrics, predicted_xy = compute_displacement_metrics(
        predicted,
        target,
        np.zeros(1),
    )

    assert metrics == pytest.approx(
        {
            "ade_3s": 0.005 * 31 * 32 / 3,
            "fde_3s": 0.005 * 30 * 31,
            "ade_5s": 0.005 * 51 * 52 / 3,
            "fde_5s": 0.005 * 50 * 51,
        }
    )
    assert predicted_xy.shape == (1, 50, 2)


@pytest.mark.parametrize(
    ("predicted", "speeds", "message"),
    [
        (np.empty((0, 64, 2)), np.empty((0,)), "must not be empty"),
        (
            np.full((1, 64, 2), np.nan),
            np.zeros(1),
            "non-finite",
        ),
        (np.zeros((1, 64, 2)), np.array([-1.0]), "non-negative"),
    ],
)
def test_displacement_metrics_reject_invalid_inputs(
    predicted, speeds, message
):
    with pytest.raises(ValueError, match=message):
        compute_displacement_metrics(
            predicted,
            np.zeros_like(predicted),
            speeds,
        )


def test_sample_uid_digest_is_order_independent():
    expected = hashlib.sha256(
        f"{_uid(1)}\n{_uid(2)}".encode("ascii")
    ).hexdigest()

    assert sample_uid_digest([_uid(2), _uid(1)]) == expected
