"""Canonical AOVL binary artifact tests."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from Platform.pipelines.overlay import (
    FLAG_DETERMINISTIC_PLANNER,
    decode_overlay,
    encode_overlay,
    overlay_s3_key,
    sample_uid_hash,
    write_overlay,
)


def _fixture():
    uids = [
        "l2d-v1-e000012-f000064",
        "l2d-v1-e000012-f000065",
        "l2d-v1-e000012-f000066",
    ]
    controls = np.arange(3 * 2 * 64 * 2, dtype=np.float32).reshape(3, 2, 64, 2)
    v0 = np.array([3.5, 4.5, 5.5], dtype=np.float32)
    return uids, controls, v0


def test_overlay_roundtrip_and_sorted_directory():
    uids, controls, v0 = _fixture()
    payload = encode_overlay(
        uids,
        controls,
        v0,
        base_seeds=(0, 7),
        deterministic_planner=True,
    )
    decoded = decode_overlay(payload)

    assert decoded.flags & FLAG_DETERMINISTIC_PLANNER
    assert decoded.base_seeds == (0, 7)
    assert list(decoded.directory) == sorted(
        (sample_uid_hash(uid), row) for row, uid in enumerate(uids)
    )
    np.testing.assert_array_equal(decoded.controls, controls)
    np.testing.assert_array_equal(decoded.v0, v0)


def test_overlay_gzip_bytes_are_deterministic():
    uids, controls, v0 = _fixture()
    first = encode_overlay(uids, controls, v0, base_seeds=(0, 1))
    second = encode_overlay(uids, controls, v0, base_seeds=(0, 1))
    assert first == second


def test_overlay_writer_returns_body_pointer_metadata(tmp_path):
    uids, controls, v0 = _fixture()
    path = tmp_path / "overlay.bin.gz"
    artifact = write_overlay(
        path, uids, controls, v0, base_seeds=(0, 7)
    )
    assert artifact.path == path
    assert artifact.sample_count == 3
    assert artifact.seed_count == 2
    assert artifact.byte_size == path.stat().st_size
    assert artifact.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_overlay_validation_rejects_ambiguous_or_bad_data():
    uids, controls, v0 = _fixture()
    with pytest.raises(ValueError, match="unique"):
        encode_overlay([uids[0]] * 3, controls, v0, base_seeds=(0, 1))
    with pytest.raises(ValueError, match="shape"):
        encode_overlay(uids, controls[:, :1], v0, base_seeds=(0, 1))
    bad = controls.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        encode_overlay(uids, bad, v0, base_seeds=(0, 1))


def test_overlay_key_is_split_free_and_validates_segments():
    model_id = "a" * 64
    key = overlay_s3_key(model_id, "l2d", "v2.1", "train-000001.tar")
    assert key == (
        "overlays/schema=v1/model=" + model_id
        + "/dataset=l2d/version=v2.1/shard=train-000001.tar/overlay.bin.gz"
    )
    assert "split=" not in key and "source=" not in key

    with pytest.raises(ValueError, match="path segment"):
        overlay_s3_key(model_id, "yaak-ai/L2D", "v2.1", "train.tar")
