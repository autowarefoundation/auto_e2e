"""Canonical AOVL binary artifact tests."""

from __future__ import annotations

import gzip
import hashlib
import struct

import numpy as np
import pytest

from Platform.pipelines.overlay import (
    BEV_HEATMAP_NAMES,
    BEV_HEATMAP_SIZE,
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
    base = np.linspace(
        0.0,
        1.0,
        BEV_HEATMAP_SIZE * BEV_HEATMAP_SIZE,
        dtype=np.float32,
    ).reshape(BEV_HEATMAP_SIZE, BEV_HEATMAP_SIZE)
    branch_scales = np.array(
        [0.01, 30.0, 0.2, 28.0, 0.5, 1.0],
        dtype=np.float32,
    )
    heatmaps = np.stack(
        [
            np.stack(
                [base * scale * (row + 1) for scale in branch_scales]
            )
            for row in range(3)
        ]
    )
    return uids, controls, v0, heatmaps


def _legacy_payload(version, uids, controls, v0, heatmaps):
    base_seeds = (0, 1)
    raw_v4 = gzip.decompress(
        encode_overlay(
            uids,
            controls,
            v0,
            base_seeds=base_seeds,
            bev_heatmaps=heatmaps,
        )
    )
    prefix_size = (
        20
        + len(base_seeds) * 8
        + len(uids) * 12
        + controls.size * 4
        + v0.size * 4
    )
    if version == 1:
        raw = (
            struct.pack(
                "<4sHHIHHHH",
                b"AOVL",
                version,
                0,
                len(uids),
                len(base_seeds),
                64,
                2,
                0,
            )
            + raw_v4[20:prefix_size]
        )
        return gzip.compress(raw, compresslevel=6, mtime=0), (), None

    indices = (0, 3, 5) if version == 2 else tuple(range(6))
    selected = heatmaps[:, indices]
    scales = selected.max(axis=(1, 2, 3)).astype("<f4")
    divisors = np.where(scales > 0, scales, 1.0).reshape(-1, 1, 1, 1)
    quantised = np.rint(
        np.clip(selected / divisors, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    raw = (
        struct.pack(
            "<4sHHIHHHH",
            b"AOVL",
            version,
            0,
            len(uids),
            len(base_seeds),
            64,
            2,
            len(indices),
        )
        + raw_v4[20:prefix_size]
        + scales.tobytes(order="C")
        + quantised.tobytes(order="C")
    )
    names = (
        ("image", "navigation", "fused")
        if version == 2
        else BEV_HEATMAP_NAMES
    )
    return gzip.compress(raw, compresslevel=6, mtime=0), names, selected


def test_overlay_roundtrip_and_sorted_directory():
    uids, controls, v0, heatmaps = _fixture()
    payload = encode_overlay(
        uids,
        controls,
        v0,
        base_seeds=(0, 7),
        deterministic_planner=True,
        bev_heatmaps=heatmaps,
    )
    decoded = decode_overlay(payload)

    assert decoded.flags & FLAG_DETERMINISTIC_PLANNER
    assert decoded.base_seeds == (0, 7)
    assert list(decoded.directory) == sorted(
        (sample_uid_hash(uid), row) for row, uid in enumerate(uids)
    )
    np.testing.assert_array_equal(decoded.controls, controls)
    np.testing.assert_array_equal(decoded.v0, v0)
    np.testing.assert_allclose(
        decoded.bev_heatmap_scales,
        heatmaps.max(axis=(2, 3)),
    )
    quantization_error = np.abs(decoded.bev_heatmaps - heatmaps)
    assert np.all(
        quantization_error
        <= decoded.bev_heatmap_scales[:, :, None, None] / 255.0
    )
    assert np.unique(
        decoded.bev_heatmaps[0, 0],
    ).size > 200
    assert decoded.bev_heatmap_names == BEV_HEATMAP_NAMES


def test_overlay_gzip_bytes_are_deterministic():
    uids, controls, v0, heatmaps = _fixture()
    first = encode_overlay(
        uids, controls, v0, base_seeds=(0, 1), bev_heatmaps=heatmaps
    )
    second = encode_overlay(
        uids, controls, v0, base_seeds=(0, 1), bev_heatmaps=heatmaps
    )
    assert first == second


def test_overlay_writer_returns_body_pointer_metadata(tmp_path):
    uids, controls, v0, heatmaps = _fixture()
    path = tmp_path / "overlay.bin.gz"
    artifact = write_overlay(
        path,
        uids,
        controls,
        v0,
        base_seeds=(0, 7),
        bev_heatmaps=heatmaps,
    )
    assert artifact.path == path
    assert artifact.sample_count == 3
    assert artifact.seed_count == 2
    assert artifact.byte_size == path.stat().st_size
    assert artifact.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_overlay_validation_rejects_ambiguous_or_bad_data():
    uids, controls, v0, heatmaps = _fixture()
    with pytest.raises(ValueError, match="unique"):
        encode_overlay(
            [uids[0]] * 3,
            controls,
            v0,
            base_seeds=(0, 1),
            bev_heatmaps=heatmaps,
        )
    with pytest.raises(ValueError, match="shape"):
        encode_overlay(
            uids,
            controls[:, :1],
            v0,
            base_seeds=(0, 1),
            bev_heatmaps=heatmaps,
        )
    bad = controls.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        encode_overlay(
            uids,
            bad,
            v0,
            base_seeds=(0, 1),
            bev_heatmaps=heatmaps,
        )
    with pytest.raises(ValueError, match="shape"):
        encode_overlay(
            uids,
            controls,
            v0,
            base_seeds=(0, 1),
            bev_heatmaps=heatmaps[:, :2],
        )
    invalid_heatmaps = heatmaps.copy()
    invalid_heatmaps[0, 0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        encode_overlay(
            uids,
            controls,
            v0,
            base_seeds=(0, 1),
            bev_heatmaps=invalid_heatmaps,
        )
    invalid_heatmaps[0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        encode_overlay(
            uids,
            controls,
            v0,
            base_seeds=(0, 1),
            bev_heatmaps=invalid_heatmaps,
        )


@pytest.mark.parametrize("version", [1, 2, 3])
def test_overlay_decoder_accepts_legacy_formats(version):
    uids, controls, v0, heatmaps = _fixture()
    payload, names, selected = _legacy_payload(
        version,
        uids,
        controls,
        v0,
        heatmaps,
    )
    decoded = decode_overlay(payload)

    np.testing.assert_array_equal(decoded.controls, controls)
    np.testing.assert_array_equal(decoded.v0, v0)
    assert decoded.bev_heatmap_names == names
    if version == 1:
        assert decoded.bev_heatmaps is None
        assert decoded.bev_heatmap_scales is None
        return

    assert selected is not None
    expected_scales = selected.max(axis=(1, 2, 3))
    np.testing.assert_array_equal(decoded.bev_heatmap_scales, expected_scales)
    quantization_error = np.abs(decoded.bev_heatmaps - selected)
    assert np.all(
        quantization_error
        <= expected_scales[:, None, None, None] / 255.0
    )


def test_overlay_key_is_split_free_and_validates_segments():
    model_id = "a" * 64
    key = overlay_s3_key(model_id, "l2d", "v2.1", "train-000001.tar")
    assert key == (
        "overlays/schema=v4/model=" + model_id
        + "/dataset=l2d/version=v2.1/shard=train-000001.tar/overlay.bin.gz"
    )
    assert "split=" not in key and "source=" not in key

    with pytest.raises(ValueError, match="path segment"):
        overlay_s3_key(model_id, "yaak-ai/L2D", "v2.1", "train.tar")
