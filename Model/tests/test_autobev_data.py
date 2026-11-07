"""AutoBEV sample members, their loader layout, and pack mixing."""

from __future__ import annotations

import io
import json
import random
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

import data_parsing.pre_extracted as pre_extracted
from data_parsing.pre_extracted import (
    AUTOBEV_BEV_ONLY_SHARE,
    collate_autobev_batch,
    make_autobev_bev_share_loader,
    make_autobev_sample_count_loader,
    make_pre_extracted_loader,
)
from data_processing.autobev_members import (
    BOX_COLUMNS,
    camera_member,
    crop_params,
    crop_projection,
    decode_boxes,
    decode_stride_depth,
    encode_boxes,
    encode_camera,
    encode_depth,
    encode_npy,
    history_camera_member,
    manifest_contract,
)
from data_processing.reactive_training_artifacts import (
    encode_bev_segmentation,
    encode_reactive_navigation,
    encode_trajectory_xy,
)
from model_components.autobev.geometry import (
    AUTOBEV_CONTRACT,
    BEV_H,
    BEV_W,
    HISTORY_FRAMES,
    IMAGE_H,
    IMAGE_W,
    NUM_VIEWS,
)
from navigation.contracts import canonical_json_bytes
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY

# Every camera looks along ego +X: image x is -Y, image y is -Z.
_CAMERA_FROM_EGO = np.asarray([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
_INTRINSIC = np.asarray([
    [500.0, 0.0, IMAGE_W / 2],
    [0.0, 500.0, IMAGE_H / 2],
    [0.0, 0.0, 1.0],
])
_PROJECTION = np.repeat(
    (_INTRINSIC @ _CAMERA_FROM_EGO[:3])[None],
    NUM_VIEWS,
    axis=0,
)


def test_crop_keeps_a_3d_point_on_the_same_pixel():
    width, height = 1920, 1080
    intrinsic = np.asarray([
        [1000.0, 0.0, 960.0],
        [0.0, 1000.0, 540.0],
        [0.0, 0.0, 1.0],
    ])
    projection = intrinsic @ np.eye(4)[:3]
    point = np.asarray([2.0, -1.0, 10.0, 1.0])
    native = projection @ point
    native_uv = native[:2] / native[2]
    image = Image.new("RGB", (width, height))
    ImageDraw.Draw(image).ellipse(
        [
            native_uv[0] - 6,
            native_uv[1] - 6,
            native_uv[0] + 6,
            native_uv[1] + 6,
        ],
        fill=(255, 255, 255),
    )

    cropped = crop_projection(projection, width, height) @ point
    cropped_uv = cropped[:2] / cropped[2]
    with Image.open(io.BytesIO(encode_camera(np.asarray(image)))) as jpeg:
        assert jpeg.size == (IMAGE_W, IMAGE_H)
        brightness = np.asarray(jpeg.convert("L"), dtype=np.float64)
    rows, cols = np.nonzero(brightness > 128)
    sx, sy, _, _, left, top = crop_params(width, height)

    np.testing.assert_allclose(
        cropped_uv,
        [native_uv[0] * sx - left, native_uv[1] * sy - top],
    )
    np.testing.assert_allclose(
        [cols.mean(), rows.mean()],
        cropped_uv,
        atol=1.0,
    )
    # A [3, 3] intrinsic follows the same transform.
    np.testing.assert_allclose(
        crop_projection(intrinsic, width, height) @ np.eye(4)[:3],
        crop_projection(projection, width, height),
    )


def test_depth_keeps_the_nearest_return_per_pixel_and_cell():
    points = np.asarray([
        [10.0, 0.0, 0.0],    # pixel (672, 384)
        [20.0, 0.0, 0.0],    # same pixel, occluded
        [15.0, -0.03, 0.0],  # pixel (673, 384), same stride cell
        [-5.0, 0.0, 0.0],    # behind the camera
        [130.0, 0.0, 0.0],   # beyond the depth range
    ])

    payload = encode_depth(
        points,
        _PROJECTION,
        np.repeat(_CAMERA_FROM_EGO[None], NUM_VIEWS, axis=0),
    )

    arrays = np.load(io.BytesIO(payload))
    assert arrays["uv_0"].tolist() == [[672, 384], [673, 384]]
    np.testing.assert_allclose(arrays["depth_0"], [10.0, 15.0])
    depth = decode_stride_depth(payload)
    assert depth.shape == (NUM_VIEWS, IMAGE_H // 8, IMAGE_W // 8)
    assert depth[0, 48, 84] == pytest.approx(10.0)
    assert np.count_nonzero(depth[0]) == 1
    # A sweep that misses every camera gives all-zero depth.
    empty = decode_stride_depth(encode_depth(
        points[3:4],
        _PROJECTION,
        np.repeat(_CAMERA_FROM_EGO[None], NUM_VIEWS, axis=0),
    ))
    assert not empty.any()


def test_boxes_round_trip_and_refuse_unknown_classes():
    boxes = np.asarray([
        [10.0, -2.0, 0.8, 4.5, 1.9, 1.6, 0.3, 5.0, 0.0],
        [6.0, 3.0, 0.9, 0.6, 0.6, 1.8, 0.0, 0.0, 1.0],
    ])

    decoded, labels = decode_boxes(encode_boxes(boxes, np.asarray([0, 1])))

    np.testing.assert_allclose(decoded, boxes, rtol=1e-6)
    assert labels.tolist() == [0, 1]
    empty, empty_labels = decode_boxes(None)
    assert empty.shape == (0, len(BOX_COLUMNS))
    assert empty_labels.shape == (0,)
    with pytest.raises(ValueError, match="BOX_CLASSES"):
        encode_boxes(boxes, np.asarray([0, 4]))


def _bev_members() -> dict[str, bytes]:
    jpeg = encode_camera(np.full((IMAGE_H, IMAGE_W, 3), 128, dtype=np.uint8))
    members = {camera_member(view): jpeg for view in range(NUM_VIEWS)}
    for frame in range(HISTORY_FRAMES):
        for view in range(NUM_VIEWS):
            members[history_camera_member(frame, view)] = jpeg
    target = np.zeros((8, BEV_H, BEV_W), dtype=np.float32)
    target[5, 100:110, 70:90] = 1.0
    valid = np.ones_like(target, dtype=np.bool_)
    valid[7] = False
    members.update({
        "projection.npy": encode_npy(_PROJECTION),
        "history_projection.npy": encode_npy(
            np.repeat(_PROJECTION[None], HISTORY_FRAMES, axis=0)
        ),
        "bev_segmentation.npz": encode_bev_segmentation(target, valid),
        "depth.npz": encode_depth(
            np.asarray([[10.0, 0.0, 0.0]]),
            _PROJECTION,
            np.repeat(_CAMERA_FROM_EGO[None], NUM_VIEWS, axis=0),
        ),
        "boxes.npz": encode_boxes(
            np.asarray([[10.0, 0.0, 0.8, 4.5, 1.9, 1.6, 0.0, 0.0, 0.0]]),
            np.asarray([0]),
        ),
    })
    return members


def _planner_members() -> dict[str, bytes]:
    geometry = AUTOE2E_NAVIGATION_GEOMETRY
    map_context = np.zeros(
        (14, geometry.height_px, geometry.width_px),
        dtype=np.float32,
    )
    map_context[0, 100:300, 100:200] = 1.0
    route = np.zeros((2, geometry.height_px, geometry.width_px), np.float32)
    route[0, 100:300, 145:155] = 1.0
    ego = np.zeros(384, dtype=np.float32)
    ego[:256].reshape(64, 4)[:, 0] = 4.0
    trajectory = np.column_stack([
        np.linspace(0.4, 25.6, 64, dtype=np.float32),
        np.zeros(64, dtype=np.float32),
    ])
    return {
        **_bev_members(),
        **encode_reactive_navigation(
            map_context,
            route,
            map_valid=True,
            route_channel_valid=np.ones(2, dtype=np.bool_),
            geometry=geometry,
            metadata={},
        ),
        "ego.npy": ego.tobytes(),
        "trajectory_xy.npz": encode_trajectory_xy(
            trajectory,
            np.ones(64, dtype=np.bool_),
        ),
    }


def _write_pack(
    root: Path,
    samples: dict[str, dict[str, bytes]],
    *,
    sample_kind: str,
    **manifest_fields: object,
) -> Path:
    root.mkdir()
    with tarfile.open(root / "shard-000.tar", "w") as archive:
        for sample_uid, members in samples.items():
            for suffix, payload in sorted(members.items()):
                info = tarfile.TarInfo(f"{sample_uid}.{suffix}")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    (root / "manifest.json").write_text(json.dumps({
        **manifest_contract(),
        "sample_kind": sample_kind,
        "shard_names": ["shard-000.tar"],
        "shard_sample_counts": {"shard-000.tar": len(samples)},
        **manifest_fields,
    }))
    return root


def _meta(sample_uid: str, split_group_uid: str) -> dict[str, bytes]:
    return {"meta.json": canonical_json_bytes({
        "frame_idx": 3,
        "sample_uid": sample_uid,
        "split_group_uid": split_group_uid,
    })}


def _single(root: Path) -> dict:
    (batch,) = list(make_pre_extracted_loader(
        str(root),
        batch_size=1,
        num_workers=0,
        shuffle=0,
    ))
    return batch


def test_loader_decodes_both_sample_kinds_into_one_layout(tmp_path):
    planner = _write_pack(
        tmp_path / "nuplan",
        {"nuplan-a": {**_planner_members(), **_meta("nuplan-a", "g-a")}},
        sample_kind="planner",
    )
    bev_only = _write_pack(
        tmp_path / "nuscenes",
        {"nusc-b": {**_bev_members(), **_meta("nusc-b", "g-b")}},
        sample_kind="autobev_only",
    )

    nuplan = _single(planner)
    nuscenes = _single(bev_only)

    assert set(nuplan) == set(nuscenes)
    for batch, supervised in ((nuplan, True), (nuscenes, False)):
        assert batch["visual_tiles"].shape == (1, 6, 3, 768, 1344)
        assert batch["visual_tiles"].dtype == torch.float32
        assert batch["camera_history_tiles"].shape == (1, 7, 6, 3, 768, 1344)
        assert batch["camera_projection_matrix"].shape == (1, 6, 3, 4)
        assert batch["camera_history_projection_matrix"].shape == (
            1,
            7,
            6,
            3,
            4,
        )
        assert batch["bev_segmentation_target"].shape == (1, 8, 220, 160)
        assert batch["bev_segmentation_valid"].dtype == torch.bool
        assert not batch["bev_segmentation_valid"][0, 7].any()
        assert batch["autobev_depth"].shape == (1, 6, 96, 168)
        assert batch["autobev_depth"][0, 0, 48, 84] == pytest.approx(10.0)
        assert [boxes.shape for boxes in batch["autobev_boxes"]] == [(1, 9)]
        assert batch["autobev_box_labels"][0].dtype == torch.int64
        assert batch["map_context"].shape == (1, 14, 450, 300)
        assert batch["route_mask"].shape == (1, 2, 450, 300)
        assert batch["egomotion_history"].shape == (1, 256)
        assert batch["trajectory_xy_m"].shape == (1, 64, 2)
        assert batch["trajectory_valid"].shape == (1, 64)
        assert batch["planner_supervised"].tolist() == [supervised]
        assert "front_camera_tile" not in batch
        assert "front_camera_projection_matrix" not in batch
    # ImageNet normalization of a mid-grey image.
    assert nuplan["visual_tiles"][0, 0, 0].mean() == pytest.approx(
        (128 / 255 - 0.485) / 0.229,
        abs=0.05,
    )
    assert nuplan["map_valid"].tolist() == [True]
    assert nuplan["trajectory_valid"].all()
    assert nuplan["initial_speed_mps"].tolist() == [4.0]
    assert nuscenes["map_valid"].tolist() == [False]
    assert nuscenes["route_valid"].tolist() == [False]
    assert not nuscenes["map_context"].any()
    assert not nuscenes["egomotion_history"].any()
    assert not nuscenes["trajectory_xy_m"].any()
    assert not nuscenes["trajectory_valid"].any()
    assert nuscenes["initial_speed_mps"].tolist() == [0.0]


def test_collate_keeps_variable_box_counts_as_lists():
    samples = [
        {
            "value": torch.tensor(1.0),
            "autobev_boxes": torch.zeros(count, 9),
            "autobev_box_labels": torch.zeros(count, dtype=torch.int64),
        }
        for count in (0, 3)
    ]

    batch = collate_autobev_batch(samples)

    assert batch["value"].shape == (2,)
    assert [boxes.shape[0] for boxes in batch["autobev_boxes"]] == [0, 3]
    assert [labels.shape[0] for labels in batch["autobev_box_labels"]] == [0, 3]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"autobev_geometry": {
                **AUTOBEV_CONTRACT,
                "pc_range": [-30.0, -40.0, -5.0, 80.0, 40.0, 3.0],
            }},
            "different AutoBEV geometry",
        ),
        ({"box_classes": ["VEHICLE"]}, "member contract"),
        ({"sample_kind": "planner_v0"}, "sample_kind"),
    ],
)
def test_loader_refuses_another_autobev_contract(tmp_path, change, message):
    root = _write_pack(
        tmp_path / "pack",
        {"a": _meta("a", "g")},
        sample_kind="planner",
    )
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.update(change)
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        make_pre_extracted_loader(str(root), num_workers=0)


def _light_decode(sample: dict, *, planner_supervised: bool) -> dict:
    return {
        "sample_uid": sample["__key__"],
        "planner_supervised": torch.tensor(planner_supervised),
        "autobev_boxes": torch.zeros(0, 9),
        "autobev_box_labels": torch.zeros(0, dtype=torch.int64),
    }


def _light_pack(
    root: Path,
    prefix: str,
    count: int,
    *,
    sample_kind: str,
    group=lambda index: "group",
) -> Path:
    return _write_pack(
        root,
        {
            f"{prefix}-{index:04d}": _meta(f"{prefix}-{index:04d}", group(index))
            for index in range(count)
        },
        sample_kind=sample_kind,
    )


def _uids(loader) -> list[str]:
    return [
        uid
        for batch, _, _ in loader
        for uid in batch["sample_uid"]
    ]


def test_pittsburgh_groups_never_reach_the_train_split(tmp_path, monkeypatch):
    monkeypatch.setattr(pre_extracted, "_decode_autobev_sample", _light_decode)
    root = _light_pack(
        tmp_path / "nuplan",
        "nuplan",
        40,
        sample_kind="planner",
        group=lambda index: (
            f"heldout-nuplan-log-{index}"
            if index % 4 == 0
            else f"nuplan-log-{index}"
        ),
    )

    def uids(split: str) -> set[str]:
        return {
            uid
            for batch in make_pre_extracted_loader(
                str(root),
                batch_size=4,
                num_workers=0,
                split=split,
                val_fraction=0.1,
                shuffle=0,
            )
            for uid in batch["sample_uid"]
        }

    held_out = {f"nuplan-{index:04d}" for index in range(0, 40, 4)}
    assert not held_out & uids("train")
    assert held_out <= uids("val")


def test_bev_share_loader_draws_the_bev_only_share(tmp_path, monkeypatch):
    monkeypatch.setattr(pre_extracted, "_decode_autobev_sample", _light_decode)
    primary = _light_pack(tmp_path / "nuplan", "nuplan", 850, sample_kind="planner")
    bev_only = [
        _light_pack(tmp_path / "nuscenes", "nusc", 30, sample_kind="autobev_only"),
        _light_pack(tmp_path / "av2", "av2", 10, sample_kind="autobev_only"),
    ]
    random.seed(0)

    loader = make_autobev_bev_share_loader(
        [str(primary)],
        [str(path) for path in bev_only],
        batch_size=10,
        num_workers=0,
        shuffle=0,
    )
    uids = _uids(loader)

    primary_uids = [uid for uid in uids if uid.startswith("nuplan-")]
    # One epoch is one pass over the primary pack; BEV-only packs repeat.
    assert sorted(primary_uids) == [f"nuplan-{index:04d}" for index in range(850)]
    share = 1.0 - len(primary_uids) / len(uids)
    assert share == pytest.approx(AUTOBEV_BEV_ONLY_SHARE, abs=0.03)
    nusc = sum(uid.startswith("nusc-") for uid in uids)
    av2 = sum(uid.startswith("av2-") for uid in uids)
    # BEV-only packs are weighted by their sample counts.
    assert nusc > 2 * av2 > 0


def test_sample_count_loader_reads_every_pack_once(tmp_path, monkeypatch):
    monkeypatch.setattr(pre_extracted, "_decode_autobev_sample", _light_decode)
    packs = [
        _light_pack(tmp_path / "nuplan", "nuplan", 60, sample_kind="planner"),
        _light_pack(tmp_path / "once", "once", 20, sample_kind="autobev_only"),
    ]
    random.seed(0)

    loader = make_autobev_sample_count_loader(
        [str(path) for path in packs],
        batch_size=8,
        num_workers=0,
        shuffle=0,
    )
    batches = list(loader)
    uids = [uid for batch, _, _ in batches for uid in batch["sample_uid"]]

    assert sorted(uids) == sorted(
        [f"nuplan-{index:04d}" for index in range(60)]
        + [f"once-{index:04d}" for index in range(20)]
    )
    # Proportional draws spread the small pack over the epoch.
    first_half = uids[:40]
    assert 5 <= sum(uid.startswith("once-") for uid in first_half) <= 15
    assert all(
        geometry_type == "rectified_pinhole" and projection is None
        for _, projection, geometry_type in batches
    )
    mixed = next(
        batch for batch, _, _ in batches
        if len(set(batch["planner_supervised"].tolist())) == 2
    )
    assert mixed["planner_supervised"].dtype == torch.bool
