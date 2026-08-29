"""Deterministic packed data for the real-model Reactive DDP canary."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from data_parsing.camera_slots import CANONICAL_SIX_CAMERA_SLOTS
from data_parsing.l2d.calibration import l2d_projection_spec
from data_parsing.l2d.camera import CAMERA_NAMES as L2D_CAMERA_NAMES
from data_parsing.nuplan.packing import NUPLAN_CAMERA_CHANNELS
from data_processing.dataset_snapshot import split_bucket
from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_MEMBER,
    BEV_SEGMENTATION_STATS_MEMBER,
    BEV_SEGMENTATION_TAXONOMY_VERSION,
    encode_bev_segmentation,
    encode_bev_segmentation_stats,
    encode_reactive_navigation,
    encode_trajectory_xy,
)
from navigation.contracts import canonical_json_bytes
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY
from reactive_training_contracts import (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    REACTIVE_BEVFORMER_FRAME_OFFSETS,
    REACTIVE_BEVFORMER_HISTORY_FRAMES,
    REACTIVE_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_INDEX,
)
from training.dataset_policy import (
    L2D_DATASET_NAME,
    NUPLAN_DATASET_NAME,
)
from training.reactive_multitask import ReactiveTrainingStage


REACTIVE_CANARY_SCHEMA_VERSION = "reactive_ddp_canary_v8"


def _jpeg_bytes(
    color: tuple[int, int, int],
    *,
    image_size: int = REACTIVE_CAMERA_IMAGE_SIZE,
) -> bytes:
    output = io.BytesIO()
    Image.new(
        "RGB",
        (image_size, image_size),
        color=color,
    ).save(
        output,
        format="JPEG",
        quality=90,
        optimize=False,
        progressive=False,
    )
    return output.getvalue()


def _split_group(*, validation: bool, index: int) -> str:
    for attempt in range(10_000):
        candidate = (
            f"canary-{'val' if validation else 'train'}-"
            f"{index:04d}-{attempt:04d}"
        )
        in_validation = split_bucket(candidate, 10) < 5
        if in_validation is validation:
            return candidate
    raise RuntimeError("could not construct deterministic canary split group")


def _add_member(
    archive: tarfile.TarFile,
    *,
    sample_uid: str,
    suffix: str,
    payload: bytes,
) -> None:
    info = tarfile.TarInfo(name=f"{sample_uid}.{suffix}")
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def _sample_members(
    *,
    stage: ReactiveTrainingStage,
    sample_uid: str,
    sample_index: int,
    split_group_uid: str,
) -> dict[str, bytes]:
    geometry = AUTOE2E_NAVIGATION_GEOMETRY
    height = geometry.height_px
    width = geometry.width_px
    num_views = len(CANONICAL_SIX_CAMERA_SLOTS)
    map_context = np.zeros((14, height, width), dtype=np.float32)
    map_context[0, 90:360, 80:220] = 1.0
    map_context[1, 100:350, 110:190] = 1.0
    route = np.zeros((2, height, width), dtype=np.float32)
    route[0, 100:350, 145:155] = 1.0
    route[1, 105:116, 145:156] = 1.0
    navigation = encode_reactive_navigation(
        map_context,
        route,
        map_valid=True,
        route_channel_valid=np.ones(2, dtype=np.bool_),
        geometry=geometry,
        metadata={
            "canary": True,
            "sample_index": sample_index,
        },
    )
    trajectory = np.column_stack([
        np.linspace(0.2, 12.8, 64, dtype=np.float32),
        np.zeros(64, dtype=np.float32),
    ])
    ego = np.zeros(384, dtype=np.float32)
    ego[:256].reshape(64, 4)[:, 0] = 2.0
    color = (
        32 + sample_index % 128,
        96 + sample_index % 96,
        160 + sample_index % 64,
    )
    dataset = (
        NUPLAN_DATASET_NAME
        if stage is ReactiveTrainingStage.NUPLAN_FULL
        else L2D_DATASET_NAME
    )
    members: dict[str, bytes] = {
        **navigation,
        "ego.npy": ego.tobytes(),
        "meta.json": canonical_json_bytes({
            "canary": True,
            "dataset": dataset,
            "frame_idx": sample_index,
            "sample_uid": sample_uid,
            "split_group_uid": split_group_uid,
        }),
        "trajectory_xy.npz": encode_trajectory_xy(
            trajectory,
            np.ones(64, dtype=np.bool_),
        ),
    }
    for view in range(num_views):
        image_size = (
            REACTIVE_FRONT_CAMERA_IMAGE_SIZE
            if (
                stage is ReactiveTrainingStage.NUPLAN_FULL
                and view == REACTIVE_FRONT_CAMERA_INDEX
            )
            else REACTIVE_CAMERA_IMAGE_SIZE
        )
        members[f"cam_{view}.jpg"] = _jpeg_bytes(
            color,
            image_size=image_size,
        )
    if stage is ReactiveTrainingStage.NUPLAN_FULL:
        projection = np.zeros((num_views, 3, 4), dtype=np.float32)
        projection[:, 0, 3] = REACTIVE_CAMERA_IMAGE_SIZE / 2
        projection[:, 1, 3] = REACTIVE_CAMERA_IMAGE_SIZE / 2
        projection[:, 2, 3] = 1.0
        front_projection = projection[
            REACTIVE_FRONT_CAMERA_INDEX:
            REACTIVE_FRONT_CAMERA_INDEX + 1
        ].copy()
        front_projection[:, :2] *= (
            REACTIVE_FRONT_CAMERA_IMAGE_SIZE
            / REACTIVE_CAMERA_IMAGE_SIZE
        )
        history_projection = np.repeat(
            projection[None],
            REACTIVE_BEVFORMER_HISTORY_FRAMES,
            axis=0,
        )
        for history_index in range(
            REACTIVE_BEVFORMER_HISTORY_FRAMES
        ):
            for view in range(num_views):
                members[
                    f"bev_hist_{history_index}_cam_{view}.jpg"
                ] = _jpeg_bytes(
                    color,
                    image_size=REACTIVE_CAMERA_IMAGE_SIZE,
                )
        members["calib.json"] = canonical_json_bytes({
            "camera_order": list(NUPLAN_CAMERA_CHANNELS),
            "camera_slots": list(CANONICAL_SIX_CAMERA_SLOTS),
            "dataset": dataset,
            "front_camera_image_size": (
                REACTIVE_FRONT_CAMERA_IMAGE_SIZE
            ),
            "front_camera_index": REACTIVE_FRONT_CAMERA_INDEX,
            "front_projection": {
                "matrix": front_projection.tolist(),
                "type": "rectified_pinhole",
            },
            "geometry_type": "rectified_pinhole",
            "history_projection": {
                "matrix": history_projection.tolist(),
                "reference_frame": "current_ego",
                "type": "rectified_pinhole",
            },
            "image_size": REACTIVE_CAMERA_IMAGE_SIZE,
            "projection": {
                "matrix": projection.tolist(),
                "type": "rectified_pinhole",
            },
            "temporal_frame_interval_us": (
                REACTIVE_BEVFORMER_FRAME_INTERVAL_US
            ),
            "temporal_frame_offsets": list(
                REACTIVE_BEVFORMER_FRAME_OFFSETS
            ),
        })
        bev = np.zeros((8, height, width), dtype=np.float32)
        bev[0] = map_context[0]
        bev[1, 100:350:20, 110:190] = 1.0
        bev[2, 180:260, 110:190] = 1.0
        bev[3, 240:260, 125:175] = 1.0
        bev[4, 278:282, 110:190] = 1.0
        bev[5, 210:230, 140:160] = 1.0
        bev[6, 190:198, 130:138] = 1.0
        bev[7, 165:177, 158:170] = 1.0
        valid = np.ones_like(bev, dtype=np.bool_)
        members[BEV_SEGMENTATION_MEMBER] = encode_bev_segmentation(
            bev,
            valid,
        )
        members[BEV_SEGMENTATION_STATS_MEMBER] = (
            encode_bev_segmentation_stats(
                bev,
                valid,
            )
        )
    else:
        members["calib.json"] = canonical_json_bytes({
            "camera_order": list(L2D_CAMERA_NAMES),
            "camera_slots": list(CANONICAL_SIX_CAMERA_SLOTS),
            "dataset": dataset,
            "geometry_type": "pinhole",
            "image_size": REACTIVE_CAMERA_IMAGE_SIZE,
        })
    return members


def write_reactive_canary_dataset(
    output_directory: str | Path,
    *,
    stage: ReactiveTrainingStage,
    shard_count: int = 2,
    train_samples_per_shard: int = 2,
    validation_samples_per_shard: int = 1,
) -> dict[str, Any]:
    """Write a tiny but production-schema-compatible Reactive corpus."""
    if shard_count < 2:
        raise ValueError("Reactive DDP canary needs at least two shards")
    if train_samples_per_shard <= 0 or validation_samples_per_shard <= 0:
        raise ValueError("each canary shard needs train and validation data")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("canary output directory must be empty")

    shard_names = []
    shard_sample_counts: dict[str, int] = {}
    sample_index = 0
    for shard_index in range(shard_count):
        shard_name = f"canary-{shard_index:03d}.tar"
        shard_names.append(shard_name)
        count = train_samples_per_shard + validation_samples_per_shard
        shard_sample_counts[shard_name] = count
        with tarfile.open(output / shard_name, mode="w") as archive:
            for offset in range(count):
                validation = offset >= train_samples_per_shard
                split_group_uid = _split_group(
                    validation=validation,
                    index=sample_index,
                )
                sample_uid = (
                    f"canary-{stage.value}-{sample_index:06d}"
                )
                members = _sample_members(
                    stage=stage,
                    sample_uid=sample_uid,
                    sample_index=sample_index,
                    split_group_uid=split_group_uid,
                )
                for suffix, payload in sorted(members.items()):
                    _add_member(
                        archive,
                        sample_uid=sample_uid,
                        suffix=suffix,
                        payload=payload,
                    )
                sample_index += 1

    shard_sha256 = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in shard_names
    }
    dataset = (
        NUPLAN_DATASET_NAME
        if stage is ReactiveTrainingStage.NUPLAN_FULL
        else L2D_DATASET_NAME
    )
    manifest: dict[str, Any] = {
        "bev_segmentation_count": (
            sample_index
            if stage is ReactiveTrainingStage.NUPLAN_FULL
            else 0
        ),
        "bev_statistics_count": (
            sample_index
            if stage is ReactiveTrainingStage.NUPLAN_FULL
            else 0
        ),
        "bev_taxonomy_version": (
            BEV_SEGMENTATION_TAXONOMY_VERSION
            if stage is ReactiveTrainingStage.NUPLAN_FULL
            else None
        ),
        "camera_order": (
            list(NUPLAN_CAMERA_CHANNELS)
            if stage is ReactiveTrainingStage.NUPLAN_FULL
            else list(L2D_CAMERA_NAMES)
        ),
        "camera_slots": list(CANONICAL_SIX_CAMERA_SLOTS),
        "dataset": dataset,
        "dataset_version": REACTIVE_CANARY_SCHEMA_VERSION,
        "geometry_type": (
            "rectified_pinhole"
            if stage is ReactiveTrainingStage.NUPLAN_FULL
            else "pinhole"
        ),
        "has_bev_segmentation": (
            stage is ReactiveTrainingStage.NUPLAN_FULL
        ),
        "has_reactive_navigation": True,
        "has_route_reconstruction": True,
        "has_trajectory_xy": True,
        "image_size": REACTIVE_CAMERA_IMAGE_SIZE,
        "map_context_channels": 14,
        "navigation_geometry": (
            AUTOE2E_NAVIGATION_GEOMETRY.contract()
        ),
        "num_views": len(CANONICAL_SIX_CAMERA_SLOTS),
        "partition_id": f"canary-{stage.value}",
        "route_channels": 2,
        "schema_version": REACTIVE_CANARY_SCHEMA_VERSION,
        "shard_names": shard_names,
        "shard_sample_counts": shard_sample_counts,
        "shard_sha256": shard_sha256,
        "source_revision": REACTIVE_CANARY_SCHEMA_VERSION,
        "total_samples": sample_index,
    }
    if stage is ReactiveTrainingStage.NUPLAN_FULL:
        manifest.update({
            "front_camera_image_size": (
                REACTIVE_FRONT_CAMERA_IMAGE_SIZE
            ),
            "front_camera_index": REACTIVE_FRONT_CAMERA_INDEX,
            "temporal_frame_interval_us": (
                REACTIVE_BEVFORMER_FRAME_INTERVAL_US
            ),
            "temporal_frame_offsets": list(
                REACTIVE_BEVFORMER_FRAME_OFFSETS
            ),
        })
    else:
        manifest["projection"] = l2d_projection_spec(
            REACTIVE_CAMERA_IMAGE_SIZE
        )
    (output / "manifest.json").write_text(
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    return manifest
