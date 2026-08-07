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

from data_processing.dataset_snapshot import split_bucket
from data_processing.reactive_training_artifacts import (
    encode_bev_segmentation,
    encode_reactive_navigation,
    encode_trajectory_xy,
)
from navigation.contracts import canonical_json_bytes
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY
from training.dataset_policy import (
    L2D_DATASET_NAME,
    NUPLAN_DATASET_NAME,
)
from training.reactive_multitask import ReactiveTrainingStage


REACTIVE_CANARY_SCHEMA_VERSION = "reactive_ddp_canary_v1"


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (256, 256), color=color).save(
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
    sample_index: int,
    split_group_uid: str,
) -> dict[str, bytes]:
    geometry = AUTOE2E_NAVIGATION_GEOMETRY
    height = geometry.height_px
    width = geometry.width_px
    num_views = (
        8 if stage is ReactiveTrainingStage.NUPLAN_FULL else 6
    )
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
    jpeg = _jpeg_bytes(color)
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
            "split_group_uid": split_group_uid,
        }),
        "trajectory_xy.npz": encode_trajectory_xy(
            trajectory,
            np.ones(64, dtype=np.bool_),
        ),
    }
    for view in range(num_views):
        members[f"cam_{view}.jpg"] = jpeg
    if stage is ReactiveTrainingStage.NUPLAN_FULL:
        bev = np.zeros((8, height, width), dtype=np.float32)
        bev[0] = map_context[0]
        bev[1] = map_context[1]
        members["bev_segmentation.npz"] = encode_bev_segmentation(
            bev,
            np.ones_like(bev, dtype=np.bool_),
        )
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
        "dataset": dataset,
        "dataset_version": REACTIVE_CANARY_SCHEMA_VERSION,
        "geometry_type": "pseudo",
        "has_bev_segmentation": (
            stage is ReactiveTrainingStage.NUPLAN_FULL
        ),
        "has_reactive_navigation": True,
        "has_route_reconstruction": True,
        "has_trajectory_xy": True,
        "map_context_channels": 14,
        "navigation_geometry": (
            AUTOE2E_NAVIGATION_GEOMETRY.contract()
        ),
        "num_views": (
            8 if stage is ReactiveTrainingStage.NUPLAN_FULL else 6
        ),
        "partition_id": f"canary-{stage.value}",
        "route_channels": 2,
        "schema_version": REACTIVE_CANARY_SCHEMA_VERSION,
        "shard_names": shard_names,
        "shard_sample_counts": shard_sample_counts,
        "shard_sha256": shard_sha256,
        "source_revision": REACTIVE_CANARY_SCHEMA_VERSION,
        "total_samples": sample_index,
    }
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
