"""PreExtractedDataset: WebDataset-backed DataLoader for training.

Reads from local EBS shard cache (init container syncs from S3).
No video decode, no lerobot dependency. Sequential tar reads at full
disk bandwidth.

Usage:
    from data_parsing.pre_extracted import make_pre_extracted_loader

    loader = make_pre_extracted_loader("/data/shards", batch_size=8)
    for batch in loader:
        # batch["visual_tiles"]       (B, V, 3, H, W)      base camera tensors
        # batch["front_camera_tile"]  (B, 3, Hf, Wf)       optional native front
        # batch["front_camera_fpn_tile"] (B, 3, H, W)      optional cache front
        # batch["map_context"]        (B, C_map, 256, 256) semantic map
        # batch["route_mask"]         (B, 2, 256, 256)     selected route
        # batch["map_valid"]          (B,)                  explicit validity
        # batch["route_valid"]        (B,)                  explicit validity
        # batch["route_supervision"]  dict of loss-only route fields
        # batch["source_frame_index"] (B,)                  source sequence index
        # batch["source_frame_interval_us"] (B,)            source cadence
        # batch["egomotion_history"]  (B, 256)
        # batch["visual_history"]     (B, 896)
        # batch["trajectory_target"]  (B, 128)
        # batch["camera_projection_matrix"] (B, V, 3, 4)   when packed per sample

Packs with the AutoBEV member contract (nuPlan and the BEV-only datasets)
decode into one layout instead: ``visual_tiles`` [B, 6, 3, 768, 1344],
``camera_history_tiles`` [B, 7, 6, 3, 768, 1344], projections, BEV labels on
the 220x160 AutoBEV grid, ``autobev_depth`` [B, 6, 96, 168], box lists, the
0.4 m navigation inputs (zeros with invalid flags for BEV-only samples), and
``planner_supervised``. ``make_autobev_bev_share_loader`` and
``make_autobev_sample_count_loader`` mix such packs per sample.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import math
import os
import random
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from PIL import Image
from reactive_training_contracts import (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    REACTIVE_BEVFORMER_FRAME_OFFSETS,
    REACTIVE_BEVFORMER_HISTORY_FRAMES,
    REACTIVE_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_INDEX,
)
from torchvision import transforms

# Stage 2 draws this share of its samples from the BEV-only packs so the
# encoder keeps what it learned on the other rigs (autobev_integration 3.3).
AUTOBEV_BEV_ONLY_SHARE: Final = 0.15
# Every AutoBEV projection is a pinhole [3, 4] matrix to the cropped pixels.
AUTOBEV_GEOMETRY_TYPE: Final = "rectified_pinhole"
# Box counts differ per sample, so a batch keeps these as lists of tensors.
AUTOBEV_VARIABLE_LENGTH_KEYS: Final = ("autobev_boxes", "autobev_box_labels")
# Stable split buckets; 100 lets a 2 % calibration split exist without moving
# a tenth of the training data out of training.
SPLIT_BUCKETS: Final = 100

_HISTORY_STEPS = 64
_FUTURE_STEPS = 64
_HISTORY_SIGNALS = 4
_TARGET_SIGNALS = 2
_VISUAL_HISTORY_DIM = 896

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Camera frames are keyed "cam_<i>.jpg"; the nav-map is "map.jpg". Matching cam_
# explicitly keeps the map out of the camera views, so V stays correct and the
# map is not projected into BEV.
_CAM_KEY_RE = re.compile(r"^cam_\d+\.jpg$")
_BEV_HIST_KEY_RE = re.compile(r"^bev_hist_(\d+)_cam_(\d+)\.jpg$")
_KITSCENES_HISTORY_FRAME_ID_RE = re.compile(r"-r(\d+)-c(\d+)$")
_BEV_HISTORY_INDEX_MEMBER = "bev_history_index.json"
# World-Model window frames: hist_<t>_cam_<v>.jpg / fut_<f>_cam_<v>.jpg.
_HIST_KEY_RE = re.compile(r"^hist_(\d+)_cam_(\d+)\.jpg$")
_FUT_KEY_RE = re.compile(r"^fut_(\d+)_cam_(\d+)\.jpg$")

NAVIGATION_REPEAT_POLICY_VERSION = "navigation_repeat_v1"
BEV_CLASS_REPEAT_POLICY_VERSION = "bev_class_repeat_v3"
_DECISIVE_ROUTE_MANEUVERS = frozenset({
    "left",
    "right",
    "u_turn",
    "merge",
    "exit",
})


def _source_frame_interval_us_from_history_index(
    history_index: object,
    *,
    current_frame_index: int,
) -> int:
    """Infer the packed source cadence from an exact KITScenes T8 index."""
    if (
        current_frame_index < 0
        or not isinstance(history_index, list)
        or len(history_index) != REACTIVE_BEVFORMER_HISTORY_FRAMES
    ):
        return -1
    rows = []
    for frame in history_index:
        if not isinstance(frame, list) or not frame:
            return -1
        frame_rows = []
        for camera_index, frame_id in enumerate(frame):
            match = _KITSCENES_HISTORY_FRAME_ID_RE.search(str(frame_id))
            if match is None or int(match.group(2)) != camera_index:
                return -1
            frame_rows.append(int(match.group(1)))
        if len(set(frame_rows)) != 1:
            return -1
        rows.append(frame_rows[0])
    differences = [
        current - previous
        for previous, current in zip(rows, rows[1:])
    ]
    if not differences or len(set(differences)) != 1:
        return -1
    row_stride = differences[0]
    if (
        row_stride <= 0
        or current_frame_index - rows[-1] != row_stride
        or REACTIVE_BEVFORMER_FRAME_INTERVAL_US % row_stride != 0
    ):
        return -1
    return REACTIVE_BEVFORMER_FRAME_INTERVAL_US // row_stride


def passthrough_nodesplitter(urls: Iterable[str]) -> Iterable[str]:
    """Keep every URL when the caller already assigned shards to this rank."""
    yield from urls


def packed_sample_tar_paths(
    shard_dir: str | Path,
) -> tuple[Path, ...]:
    """Return only sample shards, excluding sibling transport archives."""
    root = Path(shard_dir)
    manifest_path = root / "manifest.json"
    rank_inventory_path = root / "rank_shards.json"
    archive_name = "frame_pool.tar"
    if manifest_path.is_file():
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError(
                f"invalid packed manifest at {manifest_path}"
            ) from error
        if not isinstance(manifest, Mapping):
            raise ValueError(
                f"packed manifest must be an object at {manifest_path}"
            )
        configured_archive = manifest.get("frame_pool_archive")
        if configured_archive is not None:
            if (
                not isinstance(configured_archive, str)
                or not configured_archive
                or Path(configured_archive).name != configured_archive
            ):
                raise ValueError(
                    f"invalid frame-pool archive name at {manifest_path}"
                )
            archive_name = configured_archive
        configured_shards = manifest.get("shard_names")
        if configured_shards is not None:
            if (
                not isinstance(configured_shards, list)
                or any(
                    not isinstance(name, str)
                    or not name.endswith(".tar")
                    or Path(name).name != name
                    or name == archive_name
                    for name in configured_shards
                )
                or len(set(configured_shards)) != len(configured_shards)
            ):
                raise ValueError(
                    f"invalid sample shard inventory at {manifest_path}"
                )
            selected_shards = configured_shards
            if rank_inventory_path.is_file():
                try:
                    rank_inventory = json.loads(
                        rank_inventory_path.read_text(encoding="ascii")
                    )
                except (OSError, UnicodeError, ValueError) as error:
                    raise ValueError(
                        f"invalid rank shard inventory at {rank_inventory_path}"
                    ) from error
                if not isinstance(rank_inventory, Mapping):
                    raise ValueError(
                        "rank shard inventory must be an object"
                    )
                rank_shards = rank_inventory.get("shard_names")
                if (
                    rank_inventory.get("schema_version")
                    != "reactive_rank_shards_v1"
                    or rank_inventory.get("manifest_sha256")
                    != hashlib.sha256(manifest_bytes).hexdigest()
                    or not isinstance(rank_shards, list)
                    or any(
                        not isinstance(name, str)
                        or name not in configured_shards
                        for name in rank_shards
                    )
                    or not rank_shards
                    or len(set(rank_shards)) != len(rank_shards)
                ):
                    raise ValueError(
                        f"invalid rank shard inventory at {rank_inventory_path}"
                    )
                selected_shards = rank_shards
            paths = tuple(root / name for name in selected_shards)
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"packed sample shards are missing: {missing}"
                )
            return paths
    return tuple(
        path
        for path in sorted(root.glob("*.tar"))
        if path.name != archive_name
    )


def _json_mapping(value, *, member_name: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(
            value.decode()
            if isinstance(value, (bytes, bytearray))
            else value
        )
    except (TypeError, UnicodeError, ValueError) as error:
        raise ValueError(f"invalid {member_name}") from error
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{member_name} must contain a JSON object")
    return decoded


@dataclass(frozen=True)
class NavigationRepeatPolicy:
    """Deterministic raw-sample exposure policy for route-choice training."""

    version: str = NAVIGATION_REPEAT_POLICY_VERSION
    turn_repeat: int = 4
    junction_repeat: int = 2
    max_repeat: int = 4

    def __post_init__(self) -> None:
        if self.version != NAVIGATION_REPEAT_POLICY_VERSION:
            raise ValueError(
                f"unsupported navigation repeat policy {self.version!r}"
            )
        if not (
            1 <= self.junction_repeat <= self.max_repeat
            and 1 <= self.turn_repeat <= self.max_repeat
        ):
            raise ValueError(
                "navigation repeat counts must be between one and max_repeat"
            )

    def metadata(self) -> dict[str, object]:
        return {
            "version": self.version,
            "turn_repeat": self.turn_repeat,
            "junction_repeat": self.junction_repeat,
            "max_repeat": self.max_repeat,
            "decisive_maneuvers": sorted(_DECISIVE_ROUTE_MANEUVERS),
        }

    def repeat_count(self, navigation_metadata: Mapping[str, object]) -> int:
        if not bool(navigation_metadata.get("route_valid", False)):
            return 1
        repeat = 1
        maneuver = str(
            navigation_metadata.get("route_maneuver", "unknown")
        ).lower()
        if maneuver in _DECISIVE_ROUTE_MANEUVERS:
            repeat = max(repeat, self.turn_repeat)
        if bool(navigation_metadata.get("route_intersection", False)):
            repeat = max(repeat, self.junction_repeat)
        return min(repeat, self.max_repeat)

    def __call__(self, source: Iterable[dict]):
        """Repeat samples before shuffle and image decoding."""
        for sample in source:
            navigation_data = sample.get("navigation_meta.json")
            metadata = (
                _json_mapping(
                    navigation_data,
                    member_name="navigation_meta.json",
                )
                if navigation_data is not None
                else {}
            )
            for _ in range(self.repeat_count(metadata)):
                yield sample


@dataclass(frozen=True)
class NavigationExposureAudit:
    """Header-only audit of the effective navigation training distribution."""

    policy: Mapping[str, object]
    unique_sample_count: int
    effective_exposure_count: int
    route_valid_sample_count: int
    maneuver_unique_counts: Mapping[str, int]
    maneuver_exposure_counts: Mapping[str, int]
    junction_unique_counts: Mapping[str, int]
    junction_exposure_counts: Mapping[str, int]
    exposure_digest: str

    def metadata(self) -> dict[str, object]:
        return {
            "policy": dict(self.policy),
            "unique_sample_count": self.unique_sample_count,
            "effective_exposure_count": self.effective_exposure_count,
            "route_valid_sample_count": self.route_valid_sample_count,
            "maneuver_unique_counts": dict(self.maneuver_unique_counts),
            "maneuver_exposure_counts": dict(
                self.maneuver_exposure_counts
            ),
            "junction_unique_counts": dict(self.junction_unique_counts),
            "junction_exposure_counts": dict(
                self.junction_exposure_counts
            ),
            "exposure_digest": self.exposure_digest,
        }


def discover_navigation_exposure(
    shard_dirs: Sequence[str | Path],
    *,
    policy: NavigationRepeatPolicy,
    validation_group_uids: Sequence[str] | None = None,
) -> NavigationExposureAudit:
    """Audit train exposure from tar JSON members without decoding tensors."""
    import tarfile

    roots = [Path(shard_dir) for shard_dir in shard_dirs]
    if not roots:
        raise ValueError("at least one shard directory is required")
    validation_groups = frozenset(
        str(uid) for uid in (validation_group_uids or ())
    )
    if validation_group_uids is not None and (
        not validation_groups
        or len(validation_groups) != len(validation_group_uids)
        or any(not uid for uid in validation_groups)
    ):
        raise ValueError(
            "validation_group_uids must contain unique non-empty values"
        )

    records: dict[str, dict[str, Mapping[str, object]]] = {}
    for root in roots:
        tarfiles = packed_sample_tar_paths(root)
        if not tarfiles:
            raise FileNotFoundError(f"No .tar shards found in {root}")
        for tar_path in tarfiles:
            with tarfile.open(tar_path, "r:*") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    if member.name.endswith(".navigation_meta.json"):
                        suffix = ".navigation_meta.json"
                        record_key = "navigation"
                    elif member.name.endswith(".meta.json"):
                        suffix = ".meta.json"
                        record_key = "sample"
                    else:
                        continue
                    sample_uid = member.name.removesuffix(suffix)
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError(
                            f"could not read {member.name} from {tar_path}"
                        )
                    metadata = _json_mapping(
                        extracted.read(),
                        member_name=f"{member.name} in {tar_path}",
                    )
                    record = records.setdefault(sample_uid, {})
                    if record_key in record:
                        raise ValueError(
                            f"duplicate {member.name} for {sample_uid!r}"
                        )
                    record[record_key] = metadata

    repeat_records: list[tuple[str, int]] = []
    maneuver_unique: dict[str, int] = {}
    maneuver_exposure: dict[str, int] = {}
    junction_unique = {"junction": 0, "non_junction": 0}
    junction_exposure = {"junction": 0, "non_junction": 0}
    route_valid_count = 0
    for sample_uid, record in sorted(records.items()):
        if set(record) != {"sample", "navigation"}:
            raise ValueError(
                f"sample {sample_uid!r} lacks sample or navigation metadata"
            )
        sample_metadata = record["sample"]
        if sample_metadata.get("sample_uid") != sample_uid:
            raise ValueError(
                f"sample metadata UID differs for {sample_uid!r}"
            )
        group_uid = sample_metadata.get("split_group_uid")
        if not isinstance(group_uid, str) or not group_uid:
            raise ValueError(
                f"sample {sample_uid!r} has no split_group_uid"
            )
        if group_uid in validation_groups:
            continue

        navigation_metadata = record["navigation"]
        repeat = policy.repeat_count(navigation_metadata)
        repeat_records.append((sample_uid, repeat))
        route_valid = bool(
            navigation_metadata.get("route_valid", False)
        )
        route_valid_count += int(route_valid)
        maneuver = (
            str(navigation_metadata.get("route_maneuver", "unknown")).lower()
            if route_valid
            else "route_invalid"
        )
        junction = (
            "junction"
            if route_valid
            and bool(
                navigation_metadata.get("route_intersection", False)
            )
            else "non_junction"
        )
        maneuver_unique[maneuver] = maneuver_unique.get(maneuver, 0) + 1
        maneuver_exposure[maneuver] = (
            maneuver_exposure.get(maneuver, 0) + repeat
        )
        junction_unique[junction] += 1
        junction_exposure[junction] += repeat

    if not repeat_records:
        raise ValueError("navigation exposure audit selected no train samples")
    digest_payload = "".join(
        f"{sample_uid}\t{repeat}\n"
        for sample_uid, repeat in repeat_records
    ).encode("utf-8")
    return NavigationExposureAudit(
        policy=policy.metadata(),
        unique_sample_count=len(repeat_records),
        effective_exposure_count=sum(
            repeat for _, repeat in repeat_records
        ),
        route_valid_sample_count=route_valid_count,
        maneuver_unique_counts=dict(sorted(maneuver_unique.items())),
        maneuver_exposure_counts=dict(sorted(maneuver_exposure.items())),
        junction_unique_counts=junction_unique,
        junction_exposure_counts=junction_exposure,
        exposure_digest=hashlib.sha256(digest_payload).hexdigest(),
    )


@dataclass(frozen=True)
class BEVTrainingStatistics:
    """Exact train-split class counts from packed BEV metadata."""

    sample_count: int
    effective_exposure_count: int
    active_sample_count: tuple[int, ...]
    positive_sample_count: tuple[int, ...]
    positive_cell_count: tuple[int, ...]
    positive_fraction_sum: tuple[float, ...]
    positive_mass: tuple[float, ...]
    valid_cell_count: tuple[int, ...]
    exposure_digest: str

    def __post_init__(self) -> None:
        from data_processing.reactive_training_artifacts import (
            BEV_SEGMENTATION_CLASSES,
        )

        class_count = len(BEV_SEGMENTATION_CLASSES)
        vectors = (
            self.active_sample_count,
            self.positive_sample_count,
            self.positive_cell_count,
            self.positive_fraction_sum,
            self.positive_mass,
            self.valid_cell_count,
        )
        if (
            self.sample_count <= 0
            or self.effective_exposure_count < self.sample_count
            or any(len(vector) != class_count for vector in vectors)
        ):
            raise ValueError("BEV training statistics have invalid shape")
        if (
            any(
                value < 0 or value > self.effective_exposure_count
                for value in self.active_sample_count
            )
            or any(value < 0 for value in self.positive_sample_count)
            or any(value < 0 for value in self.positive_cell_count)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.positive_fraction_sum
            )
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.positive_mass
            )
            or any(value < 0 for value in self.valid_cell_count)
            or any(
                positive > active
                for positive, active in zip(
                    self.positive_sample_count,
                    self.active_sample_count,
                )
            )
            or any(
                positive > active + 1e-9
                for positive, active in zip(
                    self.positive_fraction_sum,
                    self.active_sample_count,
                )
            )
            or any(
                positive > valid
                for positive, valid in zip(
                    self.positive_cell_count,
                    self.valid_cell_count,
                )
            )
            or any(
                positive > valid + 1e-6
                for positive, valid in zip(
                    self.positive_mass,
                    self.valid_cell_count,
                )
            )
            or len(self.exposure_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.exposure_digest
            )
        ):
            raise ValueError("BEV training statistics are inconsistent")

    def metadata(self) -> dict[str, object]:
        from data_processing.reactive_training_artifacts import (
            BEV_SEGMENTATION_CLASSES,
        )

        return {
            "active_sample_count": list(self.active_sample_count),
            "classes": list(BEV_SEGMENTATION_CLASSES),
            "effective_exposure_count": self.effective_exposure_count,
            "exposure_digest": self.exposure_digest,
            "positive_cell_count": list(self.positive_cell_count),
            "positive_fraction_sum": list(self.positive_fraction_sum),
            "positive_mass": list(self.positive_mass),
            "positive_sample_count": list(self.positive_sample_count),
            "sample_count": self.sample_count,
            "valid_cell_count": list(self.valid_cell_count),
        }


@dataclass(frozen=True)
class BEVSampleStatistics:
    """Packed BEV statistics for one sample before split selection."""

    sample_uid: str
    split_group_uid: str
    positive_cell_count: tuple[int, ...]
    positive_mass: tuple[float, ...]
    valid_cell_count: tuple[int, ...]

    def __post_init__(self) -> None:
        from data_processing.reactive_training_artifacts import (
            BEV_SEGMENTATION_CLASSES,
        )

        class_count = len(BEV_SEGMENTATION_CLASSES)
        if (
            not self.sample_uid
            or not self.split_group_uid
            or len(self.positive_cell_count) != class_count
            or len(self.positive_mass) != class_count
            or len(self.valid_cell_count) != class_count
            or any(value < 0 for value in self.positive_cell_count)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.positive_mass
            )
            or any(value < 0 for value in self.valid_cell_count)
            or any(
                positive > valid
                for positive, valid in zip(
                    self.positive_cell_count,
                    self.valid_cell_count,
                )
            )
            or any(
                positive > valid + 1e-6
                for positive, valid in zip(
                    self.positive_mass,
                    self.valid_cell_count,
                )
            )
        ):
            raise ValueError("BEV sample statistics are inconsistent")

    @property
    def positive_classes(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, count in enumerate(self.positive_cell_count)
            if count > 0
        )


def _is_validation_group(group_uid: str, val_fraction: float) -> bool:
    if not group_uid:
        raise ValueError("split_group_uid must not be empty")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    buckets = SPLIT_BUCKETS
    scaled_fraction = val_fraction * buckets
    val_buckets = round(scaled_fraction)
    if not math.isclose(
        scaled_fraction,
        val_buckets,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"val_fraction must be a multiple of 1/{SPLIT_BUCKETS}"
        )
    val_buckets = max(1, min(buckets - 1, val_buckets))
    return _split_bucket(group_uid, buckets) < val_buckets


def discover_bev_sample_statistics(
    shard_dirs: Sequence[str | Path],
    *,
    shard_files: Sequence[str | Path] | None = None,
) -> tuple[BEVSampleStatistics, ...]:
    """Read each packed sample/BEV-stat pair exactly once."""
    import tarfile

    from data_processing.reactive_training_artifacts import (
        BEV_SEGMENTATION_STATS_MEMBER,
        decode_bev_segmentation_stats,
    )

    roots = [Path(shard_dir) for shard_dir in shard_dirs]
    if not roots:
        raise ValueError("at least one shard directory is required")
    if shard_files is None:
        tarfiles = [
            tar_path
            for root in roots
            for tar_path in packed_sample_tar_paths(root)
        ]
        for root in roots:
            if not any(root == path.parent for path in tarfiles):
                raise FileNotFoundError(f"No .tar shards found in {root}")
    else:
        tarfiles = [Path(path) for path in shard_files]
        if not tarfiles or len(set(tarfiles)) != len(tarfiles):
            raise ValueError(
                "explicit BEV statistic shards must be non-empty and unique"
            )
        resolved_roots = {root.resolve() for root in roots}
        for tar_path in tarfiles:
            resolved = tar_path.resolve()
            if (
                not resolved.is_file()
                or resolved.suffix != ".tar"
                or resolved.parent not in resolved_roots
            ):
                raise ValueError(
                    "BEV statistic shard must be a direct .tar child of a "
                    f"configured directory: {tar_path}"
                )
        tarfiles.sort()

    records: dict[str, dict[str, object]] = {}
    stats_suffix = f".{BEV_SEGMENTATION_STATS_MEMBER}"
    for tar_path in tarfiles:
        with tarfile.open(tar_path, "r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if member.name.endswith(stats_suffix):
                    sample_uid = member.name.removesuffix(stats_suffix)
                    record_key = "stats"
                elif member.name.endswith(".meta.json"):
                    sample_uid = member.name.removesuffix(".meta.json")
                    record_key = "sample"
                else:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(
                        f"could not read {member.name} from {tar_path}"
                    )
                record = records.setdefault(sample_uid, {})
                if record_key in record:
                    raise ValueError(
                        f"duplicate {member.name} for {sample_uid!r}"
                    )
                payload = extracted.read()
                record[record_key] = (
                    decode_bev_segmentation_stats(payload)
                    if record_key == "stats"
                    else _json_mapping(
                        payload,
                        member_name=f"{member.name} in {tar_path}",
                    )
                )

    sample_statistics = []
    for sample_uid, record in sorted(records.items()):
        if set(record) != {"sample", "stats"}:
            raise ValueError(
                f"sample {sample_uid!r} lacks metadata or BEV statistics"
            )
        sample_metadata = record["sample"]
        sample_stats = record["stats"]
        if not isinstance(sample_metadata, Mapping) or not isinstance(
            sample_stats,
            Mapping,
        ):
            raise ValueError("BEV sample statistics metadata is invalid")
        if sample_metadata.get("sample_uid") != sample_uid:
            raise ValueError(
                f"sample metadata UID differs for {sample_uid!r}"
            )
        group_uid = sample_metadata.get("split_group_uid")
        if not isinstance(group_uid, str) or not group_uid:
            raise ValueError(
                f"sample {sample_uid!r} has no split_group_uid"
            )
        sample_statistics.append(BEVSampleStatistics(
            sample_uid=sample_uid,
            split_group_uid=group_uid,
            positive_cell_count=tuple(
                int(value)
                for value in sample_stats["positive_cell_count"]
            ),
            positive_mass=tuple(
                float(value)
                for value in sample_stats["positive_mass"]
            ),
            valid_cell_count=tuple(
                int(value)
                for value in sample_stats["valid_cell_count"]
            ),
        ))
    if not sample_statistics:
        raise ValueError("BEV sample discovery found no samples")
    return tuple(sample_statistics)


def summarize_bev_training_statistics(
    records: Sequence[BEVSampleStatistics],
    *,
    val_fraction: float,
    repeat_factors: Sequence[int] | None = None,
    sample_uids: Sequence[str] | None = None,
) -> BEVTrainingStatistics:
    """Aggregate exact train counts from an in-memory sample scan."""
    from data_processing.reactive_training_artifacts import (
        BEV_SEGMENTATION_CLASSES,
    )

    class_count = len(BEV_SEGMENTATION_CLASSES)
    factors = (
        tuple(int(value) for value in repeat_factors)
        if repeat_factors is not None
        else (1,) * class_count
    )
    if (
        len(factors) != class_count
        or any(value < 1 for value in factors)
    ):
        raise ValueError("BEV repeat factors must be positive per class")
    requested_samples = (
        tuple(str(value) for value in sample_uids)
        if sample_uids is not None
        else None
    )
    if requested_samples is not None:
        allowed_samples = frozenset(requested_samples)
        if (
            not allowed_samples
            or len(allowed_samples) != len(requested_samples)
        ):
            raise ValueError(
                "BEV sample subset must be non-empty and unique"
            )
    else:
        allowed_samples = None

    active_samples: np.ndarray = np.zeros(class_count, dtype=np.int64)
    positive_samples: np.ndarray = np.zeros(class_count, dtype=np.int64)
    positive_cells: np.ndarray = np.zeros(class_count, dtype=np.int64)
    positive_fraction_sum: np.ndarray = np.zeros(
        class_count,
        dtype=np.float64,
    )
    positive_mass: np.ndarray = np.zeros(class_count, dtype=np.float64)
    valid_cells: np.ndarray = np.zeros(class_count, dtype=np.int64)
    exposure_records: list[tuple[str, int]] = []
    for record in records:
        sample_uid = record.sample_uid
        if (
            allowed_samples is not None
            and sample_uid not in allowed_samples
        ):
            continue
        if _is_validation_group(record.split_group_uid, val_fraction):
            continue
        sample_positive_cells = np.asarray(
            record.positive_cell_count,
            dtype=np.int64,
        )
        sample_positive_mass = np.asarray(
            record.positive_mass,
            dtype=np.float64,
        )
        sample_valid_cells = np.asarray(
            record.valid_cell_count,
            dtype=np.int64,
        )
        present = sample_positive_cells > 0
        active = sample_valid_cells > 0
        positive_fraction = np.divide(
            sample_positive_mass,
            sample_valid_cells,
            out=np.zeros(class_count, dtype=np.float64),
            where=active,
        )
        repeat = max(
            (
                factors[index]
                for index in np.flatnonzero(present)
            ),
            default=1,
        )
        active_samples += active.astype(np.int64) * repeat
        positive_samples += present.astype(np.int64) * repeat
        positive_cells += sample_positive_cells * repeat
        positive_fraction_sum += positive_fraction * repeat
        positive_mass += sample_positive_mass * repeat
        valid_cells += sample_valid_cells * repeat
        exposure_records.append((sample_uid, repeat))

    if not exposure_records:
        raise ValueError("BEV statistics selected no training samples")
    digest_payload = "".join(
        f"{sample_uid}\t{repeat}\n"
        for sample_uid, repeat in exposure_records
    ).encode("utf-8")
    return BEVTrainingStatistics(
        sample_count=len(exposure_records),
        effective_exposure_count=sum(
            repeat for _, repeat in exposure_records
        ),
        active_sample_count=tuple(
            int(value) for value in active_samples
        ),
        positive_sample_count=tuple(
            int(value) for value in positive_samples
        ),
        positive_cell_count=tuple(
            int(value) for value in positive_cells
        ),
        positive_fraction_sum=tuple(
            float(value) for value in positive_fraction_sum
        ),
        positive_mass=tuple(float(value) for value in positive_mass),
        valid_cell_count=tuple(int(value) for value in valid_cells),
        exposure_digest=hashlib.sha256(digest_payload).hexdigest(),
    )


def discover_bev_training_statistics(
    shard_dirs: Sequence[str | Path],
    *,
    val_fraction: float,
    repeat_factors: Sequence[int] | None = None,
    sample_uids: Sequence[str] | None = None,
) -> BEVTrainingStatistics:
    """Scan packed metadata and return exact train-split class counts."""
    return summarize_bev_training_statistics(
        discover_bev_sample_statistics(shard_dirs),
        val_fraction=val_fraction,
        repeat_factors=repeat_factors,
        sample_uids=sample_uids,
    )


def select_bev_validation_sample_uids(
    records: Sequence[BEVSampleStatistics],
    *,
    val_fraction: float,
    sample_limit: int,
) -> tuple[str, ...]:
    """Choose a deterministic class-aware validation subset."""
    from data_processing.reactive_training_artifacts import (
        BEV_SEGMENTATION_CLASSES,
    )

    if sample_limit <= 0:
        raise ValueError("BEV validation sample limit must be positive")
    candidates = sorted(
        (
            record
            for record in records
            if _is_validation_group(
                record.split_group_uid,
                val_fraction,
            )
        ),
        key=lambda record: record.sample_uid,
    )
    if not candidates:
        raise ValueError("BEV sample discovery selected no validation samples")

    selected: dict[str, BEVSampleStatistics] = {}
    covered: set[int] = set()
    all_classes = set(range(len(BEV_SEGMENTATION_CLASSES)))
    while len(selected) < sample_limit and covered != all_classes:
        remaining = all_classes - covered
        eligible = [
            record
            for record in candidates
            if record.sample_uid not in selected
            and set(record.positive_classes).intersection(remaining)
        ]
        if not eligible:
            break
        chosen = min(
            eligible,
            key=lambda record: (
                -len(set(record.positive_classes).intersection(remaining)),
                -sum(
                    record.positive_cell_count[index]
                    for index in remaining
                ),
                hashlib.sha256(record.sample_uid.encode("utf-8")).digest(),
            ),
        )
        selected[chosen.sample_uid] = chosen
        covered.update(chosen.positive_classes)

    remaining_candidates = sorted(
        candidates,
        key=lambda record: hashlib.sha256(
            record.sample_uid.encode("utf-8")
        ).digest(),
    )
    for record in remaining_candidates:
        if len(selected) >= sample_limit:
            break
        selected.setdefault(record.sample_uid, record)
    return tuple(sorted(selected))


def select_bev_validation_holdout_sample_uids(
    records: Sequence[BEVSampleStatistics],
    *,
    val_fraction: float,
    excluded_sample_uids: Sequence[str],
) -> tuple[str, ...]:
    """Return a validation holdout disjoint at split-group granularity."""
    requested = tuple(str(value) for value in excluded_sample_uids)
    excluded = frozenset(requested)
    if not excluded or len(excluded) != len(requested):
        raise ValueError(
            "excluded BEV validation samples must be non-empty and unique"
        )
    validation_records = {
        record.sample_uid: record
        for record in records
        if _is_validation_group(record.split_group_uid, val_fraction)
    }
    if not excluded.issubset(validation_records):
        missing = sorted(excluded - validation_records.keys())
        raise ValueError(
            "excluded BEV validation samples are outside the validation "
            f"split: {missing[:8]}"
        )
    excluded_groups = {
        validation_records[sample_uid].split_group_uid
        for sample_uid in excluded
    }
    holdout = tuple(sorted(
        sample_uid
        for sample_uid, record in validation_records.items()
        if record.split_group_uid not in excluded_groups
    ))
    if not holdout:
        raise ValueError("BEV validation holdout is empty")
    return holdout


def select_distributed_bev_validation_sample_uids(
    records_by_rank: Sequence[Sequence[BEVSampleStatistics]],
    *,
    val_fraction: float,
    sample_limit: int,
) -> tuple[str, ...]:
    """Reproduce the rank-local calibration subset used by DDP training."""
    world_size = len(records_by_rank)
    if world_size <= 0 or sample_limit < world_size:
        raise ValueError(
            "distributed BEV validation needs one sample per rank"
        )
    local_sample_limit = math.ceil(sample_limit / world_size)
    selected = tuple(sorted(
        sample_uid
        for rank_records in records_by_rank
        for sample_uid in select_bev_validation_sample_uids(
            rank_records,
            val_fraction=val_fraction,
            sample_limit=local_sample_limit,
        )
    ))
    if len(set(selected)) != len(selected):
        raise ValueError(
            "distributed BEV validation subset contains duplicate samples"
        )
    return selected


def discover_validation_sample_uids(
    shard_dirs: Sequence[str | Path],
    *,
    val_fraction: float,
    sample_limit: int,
) -> tuple[str, ...]:
    """Choose a deterministic validation subset from packed metadata."""
    if sample_limit <= 0:
        raise ValueError("validation sample limit must be positive")
    inventory = discover_split_inventory(shard_dirs)
    candidates = (
        sample_uid
        for group_uid, sample_uids in inventory.sample_uids_by_group
        if _is_validation_group(group_uid, val_fraction)
        for sample_uid in sample_uids
    )
    selected = sorted(
        candidates,
        key=lambda sample_uid: hashlib.sha256(
            sample_uid.encode("utf-8")
        ).digest(),
    )[:sample_limit]
    if not selected:
        raise ValueError("sample discovery selected no validation samples")
    return tuple(selected)


def derive_bev_repeat_factors(
    statistics: BEVTrainingStatistics,
    *,
    frequency_threshold: float = 0.05,
    max_repeat: int = 4,
) -> tuple[int, ...]:
    """Repeat samples only when a class is rare at the sample level."""
    if not 0.0 < frequency_threshold <= 1.0:
        raise ValueError("BEV repeat frequency threshold must be in (0,1]")
    if max_repeat < 1:
        raise ValueError("BEV max repeat must be positive")
    factors = []
    for positive_samples in statistics.positive_sample_count:
        if positive_samples <= 0:
            raise ValueError("every BEV class needs a positive train sample")
        frequency = positive_samples / statistics.sample_count
        repeat = math.ceil(
            math.sqrt(frequency_threshold / frequency)
        )
        factors.append(max(1, min(max_repeat, repeat)))
    return tuple(factors)


def derive_bev_pos_weights(
    statistics: BEVTrainingStatistics,
    *,
    max_weight: float = 64.0,
    min_positive_samples: int = 1,
    min_positive_cells: int = 1,
) -> tuple[float, ...]:
    """Derive clipped BCE positive weights from sample-normalized prevalence."""
    if max_weight < 1.0:
        raise ValueError("BEV maximum positive weight must be at least one")
    if min_positive_samples <= 0 or min_positive_cells <= 0:
        raise ValueError("BEV minimum support must be positive")
    weights = []
    for class_index, (
        sample_count,
        cell_count,
        active_sample_count,
        positive_fraction_sum,
        positive_mass,
    ) in enumerate(zip(
        statistics.positive_sample_count,
        statistics.positive_cell_count,
        statistics.active_sample_count,
        statistics.positive_fraction_sum,
        statistics.positive_mass,
    )):
        if (
            sample_count < min_positive_samples
            or cell_count < min_positive_cells
            or active_sample_count <= 0
            or positive_fraction_sum <= 0.0
            or positive_fraction_sum >= active_sample_count
            or positive_mass <= 0.0
        ):
            raise ValueError(
                "BEV class has insufficient positive support: "
                f"class_index={class_index} samples={sample_count} "
                f"cells={cell_count}"
            )
        negative_fraction_sum = (
            active_sample_count - positive_fraction_sum
        )
        ratio = negative_fraction_sum / positive_fraction_sum
        weights.append(float(np.clip(ratio, 1.0, max_weight)))
    return tuple(weights)


def derive_bev_positive_pair_frequencies(
    statistics: BEVTrainingStatistics,
) -> tuple[float, ...]:
    """Return raw per-class positive-sample frequencies for Dice scaling."""
    if statistics.sample_count <= 0:
        raise ValueError("BEV statistics must contain training samples")
    frequencies = tuple(
        positive_samples / statistics.sample_count
        for positive_samples in statistics.positive_sample_count
    )
    if any(
        not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in frequencies
    ):
        raise ValueError(
            "every BEV class needs a finite positive-sample frequency"
        )
    return frequencies


def derive_bev_gradient_budget_weights(
    statistics: BEVTrainingStatistics,
    pos_weights: Sequence[float],
    *,
    max_relative_weight: float = 8.0,
) -> tuple[float, ...]:
    """Equalize expected per-class BCE gradient mass without shifting priors."""
    positive_fraction_sum = np.asarray(
        statistics.positive_fraction_sum,
        dtype=np.float64,
    )
    active_sample_count = np.asarray(
        statistics.active_sample_count,
        dtype=np.float64,
    )
    weights = np.asarray(tuple(pos_weights), dtype=np.float64)
    if (
        positive_fraction_sum.shape != active_sample_count.shape
        or weights.shape != positive_fraction_sum.shape
        or positive_fraction_sum.ndim != 1
        or not np.isfinite(positive_fraction_sum).all()
        or not np.isfinite(active_sample_count).all()
        or not np.isfinite(weights).all()
        or np.any(positive_fraction_sum <= 0.0)
        or np.any(positive_fraction_sum >= active_sample_count)
        or np.any(weights < 1.0)
        or not math.isfinite(max_relative_weight)
        or max_relative_weight < 1.0
    ):
        raise ValueError("BEV gradient-budget statistics are invalid")
    prevalence = positive_fraction_sum / active_sample_count
    weighted_prevalence = (
        weights * prevalence
        / (1.0 - prevalence + weights * prevalence)
    )
    expected_gradient_mass = (
        weights * prevalence * (1.0 - weighted_prevalence)
        + (1.0 - prevalence) * weighted_prevalence
    )
    effective_exposure_count = float(
        statistics.effective_exposure_count
    )
    if (
        not math.isfinite(effective_exposure_count)
        or effective_exposure_count <= 0.0
        or np.any(active_sample_count > effective_exposure_count)
    ):
        raise ValueError("BEV active-sample statistics are invalid")
    expected_gradient_mass *= (
        active_sample_count / effective_exposure_count
    )
    if (
        not np.isfinite(expected_gradient_mass).all()
        or np.any(expected_gradient_mass <= 0.0)
    ):
        raise ValueError("BEV expected gradient mass is invalid")
    inverse_mass = 1.0 / expected_gradient_mass
    median = float(np.median(inverse_mass))
    bounded = np.clip(
        inverse_mass,
        median / max_relative_weight,
        median * max_relative_weight,
    )
    normalized = bounded / bounded.mean()
    return tuple(float(value) for value in normalized)


def derive_bev_rank_importance_scale(
    *,
    local_effective_exposure_count: int,
    global_sample_count: int,
    world_size: int,
) -> float:
    """Scale one rank so equal-rank DDP averaging recovers the raw corpus."""
    if (
        local_effective_exposure_count <= 0
        or global_sample_count <= 0
        or world_size <= 0
    ):
        raise ValueError("BEV rank importance inputs must be positive")
    scale = (
        world_size
        * local_effective_exposure_count
        / global_sample_count
    )
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("BEV rank importance scale is invalid")
    return float(scale)


def bev_rank_full_microbatch_capacity(
    records_by_directory: Sequence[Sequence[BEVSampleStatistics]],
    *,
    val_fraction: float,
    repeat_factors: Sequence[int],
    batch_size: int,
) -> int:
    """Count complete rank-local batches after split, repeat, and drop_last."""
    if not records_by_directory:
        raise ValueError("BEV rank capacity needs at least one directory")
    if batch_size <= 0:
        raise ValueError("BEV rank batch size must be positive")
    capacity = 0
    for records in records_by_directory:
        has_train_records = any(
            not _is_validation_group(record.split_group_uid, val_fraction)
            for record in records
        )
        if not has_train_records:
            continue
        capacity += (
            summarize_bev_training_statistics(
                records,
                val_fraction=val_fraction,
                repeat_factors=repeat_factors,
            ).effective_exposure_count
            // batch_size
        )
    if capacity <= 0:
        raise ValueError("BEV rank loader has no complete microbatches")
    return capacity


@dataclass(frozen=True)
class BEVClassRepeatPolicy:
    """Repeat rare-positive samples before decode with importance evidence."""

    repeat_factors: tuple[int, ...]
    importance_scale: float
    version: str = BEV_CLASS_REPEAT_POLICY_VERSION

    def __post_init__(self) -> None:
        from data_processing.reactive_training_artifacts import (
            BEV_SEGMENTATION_CLASSES,
        )

        if (
            self.version != BEV_CLASS_REPEAT_POLICY_VERSION
            or len(self.repeat_factors) != len(BEV_SEGMENTATION_CLASSES)
            or any(value < 1 for value in self.repeat_factors)
            or not math.isfinite(self.importance_scale)
            or self.importance_scale <= 0.0
        ):
            raise ValueError("invalid BEV class repeat policy")

    def repeat_count(self, sample: Mapping[str, object]) -> int:
        from data_processing.reactive_training_artifacts import (
            BEV_SEGMENTATION_STATS_MEMBER,
            decode_bev_segmentation_stats,
        )

        payload = sample.get(BEV_SEGMENTATION_STATS_MEMBER)
        if not isinstance(payload, (bytes, bytearray)):
            raise ValueError("sample lacks packed BEV statistics")
        stats = decode_bev_segmentation_stats(bytes(payload))
        present = stats["positive_cell_count"] > 0
        return max(
            (
                self.repeat_factors[index]
                for index in np.flatnonzero(present)
            ),
            default=1,
        )

    def __call__(self, source: Iterable[dict]):
        for sample in source:
            repeat = self.repeat_count(sample)
            enriched = {
                **sample,
                "__bev_repeat_factor__": repeat,
                "__bev_sampling_importance__": (
                    self.importance_scale / repeat
                ),
            }
            for _ in range(repeat):
                yield enriched


def _decode_image(data) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)) if isinstance(data, bytes) else data
    return _TRANSFORM(img)


def _decode_image_uint8(data: bytes) -> torch.Tensor:
    """[3, H, W] uint8 RGB, for AutoBEV samples normalized later on the GPU."""
    with Image.open(io.BytesIO(data)) as image:
        return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1).copy())


class _PoolAccessor:
    """Path-based frame-pool reader: ``frame_id -> jpeg bytes``.

    Reads ``<pool_dir>/<frame_id>.jpg`` on demand. Holding a path rather than an
    open handle lets it pickle cleanly to spawn DataLoader workers.
    """

    def __init__(self, pool_dir: str):
        self.pool_dir = pool_dir

    def __call__(self, frame_id: str) -> bytes:
        with open(f"{self.pool_dir}/{frame_id}.jpg", "rb") as f:
            return f.read()


def _make_pool_accessor(shard_dir: str):
    """Return a verified frame-pool accessor for one packed partition."""
    import hashlib
    import shutil
    import tarfile
    import tempfile

    root = Path(shard_dir)
    pool_dir = root / "pool"
    archive_path = root / "frame_pool.tar"
    if not pool_dir.is_dir() and archive_path.is_file():
        try:
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="ascii")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "frame-pool archive requires a valid manifest"
            ) from error
        expected_name = manifest.get("frame_pool_archive")
        expected_sha256 = manifest.get("frame_pool_archive_sha256")
        expected_count = manifest.get("frame_pool_frame_count")
        if (
            expected_name != archive_path.name
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_sha256
            )
            or not isinstance(expected_count, int)
            or isinstance(expected_count, bool)
            or expected_count <= 0
        ):
            raise ValueError(
                "frame-pool manifest contract is incomplete"
            )
        actual_sha256 = hashlib.sha256(
            archive_path.read_bytes()
        ).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("frame-pool archive digest mismatch")
        temporary_root = Path(
            tempfile.mkdtemp(
                prefix=".frame-pool-",
                dir=root,
            )
        )
        temporary_pool = temporary_root / "pool"
        temporary_pool.mkdir()
        try:
            with tarfile.open(archive_path, "r:") as archive:
                members = archive.getmembers()
                if len(members) != expected_count:
                    raise ValueError(
                        "frame-pool archive member count differs from manifest"
                    )
                member_names: set[str] = set()
                for member in members:
                    name = member.name
                    if (
                        not member.isfile()
                        or member.issym()
                        or member.islnk()
                        or Path(name).name != name
                        or not name.endswith(".jpg")
                        or name in member_names
                    ):
                        raise ValueError(
                            f"unsafe frame-pool archive member {name!r}"
                        )
                    member_names.add(name)
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(
                            f"could not read frame-pool member {name!r}"
                        )
                    with (
                        source,
                        (temporary_pool / name).open("wb") as destination,
                    ):
                        shutil.copyfileobj(source, destination)
            try:
                temporary_pool.rename(pool_dir)
            except FileExistsError:
                pass
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
    return _PoolAccessor(str(pool_dir)) if pool_dir.is_dir() else None


_NAVIGATION_BASE_KEYS = frozenset({
    "map_semantic.npz",
    "route_mask.npz",
    "navigation_meta.json",
})
_NAVIGATION_KEYS = _NAVIGATION_BASE_KEYS | {"route_supervision.npz"}


def _empty_route_supervision(shape) -> dict[str, torch.Tensor]:
    """Loss-only route fields of a sample without route supervision."""
    return {
        "distance_to_corridor_m": torch.zeros(shape, dtype=torch.float32),
        "distance_to_drivable_m": torch.zeros(shape, dtype=torch.float32),
        "route_heading_sin": torch.zeros(shape, dtype=torch.float32),
        "route_heading_cos": torch.zeros(shape, dtype=torch.float32),
        "route_heading_valid": torch.zeros(shape, dtype=torch.bool),
        "destination_xy_m": torch.zeros(2, dtype=torch.float32),
        "destination_visible": torch.tensor(False, dtype=torch.bool),
        "available": torch.tensor(False, dtype=torch.bool),
        "drivable_available": torch.tensor(False, dtype=torch.bool),
    }


def _empty_navigation_map() -> torch.Tensor:
    """An all-zero semantic map on the shared 0.4 m navigation grid."""
    from navigation.geometry import (
        AUTOE2E_NAVIGATION_GEOMETRY,
        MAP_CHANNEL_COUNT,
    )

    return torch.zeros(
        MAP_CHANNEL_COUNT,
        AUTOE2E_NAVIGATION_GEOMETRY.height_px,
        AUTOE2E_NAVIGATION_GEOMETRY.width_px,
        dtype=torch.float32,
    )


def _navigation_without_route(
    map_context: torch.Tensor,
    *,
    map_valid: bool,
) -> dict[str, Any]:
    """Navigation inputs of a sample that has no route members."""
    shape = map_context.shape[-2:]
    return {
        "map_context": map_context,
        "route_mask": torch.zeros(2, *shape, dtype=torch.float32),
        "map_valid": torch.tensor(map_valid, dtype=torch.bool),
        "route_valid": torch.tensor(False, dtype=torch.bool),
        "route_channel_valid": torch.zeros(2, dtype=torch.bool),
        "route_supervision": _empty_route_supervision(shape),
        "navigation_metadata": {},
    }


def _decode_navigation(sample: dict) -> dict[str, Any]:
    """Decode the semantic map, route, and loss-only route members."""
    from navigation.artifacts import (
        decode_route_supervision,
        decode_sample_navigation,
    )
    from navigation.geometry import (
        AUTOE2E_NAVIGATION_GEOMETRY,
        DEFAULT_NAVIGATION_GEOMETRY,
    )

    if not _NAVIGATION_BASE_KEYS.issubset(sample):
        raise ValueError(
            "navigation members must contain the complete schema-v8 set "
            "or the sample_navigation_v3 map, route, and metadata set"
        )
    map_array, route_array, navigation_metadata = (
        decode_sample_navigation(sample)
    )
    supervision = (
        decode_route_supervision(sample)
        if "route_supervision.npz" in sample
        else None
    )
    geometry_by_id = {
        geometry.geometry_id: geometry
        for geometry in (
            AUTOE2E_NAVIGATION_GEOMETRY,
            DEFAULT_NAVIGATION_GEOMETRY,
        )
    }
    geometry_id = navigation_metadata.get("geometry_id")
    if geometry_id not in geometry_by_id:
        raise ValueError(
            "navigation sample geometry differs from the model contract"
        )
    geometry = geometry_by_id[geometry_id]
    if map_array.shape[1:] != (
        geometry.height_px,
        geometry.width_px,
    ):
        raise ValueError(
            "navigation sample raster shape differs from its geometry"
        )
    map_context = torch.from_numpy(map_array.copy())
    raw_channel_valid = navigation_metadata.get("route_channel_valid")
    if raw_channel_valid is None:
        route_channel_valid = torch.tensor(
            [
                bool(navigation_metadata["route_valid"]),
                bool(navigation_metadata["route_valid"])
                and supervision is not None
                and supervision.destination_visible,
            ],
            dtype=torch.bool,
        )
    else:
        if (
            not isinstance(raw_channel_valid, list)
            or len(raw_channel_valid) != 2
            or any(
                not isinstance(value, bool)
                for value in raw_channel_valid
            )
        ):
            raise ValueError(
                "route_channel_valid must contain two booleans"
            )
        route_channel_valid = torch.tensor(
            raw_channel_valid,
            dtype=torch.bool,
        )
    route_supervision = (
        {
            "distance_to_corridor_m": torch.from_numpy(
                supervision.distance_to_corridor_m.copy()
            ),
            "distance_to_drivable_m": torch.from_numpy(
                supervision.distance_to_drivable_m.copy()
            ),
            "route_heading_sin": torch.from_numpy(
                supervision.route_heading_sin.copy()
            ),
            "route_heading_cos": torch.from_numpy(
                supervision.route_heading_cos.copy()
            ),
            "route_heading_valid": torch.from_numpy(
                supervision.route_heading_valid.astype(
                    np.bool_,
                    copy=True,
                )
            ),
            "destination_xy_m": torch.from_numpy(
                supervision.destination_xy_m.copy()
            ),
            "destination_visible": torch.tensor(
                supervision.destination_visible,
                dtype=torch.bool,
            ),
            "available": torch.tensor(True, dtype=torch.bool),
            "drivable_available": torch.tensor(
                supervision.drivable_available,
                dtype=torch.bool,
            ),
        }
        if supervision is not None
        else _empty_route_supervision(map_context.shape[-2:])
    )
    return {
        "map_context": map_context,
        "route_mask": torch.from_numpy(
            route_array.astype(np.float32, copy=True)
        ),
        "map_valid": torch.tensor(
            bool(navigation_metadata["map_valid"]),
            dtype=torch.bool,
        ),
        "route_valid": torch.tensor(
            bool(navigation_metadata["route_valid"]),
            dtype=torch.bool,
        ),
        "route_channel_valid": route_channel_valid,
        "route_supervision": route_supervision,
        "navigation_metadata": navigation_metadata,
    }


def _decode_motion(sample: dict) -> dict[str, torch.Tensor]:
    """Decode ego history and trajectory targets; zeros where absent."""
    ego_bytes = sample.get("ego.npy", b"")
    if isinstance(ego_bytes, bytes) and len(ego_bytes) > 0:
        ego = np.frombuffer(ego_bytes, dtype=np.float32).copy()
    else:
        ego = np.zeros(384, dtype=np.float32)
    # History: (64, 4) flattened = 256; Future: (64, 2) flattened = 128
    history_size = _HISTORY_STEPS * _HISTORY_SIGNALS
    ego_history = torch.from_numpy(ego[:history_size])
    trajectory_xy_data = sample.get("trajectory_xy.npz")
    if trajectory_xy_data is not None:
        from data_processing.reactive_training_artifacts import (
            decode_trajectory_xy,
        )

        trajectory_xy, trajectory_valid = decode_trajectory_xy(
            trajectory_xy_data
        )
        if trajectory_xy.shape != (_FUTURE_STEPS, 2):
            raise ValueError(
                "trajectory XY target must contain exactly 64 timesteps"
            )
        trajectory_xy_m = torch.from_numpy(trajectory_xy.copy())
        trajectory_valid_tensor = torch.from_numpy(trajectory_valid.copy())
    else:
        trajectory_xy_m = torch.zeros(_FUTURE_STEPS, 2, dtype=torch.float32)
        trajectory_valid_tensor = torch.zeros(_FUTURE_STEPS, dtype=torch.bool)
    return {
        "egomotion_history": ego_history,
        "trajectory_target": torch.from_numpy(ego[history_size:]),
        "trajectory_xy_m": trajectory_xy_m,
        "trajectory_valid": trajectory_valid_tensor,
        "initial_speed_mps": ego_history.reshape(
            _HISTORY_STEPS,
            _HISTORY_SIGNALS,
        )[-1, 0],
    }


def _decode_sample(
    sample: dict,
    pool=None,
    *,
    decode_history_frames: bool = True,
    decode_future_frames: bool = True,
    decode_front_camera_fpn: bool = True,
) -> dict:
    """Decode a WebDataset sample into training tensors.

    Pose-compensated datasets carry per-sample calibration here. Rig-constant
    datasets instead expose one loader-level ``.projection`` /
    ``.geometry_type`` fallback from their manifest.

    ``pool`` is a frame-pool accessor (``frame_id -> jpeg bytes``) for shards
    whose World-Model windows reference frames by id through
    ``window_index.json``; None on shards without a pool.
    """
    # Keys: "cam_0.jpg" ... "cam_{V-1}.jpg", navigation members or an optional
    # "map.jpg", plus numeric/metadata members.
    cam_keys = sorted(
        (k for k in sample if _CAM_KEY_RE.match(k)),
        key=lambda k: int(k[len("cam_"):-len(".jpg")]),
    )
    calibration = (
        _json_mapping(sample["calib.json"], member_name="calib.json")
        if "calib.json" in sample
        else None
    )
    frames = [_decode_image(sample[k]) for k in cam_keys]
    history_index: object = None
    bev_history_keys: dict[tuple[int, int], str] = {}
    for key in sample:
        match = _BEV_HIST_KEY_RE.match(key)
        if match is not None:
            bev_history_keys[
                (int(match.group(1)), int(match.group(2)))
            ] = key
    bev_history_index_blob = sample.get(_BEV_HISTORY_INDEX_MEMBER)
    if bev_history_keys and bev_history_index_blob is not None:
        raise ValueError(
            "BEVFormer T8 history cannot use direct members and a pool index"
        )
    camera_history_tiles = None
    if bev_history_keys:
        expected_history_keys = {
            (history_index, camera_index)
            for history_index in range(
                REACTIVE_BEVFORMER_HISTORY_FRAMES
            )
            for camera_index in range(len(frames))
        }
        if set(bev_history_keys) != expected_history_keys:
            raise ValueError(
                "BEVFormer T8 camera history is incomplete or misindexed"
            )
        camera_history_tiles = torch.stack([
            torch.stack([
                _decode_image(sample[bev_history_keys[
                    (history_index, camera_index)
                ]])
                for camera_index in range(len(frames))
            ])
            for history_index in range(
                REACTIVE_BEVFORMER_HISTORY_FRAMES
            )
        ])
    elif bev_history_index_blob is not None:
        if pool is None:
            raise ValueError(
                "BEVFormer T8 history index requires the sibling frame pool"
            )
        try:
            history_index = json.loads(
                bev_history_index_blob.decode()
                if isinstance(bev_history_index_blob, (bytes, bytearray))
                else bev_history_index_blob
            )
        except (AttributeError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "BEVFormer T8 history index is invalid JSON"
            ) from error
        if (
            not isinstance(history_index, list)
            or len(history_index) != REACTIVE_BEVFORMER_HISTORY_FRAMES
            or any(
                not isinstance(frame, list)
                or len(frame) != len(frames)
                or any(
                    not isinstance(frame_id, str) or not frame_id
                    for frame_id in frame
                )
                for frame in history_index
            )
        ):
            raise ValueError(
                "BEVFormer T8 history index differs from contract"
            )
        camera_history_tiles = _decode_window_from_index(
            history_index,
            pool,
        )
    if camera_history_tiles is not None:
        if tuple(camera_history_tiles.shape[2:]) != (
            3,
            REACTIVE_CAMERA_IMAGE_SIZE,
            REACTIVE_CAMERA_IMAGE_SIZE,
        ):
            raise ValueError(
                "BEVFormer T8 history camera dimensions differ from contract"
            )
    front_camera_tile = None
    front_camera_fpn_tile = None
    front_camera_fpn_available = None
    if calibration is None and "front_camera_fpn.jpg" in sample:
        raise ValueError(
            "front camera FPN image requires camera calibration"
        )
    if calibration is not None:
        front_contract_markers = (
            "front_camera_index",
            "front_camera_image_size",
            "front_projection",
        )
        required_front_fields = (
            "front_camera_index",
            "front_camera_image_size",
            "image_size",
        )
        present_front_markers = {
            field for field in front_contract_markers
            if field in calibration
        }
        has_complete_front_dimensions = all(
            field in calibration for field in required_front_fields
        )
        has_front_camera_fpn_member = "front_camera_fpn.jpg" in sample
        has_front_camera_fpn_contract = (
            "front_camera_fpn_image_size" in calibration
        )
        if (
            has_front_camera_fpn_member
            != has_front_camera_fpn_contract
        ):
            raise ValueError(
                "front camera FPN cache contract and image must coexist"
            )
        if present_front_markers and not has_complete_front_dimensions:
            raise ValueError(
                "front camera calibration contract is incomplete"
            )
        if has_complete_front_dimensions:
            front_index = calibration["front_camera_index"]
            front_size = calibration["front_camera_image_size"]
            base_size = calibration["image_size"]
            if (
                not isinstance(front_index, int)
                or isinstance(front_index, bool)
                or not isinstance(front_size, int)
                or isinstance(front_size, bool)
                or not isinstance(base_size, int)
                or isinstance(base_size, bool)
                or front_index != REACTIVE_FRONT_CAMERA_INDEX
                or front_size != REACTIVE_FRONT_CAMERA_IMAGE_SIZE
                or base_size != REACTIVE_CAMERA_IMAGE_SIZE
            ):
                raise ValueError(
                    "front camera calibration differs from model contract"
                )
            if front_index >= len(frames):
                raise ValueError(
                    "front camera index exceeds packed camera count"
                )
            front_camera_tile = frames[front_index]
            if tuple(front_camera_tile.shape) != (
                3,
                front_size,
                front_size,
            ):
                raise ValueError(
                    "packed front camera dimensions differ from calibration"
                )
            # Keep a base-resolution copy because generic consumers still read
            # visual_tiles without the optional native-front field.
            frames[front_index] = F.interpolate(
                front_camera_tile.unsqueeze(0),
                size=(base_size, base_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
            if any(
                tuple(frame.shape) != (3, base_size, base_size)
                for frame in frames
            ):
                raise ValueError(
                    "packed base camera dimensions differ from calibration"
                )
            front_camera_fpn_available = torch.tensor(
                False,
                dtype=torch.bool,
            )
            if has_front_camera_fpn_member:
                front_camera_fpn_size = calibration[
                    "front_camera_fpn_image_size"
                ]
                if (
                    not isinstance(front_camera_fpn_size, int)
                    or isinstance(front_camera_fpn_size, bool)
                    or front_camera_fpn_size != base_size
                ):
                    raise ValueError(
                        "front camera FPN dimensions differ from "
                        "the base camera contract"
                    )
                if decode_front_camera_fpn:
                    front_camera_fpn_tile = _decode_image(
                        sample["front_camera_fpn.jpg"]
                    )
                    if tuple(front_camera_fpn_tile.shape) != (
                        3,
                        base_size,
                        base_size,
                    ):
                        raise ValueError(
                            "packed front camera FPN image differs from "
                            "the base camera contract"
                        )
                    front_camera_fpn_available = torch.tensor(
                        True,
                        dtype=torch.bool,
                    )
        elif has_front_camera_fpn_member:
            raise ValueError(
                "front camera FPN image requires the native Front contract"
            )

    if _NAVIGATION_KEYS.intersection(sample):
        navigation = _decode_navigation(sample)
    elif "map.jpg" in sample:
        # L2D keeps its RGB map tile without a route.
        navigation = _navigation_without_route(
            _decode_image(sample["map.jpg"]),
            map_valid=True,
        )
    elif (
        isinstance(calibration, dict)
        and calibration.get("dataset") == "KIT-MRT/KITScenes-Multimodal"
    ):
        navigation = _navigation_without_route(
            _empty_navigation_map(),
            map_valid=False,
        )
    else:
        # NVIDIA and map-less shards receive explicit invalid inputs.
        navigation = _navigation_without_route(
            torch.zeros_like(frames[0] if frames else torch.zeros(3, 256, 256)),
            map_valid=False,
        )
    map_context = navigation["map_context"]
    motion = _decode_motion(sample)
    bev_data = sample.get("bev_segmentation.npz")
    if bev_data is not None:
        from data_processing.reactive_training_artifacts import (
            decode_bev_segmentation,
        )

        bev_target, bev_valid = decode_bev_segmentation(bev_data)
        if bev_target.shape[1:] != map_context.shape[-2:]:
            raise ValueError(
                "BEV segmentation geometry differs from navigation raster"
            )
        bev_segmentation_target = torch.from_numpy(bev_target.copy())
        bev_segmentation_valid = torch.from_numpy(bev_valid.copy())
        bev_segmentation_available = torch.tensor(
            True,
            dtype=torch.bool,
        )
    else:
        bev_segmentation_target = torch.zeros(
            8,
            *map_context.shape[-2:],
            dtype=torch.float32,
        )
        bev_segmentation_valid = torch.zeros_like(
            bev_segmentation_target,
            dtype=torch.bool,
        )
        bev_segmentation_available = torch.tensor(
            False,
            dtype=torch.bool,
        )
    sample_metadata = (
        _json_mapping(sample["meta.json"], member_name="meta.json")
        if "meta.json" in sample
        else {}
    )
    raw_split_group_uid = sample_metadata.get("split_group_uid", "")
    split_group_uid = (
        raw_split_group_uid
        if isinstance(raw_split_group_uid, str)
        else ""
    )
    raw_source_frame_index = sample_metadata.get("frame_idx", -1)
    source_frame_index = (
        raw_source_frame_index
        if (
            isinstance(raw_source_frame_index, int)
            and not isinstance(raw_source_frame_index, bool)
            and raw_source_frame_index >= 0
        )
        else -1
    )
    source_frame_interval_us = _source_frame_interval_us_from_history_index(
        history_index,
        current_frame_index=source_frame_index,
    )
    camera_projection_matrix = None
    front_camera_projection_matrix = None
    camera_history_projection_matrix = None
    camera_geometry_type = None
    if calibration is not None:
        projection_spec = calibration.get("projection")
        front_projection_spec = calibration.get("front_projection")
        history_projection_spec = calibration.get("history_projection")
        geometry_label = calibration.get("geometry_type")
        temporal_fields = (
            "temporal_frame_interval_us",
            "temporal_frame_offsets",
        )
        present_temporal_fields = {
            field for field in temporal_fields
            if field in calibration
        }
        if camera_history_tiles is not None and (
            present_temporal_fields != set(temporal_fields)
            or calibration.get("temporal_frame_interval_us")
            != REACTIVE_BEVFORMER_FRAME_INTERVAL_US
            or calibration.get("temporal_frame_offsets")
            != list(REACTIVE_BEVFORMER_FRAME_OFFSETS)
            or not isinstance(history_projection_spec, Mapping)
            or history_projection_spec.get("type")
            not in ("pinhole", "rectified_pinhole")
            or history_projection_spec.get("reference_frame")
            != "current_ego"
            or not isinstance(projection_spec, Mapping)
            or projection_spec.get("type")
            not in ("pinhole", "rectified_pinhole")
        ):
            raise ValueError(
                "BEVFormer T8 history calibration differs from contract"
            )
        if camera_history_tiles is None and (
            present_temporal_fields or history_projection_spec is not None
        ):
            raise ValueError(
                "BEVFormer T8 calibration requires packed camera history"
            )
        if front_camera_tile is not None and (
            not isinstance(projection_spec, Mapping)
            or projection_spec.get("type")
            not in ("pinhole", "rectified_pinhole")
            or not isinstance(front_projection_spec, Mapping)
            or front_projection_spec.get("type")
            not in ("pinhole", "rectified_pinhole")
        ):
            raise ValueError(
                "front camera calibration requires pinhole base and "
                "native projections"
            )
        if projection_spec is not None:
            if (
                not isinstance(projection_spec, Mapping)
                or projection_spec.get("type")
                not in ("pinhole", "rectified_pinhole")
            ):
                # Existing f-theta datasets keep their loader-level operator.
                projection_spec = None
            else:
                matrix = np.asarray(
                    projection_spec.get("matrix"),
                    dtype=np.float32,
                )
                if (
                    matrix.shape != (len(frames), 3, 4)
                    or not np.isfinite(matrix).all()
                ):
                    raise ValueError(
                        "per-sample camera projection must have shape [V,3,4]"
                    )
                camera_projection_matrix = torch.from_numpy(
                    matrix.copy()
                )
                camera_geometry_type = str(geometry_label)
                if camera_history_tiles is not None:
                    assert isinstance(history_projection_spec, Mapping)
                    history_matrix = np.asarray(
                        history_projection_spec.get("matrix"),
                        dtype=np.float32,
                    )
                    if (
                        history_matrix.shape != (
                            REACTIVE_BEVFORMER_HISTORY_FRAMES,
                            len(frames),
                            3,
                            4,
                        )
                        or not np.isfinite(history_matrix).all()
                    ):
                        raise ValueError(
                            "history projection must have shape [7,V,3,4]"
                        )
                    camera_history_projection_matrix = torch.from_numpy(
                        history_matrix.copy()
                    )
                if front_camera_tile is not None:
                    assert isinstance(front_projection_spec, Mapping)
                    front_matrix = np.asarray(
                        front_projection_spec.get("matrix"),
                        dtype=np.float32,
                    )
                    if (
                        front_matrix.shape != (1, 3, 4)
                        or not np.isfinite(front_matrix).all()
                    ):
                        raise ValueError(
                            "front camera projection must have shape [1,3,4]"
                        )
                    expected_front_matrix = matrix[
                        REACTIVE_FRONT_CAMERA_INDEX:
                        REACTIVE_FRONT_CAMERA_INDEX + 1
                    ].copy()
                    expected_front_matrix[:, :2] *= (
                        REACTIVE_FRONT_CAMERA_IMAGE_SIZE
                        / REACTIVE_CAMERA_IMAGE_SIZE
                    )
                    if not np.allclose(
                        front_matrix,
                        expected_front_matrix,
                        rtol=1e-5,
                        atol=1e-5,
                    ):
                        raise ValueError(
                            "front camera projection does not match "
                            "the base camera frame"
                        )
                    front_camera_projection_matrix = torch.from_numpy(
                        front_matrix.copy()
                    )

    out = {
        # Overlay inference derives noise from this stable identity. Keep it in
        # every batch so predictions do not depend on batch position or size.
        "sample_uid": sample.get("__key__", ""),
        "split_group_uid": split_group_uid,
        "source_frame_index": torch.tensor(
            source_frame_index,
            dtype=torch.int64,
        ),
        "source_frame_interval_us": torch.tensor(
            source_frame_interval_us,
            dtype=torch.int64,
        ),
        "visual_tiles": torch.stack(frames),
        **navigation,
        **motion,
        "visual_history": torch.zeros(_VISUAL_HISTORY_DIM),
        "bev_segmentation_target": bev_segmentation_target,
        "bev_segmentation_valid": bev_segmentation_valid,
        "bev_segmentation_available": bev_segmentation_available,
        "bev_repeat_factor": torch.tensor(
            int(sample.get("__bev_repeat_factor__", 1)),
            dtype=torch.int64,
        ),
        "bev_sampling_importance": torch.tensor(
            float(sample.get("__bev_sampling_importance__", 1.0)),
            dtype=torch.float32,
        ),
    }
    if camera_projection_matrix is not None:
        out["camera_projection_matrix"] = camera_projection_matrix
        out["camera_geometry_type"] = camera_geometry_type
    if front_camera_tile is not None:
        out["front_camera_tile"] = front_camera_tile
    if front_camera_fpn_tile is not None:
        out["front_camera_fpn_tile"] = front_camera_fpn_tile
    if front_camera_fpn_available is not None:
        out["front_camera_fpn_available"] = (
            front_camera_fpn_available
        )
    if camera_history_tiles is not None:
        out["camera_history_tiles"] = camera_history_tiles
    if camera_history_projection_matrix is not None:
        out["camera_history_projection_matrix"] = (
            camera_history_projection_matrix
        )
    if front_camera_projection_matrix is not None:
        out["front_camera_projection_matrix"] = (
            front_camera_projection_matrix
        )

    pose_data = sample.get("pose.npy")
    gps_data = sample.get("gps.npy")
    if (pose_data is None) != (gps_data is None):
        raise ValueError(
            "pose.npy and gps.npy must either both be present or both be absent"
        )
    if pose_data is not None:
        assert gps_data is not None
        from data_processing.geospatial import (
            decode_gps_future,
            decode_pose,
        )

        pose = decode_pose(pose_data)
        out["pose_current"] = torch.tensor(
            [
                pose["latitude_deg"],
                pose["longitude_deg"],
                pose["heading_deg_cw_from_north"],
            ],
            dtype=torch.float64,
        )
        out["gps_future"] = torch.from_numpy(
            decode_gps_future(gps_data)
        )

    # Optional World-Model windows: window_index.json maps (step, view) to a
    # frame_id in the sibling pool/. Rebuild history_frames [T, V, 3, H, W] and
    # future_frames [F, V, 3, H, W], oldest to newest. Absent means no JEPA loss.
    windows = _decode_windows_from_pool(
        sample,
        pool,
        decode_history_frames=decode_history_frames,
        decode_future_frames=decode_future_frames,
    )
    if windows is not None:
        history_frames, future_frames = windows
        if history_frames is not None:
            out["history_frames"] = history_frames
        if future_frames is not None:
            out["future_frames"] = future_frames

    # Optional reasoning labels: "reasoning.json" holds a serialized
    # ReasoningLabelRecord, flattened to top-level "reasoning__*" keys so the
    # default collation stacks them into [B, ...]. The keys are always emitted
    # so labeled and unlabeled samples collate together; an unlabeled sample
    # gets a fully masked target and contributes nothing to the loss.
    reasoning_data = sample.get("reasoning.json")
    for key, tensor in _decode_reasoning_targets(reasoning_data).items():
        out[f"reasoning__{key}"] = tensor

    return out


def read_autobev_manifest(shard_dir: str | Path) -> Mapping[str, object] | None:
    """Return the manifest of a pack with AutoBEV members, or None for others.

    A pack that records ``autobev_members_version`` must match the AutoBEV
    member contract in every field; any difference is refused rather than
    read with the wrong geometry.
    """
    from data_processing.autobev_members import check_manifest_contract

    path = Path(shard_dir) / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    if "autobev_members_version" not in manifest:
        return None
    check_manifest_contract(manifest, source=str(path))
    return manifest


def _decode_autobev_sample(
    sample: dict,
    *,
    planner_supervised: bool,
) -> dict:
    """Decode a sample of the AutoBEV member contract into training tensors.

    nuPlan samples carry navigation, ego, and trajectory members; BEV-only
    samples do not, and get zero inputs with invalid flags and
    ``planner_supervised=False`` so both kinds share one batch layout.
    """
    from data_processing.autobev_members import (
        BOXES_MEMBER,
        DEPTH_MEMBER,
        FEATURE_H,
        FEATURE_W,
        HISTORY_PROJECTION_MEMBER,
        PROJECTION_MEMBER,
        camera_member,
        check_projections,
        decode_boxes,
        decode_npy,
        decode_stride_depth,
        history_camera_member,
    )
    from data_processing.reactive_training_artifacts import (
        BEV_SEGMENTATION_CLASSES,
        decode_bev_segmentation,
    )
    from model_components.autobev.geometry import (
        BEV_H,
        BEV_W,
        HISTORY_FRAMES,
        NUM_VIEWS,
    )
    from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY

    projection = decode_npy(sample[PROJECTION_MEMBER])
    history_projection = decode_npy(sample[HISTORY_PROJECTION_MEMBER])
    check_projections(projection, history_projection)
    bev_target, bev_valid = decode_bev_segmentation(
        sample["bev_segmentation.npz"]
    )
    if bev_target.shape != (len(BEV_SEGMENTATION_CLASSES), BEV_H, BEV_W):
        raise ValueError(
            f"AutoBEV labels must be [{len(BEV_SEGMENTATION_CLASSES)},"
            f"{BEV_H},{BEV_W}], got {bev_target.shape}"
        )
    depth = (
        decode_stride_depth(sample[DEPTH_MEMBER])
        if DEPTH_MEMBER in sample
        else np.zeros((NUM_VIEWS, FEATURE_H, FEATURE_W), dtype=np.float32)
    )
    boxes, box_labels = decode_boxes(sample.get(BOXES_MEMBER))
    navigation = (
        _decode_navigation(sample)
        if _NAVIGATION_KEYS.intersection(sample)
        else _navigation_without_route(_empty_navigation_map(), map_valid=False)
    )
    if tuple(navigation["map_context"].shape[1:]) != (
        AUTOE2E_NAVIGATION_GEOMETRY.height_px,
        AUTOE2E_NAVIGATION_GEOMETRY.width_px,
    ):
        raise ValueError("AutoBEV samples need the 0.4 m navigation raster")
    del navigation["navigation_metadata"]
    sample_metadata = (
        _json_mapping(sample["meta.json"], member_name="meta.json")
        if "meta.json" in sample
        else {}
    )
    frame_index = sample_metadata.get("frame_idx", -1)
    return {
        "sample_uid": sample.get("__key__", ""),
        "split_group_uid": str(sample_metadata.get("split_group_uid", "")),
        "source_frame_index": torch.tensor(
            frame_index
            if isinstance(frame_index, int) and not isinstance(frame_index, bool)
            else -1,
            dtype=torch.int64,
        ),
        # uint8 here, normalized on the GPU: a float sample would be 0.6 GB.
        "visual_tiles": torch.stack([
            _decode_image_uint8(sample[camera_member(view)])
            for view in range(NUM_VIEWS)
        ]),
        "camera_history_tiles": torch.stack([
            torch.stack([
                _decode_image_uint8(sample[history_camera_member(frame, view)])
                for view in range(NUM_VIEWS)
            ])
            for frame in range(HISTORY_FRAMES)
        ]),
        "camera_projection_matrix": torch.from_numpy(
            projection.astype(np.float32)
        ),
        "camera_history_projection_matrix": torch.from_numpy(
            history_projection.astype(np.float32)
        ),
        **navigation,
        **_decode_motion(sample),
        "visual_history": torch.zeros(_VISUAL_HISTORY_DIM),
        "planner_supervised": torch.tensor(
            planner_supervised,
            dtype=torch.bool,
        ),
        "bev_segmentation_target": torch.from_numpy(
            bev_target.astype(np.float32)
        ),
        "bev_segmentation_valid": torch.from_numpy(bev_valid.astype(np.bool_)),
        "bev_segmentation_available": torch.tensor(True, dtype=torch.bool),
        "bev_repeat_factor": torch.tensor(
            int(sample.get("__bev_repeat_factor__", 1)),
            dtype=torch.int64,
        ),
        "bev_sampling_importance": torch.tensor(
            float(sample.get("__bev_sampling_importance__", 1.0)),
            dtype=torch.float32,
        ),
        "autobev_depth": torch.from_numpy(depth),
        "autobev_boxes": torch.from_numpy(boxes),
        "autobev_box_labels": torch.from_numpy(box_labels),
        **{
            f"reasoning__{key}": tensor
            for key, tensor in _decode_reasoning_targets(
                sample.get("reasoning.json")
            ).items()
        },
    }


def collate_autobev_batch(samples: Sequence[Mapping[str, object]]) -> dict:
    """Stack AutoBEV samples; variable-length boxes stay lists of tensors."""
    from torch.utils.data import default_collate

    batch = default_collate([
        {
            key: value
            for key, value in sample.items()
            if key not in AUTOBEV_VARIABLE_LENGTH_KEYS
        }
        for sample in samples
    ])
    for key in AUTOBEV_VARIABLE_LENGTH_KEYS:
        batch[key] = [sample[key] for sample in samples]
    return batch


def _decode_reasoning_targets(data) -> dict:
    """Decode the reasoning.json member into per-sample target tensors.

    When ``data`` is None, return the tensors of an abstained record (all
    IGNORE_INDEX, zero source_weight) so the sample is masked out of the loss.
    """
    from data_processing.reasoning_label_generation.schema import ReasoningLabelRecord
    from data_processing.reasoning_label_generation.targets import (
        record_from_json,
        record_to_target_tensors,
    )

    if data is None:
        record = ReasoningLabelRecord.abstain(
            sample_id="", dataset_name="", teacher_provider="none",
            teacher_model="none", prompt_version="none",
            request_mode="clip_horizons", teacher_error="no reasoning.json")
    else:
        payload = json.loads(data.decode() if isinstance(data, (bytes, bytearray)) else data)
        record = record_from_json(payload)
    return record_to_target_tensors(record)


def _decode_window_from_index(index_steps, pool) -> torch.Tensor:
    """Decode one window from a per-step list of per-view frame_ids.

    Returns ``[steps, V, 3, H, W]``, oldest to newest.
    """
    frame_steps = []
    for view_ids in index_steps:
        view_frames = [_decode_image(pool(fid)) for fid in view_ids]
        frame_steps.append(torch.stack(view_frames))   # [V, 3, H, W]
    return torch.stack(frame_steps)                     # [steps, V, 3, H, W]


def _decode_windows_from_pool(
    sample: dict,
    pool,
    *,
    decode_history_frames: bool = True,
    decode_future_frames: bool = True,
):
    """Rebuild (history_frames, future_frames) from window_index.json + the pool.

    Without window_index.json, falls back to per-sample hist_/fut_ members and
    returns None when those are absent too. Requires a pool accessor when
    window_index.json is present.
    """
    if not decode_history_frames and not decode_future_frames:
        return None
    idx_blob = sample.get("window_index.json")
    if idx_blob is None:
        return _decode_windows_legacy(
            sample,
            decode_history_frames=decode_history_frames,
            decode_future_frames=decode_future_frames,
        )
    if pool is None:
        raise ValueError(
            "sample has window_index.json but the loader has no frame pool accessor; "
            "the sibling pool/ dir must exist next to the shards (#121 §3.4d).")
    index = json.loads(idx_blob.decode() if isinstance(idx_blob, (bytes, bytearray)) else idx_blob)
    hist = (
        _decode_window_from_index(index["history"], pool)
        if decode_history_frames
        else None
    )
    fut = (
        _decode_window_from_index(index["future"], pool)
        if decode_future_frames
        else None
    )
    return hist, fut


def _decode_windows_legacy(
    sample: dict,
    *,
    decode_history_frames: bool = True,
    decode_future_frames: bool = True,
):
    """Decode per-sample hist_<t>_cam_<v>.jpg / fut_<f>_cam_<v>.jpg members.

    Returns (history, future) or None if absent."""
    def _one(key_re):
        matches = [(m, k) for k in sample if (m := key_re.match(k))]
        if not matches:
            return None
        steps = max(int(m.group(1)) for m, _ in matches) + 1
        frame_steps = []
        for t in range(steps):
            view_frames = [
                _decode_image(sample[k])
                for m, k in sorted(matches, key=lambda mk: int(mk[0].group(2)))
                if int(m.group(1)) == t
            ]
            frame_steps.append(torch.stack(view_frames))
        return torch.stack(frame_steps)
    if not decode_history_frames and not decode_future_frames:
        return None
    hist = _one(_HIST_KEY_RE) if decode_history_frames else None
    fut = _one(_FUT_KEY_RE) if decode_future_frames else None
    if (
        (decode_history_frames and hist is None)
        or (decode_future_frames and fut is None)
    ):
        return None
    return hist, fut


def load_projection_from_manifest(shard_dir: str):
    """Reconstruct the per-dataset projection operator from manifest.json.

    Returns ``(projection, geometry_type)``. A dataset with real calibration
    stores an operator spec under ``projection`` in its manifest:

        {"geometry_type": "pinhole",
         "projection": {"type": "pinhole", "matrix": [[...]]}}   # [V,3,4]
        {"geometry_type": "ftheta",
         "projection": {"type": "ftheta", "t_camera_ego": [...],  # [V,4,4]
                        "fw_poly": [...], "cx": [...], "cy": [...],
                        "image_wh": [...], "max_theta": ...}}  # native (W,H), FOV

    A dataset without calibration returns ``(None, "pseudo")`` and the caller
    runs the explicit pseudo path.
    """
    mpath = Path(shard_dir) / "manifest.json"
    # A missing manifest means pseudo geometry, but an unreadable one raises:
    # silently degrading a calibrated run to pseudo geometry would corrupt it.
    if not mpath.exists():
        return None, "pseudo"
    try:
        manifest = json.loads(mpath.read_text())
    except (ValueError, OSError) as e:
        raise ValueError(
            f"manifest.json at {mpath} exists but could not be parsed ({e}); "
            f"refusing to silently fall back to pseudo geometry."
        ) from e

    spec = manifest.get("projection")
    if spec is None:
        return None, manifest.get("geometry_type", "pseudo")
    return projection_from_spec(spec)


def projection_from_spec(spec):
    """Reconstruct ``(projection, geometry_type)`` from a serialized spec dict.

    Shared by the single-dataset manifest path and the per-sample calib.json
    path (merged loader). ``spec`` is what ``CameraProjectionModel.to_spec()``
    produced; ``None`` returns the pseudo path.
    """
    from model_components.view_fusion.projection import (
        FThetaProjection,
        PinholeProjection,
    )

    if spec is None:
        return None, "pseudo"
    kind = spec.get("type")
    if kind in ("pinhole", "rectified_pinhole"):
        matrix = torch.tensor(spec["matrix"], dtype=torch.float32).unsqueeze(0)  # [1,V,3,4]
        return PinholeProjection(matrix, geometry_type=kind), kind
    if kind == "ftheta":
        def _t(key):
            return torch.tensor(spec[key], dtype=torch.float32).unsqueeze(0)
        # fw_poly is serialized as a shared [K] or per-view [V,K]; restore the
        # matching shape ([K] or [1,V,K]) so the to_spec round-trip is exact.
        fw = spec["fw_poly"]
        if fw and isinstance(fw[0], (list, tuple)):
            fw_poly = torch.tensor(fw, dtype=torch.float32).unsqueeze(0)  # [1,V,K]
        else:
            fw_poly = torch.tensor(fw, dtype=torch.float32)               # [K] shared
        max_theta = spec.get("max_theta")
        if isinstance(max_theta, (list, tuple)):
            max_theta = torch.tensor(max_theta, dtype=torch.float32)      # per-view
        return (
            FThetaProjection(
                t_camera_ego=_t("t_camera_ego"),   # [1,V,4,4]
                fw_poly=fw_poly,
                cx=_t("cx"), cy=_t("cy"),          # [1,V]
                image_wh=_t("image_wh"),           # [1,V,2] native (W,H)
                max_theta=max_theta,
            ),
            "ftheta",
        )
    raise ValueError(f"Unknown projection type in spec: {kind!r}")


def _split_bucket(key: str, buckets: int = SPLIT_BUCKETS) -> int:
    """Deterministic bucket in [0, buckets) from a stable string.

    Uses a fixed hash rather than Python's per-process salted ``hash()`` so
    train and eval processes agree on the split.
    """
    from data_processing.dataset_snapshot import split_bucket
    return split_bucket(key, buckets)


def _split_group_of(sample, *, required: bool = False) -> str:
    """The train/val split key for a raw shard sample.

    Uses the episode/clip ``split_group_uid`` from ``meta.json`` so correlated
    frames of one episode never straddle train/val. Falls back to ``__key__``
    when the group is missing, unless ``required`` is set.
    """
    import json
    meta = sample.get("meta.json")
    if meta is not None:
        try:
            g = json.loads(meta.decode() if isinstance(meta, (bytes, bytearray)) else meta)
            grp = g.get("split_group_uid")
            if grp:
                return str(grp)
        except Exception as error:
            if required:
                raise ValueError(
                    "sample meta.json is invalid; an explicit group split "
                    "cannot fall back to the sample UID"
                ) from error
    if required:
        raise ValueError(
            "sample meta.json has no split_group_uid; an explicit group split "
            "cannot fall back to the sample UID"
        )
    return sample.get("__key__", "")


@dataclass(frozen=True)
class _ExplicitSplitGroupFilter:
    """Picklable selector for a frozen validation-group manifest."""

    validation_groups: frozenset[str]
    keep_validation: bool

    def __call__(self, sample) -> bool:
        group_uid = _split_group_of(sample, required=True)
        in_validation = group_uid in self.validation_groups
        return in_validation if self.keep_validation else not in_validation


@dataclass(frozen=True)
class _KeepAllSplitFilter:
    """Picklable selector used by unsplit and zero-validation loaders."""

    def __call__(self, sample) -> bool:
        return True


@dataclass(frozen=True)
class _HashSplitGroupFilter:
    """Picklable stable-hash train or validation group selector."""

    buckets: int
    validation_buckets: int
    keep_validation: bool

    def __call__(self, sample) -> bool:
        bucket = _split_bucket(_split_group_of(sample), self.buckets)
        in_validation = bucket < self.validation_buckets
        return (
            in_validation
            if self.keep_validation
            else not in_validation
        )


def _split_keep(
    split: str,
    val_fraction: float,
    validation_group_uids: Sequence[str] | None = None,
):
    """Return a predicate ``sample -> bool`` selecting the requested split.

    ``split="all"`` keeps everything. ``"train"`` / ``"val"`` partition by a
    stable hash of ``split_group_uid``: ``val`` is the first
    ``round(val_fraction * SPLIT_BUCKETS)`` buckets, ``train`` is the rest. A supplied
    ``validation_group_uids`` manifest replaces hash bucketing with exact
    membership and requires every sample to carry an explicit group UID.
    """
    if split not in {"all", "train", "val"}:
        raise ValueError(f"unsupported split {split!r}")
    if split == "all":
        return _KeepAllSplitFilter()
    if validation_group_uids is not None:
        requested = [str(uid) for uid in validation_group_uids]
        validation_groups = frozenset(requested)
        if not validation_groups:
            raise ValueError("validation_group_uids must not be empty")
        if len(validation_groups) != len(requested):
            raise ValueError("validation_group_uids contains duplicates")
        if any(not uid for uid in validation_groups):
            raise ValueError(
                "validation_group_uids must contain non-empty values"
            )
        return _ExplicitSplitGroupFilter(
            validation_groups=validation_groups,
            keep_validation=(split == "val"),
        )
    if val_fraction <= 0.0:
        return _KeepAllSplitFilter()
    buckets = SPLIT_BUCKETS
    val_buckets = max(1, min(buckets - 1, round(val_fraction * buckets)))
    return _HashSplitGroupFilter(
        buckets=buckets,
        validation_buckets=val_buckets,
        keep_validation=(split == "val"),
    )


@dataclass(frozen=True)
class PackedSplitInventory:
    """Identity coverage discovered from packed sample metadata."""

    group_uids: tuple[str, ...]
    sample_count: int
    sample_uid_digest: str
    sample_uids_by_group: tuple[tuple[str, tuple[str, ...]], ...]

    def sample_identity_for_groups(
        self,
        group_uids: Sequence[str],
    ) -> tuple[int, str]:
        """Return the sample count and UID digest for exact group membership."""
        requested = frozenset(str(uid) for uid in group_uids)
        if not requested:
            raise ValueError("at least one split group is required")
        inventory = dict(self.sample_uids_by_group)
        missing = requested - set(inventory)
        if missing:
            raise ValueError(
                f"split groups are absent from packed inventory: {sorted(missing)}"
            )
        sample_uids = sorted(
            sample_uid
            for group_uid in requested
            for sample_uid in inventory[group_uid]
        )
        return (
            len(sample_uids),
            hashlib.sha256(
                "\n".join(sample_uids).encode("utf-8")
            ).hexdigest(),
        )


def discover_split_inventory(
    shard_dirs: Sequence[str | Path],
    *,
    allow_single_group: bool = False,
) -> PackedSplitInventory:
    """Read exact group and sample identities from packed shard metadata.

    This is a one-time training-startup scan over tar headers and ``meta.json``
    members only; camera/map payloads are never decoded. Exact KITScenes
    holdouts use this inventory so their scene count is deterministic rather
    than an approximate hash-bucket fraction.
    """
    import tarfile

    roots = [Path(shard_dir) for shard_dir in shard_dirs]
    if not roots:
        raise ValueError("at least one shard directory is required")

    group_uids: set[str] = set()
    sample_uids: set[str] = set()
    sample_uids_by_group: dict[str, list[str]] = {}
    for root in roots:
        tarfiles = packed_sample_tar_paths(root)
        if not tarfiles:
            raise FileNotFoundError(f"No .tar shards found in {root}")
        for tar_path in tarfiles:
            with tarfile.open(tar_path, "r:*") as archive:
                for member in archive:
                    if not (
                        member.isfile()
                        and member.name.endswith(".meta.json")
                    ):
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError(
                            f"could not read {member.name} from {tar_path}"
                        )
                    try:
                        metadata = json.load(extracted)
                    except (OSError, UnicodeError, ValueError) as error:
                        raise ValueError(
                            f"invalid {member.name} in {tar_path}"
                        ) from error
                    group_uid = metadata.get("split_group_uid")
                    if not isinstance(group_uid, str) or not group_uid:
                        raise ValueError(
                            f"{member.name} in {tar_path} has no valid "
                            "split_group_uid"
                        )
                    sample_uid = metadata.get("sample_uid")
                    member_uid = member.name.removesuffix(".meta.json")
                    if (
                        not isinstance(sample_uid, str)
                        or not sample_uid
                        or sample_uid != member_uid
                    ):
                        raise ValueError(
                            f"{member.name} in {tar_path} has a mismatched "
                            "sample_uid"
                        )
                    if sample_uid in sample_uids:
                        raise ValueError(
                            f"duplicate sample_uid {sample_uid!r} in packed "
                            "shards"
                        )
                    group_uids.add(group_uid)
                    sample_uids.add(sample_uid)
                    sample_uids_by_group.setdefault(
                        group_uid, []
                    ).append(sample_uid)

    if not sample_uids:
        raise ValueError(
            "exact validation splitting requires packed sample metadata"
        )
    if not allow_single_group and len(group_uids) < 2:
        raise ValueError(
            "exact validation splitting requires metadata for at least two "
            "split groups"
        )
    return PackedSplitInventory(
        group_uids=tuple(sorted(group_uids)),
        sample_count=len(sample_uids),
        sample_uid_digest=hashlib.sha256(
            "\n".join(sorted(sample_uids)).encode("utf-8")
        ).hexdigest(),
        sample_uids_by_group=tuple(
            (
                group_uid,
                tuple(sorted(group_sample_uids)),
            )
            for group_uid, group_sample_uids in sorted(
                sample_uids_by_group.items()
            )
        ),
    )


def discover_split_group_uids(
    shard_dirs: Sequence[str | Path],
) -> tuple[str, ...]:
    """Return only the group component of :func:`discover_split_inventory`."""
    return discover_split_inventory(shard_dirs).group_uids


@dataclass(frozen=True)
class _SampleUidFilter:
    """Picklable raw-WebDataset filter for an explicit benchmark manifest."""

    allowed: frozenset[str]

    def __call__(self, sample) -> bool:
        return str(sample.get("__key__", "")) in self.allowed


def _detach_resource_tracker_pipe_from_exec(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: keep the tracker pipe out of exec'd helpers.

    A spawned worker inherits the parent's multiprocessing resource-tracker
    pipe as an inheritable fd. On macOS torch shares worker tensors through the
    ``file_system`` strategy, so the worker execs a ``torch_shm_manager`` daemon
    that would inherit that pipe too. The daemon lives until the parent process
    exits, while the parent's interpreter shutdown (Python 3.12
    ``ResourceTracker.__del__``) waits for the tracker, which only exits once
    every copy of the pipe is closed. That cycle hangs the parent forever at
    exit. Marking the fd non-inheritable in the worker breaks it; the worker
    itself keeps using the pipe.
    """
    del worker_id
    from multiprocessing import resource_tracker

    tracker_fd = getattr(resource_tracker._resource_tracker, "_fd", None)
    if tracker_fd is None:
        return
    try:
        os.set_inheritable(tracker_fd, False)
    except OSError:
        pass


def _sample_pipeline(
    shard_dir: str,
    *,
    split: str,
    val_fraction: float,
    shuffle: int,
    shuffle_seed: int | None,
    shard_files: Sequence[str | Path] | None = None,
    sample_uids: Sequence[str] | None = None,
    validation_group_uids: Sequence[str] | None = None,
    decode_history_frames: bool = True,
    decode_future_frames: bool = True,
    decode_front_camera_fpn: bool = False,
    navigation_repeat_policy: NavigationRepeatPolicy | None = None,
    bev_repeat_policy: BEVClassRepeatPolicy | None = None,
    nodesplitter=None,
    repeat: bool = False,
):
    """Return ``(decoded samples, shard count, AutoBEV manifest or None)``.

    Packs that record the AutoBEV member contract decode with
    ``_decode_autobev_sample``; every other pack keeps ``_decode_sample``.
    """
    shard_root = Path(shard_dir).resolve()
    available_tarfiles = packed_sample_tar_paths(shard_root)
    available_resolved = {
        path.resolve() for path in available_tarfiles
    }
    tarfiles = (
        sorted(Path(path) for path in shard_files)
        if shard_files is not None
        else list(available_tarfiles)
    )
    if not tarfiles:
        raise FileNotFoundError(f"No .tar shards found in {shard_dir}")
    if len(set(tarfiles)) != len(tarfiles):
        raise ValueError("shard files must be unique")
    for path in tarfiles:
        resolved = path.resolve()
        if (
            not resolved.is_file()
            or resolved.parent != shard_root
            or resolved not in available_resolved
        ):
            raise ValueError(
                "shard file must be declared as a sample shard under "
                f"{shard_root}: {path}"
            )
        if resolved.suffix != ".tar":
            raise ValueError(f"shard file must use .tar suffix: {path}")
    if shuffle > 0:
        random.Random(shuffle_seed).shuffle(tarfiles)

    urls = [str(p) for p in tarfiles]

    # webdataset's `workersplitter` already defaults to split_by_worker. Passing
    # split_by_worker as `nodesplitter` too would split twice and silently drop
    # (N-1)/N of the data, so the node slot uses single_node_only.
    dataset = wds.WebDataset(
        urls,
        shardshuffle=False,
        empty_check=False,
        nodesplitter=nodesplitter or wds.single_node_only,
    )
    if repeat:
        dataset = dataset.repeat()
    if sample_uids is not None:
        requested = [str(uid) for uid in sample_uids]
        allowed = frozenset(requested)
        if not allowed:
            raise ValueError("sample_uids must not be empty")
        if len(allowed) != len(requested):
            raise ValueError("sample_uids contains duplicates")
        dataset = dataset.select(_SampleUidFilter(allowed))
    # Split BEFORE decode so dropped groups never incur image decoding.
    keep = _split_keep(
        split,
        val_fraction,
        validation_group_uids=validation_group_uids,
    )
    if split != "all" and (
        val_fraction > 0.0 or validation_group_uids is not None
    ):
        dataset = dataset.select(keep)
    if navigation_repeat_policy is not None:
        if split != "train":
            raise ValueError(
                "navigation repeat policy is valid only for the train split"
            )
        dataset = dataset.compose(navigation_repeat_policy)
    if bev_repeat_policy is not None:
        if split != "train":
            raise ValueError(
                "BEV repeat policy is valid only for the train split"
            )
        if navigation_repeat_policy is not None:
            raise ValueError(
                "BEV and navigation repeat policies cannot be combined"
            )
        dataset = dataset.compose(bev_repeat_policy)
    if shuffle > 0:
        dataset = dataset.shuffle(shuffle, seed=shuffle_seed)
    autobev = read_autobev_manifest(shard_dir)
    # functools.partial rather than a lambda so the map fn pickles to workers.
    decode = (
        functools.partial(
            _decode_autobev_sample,
            planner_supervised=autobev["sample_kind"] == "planner",
        )
        if autobev is not None
        else functools.partial(
            _decode_sample,
            # The frame pool is a sibling pool/ dir outside `urls`, so worker
            # sharding never hides frames from a worker.
            pool=_make_pool_accessor(shard_dir),
            decode_history_frames=decode_history_frames,
            decode_future_frames=decode_future_frames,
            decode_front_camera_fpn=decode_front_camera_fpn,
        )
    )
    return dataset.map(decode), len(tarfiles), autobev


def _web_loader(
    dataset,
    *,
    shard_count: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    drop_last: bool,
    collate_fn,
) -> wds.WebLoader:
    # More workers than shards is wasted. Partition-scoped loaders are retired
    # once exhausted, so workers must not persist beyond the iterator lifetime.
    eff_workers = min(num_workers, shard_count) if num_workers > 0 else 0
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "drop_last": bool(drop_last),
        "num_workers": eff_workers,
    }
    if eff_workers > 0:
        loader_kwargs.update(
            persistent_workers=False,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            worker_init_fn=_detach_resource_tracker_pipe_from_exec,
        )
    return wds.WebLoader(dataset, **loader_kwargs)


def make_pre_extracted_loader(
    shard_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    split: str = "all",
    val_fraction: float = 0.0,
    shuffle: int = 1000,
    shuffle_seed: int | None = None,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    shard_files: Sequence[str | Path] | None = None,
    sample_uids: Sequence[str] | None = None,
    validation_group_uids: Sequence[str] | None = None,
    decode_history_frames: bool = True,
    decode_future_frames: bool = True,
    decode_front_camera_fpn: bool = False,
    navigation_repeat_policy: NavigationRepeatPolicy | None = None,
    bev_repeat_policy: BEVClassRepeatPolicy | None = None,
    drop_last: bool = False,
    nodesplitter=None,
) -> wds.WebLoader:
    """Create a WebDataset DataLoader reading from local EBS shard cache.

    Args:
        shard_dir: Path to directory containing .tar shard files.
        batch_size: Batch size.
        num_workers: DataLoader workers. Workers are sharded over the .tar
            files, so effective parallelism is capped by shard count.
        split: ``"all"`` (default, every sample), ``"train"``, or ``"val"``. With
            ``val_fraction`` > 0, ``train``/``val`` are disjoint at
            ``split_group_uid`` granularity (see ``_split_keep``).
        val_fraction: fraction of groups held out for ``val`` (0 disables the
            split → ``"all"`` behaviour regardless of ``split``).
        shuffle: Shuffle buffer size (0 to disable).
        shuffle_seed: optional deterministic seed for the shuffle buffer.
        pin_memory: pin host buffers for faster H2D copy (set True on GPU).
        prefetch_factor: batches prefetched per worker (only used when
            num_workers>0); overlaps decode with the GPU step.
        shard_files: optional explicit subset of tar files. Overlay precompute
            uses one file at a time so each output body is canonical per shard.
        sample_uids: optional exact sample allowlist. Filtering happens before
            image decode so a fixed benchmark manifest does not decode unrelated
            samples from the same source scenes.
        validation_group_uids: optional frozen group-level validation manifest.
            When supplied with ``split="train"`` or ``"val"``, membership in
            this exact set replaces approximate hash bucketing.
        decode_history_frames: decode World-Model history images. Inference
            paths that use only the current camera frame disable this to avoid
            requiring the sibling frame pool.
        decode_future_frames: decode World-Model target images. Benchmark
            inference disables this so future camera frames cannot enter its
            input batch; training keeps the default because JEPA needs them.
        decode_front_camera_fpn: decode the exact base-resolution Front
            companion required by the stateful T8 inference cache. Training
            and stateless inference leave this disabled to avoid unused image
            decode, worker IPC, and device transfer.
        navigation_repeat_policy: optional raw-sample repeat transform. Training
            applies it after split filtering and before shuffle/decode.
        bev_repeat_policy: optional BEV rare-class repeat transform. It is
            mutually exclusive with navigation repetition and train-only.
        drop_last: discard an incomplete final batch. Fixed-step DDP training
            enables this so every optimizer step has the same sample scale.
        nodesplitter: optional WebDataset node splitter. Distributed callers
            with explicit rank-owned shards use ``passthrough_nodesplitter``;
            the default rejects accidental multi-node iteration.

    The returned loader carries two extra attributes describing the dataset's
    geometry (a rig constant, so it lives on the loader, not per batch):
      - ``.projection``: a CameraProjectionModel operator, or None (pseudo).
      - ``.geometry_type``: "pinhole" / "rectified_pinhole" / "ftheta" / "pseudo".
    Pass these to the model's forward alongside each batch.
    """
    dataset, shard_count, autobev = _sample_pipeline(
        shard_dir,
        split=split,
        val_fraction=val_fraction,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        shard_files=shard_files,
        sample_uids=sample_uids,
        validation_group_uids=validation_group_uids,
        decode_history_frames=decode_history_frames,
        decode_future_frames=decode_future_frames,
        decode_front_camera_fpn=decode_front_camera_fpn,
        navigation_repeat_policy=navigation_repeat_policy,
        bev_repeat_policy=bev_repeat_policy,
        nodesplitter=nodesplitter,
    )
    loader = _web_loader(
        dataset,
        shard_count=shard_count,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        collate_fn=collate_autobev_batch if autobev is not None else None,
    )
    # Per-dataset geometry, reconstructed once from the manifest. AutoBEV
    # packs carry per-sample pinhole projections to the cropped pixels.
    projection, geometry_type = (
        (None, AUTOBEV_GEOMETRY_TYPE)
        if autobev is not None
        else load_projection_from_manifest(shard_dir)
    )
    loader.projection = projection
    loader.geometry_type = geometry_type
    return loader




@dataclass
class _ActiveLoader:
    loader: object
    iterator: object
    owned: bool
    closed: bool = False

    def close(self):
        """Release an active child iterator and its owned loader exactly once."""
        if self.closed:
            return
        self.closed = True
        iterator_close = getattr(self.iterator, "close", None)
        try:
            if iterator_close is not None:
                iterator_close()
        finally:
            if self.owned:
                loader_close = getattr(self.loader, "close", None)
                if loader_close is not None:
                    loader_close()


class MergedDatasetLoader:
    """Bounded round-robin over multiple single-dataset loaders.

    Datasets differ in camera count and geometry, so every batch stays
    same-dataset. Each yielded item is ``(batch, projection, geometry_type)``.

    Only ``max_active_loaders`` child iterators exist at once; an exhausted
    child is closed before the next pending one opens. Loader factories are
    invoked lazily per epoch, which bounds worker and prefetch state even with
    hundreds of partitions.
    """

    def __init__(
        self,
        loaders=None,
        *,
        loader_factories=None,
        max_active_loaders: int = 4,
        num_workers: int = 0,
        shuffle_seed: int | None = None,
    ):
        if loaders is not None and loader_factories is not None:
            raise ValueError("pass loaders or loader_factories, not both")
        if loaders is None and loader_factories is None:
            raise ValueError("MergedDatasetLoader needs at least one loader.")
        if max_active_loaders <= 0:
            raise ValueError("max_active_loaders must be positive")

        if loader_factories is not None:
            sources = list(loader_factories)
            owned = True
            self.loaders = []
        else:
            sources = list(loaders)
            owned = False
            self.loaders = sources
        if not sources:
            raise ValueError("MergedDatasetLoader needs at least one loader.")
        self._sources = [(source, owned) for source in sources]
        self.max_active_loaders = min(max_active_loaders, len(sources))
        self.num_workers = num_workers
        self.shuffle_seed = shuffle_seed

    @staticmethod
    def _open(source, owned: bool) -> _ActiveLoader:
        loader = source() if owned else source
        try:
            iterator = iter(loader)
        except BaseException:
            if owned:
                loader_close = getattr(loader, "close", None)
                if loader_close is not None:
                    loader_close()
            raise
        return _ActiveLoader(loader=loader, iterator=iterator, owned=owned)

    def __iter__(self):
        sources = list(self._sources)
        if self.shuffle_seed is not None:
            random.Random(self.shuffle_seed).shuffle(sources)
        pending = iter(sources)
        active: deque[_ActiveLoader] = deque()

        def fill_active():
            while len(active) < self.max_active_loaders:
                try:
                    source, owned = next(pending)
                except StopIteration:
                    return
                active.append(self._open(source, owned))

        try:
            fill_active()
            while active:
                child = active.popleft()
                try:
                    batch = next(child.iterator)
                except StopIteration:
                    child.close()
                    fill_active()
                    continue
                except BaseException:
                    child.close()
                    raise

                # Requeue before yielding so generator.close() also reaches this
                # child when the consumer stops after the current batch.
                active.append(child)
                yield (
                    batch,
                    getattr(child.loader, "projection", None),
                    getattr(child.loader, "geometry_type", "pseudo"),
                )
        finally:
            close_error = None
            while active:
                try:
                    active.popleft().close()
                except BaseException as error:
                    if close_error is None:
                        close_error = error
            if close_error is not None:
                raise close_error


def make_multi_dataset_loader(
    shard_dirs,
    batch_size: int = 8,
    num_workers: int = 4,
    split: str = "all",
    val_fraction: float = 0.0,
    shuffle: int = 1000,
    shuffle_seed: int | None = None,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    max_active_loaders: int | None = None,
    sample_uids: Sequence[str] | None = None,
    validation_group_uids: Sequence[str] | None = None,
    decode_history_frames: bool = True,
    decode_future_frames: bool = True,
    decode_front_camera_fpn: bool = False,
    navigation_repeat_policy: NavigationRepeatPolicy | None = None,
    bev_repeat_policy: BEVClassRepeatPolicy | None = None,
    drop_last: bool = False,
    nodesplitter=None,
) -> MergedDatasetLoader:
    """Build a :class:`MergedDatasetLoader` over several shard directories.

    Each directory is one dataset with its own manifest and geometry. Split
    and decode options are forwarded to every child (see
    make_pre_extracted_loader). ``num_workers`` is a global budget: each active
    partition gets one worker, and at most four partition loaders are active.
    """
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if max_active_loaders is not None and max_active_loaders <= 0:
        raise ValueError("max_active_loaders must be positive")

    shard_dirs = list(shard_dirs)
    child_workers = 1 if num_workers > 0 else 0
    default_active = min(4, num_workers) if num_workers > 0 else 1
    active_limit = default_active if max_active_loaders is None else max_active_loaders
    if num_workers > 0:
        active_limit = min(active_limit, num_workers, 4)
    else:
        active_limit = 1

    factories = [
        functools.partial(
            make_pre_extracted_loader,
            d,
            batch_size=batch_size,
            num_workers=child_workers,
            split=split,
            val_fraction=val_fraction,
            shuffle=shuffle,
            shuffle_seed=(
                None if shuffle_seed is None else shuffle_seed + index
            ),
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            sample_uids=sample_uids,
            validation_group_uids=validation_group_uids,
            decode_history_frames=decode_history_frames,
            decode_future_frames=decode_future_frames,
            decode_front_camera_fpn=decode_front_camera_fpn,
            navigation_repeat_policy=navigation_repeat_policy,
            bev_repeat_policy=bev_repeat_policy,
            drop_last=drop_last,
            nodesplitter=nodesplitter,
        )
        for index, d in enumerate(shard_dirs)
    ]
    merged = MergedDatasetLoader(
        loader_factories=factories,
        max_active_loaders=active_limit,
        num_workers=num_workers,
        shuffle_seed=shuffle_seed,
    )
    return merged


def _packed_sample_count(shard_dir: str | Path) -> int:
    """Samples in the shards this directory exposes (the rank's, when staged)."""
    manifest = json.loads((Path(shard_dir) / "manifest.json").read_text())
    counts = manifest["shard_sample_counts"]
    return sum(int(counts[path.name]) for path in packed_sample_tar_paths(shard_dir))


def _make_autobev_mixed_loader(
    shard_dirs: Sequence[str],
    weights: Sequence[float],
    *,
    repeat: Sequence[bool],
    longest: bool,
    batch_size: int,
    num_workers: int,
    split: str,
    val_fraction: float,
    shuffle: int,
    shuffle_seed: int | None,
    pin_memory: bool,
    prefetch_factor: int,
    drop_last: bool,
    nodesplitter,
) -> MergedDatasetLoader:
    """Draw each sample from one AutoBEV pack with probability ``weights``.

    Every pack must carry the AutoBEV member contract, so nuPlan and
    BEV-only samples share one batch layout. Each worker reads a slice of
    every pack, so the workers are capped by the smallest shard count.
    """
    pipelines = []
    shard_counts = []
    for index, (shard_dir, repeated) in enumerate(zip(shard_dirs, repeat, strict=True)):
        dataset, shard_count, autobev = _sample_pipeline(
            shard_dir,
            split=split,
            val_fraction=val_fraction,
            shuffle=shuffle,
            shuffle_seed=None if shuffle_seed is None else shuffle_seed + index,
            nodesplitter=nodesplitter,
            repeat=repeated,
        )
        if autobev is None:
            raise ValueError(f"{shard_dir} does not carry the AutoBEV members")
        pipelines.append(dataset)
        shard_counts.append(shard_count)
    loader = _web_loader(
        wds.RandomMix(pipelines, probs=list(weights), longest=longest),
        shard_count=min(shard_counts),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        collate_fn=collate_autobev_batch,
    )
    loader.projection = None
    loader.geometry_type = AUTOBEV_GEOMETRY_TYPE
    return MergedDatasetLoader(loaders=[loader], num_workers=num_workers)


def make_autobev_bev_share_loader(
    primary_dirs: Sequence[str],
    bev_only_dirs: Sequence[str],
    *,
    batch_size: int = 8,
    num_workers: int = 4,
    split: str = "train",
    val_fraction: float = 0.0,
    shuffle: int = 1000,
    shuffle_seed: int | None = None,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    drop_last: bool = False,
    nodesplitter=None,
) -> MergedDatasetLoader:
    """Stage 2 loader: primary samples with ``AUTOBEV_BEV_ONLY_SHARE`` BEV-only.

    Within each group the packs are weighted by their sample counts. The
    BEV-only packs repeat, so one pass over the primary packs ends the epoch.
    Yields ``(batch, projection, geometry_type)`` like ``make_multi_dataset_loader``.
    """
    primary_counts = [_packed_sample_count(path) for path in primary_dirs]
    bev_only_counts = [_packed_sample_count(path) for path in bev_only_dirs]
    weights = [
        (1.0 - AUTOBEV_BEV_ONLY_SHARE) * count / sum(primary_counts)
        for count in primary_counts
    ] + [
        AUTOBEV_BEV_ONLY_SHARE * count / sum(bev_only_counts)
        for count in bev_only_counts
    ]
    return _make_autobev_mixed_loader(
        [*primary_dirs, *bev_only_dirs],
        weights,
        repeat=[False] * len(primary_dirs) + [True] * len(bev_only_dirs),
        longest=False,
        batch_size=batch_size,
        num_workers=num_workers,
        split=split,
        val_fraction=val_fraction,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        nodesplitter=nodesplitter,
    )


def make_autobev_sample_count_loader(
    shard_dirs: Sequence[str],
    *,
    batch_size: int = 8,
    num_workers: int = 4,
    split: str = "train",
    val_fraction: float = 0.0,
    shuffle: int = 1000,
    shuffle_seed: int | None = None,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    drop_last: bool = False,
    nodesplitter=None,
) -> MergedDatasetLoader:
    """Stage 1 loader: AutoBEV packs mixed in proportion to their sample counts.

    The epoch ends when every pack is read once. Yields
    ``(batch, projection, geometry_type)`` like ``make_multi_dataset_loader``.
    """
    return _make_autobev_mixed_loader(
        shard_dirs,
        [_packed_sample_count(path) for path in shard_dirs],
        repeat=[False] * len(shard_dirs),
        longest=True,
        batch_size=batch_size,
        num_workers=num_workers,
        split=split,
        val_fraction=val_fraction,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        nodesplitter=nodesplitter,
    )
