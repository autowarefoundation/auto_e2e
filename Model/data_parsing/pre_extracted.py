"""PreExtractedDataset: WebDataset-backed DataLoader for training.

Reads from local EBS shard cache (init container syncs from S3).
No video decode, no lerobot dependency. Sequential tar reads at full
disk bandwidth.

Usage:
    from data_parsing.pre_extracted import make_pre_extracted_loader

    loader = make_pre_extracted_loader("/data/shards", batch_size=8)
    for batch in loader:
        # batch["visual_tiles"]       (B, V, 3, 256, 256)  V real cameras
        # batch["map_context"]        (B, C_map, 256, 256) semantic map
        # batch["route_mask"]         (B, 2, 256, 256)     selected route
        # batch["map_valid"]          (B,)                  explicit validity
        # batch["route_valid"]        (B,)                  explicit validity
        # batch["route_supervision"]  dict of loss-only route fields
        # batch["egomotion_history"]  (B, 256)
        # batch["visual_history"]     (B, 896)
        # batch["trajectory_target"]  (B, 128)
        # batch["camera_params"]      (B, V, 3, 4)         if the manifest has calib
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torchvision import transforms

_HISTORY_STEPS = 64
_FUTURE_STEPS = 64
_HISTORY_SIGNALS = 4
_TARGET_SIGNALS = 2
_VISUAL_HISTORY_DIM = 896

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Camera frames are keyed "cam_<i>.jpg"; the nav-map is "map.jpg". The map MUST
# NOT be picked up as a camera view — matching cam_ explicitly (not any ".jpg")
# keeps V correct and stops the map being double-counted in the BEV projection.
_CAM_KEY_RE = re.compile(r"^cam_\d+\.jpg$")
# World-Model window frames: hist_<t>_cam_<v>.jpg / fut_<f>_cam_<v>.jpg (#13).
_HIST_KEY_RE = re.compile(r"^hist_(\d+)_cam_(\d+)\.jpg$")
_FUT_KEY_RE = re.compile(r"^fut_(\d+)_cam_(\d+)\.jpg$")

NAVIGATION_REPEAT_POLICY_VERSION = "navigation_repeat_v1"
_DECISIVE_ROUTE_MANEUVERS = frozenset({
    "left",
    "right",
    "u_turn",
    "merge",
    "exit",
})


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
        tarfiles = sorted(root.glob("*.tar"))
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


def _decode_image(data) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)) if isinstance(data, bytes) else data
    return _TRANSFORM(img)


class _PoolAccessor:
    """Path-based frame-pool reader (#121 §3.4d): ``frame_id -> jpeg bytes``.

    Reads ``<pool_dir>/<frame_id>.jpg`` on demand. Path-based (not an open handle)
    so it pickles cleanly to spawn DataLoader workers; the OS page cache shares the
    bytes across workers/epochs. Returns None if the pool dir is absent (a shard
    with no deduped windows), so the loader falls back to the legacy layout.
    """

    def __init__(self, pool_dir: str):
        self.pool_dir = pool_dir

    def __call__(self, frame_id: str) -> bytes:
        with open(f"{self.pool_dir}/{frame_id}.jpg", "rb") as f:
            return f.read()


def _make_pool_accessor(shard_dir: str):
    """Return a ``_PoolAccessor`` if ``<shard_dir>/pool/`` exists, else None."""
    pool_dir = Path(shard_dir) / "pool"
    return _PoolAccessor(str(pool_dir)) if pool_dir.is_dir() else None


def _decode_sample(
    sample: dict,
    pool=None,
    *,
    decode_future_frames: bool = True,
) -> dict:
    """Decode a WebDataset sample into training tensors (geometry-free).

    Calibration is a per-dataset rig constant, not per-sample, so it is NOT
    decoded here — it is reconstructed once by ``make_pre_extracted_loader`` and
    exposed on the loader as ``.projection`` / ``.geometry_type``.

    ``pool`` is a frame-pool accessor (``frame_id -> jpeg bytes``) for shards packed
    with the deduped WM window (#121 §3.4d): the sample carries a
    ``window_index.json`` mapping (step,view)→frame_id and the pixels live in a
    sibling ``pool/`` dir. None on shards without a pool (imitation-only / legacy).
    """
    # Keys: "cam_0.jpg" ... "cam_{V-1}.jpg", schema-v8 navigation
    # members or a legacy optional "map.jpg", plus numeric/metadata members.
    cam_keys = sorted(
        (k for k in sample if _CAM_KEY_RE.match(k)),
        key=lambda k: int(k[len("cam_"):-len(".jpg")]),
    )
    frames = [_decode_image(sample[k]) for k in cam_keys]

    navigation_base_keys = {
        "map_semantic.npz",
        "route_mask.npz",
        "navigation_meta.json",
    }
    navigation_keys = navigation_base_keys | {"route_supervision.npz"}
    present_navigation = navigation_keys.intersection(sample)
    if present_navigation and not navigation_base_keys.issubset(
        present_navigation
    ):
        raise ValueError(
            "navigation members must contain the complete schema-v8 set "
            "or the sample_navigation_v3 map, route, and metadata set"
        )
    if present_navigation:
        from navigation.artifacts import (
            decode_route_supervision,
            decode_sample_navigation,
        )

        map_array, route_array, navigation_metadata = (
            decode_sample_navigation(sample)
        )
        supervision = (
            decode_route_supervision(sample)
            if "route_supervision.npz" in sample
            else None
        )
        from navigation.geometry import (
            AUTOE2E_NAVIGATION_GEOMETRY,
            DEFAULT_NAVIGATION_GEOMETRY,
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
        route_mask = torch.from_numpy(
            route_array.astype(np.float32, copy=True)
        )
        map_valid = torch.tensor(
            bool(navigation_metadata["map_valid"]),
            dtype=torch.bool,
        )
        route_valid = torch.tensor(
            bool(navigation_metadata["route_valid"]),
            dtype=torch.bool,
        )
        raw_channel_valid = navigation_metadata.get(
            "route_channel_valid"
        )
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
        shape = map_context.shape[-2:]
        if supervision is not None:
            route_supervision = {
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
        else:
            route_supervision = {
                "distance_to_corridor_m": torch.zeros(shape),
                "distance_to_drivable_m": torch.zeros(shape),
                "route_heading_sin": torch.zeros(shape),
                "route_heading_cos": torch.zeros(shape),
                "route_heading_valid": torch.zeros(
                    shape,
                    dtype=torch.bool,
                ),
                "destination_xy_m": torch.zeros(2),
                "destination_visible": torch.tensor(False),
                "available": torch.tensor(False),
                "drivable_available": torch.tensor(False),
            }
    else:
        # L2D keeps its existing RGB map contract during this KITScenes
        # milestone. NVIDIA and map-less shards receive explicit invalid inputs.
        if "map.jpg" in sample:
            map_context = _decode_image(sample["map.jpg"])
            map_valid = torch.tensor(True, dtype=torch.bool)
        else:
            ref = frames[0] if frames else torch.zeros(3, 256, 256)
            map_context = torch.zeros_like(ref)
            map_valid = torch.tensor(False, dtype=torch.bool)
        route_mask = torch.zeros(
            2,
            map_context.shape[-2],
            map_context.shape[-1],
            dtype=torch.float32,
        )
        route_valid = torch.tensor(False, dtype=torch.bool)
        route_channel_valid = torch.zeros(2, dtype=torch.bool)
        navigation_metadata = {}
        shape = map_context.shape[-2:]
        route_supervision = {
            "distance_to_corridor_m": torch.zeros(
                shape,
                dtype=torch.float32,
            ),
            "distance_to_drivable_m": torch.zeros(
                shape,
                dtype=torch.float32,
            ),
            "route_heading_sin": torch.zeros(
                shape,
                dtype=torch.float32,
            ),
            "route_heading_cos": torch.zeros(
                shape,
                dtype=torch.float32,
            ),
            "route_heading_valid": torch.zeros(
                shape,
                dtype=torch.bool,
            ),
            "destination_xy_m": torch.zeros(2, dtype=torch.float32),
            "destination_visible": torch.tensor(
                False,
                dtype=torch.bool,
            ),
            "available": torch.tensor(False, dtype=torch.bool),
            "drivable_available": torch.tensor(
                False,
                dtype=torch.bool,
            ),
        }

    # Ego: raw bytes → numpy → split into history and future
    ego_bytes = sample.get("ego.npy", b"")
    if isinstance(ego_bytes, bytes) and len(ego_bytes) > 0:
        ego = np.frombuffer(ego_bytes, dtype=np.float32).copy()
    else:
        ego = np.zeros(384, dtype=np.float32)

    # History: (64, 4) flattened = 256; Future: (64, 2) flattened = 128
    history_size = _HISTORY_STEPS * _HISTORY_SIGNALS
    ego_history = torch.from_numpy(ego[:history_size])
    ego_future = torch.from_numpy(ego[history_size:])
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
        trajectory_valid_tensor = torch.from_numpy(
            trajectory_valid.copy()
        )
    else:
        trajectory_xy_m = torch.zeros(
            _FUTURE_STEPS,
            2,
            dtype=torch.float32,
        )
        trajectory_valid_tensor = torch.zeros(
            _FUTURE_STEPS,
            dtype=torch.bool,
        )

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
    camera_projection_matrix = None
    camera_geometry_type = None
    if "calib.json" in sample:
        calibration = _json_mapping(
            sample["calib.json"],
            member_name="calib.json",
        )
        projection_spec = calibration.get("projection")
        geometry_label = calibration.get("geometry_type")
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

    out = {
        # Overlay inference derives noise from this stable identity. Keep it in
        # every batch so predictions do not depend on batch position or size.
        "sample_uid": sample.get("__key__", ""),
        "split_group_uid": split_group_uid,
        "visual_tiles": torch.stack(frames),
        "map_context": map_context,
        "route_mask": route_mask,
        "map_valid": map_valid,
        "route_valid": route_valid,
        "route_channel_valid": route_channel_valid,
        "route_supervision": route_supervision,
        "navigation_metadata": navigation_metadata,
        "egomotion_history": ego_history,
        "visual_history": torch.zeros(_VISUAL_HISTORY_DIM),
        "trajectory_target": ego_future,
        "trajectory_xy_m": trajectory_xy_m,
        "trajectory_valid": trajectory_valid_tensor,
        "initial_speed_mps": ego_history.reshape(
            _HISTORY_STEPS,
            _HISTORY_SIGNALS,
        )[-1, 0],
        "bev_segmentation_target": bev_segmentation_target,
        "bev_segmentation_valid": bev_segmentation_valid,
        "bev_segmentation_available": bev_segmentation_available,
    }
    if camera_projection_matrix is not None:
        out["camera_projection_matrix"] = camera_projection_matrix
        out["camera_geometry_type"] = camera_geometry_type

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

    # Optional World-Model windows (#13/#3.4d): the sample carries window_index.json
    # (a (step,view)→frame_id map); the frames themselves are in the sibling pool/.
    # Rebuild history_frames [T, V, 3, H, W] and future_frames [F, V, 3, H, W]
    # (oldest→newest) — IDENTICAL tensors to the old per-sample hist_/fut_ layout,
    # just deduped in storage. Present only on WM shards; absent → no JEPA loss.
    # (Legacy hist_/fut_ shards still decode via _decode_window_legacy for back-compat.)
    windows = _decode_windows_from_pool(
        sample,
        pool,
        decode_future_frames=decode_future_frames,
    )
    if windows is not None:
        history_frames, future_frames = windows
        out["history_frames"] = history_frames
        if future_frames is not None:
            out["future_frames"] = future_frames

    # Optional reasoning labels (#98): a per-sample "reasoning.json" member holds
    # a serialized ReasoningLabelRecord (same shard key → auto-aligned with this
    # sample's frames, no sample_id join). Decode it to per-sample target tensors
    # for HorizonReasoningLoss, flattened to top-level "reasoning__*" keys so
    # WebDataset's per-key default collation stacks them into [B, ...] batches.
    # Absent on shards packed without a teacher — the loader stays
    # reasoning-agnostic and training skips the reasoning loss.
    # ALWAYS emit reasoning__* keys so a batch that mixes labeled + unlabeled
    # samples collates (default_collate needs identical keys across a batch). An
    # unlabeled sample gets a fully-MASKED target (abstained record → IGNORE_INDEX
    # / zero source_weight), so it contributes nothing to the reasoning loss —
    # never a false-negative all-zero row. Shards packed with a teacher carry
    # reasoning.json; imitation-only samples don't, and both must batch together.
    reasoning_data = sample.get("reasoning.json")
    for key, tensor in _decode_reasoning_targets(reasoning_data).items():
        out[f"reasoning__{key}"] = tensor

    return out


def _decode_reasoning_targets(data) -> dict:
    """Decode the reasoning.json member into per-sample target tensors (#98).

    Lazy imports the data_processing tensorizer so importing this loader never
    pulls the label package unless training touches reasoning. When ``data`` is
    None (sample has no reasoning.json), return the tensors of an ABSTAINED
    record — all IGNORE_INDEX / zero source_weight — so the sample batches with
    labeled ones and is fully masked out of the reasoning loss (R9).
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
    """Decode one window (history or future) from a ``[[frame_id/view] /step]`` index.

    Looks each frame_id up in the pool accessor, decodes, and stacks into
    ``[steps, V, 3, H, W]`` (oldest→newest) — the exact shape/order the model
    consumes. Byte-identical to the old per-sample layout because the pool holds the
    same JPEG bytes the packer produced.
    """
    frame_steps = []
    for view_ids in index_steps:                       # one list of frame_ids per step
        view_frames = [_decode_image(pool(fid)) for fid in view_ids]
        frame_steps.append(torch.stack(view_frames))   # [V, 3, H, W]
    return torch.stack(frame_steps)                     # [steps, V, 3, H, W]


def _decode_windows_from_pool(
    sample: dict,
    pool,
    *,
    decode_future_frames: bool = True,
):
    """Rebuild (history_frames, future_frames) from window_index.json + the pool.

    Returns None when the sample has no window_index.json (imitation-only). Falls
    back to the LEGACY per-sample hist_/fut_ member layout when a shard predates the
    frame pool (so old shards still train). Requires a pool accessor when a
    window_index.json is present.
    """
    idx_blob = sample.get("window_index.json")
    if idx_blob is None:
        return _decode_windows_legacy(
            sample,
            decode_future_frames=decode_future_frames,
        )
    if pool is None:
        raise ValueError(
            "sample has window_index.json but the loader has no frame pool accessor; "
            "the sibling pool/ dir must exist next to the shards (#121 §3.4d).")
    index = json.loads(idx_blob.decode() if isinstance(idx_blob, (bytes, bytearray)) else idx_blob)
    hist = _decode_window_from_index(index["history"], pool)
    fut = (
        _decode_window_from_index(index["future"], pool)
        if decode_future_frames
        else None
    )
    return hist, fut


def _decode_windows_legacy(
    sample: dict,
    *,
    decode_future_frames: bool = True,
):
    """Legacy path: decode hist_<t>_cam_<v>.jpg / fut_<f>_cam_<v>.jpg members
    (pre-#3.4d shards). Returns (history, future) or None if absent."""
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
    hist = _one(_HIST_KEY_RE)
    fut = _one(_FUT_KEY_RE) if decode_future_frames else None
    if hist is None or (decode_future_frames and fut is None):
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

    A dataset without calibration (pseudo geometry, e.g. L2D) returns
    ``(None, "pseudo")`` and the caller runs the explicit pseudo path. This is
    the single geometry-reconstruction point, keeping the pinhole/f-theta split
    out of the training loop.
    """
    mpath = Path(shard_dir) / "manifest.json"
    # Missing manifest -> pseudo (a legacy shard has no geometry). But a manifest
    # that EXISTS and cannot be read must RAISE: silently degrading a calibrated
    # run to pseudo geometry would corrupt experiments. Corrupt/unreadable is a
    # hard error, not a fallback.
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
        # fw_poly may be serialized as a shared [K] (flat list) or per-view [V,K]
        # (nested list) — to_spec keeps a shared vector whole. Reconstruct the
        # matching shape so to_spec/load round-trip is exact: shared -> [K],
        # per-view -> [1,V,K].
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


def _split_bucket(key: str, buckets: int = 10) -> int:
    """Deterministic bucket in [0, buckets) from a stable string.

    Uses a fixed hash (blake2b) — NOT Python's ``hash()``, which is salted per
    process, so train and eval workers (and reruns) would disagree on the split.
    Reproducible across the train task and the (separate) eval task.
    """
    from data_processing.dataset_snapshot import split_bucket
    return split_bucket(key, buckets)


def _split_group_of(sample, *, required: bool = False) -> str:
    """The train/val SPLIT key for a raw shard sample (#121 §3.1).

    Hash the ``split_group_uid`` from the sample's ``meta.json`` (episode/clip
    granularity) — NOT the per-frame ``__key__`` — so all frames of one episode
    fall in the SAME bucket and never straddle train/val (adjacent frames are
    strongly correlated → a per-frame split leaks). Falls back to ``__key__`` for
    legacy shards whose meta.json predates split_group_uid.
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


def _split_keep(
    split: str,
    val_fraction: float,
    validation_group_uids: Sequence[str] | None = None,
):
    """Return a predicate ``sample -> bool`` selecting the requested split.

    ``split="all"`` (default) keeps everything (backward-compatible, single-set
    behaviour). ``"train"`` / ``"val"`` partition by a stable hash of the sample's
    ``split_group_uid`` (episode/clip) into disjoint sets: ``val`` is the first
    ``round(val_fraction*10)`` of 10 buckets, ``train`` is the rest. Splitting by
    GROUP (not per-frame) keeps a whole episode/clip on one side, so eval-on-``val``
    measures generalization to UNSEEN episodes, not memorization of neighbours.
    A supplied ``validation_group_uids`` manifest replaces hash bucketing with
    exact membership and requires every sample to carry an explicit group UID.
    """
    if split not in {"all", "train", "val"}:
        raise ValueError(f"unsupported split {split!r}")
    if split == "all":
        return lambda sample: True
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
        return lambda sample: True
    buckets = 10
    val_buckets = max(1, min(buckets - 1, round(val_fraction * buckets)))

    def keep(sample):
        b = _split_bucket(_split_group_of(sample), buckets)
        in_val = b < val_buckets
        return in_val if split == "val" else (not in_val)

    return keep


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
        tarfiles = sorted(root.glob("*.tar"))
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

    if not sample_uids or len(group_uids) < 2:
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
    decode_future_frames: bool = True,
    navigation_repeat_policy: NavigationRepeatPolicy | None = None,
) -> wds.WebLoader:
    """Create a WebDataset DataLoader reading from local EBS shard cache.

    Args:
        shard_dir: Path to directory containing .tar shard files.
        batch_size: Batch size.
        num_workers: DataLoader workers. >0 decodes JPEGs in parallel worker
            processes (the per-sample WM window is ~55 decodes; at num_workers=0
            this is fully serial and the GPU stalls — #121 P0). Workers are
            sharded over the .tar files via ``split_by_worker``, so effective
            parallelism is capped by shard count — pack more, smaller shards to use
            more workers.
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
        decode_future_frames: decode World-Model target images. Benchmark
            inference disables this so future camera frames cannot enter its
            input batch; training keeps the default because JEPA needs them.
        navigation_repeat_policy: optional raw-sample repeat transform. Training
            applies it after split filtering and before shuffle/decode.

    The returned loader carries two extra attributes describing the dataset's
    geometry (a rig constant, so it lives on the loader, not per batch):
      - ``.projection``: a CameraProjectionModel operator, or None (pseudo).
      - ``.geometry_type``: "pinhole" / "rectified_pinhole" / "ftheta" / "pseudo".
    Pass these to the model's forward alongside each batch.
    """
    tarfiles = (
        sorted(Path(path) for path in shard_files)
        if shard_files is not None
        else sorted(Path(shard_dir).glob("*.tar"))
    )
    if not tarfiles:
        raise FileNotFoundError(f"No .tar shards found in {shard_dir}")
    shard_root = Path(shard_dir).resolve()
    for path in tarfiles:
        resolved = path.resolve()
        if not resolved.is_file() or resolved.parent != shard_root:
            raise ValueError(
                f"shard file must be a direct .tar child of {shard_root}: {path}"
            )
        if resolved.suffix != ".tar":
            raise ValueError(f"shard file must use .tar suffix: {path}")

    urls = [str(p) for p in tarfiles]

    # CRITICAL (webdataset 1.0.2): WebDataset has BOTH `nodesplitter` and
    # `workersplitter`, and `workersplitter` DEFAULTS to split_by_worker. Passing
    # nodesplitter=split_by_worker applies the worker split TWICE, so with
    # num_workers=N each worker sees only 1/N of the shards → the loader silently
    # drops (N-1)/N of the data (verified: 48 samples → 24 at nw=2, 12 at nw=4).
    # Use single_node_only for the NODE slot (correct until multi-node DDP, which
    # would set split_by_node here) and let the default workersplitter do the
    # per-worker shard split exactly once.
    dataset = wds.WebDataset(urls, shardshuffle=False, empty_check=False,
                             nodesplitter=wds.single_node_only)
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
    if shuffle > 0:
        dataset = dataset.shuffle(shuffle, seed=shuffle_seed)
    # Frame-pool accessor for deduped WM windows (#121 §3.4d): a sibling pool/ dir
    # next to the .tar shards, NOT part of `urls`, so split_by_worker never shards
    # it away — every worker reaches any frame_id by path. Path-based + lazily read,
    # so it pickles cleanly to spawn workers (no open handle crosses the boundary).
    pool = _make_pool_accessor(shard_dir)
    # functools.partial (NOT a lambda) so the map fn pickles to spawn workers.
    dataset = dataset.map(functools.partial(
        _decode_sample,
        pool=pool,
        decode_future_frames=decode_future_frames,
    ))

    # split_by_worker shards the .tar list across workers, so more workers than
    # shards is wasted; cap accordingly. Partition-scoped loaders are retired as
    # soon as that partition is exhausted, so workers MUST NOT persist beyond the
    # iterator lifetime. prefetch_factor overlaps decode with the GPU step.
    eff_workers = min(num_workers, len(tarfiles)) if num_workers > 0 else 0
    loader_kwargs: dict = {"batch_size": batch_size, "num_workers": eff_workers}
    if eff_workers > 0:
        loader_kwargs.update(
            persistent_workers=False,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )
    loader = wds.WebLoader(dataset, **loader_kwargs)

    # Per-dataset geometry, reconstructed once from the manifest.
    projection, geometry_type = load_projection_from_manifest(shard_dir)
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

    Different datasets have different camera counts (L2D 6, NVIDIA 7) and
    geometries (pseudo vs f-theta), which cannot be stacked into one batch. So
    every batch remains same-dataset (uniform num_views/geometry) and carries that
    dataset's projection.

    Only ``max_active_loaders`` child iterators exist at once. Within that window
    batches retain the original round-robin ordering; when a child is exhausted
    it is closed before the next pending child is opened. Loader factories are
    invoked lazily and recreated per epoch, which bounds worker, prefetch, shuffle,
    and pin-memory state even when the input contains hundreds of partitions.

    Each yielded item is ``(batch, projection, geometry_type)`` so the training
    loop applies the right geometry to each (same-dataset) batch.
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
        pending = iter(self._sources)
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
    decode_future_frames: bool = True,
    navigation_repeat_policy: NavigationRepeatPolicy | None = None,
) -> MergedDatasetLoader:
    """Build a :class:`MergedDatasetLoader` over several shard directories.

    Each directory is one dataset (its own manifest + geometry). Datasets are
    merged through a bounded active window (see MergedDatasetLoader). A single
    directory degrades to a one-loader merge (identical to the single dataset
    path, but yielding the ``(batch, projection, geometry_type)`` tuple).

    ``split`` / ``val_fraction`` select a disjoint group-level train/val split
    applied per dataset (see make_pre_extracted_loader). ``num_workers`` is a
    GLOBAL worker budget: each active partition gets one worker, and no more than
    four partition loaders are active. Evaluation can use
    ``max_active_loaders=1`` together with a small
    ``prefetch_factor`` to bound its larger batches.
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
            decode_future_frames=decode_future_frames,
            navigation_repeat_policy=navigation_repeat_policy,
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
