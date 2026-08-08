"""Immutable semantic occupancy artifact and offline inference."""

from __future__ import annotations

import gzip
import hashlib
import io
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY

SEMANTIC_OCCUPANCY_SCHEMA = "v1"
SEMANTIC_OCCUPANCY_FORMAT_VERSION = 1
SEMANTIC_OCCUPANCY_MAGIC = b"ASOC"
SEMANTIC_OCCUPANCY_TAXONOMY_VERSION = "autoe2e-bev-semantic-v1"
SEMANTIC_OCCUPANCY_GEOMETRY_ID = (
    AUTOE2E_NAVIGATION_GEOMETRY.geometry_id
)
SEMANTIC_OCCUPANCY_HEAD_VERSION = "bev-segmentation-head-v1"
SEMANTIC_OCCUPANCY_CLASS_NAMES = (
    "drivable_area",
    "lane_area",
    "intersection",
    "crosswalk",
    "stop_line",
    "vehicle",
    "vulnerable_road_user",
    "other_obstacle",
)
FLAG_TEACHER_PRESENT = 1 << 0
_HEADER = struct.Struct("<4sHHIHHHH")
_DIRECTORY_ENTRY = struct.Struct("<QI")


@dataclass(frozen=True)
class DecodedSemanticOccupancy:
    flags: int
    height: int
    width: int
    directory: tuple[tuple[int, int], ...]
    probability: np.ndarray
    teacher: np.ndarray | None
    valid_mask: np.ndarray | None


def sample_uid_hash(sample_uid: str) -> int:
    digest = hashlib.sha256(sample_uid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def semantic_occupancy_s3_key(
    model_checkpoint_sha256: str,
    dataset_manifest_sha256: str,
    dataset: str,
    shard: str,
) -> str:
    for label, digest in (
        ("model_checkpoint_sha256", model_checkpoint_sha256),
        ("dataset_manifest_sha256", dataset_manifest_sha256),
    ):
        if len(digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in digest
        ):
            raise ValueError(f"{label} must be lowercase SHA-256")
    for label, value in (("dataset", dataset), ("shard", shard)):
        if not value or "/" in value or "\\" in value:
            raise ValueError(f"{label} must be one path segment")
    return (
        f"semantic-occupancy/schema={SEMANTIC_OCCUPANCY_SCHEMA}/"
        f"model={model_checkpoint_sha256}/"
        f"manifest={dataset_manifest_sha256}/"
        f"geometry={SEMANTIC_OCCUPANCY_GEOMETRY_ID}/"
        f"taxonomy={SEMANTIC_OCCUPANCY_TAXONOMY_VERSION}/"
        f"head={SEMANTIC_OCCUPANCY_HEAD_VERSION}/dataset={dataset}/"
        f"shard={shard}/occupancy.bin.gz"
    )


def _quantize_probability(
    probability: np.ndarray,
) -> np.ndarray:
    values = np.asarray(probability)
    if values.ndim != 4 or values.shape[1] != len(
        SEMANTIC_OCCUPANCY_CLASS_NAMES
    ):
        raise ValueError("probability must have shape [N,8,H,W]")
    if (
        not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("probability must be finite floating point in [0,1]")
    return np.ascontiguousarray(
        np.rint(values * 255.0),
        dtype=np.uint8,
    )


def encode_semantic_occupancy(
    sample_uids: Sequence[str],
    probability: np.ndarray,
    *,
    teacher: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> bytes:
    sample_uids = tuple(sample_uids)
    if not sample_uids or len(set(sample_uids)) != len(sample_uids):
        raise ValueError("sample UIDs must be non-empty and unique")
    probability_u8 = _quantize_probability(probability)
    sample_count, class_count, height, width = probability_u8.shape
    if sample_count != len(sample_uids):
        raise ValueError("sample UID count differs from probability rows")
    if max(sample_count, class_count, height, width) > 0xFFFF:
        raise ValueError("semantic occupancy dimensions exceed uint16")

    flags = 0
    teacher_u8 = None
    valid_bits = None
    if (teacher is None) != (valid_mask is None):
        raise ValueError("teacher and valid_mask must be present together")
    if teacher is not None and valid_mask is not None:
        teacher_u8 = _quantize_probability(teacher)
        valid = np.asarray(valid_mask, dtype=np.bool_)
        if teacher_u8.shape != probability_u8.shape or valid.shape != (
            probability_u8.shape
        ):
            raise ValueError("teacher tensors must match probability shape")
        valid_bits = np.packbits(
            valid.reshape(-1),
            bitorder="little",
        )
        flags |= FLAG_TEACHER_PRESENT

    hashes = [sample_uid_hash(uid) for uid in sample_uids]
    if len(set(hashes)) != len(hashes):
        raise ValueError("sample UID hash collision")
    directory = sorted(
        (uid_hash, row)
        for row, uid_hash in enumerate(hashes)
    )
    raw = io.BytesIO()
    raw.write(_HEADER.pack(
        SEMANTIC_OCCUPANCY_MAGIC,
        SEMANTIC_OCCUPANCY_FORMAT_VERSION,
        flags,
        sample_count,
        class_count,
        height,
        width,
        0,
    ))
    for uid_hash, row in directory:
        raw.write(_DIRECTORY_ENTRY.pack(uid_hash, row))
    raw.write(probability_u8.tobytes(order="C"))
    if teacher_u8 is not None and valid_bits is not None:
        raw.write(teacher_u8.tobytes(order="C"))
        raw.write(valid_bits.tobytes(order="C"))

    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=compressed,
        compresslevel=6,
        mtime=0,
    ) as stream:
        stream.write(raw.getvalue())
    return compressed.getvalue()


def decode_semantic_occupancy(
    payload: bytes,
) -> DecodedSemanticOccupancy:
    try:
        raw = gzip.decompress(payload)
    except (EOFError, OSError) as exc:
        raise ValueError("semantic occupancy artifact is not valid gzip") from exc
    if len(raw) < _HEADER.size:
        raise ValueError("semantic occupancy artifact is truncated")
    (
        magic,
        version,
        flags,
        sample_count,
        class_count,
        height,
        width,
        reserved,
    ) = _HEADER.unpack_from(raw)
    if (
        magic != SEMANTIC_OCCUPANCY_MAGIC
        or version != SEMANTIC_OCCUPANCY_FORMAT_VERSION
        or class_count != len(SEMANTIC_OCCUPANCY_CLASS_NAMES)
        or reserved != 0
        or sample_count == 0
        or height == 0
        or width == 0
        or flags & ~FLAG_TEACHER_PRESENT
    ):
        raise ValueError("unsupported semantic occupancy header")
    cell_count = sample_count * class_count * height * width
    valid_byte_count = (cell_count + 7) // 8
    expected_size = (
        _HEADER.size
        + sample_count * _DIRECTORY_ENTRY.size
        + cell_count
    )
    if flags & FLAG_TEACHER_PRESENT:
        expected_size += cell_count + valid_byte_count
    if len(raw) != expected_size:
        raise ValueError("semantic occupancy artifact size mismatch")

    cursor = _HEADER.size
    directory = []
    for _ in range(sample_count):
        directory.append(_DIRECTORY_ENTRY.unpack_from(raw, cursor))
        cursor += _DIRECTORY_ENTRY.size
    if directory != sorted(directory) or sorted(
        row for _, row in directory
    ) != list(range(sample_count)):
        raise ValueError("semantic occupancy directory is invalid")

    shape = (sample_count, class_count, height, width)
    probability = (
        np.frombuffer(raw, dtype=np.uint8, count=cell_count, offset=cursor)
        .reshape(shape)
        .astype(np.float32)
        / 255.0
    )
    cursor += cell_count
    teacher = None
    valid_mask = None
    if flags & FLAG_TEACHER_PRESENT:
        teacher = (
            np.frombuffer(
                raw,
                dtype=np.uint8,
                count=cell_count,
                offset=cursor,
            )
            .reshape(shape)
            .astype(np.float32)
            / 255.0
        )
        cursor += cell_count
        valid_mask = np.unpackbits(
            np.frombuffer(
                raw,
                dtype=np.uint8,
                count=valid_byte_count,
                offset=cursor,
            ),
            count=cell_count,
            bitorder="little",
        ).astype(np.bool_).reshape(shape)
    return DecodedSemanticOccupancy(
        flags=flags,
        height=height,
        width=width,
        directory=tuple(directory),
        probability=probability,
        teacher=teacher,
        valid_mask=valid_mask,
    )


def infer_semantic_occupancy(
    model: torch.nn.Module,
    loader: Any,
    *,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Precompute dense probabilities and optional teacher tensors."""
    was_training = model.training
    sample_uids: list[str] = []
    probabilities = []
    teachers = []
    valid_masks = []
    teacher_mode: bool | None = None
    from training.reactive_stage_runner import (
        resolve_reactive_batch_projection,
    )

    model.eval()
    try:
        with torch.no_grad():
            for item in loader:
                if isinstance(item, tuple):
                    batch, projection, geometry_type = item
                else:
                    batch, projection, geometry_type = item, None, "pseudo"
                projection, geometry_type = (
                    resolve_reactive_batch_projection(
                        batch,
                        projection,
                        geometry_type,
                        device=device,
                    )
                )
                output = model(
                    batch["visual_tiles"].to(device),
                    batch["map_context"].to(device),
                    batch["visual_history"].to(device),
                    batch["egomotion_history"].to(device),
                    route_mask=batch["route_mask"].to(device),
                    map_valid=batch["map_valid"].to(device),
                    route_valid=batch["route_valid"].to(device),
                    projection=projection,
                    geometry_type=geometry_type,
                    mode="infer",
                    return_auxiliary=True,
                    compute_bev_segmentation=True,
                    compute_route_reconstruction=False,
                )
                if not isinstance(output, tuple):
                    raise TypeError(
                        "model did not emit semantic occupancy logits"
                    )
                _, auxiliary = output
                logits = auxiliary.get("bev_segmentation_logits")
                if not torch.is_tensor(logits):
                    raise RuntimeError(
                        "checkpoint has no BEV segmentation head"
                    )
                probabilities.append(
                    logits.sigmoid().float().cpu().numpy()
                )
                batch_uids = [str(uid) for uid in batch["sample_uid"]]
                sample_uids.extend(batch_uids)
                available = batch.get("bev_segmentation_available")
                has_teacher = (
                    available is not None
                    and bool(torch.as_tensor(available).all())
                )
                if teacher_mode is None:
                    teacher_mode = has_teacher
                elif teacher_mode != has_teacher:
                    raise ValueError(
                        "semantic artifact cannot mix teacher availability"
                    )
                if has_teacher:
                    teachers.append(
                        batch["bev_segmentation_target"].numpy()
                    )
                    valid_masks.append(
                        batch["bev_segmentation_valid"].numpy()
                    )
    finally:
        model.train(was_training)
    if not sample_uids:
        raise ValueError("semantic occupancy loader yielded no samples")
    return (
        sample_uids,
        np.concatenate(probabilities, axis=0),
        np.concatenate(teachers, axis=0) if teachers else None,
        np.concatenate(valid_masks, axis=0) if valid_masks else None,
    )


def write_semantic_occupancy(
    path: str | Path,
    sample_uids: Sequence[str],
    probability: np.ndarray,
    *,
    teacher: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> str:
    payload = encode_semantic_occupancy(
        sample_uids,
        probability,
        teacher=teacher,
        valid_mask=valid_mask,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
