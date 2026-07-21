"""Binary trajectory-overlay artifact contract.

This module has no Flyte or AWS dependency. GPU inference writes one canonical
artifact per (model checkpoint, immutable dataset version, WebDataset shard);
the Go API and browser consume the same little-endian layout.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


OVERLAY_SCHEMA = "v1"
OVERLAY_FORMAT_VERSION = 1
OVERLAY_MAGIC = b"AOVL"
UID_HASH_ALGORITHM = "sha256-le64-v1"
FLAG_DETERMINISTIC_PLANNER = 1 << 0

_HEADER = struct.Struct("<4sHHIHHHH")
_SEED = struct.Struct("<q")
_DIRECTORY_ENTRY = struct.Struct("<QI")
_HORIZON = 64
_DIMS = 2


@dataclass(frozen=True)
class OverlayArtifact:
    path: Path
    sha256: str
    byte_size: int
    sample_count: int
    seed_count: int


@dataclass(frozen=True)
class DecodedOverlay:
    flags: int
    base_seeds: tuple[int, ...]
    directory: tuple[tuple[int, int], ...]
    controls: np.ndarray
    v0: np.ndarray


def sample_uid_hash(sample_uid: str) -> int:
    """Return the browser/Go-compatible 64-bit sample directory key."""
    digest = hashlib.sha256(sample_uid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def overlay_s3_key(
    model_artifact_id: str,
    dataset: str,
    version: str,
    shard: str,
) -> str:
    """Build the canonical split-free overlay key."""
    for label, value in (
        ("model_artifact_id", model_artifact_id),
        ("dataset", dataset),
        ("version", version),
        ("shard", shard),
    ):
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"{label} must be one non-empty path segment")
    if len(model_artifact_id) != 64 or any(
        char not in "0123456789abcdef" for char in model_artifact_id
    ):
        raise ValueError("model_artifact_id must be a lowercase SHA-256 hex digest")
    return (
        f"overlays/schema={OVERLAY_SCHEMA}/model={model_artifact_id}/"
        f"dataset={dataset}/version={version}/shard={shard}/overlay.bin.gz"
    )


def _normalise_controls(
    controls: np.ndarray,
    sample_count: int,
    seed_count: int,
) -> np.ndarray:
    array = np.asarray(controls)
    if array.ndim == 3:
        array = array[:, None, :, :]
    expected = (sample_count, seed_count, _HORIZON, _DIMS)
    if array.shape != expected:
        raise ValueError(f"controls must have shape {expected}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("controls must be floating point")
    if not np.isfinite(array).all():
        raise ValueError("controls contain NaN or infinity")
    return np.ascontiguousarray(array, dtype="<f4")


def _normalise_v0(v0: np.ndarray, sample_count: int) -> np.ndarray:
    array = np.asarray(v0)
    if array.shape != (sample_count,):
        raise ValueError(f"v0 must have shape ({sample_count},), got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("v0 must be floating point")
    if not np.isfinite(array).all():
        raise ValueError("v0 contains NaN or infinity")
    return np.ascontiguousarray(array, dtype="<f4")


def encode_overlay(
    sample_uids: Sequence[str],
    controls: np.ndarray,
    v0: np.ndarray,
    *,
    base_seeds: Sequence[int] = (0,),
    deterministic_planner: bool = False,
) -> bytes:
    """Encode and deterministically gzip one canonical shard overlay."""
    sample_uids = tuple(sample_uids)
    base_seeds = tuple(int(seed) for seed in base_seeds)
    if not sample_uids:
        raise ValueError("sample_uids must not be empty")
    if len(set(sample_uids)) != len(sample_uids):
        raise ValueError("sample_uids must be unique")
    if not base_seeds:
        raise ValueError("base_seeds must not be empty")
    if any(seed < 0 or seed > (2**63 - 1) for seed in base_seeds):
        raise ValueError("base seeds must fit non-negative signed int64")

    sample_count = len(sample_uids)
    seed_count = len(base_seeds)
    if sample_count > 0xFFFFFFFF:
        raise ValueError("sample_count exceeds uint32")
    if seed_count > 0xFFFF:
        raise ValueError("seed_count exceeds uint16")

    controls_f32 = _normalise_controls(
        controls, sample_count=sample_count, seed_count=seed_count
    )
    v0_f32 = _normalise_v0(v0, sample_count=sample_count)

    hashes = [sample_uid_hash(uid) for uid in sample_uids]
    if len(set(hashes)) != len(hashes):
        raise ValueError("sample_uid hash collision within overlay shard")
    directory = sorted((uid_hash, row) for row, uid_hash in enumerate(hashes))

    flags = FLAG_DETERMINISTIC_PLANNER if deterministic_planner else 0
    raw = io.BytesIO()
    raw.write(_HEADER.pack(
        OVERLAY_MAGIC,
        OVERLAY_FORMAT_VERSION,
        flags,
        sample_count,
        seed_count,
        _HORIZON,
        _DIMS,
        0,
    ))
    for seed in base_seeds:
        raw.write(_SEED.pack(seed))
    for uid_hash, row in directory:
        raw.write(_DIRECTORY_ENTRY.pack(uid_hash, row))
    raw.write(controls_f32.tobytes(order="C"))
    raw.write(v0_f32.tobytes(order="C"))

    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=compressed, compresslevel=6, mtime=0
    ) as stream:
        stream.write(raw.getvalue())
    return compressed.getvalue()


def write_overlay(
    path: str | Path,
    sample_uids: Sequence[str],
    controls: np.ndarray,
    v0: np.ndarray,
    *,
    base_seeds: Sequence[int] = (0,),
    deterministic_planner: bool = False,
) -> OverlayArtifact:
    """Atomically write one overlay and return its pointer metadata."""
    path = Path(path)
    payload = encode_overlay(
        sample_uids,
        controls,
        v0,
        base_seeds=base_seeds,
        deterministic_planner=deterministic_planner,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return OverlayArtifact(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        sample_count=len(sample_uids),
        seed_count=len(base_seeds),
    )


def decode_overlay(payload: bytes) -> DecodedOverlay:
    """Decode an overlay for contract tests and offline validation."""
    try:
        raw = gzip.decompress(payload)
    except (EOFError, OSError) as exc:
        raise ValueError("invalid gzip overlay") from exc
    if len(raw) < _HEADER.size:
        raise ValueError("overlay is shorter than its header")

    (
        magic,
        version,
        flags,
        sample_count,
        seed_count,
        horizon,
        dims,
        reserved,
    ) = _HEADER.unpack_from(raw)
    if magic != OVERLAY_MAGIC:
        raise ValueError("invalid overlay magic")
    if version != OVERLAY_FORMAT_VERSION:
        raise ValueError(f"unsupported overlay version {version}")
    if horizon != _HORIZON or dims != _DIMS or reserved != 0:
        raise ValueError("unsupported overlay dimensions or reserved bits")
    if sample_count == 0 or seed_count == 0:
        raise ValueError("overlay must contain samples and seeds")

    cursor = _HEADER.size
    expected_size = (
        _HEADER.size
        + seed_count * _SEED.size
        + sample_count * _DIRECTORY_ENTRY.size
        + sample_count * seed_count * horizon * dims * 4
        + sample_count * 4
    )
    if len(raw) != expected_size:
        raise ValueError(
            f"overlay size mismatch: expected {expected_size}, got {len(raw)}"
        )

    seeds = []
    for _ in range(seed_count):
        (seed,) = _SEED.unpack_from(raw, cursor)
        seeds.append(seed)
        cursor += _SEED.size

    directory = []
    for _ in range(sample_count):
        entry = _DIRECTORY_ENTRY.unpack_from(raw, cursor)
        directory.append(entry)
        cursor += _DIRECTORY_ENTRY.size
    if directory != sorted(directory):
        raise ValueError("overlay directory is not sorted")
    rows = [row for _, row in directory]
    if sorted(rows) != list(range(sample_count)):
        raise ValueError("overlay directory rows are not a permutation")

    control_count = sample_count * seed_count * horizon * dims
    controls = np.frombuffer(
        raw, dtype="<f4", count=control_count, offset=cursor
    ).reshape(sample_count, seed_count, horizon, dims).copy()
    cursor += control_count * 4
    speeds = np.frombuffer(
        raw, dtype="<f4", count=sample_count, offset=cursor
    ).copy()
    return DecodedOverlay(
        flags=flags,
        base_seeds=tuple(seeds),
        directory=tuple(directory),
        controls=controls,
        v0=speeds,
    )
