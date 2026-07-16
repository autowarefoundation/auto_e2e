"""Readers for canonical AOVL bodies and their matching WebDataset shard."""

from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from Platform.pipelines.overlay import (
    DecodedOverlay,
    decode_overlay,
    sample_uid_hash,
)


_EGO_FLOATS = 64 * 4 + 64 * 2
_TARGET_OFFSET = 64 * 4
_MAX_JSON_BYTES = 1024 * 1024
_MAX_JPEG_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class ShardSample:
    sample_uid: str
    scene_uid: str
    frame_idx: int
    dataset: str
    camera_jpeg: bytes
    initial_speed: float
    target_controls: np.ndarray
    calibration: Mapping[str, Any]


@dataclass(frozen=True)
class OverlayReader:
    decoded: DecodedOverlay
    sha256: str
    rows_by_hash: Mapping[int, int]

    @property
    def base_seeds(self) -> tuple[int, ...]:
        return self.decoded.base_seeds

    def sample(
        self,
        sample_uid: str,
        seed_index: int,
    ) -> tuple[np.ndarray, float]:
        if seed_index < 0 or seed_index >= len(self.decoded.base_seeds):
            raise IndexError(
                f"seed_index {seed_index} is outside "
                f"[0, {len(self.decoded.base_seeds)})"
            )
        row = self.rows_by_hash.get(sample_uid_hash(sample_uid))
        if row is None:
            raise KeyError(f"overlay has no row for sample_uid {sample_uid!r}")
        return self.decoded.controls[row, seed_index], float(self.decoded.v0[row])

    def validate_sample_uids(self, sample_uids: list[str]) -> None:
        expected = {sample_uid_hash(sample_uid) for sample_uid in sample_uids}
        if len(expected) != len(sample_uids):
            raise ValueError("shard sample_uid values are not hash-unique")
        actual = set(self.rows_by_hash)
        if expected != actual:
            raise ValueError(
                "overlay directory does not exactly match the shard samples: "
                f"{len(expected - actual)} missing, "
                f"{len(actual - expected)} unexpected"
            )


def load_overlay(path: str | Path) -> OverlayReader:
    payload = Path(path).read_bytes()
    decoded = decode_overlay(payload)
    return OverlayReader(
        decoded=decoded,
        sha256=hashlib.sha256(payload).hexdigest(),
        rows_by_hash={uid_hash: row for uid_hash, row in decoded.directory},
    )


def _read_member(stream: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    extracted = stream.extractfile(member)
    if extracted is None:
        raise ValueError(f"tar member is not readable: {member.name}")
    return extracted.read()


def _sample_key(member_name: str, suffix: str) -> str | None:
    if not member_name.endswith(suffix):
        return None
    key = member_name[: -len(suffix)]
    if not key or "/" in key or "\\" in key:
        raise ValueError(f"unsafe sample member name {member_name!r}")
    return key


def read_shard_samples(
    path: str | Path,
    *,
    camera_index: int = 0,
) -> list[ShardSample]:
    """Read the report inputs without extracting untrusted tar paths."""
    if camera_index < 0:
        raise ValueError("camera_index must be non-negative")
    camera_suffix = f".cam_{camera_index}.jpg"
    records: dict[str, dict[str, bytes]] = {}

    with tarfile.open(path, mode="r:*") as stream:
        for member in stream:
            if not member.isfile():
                continue
            matched: tuple[str, str] | None = None
            for field, suffix, limit in (
                ("meta", ".meta.json", _MAX_JSON_BYTES),
                ("ego", ".ego.npy", _EGO_FLOATS * 4),
                ("camera", camera_suffix, _MAX_JPEG_BYTES),
                ("calibration", ".calib.json", _MAX_JSON_BYTES),
            ):
                key = _sample_key(member.name, suffix)
                if key is None:
                    continue
                if member.size > limit:
                    raise ValueError(
                        f"{member.name} exceeds the {limit}-byte report limit"
                    )
                matched = key, field
                break
            if matched is None:
                continue
            key, field = matched
            record = records.setdefault(key, {})
            if field in record:
                raise ValueError(f"duplicate {field} member for {key}")
            record[field] = _read_member(stream, member)

    samples: list[ShardSample] = []
    seen_positions: set[tuple[str, int]] = set()
    required = {"meta", "ego", "camera", "calibration"}
    for key, record in records.items():
        missing = required.difference(record)
        if missing:
            raise ValueError(
                f"sample {key!r} is missing report members: "
                + ", ".join(sorted(missing))
            )
        try:
            metadata = json.loads(record["meta"])
            calibration = json.loads(record["calibration"])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"sample {key!r} has invalid JSON metadata") from exc
        if not isinstance(metadata, dict) or not isinstance(calibration, dict):
            raise ValueError(f"sample {key!r} metadata must be JSON objects")

        sample_uid = str(metadata.get("sample_uid", ""))
        scene_uid = str(metadata.get("split_group_uid", ""))
        dataset = str(metadata.get("dataset", ""))
        if sample_uid != key:
            raise ValueError(
                f"tar key {key!r} disagrees with sample_uid {sample_uid!r}"
            )
        if not scene_uid or not dataset:
            raise ValueError(f"sample {key!r} has no scene or dataset identity")
        frame_idx = metadata.get("frame_idx")
        if (
            isinstance(frame_idx, bool)
            or not isinstance(frame_idx, int)
            or frame_idx < 0
        ):
            raise ValueError(f"sample {key!r} has invalid frame_idx")
        position = (scene_uid, frame_idx)
        if position in seen_positions:
            raise ValueError(
                f"duplicate frame {frame_idx} in scene {scene_uid!r}"
            )
        seen_positions.add(position)

        ego = np.frombuffer(record["ego"], dtype="<f4")
        if ego.size != _EGO_FLOATS:
            raise ValueError(
                f"sample {key!r} ego payload has {ego.size} floats, "
                f"expected {_EGO_FLOATS}"
            )
        if not np.isfinite(ego).all():
            raise ValueError(f"sample {key!r} ego payload is not finite")
        samples.append(ShardSample(
            sample_uid=sample_uid,
            scene_uid=scene_uid,
            frame_idx=frame_idx,
            dataset=dataset,
            camera_jpeg=record["camera"],
            initial_speed=float(ego[_TARGET_OFFSET - 4]),
            target_controls=ego[_TARGET_OFFSET:].reshape(64, 2).copy(),
            calibration=calibration,
        ))

    if not samples:
        raise ValueError(f"shard {path} contains no reportable samples")
    samples.sort(key=lambda sample: (
        sample.scene_uid,
        sample.frame_idx,
        sample.sample_uid,
    ))
    return samples
