"""Materialize immutable nuPlan snapshot archives for local data preparation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
from typing import Any
from urllib.parse import urlsplit
import zipfile

from .packing import NUPLAN_CAMERA_CHANNELS


NUPLAN_SNAPSHOT_SCHEMA_VERSION = "nuplan_raw_snapshot_v1"
_REQUIRED_COMPONENTS = frozenset({"maps", "database", "sensor_blobs"})


@dataclass(frozen=True)
class MaterializedNuPlanDataset:
    """Local roots and DB files accepted by the nuPlan scenario builder."""

    data_root: Path
    map_root: Path
    map_version: str
    sensor_root: Path
    db_files: tuple[Path, ...]
    sensor_log_names: tuple[str, ...]


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_relative_path(value: Any, field: str) -> str:
    path = _require_text(value, field)
    parts = PurePosixPath(path).parts
    if (
        path.startswith("/")
        or path.endswith("/")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"{field} must be a normalized relative path")
    return path


def validate_nuplan_snapshot_manifest(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the redacted immutable snapshot consumed by data preparation."""
    if payload.get("schema_version") != NUPLAN_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            "nuPlan snapshot schema_version must be "
            f"{NUPLAN_SNAPSHOT_SCHEMA_VERSION!r}"
        )
    snapshot_id = _require_text(payload.get("snapshot_id"), "snapshot_id")
    dataset_revision = _require_text(
        payload.get("dataset_revision"),
        "dataset_revision",
    )
    map_version = _require_text(payload.get("map_version"), "map_version")
    source_contract_sha256 = _require_text(
        payload.get("source_contract_sha256"),
        "source_contract_sha256",
    )
    if (
        len(source_contract_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_contract_sha256)
    ):
        raise ValueError("source_contract_sha256 must be lowercase SHA-256")

    raw_archives = payload.get("archives")
    if not isinstance(raw_archives, list) or not raw_archives:
        raise ValueError("archives must be a non-empty list")
    archives: list[dict[str, Any]] = []
    archive_ids: set[str] = set()
    for index, raw_archive in enumerate(raw_archives):
        if not isinstance(raw_archive, Mapping):
            raise ValueError(f"archives[{index}] must be an object")
        archive_id = _require_text(
            raw_archive.get("archive_id"),
            f"archives[{index}].archive_id",
        )
        if archive_id in archive_ids:
            raise ValueError(f"duplicate archive_id {archive_id!r}")
        archive_ids.add(archive_id)
        component = _require_text(
            raw_archive.get("component"),
            f"archives[{index}].component",
        )
        if component not in _REQUIRED_COMPONENTS:
            raise ValueError(
                f"archives[{index}].component must be one of "
                f"{sorted(_REQUIRED_COMPONENTS)}"
            )
        object_uri = _require_text(
            raw_archive.get("object_uri"),
            f"archives[{index}].object_uri",
        )
        parsed = urlsplit(object_uri)
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or not parsed.path.lstrip("/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                f"archives[{index}].object_uri must be an S3 object URI"
            )
        size_bytes = raw_archive.get("size_bytes")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise ValueError(
                f"archives[{index}].size_bytes must be positive"
            )
        filename = _require_text(
            raw_archive.get("filename"),
            f"archives[{index}].filename",
        )
        if PurePosixPath(filename).name != filename:
            raise ValueError(
                f"archives[{index}].filename must be a basename"
            )
        archives.append({
            "archive_id": archive_id,
            "component": component,
            "extract_to": _validate_relative_path(
                raw_archive.get("extract_to"),
                f"archives[{index}].extract_to",
            ),
            "filename": filename,
            "md5": str(raw_archive.get("md5", "")),
            "object_uri": object_uri,
            "sha256": str(raw_archive.get("sha256", "")),
            "size_bytes": size_bytes,
        })

    components = {archive["component"] for archive in archives}
    missing = _REQUIRED_COMPONENTS - components
    if missing:
        raise ValueError(
            "nuPlan snapshot is missing required components: "
            f"{sorted(missing)}"
        )
    declared_total = payload.get("total_size_bytes")
    actual_total = sum(archive["size_bytes"] for archive in archives)
    if declared_total != actual_total:
        raise ValueError(
            "nuPlan snapshot total_size_bytes does not match its archives"
        )
    return {
        "archives": archives,
        "dataset_revision": dataset_revision,
        "map_version": map_version,
        "schema_version": NUPLAN_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "source_contract_sha256": source_contract_sha256,
        "total_size_bytes": actual_total,
    }


def load_nuplan_snapshot_manifest(payload: bytes) -> dict[str, Any]:
    """Decode and validate one immutable snapshot manifest."""
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("nuPlan snapshot is not valid UTF-8 JSON") from error
    if not isinstance(parsed, Mapping):
        raise ValueError("nuPlan snapshot root must be an object")
    return validate_nuplan_snapshot_manifest(parsed)


def select_snapshot_archives(
    manifest: Mapping[str, Any],
    archive_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Select a complete maps/database/sensor set by stable archive ID."""
    archives = [dict(archive) for archive in manifest["archives"]]
    if archive_ids:
        requested = list(archive_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("archive_ids must not contain duplicates")
        by_id = {archive["archive_id"]: archive for archive in archives}
        missing_ids = sorted(set(requested) - set(by_id))
        if missing_ids:
            raise ValueError(f"unknown nuPlan archive_ids: {missing_ids}")
        archives = [by_id[archive_id] for archive_id in requested]
    non_zip_ids = [
        archive["archive_id"]
        for archive in archives
        if not archive["filename"].endswith(".zip")
    ]
    if non_zip_ids:
        raise ValueError(
            "selected nuPlan archives must be ZIP files: "
            f"{non_zip_ids}"
        )
    components = {archive["component"] for archive in archives}
    missing_components = _REQUIRED_COMPONENTS - components
    if missing_components:
        raise ValueError(
            "selected nuPlan archives are missing components: "
            f"{sorted(missing_components)}"
        )
    sensor_labels = [
        f"{archive['archive_id']} {archive['filename']}".lower()
        for archive in archives
        if archive["component"] == "sensor_blobs"
    ]
    missing_modalities = [
        modality
        for modality in ("camera", "lidar")
        if not any(modality in label for label in sensor_labels)
    ]
    if missing_modalities:
        raise ValueError(
            "selected nuPlan sensor archives are missing modalities: "
            f"{missing_modalities}"
        )
    return archives


def verify_archive_file(
    archive_path: Path,
    archive: Mapping[str, Any],
) -> None:
    """Verify a downloaded snapshot object before extraction."""
    if archive_path.stat().st_size != archive["size_bytes"]:
        raise ValueError(
            f"nuPlan archive size mismatch for {archive['archive_id']!r}"
        )
    expected_sha256 = archive.get("sha256", "")
    expected_md5 = archive.get("md5", "")
    if not expected_sha256 and not expected_md5:
        return
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with archive_path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            sha256.update(chunk)
            md5.update(chunk)
    if expected_sha256 and sha256.hexdigest() != expected_sha256:
        raise ValueError(
            f"nuPlan archive SHA-256 mismatch for {archive['archive_id']!r}"
        )
    if expected_md5 and md5.hexdigest() != expected_md5:
        raise ValueError(
            f"nuPlan archive MD5 mismatch for {archive['archive_id']!r}"
        )


def _safe_member_parts(info: zipfile.ZipInfo) -> tuple[str, ...]:
    path = PurePosixPath(info.filename)
    parts = path.parts
    if (
        path.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"unsafe ZIP member path: {info.filename!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"ZIP symlink is not allowed: {info.filename!r}")
    return parts


def _member_destination_from_parts(
    parts: tuple[str, ...],
    archive: Mapping[str, Any],
    *,
    is_directory: bool,
    map_version: str,
) -> PurePosixPath | None:
    if parts == ("LICENSE",):
        return None
    component = archive["component"]
    extract_to = PurePosixPath(archive["extract_to"])
    if component == "maps":
        if parts[0] != map_version:
            raise ValueError(
                f"map archive member is outside {map_version!r}: "
                f"{PurePosixPath(*parts)!s}"
            )
        relative = PurePosixPath(*parts[1:])
        return extract_to / relative if relative.parts else None
    if component == "database":
        if is_directory or PurePosixPath(*parts).suffix != ".db":
            return None
        return extract_to / parts[-1]
    if len(parts) < 2:
        return None
    relative = PurePosixPath(*parts[1:])
    return extract_to / relative if relative.parts else None


def _safe_tar_member_parts(info: tarfile.TarInfo) -> tuple[str, ...]:
    path = PurePosixPath(info.name)
    parts = path.parts
    if (
        path.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"unsafe TAR member path: {info.name!r}")
    if info.issym() or info.islnk():
        raise ValueError(f"TAR link is not allowed: {info.name!r}")
    if not (info.isdir() or info.isreg()):
        raise ValueError(f"unsupported TAR member type: {info.name!r}")
    return parts


def _archive_map_version(
    member_paths: Sequence[tuple[str, ...]],
) -> str:
    map_versions = {
        parts[0]
        for parts in member_paths
        if len(parts) == 2
        and parts[0].startswith("nuplan-maps-v")
        and parts[1] == f"{parts[0]}.json"
    }
    if len(map_versions) != 1:
        raise ValueError(
            "nuPlan map archive must contain exactly one "
            "version metadata file"
        )
    return map_versions.pop()


def _write_archive_member(
    source: Any,
    target: Path,
    dataset_root: Path,
) -> None:
    if dataset_root.resolve() not in target.parents:
        raise ValueError(f"archive member escapes dataset root: {target}")
    if target.exists():
        if not target.is_file():
            raise FileExistsError(
                f"duplicate nuPlan materialized path is not a file: {target}"
            )
        with source, target.open("rb") as existing:
            while True:
                incoming_chunk = source.read(16 * 1024 * 1024)
                existing_chunk = existing.read(16 * 1024 * 1024)
                if incoming_chunk != existing_chunk:
                    raise FileExistsError(
                        "conflicting duplicate nuPlan materialized file: "
                        f"{target}"
                    )
                if not incoming_chunk:
                    return
    target.parent.mkdir(parents=True, exist_ok=True)
    with source, target.open("wb") as output:
        shutil.copyfileobj(
            source,
            output,
            length=16 * 1024 * 1024,
        )


def extract_nuplan_archive(
    archive_path: Path,
    archive: Mapping[str, Any],
    dataset_root: Path,
    *,
    map_version: str,
) -> dict[str, int | str]:
    """Safely normalize one official nuPlan ZIP or TAR archive."""
    file_count = 0
    uncompressed_bytes = 0
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as source:
            members = source.infolist()
            member_parts = [
                _safe_member_parts(info) for info in members
            ]
            archive_map_version = map_version
            if archive["component"] == "maps":
                archive_map_version = _archive_map_version(member_parts)
            for info, parts in zip(members, member_parts):
                relative = _member_destination_from_parts(
                    parts,
                    archive,
                    is_directory=info.is_dir(),
                    map_version=archive_map_version,
                )
                if relative is None:
                    continue
                target = (
                    dataset_root / Path(*relative.parts)
                ).resolve()
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                _write_archive_member(
                    source.open(info),
                    target,
                    dataset_root,
                )
                file_count += 1
                uncompressed_bytes += info.file_size
        return {
            "archive_format": "zip",
            "file_count": file_count,
            "uncompressed_bytes": uncompressed_bytes,
        }

    if not tarfile.is_tarfile(archive_path):
        raise ValueError(
            f"unsupported nuPlan archive format: {archive['archive_id']!r}"
        )
    with tarfile.open(archive_path, mode="r:*") as source:
        members = source.getmembers()
        member_parts = [
            _safe_tar_member_parts(info) for info in members
        ]
        archive_map_version = map_version
        if archive["component"] == "maps":
            archive_map_version = _archive_map_version(member_parts)
        for info, parts in zip(members, member_parts):
            relative = _member_destination_from_parts(
                parts,
                archive,
                is_directory=info.isdir(),
                map_version=archive_map_version,
            )
            if relative is None:
                continue
            target = (dataset_root / Path(*relative.parts)).resolve()
            if info.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            input_stream = source.extractfile(info)
            if input_stream is None:
                raise ValueError(
                    f"failed to read TAR member: {info.name!r}"
                )
            _write_archive_member(input_stream, target, dataset_root)
            file_count += 1
            uncompressed_bytes += info.size
    return {
        "archive_format": "tar",
        "file_count": file_count,
        "uncompressed_bytes": uncompressed_bytes,
    }


def complete_sensor_log_names(sensor_root: Path) -> tuple[str, ...]:
    """Return logs that have every camera channel and a LiDAR point cloud."""
    complete: list[str] = []
    if not sensor_root.is_dir():
        return ()
    for log_directory in sorted(sensor_root.iterdir()):
        if not log_directory.is_dir():
            continue
        cameras_ready = all(
            any((log_directory / channel).glob("*.jpg"))
            for channel in NUPLAN_CAMERA_CHANNELS
        )
        lidar_ready = any(
            (log_directory / "MergedPointCloud").glob("*.pcd")
        )
        if cameras_ready and lidar_ready:
            complete.append(log_directory.name)
    return tuple(complete)


def discover_materialized_nuplan(
    dataset_root: Path,
) -> MaterializedNuPlanDataset:
    """Discover DBs whose complete camera and LiDAR blobs are available."""
    data_root = dataset_root.resolve()
    map_root = data_root / "maps"
    map_metadata = tuple(sorted(map_root.glob("nuplan-maps-v*.json")))
    if len(map_metadata) != 1:
        raise ValueError(
            "nuPlan map root must contain exactly one version metadata file"
        )
    map_version = map_metadata[0].stem
    sensor_root = data_root / "nuplan-v1.1" / "sensor_blobs"
    sensor_logs = complete_sensor_log_names(sensor_root)
    available = set(sensor_logs)
    db_root = data_root / "nuplan-v1.1" / "splits"
    db_files = tuple(
        path
        for path in sorted(db_root.rglob("*.db"))
        if path.stem in available
    )
    if not map_root.is_dir():
        raise FileNotFoundError("nuPlan map root was not materialized")
    if not sensor_logs:
        raise ValueError(
            "nuPlan snapshot has no log with complete camera and LiDAR blobs"
        )
    if not db_files:
        raise ValueError(
            "nuPlan snapshot has no DB matching complete sensor blobs"
        )
    return MaterializedNuPlanDataset(
        data_root=data_root,
        map_root=map_root,
        map_version=map_version,
        sensor_root=sensor_root,
        db_files=db_files,
        sensor_log_names=sensor_logs,
    )


def materialize_nuplan_snapshot(
    manifest: Mapping[str, Any],
    archive_paths: Mapping[str, Path],
    dataset_root: Path,
    *,
    archive_ids: Sequence[str] | None = None,
    on_archive_extracted: Callable[[Mapping[str, Any]], None] | None = None,
) -> MaterializedNuPlanDataset:
    """Materialize selected archives and discover a coherent local dataset."""
    selected = select_snapshot_archives(manifest, archive_ids)
    dataset_root.mkdir(parents=True, exist_ok=True)
    for archive in selected:
        archive_id = archive["archive_id"]
        if archive_id not in archive_paths:
            raise ValueError(f"archive path is missing for {archive_id!r}")
        archive_path = archive_paths[archive_id]
        verify_archive_file(archive_path, archive)
        extract_nuplan_archive(
            archive_path,
            archive,
            dataset_root,
            map_version=manifest["map_version"],
        )
        if on_archive_extracted is not None:
            on_archive_extracted(archive)
    return discover_materialized_nuplan(dataset_root)
