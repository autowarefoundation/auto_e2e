"""Pure contracts for resolving a wf_autoe2e dataset by content identity.

A dataset is identified by ``dataset_id``: the SHA-256 of the canonical JSON
of the fields that decide its bytes. Publications keep the Console-facing
``<dataset>/vN.M/`` layout; ``<dataset>/by-id/<dataset_id>.json`` is written
after the manifest and maps the identity to that version.
"""

from __future__ import annotations

import enum
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

from Platform.pipelines.dataset_publication import (
    canonical_json_bytes,
    dataset_prefix,
    sha256_bytes,
)


DATASET_SPEC_SCHEMA = "autoe2e_dataset_spec_v1"
DATASET_INDEX_SCHEMA = "autoe2e_dataset_index_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_S3_URI_RE = re.compile(r"^s3://[a-z0-9][a-z0-9.-]+/.+[^/]$")
_VERSION_PARTS_RE = re.compile(r"^v([0-9]+)(?:\.([0-9]+))?$")


class AutoE2EDataset(enum.Enum):
    """Published dataset ids that wf_autoe2e can build."""

    KITSCENES = "kitscenes"
    NUPLAN = "nuplan"
    L2D = "l2d"
    # AutoBEV-only datasets: images, projections, BEV labels, depth and boxes.
    NUSCENES = "nuscenes"
    ARGOVERSE2 = "argoverse2"
    ONCE = "once"
    SIMBEV = "simbev"


@dataclass(frozen=True)
class ManifestRef:
    """An immutable S3 object pinned by version id and content digest."""

    uri: str
    version_id: str
    sha256: str


@dataclass(frozen=True)
class ReasoningSpec:
    teacher: str
    teacher_model: str
    prompt_version: str
    label_stride: int


@dataclass(frozen=True)
class DatasetSpec:
    """What a dataset contains. Every field enters ``dataset_id``.

    ``source_manifest`` is a raw snapshot manifest; it pins the unit list and
    the upstream revision. ``pack_params`` carries adapter-specific pack
    inputs as strings so the workflow signature is the same for every dataset.
    """

    dataset: AutoE2EDataset
    source_manifest: ManifestRef
    pack_version: str
    image_size: int
    unit_limit: int = 0
    pack_params: Dict[str, str] = field(default_factory=dict)
    reasoning: Optional[ReasoningSpec] = None


def validate_manifest_ref(ref: ManifestRef, name: str) -> None:
    if not _S3_URI_RE.fullmatch(ref.uri):
        raise ValueError(f"{name} uri is not an S3 object: {ref.uri!r}")
    if not ref.version_id:
        raise ValueError(f"{name} version_id is required")
    if not _SHA256_RE.fullmatch(ref.sha256):
        raise ValueError(f"{name} sha256 must be 64 lowercase hex digits")


def validate_dataset_spec(spec: DatasetSpec) -> None:
    if not isinstance(spec.dataset, AutoE2EDataset):
        raise ValueError(f"unsupported dataset {spec.dataset!r}")
    validate_manifest_ref(spec.source_manifest, "source_manifest")
    if spec.unit_limit < 0:
        raise ValueError("unit_limit must be zero or positive")
    for key, value in spec.pack_params.items():
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise ValueError(f"pack_params must map non-empty strings: {key!r}")
    if not spec.pack_version:
        raise ValueError("pack_version is required")
    if spec.image_size <= 0:
        raise ValueError("image_size must be positive")
    if spec.reasoning is not None:
        if not (
            spec.reasoning.teacher
            and spec.reasoning.teacher_model
            and spec.reasoning.prompt_version
        ):
            raise ValueError("reasoning teacher, teacher_model, and prompt_version are required")
        if spec.reasoning.label_stride <= 0:
            raise ValueError("reasoning label_stride must be positive")


def canonical_dataset_spec(spec: DatasetSpec) -> dict[str, Any]:
    """Return the JSON object whose digest is the dataset identity."""
    validate_dataset_spec(spec)
    payload = asdict(spec)
    payload["dataset"] = spec.dataset.value
    payload["pack_params"] = dict(sorted(spec.pack_params.items()))
    payload["schema_version"] = DATASET_SPEC_SCHEMA
    return payload


def compute_dataset_id(spec: DatasetSpec) -> str:
    return sha256_bytes(canonical_json_bytes(canonical_dataset_spec(spec)))


def dataset_index_key(published_name: str, dataset_id: str) -> str:
    if not _SHA256_RE.fullmatch(dataset_id):
        raise ValueError(f"invalid dataset id {dataset_id!r}")
    dataset_prefix(published_name, "v1")
    return f"{published_name}/by-id/{dataset_id}.json"


def build_reservation_key(published_name: str, dataset_id: str) -> str:
    if not _SHA256_RE.fullmatch(dataset_id):
        raise ValueError(f"invalid dataset id {dataset_id!r}")
    dataset_prefix(published_name, "v1")
    return f"{published_name}/_stages/builds/{dataset_id}.json"


def version_reservation_key(published_name: str, version: str) -> str:
    dataset_prefix(published_name, version)
    return f"{published_name}/_stages/versions/{version}.json"


def lineage_key(published_name: str, version: str) -> str:
    return f"{dataset_prefix(published_name, version)}/lineage/dataset.json"


def units_key(published_name: str, version: str) -> str:
    return f"{dataset_prefix(published_name, version)}/lineage/units.jsonl"


def partition_view_prefix(published_name: str, version: str, partition_id: str) -> str:
    """Per-partition directory that Ray Train reads (manifest.json and shards)."""
    import re as _re

    if not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", partition_id):
        raise ValueError(f"invalid partition id {partition_id!r}")
    return f"{dataset_prefix(published_name, version)}/partitions/{partition_id}"


def _version_parts(version: str) -> tuple[int, int]:
    match = _VERSION_PARTS_RE.fullmatch(version)
    if match is None:
        raise ValueError(f"invalid dataset version {version!r}")
    return int(match.group(1)), int(match.group(2) or 0)


def next_dataset_version(existing_versions: Iterable[str]) -> str:
    """Allocate the next minor version after the newest well-formed one."""
    parsed = []
    for version in existing_versions:
        try:
            parsed.append(_version_parts(version))
        except ValueError:
            continue
    if not parsed:
        return "v1.0"
    major, minor = max(parsed)
    return f"v{major}.{minor + 1}"


VIEWS_SCHEMA = "autoe2e_partition_views_v1"


def views_manifest_key(published_name: str, version: str) -> str:
    return f"{dataset_prefix(published_name, version)}/partitions/manifest.json"


def views_manifest(
    *,
    dataset_id: str,
    published_name: str,
    version: str,
    contracts: Mapping[str, Any],
    image_size: int,
    partitions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """The reuse gate: every training partition view and its manifest digest."""
    ordered = sorted((dict(p) for p in partitions), key=lambda p: p["partition_id"])
    ids = [p["partition_id"] for p in ordered]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("partition views must be non-empty and unique")
    for partition in ordered:
        if set(partition) != {"partition_id", "prefix", "manifest_sha256", "sample_count"}:
            raise ValueError(f"partition view {partition['partition_id']} has unexpected fields")
        if partition["prefix"] != partition_view_prefix(published_name, version, partition["partition_id"]):
            raise ValueError(f"partition view {partition['partition_id']} is outside the publication")
        if not _SHA256_RE.fullmatch(partition["manifest_sha256"]):
            raise ValueError(f"partition view {partition['partition_id']} has an invalid digest")
    return {
        "schema_version": VIEWS_SCHEMA,
        "dataset_id": dataset_id,
        "published_name": published_name,
        "version": version,
        "contracts": dict(contracts),
        "image_size": int(image_size),
        "partitions": ordered,
        "samples": sum(int(p["sample_count"]) for p in ordered),
    }


def dataset_index_record(
    spec: DatasetSpec,
    *,
    published_name: str,
    version: str,
    views_sha256: str,
    lineage_sha256: str,
) -> dict[str, Any]:
    canonical = canonical_dataset_spec(spec)
    for name, digest in (("views_sha256", views_sha256), ("lineage_sha256", lineage_sha256)):
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"{name} must be 64 lowercase hex digits")
    return {
        "schema_version": DATASET_INDEX_SCHEMA,
        "dataset_id": sha256_bytes(canonical_json_bytes(canonical)),
        "spec": canonical,
        "published_name": published_name,
        "version": version,
        "views_key": views_manifest_key(published_name, version),
        "views_sha256": views_sha256,
        "lineage_key": lineage_key(published_name, version),
        "lineage_sha256": lineage_sha256,
    }


def validate_dataset_index_record(
    record: Mapping[str, Any],
    spec: DatasetSpec,
    *,
    published_name: str,
) -> None:
    """Reject an index record that does not describe exactly ``spec``."""
    if record.get("schema_version") != DATASET_INDEX_SCHEMA:
        raise ValueError("unsupported dataset index schema")
    canonical = canonical_dataset_spec(spec)
    if record.get("spec") != canonical:
        raise ValueError("dataset index spec differs from the requested spec")
    if record.get("published_name") != published_name:
        raise ValueError("dataset index is for another published dataset")
    expected = dataset_index_record(
        spec,
        published_name=published_name,
        version=str(record.get("version")),
        views_sha256=str(record.get("views_sha256")),
        lineage_sha256=str(record.get("lineage_sha256")),
    )
    if dict(record) != expected:
        raise ValueError("dataset index record has unexpected fields")


def validate_views(
    record: Mapping[str, Any],
    *,
    views_payload: bytes,
    expected_contracts: Mapping[str, Any],
) -> dict[str, Any]:
    """Check the partition views manifest against the index and running code.

    A publication packed under other contracts or another image size is
    never reused. Returns the parsed views manifest.
    """
    if sha256_bytes(views_payload) != record["views_sha256"]:
        raise ValueError("partition views manifest differs from the index")
    views = json.loads(views_payload)
    if views.get("schema_version") != VIEWS_SCHEMA:
        raise ValueError("unsupported partition views schema")
    rebuilt = views_manifest(
        dataset_id=record["dataset_id"],
        published_name=record["published_name"],
        version=record["version"],
        contracts=views["contracts"],
        image_size=views["image_size"],
        partitions=views["partitions"],
    )
    if canonical_json_bytes(rebuilt) != views_payload:
        raise ValueError("partition views manifest is not canonical")
    if views["contracts"] != dict(expected_contracts):
        raise ValueError("publication was packed under other contracts")
    if int(views["image_size"]) != int(record["spec"]["image_size"]):
        raise ValueError("publication image size differs from the spec")
    return views
