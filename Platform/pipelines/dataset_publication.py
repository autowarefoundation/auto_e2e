"""Pure contracts for publishing packed Flyte outputs as one dataset snapshot."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from typing import Any, Mapping, NamedTuple, Sequence


PUBLICATION_SCHEMA = "v1"
_DATASET_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_VERSION_RE = re.compile(r"^v[0-9]+(?:\.[0-9]+)*$")


class DatasetPublication(NamedTuple):
    manifest_key: str
    manifest_sha256: str


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {"sort_keys": True}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return json.dumps(value, **options).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def dataset_prefix(dataset: str, version: str) -> str:
    if not _DATASET_RE.fullmatch(dataset):
        raise ValueError(f"invalid published dataset id {dataset!r}")
    if not _VERSION_RE.fullmatch(version):
        raise ValueError(f"invalid dataset version {version!r}")
    return f"{dataset}/{version}"


def shard_key(dataset: str, version: str, shard: str) -> str:
    if not shard.endswith(".tar") or "/" in shard or shard in {".tar", "..tar"}:
        raise ValueError(f"invalid shard name {shard!r}")
    return f"{dataset_prefix(dataset, version)}/shards/{shard}"


def pool_key(dataset: str, version: str, relative_key: str) -> str:
    if (
        not relative_key.startswith("pool/")
        or relative_key.endswith("/")
        or ".." in relative_key.split("/")
    ):
        raise ValueError(f"invalid pool key {relative_key!r}")
    return f"{dataset_prefix(dataset, version)}/{relative_key}"


def episode_path_key(
    dataset: str,
    version: str,
    episode_filename: str,
) -> str:
    if (
        not episode_filename.endswith(".f64")
        or "/" in episode_filename
        or not re.fullmatch(r"[A-Za-z0-9_-]+\.f64", episode_filename)
    ):
        raise ValueError(f"invalid episode path name {episode_filename!r}")
    return (
        f"{dataset_prefix(dataset, version)}/geo/episode_paths/"
        f"{episode_filename}"
    )


def geo_pointer_item(
    dataset: str,
    version: str,
    *,
    summary: Mapping[str, Any],
    n_samples: int,
    computed_at: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "pk": f"GEO#{dataset}#{version}",
        "sk": "META",
        "summary": canonical_json_bytes(summary).decode(),
        "geojson_key": (
            f"{dataset_prefix(dataset, version)}/geo/heatmap.geojson.gz"
        ),
        "n_samples": int(n_samples),
        "computed_at": computed_at,
        "dataset_manifest_sha256": manifest_sha256,
    }


def _single_value(
    results: Sequence[Mapping[str, Any]],
    path: Sequence[str],
    label: str,
) -> Any:
    values = []
    for result in results:
        value: Any = result
        for part in path:
            value = value[part]
        values.append(value)
    encoded = {canonical_json_bytes(value) for value in values}
    if len(encoded) != 1:
        raise ValueError(f"partition {label} values differ")
    return values[0]


def _merge_geo(
    results: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    version: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    nonempty = [result for result in results if result["manifest"]["total_samples"]]
    geo_results = [result["geo"] for result in nonempty if result.get("geo")]
    has_gps = any(result["manifest"].get("has_gps", False) for result in nonempty)
    if not geo_results:
        if has_gps:
            raise ValueError("GPS shards have no dataset-level geo artifacts")
        return None, None
    if len(geo_results) != len(nonempty):
        raise ValueError("geo artifacts are missing from a non-empty GPS partition")

    privacy = _single_value(
        [{"privacy": geo["privacy"]} for geo in geo_results],
        ("privacy",),
        "geo privacy",
    )
    grid_deg = float(privacy["heatmap_grid_degrees"])
    k_anonymity = int(privacy["k_anonymity"])
    if grid_deg <= 0 or k_anonymity <= 0:
        raise ValueError("invalid geo privacy policy")

    episode_names: set[str] = set()
    path_point_count = 0
    cell_stats: dict[tuple[int, int], dict[str, int]] = {}
    for geo in geo_results:
        for path in geo["episode_paths"]:
            filename = str(path["filename"])
            if filename in episode_names:
                raise ValueError(f"duplicate episode path {filename}")
            episode_names.add(filename)
            path_point_count += int(path["point_count"])
        for cell in geo["cells"]:
            key = (int(cell["lat_cell"]), int(cell["lon_cell"]))
            merged = cell_stats.setdefault(
                key, {"sample_count": 0, "episode_count": 0}
            )
            merged["sample_count"] += int(cell["sample_count"])
            merged["episode_count"] += int(cell["episode_count"])

    features = []
    published_cells = []
    for (lat_cell, lon_cell), counts in sorted(cell_stats.items()):
        if counts["episode_count"] < k_anonymity:
            continue
        published_cells.append((lat_cell, lon_cell))
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    (lon_cell + 0.5) * grid_deg,
                    (lat_cell + 0.5) * grid_deg,
                ],
            },
            "properties": counts,
        })

    bbox = None
    if published_cells:
        lat_cells = [cell[0] for cell in published_cells]
        lon_cells = [cell[1] for cell in published_cells]
        bbox = [
            min(lon_cells) * grid_deg,
            min(lat_cells) * grid_deg,
            (max(lon_cells) + 1) * grid_deg,
            (max(lat_cells) + 1) * grid_deg,
        ]

    summary = {
        "schema_version": "v1",
        "dataset": dataset,
        "version": version,
        "bbox": bbox,
        "episode_count": len(episode_names),
        "path_point_count": path_point_count,
        "sample_pose_count": sum(
            int(geo["sample_pose_count"]) for geo in geo_results
        ),
        "source_coordinate_dtype": _single_value(
            geo_results,
            ("source_coordinate_dtype",),
            "source coordinate dtype",
        ),
        "stored_coordinate_dtype": _single_value(
            geo_results,
            ("stored_coordinate_dtype",),
            "stored coordinate dtype",
        ),
        "timestamp_dtype": _single_value(
            geo_results, ("timestamp_dtype",), "timestamp dtype"
        ),
        "gps_accuracy_available": any(
            bool(geo.get("gps_accuracy_available", False))
            for geo in geo_results
        ),
        "privacy": privacy,
    }
    heatmap = {"type": "FeatureCollection", "features": features}
    return summary, heatmap


def merge_partition_results(
    partition_results: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    version: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Validate partition receipts and build the canonical manifest and geo set."""
    if not partition_results:
        raise ValueError("dataset publication has no partition results")
    dataset_prefix(dataset, version)
    results = sorted(
        partition_results,
        key=lambda result: (
            str(result["manifest"].get("partition_id") or ""),
            str(result["source_uri"]),
        ),
    )
    for result in results:
        if result.get("schema_version") != PUBLICATION_SCHEMA:
            raise ValueError("unsupported partition publication schema")
        if result["dataset_version"] != version:
            raise ValueError("partition dataset version differs from publication")

    nonempty = [result for result in results if result["manifest"]["total_samples"]]
    if not nonempty:
        raise ValueError("dataset publication contains no samples")

    source_dataset = _single_value(
        results, ("manifest", "dataset"), "source dataset"
    )
    source_revision = _single_value(
        results, ("manifest", "source_revision"), "source revision"
    )
    contracts = _single_value(results, ("manifest", "contracts"), "contracts")
    hz = _single_value(results, ("manifest", "hz"), "sample rate")
    image_size = _single_value(
        results, ("manifest", "image_size"), "image size"
    )
    num_views = _single_value(
        nonempty, ("manifest", "num_views"), "camera count"
    )
    geometry_type = _single_value(
        nonempty, ("manifest", "geometry_type"), "geometry type"
    )
    rig = _single_value(nonempty, ("rig",), "rig projection")

    partition_ids: set[str] = set()
    shard_names: set[str] = set()
    shards = []
    partitions = []
    for result in results:
        source_manifest = result["manifest"]
        partition_id = str(source_manifest.get("partition_id") or "")
        if partition_id in partition_ids:
            raise ValueError(f"duplicate partition id {partition_id!r}")
        partition_ids.add(partition_id)

        expected_shards = int(source_manifest["shards"])
        if len(result["shards"]) != expected_shards:
            raise ValueError(
                f"partition {partition_id!r} has {len(result['shards'])} "
                f"shards, manifest declares {expected_shards}"
            )
        result_names = sorted(str(shard["name"]) for shard in result["shards"])
        if result_names != sorted(source_manifest.get("shard_names", [])):
            raise ValueError(f"partition {partition_id!r} shard names differ")
        for shard in result["shards"]:
            name = str(shard["name"])
            if name in shard_names:
                raise ValueError(f"duplicate published shard {name}")
            shard_names.add(name)
            shards.append(dict(shard))

        partitions.append({
            "partition_id": partition_id or None,
            "source_uri": result["source_uri"],
            "source_manifest_sha256": result["source_manifest_sha256"],
            "sample_count": int(source_manifest["total_samples"]),
            "reasoning_label_count": int(
                source_manifest.get("reasoning_label_count", 0)
            ),
            "shard_count": expected_shards,
            "pool": result["pool"],
        })

    shards.sort(key=lambda shard: shard["name"])
    geo_summary, heatmap = _merge_geo(
        results, dataset=dataset, version=version
    )
    episode_count = (
        int(geo_summary["episode_count"])
        if geo_summary is not None
        else sum(int(result["manifest"].get("episodes", 0)) for result in results)
    )
    prefix = dataset_prefix(dataset, version)
    rig_payload = canonical_json_bytes(rig, pretty=True)
    manifest = {
        "schema_version": PUBLICATION_SCHEMA,
        "status": "ready",
        "dataset": dataset,
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "version": version,
        "contracts": contracts,
        "hz": int(hz),
        "image_size": int(image_size),
        "num_views": int(num_views),
        "geometry_type": geometry_type,
        "total_samples": sum(
            int(result["manifest"]["total_samples"]) for result in results
        ),
        "reasoning_label_count": sum(
            int(result["manifest"].get("reasoning_label_count", 0))
            for result in results
        ),
        "shards": len(shards),
        "shard_count": len(shards),
        "shard_entries": shards,
        "episodes": episode_count,
        "has_map": any(
            bool(result["manifest"].get("has_map", False)) for result in nonempty
        ),
        "has_world_model": any(
            bool(result["manifest"].get("has_world_model", False))
            for result in nonempty
        ),
        "has_reasoning_labels": any(
            bool(result["manifest"].get("has_reasoning_labels", False))
            for result in nonempty
        ),
        "has_gps": geo_summary is not None,
        "partitions": partitions,
        "rig": {
            "key": f"{prefix}/rig/projection.json",
            "sha256": sha256_bytes(rig_payload),
        },
        "geo": geo_summary,
    }
    return manifest, rig, heatmap


def gzip_json_bytes(value: Any) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as stream:
        stream.write(canonical_json_bytes(value))
    return output.getvalue()
