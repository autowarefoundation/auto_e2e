"""Geospatial shard codecs and dataset-level artifact generation.

The packed member names retain the historical ``.npy`` suffix, but, like
``ego.npy``, their payloads are fixed little-endian binary rather than NumPy
``.npy`` containers. Keeping the encoding explicit makes it straightforward for
the Go console API and browser clients to decode without a NumPy dependency.
"""

from __future__ import annotations

import gzip
import json
import math
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np


POSE_SCHEMA_VERSION = "v1"
GPS_SCHEMA_VERSION = "v1"
EPISODE_PATH_SCHEMA_VERSION = "v1"

# latitude_deg:f64, longitude_deg:f64, heading_deg_cw_from_north:f64,
# timestamp_ns:i64, gps_accuracy_m:f32.
_POSE_STRUCT = struct.Struct("<dddqf")
POSE_BINARY_SIZE = _POSE_STRUCT.size
GPS_FUTURE_POINTS = 65


def encode_pose(pose: Mapping[str, Any]) -> bytes:
    """Encode one absolute vehicle pose using the stable v1 layout."""
    return _POSE_STRUCT.pack(
        float(pose["latitude_deg"]),
        float(pose["longitude_deg"]),
        float(pose["heading_deg_cw_from_north"]),
        int(pose["timestamp_ns"]),
        float(pose.get("gps_accuracy_m", math.nan)),
    )


def decode_pose(payload: bytes) -> dict[str, float | int]:
    """Decode a v1 pose member, rejecting truncated or ambiguous payloads."""
    if len(payload) != POSE_BINARY_SIZE:
        raise ValueError(
            f"pose payload must be {POSE_BINARY_SIZE} bytes, got {len(payload)}"
        )
    lat, lon, heading, timestamp_ns, accuracy = _POSE_STRUCT.unpack(payload)
    return {
        "latitude_deg": lat,
        "longitude_deg": lon,
        "heading_deg_cw_from_north": heading,
        "timestamp_ns": timestamp_ns,
        "gps_accuracy_m": accuracy,
    }


def encode_gps_future(points: Any) -> bytes:
    """Encode current + 64 future ``[lat, lon]`` points as little-endian f64."""
    array = np.asarray(points, dtype="<f8")
    expected = (GPS_FUTURE_POINTS, 2)
    if array.shape != expected:
        raise ValueError(f"gps_future must have shape {expected}, got {array.shape}")
    return np.ascontiguousarray(array).tobytes()


def decode_gps_future(payload: bytes) -> np.ndarray:
    """Decode a v1 current+future GPS member into a copied ``[65,2]`` array."""
    expected_bytes = GPS_FUTURE_POINTS * 2 * np.dtype("<f8").itemsize
    if len(payload) != expected_bytes:
        raise ValueError(
            f"gps payload must be {expected_bytes} bytes, got {len(payload)}"
        )
    return np.frombuffer(payload, dtype="<f8").reshape(GPS_FUTURE_POINTS, 2).copy()


def geospatial_members(sample: Mapping[str, Any]) -> dict[str, bytes]:
    """Return optional ``pose.npy`` and ``gps.npy`` members for one sample."""
    pose = sample.get("pose_current")
    gps_future = sample.get("gps_future")
    if pose is None and gps_future is None:
        return {}
    if pose is None or gps_future is None:
        raise ValueError("pose_current and gps_future must be present together")
    return {
        "pose.npy": encode_pose(pose),
        "gps.npy": encode_gps_future(gps_future),
    }


def _valid_coordinate(lat: float, lon: float) -> bool:
    return (
        math.isfinite(lat)
        and math.isfinite(lon)
        and -90.0 <= lat <= 90.0
        and -180.0 <= lon <= 180.0
        and not (lat == 0.0 and lon == 0.0)
    )


def write_geo_artifacts(
    dataset: Any,
    output_dir: str | Path,
    *,
    dataset_name: str,
    dataset_version: str,
    k_anonymity: int = 5,
    endpoint_exclusion_frames: int = 10,
) -> dict[str, Any]:
    """Write full episode paths, sample poses, a summary, and a coarse heatmap.

    ``dataset`` is an L2DDataset-like object exposing ``episode_indices()``,
    ``episode_path(ep)`` and ``sample_pose_records()``. Exact paths and the
    per-sample parquet remain access-controlled source artifacts. The GeoJSON
    heatmap excludes trip endpoints and suppresses cells represented by fewer
    than ``k_anonymity`` distinct episodes.
    """
    root = Path(output_dir) / "geo"
    paths_dir = root / "episode_paths"
    paths_dir.mkdir(parents=True, exist_ok=True)

    cell_stats: dict[tuple[int, int], dict[str, Any]] = {}
    episode_count = 0
    path_point_count = 0
    grid_deg = 0.01

    for episode_index in dataset.episode_indices():
        path = np.asarray(dataset.episode_path(episode_index), dtype="<f8")
        if path.ndim != 2 or path.shape[1] != 4:
            raise ValueError(
                f"episode path must have shape [N,4], got {path.shape}"
            )
        episode_count += 1

        valid_rows = []
        for row in path:
            lat, lon = float(row[0]), float(row[1])
            if _valid_coordinate(lat, lon):
                valid_rows.append(row)

        start = min(endpoint_exclusion_frames, len(valid_rows))
        end = max(start, len(valid_rows) - endpoint_exclusion_frames)
        published_path = np.asarray(valid_rows[start:end], dtype="<f8")
        if published_path.size:
            published_path = published_path.reshape(-1, 4)
            (paths_dir / f"{int(episode_index):06d}.f64").write_bytes(
                np.ascontiguousarray(published_path).tobytes()
            )
            path_point_count += int(published_path.shape[0])
        for row in published_path:
            lat, lon = float(row[0]), float(row[1])
            cell = (math.floor(lat / grid_deg), math.floor(lon / grid_deg))
            stats = cell_stats.setdefault(
                cell, {"sample_count": 0, "episodes": set()}
            )
            stats["sample_count"] += 1
            stats["episodes"].add(int(episode_index))

    records = list(dataset.sample_pose_records())
    parquet_path = root / "sample_pose.parquet"
    if records:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(records), parquet_path)

    features = []
    published_cells = []
    for (lat_cell, lon_cell), stats in sorted(cell_stats.items()):
        episode_ids: set[int] = stats["episodes"]
        if len(episode_ids) < k_anonymity:
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
            "properties": {
                "sample_count": stats["sample_count"],
                "episode_count": len(episode_ids),
            },
        })
    if published_cells:
        lat_cells = [cell[0] for cell in published_cells]
        lon_cells = [cell[1] for cell in published_cells]
        bbox: list[float] | None = [
            min(lon_cells) * grid_deg,
            min(lat_cells) * grid_deg,
            (max(lon_cells) + 1) * grid_deg,
            (max(lat_cells) + 1) * grid_deg,
        ]
    else:
        bbox = None

    summary = {
        "schema_version": "v1",
        "dataset": dataset_name,
        "version": dataset_version,
        "bbox": bbox,
        "episode_count": episode_count,
        "path_point_count": path_point_count,
        "sample_pose_count": len(records),
        "source_coordinate_dtype": "float32",
        "stored_coordinate_dtype": "float64",
        "timestamp_dtype": "int64_ns",
        "gps_accuracy_available": False,
        "privacy": {
            "k_anonymity": k_anonymity,
            "endpoint_exclusion_frames": endpoint_exclusion_frames,
            "heatmap_grid_degrees": grid_deg,
        },
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    heatmap = {
        "type": "FeatureCollection",
        "features": features,
    }
    with open(root / "heatmap.geojson.gz", "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(
                json.dumps(
                    heatmap, separators=(",", ":"), sort_keys=True
                ).encode()
            )

    return summary
