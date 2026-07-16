"""Canonical dataset publication contract tests."""

from __future__ import annotations

import gzip
import json

import pytest

from Platform.pipelines.dataset_publication import (
    PUBLICATION_SCHEMA,
    episode_path_key,
    geo_pointer_item,
    gzip_json_bytes,
    merge_partition_results,
    pool_key,
    rig_key,
    shard_key,
)


def _result(
    partition_id: str,
    *,
    shard: str,
    episode_count: int,
    samples: int = 10,
) -> dict:
    return {
        "schema_version": PUBLICATION_SCHEMA,
        "source_uri": f"s3://flyte/{partition_id}",
        "source_manifest_sha256": partition_id * 8,
        "dataset_version": "v2.1",
        "manifest": {
            "dataset": "yaak-ai/L2D",
            "source_revision": "main",
            "dataset_version": "v2.1",
            "partition_id": partition_id,
            "total_samples": samples,
            "shards": 1,
            "shard_names": [shard],
            "reasoning_label_count": 1,
            "contracts": {
                "uid_schema_version": "v1",
                "shard_schema_version": "v3",
            },
            "hz": 10,
            "image_size": 256,
            "num_views": 6,
            "geometry_type": "pinhole",
            "has_map": True,
            "has_world_model": True,
            "has_reasoning_labels": True,
            "has_gps": True,
        },
        "rig": {
            "schema_version": "v1",
            "dataset": "yaak-ai/L2D",
            "geometry_type": "pinhole",
            "image_size": 256,
            "projection": {
                "type": "pinhole",
                "matrix": [[
                    [128.0, 0.0, 0.0, 0.0],
                    [0.0, 128.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]],
            },
        },
        "shards": [{
            "name": shard,
            "key": f"l2d/v2.1/shards/{shard}",
            "byte_size": 123,
            "etag": partition_id,
            "content_identity": partition_id * 8,
        }],
        "pool": {
            "object_count": 4,
            "byte_size": 400,
            "digest": partition_id * 8,
        },
        "geo": {
            "privacy": {
                "k_anonymity": 5,
                "endpoint_exclusion_frames": 10,
                "heatmap_grid_degrees": 0.01,
            },
            "source_coordinate_dtype": "float32",
            "stored_coordinate_dtype": "float64",
            "timestamp_dtype": "int64_ns",
            "gps_accuracy_available": False,
            "sample_pose_count": samples,
            "sample_pose_source": {
                "bucket": "flyte",
                "key": f"{partition_id}/geo/sample_pose.parquet",
                "etag": partition_id,
                "size": 100,
                "content_identity": partition_id * 8,
            },
            "episode_paths": [
                {
                    "filename": f"{partition_id}-{index}.f64",
                    "point_count": 20,
                    "key": (
                        f"l2d/v2.1/geo/episode_paths/"
                        f"{partition_id}-{index}.f64"
                    ),
                    "byte_size": 640,
                    "content_identity": f"{partition_id}-{index}",
                }
                for index in range(episode_count)
            ],
            "cells": [{
                "lat_cell": 4900,
                "lon_cell": 1100,
                "sample_count": episode_count * 20,
                "episode_count": episode_count,
            }],
        },
    }


def test_publication_keys_are_version_scoped_and_traversal_safe():
    assert shard_key("l2d", "v2.1", "train-000000.tar") == (
        "l2d/v2.1/shards/train-000000.tar"
    )
    assert pool_key("l2d", "v2.1", "pool/frame.jpg") == (
        "l2d/v2.1/pool/frame.jpg"
    )
    assert rig_key("l2d", "v2.1", "a" * 64) == (
        f"l2d/v2.1/rig/{'a' * 64}.json"
    )
    assert episode_path_key("l2d", "v2.1", "000012.f64") == (
        "l2d/v2.1/geo/episode_paths/000012.f64"
    )
    with pytest.raises(ValueError):
        shard_key("l2d", "v2.1", "../escape.tar")
    with pytest.raises(ValueError):
        pool_key("l2d", "v2.1", "pool/../escape.jpg")
    with pytest.raises(ValueError):
        rig_key("l2d", "v2.1", "../projection")


def test_merge_applies_k_anonymity_across_all_partitions():
    manifest, rigs, heatmap = merge_partition_results(
        [
            _result(
                "part-a",
                shard="part-a-train-000000.tar",
                episode_count=3,
            ),
            _result(
                "part-b",
                shard="part-b-train-000000.tar",
                episode_count=2,
            ),
        ],
        dataset="l2d",
        version="v2.1",
    )

    assert manifest["total_samples"] == 20
    assert manifest["shards"] == 2
    assert manifest["shard_count"] == 2
    assert manifest["rig_count"] == 1
    assert [entry["name"] for entry in manifest["shard_entries"]] == [
        "part-a-train-000000.tar",
        "part-b-train-000000.tar",
    ]
    assert manifest["episodes"] == 5
    assert manifest["has_gps"] is True
    assert manifest["reasoning_label_count"] == 2
    assert manifest["geo"]["episode_count"] == 5
    assert manifest["geo"]["sample_pose_count"] == 20
    assert manifest["geo"]["bbox"] == [11.0, 49.0, 11.01, 49.01]
    assert len(rigs) == 1
    rig_digest, rig = next(iter(rigs.items()))
    assert rig["geometry_type"] == "pinhole"
    assert {
        entry["rig"]["sha256"] for entry in manifest["shard_entries"]
    } == {rig_digest}
    assert manifest["shard_entries"][0]["rig"]["key"] == (
        f"l2d/v2.1/rig/{rig_digest}.json"
    )
    assert heatmap is not None
    assert len(heatmap["features"]) == 1
    assert heatmap["features"][0]["properties"]["episode_count"] == 5


def test_merge_scopes_different_rigs_to_their_shards():
    first = _result(
        "part-a", shard="part-a-train-000000.tar", episode_count=3
    )
    second = _result(
        "part-b", shard="part-b-train-000000.tar", episode_count=2
    )
    second["rig"]["projection"]["matrix"][0][0][0] = 129.0

    manifest, rigs, _ = merge_partition_results(
        [first, second], dataset="l2d", version="v2.1"
    )

    assert manifest["rig_count"] == 2
    assert len(rigs) == 2
    assert len({
        entry["rig"]["key"] for entry in manifest["shard_entries"]
    }) == 2


def test_merge_rejects_invalid_rig_and_duplicate_shards():
    first = _result(
        "part-a", shard="part-a-train-000000.tar", episode_count=3
    )
    second = _result(
        "part-b", shard="part-b-train-000000.tar", episode_count=2
    )
    second["rig"]["image_size"] = 512
    with pytest.raises(ValueError, match="rig"):
        merge_partition_results(
            [first, second], dataset="l2d", version="v2.1"
        )

    second = _result(
        "part-b", shard="part-a-train-000000.tar", episode_count=2
    )
    with pytest.raises(ValueError, match="duplicate published shard"):
        merge_partition_results(
            [first, second], dataset="l2d", version="v2.1"
        )


def test_merge_rejects_gps_snapshot_without_geo_products():
    result = _result(
        "part-a", shard="part-a-train-000000.tar", episode_count=1
    )
    result["geo"] = None
    with pytest.raises(ValueError, match="no dataset-level geo"):
        merge_partition_results([result], dataset="l2d", version="v2.1")


def test_merge_rejects_unmaterializable_reasoning_label_count():
    result = _result(
        "part-a", shard="part-a-train-000000.tar", episode_count=1
    )
    result["manifest"]["reasoning_label_count"] = 100_001
    with pytest.raises(ValueError, match="materialization limit"):
        merge_partition_results([result], dataset="l2d", version="v2.1")

    result["manifest"]["reasoning_label_count"] = -1
    with pytest.raises(ValueError, match="must not be negative"):
        merge_partition_results([result], dataset="l2d", version="v2.1")


def test_geo_pointer_is_small_and_manifest_scoped():
    item = geo_pointer_item(
        "l2d",
        "v2.1",
        summary={"bbox": [11.0, 49.0, 11.1, 49.1]},
        n_samples=10,
        computed_at="2026-07-15T00:00:00Z",
        manifest_sha256="a" * 64,
    )
    assert item["pk"] == "GEO#l2d#v2.1"
    assert item["geojson_key"] == "l2d/v2.1/geo/heatmap.geojson.gz"
    assert item["dataset_manifest_sha256"] == "a" * 64


def test_geojson_gzip_is_deterministic():
    value = {"type": "FeatureCollection", "features": []}
    first = gzip_json_bytes(value)
    second = gzip_json_bytes(value)
    assert first == second
    assert json.loads(gzip.decompress(first)) == value
