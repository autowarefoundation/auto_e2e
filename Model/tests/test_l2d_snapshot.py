"""L2D snapshot layout: safe paths, injective unit ids, and pinned OSM extracts."""

from __future__ import annotations

import pytest

from Platform.pipelines.l2d_snapshot import file_unit_id, partition_plan, plan_units, validate_osm_extract

SHA = "a" * 64


def test_unit_ids_refuse_unsafe_or_foreign_paths():
    assert file_unit_id("videos/observation.images.rear/chunk-000/file-001.mp4") == (
        "videos.observation.images.rear.chunk-000.file-001.mp4"
    )
    for path in ("data/../../etc/passwd", "data//x", "/data/x", "meta/info.json", "osm/x.pbf", "data\\x"):
        with pytest.raises(ValueError):
            file_unit_id(path)
    with pytest.raises(ValueError):
        plan_units([0], meta_files=["meta/../x"], data_file_of=lambda e: "data/a", video_files_of=lambda e: [])


def test_two_paths_on_one_unit_id_are_refused():
    with pytest.raises(ValueError, match="share unit id"):
        plan_units(
            [0, 1], meta_files=["meta/info.json"],
            data_file_of=lambda e: ["data/a/b.c", "data/a.b/c"][e], video_files_of=lambda e: [],
        )


def test_partitions_group_episodes_in_numeric_order():
    _, episode_units = plan_units(
        [10, 2, 1], meta_files=["meta/info.json"],
        data_file_of=lambda e: f"data/file-{e}.parquet", video_files_of=lambda e: [],
    )
    units, groups = partition_plan({"episode_units": episode_units}, 2)
    assert groups == [["1", "2"], ["10"]]
    assert units[1] == ["data.file-10.parquet", "meta"]


@pytest.mark.parametrize("url", [
    "http://download.geofabrik.de/europe/germany-261027.osm.pbf",
    "https://evil.example/europe/germany-261027.osm.pbf",
    "https://download.geofabrik.de/europe/germany-latest.osm.pbf",
    "https://user:pw@download.geofabrik.de/europe/germany-261027.osm.pbf",
    "https://download.geofabrik.de/europe/germany-261027.osm.pbf?token=x",
    "https://download.geofabrik.de:8443/europe/germany-261027.osm.pbf",
])
def test_osm_extracts_come_only_from_dated_geofabrik_files(url):
    with pytest.raises(ValueError):
        validate_osm_extract(url, SHA)


def test_osm_extract_needs_a_pinned_sha256():
    url = "https://download.geofabrik.de/europe/germany-261027.osm.pbf"
    assert validate_osm_extract(url, SHA) == "germany-261027.osm.pbf"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_osm_extract(url, "")
