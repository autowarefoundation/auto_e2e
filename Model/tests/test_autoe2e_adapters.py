"""Dataset adapter tests: snapshot manifests to build plans."""

from __future__ import annotations

import io
import json

import pytest

from Platform.pipelines.autoe2e_adapters import (
    KITSCENES_UPSTREAM,
    NUPLAN_PACK_TASK_VERSION,
    current_pack_version,
    extract_unit_tar,
    normalized_nuplan_pack_params,
    read_source,
)
from Platform.pipelines.autoe2e_dataset import (
    AutoE2EDataset,
    DatasetSpec,
    ManifestRef,
    ReasoningSpec,
)
from Platform.pipelines.autoe2e_snapshot import (
    build_manifest,
    index_bytes,
    index_tar,
    manifest_bytes,
    snapshot_prefix,
    unit_entry,
    unit_index_key,
    unit_tar_key,
    write_unit_tar,
)
from Platform.pipelines.dataset_publication import sha256_bytes

KIT = AutoE2EDataset.KITSCENES
REASONING = ReasoningSpec("openai_compatible", "nvidia/Cosmos3-Nano", "action_relevant_reasoning_v3_temporal_front256", 10)


def _unit_tar(unit: str, files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    write_unit_tar(out, unit, ((p, len(b), io.BytesIO(b)) for p, b in sorted(files.items())))
    return out.getvalue()


def _snapshot(units=("scene-b", "scene-a", "scene-c"), split="train", upstream=KITSCENES_UPSTREAM):
    prefix = snapshot_prefix(KIT, "6fde0034", "snap")
    entries = []
    for unit in units:
        payload = _unit_tar(unit, {f"data/{split}/{unit}/a.jpg": b"jpg", "data/sequence_archives.csv": b"csv"})
        members = index_tar(io.BytesIO(payload))
        entries.append(unit_entry(
            unit, tar_key=unit_tar_key(prefix, unit), tar_size=len(payload),
            tar_sha256=sha256_bytes(payload), index_key=unit_index_key(prefix, unit),
            index_sha256=sha256_bytes(index_bytes(members)), members=members,
        ))
    return manifest_bytes(build_manifest(
        dataset=KIT, upstream=upstream, revision="6fde0034", snapshot_id="snap",
        provenance={"method": "s3-copy", "split": split}, units=entries,
    ))


def _spec(payload: bytes, **overrides) -> DatasetSpec:
    values = dict(
        dataset=KIT,
        source_manifest=ManifestRef("s3://datasets/kitscenes/raw-snapshots/6fde0034/snap/manifest.json", "v", sha256_bytes(payload)),
        pack_version=None,
        image_size=256,
        reasoning=REASONING,
    )
    values.update(overrides)
    if values["pack_version"] is None:
        values["pack_version"] = current_pack_version(values["dataset"])
    return DatasetSpec(**values)


def test_kitscenes_plan_has_one_partition_per_scene_in_manifest_order():
    payload = _snapshot()
    plan = read_source(_spec(payload), payload)
    assert [u.unit_id for u in plan.units] == ["scene-a", "scene-b", "scene-c"]
    assert plan.partitions == [["scene-a"], ["scene-b"], ["scene-c"]]
    assert (plan.published_name, plan.split, plan.revision, plan.labels) == ("kitscenes", "train", "6fde0034", True)


def test_unit_limit_takes_the_first_units():
    payload = _snapshot()
    plan = read_source(_spec(payload, unit_limit=2), payload)
    assert [u.unit_id for u in plan.units] == ["scene-a", "scene-b"]
    with pytest.raises(ValueError, match="unit_limit"):
        read_source(_spec(payload, unit_limit=4), payload)


def test_kitscenes_val_is_published_separately_and_unlabelled():
    payload = _snapshot(split="val")
    plan = read_source(_spec(payload, reasoning=None), payload)
    assert plan.published_name == "kitscenes-val" and plan.labels is False
    with pytest.raises(ValueError, match="not reasoning-labelled"):
        read_source(_spec(payload), payload)


def test_manifest_must_match_the_pinned_digest_and_dataset():
    payload = _snapshot()
    with pytest.raises(ValueError, match="pinned"):
        read_source(_spec(_snapshot(units=("x",))), payload)
    with pytest.raises(ValueError, match="upstream"):
        other = _snapshot(upstream="someone/else")
        read_source(_spec(other), other)


def test_kitscenes_rejects_pack_params():
    payload = _snapshot()
    with pytest.raises(ValueError, match="unsupported pack_params"):
        read_source(_spec(payload, pack_params={"partition_size": "2"}), payload)


def _l2d_snapshot(episodes=(0, 1, 2), with_osm=True) -> bytes:
    from Platform.pipelines.autoe2e_adapters import L2D_UPSTREAM
    from Platform.pipelines.l2d_snapshot import plan_units

    unit_files, episode_units = plan_units(
        episodes,
        meta_files=["meta/info.json"],
        # Episodes 0 and 1 share their files; episode 2 is in the next file.
        data_file_of=lambda e: f"data/chunk-000/file-{e // 2:03d}.parquet",
        video_files_of=lambda e: [f"videos/observation.images.rear/chunk-000/file-{e // 2:03d}.mp4"],
    )
    if with_osm:
        unit_files["osm"] = ["osm/germany-251020.osm.pbf"]
    prefix = snapshot_prefix(AutoE2EDataset.L2D, "1e7578b1", "snap")
    entries = []
    for unit, files in sorted(unit_files.items()):
        payload = _unit_tar(unit, {path: b"x" for path in files})
        members = index_tar(io.BytesIO(payload))
        entries.append(unit_entry(
            unit, tar_key=unit_tar_key(prefix, unit), tar_size=len(payload),
            tar_sha256=sha256_bytes(payload), index_key=unit_index_key(prefix, unit),
            index_sha256=sha256_bytes(index_bytes(members)), members=members,
        ))
    return manifest_bytes(build_manifest(
        dataset=AutoE2EDataset.L2D, upstream=L2D_UPSTREAM, revision="1e7578b1", snapshot_id="snap",
        provenance={"method": "hf", "split": "train", "episode_units": episode_units, "unit_files": unit_files},
        units=entries,
    ))


def _l2d_spec(payload: bytes, **overrides) -> DatasetSpec:
    values = dict(dataset=AutoE2EDataset.L2D, reasoning=None, image_size=512, pack_params={"partition_size": "2"})
    values.update(overrides)
    return _spec(payload, **values)


def test_l2d_partitions_are_episode_groups_over_shared_file_units():
    payload = _l2d_snapshot()
    plan = read_source(_l2d_spec(payload), payload)
    assert plan.published_name == "l2d" and plan.split == "train" and plan.labels is False
    assert plan.partition_groups == [["0", "1"], ["2"]]
    assert plan.partitions == [
        ["data.chunk-000.file-000.parquet", "meta", "videos.observation.images.rear.chunk-000.file-000.mp4"],
        ["data.chunk-000.file-001.parquet", "meta", "videos.observation.images.rear.chunk-000.file-001.mp4"],
    ]
    # The OSM unit is no partition's input but stays in the lineage.
    assert "osm" in {unit.unit_id for unit in plan.units}


def test_l2d_rejects_what_its_snapshot_fixes():
    payload = _l2d_snapshot()
    with pytest.raises(ValueError, match="partition_size"):
        read_source(_l2d_spec(payload, pack_params={}), payload)
    with pytest.raises(ValueError, match="unsupported pack_params"):
        read_source(_l2d_spec(payload, pack_params={"partition_size": "2", "osm": "x"}), payload)
    with pytest.raises(ValueError, match="reasoning"):
        read_source(_l2d_spec(payload, reasoning=REASONING), payload)
    with pytest.raises(ValueError, match="unit_limit"):
        read_source(_l2d_spec(payload, unit_limit=1), payload)
    without_osm = _l2d_snapshot(with_osm=False)
    with pytest.raises(ValueError, match="OSM"):
        read_source(_l2d_spec(without_osm), without_osm)


def _nuplan_manifest() -> bytes:
    return json.dumps({
        "schema_version": "nuplan_raw_snapshot_v1",
        "dataset": "nuplan/nuplan-v1.1",
        "dataset_revision": "nuplan-v1.1-complete",
        "archives": [
            {"archive_id": "maps", "object_uri": "s3://datasets/nuplan/raw-snapshots/x/archives/maps/m.zip", "sha256": "1" * 64},
        ],
    }).encode()


NUPLAN_PARAMS = normalized_nuplan_pack_params({"total_scenario_limit": "4096"})


def test_nuplan_plan_carries_validated_pack_params_and_archive_units():
    payload = _nuplan_manifest()
    spec = _spec(payload, dataset=AutoE2EDataset.NUPLAN, reasoning=None, pack_params=NUPLAN_PARAMS)
    plan = read_source(spec, payload)
    assert (plan.published_name, plan.revision, plan.labels) == ("nuplan", "nuplan-v1.1-complete", False)
    assert plan.pack_params == NUPLAN_PARAMS
    assert [(u.unit_id, u.tar_key) for u in plan.units] == [("maps", "nuplan/raw-snapshots/x/archives/maps/m.zip")]


def test_nuplan_requires_explicit_params_so_equal_meaning_is_one_id():
    payload = _nuplan_manifest()
    with pytest.raises(ValueError, match="explicitly"):
        read_source(_spec(payload, dataset=AutoE2EDataset.NUPLAN, reasoning=None, pack_params={"total_scenario_limit": "4096"}), payload)


def test_stale_pack_version_is_rejected():
    payload = _snapshot()
    with pytest.raises(ValueError, match="this code"):
        read_source(_spec(payload, pack_version="v3.3"), payload)
    assert NUPLAN_PACK_TASK_VERSION in current_pack_version(AutoE2EDataset.NUPLAN)
    assert current_pack_version(AutoE2EDataset.KITSCENES) != current_pack_version(AutoE2EDataset.NUPLAN)


@pytest.mark.parametrize(
    "overrides",
    [
        {"reasoning": REASONING},
        {"unit_limit": 3},
        {"pack_params": normalized_nuplan_pack_params({"total_scenario_limit": "zero"})},
        {"pack_params": normalized_nuplan_pack_params({"max_rejection_fraction": "1.5"})},
        {"pack_params": {**NUPLAN_PARAMS, "partition_size": "2"}},
    ],
)
def test_nuplan_rejects_unsupported_settings(overrides):
    payload = _nuplan_manifest()
    values = {"dataset": AutoE2EDataset.NUPLAN, "reasoning": None, "pack_params": NUPLAN_PARAMS, **overrides}
    with pytest.raises(ValueError):
        read_source(_spec(payload, **values), payload)


def test_extract_rebuilds_the_upstream_tree_and_shares_identical_files(tmp_path):
    a = _unit_tar("scene-a", {"data/train/scene-a/a.jpg": b"A", "data/sequence_archives.csv": b"csv"})
    b = _unit_tar("scene-b", {"data/train/scene-b/b.jpg": b"B", "data/sequence_archives.csv": b"csv"})
    assert extract_unit_tar(io.BytesIO(a), "scene-a", tmp_path) == ["data/sequence_archives.csv", "data/train/scene-a/a.jpg"]
    assert extract_unit_tar(io.BytesIO(b), "scene-b", tmp_path) == ["data/train/scene-b/b.jpg"]
    assert (tmp_path / "data/train/scene-b/b.jpg").read_bytes() == b"B"


def test_extract_rejects_conflicting_shared_files_and_foreign_members(tmp_path):
    extract_unit_tar(io.BytesIO(_unit_tar("scene-a", {"data/x.csv": b"1"})), "scene-a", tmp_path)
    with pytest.raises(ValueError, match="disagree"):
        extract_unit_tar(io.BytesIO(_unit_tar("scene-b", {"data/x.csv": b"2"})), "scene-b", tmp_path)
    with pytest.raises(ValueError, match="unexpected"):
        extract_unit_tar(io.BytesIO(_unit_tar("scene-c", {"y": b"1"})), "scene-d", tmp_path)
