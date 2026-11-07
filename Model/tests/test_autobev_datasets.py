"""nuScenes, Argoverse 2, ONCE and SimBEV as wf_autoe2e datasets: acquisition plans, adapters, packs."""

from __future__ import annotations

import io
import json
import zipfile
import zlib
from pathlib import Path

import pytest

from Platform.pipelines import autobev_sources
from Platform.pipelines.autoe2e_adapters import AUTOBEV_SOURCES, current_pack_version, read_source
from Platform.pipelines.autoe2e_dataset import AutoE2EDataset, DatasetSpec, ManifestRef, ReasoningSpec
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

AUTOBEV = tuple(AUTOBEV_SOURCES)


def _tar(unit: str, files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    write_unit_tar(out, unit, ((path, len(data), io.BytesIO(data)) for path, data in sorted(files.items())))
    return out.getvalue()


def _snapshot(dataset: AutoE2EDataset, units: dict[str, dict[str, bytes]], upstream: str | None = None):
    """(manifest bytes, {S3 key: object bytes}) of a snapshot holding ``units``."""
    prefix = snapshot_prefix(dataset, "rev1", "snap")
    objects, entries = {}, []
    for unit, files in units.items():
        payload = _tar(unit, files)
        members = index_tar(io.BytesIO(payload))
        index = index_bytes(members)
        objects[unit_tar_key(prefix, unit)] = payload
        objects[unit_index_key(prefix, unit)] = index
        entries.append(unit_entry(
            unit, tar_key=unit_tar_key(prefix, unit), tar_size=len(payload), tar_sha256=sha256_bytes(payload),
            index_key=unit_index_key(prefix, unit), index_sha256=sha256_bytes(index), members=members,
        ))
    manifest = manifest_bytes(build_manifest(
        dataset=dataset, upstream=upstream or AUTOBEV_SOURCES[dataset].upstream, revision="rev1",
        snapshot_id="snap", provenance={"method": "upstream", "split": "train"}, units=entries,
    ))
    return manifest, objects


def _spec(dataset: AutoE2EDataset, payload: bytes, **overrides) -> DatasetSpec:
    values = dict(
        dataset=dataset,
        source_manifest=ManifestRef(f"s3://datasets/{dataset.value}/raw-snapshots/rev1/snap/manifest.json", "v",
                                    sha256_bytes(payload)),
        pack_version=current_pack_version(dataset),
        image_size=1344,
    )
    values.update(overrides)
    return DatasetSpec(**values)


@pytest.mark.parametrize("dataset", [d for d in AUTOBEV if d is not AutoE2EDataset.ARGOVERSE2])
def test_shared_unit_datasets_plan_the_packer_partitions_over_every_unit(dataset):
    from Platform.pipelines.autoe2e_adapters import autobev_packer

    payload, _ = _snapshot(dataset, {"b": {"x": b"1"}, "a": {"y": b"2"}})
    plan = read_source(_spec(dataset, payload), payload)
    assert plan.published_name == dataset.value and plan.split == "train" and plan.revision == "rev1"
    assert plan.shared_units and not plan.labels and not plan.console
    assert plan.partitions == [["a", "b"]] * autobev_packer(dataset).PARTITIONS
    with pytest.raises(ValueError, match="unit_limit"):
        read_source(_spec(dataset, payload, unit_limit=1), payload)


def test_argoverse2_plans_one_partition_per_log_and_allows_a_unit_limit():
    dataset = AutoE2EDataset.ARGOVERSE2
    payload, _ = _snapshot(dataset, {"train-b": {"x": b"1"}, "train-a": {"y": b"2"}, "val-c": {"z": b"3"}})
    plan = read_source(_spec(dataset, payload), payload)
    assert plan.partitions == [["train-a"], ["train-b"], ["val-c"]] and not plan.shared_units
    assert read_source(_spec(dataset, payload, unit_limit=2), payload).partitions == [["train-a"], ["train-b"]]


@pytest.mark.parametrize("dataset", AUTOBEV)
def test_adapter_refuses_what_would_not_describe_the_content(dataset):
    payload, _ = _snapshot(dataset, {"a": {"x": b"1"}})
    with pytest.raises(ValueError, match="pack_version"):
        read_source(_spec(dataset, payload, pack_version="stale"), payload)
    with pytest.raises(ValueError, match="unsupported pack_params"):
        read_source(_spec(dataset, payload, pack_params={"partitions": "8"}), payload)
    with pytest.raises(ValueError, match="image_size"):
        read_source(_spec(dataset, payload, image_size=512), payload)
    with pytest.raises(ValueError, match="reasoning"):
        read_source(_spec(dataset, payload, reasoning=ReasoningSpec("t", "m", "p", 10)), payload)
    other, _ = _snapshot(dataset, {"a": {"x": b"1"}}, upstream="https://elsewhere")
    with pytest.raises(ValueError, match="upstream"):
        read_source(_spec(dataset, other), other)


def test_pack_versions_name_each_packer_contract():
    versions = {current_pack_version(dataset) for dataset in AUTOBEV}
    assert len(versions) == len(AUTOBEV)
    assert all("autobev_members_v1" in version for version in versions)


# --- acquisition ----------------------------------------------------------------


def test_drive_snapshots_list_every_archive_and_their_license():
    units, provenance = autobev_sources.plan_units(AutoE2EDataset.ONCE)
    assert "cam03" not in units and {"infos", "lidar", "lanes", "cam01"} <= set(units)
    assert provenance["license"]["commercial_use"] is False and provenance["method"] == "upstream"
    units, provenance = autobev_sources.plan_units(AutoE2EDataset.SIMBEV)
    assert len(units) == 10 and provenance["drive_files"]["infos"] == "1E44LJVwGj1tV5t1cCBacBCXkn4jKHKAD"


class _ListingS3:
    """list_objects_v2 over an in-memory key set."""

    def __init__(self, keys):
        self.keys = sorted(keys)

    def get_paginator(self, _name):
        return self

    def paginate(self, Bucket, Prefix, Delimiter=None):  # noqa: N803 - boto3 names
        keys = [key for key in self.keys if key.startswith(Prefix)]
        if Delimiter:
            prefixes = sorted({Prefix + key[len(Prefix):].split("/", 1)[0] + "/" for key in keys})
            return [{"CommonPrefixes": [{"Prefix": prefix} for prefix in prefixes]}]
        return [{"Contents": [{"Key": key, "Size": 1} for key in keys]}]


def test_argoverse2_acquisition_skips_pittsburgh_logs(monkeypatch):
    base = "datasets/av2/sensor"
    keys = [
        f"{base}/train/a/map/log_map_archive_a____PIT_city_1.json", f"{base}/train/a/map/a___img_Sim2_city.json",
        f"{base}/train/b/map/log_map_archive_b____MIA_city_2.json",
        f"{base}/val/c/map/log_map_archive_c____ATX_city_3.json",
        f"{base}/test/d/map/log_map_archive_d____MIA_city_4.json",
    ]
    monkeypatch.setattr(autobev_sources, "unsigned_s3", lambda: _ListingS3(keys))
    units, provenance = autobev_sources.plan_units(AutoE2EDataset.ARGOVERSE2)
    assert units == ["train-b", "val-c"]
    assert provenance["held_out_logs"] == ["train-a"]
    assert provenance["logs"]["val-c"] == {"split": "val", "log": "c", "map": "log_map_archive_c____ATX_city_3.json"}


def test_drive_download_resumes_after_a_quota_cut(tmp_path, monkeypatch):
    payload = bytes(range(256)) * 40
    calls = {"failures": 1}

    def fake_range(url, start, end):
        if calls["failures"] and start > 0:
            calls["failures"] -= 1
            raise IOError("quota")
        return payload[start: end + 1]

    monkeypatch.setattr(autobev_sources, "DRIVE_CHUNK", 4096)
    monkeypatch.setattr(autobev_sources, "http_size", lambda url: len(payload))
    monkeypatch.setattr(autobev_sources, "http_range", fake_range)
    target = tmp_path / "file.tar"
    with pytest.raises(IOError):
        autobev_sources.drive_download("id", target)
    partial = tmp_path / "file.tar.partial"
    assert partial.stat().st_size == 4096 and not target.exists()
    autobev_sources.drive_download("id", target)
    assert target.read_bytes() == payload and not partial.exists()


def test_retry_backs_off_and_gives_up(monkeypatch):
    waits = []
    with pytest.raises(IOError, match="gave up"):
        autobev_sources._retry(lambda: (_ for _ in ()).throw(IOError("403")), "x", sleep=waits.append)
    assert len(waits) == autobev_sources.DRIVE_ATTEMPTS - 1 and max(waits) == autobev_sources.DRIVE_MAX_BACKOFF_S


def test_zip_members_are_read_by_range_and_crc_checked():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("RGB-CAM_FRONT/SimBEV-scene-0000-frame-0070-RGB-CAM_FRONT.jpg", b"jpeg" * 100)
        archive.writestr("RGB-CAM_FRONT/SimBEV-scene-0000-frame-0071-RGB-CAM_FRONT.jpg", b"other")
    blob = out.getvalue()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        info = archive.infolist()[0]
    assert autobev_sources.zip_member("u", info, get=lambda url, start, end: blob[start: end + 1]) == b"jpeg" * 100
    broken = bytearray(blob)
    broken[info.header_offset + 30 + len(info.filename) + info.compress_size - 1] ^= 0xFF
    with pytest.raises((IOError, zlib.error)):
        autobev_sources.zip_member("u", info, get=lambda url, start, end: bytes(broken[start: end + 1]))


# --- packing from snapshot members ---------------------------------------------


class _ObjectS3:
    def __init__(self, objects):
        self.objects = objects

    def get_object(self, Bucket, Key, Range=None):  # noqa: N803 - boto3 names
        payload = self.objects[Key]
        if Range:
            start, end = (int(v) for v in Range.removeprefix("bytes=").split("-"))
            payload = payload[start: end + 1]
        return {"Body": io.BytesIO(payload)}


def test_snapshot_members_read_verified_members_by_upstream_path():
    from Platform.pipelines.autoe2e_tasks import SnapshotMembers

    payload, objects = _snapshot(AutoE2EDataset.ONCE, {"infos": {"data/1/1.json": b"{}"}, "lanes": {"x/y": b"z"}})
    plan = read_source(_spec(AutoE2EDataset.ONCE, payload), payload)
    files = SnapshotMembers(_ObjectS3(objects), "datasets", plan.units)
    assert sorted(files) == ["data/1/1.json", "x/y"] and files["x/y"] == b"z" and "x/y" in files
    tampered = dict(objects)
    key = plan.units[0].tar_key
    tampered[key] = tampered[key].replace(b"{}", b"[]")
    with pytest.raises(ValueError, match="differs"):
        SnapshotMembers(_ObjectS3(tampered), "datasets", plan.units)["data/1/1.json"]


def test_an_autobev_partition_is_packed_from_snapshot_members(tmp_path):
    from Platform.pipelines.autoe2e_tasks import SnapshotMembers, pack_autobev_partition
    from tests.test_autobev_packers import _once_files

    files = _once_files()
    units = {
        "infos": {k: v for k, v in files.items() if k.endswith(".json") and k.startswith("data/")},
        "lanes": {k: v for k, v in files.items() if k.startswith("ONCE_3DLanes/")},
        "sensors": {k: v for k, v in files.items() if not k.endswith(".json")},
    }
    payload, objects = _snapshot(AutoE2EDataset.ONCE, units)
    plan = read_source(_spec(AutoE2EDataset.ONCE, payload), payload)
    out = pack_autobev_partition(plan, 0, SnapshotMembers(_ObjectS3(objects), "datasets", plan.units))
    manifest = json.loads((Path(out) / "manifest.json").read_text())
    assert manifest["partition_id"] == "once-0000" and manifest["total_samples"] == 1
    assert manifest["sample_kind"] == "autobev_only" and manifest["dataset"] == "once"
    empty = json.loads((pack_autobev_partition(plan, 1, files) / "manifest.json").read_text())
    assert empty["total_samples"] == 0 and empty["shard_names"] == []
