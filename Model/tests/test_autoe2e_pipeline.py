"""Task-level helpers of wf_autoe2e and wf_acquire_raw_snapshot."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from Platform.pipelines.autoe2e_adapters import SourcePlan, UnitRef
from Platform.pipelines.autoe2e_dataset import AutoE2EDataset
from Platform.pipelines.autoe2e_snapshot import read_index, snapshot_prefix, unit_tar_key, write_unit_tar
from Platform.pipelines.dataset_publication import sha256_bytes

BUCKET = "datasets"


def _error(code, status):
    return ClientError({"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, "Op")


class S3:
    def __init__(self):
        self.objects = {}
        self.metadata = {}

    def get_object(self, *, Bucket, Key, **_):
        if (Bucket, Key) not in self.objects:
            raise _error("NoSuchKey", 404)
        payload = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(payload), "ContentLength": len(payload)}

    def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _error("404", 404)
        return {"ContentLength": len(self.objects[(Bucket, Key)]), "Metadata": self.metadata.get((Bucket, Key), {})}

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch=None, Metadata=None, **_):
        if IfNoneMatch == "*" and (Bucket, Key) in self.objects:
            raise _error("PreconditionFailed", 412)
        self.objects[(Bucket, Key)] = Body
        self.metadata[(Bucket, Key)] = Metadata or {"sha256": sha256_bytes(Body)}

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.objects[(bucket, key)] = Path(path).read_bytes()
        self.metadata[(bucket, key)] = (ExtraArgs or {}).get("Metadata", {})

    def copy(self, source, bucket, key):
        self.objects[(bucket, key)] = self.objects[(source["Bucket"], source["Key"])]


def _unit_tar(unit, files):
    out = io.BytesIO()
    write_unit_tar(out, unit, ((p, len(b), io.BytesIO(b)) for p, b in sorted(files.items())))
    return out.getvalue()


def test_materialize_rebuilds_the_tree_and_checks_the_tar_digest(tmp_path):
    from Platform.pipelines.autoe2e_tasks import materialize_units

    s3 = S3()
    payload = _unit_tar("scene-a", {"data/train/scene-a/x.jpg": b"jpg", "data/sequence_archives.csv": b"csv"})
    s3.objects[(BUCKET, "k/scene-a.tar")] = payload
    unit = UnitRef("scene-a", "k/scene-a.tar", "sha256:" + sha256_bytes(payload), "k/i.jsonl", "0" * 64)
    materialize_units(s3, BUCKET, [unit], tmp_path / "raw")
    assert (tmp_path / "raw/data/train/scene-a/x.jpg").read_bytes() == b"jpg"
    bad = UnitRef("scene-a", "k/scene-a.tar", "sha256:" + "f" * 64, "k/i.jsonl", "0" * 64)
    with pytest.raises(ValueError, match="differs"):
        materialize_units(s3, BUCKET, [bad], tmp_path / "raw2")


def _plan(units, partitions, labels=True, dataset=AutoE2EDataset.KITSCENES):
    return SourcePlan(
        dataset=dataset, upstream="u", revision="r", split="train", published_name="kitscenes",
        snapshot_manifest_sha256="a" * 64,
        units=[UnitRef(u, f"k/{u}.tar", "sha256:" + "1" * 64, f"k/{u}.jsonl", "2" * 64) for u in units],
        partitions=partitions, labels=labels,
    )


def test_unit_rows_join_labels_and_partition_views():
    from Platform.pipelines.autoe2e_tasks import PartitionView, UnitLabel, unit_rows

    plan = _plan(["a", "b"], [["a"], ["b"]])
    labels = [UnitLabel("a", "3" * 64, "4" * 64, "reused", 2, "s3://x"), UnitLabel("b", "5" * 64, "6" * 64, "created", 0, "s3://y")]
    views = [PartitionView("p-a", "pa", "7" * 64, 10, ["a"]), PartitionView("p-b", "pb", "8" * 64, 0, ["b"])]
    rows = unit_rows(plan, labels, views)
    assert [(r.unit_id, r.partition_id, r.sample_count, r.label_origin) for r in rows] == [
        ("a", "p-a", 10, "reused"), ("b", "p-b", 0, "created")]


def test_unit_rows_reject_missing_labels_or_partitions():
    from Platform.pipelines.autoe2e_tasks import PartitionView, UnitLabel, unit_rows

    plan = _plan(["a", "b"], [["a"], ["b"]])
    views = [PartitionView("p-a", "pa", "7" * 64, 10, ["a"]), PartitionView("p-b", "pb", "8" * 64, 0, ["b"])]
    with pytest.raises(ValueError, match="missing its label"):
        unit_rows(plan, [UnitLabel("a", "3" * 64, "4" * 64, "reused", 2, "s3://x")], views)
    with pytest.raises(ValueError, match="no partition"):
        unit_rows(_plan(["a"], [["a"]], labels=False), [], [])


def test_nuplan_archives_are_shared_units():
    from Platform.pipelines.autoe2e_lineage import units_jsonl
    from Platform.pipelines.autoe2e_tasks import PartitionView, unit_rows

    plan = _plan(["maps", "db-1"], [["maps", "db-1"]], labels=False, dataset=AutoE2EDataset.NUPLAN)
    plan.units = [UnitRef(u, f"k/{u}.zip", "crc64nvme:8BLx0oBv5Jo=", "", "") for u in ("maps", "db-1")]
    # read_nuplan declares this; the hand-built plan must too.
    plan.shared_units = True
    rows = unit_rows(plan, [], [PartitionView("g0", "p", "7" * 64, 10, ["maps", "db-1"])])
    assert {r.partition_id for r in rows} == {"*"}
    assert {r.raw_index_sha256 for r in rows} == {None}
    assert units_jsonl(rows)


def test_nuplan_archive_digest_prefers_sha256_then_crc64nvme():
    from Platform.pipelines.autoe2e_adapters import nuplan_archive_digest

    assert nuplan_archive_digest({"sha256": "a" * 64, "checksum_crc64nvme": "x"}) == "sha256:" + "a" * 64
    assert nuplan_archive_digest({"sha256": "", "checksum_crc64nvme": "8BLx0oBv5Jo="}) == "crc64nvme:8BLx0oBv5Jo="
    with pytest.raises(ValueError, match="pins no digest"):
        nuplan_archive_digest({"archive_id": "db-x", "sha256": "", "checksum_crc64nvme": ""})


def test_acquired_unit_tar_is_deterministic_and_indexed(tmp_path):
    from Platform.pipelines.autoe2e_acquire import tar_tree, upload_unit

    root = tmp_path / "tree"
    (root / "data/train/scene-a").mkdir(parents=True)
    (root / "data/train/scene-a/x.jpg").write_bytes(b"jpg" * 100)
    (root / "data/sequence_archives.csv").write_bytes(b"csv")
    members = tar_tree(root, "scene-a", tmp_path / "a.tar")
    assert tar_tree(root, "scene-a", tmp_path / "b.tar") == members
    assert (tmp_path / "a.tar").read_bytes() == (tmp_path / "b.tar").read_bytes()
    s3 = S3()
    prefix = snapshot_prefix(AutoE2EDataset.KITSCENES, "rev", "snap")
    entry = upload_unit(s3, bucket=BUCKET, prefix=prefix, unit_id="scene-a", tar_path=tmp_path / "a.tar", members=members)
    assert entry.tar_key == unit_tar_key(prefix, "scene-a")
    assert entry.tar_sha256 == hashlib.sha256((tmp_path / "a.tar").read_bytes()).hexdigest()
    assert read_index(s3.objects[(BUCKET, entry.index_key)]) == sorted(members, key=lambda m: m.name)
    again = upload_unit(s3, bucket=BUCKET, prefix=prefix, unit_id="scene-a", tar_path=tmp_path / "a.tar", members=members)
    assert again == entry


def test_reacquiring_a_unit_with_other_bytes_fails(tmp_path):
    from Platform.pipelines.autoe2e_acquire import tar_tree, upload_unit

    root = tmp_path / "tree"
    root.mkdir()
    (root / "x").write_bytes(b"one")
    members = tar_tree(root, "u", tmp_path / "a.tar")
    s3 = S3()
    prefix = snapshot_prefix(AutoE2EDataset.KITSCENES, "rev", "snap")
    upload_unit(s3, bucket=BUCKET, prefix=prefix, unit_id="u", tar_path=tmp_path / "a.tar", members=members)
    (root / "x").write_bytes(b"two")
    other = tar_tree(root, "u", tmp_path / "b.tar")
    with pytest.raises(RuntimeError, match="other bytes"):
        upload_unit(s3, bucket=BUCKET, prefix=prefix, unit_id="u", tar_path=tmp_path / "b.tar", members=other)


def test_checkpoints_are_stored_content_addressed_and_verified():
    from Platform.pipelines.autoe2e import store_checkpoint

    s3 = S3()
    s3.objects[("ray", "run/checkpoint.pt")] = b"weights"
    sha = sha256_bytes(b"weights")
    uri = store_checkpoint(s3, source_uri="s3://ray/run/checkpoint.pt", sha256=sha, bucket="ckpt")
    assert uri == f"s3://ckpt/autoe2e/checkpoints/{sha}.pt"
    with pytest.raises(RuntimeError, match="differs"):
        store_checkpoint(s3, source_uri="s3://ray/run/checkpoint.pt", sha256="f" * 64, bucket="ckpt")


def test_numeric_leaves_flatten_only_finite_numbers():
    from Platform.pipelines.autoe2e import numeric_leaves

    assert numeric_leaves("", {"trajectory": {"ade_3s_m": 0.7, "ade_6p4s_m": None, "n": 3},
                               "route": {"available": False}, "x": float("nan")}) == {
        "trajectory.ade_3s_m": 0.7, "trajectory.n": 3.0}


class ListingS3(S3):
    def get_paginator(self, name):
        s3 = self

        class Paginator:
            def paginate(self, *, Bucket, Prefix, Delimiter=None):
                keys = sorted(k for b, k in s3.objects if b == Bucket and k.startswith(Prefix))
                yield {"Contents": [
                    {"Key": k, "Size": len(s3.objects[(Bucket, k)]),
                     "ETag": f'"{sha256_bytes(s3.objects[(Bucket, k)])[:32]}"'}
                    for k in keys
                ]}

        return Paginator()


def _july_labels(s3, *, teacher_model="nvidia/Cosmos3-Nano", sample_unit="scene-a", count=1):
    import json

    prefix = "a2/run/abc"
    meta = {"source_revision": "rev", "split": "train", "teacher": "openai_compatible",
            "prompt_version": "p", "label_policy_version": "v2", "num_records": count}
    s3.objects[("artifacts", f"{prefix}/meta.json")] = json.dumps(meta).encode()
    rows = "\n".join(json.dumps({"sample_id": f"kitscenes-v1-{sample_unit}-f{i:06d}",
                                 "teacher_model": teacher_model, "prompt_version": "p"})
                     for i in range(count))
    s3.objects[("artifacts", f"{prefix}/dataset=x/records.jsonl")] = rows.encode()
    return f"s3://artifacts/{prefix}"


def _adopt(monkeypatch, s3, uri, expected=1):
    from Platform.pipelines import autoe2e_acquire
    from Platform.pipelines.autoe2e_acquire import UnitSource, adopt_unit_labels
    from Platform.pipelines.autoe2e_dataset import ReasoningSpec

    monkeypatch.setattr(autoe2e_acquire, "_s3", lambda: s3)
    return adopt_unit_labels.task_function(
        dataset=AutoE2EDataset.KITSCENES, revision="rev",
        reasoning=ReasoningSpec("openai_compatible", "nvidia/Cosmos3-Nano", "p", 10),
        source=UnitSource("scene-a", "", uri, expected), datasets_bucket=BUCKET, adopted_from="snap",
    )


def test_adopted_labels_are_committed_once(monkeypatch):
    s3 = ListingS3()
    uri = _july_labels(s3)
    first = _adopt(monkeypatch, s3, uri)
    assert _adopt(monkeypatch, s3, uri) == first
    assert any(k.startswith("kitscenes/_stages/labels/") for _, k in s3.objects)


@pytest.mark.parametrize(
    ("kwargs", "expected", "message"),
    [
        ({"teacher_model": "nvidia/Cosmos3-Super"}, 1, "teacher_model"),
        ({"sample_unit": "scene-b"}, 1, "another unit"),
        ({}, 2, "manifest"),
    ],
)
def test_adoption_rejects_labels_that_do_not_match(monkeypatch, kwargs, expected, message):
    s3 = ListingS3()
    uri = _july_labels(s3, **kwargs)
    with pytest.raises(ValueError, match=message):
        _adopt(monkeypatch, s3, uri, expected)
    assert not any(k.startswith("kitscenes/") for _, k in s3.objects)


def test_tar_tree_keeps_one_file_open_at_a_time(tmp_path):
    import resource

    from Platform.pipelines.autoe2e_acquire import tar_tree

    root = tmp_path / "tree"
    root.mkdir()
    for i in range(300):
        (root / f"f{i:04d}.bin").write_bytes(b"x")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, hard))
    try:
        members = tar_tree(root, "u", tmp_path / "u.tar")
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert len(members) == 300


def _source(metrics='{"validation_ade_6p4s_m": 2.5, "validation_fde_6p4s_m": 6.0, "route": {"x": 1}}'):
    from Platform.pipelines.autoe2e import OverlaySource, RegisteredModel

    model = RegisteredModel("best", "m-abc", "71", "s3://ckpt/autoe2e/checkpoints/x.pt", "c" * 64, "l" * 64)
    return OverlaySource(model, "r" * 32, metrics, "res_net_50", 0.1, "exec", "v3.7")


def test_overlay_metadata_names_the_registered_version_and_validation_metrics():
    from Platform.pipelines.autoe2e import overlay_metadata

    metadata = overlay_metadata(_source(), dataset_source="KIT-MRT/KITScenes-Multimodal")
    assert metadata["registered_model_name"] == "auto-e2e-driving-policy"
    assert metadata["model_version"] == "71"
    assert metadata["artifact_uri"] == "models:/m-abc"
    assert (metadata["eval_ade"], metadata["eval_fde"]) == (2.5, 6.0)
    assert (metadata["model_name"], metadata["dataset_version_source"], metadata["train_execution_id"]) == (
        "res_net_50", "v3.7", "exec")
    with pytest.raises(ValueError, match="validation ADE"):
        overlay_metadata(_source('{"bev_iou": 0.3}'), dataset_source="d")


def test_overlays_need_a_console_publication(monkeypatch):
    from Platform.pipelines import autoe2e_store
    from Platform.pipelines.autoe2e import RuntimeOptions, console_manifest_sha256
    from Platform.pipelines.autoe2e_store import DatasetHandle

    handle = DatasetHandle("d" * 64, "kitscenes", "v3.7", True, "exec", lineage_key="k", lineage_sha256="s")
    runtime = RuntimeOptions(datasets_bucket=BUCKET)
    for console, expected in (({"manifest_sha256": "m" * 64}, "m" * 64), (None, "")):
        monkeypatch.setattr(autoe2e_store, "read_dataset_lineage",
                            lambda s3, *, bucket, key, sha256, c=console: {"console": c})
        assert console_manifest_sha256(S3(), handle, runtime) == expected


def test_overlay_chunks_balance_samples_and_cover_every_partition_once():
    from Platform.pipelines.autoe2e import overlay_chunks

    counts = {f"s3://d/p{i:03d}": c for i, c in enumerate([500, 10, 10, 10, 480, 20, 30, 5])}
    chunks = overlay_chunks(counts, 2)
    assert sorted(u for c in chunks for u in c) == sorted(counts)
    loads = sorted(sum(counts[u] for u in c) for c in chunks)
    assert loads == [530, 535]
    for chunk in chunks:
        assert chunk == sorted(chunk)  # dataset order inside a chunk
    assert [len(c) for c in overlay_chunks(dict(list(counts.items())[:2]), 16)] == [1, 1]
    with pytest.raises(ValueError, match="at least 1"):
        overlay_chunks(counts, 0)
    with pytest.raises(ValueError, match="no partitions"):
        overlay_chunks({}, 4)


def _parent_logged(**params):
    from types import SimpleNamespace

    metric = lambda key, value, ts: SimpleNamespace(key=key, value=value, timestamp=ts)  # noqa: E731
    return SimpleNamespace(
        source_run_id="r" * 32,
        params={"dataset.dataset": "kitscenes", "dataset_id": "d" * 64, "training.backbone": "res_net_50",
                "training.val_fraction": "0.1", "provenance.flyte_execution_id": "train-exec", **params},
        metrics=[
            metric("eval/internal_val/kitscenes-v3.7/validation_ade_6p4s_m", 9.0, 1),
            metric("eval/internal_val/kitscenes-v3.7/validation_ade_6p4s_m", 3.0, 2),
            metric("eval/internal_val/kitscenes-v3.7/validation_fde_6p4s_m", 7.0, 2),
            metric("eval/kitscenes_benchmark/kitscenes-val-v3.6/ade_3s_m", 1.0, 3),
        ],
    )


def test_overlay_source_reads_a_parent_from_its_own_logged_model():
    from types import SimpleNamespace

    from Platform.pipelines.autoe2e import RegisteredModel, RuntimeOptions, TrainedCheckpoints, TrainingSpec, overlay_source
    from Platform.pipelines.autoe2e_store import DatasetHandle

    s3 = S3()
    s3.objects[(BUCKET, f"kitscenes/by-id/{'d' * 64}.json")] = b'{"version": "v3.7"}'
    parent = RegisteredModel("parent", "m-p", "70", "s3://c/p.pt", "c" * 64, "l" * 64)
    logged = _parent_logged()
    client = SimpleNamespace(get_logged_model=lambda model_id: logged)
    # The epochs=0 run's own spec and dataset must not leak into the attribution.
    args = (TrainedCheckpoints(trained=False), TrainingSpec(backbone="other", val_fraction=0.3),
            DatasetHandle("e" * 64, "kitscenes", "v9.9", True, "x"), RuntimeOptions(datasets_bucket=BUCKET))
    source = overlay_source(client, s3, [parent], *args)
    assert (source.model, source.run_id, source.backbone, source.val_fraction) == (parent, "r" * 32, "res_net_50", 0.1)
    assert (source.train_execution_id, source.train_dataset_version) == ("train-exec", "v3.7")
    assert json.loads(source.metrics_json) == {"validation_ade_6p4s_m": 3.0, "validation_fde_6p4s_m": 7.0}
    for change in ({"source_run_id": None}, {"params": {**logged.params, "dataset.dataset": "nuplan"}},
                   {"metrics": logged.metrics[:2]}):
        client = SimpleNamespace(get_logged_model=lambda model_id, c=change: SimpleNamespace(**{**vars(logged), **c}))
        assert overlay_source(client, s3, [parent], *args) is None


def test_overlay_source_takes_a_training_best_but_not_a_one_gpu_smoke():
    from Platform.pipelines.autoe2e import (
        GpuProfile, RegisteredModel, RuntimeOptions, TrainedCheckpoints, TrainingSpec, overlay_source)
    from Platform.pipelines.autoe2e_store import DatasetHandle

    best = RegisteredModel("best", "m-b", "71", "s3://c/b.pt", "b" * 64, "l" * 64)
    trained = TrainedCheckpoints(trained=True, run_id="t" * 32, best_metrics_json="{}")
    handle = DatasetHandle("e" * 64, "kitscenes", "v3.7", True, "x")
    source = overlay_source(None, None, [best], trained, TrainingSpec(gpu_profile=GpuProfile.RAY_8), handle,
                            RuntimeOptions())
    assert (source.model, source.run_id, source.train_dataset_version) == (best, "t" * 32, "v3.7")
    assert overlay_source(None, None, [best], trained, TrainingSpec(gpu_profile=GpuProfile.RAY_1), handle,
                          RuntimeOptions()) is None


def test_overlays_skip_partitions_without_samples():
    from Platform.pipelines.autoe2e import RuntimeOptions, partition_sample_counts
    from Platform.pipelines.autoe2e_store import DatasetHandle

    s3 = S3()
    views = json.dumps({"partitions": [{"prefix": "kitscenes/v3.7/partitions/p0", "sample_count": 3},
                                       {"prefix": "kitscenes/v3.7/partitions/p1", "sample_count": 0}]}).encode()
    s3.objects[(BUCKET, "kitscenes/v3.7/partitions/manifest.json")] = views
    uris = [f"s3://{BUCKET}/kitscenes/v3.7/partitions/p0", f"s3://{BUCKET}/kitscenes/v3.7/partitions/p1"]
    handle = DatasetHandle("e" * 64, "kitscenes", "v3.7", True, "x", views_sha256=sha256_bytes(views),
                           partition_uris=uris)
    assert partition_sample_counts(s3, handle, RuntimeOptions(datasets_bucket=BUCKET)) == {uris[0]: 3}
