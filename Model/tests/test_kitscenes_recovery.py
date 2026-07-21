"""Recovery-manifest contracts for reusing KITScenes artifacts."""

from __future__ import annotations

import copy

import pytest

from Platform.pipelines.kitscenes_recovery import (
    RECOVERY_MANIFEST_SCHEMA,
    artifact_set_sha256,
    validate_recovery_manifest,
)


DATASET = "KIT-MRT/KITScenes-Multimodal"
REVISION = "6fde0034446669e2ed7235e4c7fe323cd23d599d"
SCENES = [
    "008fba36-5e82-e02b-8edf-a55f5271758d",
    "00efe646-1e8f-cadc-d642-e676781187ef",
]


def _entries():
    return [
        {
            "index": index,
            "scene_id": scene_id,
            "raw_uri": f"s3://artifacts/raw/{scene_id}",
            "label_uri": f"s3://artifacts/labels/{scene_id}",
            "expected_label_count": index,
        }
        for index, scene_id in enumerate(SCENES)
    ]


def _manifest():
    entries = _entries()
    return {
        "schema_version": RECOVERY_MANIFEST_SCHEMA,
        "dataset": DATASET,
        "source_revision": REVISION,
        "split": "train",
        "artifact_set_sha256": artifact_set_sha256(entries),
        "entries": entries,
    }


def _validate(manifest, expected_digest=None, **kwargs):
    return validate_recovery_manifest(
        manifest,
        expected_artifact_set_sha256=(
            expected_digest or manifest["artifact_set_sha256"]
        ),
        expected_dataset=DATASET,
        expected_source_revision=REVISION,
        expected_scene_ids=SCENES,
        **kwargs,
    )


def test_manifest_validates_ordered_atomic_scene_entries():
    entries = _validate(_manifest())

    assert [entry["index"] for entry in entries] == [0, 1]
    assert [entry["expected_label_count"] for entry in entries] == [0, 1]


def test_artifact_digest_excludes_observed_label_counts():
    before = _entries()
    after = copy.deepcopy(before)
    after[0]["expected_label_count"] = 99

    assert artifact_set_sha256(before) == artifact_set_sha256(after)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: manifest["entries"][0].update(index=1),
            "contiguous and ordered",
        ),
        (
            lambda manifest: manifest["entries"][1].update(
                scene_id=manifest["entries"][0]["scene_id"]
            ),
            "duplicate recovery scene_id",
        ),
        (
            lambda manifest: manifest["entries"][0].update(
                raw_uri="https://example.test/raw"
            ),
            "must be an S3 directory URI",
        ),
        (
            lambda manifest: manifest["entries"][0].update(
                expected_label_count=-1
            ),
            "negative label count",
        ),
    ],
)
def test_manifest_rejects_invalid_entries(mutation, message):
    manifest = _manifest()
    mutation(manifest)

    with pytest.raises(ValueError, match=message):
        _validate(manifest)


def test_manifest_rejects_artifact_tuple_tampering():
    manifest = _manifest()
    manifest["entries"][0]["raw_uri"] += "-different"

    with pytest.raises(ValueError, match="declared=.*computed"):
        _validate(manifest)


def test_manifest_rejects_launch_digest_mismatch():
    manifest = _manifest()

    with pytest.raises(ValueError, match="differs from the launch input"):
        _validate(manifest, expected_digest="f" * 64)


def test_manifest_rejects_audited_label_and_empty_scene_drift():
    manifest = _manifest()

    with pytest.raises(ValueError, match="label total"):
        _validate(manifest, expected_label_count=2)
    with pytest.raises(ValueError, match="empty-scene count"):
        _validate(manifest, expected_empty_scene_count=2)

    entries = _validate(
        manifest,
        expected_label_count=1,
        expected_empty_scene_count=1,
    )
    assert sum(entry["expected_label_count"] for entry in entries) == 1


def test_manifest_rejects_official_scene_order_drift():
    manifest = _manifest()
    manifest["entries"].reverse()
    for index, entry in enumerate(manifest["entries"]):
        entry["index"] = index
    manifest["artifact_set_sha256"] = artifact_set_sha256(
        manifest["entries"]
    )

    with pytest.raises(ValueError, match="official inventory"):
        _validate(manifest)
