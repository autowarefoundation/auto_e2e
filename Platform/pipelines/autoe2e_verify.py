"""verify_lineage: recompute every digest from a model version down to raw bytes.

Walks the registry version to its LoggedModel, the checkpoint lineage and the
stored checkpoint, the dataset lineage (the MLflow copy must equal the S3
original), the unit rows, the raw snapshot manifest, every label record, the
evaluation records, and then the parent recursively. The first mismatch
raises, so a pipeline change that breaks traceability fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from typing import Any, Mapping, Optional

from Platform.pipelines.autoe2e_lineage import (
    CHECKPOINT_LINEAGE_SCHEMA,
    DATASET_LINEAGE_SCHEMA,
    EVALUATION_SCHEMA,
    parse_record,
    read_units_jsonl,
)
from Platform.pipelines.dataset_publication import canonical_json_bytes, sha256_bytes


def _download(model, relative: str) -> bytes:
    import mlflow

    with tempfile.TemporaryDirectory() as root:
        path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{model.artifact_location.rstrip('/')}/{relative}", dst_path=root,
        )
        with open(path, "rb") as handle:
            return handle.read()


def _list(model, relative: str) -> list[str]:
    """List artifact files under ``relative``; an absent directory is empty.

    Only a missing directory counts as empty. Any other listing error raises,
    so a transient failure can never look like "no evaluations".
    """
    import mlflow
    from mlflow.exceptions import MlflowException

    try:
        entries = mlflow.artifacts.list_artifacts(f"{model.artifact_location.rstrip('/')}/{relative}")
    except MlflowException as error:
        if error.error_code in {"RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"}:
            return []
        raise
    names = []
    for entry in entries:
        if entry.is_dir:
            names.extend(f"{entry.path.rsplit('/', 1)[-1]}/{child}" for child in _list(model, f"{relative}/{entry.path.rsplit('/', 1)[-1]}"))
        else:
            names.append(entry.path.rsplit("/", 1)[-1])
    return names


def _s3_bytes(s3, uri: str, version_id: Optional[str] = None) -> bytes:
    from Platform.pipelines.dataset_publication_tasks import _split_s3_uri

    bucket, key = _split_s3_uri(uri)
    kwargs = {"VersionId": version_id} if version_id else {}
    return s3.get_object(Bucket=bucket, Key=key, **kwargs)["Body"].read()


def _s3_sha256(s3, uri: str) -> str:
    from Platform.pipelines.dataset_publication_tasks import _split_s3_uri

    bucket, key = _split_s3_uri(uri)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    digest = hashlib.sha256()
    for chunk in iter(lambda: body.read(8 * 1024**2), b""):
        digest.update(chunk)
    return digest.hexdigest()


def verify_model(client, s3, model_id: str, *, rehash_checkpoint: bool = True, _seen: Optional[set] = None) -> dict:
    """Verify one LoggedModel's chain; return a summary of what was checked."""
    seen = _seen if _seen is not None else set()
    if model_id in seen:
        raise ValueError(f"lineage cycle at {model_id}")
    seen.add(model_id)
    model = client.get_logged_model(model_id)
    params = model.params
    checkpoint = parse_record(
        _download(model, "lineage/checkpoint.json"),
        expected_sha256=params["lineage_sha256"], schema=CHECKPOINT_LINEAGE_SCHEMA,
    )
    if checkpoint["checkpoint"]["sha256"] != params["checkpoint_sha256"]:
        raise ValueError(f"{model_id}: checkpoint lineage names other weights")
    if int(checkpoint["stage_index"]) != int(params["stage_index"]):
        raise ValueError(f"{model_id}: stage_index param differs from lineage")
    if rehash_checkpoint and _s3_sha256(s3, params["checkpoint_uri"]) != params["checkpoint_sha256"]:
        raise ValueError(f"{model_id}: stored checkpoint bytes changed")
    summary: dict[str, Any] = {"model_id": model_id, "checkpoint_sha256": params["checkpoint_sha256"], "datasets": [], "evaluations": 0}

    dataset_ref = checkpoint["dataset"]
    if dataset_ref.get("status") != "unknown":
        summary["datasets"].append(_verify_dataset(s3, model, dataset_ref))
    for name in _list(model, "evaluations"):
        record_bytes_ = _download(model, f"evaluations/{name}")
        record = parse_record(record_bytes_, expected_sha256=sha256_bytes(record_bytes_), schema=EVALUATION_SCHEMA)
        if record["checkpoint_sha256"] != params["checkpoint_sha256"]:
            raise ValueError(f"{model_id}: evaluation {name} is for other weights")
        if record["checkpoint_lineage_sha256"] != params["lineage_sha256"]:
            raise ValueError(f"{model_id}: evaluation {name} cites another lineage")
        summary["datasets"].append(_verify_dataset(s3, model, record["dataset"]))
        summary["evaluations"] += 1

    parent = checkpoint.get("parent")
    if parent:
        parent_summary = verify_model(client, s3, parent["model_id"], rehash_checkpoint=rehash_checkpoint, _seen=seen)
        parent_model = client.get_logged_model(parent["model_id"])
        if parent_model.params["lineage_sha256"] != parent["lineage_sha256"]:
            raise ValueError(f"{model_id}: parent lineage digest differs from the parent model")
        summary["parent"] = parent_summary
    return summary


def _verify_dataset(s3, model, ref: Mapping[str, Any]) -> dict:
    """Verify the MLflow copy of a dataset lineage against S3, then the chain."""
    dataset_id = ref["dataset_id"]
    copy = _download(model, f"lineage/datasets/{dataset_id}/dataset.json")
    lineage = parse_record(copy, expected_sha256=ref["lineage_sha256"], schema=DATASET_LINEAGE_SCHEMA)
    units_payload = _download(model, f"lineage/datasets/{dataset_id}/units.jsonl")
    return _verify_dataset_chain(s3, dataset_id, lineage, copy, units_payload)


def verify_published_dataset(s3, *, lineage_uri: str, lineage_sha256: str) -> dict:
    """Verify a published dataset's lineage chain from S3 alone."""
    payload = _s3_bytes(s3, lineage_uri)
    lineage = parse_record(payload, expected_sha256=lineage_sha256, schema=DATASET_LINEAGE_SCHEMA)
    return _verify_dataset_chain(
        s3, lineage["dataset_id"], lineage, payload, _s3_bytes(s3, lineage["units"]["uri"]),
    )


def _verify_dataset_chain(s3, dataset_id: str, lineage: Mapping[str, Any], payload: bytes, units_payload: bytes) -> dict:
    if lineage["dataset_id"] != dataset_id:
        raise ValueError(f"dataset lineage is for {lineage['dataset_id']}, not {dataset_id}")
    units_uri = lineage["units"]["uri"]
    if not units_uri.endswith("/lineage/units.jsonl"):
        raise ValueError(f"{dataset_id}: unexpected units location {units_uri}")
    original_uri = units_uri[: -len("units.jsonl")] + "dataset.json"
    if _s3_bytes(s3, original_uri) != payload:
        raise ValueError(f"lineage copy of {dataset_id} differs from S3")
    if sha256_bytes(units_payload) != lineage["units"]["sha256"] or _s3_bytes(s3, units_uri) != units_payload:
        raise ValueError(f"{dataset_id}: units.jsonl differs from its lineage digest")
    units = read_units_jsonl(units_payload)
    if sha256_bytes(canonical_json_bytes(lineage["spec"])) != dataset_id:
        raise ValueError(f"{dataset_id}: spec digest differs")
    manifest_ref = lineage["spec"]["source_manifest"]
    manifest = _s3_bytes(s3, manifest_ref["uri"], manifest_ref["version_id"])
    if sha256_bytes(manifest) != lineage["raw"]["snapshot_manifest_sha256"]:
        raise ValueError(f"{dataset_id}: raw snapshot manifest changed")
    snapshot = json.loads(manifest)
    # autoe2e snapshots list units; the nuPlan snapshot lists archives.
    from Platform.pipelines.autoe2e_adapters import nuplan_archive_digest

    tar_digests = {u["unit_id"]: f"sha256:{u['tar_sha256']}" for u in snapshot.get("units", [])}
    tar_digests.update({a["archive_id"]: nuplan_archive_digest(a) for a in snapshot.get("archives", [])})
    bucket = units_uri.split("/")[2]
    views_key = lineage["views"]["key"]
    views_payload = s3.get_object(Bucket=bucket, Key=views_key)["Body"].read()
    if sha256_bytes(views_payload) != lineage["views"]["sha256"]:
        raise ValueError(f"{dataset_id}: partition views manifest changed")
    labels_checked = 0
    for unit in units:
        if tar_digests and tar_digests.get(unit.unit_id) != unit.raw_tar_digest:
            raise ValueError(f"{dataset_id}: {unit.unit_id} raw tar digest differs from the snapshot")
        if unit.label_id is not None:
            from Platform.pipelines.autoe2e_dataset import AutoE2EDataset
            from Platform.pipelines.autoe2e_labels import label_record_key

            key = label_record_key(AutoE2EDataset(lineage["spec"]["dataset"]), unit.label_id)
            if sha256_bytes(s3.get_object(Bucket=bucket, Key=key)["Body"].read()) != unit.label_record_sha256:
                raise ValueError(f"{dataset_id}: label record of {unit.unit_id} changed")
            labels_checked += 1
    from Platform.pipelines.l2d_snapshot import L2D_OSM_UNIT

    # One region per partition; datasets packed before that name one region.
    pack = lineage["pack"]
    regions = pack.get("osm_regions") or ([pack["osm_region"]] if pack.get("osm_region") else [])
    # L2D map channels come only from the regions, so its lineage must name them.
    if lineage["spec"]["dataset"] == "l2d" and not regions:
        raise ValueError(f"{dataset_id}: L2D lineage names no OSM region")
    episodes_of: dict[str, set | None] = {}
    for region in regions:
        # Each region is derived from the snapshot's OSM unit; it must still be
        # the bytes its packs rendered navigation from, and name that unit.
        region_payload = _s3_bytes(s3, region["uri"], region["version_id"])
        if sha256_bytes(region_payload) != region["sha256"]:
            raise ValueError(f"{dataset_id}: OSM region changed")
        provenance = json.loads(region_payload)["provenance"]
        if tar_digests.get(L2D_OSM_UNIT) != provenance["osm_unit_digest"]:
            raise ValueError(f"{dataset_id}: OSM region was not built from the snapshot's OSM extract")
        if provenance["snapshot_manifest_sha256"] != lineage["raw"]["snapshot_manifest_sha256"]:
            raise ValueError(f"{dataset_id}: OSM region was built for another snapshot")
        if pack.get("osm_regions") and "episodes" not in provenance:
            raise ValueError(f"{dataset_id}: per-partition OSM region names no episodes")
        episodes_of[region["sha256"]] = set(provenance["episodes"]) if "episodes" in provenance else None
    # One region per published partition, no more and no fewer.
    if pack.get("osm_regions") and len(regions) != int(lineage["views"]["partitions"]):
        raise ValueError(f"{dataset_id}: {len(regions)} OSM regions for {lineage['views']['partitions']} partitions")
    if regions:
        # Every packed partition must have rendered from one of these regions,
        # and that region must cover the partition's episodes.
        for partition in json.loads(views_payload)["partitions"]:
            manifest_payload = s3.get_object(Bucket=bucket, Key=f"{partition['prefix']}/manifest.json")["Body"].read()
            if sha256_bytes(manifest_payload) != partition["manifest_sha256"]:
                raise ValueError(f"{dataset_id}: partition {partition['partition_id']} manifest changed")
            manifest = json.loads(manifest_payload)
            if not int(manifest.get("total_samples", 0)):
                continue
            navigation = manifest.get("navigation") or {}
            used = navigation.get("osm_region_sha256")
            if used not in episodes_of:
                raise ValueError(f"{dataset_id}: partition {partition['partition_id']} used another OSM region")
            covered = episodes_of[used]
            packed = {int(e["episode_index"]) for e in navigation.get("episodes", [])}
            if not packed:
                raise ValueError(f"{dataset_id}: partition {partition['partition_id']} has samples but lists no episodes")
            if covered is not None and not packed <= covered:
                raise ValueError(f"{dataset_id}: partition {partition['partition_id']} packed episodes its region lacks")
    return {"dataset_id": dataset_id, "units": len(units), "labels": labels_checked}


def verify_lineage(client, s3, model_version: str, *, rehash_checkpoint: bool = True) -> dict:
    """Verify a registry version ("auto-e2e-driving-policy/N") end to end."""
    name, _, version = model_version.rpartition("/")
    registered = client.get_model_version(name, version)
    if not registered.source.startswith("models:/m-"):
        raise ValueError(f"{model_version} does not point at a LoggedModel")
    model_id = registered.source.removeprefix("models:/")
    params = client.get_logged_model(model_id).params
    for key, value in (registered.tags or {}).items():
        if key in params and params[key] != value:
            raise ValueError(f"{model_version}: version tag {key} differs from the model")
    return verify_model(client, s3, model_id, rehash_checkpoint=rehash_checkpoint)
