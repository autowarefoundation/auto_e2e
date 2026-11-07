"""wf_autoe2e: one pipeline from a raw snapshot to traceable MLflow models.

1. plan and resolve the training dataset, the BEV datasets mixed into it, and
   each evaluation dataset;
2. build and publish any dataset that is not published yet;
3. resolve the parent model, if any (stage 2 starts from a stage 1 AutoBEV);
4. train one stage with Ray Train while a second task validates every
   checkpoint as it lands (skipped when steps=0);
5. register one MLflow LoggedModel and version for the best and the final
   checkpoint;
6. evaluate every suite and attach the metrics to the LoggedModels.
"""

from __future__ import annotations

import enum
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from flytekit import Resources, dynamic, task, workflow

from Platform.pipelines.autoe2e_adapters import SourcePlan
from Platform.pipelines.autoe2e_dataset import DatasetSpec, canonical_dataset_spec
from Platform.pipelines.autoe2e_env import PIPELINE_ENV
from Platform.pipelines.autoe2e_store import DatasetHandle
from Platform.pipelines.autoe2e_tasks import build_dataset, plan_dataset, resolve_dataset
from Platform.pipelines.autoe2e_training import (
    MLFLOW_URI,
    QUEUES,
    RAY_STORAGE_PATH,
    TRAIN_TASKS,
    TRAINING_IMAGE,
    VALIDATION_POD,
    TrainedCheckpoints,
    TrainedRun,
    TrainingData,
    TrainingSpec,
    select_checkpoints,
    validate_checkpoints,
)

EXPERIMENT_ENV = {
    "AUTO_E2E_MLFLOW_EXPERIMENT": "auto-e2e",
    "MLFLOW_TRACKING_URI": MLFLOW_URI,
    # Read at registration and baked into the task, so lineage names the code.
    "AUTO_E2E_GIT_COMMIT": os.environ.get("AUTO_E2E_GIT_COMMIT", "unknown"),
}


class EvalSuite(enum.Enum):
    # BEV IoU (and in stage 2 ADE/FDE) on the full held-out Pittsburgh split of
    # the training publication, thresholds calibrated on its calibration split.
    GEO_VAL = "geo_val"


@dataclass
class EvalTarget:
    suite: EvalSuite
    dataset: Optional[DatasetSpec] = None


@dataclass
class EvaluationSpec:
    targets: List[EvalTarget] = field(default_factory=list)


@dataclass
class ModelRef:
    """Exactly one field is set: a registry version ("name/version") or an S3 checkpoint."""

    model_version: str = ""
    checkpoint_uri: str = ""


@dataclass
class RuntimeOptions:
    datasets_bucket: str = ""
    checkpoints_bucket: str = ""
    dynamo_table: str = "auto-e2e-console"
    label_concurrency: int = 5
    pack_concurrency: int = 60
    tracking_url: str = ""


@dataclass
class ParentHandle:
    present: bool = False
    checkpoint_uri: str = ""
    checkpoint_sha256: str = ""
    model_id: str = ""
    model_version: str = ""
    lineage_sha256: str = ""
    stage_index: int = 0
    step: int = 0


@dataclass
class RegisteredModel:
    role: str
    model_id: str
    version: str
    checkpoint_uri: str
    checkpoint_sha256: str
    lineage_sha256: str


@dataclass
class PipelineResult:
    dataset_id: str
    dataset_lineage_sha256: str
    models: List[RegisteredModel] = field(default_factory=list)


def _client():
    from mlflow import MlflowClient

    return MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI))


def _s3():
    import boto3

    return boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))


def _execution_id() -> str:
    from flytekit import current_context

    try:
        return current_context().execution_id.name or "local"
    except Exception:
        return "local"


# --- datasets -----------------------------------------------------------------


@dynamic(container_image=TRAINING_IMAGE, environment=PIPELINE_ENV)
def ensure_dataset(
    spec: DatasetSpec,
    plan: SourcePlan,
    handle: DatasetHandle,
    runtime: RuntimeOptions,
) -> DatasetHandle:
    """Return a published dataset: the resolved one, or a freshly built one."""
    if handle.published:
        return handle
    return build_dataset(
        spec=spec, plan=plan, handle=handle,
        datasets_bucket=runtime.datasets_bucket, dynamo_table=runtime.dynamo_table,
        label_concurrency=runtime.label_concurrency, pack_concurrency=runtime.pack_concurrency,
    )


@dynamic(container_image=TRAINING_IMAGE, environment=PIPELINE_ENV)
def ensure_datasets(specs: List[DatasetSpec], runtime: RuntimeOptions) -> List[DatasetHandle]:
    """Resolve or build every dataset of a list (the BEV datasets mixed into training)."""
    handles = []
    for spec in specs:
        plan = plan_dataset(spec=spec, datasets_bucket=runtime.datasets_bucket)
        resolved = resolve_dataset(spec=spec, plan=plan, datasets_bucket=runtime.datasets_bucket)
        handles.append(ensure_dataset(spec=spec, plan=plan, handle=resolved, runtime=runtime))
    return handles


# --- parent ---------------------------------------------------------------------


def checkpoint_sha256(s3, uri: str) -> str:
    import hashlib

    from Platform.pipelines.dataset_publication_tasks import _split_s3_uri

    bucket, key = _split_s3_uri(uri)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    digest = hashlib.sha256()
    for chunk in iter(lambda: body.read(8 * 1024**2), b""):
        digest.update(chunk)
    return digest.hexdigest()


def stored_step(uri: str) -> int:
    """Training step recorded inside a checkpoint (never a caller's claim)."""
    import tempfile

    import torch

    from Platform.pipelines.dataset_publication_tasks import _split_s3_uri

    bucket, key = _split_s3_uri(uri)
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        _s3().download_file(bucket, key, handle.name)
        payload = torch.load(handle.name, map_location="cpu", weights_only=False)
    step = payload.get("metadata", {}).get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"checkpoint {uri} records no valid step")
    return step


@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(cpu="2", mem="16Gi", ephemeral_storage="20Gi"),
    limits=Resources(cpu="2", mem="16Gi", ephemeral_storage="20Gi"),
    environment=EXPERIMENT_ENV,
    retries=2,
)
def resolve_parent(parent: Optional[ModelRef]) -> ParentHandle:
    """Pin the parent checkpoint by digest, from a registry version or a bare URI."""
    from Platform.pipelines.autoe2e_mlflow import REGISTERED_MODEL

    if parent is None or (not parent.model_version and not parent.checkpoint_uri):
        return ParentHandle()
    if parent.model_version and parent.checkpoint_uri:
        raise ValueError("ModelRef sets both model_version and checkpoint_uri")
    s3 = _s3()
    if parent.checkpoint_uri:
        return ParentHandle(
            present=True, checkpoint_uri=parent.checkpoint_uri,
            checkpoint_sha256=checkpoint_sha256(s3, parent.checkpoint_uri),
            step=stored_step(parent.checkpoint_uri),
        )
    name, _, version = parent.model_version.rpartition("/")
    if name != REGISTERED_MODEL or not version:
        raise ValueError(f"model_version must be {REGISTERED_MODEL}/<version>")
    client = _client()
    registered = client.get_model_version(name, version)
    if not registered.source.startswith("models:/m-"):
        raise ValueError(f"version {version} is not a wf_autoe2e LoggedModel ({registered.source})")
    model = client.get_logged_model(registered.source.removeprefix("models:/"))
    params = model.params
    actual = checkpoint_sha256(s3, params["checkpoint_uri"])
    if actual != params["checkpoint_sha256"]:
        raise ValueError(f"parent checkpoint bytes differ from version {version}")
    return ParentHandle(
        present=True, checkpoint_uri=params["checkpoint_uri"], checkpoint_sha256=actual,
        model_id=model.model_id, model_version=parent.model_version,
        lineage_sha256=params["lineage_sha256"], stage_index=int(params["stage_index"]),
        step=int(params["checkpoint_step"]),
    )


# --- training -------------------------------------------------------------------


@task(container_image=TRAINING_IMAGE, requests=Resources(cpu="1", mem="1Gi"))
def parent_as_checkpoints(parent: ParentHandle) -> TrainedCheckpoints:
    """steps=0: evaluate the parent as both best and final, without training."""
    if not parent.present:
        raise ValueError("steps=0 evaluates a parent model; set parent")
    return TrainedCheckpoints(
        trained=False, best_uri=parent.checkpoint_uri, best_sha256=parent.checkpoint_sha256,
        best_step=parent.step, final_uri=parent.checkpoint_uri,
        final_sha256=parent.checkpoint_sha256, final_step=parent.step,
    )


def training_data(dataset: DatasetHandle, mix: List[DatasetHandle], early_mix: List[DatasetHandle]) -> TrainingData:
    return TrainingData(
        primary=list(dataset.partition_uris),
        mix=[uri for handle in mix for uri in handle.partition_uris],
        early=[uri for handle in early_mix for uri in handle.partition_uris],
        dataset_ids=[dataset.dataset_id, *(h.dataset_id for h in mix), *(h.dataset_id for h in early_mix)],
    )


@dynamic(container_image=TRAINING_IMAGE, environment=PIPELINE_ENV)
def train(
    spec: TrainingSpec,
    dataset: DatasetHandle,
    mix: List[DatasetHandle],
    early_mix: List[DatasetHandle],
    parent: ParentHandle,
) -> TrainedCheckpoints:
    """Train on the profile's queue while a validation task follows the checkpoints."""
    if spec.steps == 0:
        return parent_as_checkpoints(parent=parent)
    name = f"{_execution_id()}-stage{spec.stage}"
    run = TrainedRun(trained=True, run_uri=f"{RAY_STORAGE_PATH.rstrip('/')}/{name}", run_name=name)
    data = training_data(dataset, mix, early_mix)
    trained = TRAIN_TASKS[spec.gpu_profile](spec=spec, data=data, run=run, parent_uri=parent.checkpoint_uri)
    validated = validate_checkpoints(spec=spec, data=data, run=run, parent_uri=parent.checkpoint_uri)
    selected = select_checkpoints(spec=spec, run=run, validated=validated)
    trained >> selected
    return selected


# --- registration ---------------------------------------------------------------


def store_checkpoint(s3, *, source_uri: str, sha256: str, bucket: str) -> str:
    """Copy a checkpoint to a content-addressed, write-once key and verify it."""
    from Platform.pipelines.dataset_publication_tasks import _split_s3_uri

    key = f"autoe2e/checkpoints/{sha256}.pt"
    target = f"s3://{bucket}/{key}"
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except Exception as error:
        if getattr(error, "response", {}).get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
        source_bucket, source_key = _split_s3_uri(source_uri)
        s3.copy({"Bucket": source_bucket, "Key": source_key}, bucket, key)
    if checkpoint_sha256(s3, target) != sha256:
        raise RuntimeError(f"stored checkpoint differs from {sha256}: {target}")
    return target


@task(container_image=TRAINING_IMAGE, requests=Resources(cpu="2", mem="8Gi"), environment=EXPERIMENT_ENV, retries=2)
def register_models(
    dataset_spec: DatasetSpec,
    dataset: DatasetHandle,
    mix: List[DatasetHandle],
    training: TrainingSpec,
    parent: ParentHandle,
    trained: TrainedCheckpoints,
    runtime: RuntimeOptions,
) -> List[RegisteredModel]:
    """One LoggedModel and version per distinct checkpoint, with full lineage."""
    from Platform.pipelines import autoe2e_mlflow as am
    from Platform.pipelines.autoe2e_lineage import checkpoint_lineage, record_bytes, record_sha256
    from Platform.pipelines.autoe2e_store import read_dataset_lineage

    client = _client()
    s3 = _s3()
    experiment = am.experiment_id(client)
    if parent.present and not parent.model_id:
        # A bare checkpoint is imported first, so every parent has a model page
        # and a lineage digest for its children to point at.
        imported = _register_imported(client, s3, experiment, parent, runtime)
        parent = ParentHandle(
            present=True, checkpoint_uri=imported.checkpoint_uri,
            checkpoint_sha256=imported.checkpoint_sha256, model_id=imported.model_id,
            model_version=f"{am.REGISTERED_MODEL}/{imported.version}",
            lineage_sha256=imported.lineage_sha256, stage_index=0, step=parent.step,
        )
    if not trained.trained:
        return [RegisteredModel("parent", parent.model_id, parent.model_version.rpartition("/")[2],
                                parent.checkpoint_uri, parent.checkpoint_sha256, parent.lineage_sha256)]
    lineage = read_dataset_lineage(
        s3, bucket=runtime.datasets_bucket, key=dataset.lineage_key, sha256=dataset.lineage_sha256,
    )
    dataset_json = record_bytes(lineage)
    units_bucket_key = lineage["units"]["uri"].removeprefix(f"s3://{runtime.datasets_bucket}/")
    units_payload = s3.get_object(Bucket=runtime.datasets_bucket, Key=units_bucket_key)["Body"].read()
    roles = [("best", trained.best_uri, trained.best_sha256, trained.best_step, trained.best_metrics_json)]
    if trained.final_sha256 != trained.best_sha256:
        roles.append(("final", trained.final_uri, trained.final_sha256, trained.final_step, trained.final_metrics_json))
    mixed = []
    for handle in mix:
        mix_lineage = read_dataset_lineage(
            s3, bucket=runtime.datasets_bucket, key=handle.lineage_key, sha256=handle.lineage_sha256)
        mix_units = mix_lineage["units"]["uri"].removeprefix(f"s3://{runtime.datasets_bucket}/")
        mixed.append((handle.dataset_id, record_bytes(mix_lineage),
                      s3.get_object(Bucket=runtime.datasets_bucket, Key=mix_units)["Body"].read()))
    results = []
    for role, uri, sha, step, metrics_json in roles:
        stored = store_checkpoint(s3, source_uri=uri, sha256=sha, bucket=runtime.checkpoints_bucket)
        record = checkpoint_lineage(
            checkpoint={"sha256": sha, "uri": stored, "role": role, "step": step},
            parent=None if not parent.present else {
                "model_id": parent.model_id, "model_version": parent.model_version,
                "checkpoint_sha256": parent.checkpoint_sha256,
                "lineage_sha256": parent.lineage_sha256, "stage_index": parent.stage_index,
            },
            dataset={"dataset_id": dataset.dataset_id, "lineage_uri": f"s3://{runtime.datasets_bucket}/{dataset.lineage_key}",
                     "lineage_sha256": dataset.lineage_sha256},
            split={"policy": "pittsburgh_held_out"},
            training={"spec": _spec_dict(training), "flyte_execution_id": _execution_id(),
                      "mixed_dataset_ids": [handle.dataset_id for handle in mix],
                      "mlflow_run_id": trained.run_id, "image_digest": TRAINING_IMAGE,
                      "git_commit": os.environ.get("AUTO_E2E_GIT_COMMIT", "unknown")},
            selection={"metrics": json.loads(metrics_json)},
        )
        lineage_sha = record_sha256(record)
        params = am.model_params(
            identity={
                "checkpoint_sha256": sha, "checkpoint_role": role, "checkpoint_uri": stored,
                "checkpoint_step": step, "lineage_sha256": lineage_sha,
                "dataset_id": dataset.dataset_id, "dataset_lineage_sha256": dataset.lineage_sha256,
                "parent_model_id": parent.model_id or None, "stage_index": record["stage_index"],
            },
            dataset_spec=canonical_dataset_spec(dataset_spec),
            training_spec=_spec_dict(training),
            provenance={"flyte_execution_id": _execution_id(), "training_image": TRAINING_IMAGE,
                        "git_commit": os.environ.get("AUTO_E2E_GIT_COMMIT", "unknown")},
        )
        model = am.ensure_logged_model(
            client, experiment=experiment, source_run_id=trained.run_id, name=role, params=params,
            description_text=am.description(
                checkpoint={"role": role, "sha256": sha}, lineage=record,
                tracking_url=runtime.tracking_url, experiment=experiment,
            ),
            artifacts=am.lineage_artifacts(
                checkpoint_lineage=record_bytes(record),
                datasets=[(dataset.dataset_id, dataset_json, units_payload), *mixed],
            ),
        )
        version = am.ensure_version(client, model=model, run_id=trained.run_id)
        results.append(RegisteredModel(role, model.model_id, version, stored, sha, lineage_sha))
    training_dataset = am.mlflow_dataset(
        dataset_lineage=lineage, dataset_lineage_sha256=dataset.lineage_sha256,
        index_uri=f"s3://{runtime.datasets_bucket}/{dataset.published_name}/by-id/{dataset.dataset_id}.json",
    )
    am.log_dataset_input(client, run_id=trained.run_id, dataset=training_dataset, context="training",
                         model_id=parent.model_id or None)
    from mlflow.entities import LoggedModelOutput
    from mlflow.exceptions import MlflowException

    run_outputs = client.get_run(trained.run_id).outputs
    linked = {o.model_id for o in ((run_outputs.model_outputs if run_outputs else None) or [])}
    missing = [LoggedModelOutput(m.model_id, 0) for m in results if m.model_id not in linked]
    if missing:
        try:
            client.log_outputs(trained.run_id, missing)
        except MlflowException:
            # A retry may race an earlier attempt that already linked them.
            linked = {o.model_id for o in (client.get_run(trained.run_id).outputs.model_outputs or [])}
            if any(m.model_id not in linked for m in missing):
                raise
    return results


def _spec_dict(spec: TrainingSpec) -> dict:
    payload = asdict(spec)
    payload["gpu_profile"] = spec.gpu_profile.value
    return payload


def _register_imported(client, s3, experiment, parent: ParentHandle, runtime: RuntimeOptions) -> RegisteredModel:
    from Platform.pipelines import autoe2e_mlflow as am
    from Platform.pipelines.autoe2e_lineage import checkpoint_lineage, record_bytes, record_sha256

    stored = store_checkpoint(s3, source_uri=parent.checkpoint_uri, sha256=parent.checkpoint_sha256,
                              bucket=runtime.checkpoints_bucket)
    record = checkpoint_lineage(
        checkpoint={"sha256": parent.checkpoint_sha256, "uri": stored, "role": "imported", "step": parent.step},
        parent=None, dataset=None, split=None, training=None, selection=None,
    )
    lineage_sha = record_sha256(record)
    params = am.model_params(
        identity={"checkpoint_sha256": parent.checkpoint_sha256, "checkpoint_role": "imported",
                  "checkpoint_uri": stored, "checkpoint_step": parent.step,
                  "lineage_sha256": lineage_sha, "stage_index": 0},
        dataset_spec=None, training_spec=None,
        # Only stable values: a second import of the same weights must match.
        provenance={"imported": "true"},
    )
    model = am.ensure_logged_model(
        client, experiment=experiment, source_run_id=None, name="imported", params=params,
        description_text=am.description(checkpoint={"role": "imported", "sha256": parent.checkpoint_sha256},
                                        lineage=record, tracking_url=runtime.tracking_url, experiment=experiment),
        artifacts=am.lineage_artifacts(checkpoint_lineage=record_bytes(record), datasets=[]),
    )
    version = am.ensure_version(client, model=model, run_id=None)
    return RegisteredModel("imported", model.model_id, version, stored, parent.checkpoint_sha256, lineage_sha)


# --- evaluation -----------------------------------------------------------------


def _evaluation_run(client, experiment: str, name: str) -> str:
    return client.create_run(experiment, run_name=name, tags={"flyte_execution_id": _execution_id()}).info.run_id


def _log(client, s3, runtime, *, run_id, model: RegisteredModel, dataset: DatasetHandle, suite: str,
         metrics: dict, protocol: dict, per_sample_uri: str, per_sample_sha: str,
         model_reexecuted: bool = True) -> None:
    from Platform.pipelines import autoe2e_mlflow as am
    from Platform.pipelines.autoe2e_lineage import evaluation_record, record_bytes
    from Platform.pipelines.autoe2e_store import read_dataset_lineage

    lineage = read_dataset_lineage(s3, bucket=runtime.datasets_bucket, key=dataset.lineage_key, sha256=dataset.lineage_sha256)
    record = evaluation_record(
        checkpoint_sha256=model.checkpoint_sha256, checkpoint_lineage_sha256=model.lineage_sha256,
        suite=suite,
        dataset={"dataset_id": dataset.dataset_id, "lineage_sha256": dataset.lineage_sha256,
                 "published": f"{dataset.published_name}/{dataset.version}"},
        protocol=protocol, metrics=metrics,
        per_sample={"uri": per_sample_uri, "sha256": per_sample_sha} if model_reexecuted else None,
        model_reexecuted=model_reexecuted,
        code={"image_digest": TRAINING_IMAGE, "git_commit": os.environ.get("AUTO_E2E_GIT_COMMIT", "unknown")},
        flyte_execution_id=_execution_id(), finished_at=datetime.now(timezone.utc).isoformat(),
    )
    mlflow_dataset = am.mlflow_dataset(
        dataset_lineage=lineage, dataset_lineage_sha256=dataset.lineage_sha256,
        index_uri=f"s3://{runtime.datasets_bucket}/{dataset.published_name}/by-id/{dataset.dataset_id}.json",
    )
    # MLflow metric names allow "/", ".", "-", and "_" but not "@".
    suite_name = f"{suite}/{dataset.published_name}-{dataset.version}"
    _attach_dataset_lineage(client, s3, runtime, model.model_id, dataset, lineage)
    am.log_evaluation(
        client, run_id=run_id, model_id=model.model_id, dataset=mlflow_dataset,
        context=f"eval:{suite}", metrics=metrics, suite=suite_name,
        # Attempt-scoped name: a re-evaluation or a retry adds a record, never
        # replaces one.
        record_payload=record_bytes(record),
        record_name=f"{dataset.dataset_id}-{_execution_id()}-{os.environ.get('FLYTE_ATTEMPT_NUMBER', '0')}.json",
    )


def _attach_dataset_lineage(client, s3, runtime, model_id: str, dataset: DatasetHandle, lineage: dict) -> None:
    """Keep a byte-identical copy of an evaluation dataset's lineage on the model."""
    import tempfile
    from pathlib import Path

    from Platform.pipelines.autoe2e_lineage import record_bytes

    units_key = lineage["units"]["uri"].removeprefix(f"s3://{runtime.datasets_bucket}/")
    units_payload = s3.get_object(Bucket=runtime.datasets_bucket, Key=units_key)["Body"].read()
    with tempfile.TemporaryDirectory() as root:
        base = Path(root, "lineage", "datasets", dataset.dataset_id)
        base.mkdir(parents=True)
        (base / "dataset.json").write_bytes(record_bytes(lineage))
        (base / "units.jsonl").write_bytes(units_payload)
        client.log_model_artifacts(model_id, root)


@task(container_image=TRAINING_IMAGE, labels=QUEUES["gpu-validation"], pod_template=VALIDATION_POD,
      requests=Resources(cpu="6", mem="48Gi", gpu="1", ephemeral_storage="400Gi"),
      limits=Resources(cpu="6", mem="48Gi", gpu="1", ephemeral_storage="400Gi"),
      environment=EXPERIMENT_ENV, retries=1)
def evaluate_geo_val(models: List[RegisteredModel], training: TrainingSpec, dataset: DatasetHandle,
                     runtime: RuntimeOptions) -> None:
    """Re-run every model on the whole held-out Pittsburgh split and record per-sample results."""
    import tempfile

    import torch

    from training.autobev_data import autobev_validation_batches
    from evaluation.checkpoint_validation import validate
    from model_components import AutoE2E
    from model_components.autobev import AutoBEV
    from Platform.pipelines import autoe2e_mlflow as am
    from Platform.pipelines.dataset_publication import sha256_bytes
    from Platform.pipelines.dataset_publication_tasks import _put_immutable, _split_s3_uri

    client = _client()
    s3 = _s3()
    device = torch.device("cuda")
    run_id = _evaluation_run(client, am.experiment_id(client), f"geo-val-{_execution_id()}")
    for model in models:
        bucket, key = _split_s3_uri(model.checkpoint_uri)
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            s3.download_file(bucket, key, handle.name)
            state = torch.load(handle.name, map_location="cpu", weights_only=False)
        stage = int(state["metadata"]["stage"])
        network = (AutoBEV(pretrained_backbone=False) if stage == 1 else AutoE2E(pretrained_backbone=False))
        network.load_state_dict(state["model"])
        result = validate(
            network.to(device), stage,
            calibration=autobev_validation_batches(dataset.partition_uris, split="calibration", limit=0),
            validation=autobev_validation_batches(dataset.partition_uris, split="validation", limit=0),
            device=device,
        )
        report = json.dumps({"checkpoint_sha256": model.checkpoint_sha256, **result}, sort_keys=True).encode()
        report_sha = sha256_bytes(report)
        report_key = f"autoe2e/evaluations/{model.checkpoint_sha256}/geo_val/{dataset.dataset_id}/{report_sha}.json"
        _put_immutable(s3, bucket=runtime.checkpoints_bucket, key=report_key, payload=report,
                       content_type="application/json")
        _log(client, s3, runtime, run_id=run_id, model=model, dataset=dataset, suite="geo_val",
             metrics=result["metrics"], protocol={"name": "autoe2e_geo_val_v1", "stage": stage},
             per_sample_uri=f"s3://{runtime.checkpoints_bucket}/{report_key}", per_sample_sha=report_sha)
    client.set_terminated(run_id)


@dynamic(container_image=TRAINING_IMAGE, environment=PIPELINE_ENV)
def evaluate(
    models: List[RegisteredModel],
    evaluation: EvaluationSpec,
    training: TrainingSpec,
    dataset: DatasetHandle,
    runtime: RuntimeOptions,
) -> None:
    for target in evaluation.targets:
        if target.dataset is not None:
            raise ValueError("geo_val reads the training publication's held-out split; leave its dataset empty")
        evaluate_geo_val(models=models, training=training, dataset=dataset, runtime=runtime)


@task(container_image=TRAINING_IMAGE, requests=Resources(cpu="2", mem="8Gi"), environment=EXPERIMENT_ENV, retries=2)
def verify_models(dataset: DatasetHandle, models: List[RegisteredModel], runtime: RuntimeOptions) -> PipelineResult:
    """Recompute every lineage digest of the registered models before returning."""
    from Platform.pipelines.autoe2e_mlflow import REGISTERED_MODEL
    from Platform.pipelines.autoe2e_verify import verify_lineage

    from Platform.pipelines.autoe2e_verify import verify_published_dataset

    client = _client()
    s3 = _s3()
    # The training dataset is verified even when no model was trained on it.
    print(json.dumps(verify_published_dataset(
        s3, lineage_uri=f"s3://{runtime.datasets_bucket}/{dataset.lineage_key}",
        lineage_sha256=dataset.lineage_sha256,
    ), sort_keys=True))
    for model in models:
        summary = verify_lineage(client, s3, f"{REGISTERED_MODEL}/{model.version}")
        print(json.dumps(summary, sort_keys=True))
    return PipelineResult(dataset.dataset_id, dataset.lineage_sha256, models)


@workflow
def wf_autoe2e(
    dataset: DatasetSpec,
    training: TrainingSpec,
    evaluation: EvaluationSpec,
    runtime: RuntimeOptions,
    mix: List[DatasetSpec] = [],
    early_mix: List[DatasetSpec] = [],
    parent: Optional[ModelRef] = None,
) -> PipelineResult:
    """Build or reuse the datasets, train one stage, register, and evaluate, all traceable in MLflow.

    ``mix`` are the BEV datasets trained together with ``dataset`` (all of
    stage 1, and 15 % of stage 2's samples with the segmentation loss only);
    ``early_mix`` (SimBEV) joins stage 1 for its first half only.
    """
    plan = plan_dataset(spec=dataset, datasets_bucket=runtime.datasets_bucket)
    resolved = resolve_dataset(spec=dataset, plan=plan, datasets_bucket=runtime.datasets_bucket)
    published = ensure_dataset(spec=dataset, plan=plan, handle=resolved, runtime=runtime)
    mixed = ensure_datasets(specs=mix, runtime=runtime)
    early = ensure_datasets(specs=early_mix, runtime=runtime)
    parent_handle = resolve_parent(parent=parent)
    trained = train(spec=training, dataset=published, mix=mixed, early_mix=early, parent=parent_handle)
    models = register_models(
        dataset_spec=dataset, dataset=published, mix=mixed, training=training,
        parent=parent_handle, trained=trained, runtime=runtime,
    )
    done = evaluate(models=models, evaluation=evaluation, training=training, dataset=published, runtime=runtime)
    result = verify_models(dataset=published, models=models, runtime=runtime)
    done >> result
    return result
