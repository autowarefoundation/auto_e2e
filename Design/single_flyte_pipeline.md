# Design: Single Flyte Pipeline from Raw Data to a Traceable Model

## Document Metadata

| Field | Value |
|---|---|
| Status | Draft |
| Owner | riita10069 |
| Created | 2026-10-20 |
| Replaces | All 40 `wf_*` entry points in `Platform/pipelines/` on main at `c7fc884d` |
| Entry points | `wf_acquire_raw_snapshot` once per upstream revision, `wf_autoe2e` for everything else |
| Training path | Ray Train on KubeRay, admitted by Kueue |

## 1. Executive Summary

Producing one evaluated model today takes several separate launches. A
KITScenes model needs `wf_sharded_full_run` or a Ray training workflow, then
`wf_evaluate_kitscenes_benchmark`, then `wf_publish_full_run_overlays` from a
CodeBuild launcher, then a manual Console materialization Job. A nuPlan model
also needs `wf_acquire_nuplan_raw_snapshot`, a pack workflow, and a
hand-written S3 list of shard URIs. Results end up in different MLflow runs
and experiments, up to three model versions are created per training run,
and no record connects a checkpoint to the raw data and labels it was
trained on.

This design keeps two entry points:

- `wf_acquire_raw_snapshot` copies upstream data into an immutable S3 raw
  snapshot once per upstream revision. It is the only code that can reach
  Hugging Face.
- `wf_autoe2e` does everything else from S3: it builds or reuses a
  dataset, trains, evaluates, registers, and precomputes Console overlays.

The data flows through four layers, each immutable and addressed by digest:

```
raw snapshot ──► labels ──► published dataset ──► checkpoint ──► evaluations
 (tar + index)   (per unit)   (WebDataset shards)   (model version)   (same run)
```

Every MLflow model version carries a lineage record that names each layer
by digest, so any checkpoint can be traced back to the exact raw bytes,
labels, pack code, and parent model it came from, and every evaluation ever
run on it is on its run.

## 2. Decisions

| Decision | Choice | Reason |
|---|---|---|
| Entry points | `wf_acquire_raw_snapshot` and `wf_autoe2e` only | One interface to learn. Acquisition has a different lifecycle and is the only code that holds the HF token |
| Upstream access | `wf_autoe2e` tasks never receive the `hf-token` secret | Processing never depends on the Hub, and the guarantee is structural |
| Raw format | Uncompressed tar per unit plus a canonical JSON Lines member index | nuPlan archives already are uncompressed tar. Range reads of one member need only a key, an offset, and a length |
| Pack format | WebDataset shards, as today | Training streams samples sequentially |
| Datasets | KITScenes, nuPlan, L2D behind one adapter interface. NVIDIA PhysicalAI is deleted | The workflow never branches on the dataset. NVIDIA has no fan-out planner and no Reactive stage |
| Datasets per execution | Exactly one training dataset | Keeps lineage unambiguous. Evaluation datasets are separate inputs |
| Expensive stages | Raw is acquired once; labels are recorded per unit and never repeated | A pack contract bump must only re-pack |
| Dataset reuse | By-id index lookup by `dataset_id` | Flyte cache keys include non-semantic inputs, and DataCatalog write failures already forced a manual recovery manifest (#121) |
| Training backend | Ray only. `train_il` and `train_offline_rl` are removed | One code path for 1, 2, 4, and 8 GPUs |
| Multi-stage training | `parent` accepts a model version or a checkpoint | Chains of any depth, one execution per stage |
| Model versions | One per distinct checkpoint: `best`, and `final` only if its SHA-256 differs | No duplicate versions for the same weights |
| MLflow center | One MLflow LoggedModel per distinct checkpoint; the registered version links to it with `model_id` | The model page, not a run, shows lineage, parameters, and every evaluation |
| Evaluation-only runs | A separate run that takes the model as input; metrics attach to the LoggedModel | Evaluations added later land on the same model without editing older runs |
| MLflow layout | One experiment `auto-e2e` | One place to compare everything |

## 3. Interface

```python
@workflow
def wf_acquire_raw_snapshot(
    dataset: AutoE2EDataset,
    upstream_revision: str,
    units: List[str],               # empty list = every unit at the revision
    datasets_bucket: str,
) -> ManifestRef: ...               # the snapshot manifest

@workflow
def wf_autoe2e(
    dataset: DatasetSpec,
    training: TrainingSpec,
    evaluation: EvaluationSpec,
    parent: Optional[ModelRef] = None,
    runtime: RuntimeOptions = RuntimeOptions(),
) -> PipelineResult: ...
```

```python
@dataclass
class DatasetSpec:                   # every field enters dataset_id
    dataset: AutoE2EDataset          # KITSCENES, NUPLAN, L2D
    source_manifest: ManifestRef     # raw snapshot manifest: URI + VersionId + SHA-256
    unit_limit: int                  # 0 = all units; N = first N in manifest order
    pack_version: str                # must equal current_pack_version(dataset), derived from the contract constants
    image_size: int
    pack_params: Dict[str, str]      # adapter-validated, e.g. nuPlan scenario budget
    reasoning: Optional[ReasoningSpec]  # teacher, prompt_version, label_stride

@dataclass
class ModelRef:                      # exactly one field is set
    model_version: Optional[str]     # e.g. "auto-e2e-driving-policy/66"
    checkpoint: Optional[FlyteFile]  # a checkpoint that was never registered

@dataclass
class TrainingSpec:
    gpu_profile: GpuProfile          # RAY_1, RAY_2, RAY_4, RAY_8
    epochs: int                      # 0 = evaluate the parent only
    learning_rate: float
    per_rank_batch_size: int
    seed: int
    objectives: ObjectiveSpec        # loss weights

@dataclass
class EvaluationSpec:
    suites: List[EvalSuite]          # INTERNAL_VAL, KITSCENES_BENCHMARK, REACTIVE
    datasets: Dict[str, DatasetSpec] # suite name -> evaluation dataset, if not INTERNAL_VAL

@dataclass
class RuntimeOptions:                # speed only, never enters any identity
    label_concurrency: int
    pack_concurrency: int
    copy_workers: int
```

The upstream revision is read from `source_manifest`, so the manifest digest
pins it. Dataset-specific behavior lives behind the adapter for
`DatasetSpec.dataset`.

## 4. Data Layers

### 4.1 Raw snapshots

```
s3://<datasets_bucket>/<dataset>/raw-snapshots/<upstream_revision>/<snapshot_id>/
  units/<unit_id>.tar      # uncompressed tar, upstream files byte for byte
  index/<unit_id>.jsonl    # one canonical JSON line per member
  manifest.json            # written last, write-once
```

Each tar member is named `<unit_id>/<upstream relative path>`, so the tars
also stream with WebDataset tools. JPEG and Parquet sources are already
compressed, so the tar is not.

The index has one row per member: tar key, `offset_data`, size, SHA-256, and
the adapter's parsed fields (sensor, timestamp, frame index). It is written
once by streaming the tar with the standard `tarfile` module. Reading one
member is a single ranged GET, `fs.cat_file(tar_key, offset, offset + size)`;
no FUSE and no extraction. Planners select members by querying the index
instead of listing S3.

The manifest records the dataset, upstream repository and revision, how the
bytes were obtained, and per unit: tar key, tar size, tar SHA-256, member
count, and index SHA-256. `wf_autoe2e` checks the manifest digest against
`DatasetSpec.source_manifest` before reading anything.

`wf_acquire_raw_snapshot` downloads each requested unit, verifies upstream
hashes, writes the tar and index with `IfNoneMatch="*"`, and writes the
manifest last. Rerunning it rewrites nothing.

| Dataset | Unit | Snapshot | Bytes come from |
|---|---|---|---|
| KITScenes train | scene | `kitscenes/raw-snapshots/6fde0034…/` | the 533 `raw_uri` prefixes of Full Run `a5nk56zsbcg7g6npjfpt` in the Flyte artifacts bucket, tarred in-region (1.75 TB, 14.47 million objects become 533 tars). No Hub access |
| nuPlan | archive | `nuplan/raw-snapshots/nuplan-v1.1-complete/nuplan-v1.1-complete-20260809/` | already exists. Its 341 archives are uncompressed tar despite the `.zip` name; only the index is added |
| L2D | one upstream file, all of `meta/`, and the OSM extract | `l2d/raw-snapshots/1e7578b1…/` | the first `wf_acquire_raw_snapshot` run downloads the episodes' files from the Hub and one dated Geofabrik extract pinned by a caller-given SHA-256 |
| KITScenes val and test | scene | `kitscenes-val-hub` | the first `wf_acquire_raw_snapshot` run is the only Hub access |

### 4.2 Labels

Labels are billed Cosmos calls and are never repeated. A label set is per
unit, and its identity excludes everything about the pack:

| Identity fields | Not included |
|---|---|
| dataset, upstream revision, unit id, teacher, teacher model, `prompt_version`, `label_stride`, `REASONING_LABEL_POLICY_VERSION`, `UID_SCHEMA_VERSION` | parser, shard, and geometry versions |

Labelling one unit is a three-step commit:

1. a write-once claim `<dataset>/_stages/labels/<label_id>.claim` names the
   one Flyte execution allowed to call the teacher; another execution fails
   instead of paying twice, and a retry of the same execution proceeds;
2. that execution writes the files under a fresh attempt prefix
   `<dataset>/labels/<label_id>/<attempt_id>/`, so nothing is overwritten;
3. the write-once record `<dataset>/_stages/labels/<label_id>.json` commits
   one attempt with a digest over every file's relative key, size, and ETag.

A build verifies that digest before use and fails on a mismatch instead of
re-labeling. The label identity names the teacher model
(`ReasoningSpec.teacher_model`); the task refuses to label when the endpoint
serves another model.

A parser change can change which frames exist, but it does not re-label. The
pack JOIN treats a sample without a label as fully masked, and the
publication records `reasoning_label_count`, so lost coverage is visible.

The 533 KITScenes label sets of Full Run `a5nk56zsbcg7g6npjfpt` are copied
into the label store once. A read-only dry run on 2026-10-20 checked every
label `meta.json` against the label identity.

### 4.3 Published datasets

`dataset_id` is the SHA-256 of the canonical JSON of `DatasetSpec`.
Publications keep the `<dataset>/vN.M/` layout that the Console reads. After
the manifest gate, the build writes two write-once objects:

- `<published>/vN.M/partitions/<partition_id>/`: a byte-identical copy of
  each packed partition, which is what Ray Train reads, and
  `<published>/vN.M/partitions/manifest.json`, the views manifest that lists
  every partition manifest digest. This is the reuse gate for every dataset.
  KITScenes additionally gets the Console's flat `shards/` publication, whose
  manifest digest is recorded in the lineage;
- `<published>/by-id/<dataset_id>.json`: the canonical spec, version, and
  manifest key and SHA-256;
- `<dataset>/vN.M/lineage/dataset.json` and `units.jsonl`: the dataset
  lineage record (6.1), naming the raw snapshot, every label record, and the
  pack code.

The Console ignores both because they are not version directories or
shards. A publication is reused only if the index's spec equals the request,
the manifest SHA-256 matches the index and `.publication-lock.json`, the
manifest is `ready`, and every shard exists with the recorded size and
ETag. Any mismatch fails the execution.

### 4.4 Dataset adapters

| Member | Meaning |
|---|---|
| `published_name(plan)` | e.g. `kitscenes`, `kitscenes-val`, `nuplan` |
| `read_snapshot(manifest, unit_limit, pack_params)` | returns a `SourcePlan`: upstream revision, units, and pack partitions |
| `acquire_task` | downloads one unit into a snapshot. Used only by `wf_acquire_raw_snapshot` |
| `index_member(name)` | parses sensor, timestamp, and frame index from a member name |
| `label_task` | labels one unit from the snapshot, or `None` |
| `pack_task` | packs one partition from snapshot members and labels into WebDataset shards |

| | KITScenes | nuPlan | L2D |
|---|---|---|---|
| `acquire_task` | HF download of one scene | `acquire_nuplan_archive` | HF download of one file, `meta/`, or the OSM extract |
| `label_task` | `generate_reasoning_labels` | `None` | `None` |
| Partition | one scene | one sensor group (43 for train) | `pack_params["partition_size"]` episodes |
| `pack_task` | `data_processing` | `pack_nuplan_snapshot_reactive_dataset` | `data_processing` |
| `pack_params` | none | `total_scenario_limit`, `samples_per_shard`, `max_rejection_fraction` | `partition_size` |
| Validation split | frozen holdout `kitscenes_train_dev_v4.json` | hash buckets of the log id | hash buckets |

Pack tasks read only the members their partition needs. A nuPlan pack no
longer downloads and extracts whole 157 to 250 GB archives, which is what
forces 1200Gi of ephemeral storage today, so pack partitions can be small
and many.

The code that fetches metadata from the Hub today
(`plan_fanout_partitions` through `LeRobotDatasetMetadata`,
`Model/data_parsing/kit_scenes/source.py`, and the KITScenes test direct
pack) reads the same files from the snapshot instead.

### 4.4.1 L2D navigation

L2D has no HD map. It ships a 640x360 north-up OpenStreetMap tile per frame
and ten GPS route waypoints, while the Reactive contract needs the 14 map
channels and 2 route channels of nuPlan and KITScenes. They are rebuilt from
two sources:

| Source | Channels |
|---|---|
| OSM lane graph (`osm_lane_graph.py`): drivable ways split at junctions into directed lanes with boundaries, successors, neighbours, and levels; traffic signals, stop signs, and crossings become signals, stop lines, and crosswalks | all 14, through the shared `OSMMapAdapter` and native rasterizer |
| The dataset's map tile, registered by fitting its route dots to the waypoints at the known north-up rotation (`map_image.py`) | road surface added to `DRIVABLE_AREA`, footprint to `KNOWN_MAP_AREA` |

The route is matched from the recorded trace onto the lane graph, rendered on
the 500 ms anchor grid and warped to each frame, as KITScenes does. Members
are the same four KITScenes writes, so the loader reads every dataset alike.

Bytes and identity:

| Artifact | Where | Identity |
|---|---|---|
| OSM extract | the snapshot's `osm` unit | tar digest in the snapshot manifest |
| OSM region (ways near every pose and waypoint, 1 km buffer) | `l2d/osm-regions/<sha256>.json`, built once per plan by `build_l2d_osm_region` | its SHA-256 in every partition manifest and in the lineage `pack.osm_region`; `verify_lineage` rechecks both and the source unit |
| Every rule that decides the bytes (buffers, margins, lane widths, highway set, versions) | `navigation_contract.py` | `L2D_NAVIGATION_CONTRACT`, part of the L2D `pack_version` and so of `dataset_id` |

Each episode builds its own lane graph from the region with its own UTM
origin, and GPS heading is turned into grid yaw, since meridian convergence
reaches several degrees across Germany.

Open for scale (100k episodes): one region per dataset makes every pack read
the whole region and the plan carry every episode's units. Per-partition
subregions and an episode index outside the manifest would bound both.

## 5. `wf_autoe2e` Stages

| # | Stage | Skipped when |
|---|---|---|
| 1 | Resolve the training dataset and each evaluation dataset by `dataset_id` | — |
| 2 | Build: label units without a label record, pack every partition, publish, write by-id and lineage | the dataset is already published |
| 3 | Resolve the parent: download the checkpoint, verify its SHA-256, read its `stage_index` | no parent |
| 4 | Train with Ray Train | `epochs == 0` |
| 5 | Evaluate every suite on every distinct checkpoint | — |
| 6 | Register one model version per new distinct checkpoint | `epochs == 0` |
| 7 | Verify lineage: `verify_lineage` recomputes every digest of the new models | — |
| 8 | Precompute Console overlays on the training dataset for the new `best`, or for the parent of an `epochs=0` run | the dataset is not `kitscenes` or has no Console publication (`unit_limit > 0`), `runtime.artifacts_bucket` is empty, the run is a `ray_1` smoke, or the parent was not trained by wf_autoe2e on KITScenes (no Ray run or no internal_val ADE/FDE) |

Stage 8 reuses `prepare_overlay_set`, `precompute_overlay_partition`, and
`finalize_overlay_set`. Reactive inference costs about 15 to 20 s per sample on an
L4 in bf16, so the non-empty partitions are split into
`runtime.overlay_workers` chunks of similar sample count that run in parallel
on the scale-to-zero `gpu-overlay` pool. The
development namespace quota reserves 16 of its 34 GPU requests for this
fan-out, so the default is 16 chunks; the v3.7 set took about 30 hours on 10. A smoke model is
put on the Console only on purpose: an `epochs=0` run with that model as
`parent`, whose attribution (backbone, val_fraction, training execution,
training dataset version, Ray run, internal_val ADE/FDE) is read from the
parent's own LoggedModel. Overlays are keyed by the checkpoint SHA-256 and the
Console publication digest from the dataset lineage
(`console.manifest_sha256`), which is why a build without a Console
publication has none. The overlay metadata names the registry version
(`models:/<model_id>`, numeric version, Ray run id) and the Ray validation
ADE/FDE at 6.4 s; a model without them is refused rather than shown with
0.0. `load_policy` rebuilds a Reactive checkpoint from its stage constructor
with pretrained weights off, as the KITScenes benchmark evaluator does.

`train_ray` is one task factory. `gpu_profile` picks the `RayJobConfig`,
Kueue queue, priority class, and node selector from a single table. The task
reads the active Capacity Block end time itself.

Registration first searches `auto-e2e-driving-policy` for a version whose
`checkpoint_sha256` tag equals the checkpoint, so one set of weights never
gets two versions, even across executions.

With `epochs == 0`, nothing is trained. The execution creates an
evaluation run that takes the parent's LoggedModel as input and attaches its
metrics to that LoggedModel. If the parent is an unregistered checkpoint, the
execution first creates a LoggedModel and a version for it once, with
`checkpoint_role=imported`; its checkpoint lineage marks `dataset`, `split`,
and `training` as unknown instead of guessing them.

## 6. Traceability in MLflow

MLflow 3 tracks models as first-class LoggedModels, separate from runs. A
run keeps only what MLflow and the training loop record on their own. The
`best` and `final` LoggedModels hold everything else, and the LoggedModel
page is where a checkpoint is read.

```
parent LoggedModel ──input──► training run ──output──► LoggedModel (best) ◄── version N
                                   │                    LoggedModel (final) ◄── version N+1
training dataset ──input───────────┘                         ▲
evaluation dataset ──input──► evaluation run ──metrics───────┘
```

| MLflow object | Created by | Holds |
|---|---|---|
| Training run | stage 4 | training curves, system metrics, Flyte execution id. inputs: the training dataset (`training`) and the parent LoggedModel. outputs: the new LoggedModels. No artifacts |
| Evaluation run, one per execution of stage 5 | stage 5 | inputs: the LoggedModel and each evaluation dataset (`eval:<suite>`). No artifacts |
| LoggedModel, one per distinct checkpoint | stage 6 with `initialize_logged_model(source_run_id=training run)` | params (6.0), description (6.0), metrics, and artifacts: the checkpoint, its checkpoint lineage, the dataset lineage copies, and its evaluation records |
| Metrics | `log_metric(model_id=…, dataset=…)` | every evaluation metric, attached to the LoggedModel and the MLflow Dataset it was measured on |
| Model version | `create_model_version(source="models:/<model_id>", run_id=…)` | a pointer to the LoggedModel, plus the identity tags for registry search |
| MLflow Dataset | `log_input` with an `mlflow.entities.Dataset` (6.0) | name `<published name>/<version>`, digest `dataset_id`, source the by-id index URI, and a profile with the lineage summary |

Because metrics carry the dataset, the same metric name measured on two
evaluation datasets stays two results, and an evaluation added a month later
appears next to the original ones on the LoggedModel page.
`search_logged_models(datasets=[{"dataset_name": …, "dataset_digest": …}])`
lists every model evaluated on a given dataset; the digest is required for
a match. The parent input on each training run makes the stage chain a graph
that MLflow itself stores.

### 6.0 Constraints found on the production server

A probe against the production MLflow 3.13.0 on 2026-10-20 created a
LoggedModel, dataset-tied metrics, a parent input, and a registered version,
then removed them. It settled three placement choices:

| Finding | Consequence |
|---|---|
| The LoggedModel page shows metrics with their dataset and source run, params, runs, datasets used, model versions, and artifacts, but not tags | Every lineage digest is a param: `checkpoint_sha256`, `checkpoint_role`, `lineage_sha256`, `dataset_id`, `dataset_lineage_sha256`, `parent_model_id`, `stage_index`, image digests, and git commit, next to the flattened `DatasetSpec` and `TrainingSpec`. Params are write-once, which fits immutable identity. The description is generated Markdown that links the parent model page, the published dataset, and the evaluation records |
| The OSS registry has no `model_id` column; a version links to a LoggedModel only through `source="models:/<model_id>"` | Versions are created with that source and the training run id. The same identity values are also set as version tags, because the registry list filters on tags and not on LoggedModel params |
| `MetaDataset` does not serialize a profile, and the skinny client has no `mlflow.data` | The pipeline builds `mlflow.entities.Dataset` directly with a JSON profile: `dataset_lineage_sha256`, `snapshot_manifest_sha256`, `label_set_sha256`, pack contracts, and sample and shard totals |

Cleanup of a probe must delete objects by id. `mlflow gc` without
`--run-ids` permanently deletes every run in the deleted lifecycle stage of
the whole server, not only the given experiment; the probe cleanup did this
and purged four unrelated runs that were already in the trash.

### 6.1 Lineage records

Lineage is three kinds of record. Each is canonical JSON (sorted keys,
compact separators), carries a `schema_version`, is written once with
`IfNoneMatch="*"`, and is identified by the SHA-256 of its bytes. A record
refers to another record only by URI plus SHA-256, never by copying it, so
the chain is a hash chain: changing any byte upstream breaks every digest
downstream.

| Record | Written by | Location | One per |
|---|---|---|---|
| dataset lineage | stage 2, after the shards and before the by-id index | `<dataset>/vN.M/lineage/dataset.json` and `units.jsonl` | published dataset |
| checkpoint lineage | stage 6, or stage 5 for an imported checkpoint | LoggedModel artifact `lineage/checkpoint.json` | distinct checkpoint |
| evaluation record | stage 5 | LoggedModel artifact `evaluations/<suite>/<dataset_id>.json` | checkpoint, suite, and evaluation dataset |

A dataset lineage is shared by every checkpoint trained on that dataset, so
its canonical copy lives with the publication in S3. Every LoggedModel also
stores byte-identical copies of `dataset.json` and `units.jsonl` for its
training dataset and for each dataset it was evaluated on, under
`lineage/datasets/<dataset_id>/`. The copies are checked against the
recorded SHA-256, so the hash chain holds, and the model page alone is
enough to trace a checkpoint even if the S3 publication is gone. Checkpoint
lineage and evaluation records live only on the LoggedModel. Evaluation
records are only ever added, so a checkpoint's history is the set of its
records, and no earlier record is rewritten.

#### Dataset lineage (`autoe2e_dataset_lineage_v1`)

`units.jsonl` has one row per unit used:

| Column | Meaning |
|---|---|
| `unit_id` | scene, archive, or episode id |
| `partition_id` | pack partition that consumed the unit |
| `raw_tar_key`, `raw_tar_digest`, `raw_index_sha256` | from the raw snapshot manifest; the digest is `sha256:<hex>` for tars this code wrote and `crc64nvme:<base64>` for the S3-copied nuPlan archives, which have no member index (`raw_index_sha256` is `null`) |
| `label_id`, `label_record_sha256` | null for a dataset without labels |
| `label_origin` | `reused` or `created` in this build |
| `sample_count`, `label_count` | contribution of the unit to the published shards |

`dataset.json`:

```json
{
  "schema_version": "autoe2e_dataset_lineage_v1",
  "dataset_id": "…",
  "spec": {"dataset": "kitscenes", "source_manifest": {"uri": "s3://…/manifest.json", "version_id": "…", "sha256": "…"}, "unit_limit": 0, "pack_version": "…", "image_size": 256, "pack_params": {}, "reasoning": {"teacher": "openai_compatible", "teacher_model": "nvidia/Cosmos3-Nano", "prompt_version": "action_relevant_reasoning_v3_temporal_front256", "label_stride": 10}},
  "published": {"name": "kitscenes", "version": "v3.6", "manifest_key": "kitscenes/v3.6/shards/manifest.json", "manifest_sha256": "…"},
  "raw": {"upstream": "KIT-MRT/KITScenes-Multimodal", "revision": "6fde0034…", "snapshot_manifest_sha256": "…", "units": 533},
  "labels": {"label_policy_version": "v2", "uid_schema_version": "v1", "label_set_sha256": "…", "reused": 533, "created": 0},
  "units": {"uri": "s3://…/kitscenes/v3.6/lineage/units.jsonl", "sha256": "…", "rows": 533},
  "pack": {"contracts": {"uid": "v1", "parser": "v3", "shard": "v13", "geometry": "v7"}, "image_digest": "sha256:…", "git_commit": "…", "task": "data_processing"},
  "build": {"flyte_execution_id": "…", "finished_at": "2026-10-21T03:12:00Z"},
  "totals": {"samples": 42667, "shards": 404, "reasoning_labels": 4598}
}
```

`label_set_sha256` is the digest of the sorted `(unit_id, label_id,
label_record_sha256)` rows, so two datasets that share exactly the same labels
have the same value even if their packs differ. `spec` repeats the
`DatasetSpec` so that `dataset_id` can be recomputed from this record alone.

#### Checkpoint lineage (`autoe2e_checkpoint_lineage_v1`)

```json
{
  "schema_version": "autoe2e_checkpoint_lineage_v1",
  "checkpoint": {"sha256": "…", "uri": "s3://…/best.pt", "role": "best", "epoch": 6, "step": 21333},
  "stage_index": 2,
  "parent": {"model_id": "m-…", "model_version": "auto-e2e-driving-policy/66", "checkpoint_sha256": "…", "lineage_sha256": "…"},
  "dataset": {"dataset_id": "…", "lineage_uri": "s3://…/kitscenes/v3.6/lineage/dataset.json", "lineage_sha256": "…"},
  "split": {"policy": "kitscenes_train_dev_v4", "validation_units_sha256": "…", "validation_units": 40},
  "training": {"spec": {"gpu_profile": "RAY_8", "epochs": 10, "learning_rate": 3e-05, "per_rank_batch_size": 2, "seed": 149, "objectives": {}}, "flyte_execution_id": "…", "mlflow_run_id": "…", "image_digest": "sha256:…", "git_commit": "…", "kueue_queue": "p5en-capacity-block", "capacity_reservation_id": "cr-…"},
  "selection": {"metric": "eval/internal_val/ade_6p4s", "value": 2.633}
}
```

`parent` is `null` for a model trained from scratch. For an imported
checkpoint, `dataset`, `split`, and `training` are
`{"status": "unknown"}` instead of guesses, and `stage_index` is 0.

#### Evaluation record (`autoe2e_evaluation_v1`)

```json
{
  "schema_version": "autoe2e_evaluation_v1",
  "checkpoint_sha256": "…",
  "checkpoint_lineage_sha256": "…",
  "suite": "kitscenes_benchmark",
  "dataset": {"dataset_id": "…", "lineage_sha256": "…", "published": "kitscenes-val/v3.5"},
  "protocol": {"name": "kitscenes_paper_approximation", "version": "v1", "horizons_s": [3, 5]},
  "model_reexecuted": true,
  "metrics": {"ade_3s": 0.71, "fde_3s": 1.90, "ade_5s": 1.85, "fde_5s": 5.38},
  "per_sample": {"uri": "s3://…/per_sample.parquet", "sha256": "…", "rows": 200},
  "code": {"image_digest": "sha256:…", "git_commit": "…"},
  "flyte_execution_id": "…",
  "finished_at": "…"
}
```

`model_reexecuted` is `true` for every suite that runs the model on the
dataset, and then `per_sample` pins the output. It is `false` only for
`internal_val`, which copies the training run's own validation metrics and
has no per-sample output (`per_sample` is `null`). A suite that replays
stored overlays instead of running the model is not written at all.

#### Verification

`verify_lineage(model_version)` walks the chain and recomputes every digest:
the LoggedModel's checkpoint lineage, the dataset lineage and `units.jsonl` in
S3, the raw snapshot manifest, the label records, each evaluation record,
and then the parent's chain recursively. It fails on the first mismatch. The
smoke launch plan runs it after registration, so a pipeline change that
breaks lineage fails the smoke run.

### 6.2 Questions this answers

All answers start from the model version page and stay inside MLflow.

| Question | On the model page |
|---|---|
| Which data trained this checkpoint? | the training dataset input and its profile; `lineage/datasets/<dataset_id>/dataset.json` |
| Which raw bytes and upstream revision? | dataset lineage `raw`, and `units.jsonl` tar digests per unit |
| Which labels, from which teacher and prompt? | dataset lineage `spec.reasoning` and `labels`; `units.jsonl` names every label record |
| Which code packed and trained it? | dataset lineage `pack`; LoggedModel params for the training image and commit |
| What was it initialized from? | `parent_model_id`, which links to the parent's model page, recursively |
| Every evaluation ever run on it, and on which data? | the LoggedModel metrics, each tied to its evaluation dataset, and `evaluations/` |

## 7. Launching and the smoke run

There are no launch plans with defaults: bucket names and manifest digests
are account-specific and stay out of git. `Platform/scripts/launch_autoe2e.py`
launches either workflow from a JSON run file under the gitignored
`Platform/scripts/runs/`, and prints the typed inputs first.

The smoke run exercises every stage that current capacity allows:

- KITScenes train with `unit_limit=10`: snapshot read, label reuse, pack,
  views, Console publication, lineage, and index;
- KITScenes val with a few units for the benchmark;
- `epochs=0` with the existing fine-tuned checkpoint as a bare parent, so
  import, benchmark re-execution, dataset-tied metrics, and
  `verify_lineage` all run.

Training itself is not in the smoke run. `ray_1` accepts only the nuPlan
BEV-only single-rank smoke contract, `ray_2` needs two validation GPUs while
the reservation has one, and `ray_4`/`ray_8` need a Capacity Block.

## 8. Deletion

| Removed | Where its logic goes |
|---|---|
| `wf_sharded_full_run`, `wf_train_il`, `wf_train_offline_rl`, `wf_full_pipeline`, `wf_ingest_train_eval` | `wf_autoe2e`. The IL and offline RL training loops are deleted |
| `wf_train_reactive_*` (9 workflows), `wf_reactive_multistage_ray_2_canary` | `train_ray` with `gpu_profile` and `parent` |
| `wf_data_ingest`, `wf_acquire_nuplan_raw_snapshot` | `wf_acquire_raw_snapshot` |
| `wf_data_processing`, `wf_generate_reasoning_labels`, `wf_create_dataset*`, `wf_create_kitscenes_*evaluation_sharded`, nuPlan pack workflows | adapters and stage 2 |
| `wf_evaluate_*` | stage 5 |
| overlay and dataset publication workflows | stages 2 and 7 |
| `wf_audit_kitscenes_*` | validation inside stages 1 and 2 |
| `wf_prepare_kitscenes_paper_approximation`, `wf_export_trajectory_report` | deleted |
| NVIDIA PhysicalAI branches and `physical_ai_av` in the data-prep image | deleted |

The CodeBuild launcher buildspecs for removed workflows and the tasks no
stage calls are deleted with them. After `wf_autoe2e` is registered, the old
entry points are archived in Flyte.

## 9. Out of Scope

- Making Kueue admission depend on an existing Capacity Block
  (ProvisioningRequest or an AdmissionCheck).
- Automatic Capacity Block purchase. `Platform/README.md` keeps purchase as
  an operator action.
- Multi-node training and EFA.
