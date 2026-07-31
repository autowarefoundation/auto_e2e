# Design Document: Automatic ODD Labeling Pipeline

## Document Metadata

| Field | Value |
|---|---|
| Status | Proposed |
| Owner | riita10069 |
| Created | 2026-07-27 |
| Initial dataset | KITScenes |
| Future datasets | L2D, NVIDIA PhysicalAI-AV, and adapter-conformant datasets |
| Related design | `Docs/navigation_input_design.md` |
| Related implementation | `Model/navigation/`, `Model/data_processing/`, `Platform/pipelines/` |

## 1. Executive Summary

This document defines an automatic labeling pipeline for Operational Design
Domain (ODD), driving events, and perception-difficulty conditions. Its primary
outputs are:

1. corpus statistics that describe which conditions are represented and for
   how long;
2. searchable scene and interval metadata for dataset curation and failure
   analysis;
3. reproducible evaluation slices that can be joined to model results by stable
   scene and time identities.

These labels are not the existing Reasoning Labels. Reasoning Labels are sparse,
prompt-dependent supervision for the model's action-relevant reasoning branch.
ODD Labels are scene catalog metadata. They belong to a scene, not to a training
sample, and are never fed to AutoE2E as an input or target. Their primary
consumer is the DataModelConsole Dashboard: users inspect the available
ontology, understand dataset composition, search for scenes, and inspect one
scene's labels during playback. A future Active Learning extension may use the
same catalog to select scenes for a subsequent training dataset, but that is a
downstream use rather than the initial product. ODD Labels must provide broad
corpus coverage, retain source evidence, represent missingness explicitly, and
remain usable without a model, checkpoint, MLflow run, or training execution.

A scene can contain time-varying ODD conditions, event intervals, camera
conditions, and actor-scoped observations. These are children of one
`SceneLabelRecord`; they are not sample labels. A later training pipeline may
select the scene and independently enumerate its samples, but that enumeration
does not change the ODD LabelSet.

The initial implementation targets KITScenes, but dataset-specific logic is
confined to adapters. Every adapter emits the same canonical time-aligned
evidence contract. Source labelers then operate on canonical map/route,
GNSS/INS, camera, image-quality, and optional CAN evidence. Object tracks,
detector output, and LiDAR are not dependencies of the initial implementation.
The pipeline never encodes KITScenes directory names, frame numbering, or
Lanelet2 types in the label schema.

The design makes six central decisions:

1. Deterministic evidence is authoritative where it is semantically sufficient.
   Map/route and GNSS/INS derive road topology, planned route action, ego speed,
   and actual trajectory properties. A VLM does not re-guess these values.
2. Source evidence and resolved labels are separate artifacts. Fusion never
   destroys disagreement or hides which input produced a value.
3. Missing, unobservable, ambiguous, and explicitly absent conditions are
   different states. `none` is a valid observed negative; it is never a
   substitute for missing or unobservable evidence.
4. Labels are temporal intervals with explicit scope. Event labels describe
   actual behavior and interaction intervals; object-level perception labels
   require an actor or camera subject.
5. ODD ownership is scene-native. Timeline intervals exist to describe changes
   inside a scene, never to mirror the model's sliding sample windows.
6. An ODD LabelSet is an immutable sidecar bound to an immutable dataset
   manifest. KITScenes is republished once as v3.1 to carry lossless navigation
   contract v2; later ODD Label revisions do not require another repack.

The high-level data flow is:

```text
immutable dataset snapshot
  -> DatasetEvidenceAdapter
  -> canonical scene timeline + capability manifest
  -> source labelers
       map_route
       gnss_ins
       image_qc
       vlm
       can_optional
       fusion
  -> immutable source evidence
  -> deterministic fusion + temporal segmentation
  -> label/event validation and quality audit
  -> immutable ODD LabelSet
  -> Dashboard statistics/search
  -> optional query-time evaluation-slice projection
  -> optional future Active Learning SceneSelectionManifest
```

## 2. Problem Statement

The current Reasoning pipeline answers action-facing questions such as why the
planner should slow down or turn. It does not provide a complete description of
the dataset's ODD composition. In particular, it is not designed to answer:

- How much urban, motorway, workzone, wet-road, night, or high-traffic data is
  present?
- Which intervals contain a left route action versus an actually executed left
  turn?
- Which scenes have difficult image quality, occlusion, glare, small actors, or
  temporary traffic control?
- Which labels were determined from map/route, GNSS/INS, a VLM, or a fusion
  algorithm?
- Was a condition absent, unavailable in the dataset, outside the sensors'
  observability, or genuinely ambiguous?
- Can the same queries and statistics be produced for a dataset other than
  KITScenes?

A flat per-image VLM taxonomy does not solve this problem. It would duplicate
facts that deterministic sources already know, make temporal events unreliable,
and collapse data absence into false negatives. It also would not distinguish
navigation intent from the trajectory actually driven.

The required system is therefore a multi-source, time-aligned labeling pipeline,
not a larger VLM prompt.

## 3. Architectural Decisions

### 3.1 ODD LabelSet is separate from Reasoning Labels

The two artifacts have different ownership and use:

| Property | Reasoning Label | ODD LabelSet |
|---|---|---|
| Primary purpose | Model supervision | Dataset description, search, slicing |
| Typical coverage | Sparse samples | Complete eligible scene timeline |
| Ownership unit | Training sample/horizon | Scene and children of that scene |
| Primary producer | Offline teacher | Deterministic labelers + VLM + fusion |
| Time model | Fixed 0, 1, 2, 3, 4 s horizons | Arbitrary half-open intervals |
| Schema owner | Reasoning taxonomy | ODD ontology registry |
| Storage | Training-oriented `reasoning.json` / records | Immutable sidecar Parquet |
| Identity | `sample_uid` | `scene_uid` |
| Model dependency | Training target | Never a model input or target |
| Primary consumer | Training loss | DataModelConsole Dashboard |
| Optional downstream | None | Future Active Learning scene selection |
| Prompt dependency | Yes | Only VLM evidence rows |

The implementation may reuse low-level clip encoding and OpenAI-compatible HTTP
utilities, but it must not reuse `ReasoningLabelRecord`, its taxonomy, prompt
version, or training masks. ODD labels must not be embedded in training shards,
collated into model batches, or exposed as model features or losses.

### 3.2 Scene-native ownership

The LabelSet's logical root is a scene. Every observation and event carries
`scene_uid` and is published under that scene's inventory. Time intervals,
camera subjects, and actor subjects refine where a condition occurred inside
the scene; they do not create a sample-level ownership model.

This boundary is intentional:

- the ODD catalog remains stable if sample cadence, history length, future
  horizon, or parser enumeration changes;
- corpus statistics count actual scene time/distance instead of overlapping
  model windows;
- a future Active Learning extension can select coherent scenes with temporal
  context;
- train/validation leakage remains controlled at scene or split-group level;
- no future-aware ODD label can accidentally enter a model input tensor.

If Active Learning is implemented later, KITScenes selects complete scenes. A
future dataset with very long recordings may allow contiguous scene clips only
through a separately versioned selection policy. Individual training samples
are never the ODD selection unit.

### 3.3 Canonical source enum is small; evidence kinds are extensible

Every resolved label has exactly one owning `source`:

```text
map_route | gnss_ins | vlm | image_qc | fusion | can_optional
```

This enum identifies the algorithm class responsible for the result. It is not
an exhaustive list of sensor types. Detailed inputs are recorded as evidence
references, whose `kind` may include:

```text
hd_map | osm | selected_route | pose | gnss | ins | camera | timestamp
object_track | lidar | can | image_metric | vlm_response | bedrock_response
derived_trajectory
```

`object_track` and `lidar` are reserved for a future extension and are not
emitted by the initial KITScenes pipeline. Day phase derived from an absolute
timestamp and location has `source=fusion`. This preserves the requested stable
source enum without hiding the actual inputs.

### 3.4 Deterministic sources precede semantic inference

The source precedence is semantic, not a universal numeric ranking:

- canonical map/route owns static road topology and planned route action;
- GNSS/INS or CAN owns ego motion and the actually driven trajectory;
- image-quality algorithms own measurable pixel/signal defects;
- VLM owns semantic visual appearance that is not available deterministically;
- fusion owns labels that require several sources, temporal tracks, or conflict
  resolution;
- optional CAN corroborates motion and vehicle-state labels but is never a
  required cross-dataset dependency.

A VLM can provide fallback evidence for a map-derived field only when the
ontology explicitly allows it. A fallback remains `source=vlm`; it is not
presented as map truth.

### 3.5 Label values never encode missingness

Every record has one of four statuses:

```text
valid | unavailable | not_observable | ambiguous
```

No enum receives generic `unknown`, `missing`, or `unobservable` values.
Existing domain values such as `not_applicable` and `no_response_required` are
valid semantic outcomes and are not aliases for status.

### 3.6 Sidecar publication preserves dataset immutability

An ODD LabelSet references:

- dataset name and immutable source revision;
- published dataset version;
- dataset manifest SHA-256;
- complete scene identity digest;
- ontology SHA-256;
- labeler and fusion configuration digests.

Changing a taxonomy, labeler, prompt, threshold, or source input creates a new
LabelSet. It never mutates a published dataset version or previous LabelSet.

### 3.7 Offline retrospective context is allowed and declared

Scene search and dataset statistics are offline tasks. Event segmentation may
therefore use frames or trajectory points after event onset. Every labeler
declares `lookback_ns` and `lookahead_ns`, and VLM evidence records the exact
clip timestamps.

These labels are never model inputs under this design. Their retrospective
context is valid for Dashboard, statistics, evaluation slicing, and selecting
which scenes enter a future training dataset. The selected raw scene is then
processed by the normal training pipeline without injecting ODD fields into the
model batch.

## 4. Goals and Non-Goals

### 4.1 Goals

1. Define a dataset-independent schema for ODD, event, and perception labels.
2. Preserve status, confidence, source, evidence, and provenance per label.
3. Use map/route, GNSS/INS, image QC, VLM, fusion, and optional CAN according to
   their actual strengths.
4. Distinguish planned `odd.route.action` from executed
   `event.ego.maneuver`.
5. Represent labels as temporal intervals and events as stable event instances.
6. Support scene-, camera-, and actor-scoped labels without overloading one
   field.
7. Derive correct duration-, distance-, and scene-weighted corpus statistics.
8. Support fast label search and model-metric slicing in DataModelConsole.
9. Make every result reproducible from immutable inputs and versioned code.
10. Start with KITScenes while requiring adapter conformance for future
    datasets.
11. Preserve all source disagreements for audit and confidence calibration.
12. Avoid additional KITScenes repacks when only labels change.
13. Keep a clean future extension point for Active Learning scene selection
    without making it part of the initial Dashboard milestone.

### 4.2 Non-goals for the first implementation

- Training a new runtime ODD classifier.
- Feeding ODD Labels into AutoE2E.
- Using ODD Labels as direct supervision or attaching them to model samples.
- Replacing the action-relevant Reasoning branch.
- Producing ground-truth model error labels such as false positive, false
  negative, or tracking loss.
- Estimating physical quantities that the available sensors cannot support,
  such as road friction or absolute visibility distance.
- Claiming ground truth solely because a VLM produced a high score.
- Live vehicle labeling or an online 10 Hz labeling service.
- Automatically selecting training scenes or launching training from the
  initial Dashboard milestone.
- Labeling `near_collision` without relative trajectories and a validated TTC
  contract.
- Adding an object detector, tracker, or LiDAR processing pipeline to the
  initial KITScenes implementation.
- Reconstructing semantic map attributes from the raster model-input masks.

## 5. Terminology and Identity

### 5.1 Core terms

- Dataset snapshot: immutable raw-source revision plus parser and publication
  contracts.
- Scene: one continuous, dataset-native recording after adapter normalization.
- Timeline: ordered nanosecond timestamps in the scene's canonical clock.
- Evidence: one source labeler's auditable claim before cross-source fusion.
- Scene label record: the logical root containing one scene's observations,
  events, coverage, and provenance.
- Label observation: one resolved child of a scene record over a half-open time
  interval.
- Event instance: one temporally contiguous occurrence with a stable
  `event_uid`.
- LabelSet: immutable collection of evidence, labels, events, statistics, and a
  manifest.
- Scene selection manifest: a future, optional Active Learning result that
  selects scenes from a LabelSet without copying labels into model samples.
- Subject: the scene, camera, actor track, traffic control, or ego entity to
  which a label applies.

### 5.2 Stable identities

The following identities are mandatory:

```text
dataset_name
dataset_version
dataset_manifest_sha256
scene_uid
```

Where the published dataset already provides them, the pipeline also retains:

```text
source_frame_id
source_timestamp_ns
camera_id
actor_track_uid
```

`scene_uid` is dataset-global and partition-independent. It must not be a Flyte
partition index. `sample_uid` and `split_group_uid` remain dataset/training
identities outside the ODD schema. They may be resolved from a selected
`scene_uid` by downstream data processing, but neither is stored on an ODD
observation or used as ODD ownership.

`observation_uid` is content-derived:

```text
sha256(
  labelset_contract_version,
  dataset_manifest_sha256,
  scene_uid,
  start_timestamp_ns,
  end_timestamp_ns,
  subject,
  label_key,
  source,
  labeler_contract_sha256
)
```

`event_uid` is stable within a LabelSet and derives from scene, event family,
participants, and the final segmented interval. It does not depend on Flyte
task ordering.

## 6. System Architecture

### 6.1 End-to-end flow

```text
                                      immutable dataset manifest
                                                   |
                                                   v
                                    DatasetEvidenceAdapter
                                    | capability manifest
                                    | canonical scene clock
                                    | source references
                                                   |
                  +----------------+---------------+----------------+
                  |                |               |                |
                  v                v               v                v
           MapRouteLabeler  GNSSINSLabeler  ImageQCLabeler    VLMLabeler
                  |                |               |                |
                  +----------------+-------+-------+----------------+
                                           |
                                           v
                                    source evidence rows
                                           |
                                           v
                                Fusion + temporal segmentation
                                 | conflict preservation
                                 | confidence calibration
                                 | event instance building
                                           |
                                           v
                                schema and quality validation
                                           |
                                           v
                           immutable ODD LabelSet publication
                              | evidence.parquet
                              | scene_records.parquet
                              | observations.parquet
                              | events.parquet
                              | statistics.parquet
                              | manifest.json
                                           |
                              DataModelConsole
                    +----------------------+----------------------+
                    |                      |                      |
                    v                      v                      v
             ontology catalog      scene statistics       search + playback
                                                                  |
                                                         optional future
                                                         Active Learning
```

### 6.2 Module boundaries

The proposed implementation boundaries are:

```text
Model/odd_labeling/
  contracts.py
  ontology.py
  adapters/
  labelers/
  fusion/
  temporal/
  quality/
  publication/

Platform/pipelines/
  odd_labeling_tasks.py
  odd_labeling_workflows.py

Tools/DataModelConsole/
  ODD LabelSet materialization, APIs, statistics, and search UI
```

`Model/odd_labeling` contains offline data processing and must not be imported
by `Model/model_components`. The Flyte layer orchestrates typed, local-testable
functions; it does not contain label semantics. The labeling workflow is a
standalone dataset operation and is not called by any training workflow.

### 6.3 Reuse of navigation contracts

The pipeline consumes, but does not replace:

- `NavigationMap`;
- `NavigationRoute`;
- `navigation_meta.json`;
- `scene_navigation.json`;
- route quality and provenance.

KITScenes map/route evidence comes from the canonical Lanelet2-derived
navigation artifacts. Future OSM datasets use the same provider-independent
contract. No ODD labeler reads rendered RGB map pixels to infer topology.

## 7. Canonical Data Contracts

### 7.1 Dataset capability manifest

Every adapter first emits one capability manifest:

```text
DatasetCapabilityManifest
  schema_version
  dataset_name
  dataset_version
  dataset_manifest_sha256
  source_revision
  adapter_name
  adapter_version
  scene_inventory_sha256
  timebase:
    canonical_clock
    timestamp_unit
    absolute_time_available
    timezone_resolution_available
  channels:
    cameras: [CameraCapability]
    map: ChannelCapability
    route: ChannelCapability
    gnss: ChannelCapability
    ins: ChannelCapability
    lidar: ChannelCapability
    object_tracks: ChannelCapability
    can: ChannelCapability
  coordinate_frames
  calibration_refs
  known_limitations
```

Each `ChannelCapability` states:

```text
availability: complete | partial | absent
coverage_start_ns
coverage_end_ns
nominal_rate_hz
observed_count
missing_count
source_artifact_sha256
quality_summary
```

An absent channel causes `status=unavailable` for labels that require it. A
present channel that cannot observe a particular interval or subject causes
`status=not_observable`. This distinction is decided from the capability and
per-interval coverage, not guessed by a VLM.

### 7.2 Scene label record

The logical publication unit is:

```text
SceneLabelRecord
  schema_version
  labelset_id
  dataset_name
  dataset_version
  dataset_manifest_sha256
  scene_uid
  scene_start_timestamp_ns
  scene_end_timestamp_ns
  scene_duration_ns
  scene_distance_m
  observation_uids: list[string]
  event_uids: list[string]
  source_coverage
  ontology_coverage
  quality_summary
  provenance
```

Evidence and observations may be physically stored in normalized Parquet tables
for efficient filtering, but every row is a child of exactly one
`SceneLabelRecord`. There is no `SampleLabelRecord` contract.

### 7.3 Evidence record

Source labelers emit evidence before fusion:

```text
LabelEvidence
  schema_version
  evidence_uid
  label_key
  cardinality: single | multi
  values: list[string]
  candidate_values: list[CandidateValue]
  status: valid | unavailable | not_observable | ambiguous
  confidence: float32
  source: map_route | gnss_ins | vlm | image_qc | fusion | can_optional
  scope: LabelScope
  measurements: list[Measurement]
  evidence_refs: list[EvidenceRef]
  provenance: SemanticLabelerProvenance
```

`CandidateValue` contains a value, score, and optional evidence reference. It is
used for ambiguity and audit, not as a resolved value.

`Measurement` contains:

```text
name
value
unit
quality
aggregation
```

Examples include `ego_speed_kph`, signed road curvature, grade, clipped-pixel
ratio, blur score, TTC, or actor count. Raw and continuous measurements remain
available even when a categorical label is also emitted.

### 7.4 Resolved label observation

Fusion emits:

```text
LabelObservation
  schema_version
  observation_uid
  labelset_id
  label_key
  cardinality: single | multi
  values: list[string]
  status: valid | unavailable | not_observable | ambiguous
  confidence: float32
  source: map_route | gnss_ins | vlm | image_qc | fusion | can_optional
  scope: LabelScope
  supporting_evidence_uids: list[string]
  conflicting_evidence_uids: list[string]
  measurements: list[Measurement]
  fusion_provenance: SemanticLabelerProvenance
```

Single-select labels have exactly one value when `status=valid`. Multi-select
labels have one or more values when `status=valid`. Other statuses have an
empty `values` list. Candidate values remain in evidence rows.

The owning source is:

- the original source when fusion performs only validation, smoothing, or
  interval coalescing without changing semantics;
- `fusion` when several sources are required, one source overrides another
  under an explicit rule, or a conflict is resolved.

### 7.5 Label scope

```text
LabelScope
  dataset_name
  dataset_version
  scene_uid
  start_timestamp_ns
  end_timestamp_ns
  anchor_timestamp_ns
  subject_type: scene | ego | camera | actor | traffic_control | route_segment
  subject_id: optional
  camera_ids: list[string]
  coordinate_frame: optional
  spatial_roi: optional
```

Intervals are half-open: `[start_timestamp_ns, end_timestamp_ns)`. A point
observation uses the smallest adapter-supported interval around its timestamp;
zero-duration intervals are prohibited.

`subject_id` is mandatory for `camera`, `actor`, `traffic_control`, and
`route_segment`. An object-level perception label without an actor identity is
invalid. A scene-level summary may report the worst or aggregate condition only
when the ontology defines that aggregation explicitly.

### 7.6 Event instance

Event labels are grouped into instances:

```text
EventInstance
  schema_version
  event_uid
  labelset_id
  scene_uid
  start_timestamp_ns
  end_timestamp_ns
  primary_event_key
  actor_track_uids
  labels: list[LabelObservation]
  phases:
    - phase: onset | active | resolution
      start_timestamp_ns
      end_timestamp_ns
  confidence
  status
  supporting_evidence_uids
  provenance
```

One event instance can carry, for example:

```text
event.vehicle.interaction = [cut_in]
event.interaction.actor = [vehicle]
event.ego.strong_response = hard_brake
event.hazard.response = slow_down
event.outcome = hazard_avoided
```

Overlapping independent events are allowed and receive different `event_uid`
values. Repeated frame detections of one cut-in are not separate events.

### 7.7 Provenance

Every canonical evidence and resolved record retains semantic provenance:

```text
SemanticLabelerProvenance
  labeler_name
  labeler_version
  code_commit
  container_image_digest
  config_sha256
  ontology_sha256
  input_artifact_sha256s
  model_provider: optional
  model_name: optional
  model_revision: optional
  prompt_sha256: optional
  decoding_config_sha256: optional
  lookback_ns
  lookahead_ns
```

VLM provenance must contain the exact model revision, prompt hash, response
schema version, request image timestamps, sampling parameters, and raw response
artifact digest. A mutable model name alone is insufficient.

Wall-clock and orchestration metadata is stored separately:

```text
ExecutionReceipt
  receipt_schema_version
  semantic_partition_sha256
  created_at
  flyte_execution_id
  flyte_task_execution_id
  attempt
  runtime_metrics
```

An execution retry may create a different receipt, but it must produce the same
semantic partition SHA-256. Execution IDs and timestamps never enter
`observation_uid`, `event_uid`, semantic Parquet rows, or `labelset_id`.

## 8. Status, `none`, and Confidence Semantics

### 8.1 Status definitions

| Status | Meaning | Values |
|---|---|---|
| `valid` | The source had sufficient evidence and the label is resolved | Cardinality-valid list |
| `unavailable` | A required channel, contract field, or provider capability is absent | Empty |
| `not_observable` | The channel exists, but this interval/subject cannot be judged | Empty |
| `ambiguous` | Relevant evidence exists but cannot support one canonical result | Empty |

Examples:

- A dataset has no CAN stream: CAN-only evidence is `unavailable`.
- A camera exists but a road sign is fully occluded: the sign state is
  `not_observable`.
- Map says asphalt while visible construction plates cover most of the road:
  surface type may be `ambiguous`, with both evidence rows retained.
- A multi-select workzone label with a clear view and no workzone features is
  `valid` with `values=["none"]`.

### 8.2 Explicit negative rule

For multi-select labels, an observed negative is explicit. The ontology
registry defines a neutral token, normally `none`. It is mutually exclusive
with all positive values.

```text
valid + ["none"]         observed and no condition is present
not_observable + []      the condition could not be checked
unavailable + []         the required source does not exist
ambiguous + []           evidence conflicts or is insufficiently separable
```

`[]` is never a valid multi-select result.

Some proposed candidate sets contain a positive neutral value such as `normal`
or `dry`. These describe an observed condition and do not replace the missing
state. Before implementation, the machine-readable ontology adds `none` only to
a multi-select group that lacks an explicit negative and has a meaningful
absence case. In particular, `odd.dynamic.agent_type_present` requires `none`
for an observed empty road. `odd.road.surface_state` uses `dry` as its observed
neutral and does not add a redundant `none`. The taxonomy table in Section 15
marks normalized additions.

For a single-select label whose domain includes `none`, `not_applicable`, or
`no_response_required`, that token is also a valid observed semantic value.

### 8.3 Confidence

`confidence` is a calibrated estimate in `[0.0, 1.0]` that the record's status
and, for `valid`, its resolved values are correct under the ontology
definition.

- It is not a VLM logit copied directly into the final artifact.
- Deterministic rules may have confidence below 1.0 when map matching, sensor
  quality, or interpolation is uncertain.
- `unavailable` may have confidence 1.0 when the capability manifest
  conclusively records an absent source.
- `not_observable` may have high confidence when geometry proves the subject is
  outside all camera fields of view.
- `ambiguous` confidence describes confidence that the ambiguity classification
  is correct. Candidate scores remain on evidence rows.

Both raw source scores and calibrated final confidence are retained. Confidence
calibration is label/source-specific and versioned.

## 9. Ontology Registry

### 9.1 Registry is the source of truth

The implementation must introduce one machine-readable ontology registry,
proposed as:

```text
Model/odd_labeling/ontology/odd_labeling_v1.yaml
```

Each label definition contains:

```text
key
display_name
description
namespace
cardinality
allowed_values:
  - value
    display_name
    description
neutral_value
allowed_statuses
allowed_sources
authoritative_sources
fallback_sources
subject_types
required_capabilities
temporal_resolution
spatial_context
aggregation_rule
conflict_rule
minimum_duration_ns
hysteresis
quality_tier
introduced_in
```

Code, JSON Schema, prompt response schema, Console facets, and documentation
tables are generated or validated from this registry. Label strings are not
duplicated manually in source code.

### 9.2 Versioning

The registry has an explicit schema version and SHA-256.

- Removing or renaming a key/value, changing meaning, cardinality, scope, or
  neutral semantics is a major version change.
- Adding a value is a minor version change and still creates a new LabelSet.
- Fixing documentation without semantic change is a patch version.
- Value order is not a model-loss ABI, but canonical serialization sorts values
  to make content hashes deterministic.

There is no compatibility shim in the research phase. Consumers must request
an exact ontology version or declare a tested migration.

### 9.3 Quality tiers

Each label/source pair is classified:

```text
certified       passes the frozen human-audit gate
experimental    published and searchable with visible quality warning
disabled        evidence may be retained but no resolved label is published
```

A key can be certified for `map_route` and experimental for `vlm`. Quality is
not assigned only at the key level.

## 10. Temporal and Spatial Semantics

### 10.1 Canonical processing clocks

The pipeline keeps source-native rates and produces interval labels:

| Source | Typical KITScenes input rate | Processing policy |
|---|---:|---|
| GNSS/INS pose | 10 Hz | Derive continuous motion at native timestamps |
| Map/route raster anchor | 2 Hz | Use canonical vectors; evaluate at pose timestamps |
| Camera | 10 Hz reference timeline | QC per frame; semantic clips on selected cadence |
| VLM | 0.25 Hz regular cadence | Front-view evidence, plus event-triggered temporal clips |
| Object tracks | Dataset-dependent | Native track timestamps |
| CAN | Dataset-dependent | Resample only with bounded timestamp skew |

Resolved ODD intervals are coalesced only when key, values, status, source,
subject, and compatible confidence remain unchanged. Original evidence timing
is not discarded.

### 10.2 ODD labels

`odd.*` describes conditions around the ego and selected route. It is not
necessarily constant for a whole scene. Unless a label definition specifies
otherwise, its spatial context is:

- current ego-connected road element;
- selected route up to 100 m ahead;
- ego-local visible/drivable context within the navigation raster;
- current one-second output interval.

The exact lookahead and ROI are versioned per labeler. Statistics are computed
from interval duration or distance, not the number of overlapping training
samples.

### 10.3 Planned route versus actual trajectory

This distinction is normative:

- `odd.route.action` is navigation intent from the selected
  `NavigationRoute`. It answers what the route plans.
- `event.ego.maneuver` is reconstructed from the driven trajectory, lane
  sequence, and map context. It answers what ego actually executed.

A planned left turn followed by a straight or interrupted trajectory therefore
has:

```text
odd.route.action = turn_left
event.ego.maneuver = lane_follow or interrupted event
```

No fusion rule may overwrite one with the other.

### 10.4 Events

Events are retrospective intervals. Detectors first produce candidates at
native timestamps, then an event segmenter applies:

- minimum duration;
- entry and exit hysteresis;
- maximum merge gap;
- actor continuity;
- mutually exclusive event rules;
- onset, active, and resolution boundaries.

Thresholds are key-specific configuration, not hidden constants.

### 10.5 Perception labels

`perception.*` describes observable scene difficulty, not whether a specific
model failed.

- `perception.image.*` is camera-scoped.
- `perception.object.*` requires `subject_type=actor`.
- `perception.fov.state` requires actor and camera geometry or a temporally
  tracked VLM claim.
- `perception.scene.*` is scene/window-scoped.
- A multi-camera aggregate must declare `aggregation_rule`, for example
  `worst_camera`, `front_center_only`, or `any_surround`.

The system does not silently turn front-camera visibility into whole-scene
visibility.

## 11. Dataset Adapter Contract

### 11.1 Adapter interface

Every dataset implements:

```text
DatasetEvidenceAdapter
  describe_capabilities() -> DatasetCapabilityManifest
  list_scenes() -> list[SceneDescriptor]
  open_scene(scene_uid) -> CanonicalSceneEvidence
```

`CanonicalSceneEvidence` provides lazy, typed access to:

```text
scene metadata and canonical clock
camera frames + calibration + frame validity
camera frame inventory mode: capture_timeline | sampled_evidence | unknown
ego poses + covariance/quality where available
canonical NavigationMap
canonical NavigationRoute
CAN signals where available
dataset-native source references
```

An adapter normalizes identity, time, units, frames, and availability. It does
not make semantic ODD decisions.

### 11.2 Adapter invariants

1. Timestamps are `int64` nanoseconds in one declared scene clock.
2. Source timestamps are retained when synchronization creates a canonical
   timestamp.
3. Position frames and transforms are named; unlabeled XY arrays are invalid.
4. Speed is m/s internally; published `ego_speed_kph` is explicitly converted.
5. Missing frames remain missing and are not silently repeated.
6. Duplicate or non-monotonic timestamps fail validation.
7. Map and route carry provider versions and source hashes.
8. Any future actor identities must be stable within a Scene and never
   synthesized from array row order.
9. The adapter reports capability absence before labelers run.
10. All labeler outputs remain owned by the opened `scene_uid`.
11. `dropped_frame` requires `frame_inventory_mode=capture_timeline`.
    A gap in `sampled_evidence` or `unknown` is source-unavailable coverage,
    not evidence that the camera or transport dropped a frame.

### 11.3 KITScenes adapter

The initial KITScenes adapter consumes:

- the pinned source revision and immutable dataset publication manifest;
- 10 Hz pose timestamps, translations, and quaternion-derived yaw;
- the existing derived speed, acceleration, yaw-rate, and curvature signals;
- seven selected camera views and their calibration;
- `scene_navigation.json` with canonical `NavigationMap` and
  `NavigationRoute`;
- `navigation_meta.json` and route-quality fields.

The first implementation does not consume object tracks, detector output, or
LiDAR. It must not assume that every published KITScenes Scene has absolute
civil time, CAN, or every Lanelet2 attribute. The capability audit decides this
from actual artifacts. Unsupported labels become `unavailable`; they are not
guessed to improve coverage.

The KITScenes ODD adapter operates on complete scenes. It does not iterate the
42,667 training samples as labeling units. Existing sample identities are
irrelevant to Dashboard labeling. If Active Learning is added later, the normal
training pipeline enumerates samples only after scenes have been selected.

### 11.4 Future adapters

L2D, NVIDIA PhysicalAI-AV, and future datasets implement the same interface.
Dataset-specific aliases are resolved before label generation:

```text
dataset camera name -> canonical camera role
dataset pose frame -> canonical ENU / ego FLU
dataset map class -> canonical NavigationMap primitive
dataset track class -> ontology actor class
dataset timebase -> canonical nanoseconds
```

An adapter conformance suite uses synthetic fixtures so a new adapter cannot
publish labels until identity, time, units, missingness, and frame transforms
pass.

## 12. Source Labelers

### 12.1 MapRouteLabeler

Inputs:

- canonical `NavigationMap`;
- selected `NavigationRoute`;
- ego pose and map match;
- map layer availability;
- route/map quality and provenance.

Responsibilities:

- road context, class, division, directionality, geometry, junction, lane
  count/type, static structures, and mapped controls;
- planned route action;
- map-supported surface and road-edge evidence;
- route-relevant signal/control association;
- quality-aware abstention and map/VLM conflict evidence.

The labeler uses vectors and attributes, not raster color. OSM and Lanelet2
attribute mappings live in provider adapters, while final ontology mapping is
shared.

Most map/route labels use deterministic topology and geometry. Map-only labels
are gated by the ego-local map match, never by whole-scene Route quality. The
initial local gate requires lateral distance at most 8 m and heading error at
most 75 degrees. WGS84 ego evidence is first transformed with the exact
`MapFrame.projection`; an EPSG-backed UTM map must not be matched with an
equirectangular approximation. Map-only matching treats a centerline as
undirected geometry because provider storage direction is not evidence that
the road shape, junction, or context is unusable. The applied projection and
`heading_semantics=undirected_centerline_geometry` are auditable provenance.
Whole-scene `route.valid`, Route confidence, matched-pose ratio, and
unresolved-discontinuity count remain in provenance for audit, but cannot make
an otherwise usable current road segment unavailable.

`odd.route.action` is the only label in this group that requires the selected
Route. It uses a separately matched local Route segment with lateral distance
at most 10 m, heading error at most 75 degrees, segment confidence at least
0.5, and local continuity from the preceding segment. Unlike Map-only matching,
Route matching remains directed because segment direction is part of intended
action. A discontinuity elsewhere in the Scene is irrelevant. For KITScenes,
the selected Route is reconstructed from the driven trace and estimated
destination rather than supplied planner intent, so every result records
`intent_semantics=reconstructed_from_ego_trace`. It must still remain distinct
from the trajectory-derived `event.ego.maneuver`.

When a locally valid three-arm junction falls in the deterministic T/Y angular
boundary, a task-specific Bedrock Claude Opus 5 resolver may inspect an
ego-local semantic map render plus a structured graph summary. This request
uses the same declared-map projection as deterministic matching before exact
geography is removed. No other junction class and no Route action are sent to
Bedrock. It is a bounded tie-breaker, not a replacement for map matching or
missing map attributes. Its request and acceptance policy are defined in
Section 15.5.

Road context and road type are distinct:

- context describes surrounding functional environment such as urban,
  residential, or industrial;
- type describes the current road hierarchy/use such as primary, residential,
  or service.

If a map has only road class but lacks defensible land-use context, road type
may be valid while road context is unavailable or ambiguous.

Lane count is for ego's current travel direction, not the sum across both
directions. Junction position is relative to the ego and selected route.

### 12.2 GNSSINSLabeler

Inputs:

- metric pose trajectory and timestamps;
- pose quality/covariance when available;
- map-matched road tangent and elevation when available;
- optional corroborating CAN evidence.

Responsibilities:

- raw `ego_speed_kph`;
- speed bin;
- motion state;
- actual maneuver candidates;
- acceleration/deceleration and strong-response candidates;
- road vertical geometry when reliable.

Trajectory derivatives use a versioned, gap-aware smoothing configuration.
Values spanning a gap above the configured threshold are
`status=not_observable`; they are not interpolated through arbitrary outages.

#### Gap and derivative contract

`odd_gnss_ins_policy_v2` freezes the following initial policy:

1. Determine the expected period from the highest declared, non-absent
   GNSS/INS nominal rate. If a dataset adapter has no declared rate, use the
   median positive pose timestamp delta.
2. Define `maximum_gap_ns` as the greater of 500 ms and three expected
   periods. A timestamp delta strictly above this threshold starts a new
   contiguous segment.
3. Compute local metric velocity, speed, acceleration, unwrapped heading
   change, and yaw rate independently inside each contiguous segment. Never
   smooth, differentiate, or integrate across segment boundaries.
4. Require at least three ordered pose samples in a contiguous segment before
   publishing derived kinematics. A shorter segment is
   `status=not_observable`, not a zero-motion segment.
5. Treat the missing interval as beginning one expected period after the last
   sample and ending at the next sample. Every one-second observation interval
   overlapping it publishes `status=not_observable` with no resolved value for
   the KIN-owned speed, motion, maneuver, and strong-response keys.
6. Reset stationary, starting, and strong-response state-machine history after
   a gap. State before an outage cannot establish dwell or event continuity
   after it.

Each observation records the policy version, expected period, effective
maximum gap, stationary epsilon, and dwell thresholds in provenance. Any
change to these rules increments both the GNSS/INS source policy and Flyte task
cache version. A full LabelSet produced under an earlier source policy is not
silently reinterpreted.

#### Speed bin contract

The recommended semantic boundaries are:

```text
stationary: physical zero
creeping:   above zero and below 5 km/h
low_speed:  5 km/h to below 30 km/h
medium_speed: 30 km/h through 60 km/h
high_speed: above 60 km/h
```

Real GNSS/INS speed has noise, so the implementation uses a versioned
`stationary_epsilon_kph` and dwell time:

```text
stationary    abs(speed) <= stationary_epsilon_kph for the dwell interval
creeping      stationary_epsilon_kph < speed < 5
low_speed     5 <= speed < 30
medium_speed  30 <= speed <= 60
high_speed    speed > 60
```

The v2 experimental defaults are `stationary_epsilon_kph=0.5` and
`stationary_dwell_ns=1_000_000_000`. They must be revalidated against the
KITScenes stationary-noise audit before a LabelSet is certified. A threshold
change requires a new source policy version rather than mutating an existing
LabelSet. The continuous `ego_speed_kph`, its quality, aggregation, and source
timestamps are always stored alongside the bin.

#### Actual maneuver contract

`event.ego.maneuver` uses the driven trajectory, with map/route used only for
topology and lane-reference context. It considers:

- integrated heading change;
- signed path curvature;
- lanelet transitions or lane-boundary crossings;
- longitudinal displacement;
- entry/exit from junction and merge topology;
- motion-state and stop intervals.

Route maneuver alone cannot create an actual maneuver event.

### 12.3 ImageQCLabeler

Inputs are decoded camera frames, frame timestamps, and frame validity. The
labeler computes deterministic evidence for:

- exposure and mixed exposure;
- blur candidates;
- contrast;
- black or corrupted frames;
- frozen frames;
- dropped-frame gaps;
- partial/full obstruction candidates;
- glare-related pixel metrics.

Semantic causes such as water droplets, mud, fog, or wet-road reflection may
require VLM evidence. Image QC records the measurable signal and fusion assigns
the semantic label only when supported.

Frozen-frame detection compares temporal image fingerprints and timestamps. A
stationary scene alone must not be mislabeled frozen; motion cues from other
cameras, ego motion, and encoding metadata are used in fusion.

`odd_image_qc_policy_v4` applies these status rules:

1. Build expected camera-frame inventory from the canonical scene clock and
   dataset camera capability manifest only when the adapter declares
   `frame_inventory_mode=capture_timeline`. This inventory is native-timeline
   evidence and is independent of the lower pixel-QC sampling cadence.
2. A declared absent camera is `status=unavailable` for the scene. A missing
   expected frame in an authoritative capture timeline is `dropped_frame`;
   adjacent missing frames are coalesced without hiding their count. When the
   artifact exposes only sampled model-input windows, as KITScenes does, the
   published synchronized sample timestamps form the expected inventory.
   Capture timestamps between those anchors are outside the artifact contract
   and produce no observation. A declared camera missing at a published sample
   anchor is `status=unavailable` with
   `frame_inventory_mode=sampled_evidence`; it never becomes `dropped_frame`
   and never creates a VLM trigger anchor.
3. An object with invalid size or an undecodable image is
   `corrupted_frame`. The scene task continues and records the failure instead
   of discarding every other camera observation.
4. A decoded sampled frame covers only its native frame interval. It never
   labels the time up to the next sampled anchor as normal.
5. Exact repeated content becomes `frozen_frame` only when the ego moved at
   least the configured distance or another synchronized camera changed.
   Repeated content without independent motion evidence is `ambiguous`, not
   frozen.
6. An effectively black decoded image is `black_frame`; dependent exposure,
   blur, contrast, lighting, and glare observations are
   `not_observable` for that interval.
7. `partial_obstruction` and `full_obstruction` require semantic visual
   evidence. Pixel uniformity alone is not enough because sky, walls, and dark
   scenes can look similar. The OpenAI-compatible observer may resolve these
   states, but it cannot override authoritative decode, dropped-frame, or
   timing facts.

Every observation carries the Image QC policy and labeler versions. Pixel
thresholds and frozen-motion thresholds remain experimental until the
stratified human audit passes; changing them requires a new source policy and
Flyte cache version.

### 12.4 VLMLabeler

The VLM labeler handles visual semantics not reliably available from
deterministic sources:

- lane-marking quality;
- road surface appearance/state;
- weather appearance and visibility degradation;
- workzones and temporary controls;
- traffic participants and interaction candidates;
- occlusion, clutter, unusual appearance, and scene complexity.

It calls a pinned OpenAI-compatible multimodal API; the expected production
backend is a road-capable model such as Cosmos. The primary forward camera is
the default semantic observation surface:

- road/environment and traffic/dynamic tasks use front-center at the current
  evidence time;
- forward perception uses a short front-center temporal window and carries
  `camera_id=front_center`;
- Event interaction tasks run only at deterministic trigger anchors and use
  ordered front-center frames describing before, active, and after;
- Image QC may inspect every available camera without sending those images to
  the VLM.

These visual labels describe the observable forward corridor, not unobserved
surround conditions. No Event claim such as cut-in or sudden emergence is
originated from a single image. Hard brake and evasive steer are never
originated by the VLM.

The initial scheduling policy is:

1. one regular semantic anchor every four seconds for ODD/perception coverage,
   with at most 32 combined regular and triggered anchors per Scene;
2. event-triggered three-frame front-center clips around GNSS/INS, QC, map, or
   visual-semantic changes. Trigger context anchors are Event-only evidence
   and never schedule regular ODD/perception bundles;
3. issue three requests per regular anchor: road/environment, traffic/dynamic,
   and forward perception;
4. permit at most one focused refinement per bundle and anchor when evidence
   is ambiguous, not observable, low-confidence, or safety-relevant positive;
5. de-duplicate overlapping requests by content/timestamp digest;
6. retain one explicit evidence row for every attempted field, including
   abstention.

VLM output is schema-constrained. Free-form rationale is an audit field, not a
label. Prompted confidence is raw evidence only and must be calibrated against
human audit.

### 12.5 CANOptionalLabeler

CAN is optional and never part of minimum adapter conformance. When present, it
may provide:

- wheel speed;
- brake pressure or brake-pedal state;
- accelerator position;
- steering angle/rate;
- gear/reverse state;
- turn signal state.

CAN can improve motion, hard-brake, reverse, and evasive-steer evidence. Signal
names and units are normalized by the dataset adapter. A dataset without CAN
still labels supported motion from GNSS/INS and records CAN-only claims as
unavailable.

### 12.6 FusionLabeler

Fusion owns labels that require:

- map plus visual state;
- planned route plus driven trajectory;
- timestamp plus location;
- temporal interaction logic;
- conflict resolution between sources.

The initial KITScenes implementation deliberately does not consume object
tracks, detector output, or LiDAR. Traffic density and actor presence use
task-specific ORV observations; temporal interactions use front-center clips;
motion response uses GNSS/INS. Labels that fundamentally require stable actor
identity, metric relative trajectories, object boxes, or depth remain
`unsupported_missing_source` rather than introducing a detector/tracker
pipeline. Actor range and TTC-based near-collision are examples. A future
LabelSet version may add a separately reviewed geometry backend without making
it a prerequisite for this pipeline.

## 13. Fusion and Conflict Resolution

### 13.1 Fusion is label-specific

There is no global "map beats VLM" function. Each ontology key defines a
fusion policy. Common policies are:

```text
authoritative_only
authoritative_with_fallback
corroborated_union
temporal_state_machine
weighted_consensus
conflict_to_ambiguous
```

Examples:

- `odd.road.type`: map authoritative; VLM fallback only if explicitly enabled.
- `odd.road.surface_state`: VLM primary; image QC corroborates wet/spray cues.
- `odd.traffic_light.state`: VLM state joined to the route-relevant mapped
  signal; neither source is sufficient alone when several signals are visible.
- `event.ego.strong_response`: GNSS/INS and optional CAN temporal state machine;
  VLM cannot create hard brake.
- `perception.image.exposure`: image QC authoritative; VLM may explain but not
  override the metric rule.

### 13.2 Conflict policy

For every conflict:

1. retain all source evidence;
2. determine whether sources refer to the same subject, interval, and spatial
   context;
3. apply the key's explicit priority and freshness policy;
4. lower confidence for stale or low-quality evidence;
5. resolve only if one result satisfies the policy margin;
6. otherwise publish `status=ambiguous` with no resolved value.

Map-versus-visual disagreement is often useful information. A temporary lane
closure, construction plate, or missing sign can make the visual state differ
from the static map. Such disagreement must not be erased as VLM error.

For a multi-select label, fusion may union positive values only when no active
evidence contains the ontology-defined neutral value. A neutral claim such as
`none` or `normal` and a positive/abnormal claim from evidence at the same
authority are conflicting observations, so fusion publishes `ambiguous` and
retains both evidence IDs. A label-specific authoritative-source rule may
resolve the value, but the overridden evidence remains in
`conflicting_evidence_uids`. Fusion never emits a neutral value together with a
positive value.

### 13.3 Temporal smoothing

Smoothing never changes a value across a real transition merely to improve
stability. Each key specifies:

- minimum on/off duration;
- entry and exit thresholds;
- permitted short gap;
- confidence aggregation;
- whether interval boundaries snap to source timestamps.

Map-derived topology can change at a lane boundary. Weather and lighting change
more slowly. Traffic-light state and interaction events change quickly. They
must not share one median filter.

### 13.4 Confidence calibration

Calibration is performed per:

```text
ontology key x source x labeler version
```

Candidate methods include isotonic regression or temperature scaling selected
on a frozen audit set. The manifest records:

- calibration dataset digest;
- method and parameters;
- class counts;
- Brier score or log loss;
- expected calibration error;
- confidence coverage curves.

An uncalibrated source can publish experimental evidence but cannot claim
certified confidence.

## 14. Key Derivation Rules

### 14.1 Road horizontal and vertical geometry

- ODD horizontal geometry uses the current ego-connected road or route
  centerline, not ego steering noise.
- Signed curvature determines `curve_left` or `curve_right`; a deadband and
  minimum arc length define `straight`.
- Vertical geometry uses map elevation where available and GNSS/INS vertical
  profile after quality filtering.
- `crest` and `sag` require a signed grade transition over a minimum baseline.
- Without reliable elevation, vertical geometry is unavailable; a camera VLM
  does not infer precise grade in v1.

### 14.2 Junction and control

- Junction type comes from canonical topology and route connectivity.
- Junction position is a temporal state machine around route distance to the
  junction polygon: `approach -> inside -> exit -> midblock`.
- `midblock` is valid only when the map coverage and current match are valid.
- Junction control is the control applicable to ego's connected movement, not
  any visible sign or signal in the image.
- Multiple applicable controls without a resolvable lane/control association
  produce ambiguity.

### 14.3 Traffic-light state

`odd.traffic_light.state` is scoped to a route-relevant traffic-control subject.
The mapped signal position identifies candidates; VLM supplies visual state;
camera projection and lane/control association disambiguate the controlling
signal.

`not_applicable` is valid when the route movement has no applicable traffic
light. A signal that exists but is not visible is `not_observable`, not
`off` or `not_applicable`.

### 14.4 Density labels

Density thresholds use a versioned ego-local ROI and observable area. In the
initial KITScenes implementation, ORV observes the forward-visible corridor at
one timestamp and returns the ontology bin directly; no detector count or track
duration is manufactured. Provenance records the front-center observation
scope.

- `empty` or `none` is valid only when the ROI is sufficiently observable.
- A front-only image does not establish surround traffic density.
- Map-derived parked areas can support context but cannot establish that
  vehicles are currently present.
- Stop-and-go requires temporal evidence and therefore cannot be established by
  the static forward-view density request alone.

### 14.5 Strong response and near collision

`hard_brake`, `emergency_stop`, and `evasive_steer` require temporal motion
signals. Initial thresholds are proposed by a KITScenes distribution audit and
frozen before labels are generated. At minimum:

- hard brake requires sustained longitudinal deceleration above a configured
  magnitude and duration;
- emergency stop requires a hard response followed by a near-stationary state
  within a configured horizon;
- evasive steer requires an abnormal lateral/yaw response relative to road
  curvature, not merely a planned turn.

`near_collision` is not part of the initial value set. It may be introduced only
when relative actor trajectories, distance, and TTC are available under a
validated coordinate and synchronization contract. The raw TTC definition,
minimum prediction quality, threshold, actor identity, and uncertainty must be
stored.

### 14.6 Event response and outcome

Event response is reconstructed from ego trajectory relative to event onset.
Outcome is not inferred from one final image:

- `normal_completion`: event completed without detected hazard escalation;
- `interrupted`: evidence ended or the scene cut before resolution;
- `hazard_avoided`: a supported hazard was resolved without collision;
- `unresolved`: the event remained active at coverage end;
- `collision`: requires corroborated collision evidence.

An unavailable collision sensor does not turn every event into
`normal_completion`.

## 15. Ontology v1

The values below are the initial ontology. All keys also carry the status,
confidence, source, scope, evidence, and provenance fields defined above.

### 15.1 ODD labels

This table is both the labeling contract and the initial Console ODD catalog.
The Console must expose every key and every allowed value, including candidates
with zero observed scenes, so users can understand what the system is intended
to label instead of seeing only values that happened to appear in one dataset.
For each key, the UI obtains from the ontology registry:

- display name and definition;
- single- or multi-select cardinality;
- complete allowed-value list and per-value descriptions;
- neutral/`none` semantics;
- primary, fallback, and authoritative sources;
- supported subject/scope and temporal semantics;
- current quality tier;
- counts and observable coverage for the selected LabelSet.

Candidate values and observed values are separate concepts in the UI. An
unobserved candidate displays a zero count or unsupported status; it is never
removed from the catalog. Frontend constants must not duplicate this table.

| Key | Type | Allowed values | Primary source |
|---|---|---|---|
| `odd.road.context` | single | `urban`, `suburban`, `rural`, `motorway`, `residential`, `industrial`, `parking` | `map_route` |
| `odd.road.type` | single | `motorway`, `trunk`, `primary`, `secondary`, `tertiary`, `residential`, `service`, `ramp`, `parking_aisle`, `shared_space` | `map_route` |
| `odd.road.division` | single | `divided`, `undivided` | `map_route` |
| `odd.road.directionality` | single | `one_way`, `two_way` | `map_route` |
| `odd.road.horizontal_geometry` | single | `straight`, `curve_left`, `curve_right` | `map_route` |
| `odd.road.vertical_geometry` | single | `level`, `uphill`, `downhill`, `crest`, `sag` | `map_route`, `gnss_ins` |
| `odd.road.junction_type` | single | `none`, `t_junction`, `y_junction`, `crossroad`, `staggered`, `roundabout`, `merge`, `diverge`, `grade_separated` | `map_route` |
| `odd.road.junction_position` | single | `approach`, `inside`, `exit`, `midblock` | `map_route` |
| `odd.road.junction_control` | single | `traffic_light`, `stop_sign`, `yield_sign`, `uncontrolled`, `other` | `map_route` |
| `odd.route.action` | single | `lane_follow`, `straight`, `turn_left`, `turn_right`, `u_turn`, `merge`, `diverge`, `roundabout_enter`, `roundabout_exit` | `map_route` |
| `odd.road.lane_count_bin` | single | `one`, `two`, `three`, `four_plus` | `map_route` |
| `odd.road.lane_type_present` | multi | `general`, `bus`, `bicycle`, `tram`, `emergency`, `turn_only`, `parking`, `shared`, `none` | `map_route` |
| `odd.road.lane_marking_quality` | single | `clear`, `faded`, `missing`, `temporary`, `occluded` | `vlm` |
| `odd.road.surface_type` | single | `asphalt`, `concrete`, `paving_stone`, `gravel`, `unpaved` | `vlm`, `map_route` |
| `odd.road.surface_state` | multi | `dry`, `wet`, `standing_water`, `snow_covered`, `visually_contaminated` | `vlm` |
| `odd.road.edge_type_present` | multi | `curb`, `guardrail`, `solid_barrier`, `temporary_barrier`, `paved_shoulder`, `unpaved_shoulder`, `grass`, `none` | `map_route`, `vlm` |
| `odd.road.special_structure` | multi | `bridge`, `tunnel`, `railway_crossing`, `pedestrian_crossing`, `access_gate`, `none` | `map_route` |
| `odd.road.workzone_state` | multi | `roadworks`, `lane_closure`, `detour`, `cones`, `temporary_barrier`, `temporary_signage`, `none` | `vlm`, `map_route` |
| `odd.traffic_control.present` | multi | `traffic_light`, `stop_sign`, `yield_sign`, `speed_limit_sign`, `temporary_sign`, `traffic_officer`, `none` | `map_route`, `vlm` |
| `odd.traffic_light.state` | single | `red`, `red_yellow`, `yellow`, `green`, `flashing`, `off`, `not_applicable` | `vlm`, `map_route` |
| `odd.environment.day_phase` | single | `day`, `dawn`, `dusk`, `night` | `fusion` from timestamp/location, VLM fallback |
| `odd.environment.sky` | single | `clear`, `partly_cloudy`, `overcast` | `vlm` |
| `odd.environment.precipitation_visual` | single | `none_visible`, `rain`, `snow`, `mixed` | `vlm` |
| `odd.environment.visibility_degradation` | multi | `fog`, `haze`, `precipitation`, `water_spray`, `smoke_or_dust`, `none` | `vlm` |
| `odd.environment.road_lighting` | single | `daylight`, `street_lit`, `unlit`, `tunnel_lit` | `vlm` |
| `odd.environment.glare` | multi | `sun_front`, `sun_side`, `headlight`, `wet_road_reflection`, `none` | `vlm`, `image_qc` |
| `odd.dynamic.traffic_density` | single | `empty`, `low`, `medium`, `high`, `stop_and_go` | `fusion` from VLM/tracks |
| `odd.dynamic.vru_density` | single | `none`, `low`, `medium`, `high` | `fusion` from VLM/tracks |
| `odd.dynamic.parked_vehicle_density` | single | `none`, `low`, `medium`, `high` | `fusion` from VLM/map |
| `odd.dynamic.oncoming_traffic` | single | `absent`, `present` | `fusion` from VLM/tracks |
| `odd.dynamic.agent_type_present` | multi | `passenger_vehicle`, `light_commercial_vehicle`, `heavy_truck`, `bus`, `motorcycle`, `bicycle`, `e_scooter`, `pedestrian`, `wheelchair_user`, `animal`, `emergency_vehicle`, `construction_vehicle`, `none` | `fusion` from VLM/tracks |
| `odd.ego.speed_bin` | single | `stationary`, `creeping`, `low_speed`, `medium_speed`, `high_speed` | `gnss_ins`, `can_optional` |

`odd.ego.speed_bin` always includes the continuous measurement
`ego_speed_kph`.

The ontology normalizes `none` into the multi-select lane type and actor type
groups to enforce the explicit-negative rule. For lane type, `none` is valid
only when the road is observed and none of the listed lane types is represented.
It must not be used when lane attribution is absent. Surface state retains
`dry` as its explicit observed neutral.

### 15.2 Event labels

| Key | Type | Allowed values | Primary source |
|---|---|---|---|
| `event.ego.motion_state` | single | `stopped`, `starting`, `moving`, `creeping`, `accelerating`, `decelerating`, `reversing` | `gnss_ins`, `can_optional` |
| `event.ego.maneuver` | single | `lane_follow`, `turn_left`, `turn_right`, `u_turn`, `lane_change_left`, `lane_change_right`, `merge`, `diverge`, `pull_over`, `pull_out`, `overtake`, `stop` | `fusion` from trajectory/map |
| `event.ego.strong_response` | single | `none`, `hard_brake`, `emergency_stop`, `evasive_steer` | `gnss_ins`, `can_optional`, `fusion` |
| `event.vehicle.interaction` | multi | `cut_in`, `cut_out`, `lead_vehicle_braking`, `vehicle_merging_ahead`, `vehicle_crossing_path`, `oncoming_encroachment`, `parked_vehicle_pull_out`, `door_opening`, `vehicle_yielding`, `vehicle_not_yielding`, `being_overtaken`, `none` | `fusion` from temporal VLM/tracks |
| `event.vru.interaction` | multi | `pedestrian_crossing`, `pedestrian_entering_road`, `pedestrian_waiting_to_cross`, `pedestrian_walking_along_road`, `cyclist_crossing`, `cyclist_merging`, `vru_sudden_emergence`, `occluded_vru_emergence`, `vru_yielding`, `vru_not_yielding`, `none` | `fusion` from temporal VLM/tracks |
| `event.traffic_control.response` | single | `stop_at_red`, `proceed_on_green`, `stop_at_stop_sign`, `yield_at_yield_sign`, `stop_for_crosswalk`, `stop_at_rail_crossing`, `follow_traffic_officer`, `no_response_required` | `fusion` from map/route, VLM, trajectory |
| `event.right_of_way` | single | `ego_has_priority`, `other_has_priority`, `ambiguous_priority`, `not_applicable` | `fusion` from map/route and VLM |
| `event.hazard.type` | multi | `obstacle_on_road`, `debris_on_road`, `blocked_lane`, `wrong_way_vehicle`, `emergency_vehicle_approach`, `collision`, `none` | `vlm`, `fusion` |
| `event.hazard.response` | single | `none`, `slow_down`, `stop`, `obstacle_avoidance`, `lane_change_avoidance`, `yield` | `fusion` from trajectory, GNSS/INS, VLM |
| `event.traffic_flow` | multi | `congestion_entry`, `congestion_exit`, `queue_entry`, `queue_exit`, `stop_and_go`, `road_closure_encounter`, `workzone_entry`, `workzone_exit`, `none` | `fusion` from trajectory, VLM, map/route |
| `event.interaction.actor` | multi | `vehicle`, `pedestrian`, `cyclist`, `motorcycle`, `emergency_vehicle`, `animal`, `static_obstacle`, `none` | `fusion` from VLM/tracks |
| `event.outcome` | single | `normal_completion`, `interrupted`, `hazard_avoided`, `unresolved`, `collision` | `fusion` |
| `event.phase` | single | `onset`, `active`, `resolution` | `fusion` |

Required event rules:

- `odd.route.action=turn_left` means the selected route plans a left turn.
- `event.ego.maneuver=turn_left` means the driven trajectory executes a left
  turn.
- `hard_brake`, `evasive_steer`, and any future `near_collision` are prohibited
  from single-image VLM inference.
- `near_collision` may be added only with relative trajectories and a validated
  TTC contract.
- `event.phase` is stored as phase subintervals on an `EventInstance`; it is not
  copied as an unrelated scene label.

### 15.3 Perception labels

| Key | Type | Allowed values | Primary source |
|---|---|---|---|
| `perception.occlusion.source` | multi | `static_object`, `dynamic_object`, `ego_body`, `weather`, `none` | `vlm` |
| `perception.occlusion.level` | single | `none`, `partial`, `major`, `full` | `vlm` |
| `perception.object.visibility` | single | `fully_visible`, `partially_visible`, `barely_visible`, `not_visible` | future actor geometry; initially disabled |
| `perception.object.scale` | single | `normal`, `small`, `very_small` | future actor geometry; initially disabled |
| `perception.object.range` | single | `near`, `mid`, `far`, `very_far` | future metric actor geometry; initially disabled |
| `perception.fov.state` | single | `centered`, `edge_of_fov`, `truncated`, `entering_fov`, `leaving_fov` | future temporal actor geometry; initially disabled |
| `perception.scene.clutter` | single | `low`, `medium`, `high` | `vlm` |
| `perception.object.overlap` | single | `none`, `moderate`, `heavy` | future actor geometry; initially disabled |
| `perception.visual.contrast` | single | `normal`, `low_contrast`, `silhouette` | `image_qc`, `vlm` |
| `perception.visual.lighting` | multi | `normal`, `backlit`, `deep_shadow`, `high_dynamic_range`, `tunnel_transition` | `image_qc`, `vlm` |
| `perception.visual.glare` | multi | `sun`, `headlight`, `wet_road_reflection`, `none` | `image_qc`, `vlm` |
| `perception.image.exposure` | single | `normal`, `overexposed`, `underexposed`, `mixed` | `image_qc` |
| `perception.image.blur` | single | `none`, `motion_blur`, `defocus_blur` | `image_qc` |
| `perception.image.weather_artifact` | multi | `rain_streak`, `snow_streak`, `water_spray`, `fog_or_haze`, `none` | `image_qc`, `vlm` |
| `perception.image.lens_contamination` | multi | `water_droplet`, `dirt`, `mud`, `condensation`, `none` | `image_qc`, `vlm` |
| `perception.image.frame_status` | single | `normal`, `partial_obstruction`, `full_obstruction`, `black_frame`, `frozen_frame`, `dropped_frame`, `corrupted_frame` | `image_qc` |
| `perception.object.appearance` | multi | `normal`, `unusual_object`, `unusual_pose`, `temporary_object`, `ambiguous_class`, `deceptive_appearance` | future actor-scoped VLM; initially disabled |
| `perception.map_element_condition` | single | `clear`, `occluded`, `faded`, `temporary_conflict`, `visually_missing` | `fusion` from map/route and VLM |
| `perception.scene.complexity` | single | `simple`, `moderate`, `complex`, `extreme` | `vlm` |
| `perception.mixed_traffic` | single | `absent`, `present` | `vlm` |
| `perception.temporary_traffic_control` | single | `absent`, `present` | `fusion` from VLM and map/route |

`perception.visual.lighting` and `perception.object.appearance` use `normal` as
an observed neutral condition. The ontology must prohibit `normal` from
co-occurring with abnormal values. Missing evidence still uses status, never
`normal`.

Object-level keys require a future, explicitly designed stable actor identity
and geometry contract. The initial VLM pipeline does not create
`visual_actor_uid` values and does not present Scene descriptions as object
tracks.

### 15.4 Explicitly excluded labels

| Excluded label | Reason |
|---|---|
| temperature, humidity, wind speed | Requires corresponding environmental sensors |
| rainfall rate in mm/h | Absolute rate is unstable from imagery |
| road friction, road-surface temperature | Requires dedicated measurement or a separately validated estimator |
| visibility distance in meters | Requires a defensible metric-distance protocol |
| illuminance in lux | Absolute illuminance is not recoverable from auto-exposed images |
| ice, black ice | Cannot be confirmed reliably from current imagery |
| sensor calibration drift | Requires comparison to a calibration reference |
| false positive, false negative | Requires model output and ground truth |
| misclassification, tracking loss | Requires model output and ground truth |
| localization failure | Requires an independent position reference |
| near collision | Requires synchronized relative trajectories, distance, and TTC |

These exclusions are schema constraints, not merely prompt instructions. A VLM
response containing an excluded label fails validation and is retained only as
an invalid raw response for audit.

### 15.5 Inference-provider routing

Label acquisition uses the most direct source that can support the semantic
claim. Model calls are selective fallbacks or semantic observers; they are not
used to re-label deterministic facts.

The matrices below use these backend codes:

| Code | Backend | Permitted input | Canonical source |
|---|---|---|---|
| `DMR` | Deterministic map/route resolver | Canonical vectors, graph, attributes, route, map-match quality | `map_route` |
| `BMR` | Bedrock Claude map/route resolver | Privacy-filtered ego-local semantic render and structured topology summary | `map_route` |
| `KIN` | Deterministic GNSS/INS trajectory resolver | Pose, timestamps, quality, derived kinematics | `gnss_ins` |
| `CAN` | Optional deterministic CAN resolver | Normalized vehicle signals | `can_optional` |
| `IQC` | Deterministic image-quality resolver | Decoded camera frames and frame metadata | `image_qc` |
| `ORV` | OpenAI-compatible road VLM | Timestamped vehicle-camera images/clips; expected backend is Cosmos-like | `vlm` |
| `FUS` | Deterministic cross-source fusion/state machine | Validated evidence from the other backends | `fusion` |

`BMR` and `ORV` are different trust and data boundaries:

- `ORV` is the only general model backend that receives real vehicle-camera
  imagery. It handles visual road knowledge, weather appearance, agents,
  interactions, and perception difficulty.
- `BMR` receives no vehicle-camera image. In the initial implementation it
  handles only a locally valid three-arm junction whose branch angles leave T
  and Y as the two deterministic candidates.
- A raw camera frame must not be sent to Bedrock by this design.
- A rendered map must not be presented to `ORV` as if it were a camera frame.
- Provider names and model revisions are provenance; the canonical `source`
  remains one of the six values defined in Section 3.3.

#### Bedrock Claude map/route request

`BMR` uses the Bedrock Converse API with the pinned US inference profile
`us.anthropic.claude-opus-5` and model revision `claude-opus-5`. The request is
task-specific and contains:

```text
MapRouteResolverRequest
  schema_version
  label_key
  allowed_values
  geometry_id
  coordinate_convention: ego FLU, +X forward, +Y left
  semantic_layers:
    drivable_area
    lane_boundaries
    lane_centerlines
    lane_directions
    intersection_polygons
    route_corridor
  ego_local_render_png
  lane_graph_summary
  deterministic_candidates_with_scores
  ego_local_map_match
  required_evidence
```

The render removes latitude/longitude, street names, provider object IDs,
dataset IDs, and unrelated imagery. Primitive IDs in the request are ephemeral
local references. The response is constrained to:

```text
status
value
confidence
cited_primitive_ids
candidate_rejections
```

The Converse request uses a tool schema for this response object and rejects a
plain-text-only completion as a parse failure. Tool choice, inference
configuration, model/inference-profile identity, and request body digest are
recorded in provenance.

`BMR` is called only when:

1. `odd.road.junction_type` has a valid ego-local map match;
2. all required lane-graph primitives are present;
3. exactly three junction arms are available;
4. the deterministic largest-angle gap is in the 140-150 degree boundary, so
   `DMR` returns `ambiguous` with candidates `t_junction` and `y_junction`.

Its result is accepted only when the cited primitives exist and an independent
geometry validator confirms the three-arm invariant. Opus 5 may select only T
or Y from the supplied candidates. It cannot return crossroad, merge, diverge,
roundabout, or a Route action. Otherwise the final status remains `ambiguous`.

`BMR` cannot invent map attributes. It is prohibited for surface material,
weather, lane-marking quality, traffic-light color, actor presence, and other
facts absent from the semantic input.

#### OpenAI-compatible road VLM request

`ORV` calls a pinned OpenAI-compatible multimodal endpoint backed by a
road-capable model such as Cosmos. Requests preserve camera role, timestamp,
ordering, and aspect ratio. They use small task bundles rather than one prompt
for the entire ontology:

```text
RoadVLMRequest
  schema_version
  task_bundle:
    road_appearance | environment | traffic_control
    dynamic_agents | interaction | perception_condition
  allowed_keys_and_values
  scene_uid_hash
  clip_start_timestamp_ns
  clip_end_timestamp_ns
  camera_frames:
    camera_role
    timestamp_ns
    image
    frame_quality
  observability_requirements
```

The response contains one evidence object per requested key, including explicit
status, values, confidence, supporting camera/timestamps, and observability
reason. Free-form values are rejected. The model's reported confidence is raw
evidence and is calibrated before publication.

The client requests JSON Schema structured output when the endpoint supports
it. Otherwise it applies the same schema validator to returned JSON and
abstains on any extra value, missing key, or invalid cardinality.

The client may apply a versioned, deterministic protocol repair before strict
validation. This is wire-format canonicalization, not semantic inference, and
is limited to the following cases:

1. For a scene-scoped observation, replace a non-null `camera_id` with `null`
   only when that ID is both a supplied camera role and explicitly cited in the
   observation's `supporting_cameras`.
2. For a valid multi-select observation, remove its ontology-defined neutral
   value (`none` or `normal`) only when one or more other returned values are
   present and every returned value belongs to the allowed candidate set.
3. Remove unsupported entries from `supporting_timestamps_ns` only when the
   response also cites at least one exact timestamp of an input frame. This
   handles providers that append the requested half-open interval boundary to
   otherwise valid frame citations. The client never snaps, substitutes, or
   invents a timestamp, and an unsupported-only citation remains invalid.

Unknown values, missing or extra keys, malformed JSON, unsupported-only
timestamps, unknown cameras, a wrong camera-scoped identity, missing evidence
citations, and every other schema violation remain invalid. The raw provider
response and its digest are never changed. Each applied repair records its
version, label key, repair kind, and exact before/after values in both the
provider exchange artifact and the resulting observation provenance. The
provider report aggregates repair counts by backend, kind, and label key so
that a high repair rate remains visible as a provider-quality issue rather
than being hidden by a successful validation result.

Requests follow the task-specific camera policy in Section 12.4. Regular
questions use front-center evidence; temporal Event questions use ordered
front-center frames; camera-scoped perception is explicitly scoped to
front-center. A single image is sufficient only for static appearance when the
ontology allows it; it is never sufficient for an interaction or
strong-response event.

For rare, safety-relevant, or low-confidence observations, at most one second
pass per bundle and anchor uses a wider temporal window while keeping
temperature and schema fixed. Agreement raises evidence strength; disagreement
produces `ambiguous` rather than choosing the more convenient answer.

#### Cross-provider acceptance rule

Model inference cannot repair missing source evidence:

- required channel absent: `unavailable`;
- channel present but required area/actor is not visible: `not_observable`;
- valid evidence supports conflicting values: `ambiguous`;
- model output is invalid or cites no required evidence: reject the output and
  retain the pre-inference status.

`FUS` never converts an `ORV` or `BMR` response directly into certified truth.
It applies source quality, geometry checks, temporal consistency, calibration,
and the label-specific rules below.

### 15.6 ODD label acquisition matrix

The ODD spatial default is the ego-connected road and selected route up to 100 m
ahead. The temporal default is a one-second scene interval. A row overrides
these defaults where needed.

| Key | Required evidence | Acquisition and backend | Window / scope | Fallback and status gate |
|---|---|---|---|---|
| `odd.road.context` | Land-use/road-context map attributes or observable surround imagery | `DMR` maps provider attributes; otherwise `ORV` classifies the visible built environment | 1 s, surround views, current road neighborhood | Do not use `BMR` from geometry alone. `ORV` fallback is experimental; no land-use attributes and insufficient view is `not_observable` |
| `odd.road.type` | Current map-matched lane/road with provider class | `DMR` applies a versioned Lanelet2/OSM-to-ontology mapping | Current road segment | Unmapped but present class is `ambiguous`; missing/invalid map is `unavailable`. Camera shape alone does not determine functional class |
| `odd.road.division` | Opposing-direction lane topology, carriageway identity, median/barrier attributes | `DMR` groups carriageways and opposing lanes | Ego-local current segment | Missing separation attributes and insufficient opposing-lane topology are `unavailable`; `BMR` and `ORV` do not guess division |
| `odd.road.directionality` | Traffic-rule direction or one-way map attribute | `DMR` reads canonical traffic rules | Current connected lane | `ORV` fallback requires visible one-way signs/arrows and is experimental. Geometry without direction metadata is `ambiguous`, not two-way |
| `odd.road.horizontal_geometry` | Valid ego-local current-lane centerline | `DMR` computes robust signed curvature over configurable arc length | 30-80 m local arc | A distant Route defect is irrelevant. Invalid local centerline is `unavailable`; curvature near deadband is `straight`; competing signs over the arc are `ambiguous` |
| `odd.road.vertical_geometry` | Reliable map Z or GNSS/INS altitude profile and pose quality | `KIN`/`DMR` estimate smoothed grade and grade derivative | Configured 30-100 m baseline | No reliable elevation is `unavailable`. Neither `BMR` nor `ORV` may infer precise grade |
| `odd.road.junction_type` | Ego-local lane graph connectivity, intersection polygons, levels | `DMR` classifies graph degree, branch angles, roundabout cycles, merge/split, and levels; Opus 5 `BMR` resolves only the 140-150 degree three-arm T/Y boundary | Junction reached within the local map ROI | Missing local graph is `unavailable`; all non-T/Y ambiguity remains `ambiguous` without a model call |
| `odd.road.junction_position` | Valid ego-local junction polygon/distance and local map match | `DMR` state machine: approach, inside, exit, midblock | Native pose timeline, temporally coalesced | Global Route quality and distant discontinuities are ignored; poor local match is `ambiguous`; no local map coverage is `unavailable` |
| `odd.road.junction_control` | Route-relevant regulatory elements and lane association | `DMR` associates mapped controls; `FUS` corroborates with `ORV` | Current/next junction movement | Visible control without lane association is `ambiguous`; no applicable mapped control with complete coverage is `uncontrolled`; missing coverage is `unavailable`; no Bedrock fallback |
| `odd.route.action` | Locally matched selected-Route segment and transition | `DMR` uses transition type, signed heading change, and local continuity; no model fallback | Current locally matched Route segment | No selected Route is `unavailable`; only a discontinuity at the matched segment is `ambiguous`; KITScenes records reconstructed intent in provenance; actual maneuver remains a separate trajectory label |
| `odd.road.lane_count_bin` | Parallel ego-direction lane group and traffic rules | `DMR` counts travel-direction lanes from adjacency and carriageway identity | Current carriageway, 50 m stable span | Partial map lane coverage is `unavailable` or `ambiguous`, never `one` by default; no model fallback |
| `odd.road.lane_type_present` | Lane subtype/regulatory attributes; visible lane symbols | `DMR` maps lane attributes; `ORV` detects visible bus/bicycle/turn/parking markings; `FUS` unions corroborated types | Current carriageway, 1-3 s clip | Complete valid map with no listed type can emit `none`; missing attributes are `unavailable`; map/visual temporary conflict is `ambiguous` or workzone evidence |
| `odd.road.lane_marking_quality` | Road surface around lane boundaries, valid camera exposure | `ORV` evaluates clear/faded/missing/temporary/occluded with `IQC` observability gate | Front and front-side clips, 1-3 s | Boundary outside view or image failure is `not_observable`; mapped boundary plus visually absent marking supports `missing`; temporary markings require temporal/multi-view support |
| `odd.road.surface_type` | Map surface attribute and visible drivable surface | `DMR` uses explicit surface material; `ORV` classifies appearance; `FUS` prefers fresh explicit map unless temporary coverage is evident | Ego lane, 1-3 s clip | No visible road and no map attribute is `not_observable`/`unavailable`; disagreement without temporary evidence is `ambiguous` |
| `odd.road.surface_state` | Visible road surface, precipitation/reflection QC cues | `ORV` classifies dry/wet/water/snow/contamination; `IQC` supplies reflection and visibility metrics; `FUS` enforces temporal consistency | Ego lane ahead, 2-5 s clip | Occluded or saturated road is `not_observable`; `dry` requires sufficient visible road area; uncertain wet versus reflection is `ambiguous` |
| `odd.road.edge_type_present` | Map edge/barrier/shoulder primitives and visible road edge | `DMR` extracts available primitives; `ORV` detects visible curb/barrier/grass/shoulder; `FUS` forms union | Both road edges within local ROI | An edge outside camera/map coverage is not negative. `none` requires complete observable ROI; map/visual temporary barriers are retained as distinct evidence |
| `odd.road.special_structure` | Map bridge/tunnel/crossing/gate tags and visible structure | `DMR` uses explicit structures; `ORV` corroborates tunnel, crossing, or gate appearance | Current road plus 100 m | `BMR` cannot infer an untagged bridge from flat geometry. Missing map and non-visible structure is `not_observable`; conflicts are `ambiguous` |
| `odd.road.workzone_state` | Temporal camera clip and optional fresh roadworks/map data | `ORV` detects cones, barriers, signage, closures, detours; `FUS` combines mapped roadworks and route deviation | 3-8 s multi-view clip | Single isolated cone is low confidence; map-only stale roadworks is not enough for visual subtypes; insufficient road visibility is `not_observable` |
| `odd.traffic_control.present` | Mapped regulatory elements and visible controls | `DMR` lists route-local controls; `ORV` detects signs/lights/officer; `FUS` associates subjects and unions current controls | Current road and next junction, 2-5 s | `none` requires complete mapped/visible coverage; visible but unassociated control remains valid as scene-present but not junction-applicable |
| `odd.traffic_light.state` | Route-relevant mapped signal, camera projection/association, temporal visual state | `DMR` identifies candidate signal; `ORV` reads color/flash/off over several frames; `FUS` resolves controlling subject | Signal-specific, 0.5-2 s at native frames | Signal exists but is occluded/out of FOV is `not_observable`; several plausible signals is `ambiguous`; `not_applicable` requires no applicable signal |
| `odd.environment.day_phase` | Absolute timestamp plus location, or visible illumination/sky | `FUS` uses astronomical solar elevation when time/location are valid; `ORV` is fallback | Scene interval, slowly varying | Timestamp without timezone is acceptable with UTC epoch/location; no absolute time and no observable sky/lighting is `not_observable`; dawn/dusk boundary uses configured solar angles |
| `odd.environment.sky` | Sufficient visible sky pixels | `ORV` classifies clear/partly cloudy/overcast across cameras | 3-10 s, sky-visible cameras | Too little sky, tunnel, or severe exposure is `not_observable`; inconsistent views are fused by visible solid angle or marked `ambiguous` |
| `odd.environment.precipitation_visual` | Temporal images with visible precipitation/road cues | `ORV` classifies none/rain/snow/mixed; `IQC` contributes streak/spray evidence | 3-8 s multi-view clip | `none_visible` means no visible precipitation, not meteorological no-rain; insufficient visibility is `not_observable` |
| `odd.environment.visibility_degradation` | Distant scene contrast, temporal images, QC metrics | `ORV` distinguishes fog/haze/precipitation/spray/dust; `IQC` measures contrast and veiling | 3-8 s, forward and surround | Camera contamination must be separated from atmosphere; no distant view is `not_observable`; several supported causes may coexist |
| `odd.environment.road_lighting` | Illumination, street-light/tunnel context, day phase | `ORV` classifies visible road lighting; `FUS` combines day phase and mapped tunnel | 3-5 s road-facing clip | Street lamps present but visibly off do not imply `street_lit`; clipped/black imagery is `not_observable`; tunnel map plus visible lamps supports `tunnel_lit` |
| `odd.environment.glare` | Saturated bright regions, direction, semantic source | `IQC` detects glare candidates; `ORV` assigns sun/headlight/wet-road cause and direction; `FUS` validates persistence | 1-3 s per camera | Saturation from exposure alone is not glare; cause unresolvable is `ambiguous`; sufficient clean imagery with no glare emits `none` |
| `odd.dynamic.traffic_density` | Observable surround imagery | `ORV` classifies the ontology bin from all available cameras at one timestamp | Current timestamp, ego-local road ROI | Front-only coverage cannot certify surround `empty`; low observable area is `not_observable`; static input does not establish `stop_and_go` |
| `odd.dynamic.vru_density` | Observable sidewalks, crossings, and road imagery | `ORV` classifies the ontology bin from all available cameras at one timestamp | Current timestamp, relevant surround ROI | `none` requires sufficient relevant-area coverage; occluded areas are not negative |
| `odd.dynamic.parked_vehicle_density` | Observable curb/parking imagery and optional map context | `ORV` classifies parked-vehicle density from all available cameras at one timestamp | Current timestamp, curb/parking ROI | Visually uncertain stopped-versus-parked vehicles produce `ambiguous`; no observable parking edge is `not_observable` |
| `odd.dynamic.oncoming_traffic` | Observable surround imagery and optional map direction | `ORV` classifies visible opposing traffic; `FUS` may corroborate lane direction | Current timestamp, forward/side ROI | `absent` requires sufficient coverage; visual heading uncertainty is `ambiguous` |
| `odd.dynamic.agent_type_present` | Observable surround imagery | `ORV` classifies scene-present actor types from all available cameras at one timestamp | Current timestamp, all observable cameras | `none` requires sufficient road/sidewalk coverage; unsupported fine class remains absent from resolved values rather than guessed |
| `odd.ego.speed_bin` | Metric pose timestamps and/or wheel speed | `KIN` derives gap-aware speed; `CAN` corroborates; `FUS` applies frozen epsilon, dwell, and bin boundaries | Native motion timeline, coalesced intervals | Missing/poor pose and CAN is `unavailable`; disagreement above tolerance is `ambiguous`; raw `ego_speed_kph` is always retained |

### 15.7 Event label acquisition matrix

Event rows are produced only after candidate detection and temporal
segmentation. `ORV` event requests must span pre-event, active, and post-event
frames.

| Key | Required evidence | Acquisition and backend | Window / scope | Fallback and status gate |
|---|---|---|---|---|
| `event.ego.motion_state` | Signed speed, acceleration, timestamps, optional gear | `KIN`/`CAN` state machine with priority: stopped, reversing, starting, creeping, accelerating/decelerating, moving | Native timeline with dwell/hysteresis | No reliable motion source is `unavailable`; threshold oscillation is coalesced, not emitted as rapid events |
| `event.ego.maneuver` | Actual ego trajectory, map match/lane transitions, road topology | `KIN` + `DMR` + `FUS` classify heading change, lane crossing, merge/diverge, stop, pull-over/out, and overtake | Complete maneuver, including approach and exit | Planned route alone is prohibited. Invalid trajectory is `unavailable`; competing lane/turn interpretations are `ambiguous` |
| `event.ego.strong_response` | Acceleration, speed, yaw/steering response, road curvature, optional brake/steer CAN | `KIN`/`CAN`/`FUS` apply sustained deceleration and curvature-relative evasive thresholds | Trigger plus configured pre/post seconds | `ORV` cannot originate this label. No quality motion signal is `unavailable`; planned sharp turn must not become evasive steer |
| `event.vehicle.interaction` | Ordered front-center frames with visible vehicle continuity | Focused `ORV` identifies cut-in/out, braking, yielding, door opening, and related visual interactions | Three timestamps describing before, active, and after | Single image is invalid. Occlusion or loss of visual continuity is `not_observable` or `ambiguous`; no metric relative-trajectory claim is made |
| `event.vru.interaction` | Ordered front-center frames with visible VRU continuity and ego-path context | Focused `ORV` identifies crossing, entering, waiting, walking, emergence, and yielding semantics | Three timestamps describing before, active, and after | Single image cannot establish entering/emergence/yielding. Occluded actor without temporal evidence is `not_observable` |
| `event.traffic_control.response` | Applicable control, current control state, ego trajectory | `DMR` identifies control; `ORV` reads visual state/officer; `KIN` provides response; `FUS` matches stop/proceed/yield behavior | Approach through control clearance | Any missing component makes response `unavailable` or `not_observable`; unrelated visible control must not be associated |
| `event.right_of_way` | Static priority rules, route movement, temporal visual actor/control evidence | `DMR` resolves mapped priority; `ORV` supplies visual interaction semantics; `FUS` determines current priority | Interaction/junction interval | Opus 5 is not used for right of way. Insufficient rule/actor evidence is `ambiguous`; no interaction/control context yields `not_applicable` |
| `event.hazard.type` | Temporal front-road imagery and map/route blockage context | `ORV` identifies debris, obstacle, wrong-way vehicle, and emergency vehicle; `FUS` may corroborate with map and ego motion | Hazard onset through clearance | `collision` requires corroborating motion/contact evidence, not one VLM frame; no visible path is `not_observable` |
| `event.hazard.response` | Valid hazard event plus ego trajectory after onset | `KIN`/`DMR`/`FUS` classify slow, stop, yield, lateral avoidance, or lane-change avoidance | Hazard onset through response completion | No valid hazard cannot produce a response; causality uncertain with unrelated maneuver is `ambiguous` |
| `event.traffic_flow` | Ego speed history and temporal visual traffic/workzone context | `KIN` detects ego flow transitions; `ORV` identifies queue, closure, and workzone semantics; `FUS` segments entry/exit | 5-30 s depending on flow state | Stop at a signal is not congestion without visual traffic evidence; Scene boundary before transition yields interrupted context |
| `event.interaction.actor` | Visually identifiable actor class linked to an Event | `ORV` resolves the semantic participant class; `FUS` attaches only Event participants | Same interval as parent Event | No persistent global track ID is created. Scene-present but uninvolved actors are excluded; unknown class is `ambiguous`, not `none` |
| `event.outcome` | Complete event interval, actor/hazard state, ego motion, collision evidence | `FUS` applies outcome state machine after all event evidence | Event end plus resolution horizon | Scene ends early gives `interrupted` or `unresolved`; `normal_completion` requires observable resolution; collision requires corroboration |
| `event.phase` | Final event boundaries and confidence | `FUS` temporal segmenter assigns onset/active/resolution subintervals | Within one `EventInstance` | No independent model call. Missing resolution remains an active event ending as `unresolved`; phases cannot overlap or exceed event bounds |

### 15.8 Perception label acquisition matrix

The initial KITScenes pipeline publishes Scene- and camera-scoped perception
conditions without running a detector or tracker. Object-level keys that
require stable actor geometry remain `unsupported_missing_source`.
Camera-level rows retain `camera_id`.

| Key | Required evidence | Acquisition and backend | Window / scope | Fallback and status gate |
|---|---|---|---|---|
| `perception.occlusion.source` | Per-camera Scene visibility over a short window | `ORV` classifies static/dynamic/ego/weather source for each camera | Camera interval, 1-5 s | No visible reference region is `not_observable`; `none` requires a clear observable view |
| `perception.occlusion.level` | Per-camera Scene visibility over a short window | `ORV` estimates Scene-level none/partial/major/full occlusion for each camera | Camera interval, 1-5 s | This is not an object-visible-fraction measurement. Invalid or fully unobservable input is `not_observable` |
| `perception.object.visibility` | Stable actor identity and visible geometry | Disabled in the initial KITScenes LabelSet | Object-camera interval | `unsupported_missing_source`; a VLM Scene description does not create an actor identity |
| `perception.object.scale` | Stable actor box/mask and camera resolution | Disabled in the initial KITScenes LabelSet | Object-camera interval | `unsupported_missing_source`; no detector output is introduced |
| `perception.object.range` | Metric depth or calibrated 3D actor geometry | Disabled in the initial KITScenes LabelSet | Object interval | `unsupported_missing_source`; VLM-only range is not published as metric evidence |
| `perception.fov.state` | Stable actor geometry across camera frames | Disabled in the initial KITScenes LabelSet | Object-camera sequence | `unsupported_missing_source`; Scene-level camera coverage is handled separately |
| `perception.scene.clutter` | Observable camera imagery and image structure | `ORV` gives semantic clutter; `IQC` contributes edge/quality observability | Per camera, 1-3 s | Severe blur/occlusion is `not_observable`; high traffic alone is not necessarily high clutter |
| `perception.object.overlap` | Stable actor boxes/masks and depth ordering | Disabled in the initial KITScenes LabelSet | Object-camera interval | `unsupported_missing_source`; no detector output is introduced |
| `perception.visual.contrast` | Pixel luminance/chroma and target/background context | `IQC` measures local/global contrast; `ORV` distinguishes low-contrast versus silhouette | Camera or actor-camera interval | Invalid exposure first uses exposure/frame status; no identifiable target for object contrast is `not_observable` |
| `perception.visual.lighting` | Exposure/HDR/shadow metrics and temporal semantic imagery | `IQC` proposes backlight, shadow, HDR, transition metrics; `ORV` assigns semantic lighting; `FUS` enforces mutual exclusion of normal | Per camera, 1-3 s | Invalid frame is `not_observable`; `normal` cannot co-occur with abnormal values |
| `perception.visual.glare` | Saturation/bloom metrics and semantic light source | `IQC` detects candidate regions; `ORV` classifies sun/headlight/wet-road reflection | Per camera, 1-3 s | Exposure clipping without directional bloom is not glare; no cause support is `ambiguous`; clean visible frame can emit `none` |
| `perception.image.exposure` | Decoded pixel histogram and spatial luminance | `IQC` deterministically classifies clipped dark/bright fractions and mixed regions | Per camera frame, temporally coalesced | Decode failure is frame status `corrupted_frame`; exposure is `not_observable` for invalid pixels |
| `perception.image.blur` | Spatial frequency, edge spread, optical flow/ego motion | `IQC` detects blur and distinguishes motion versus defocus using temporal/flow cues; focused `ORV` only adjudicates ambiguous cause | Per camera, 0.5-2 s | Low-texture scene without enough edges is `not_observable`; no blur with adequate texture is `none` |
| `perception.image.weather_artifact` | Temporal camera pixels and environment context | `IQC` proposes streak/spray/veiling patterns; `ORV` classifies artifact type; `FUS` separates lens-fixed contamination | Per camera, 2-5 s | No clean observable region is `not_observable`; atmospheric fog and lens condensation require temporal distinction |
| `perception.image.lens_contamination` | Camera-fixed artifacts persistent across scene motion | `IQC` detects image-coordinate persistence; `ORV` classifies droplet/dirt/mud/condensation; `FUS` validates persistence | Per camera, preferably 3-10 s | A one-frame splash is weather artifact until persistent; insufficient temporal baseline is `ambiguous`; clean lens emits `none` |
| `perception.image.frame_status` | Decoder result, timestamps, frame hashes, authoritative frame inventory, neighboring cameras/ego motion | `IQC` deterministically detects normal/obstruction/black/frozen/dropped/corrupted | Per camera at native rate only with `capture_timeline`; otherwise at sampled evidence intervals | `dropped_frame` requires an authoritative capture inventory; a missing sampled model-input frame is `unavailable`. Stationary scene alone cannot imply frozen; full/partial obstruction semantic boundary may use `ORV` after deterministic candidate |
| `perception.object.appearance` | Stable actor crop plus temporal identity | Disabled in the initial KITScenes LabelSet | Object-camera clip | `unsupported_missing_source`; a Scene-level VLM request does not manufacture stable actor crops |
| `perception.map_element_condition` | Projected mapped element and corresponding camera region | `DMR` identifies expected lane/sign/control element; `ORV` classifies visible condition; `FUS` detects faded/occluded/temporary conflict/missing | Element-camera interval | Projection or association failure is `unavailable`; outside FOV is `not_observable`; visually missing requires valid expected map element and sufficient view |
| `perception.scene.complexity` | Front Scene imagery, topology, controls, and occlusion | `ORV` provides semantic complexity; `DMR` provides topology context; `FUS` applies audited calibration | Front camera set at the current evidence interval | Missing major visual coverage lowers confidence; thresholds are frozen from human audit |
| `perception.mixed_traffic` | Co-present heterogeneous participants in observable surround imagery | `ORV` classifies from all available cameras at one timestamp | Current timestamp, surround Scene | Different actors at unrelated times do not establish mixed traffic; insufficient surround coverage is `not_observable` |
| `perception.temporary_traffic_control` | Visible temporary signs/cones/barriers/officer and static map comparison | `ORV` detects temporary controls; `DMR` provides expected static state; `FUS` resolves present/absent and conflict | 3-8 s road/junction interval | `absent` requires sufficient road/control visibility; static mapped control alone is not temporary; map/visual mismatch supports present only with visual evidence |

### 15.9 Acquisition completeness and KITScenes support

The matrix defines how a label is acquired when the required capabilities
exist. It does not claim that KITScenes supplies every capability.

Before a KITScenes full run, the capability audit produces one row per
key/backend:

```text
supported_certified
supported_experimental
unsupported_missing_source
disabled_pending_audit
```

The Console Ontology tab displays this support state next to every candidate.
Unsupported labels remain visible in the ontology but publish
`status=unavailable`; they are not routed to a model to manufacture coverage.

The initial expected routing is:

- run `DMR`, `KIN`, and `IQC` wherever their source contracts are available;
- run Opus 5 `BMR` only on the small subset of locally valid three-arm junctions
  that remain T/Y ambiguous after deterministic resolution;
- run `ORV` on task-specific camera clips for visual ODD/perception coverage;
- do not run an object detector, tracker, or LiDAR geometry pipeline; mark keys
  that fundamentally require those sources as `unsupported_missing_source`;
- publish the exact supported/unsupported matrix with the LabelSet.

The KITScenes source audit of the current `map.osm` found 75 Lanelets:

- `location` exists on all 75 (`urban` on 43 and `nonurban` on 32), so
  `odd.road.context` can use explicit provider data;
- `highway`/`road_class` exists on 0 of 75, so `odd.road.type` must remain
  `unavailable` rather than inferring suburban/residential from camera
  appearance;
- `one_way` exists on 18 of 75 and `road_section` on 3 of 75, so
  directionality, lane grouping, and division coverage are necessarily partial.

Navigation contract v2 preserves road class, lane subtype, one-way,
carriageway, median/barrier, predecessor/successor, adjacency, boundary
attributes, and raw provider attributes without reducing them to raster masks.
This makes richer source maps usable end to end, but cannot recover attributes
that are absent in the original KITScenes map.

The published KITScenes camera artifact is a sampled model-input window, not an
authoritative capture timeline. Image QC therefore evaluates decoded sampled
frames and emits no evidence for unpublished capture intervals. It reports
`unavailable` only when a declared synchronized camera is missing at a
published sample anchor. A KITScenes v3.1 real-scene regression initially
produced 378 false `unavailable` rows by comparing sampled cameras with the
10 Hz pose timeline. Policy v4 corrects the inventory boundary; the same scene
now produces 363 valid, 14 ambiguous, one not-observable, zero unavailable,
and zero decode failures.

## 16. Flyte Workflow

### 16.1 Standalone dataset-labeler lifecycle

ODD Labeling is a standalone Dataset Labeler workflow. It is not a stage of
`wf_train_il`, Reasoning label generation, model evaluation, or dataset
training. Its only required domain input is an immutable published dataset
snapshot.

The following boundaries are mandatory:

- training workflows never invoke the ODD workflow;
- the ODD workflow never starts training or evaluation;
- no checkpoint, model version, MLflow run, Model Registry entry, optimizer
  configuration, or train/validation split is required;
- completion publishes an ODD LabelSet and optionally refreshes the Console
  read index;
- failure cannot block an otherwise valid training run;
- generating a new LabelSet does not modify packed training shards;
- Active Learning, if added later, is a separate consumer of a ready LabelSet.

The workflow can be launched manually, on a dataset-publication event, on a
schedule, or from a Console administrative action. Re-running it with a new
ontology, labeler, prompt, or source configuration creates a new immutable
LabelSet without requiring any training activity.

Flyte registers `wf_generate_odd_labelset` with a dedicated
`odd-dataset-labeler` LaunchPlan. This LaunchPlan is independently executable
and is not nested in a training LaunchPlan.

### 16.2 Workflow interface

The proposed top-level workflow is:

```text
wf_generate_odd_labelset(
  dataset_name,
  dataset_version,
  dataset_manifest_uri,
  dataset_manifest_sha256,
  ontology_version,
  ontology_sha256,
  labeler_bundle_version,
  labeler_config_uri,
  labeler_config_sha256,
  enabled_sources,
  road_vlm_provider,
  road_vlm_model_revision,
  road_vlm_prompt_bundle_sha256,
  map_resolver_provider,
  map_resolver_model_revision,
  map_resolver_prompt_bundle_sha256,
  calibration_bundle_sha256,
  publication_prefix,
)
```

Mutable defaults are prohibited for full runs. Image digests, source revisions,
model revisions, ontology hashes, prompt hashes, and configuration hashes are
explicit task inputs.

The workflow has no training-related input. In particular, it does not accept a
checkpoint URI, MLflow run ID, registry model version, training execution ID,
or training hyperparameters.

### 16.3 Task graph

```text
resolve_dataset_snapshot
  -> audit_dataset_capabilities
  -> plan_scene_partitions
  -> map over scene partitions:
       extract_canonical_evidence
       +-> label_map_route_deterministic
             -> select_ambiguous_map_route_requests
             -> label_map_route_bedrock
       +-> label_gnss_ins
       +-> label_image_qc
       +-> label_can_optional
       +-> select_road_vlm_requests
             -> label_openai_compatible_road_vlm
       -> label_fusion_candidates
       -> segment_events
       -> validate_partition
       -> write_partition_receipt
  -> merge_partition_receipts
  -> calibrate_or_apply_frozen_calibration
  -> resolve_and_coalesce_labels
  -> compute_statistics
  -> validate_labelset
  -> publish_labelset
  -> materialize_console_index
```

Source labelers are independently cacheable. Changing a VLM prompt does not
rerun map, GNSS/INS, or image-QC labeling. Changing a fusion policy reuses all
source evidence. Changing a Bedrock topology prompt reruns only the cached
deterministic-ambiguous `BMR` request set, not deterministic map extraction or
camera labeling.

`materialize_console_index` is a downstream read-model update. A successful
LabelSet remains valid even if Console materialization needs to be retried.

### 16.4 Cache keys

Task cache identity includes only semantic inputs:

```text
dataset manifest SHA-256
scene identity digest
source artifact digests
adapter version/config
ontology SHA-256
labeler version/config
OpenAI-compatible VLM model/prompt/decoding hashes where applicable
Bedrock map resolver model/prompt/inference-profile hashes where applicable
fusion version/config
calibration bundle SHA-256
```

Worker count, Flyte partition size, retry count, and resource requests do not
change semantic identity.

### 16.5 Partitioning and idempotency

KITScenes uses one scene per logical partition, consistent with current
navigation processing. Other datasets may group small scenes, but stable
`scene_uid` and `observation_uid` remain partition-independent.

Every task writes to a temporary content-addressed prefix and emits a receipt.
The final publisher:

1. validates all expected scene receipts;
2. rejects duplicate scenes or observations;
3. checks source and ontology consistency;
4. writes immutable artifacts;
5. verifies object hashes;
6. writes `manifest.json` last;
7. atomically updates an optional ready pointer only after validation.

A retry with identical inputs produces the same records and safely reuses
content. Partial output is never discoverable as ready.

### 16.6 Model-inference resource and failure policy

- Deterministic tasks do not depend on the Cosmos endpoint.
- Deterministic map/route tasks do not depend on Bedrock.
- OpenAI-compatible road-VLM and Bedrock map-resolver requests have separate
  bounded concurrency, retry, timeout, and cost budgets.
- Camera requests are routed only to the configured OpenAI-compatible endpoint.
- Bedrock requests contain only privacy-filtered map/route renders and
  structured topology summaries.
- Transport retries are bounded and idempotent by request digest.
- A failed request produces evidence with the appropriate unavailable or
  not-observable status; it does not create `none`.
- A failed `BMR` fallback preserves the deterministic `ambiguous` result; it
  does not make the underlying map channel unavailable.
- A full LabelSet may publish with experimental model-inference gaps only when
  the manifest reports coverage and the frozen completeness gate permits it.
- The existing Cosmos cluster is treated as an external protected dependency;
  this workflow does not modify its infrastructure.
- The workflow invokes Bedrock Runtime only; it does not create or modify
  Bedrock models, inference profiles, IAM roles, or guardrail resources.

## 17. Artifact and Publication Contract

### 17.1 LabelSet layout

The proposed logical layout is:

```text
odd-labelsets/
  dataset={dataset_name}/
    version={dataset_version}/
      labelset={labelset_id}/
        manifest.json
        capabilities.json
        ontology.yaml
        evidence/
          source=map_route/*.parquet
          source=gnss_ins/*.parquet
          source=image_qc/*.parquet
          source=vlm/*.parquet
          source=can_optional/*.parquet
          source=fusion/*.parquet
        scene_records/*.parquet
        observations/*.parquet
        events/*.parquet
        statistics/*.parquet
        quality/
          coverage.json
          conflicts.parquet
          audit_manifest.json
          calibration.json
        receipts/*.json
```

Paths are illustrative; the manifest, not path parsing, is authoritative.

### 17.2 LabelSet ID

`labelset_id` is a content identity derived from:

```text
dataset manifest SHA-256
ontology SHA-256
adapter bundle SHA-256
labeler bundle SHA-256
source configuration SHA-256
fusion configuration SHA-256
calibration bundle SHA-256
semantic output Merkle root
```

The semantic output Merkle root is computed over canonical scene, evidence,
observation, event, statistics, and quality payloads with the `labelset_id`
field omitted. `labelset_id` is then computed once and inserted during final
serialization. This avoids a self-referential hash while still making output
content part of identity. It is not a timestamp or Flyte execution ID.

### 17.3 Parquet requirements

- Explicit Arrow schema; no inferred mixed types.
- Nanosecond timestamps stored as signed int64.
- Dictionary encoding for label keys, values, statuses, and sources.
- Scene-based row groups for selective reads.
- Canonical sorted value lists for multi-select labels.
- No NaN confidence; values must be finite and in range.
- Evidence and label rows sorted deterministically.
- Schema metadata includes ontology and LabelSet identities.

JSONL may be emitted for debugging small smoke runs, but Parquet is the
authoritative full-corpus format.

### 17.4 Scene ownership and physical layout

Parquet normalization does not change the ownership model. `scene_records`
contains one logical root per scene. Evidence, observations, and events are
partitioned and indexed by `scene_uid`; no table contains a canonical
`sample_uid -> label` relation.

The LabelSet therefore remains unchanged when model sample cadence, history,
target horizon, or sample eligibility changes.

### 17.5 Derived evaluation projection

Model analysis may need to ask which scene labels overlap a model evaluation
row. The Console or evaluation service may build an ephemeral derived
projection:

```text
sample_uid
scene_uid
sample_anchor_timestamp_ns
label_observation_uids
overlapping_event_uids
projection_policy_version
```

This projection is not part of the ODD LabelSet, is not an ODD label artifact,
and is never supplied to the model. It is regenerated from a model/dataset
manifest plus scene intervals, and its cache identity includes both manifests.
The join uses interval containment or overlap rules from the ontology.

## 18. Console Dashboard Experience, Statistics, and Search

### 18.1 Correct denominators

For each key/value, publish:

- scene occurrence count;
- event instance count where applicable;
- valid duration and distance;
- value duration and distance;
- valid interval count;
- unavailable duration;
- not-observable duration;
- ambiguous duration;
- confidence distribution;
- source distribution;
- dataset and scene coverage.

Percentages use valid observable coverage as the semantic denominator:

```text
value_fraction = value_duration / valid_duration
```

Missingness is reported separately:

```text
observable_coverage =
  valid_duration / total_eligible_duration
```

Unavailable and not-observable intervals are never counted as negative values.

### 18.2 Avoiding sampling bias

Statistics must not count overlapping 6.4-second training windows as independent
time. Corpus composition is based on the normalized scene timeline. Training
sample counts are not ODD LabelSet statistics.

Publish duration-weighted, distance-weighted, and scene-weighted views. A long
stationary scene must not dominate every reported composition without that
weighting being visible.

### 18.3 Co-occurrence

The statistics artifact supports:

- pairwise ODD co-occurrence;
- ODD x event co-occurrence;
- ODD/perception x model-metric joins;
- rare-combination counts;
- status and source cross-tabs.

Combinations are computed from interval overlap with a versioned minimum
overlap, not by joining arbitrary nearest samples.

### 18.4 Product hierarchy

The initial product is a Dashboard, not an Active Learning application. The
Console adds one ODD workspace, proposed at `/odd`, with these tabs:

```text
Overview | Search | Ontology | LabelSets
```

- Overview answers "what kinds of scenes are in this dataset, and in what
  proportion?"
- Search answers "which scenes have these conditions?"
- Ontology answers "which labels and candidate values can the system produce?"
- LabelSets answers "which labeling version is ready and how complete is it?"

The selected dataset version and LabelSet remain visible and synchronized
across all tabs. A user never sees statistics from one LabelSet and scene
details from another without an explicit version change.

### 18.5 Ontology catalog

The Ontology tab renders the machine-readable registry that defines Section 15.
It is not generated from observed label rows. The catalog is therefore complete
even before a full labeling run or when a candidate value has zero scenes.

The initial catalog groups keys into:

```text
ODD | Events | Perception
```

Each key row shows:

- display name and canonical key;
- concise definition;
- single- or multi-select type;
- all allowed candidate values;
- neutral or `none` behavior;
- primary and fallback sources;
- inference backend where applicable (`deterministic`,
  `openai_compatible`, or `bedrock_claude`);
- applicable subject and temporal scope;
- quality tier;
- support state for the selected dataset;
- scene count and observable coverage when a ready LabelSet exists.

Expanding a key shows per-value definitions, source/fusion rules, status
semantics, and examples. Candidate values with no observations show `0 scenes`;
unsupported values show why they are unavailable. Neither case removes the
candidate from the UI.

The catalog uses ontology API data. Frontend code must not contain a second
hard-coded copy of the keys or values in Section 15.1.

### 18.6 Dataset composition overview

Overview defaults to ODD scene composition. It includes:

1. dataset/LabelSet coverage summary;
2. scene-presence distribution by ODD key and candidate value;
3. duration- and distance-weighted distribution;
4. status coverage (`valid`, `unavailable`, `not_observable`, `ambiguous`);
5. source distribution;
6. common and rare ODD combinations;
7. event and perception summaries as secondary tabs.

The default scene ratio is:

```text
scene_presence_ratio(key, value) =
  count(distinct scene_uid with status=valid and value present)
  / count(distinct scene_uid with at least one valid observation for key)
```

The UI always shows numerator, denominator, and observable scene coverage.
Presence ratios for different values may sum above 100% because one scene can
contain several conditions over time. The chart labels this behavior. A
segmented control switches between:

```text
Scene presence | Duration share | Distance share
```

Duration and distance views use the formulas in Section 18.1. Status coverage
is displayed next to value distribution so a high value percentage over a
small observable subset cannot be mistaken for broad dataset coverage.

Scene-presence ratios include a 95% Wilson confidence interval using scenes as
the independent units. Duration and distance views use a scene-clustered
bootstrap interval. Frame or interval rows must not be treated as independent
observations because adjacent labels are temporally correlated.

For single-select labels, an optional dominant-scene view assigns each scene the
value with the greatest valid duration and sums to 100%. It is visibly labeled
as a derived view and never replaces scene presence.

Selecting a chart segment opens Search with the exact dataset, LabelSet,
key/value, status, and weighting encoded in the URL.

### 18.7 Scene search

Search provides a structured query builder rather than free-form text. One
predicate contains:

```text
namespace/key
value or values
status
source
minimum confidence
minimum occurrence duration
camera/actor scope where applicable
```

Predicates support AND/OR groups. Initial required examples are:

```text
odd.environment.road_lighting = unlit
AND odd.road.junction_type = crossroad

event.vru.interaction contains pedestrian_crossing
AND perception.occlusion.level in [major, full]

odd.route.action = turn_left
AND event.ego.maneuver != turn_left
```

Key and value menus are sourced from the Ontology catalog. Every candidate
value is visible with its current scene count, including zero. Status and source
filters are first-class; a user can search for unavailable or ambiguous
coverage instead of only valid values.

Value predicates default to `status=valid`. In particular, `value != X` means a
different valid value and does not match unavailable, not-observable, or
ambiguous records unless the query explicitly requests those statuses.

Results are scene rows, not sample rows. Each result shows:

- scene identifier and playback thumbnail;
- matched key/value summary;
- status, confidence, and source;
- matched interval count and total duration;
- first matched timestamp;
- relevant event/actor summary;
- a command to open playback at the matched interval.

Pagination and sorting are stable. Supported sort modes include confidence,
matched duration, scene duration, and recording time. Query state is
deep-linkable and can be restored from the URL.

### 18.8 Scene detail and playback

The existing Scene page keeps Reasoning and ODD semantically separate while
making both easy to inspect. The required vertical order is:

```text
scene media and playback controls
trajectory/navigation/model overlays
Reasoning Labels
ODD Labels
remaining diagnostics
```

The ODD Labels section appears directly below Reasoning Labels, with no
unrelated section between them. It is an unframed page section, not a card
nested inside the Reasoning panel.

The section has:

```text
Current time | Whole scene
ODD | Events | Perception
```

Current time displays observations active at the playback timestamp. Whole
scene displays value presence, duration, and status coverage across the scene.
Each row shows key, value, status, confidence, source, and active interval.
Clicking a key opens the same ontology definition and complete candidate list
used by the Ontology tab.

The scene timeline renders:

- ODD state intervals;
- event onset, active, and resolution phases;
- camera/actor-scoped perception intervals;
- ambiguous and not-observable spans with distinct non-value styling.

Selecting an interval seeks playback to its start. Playback updates Current
time labels without shifting page layout. Planned `odd.route.action` and actual
`event.ego.maneuver` are shown as separate rows even when their values agree.

`none`, `not_observable`, `unavailable`, and `ambiguous` must be visually
distinct. A missing response must never render as `none`. Evidence details show
supporting/conflicting sources, inference backend/model revision, measurements,
labeler version, and LabelSet provenance without replacing the concise default
view.

### 18.9 LabelSet state and operational visibility

The LabelSets tab lists ontology version, dataset manifest, sources, quality
tier, created time, coverage, and immutable identity. States are:

```text
not_started | running | ready | failed | superseded
```

When labeling is running, the Console shows scene progress and per-source
coverage without exposing partial results as ready. A failed run shows the
failure stage and retains the last ready LabelSet. Switching LabelSets is
explicit; the Dashboard never silently follows a newer experimental run.

The Console may launch or retry the standalone Flyte Dataset Labeler when the
user has operational permission. This action labels the dataset only. It does
not launch training, evaluation, model registration, or Active Learning.

### 18.10 Read APIs and materialization

DataModelConsole materializes search and aggregate projections only. Parquet,
the ontology registry, and the LabelSet manifest remain authoritative.

The initial API surface is:

```text
GET  /api/v1/odd/ontology
GET  /api/v1/odd/labelsets
GET  /api/v1/odd/statistics
POST /api/v1/odd/scenes/search
GET  /api/v1/scenes/{scene_uid}/odd
GET  /api/v1/scenes/{scene_uid}/odd/evidence/{observation_uid}
```

The ontology response includes all candidate values independent of observed
counts. Statistics return numerator, denominator, weighting mode, status
coverage, source coverage, and LabelSet identity. Scene detail returns
coalesced intervals and event phases, not one row per camera frame.

The read index is generation-pinned to the dataset manifest and LabelSet.
Materialization publishes last after complete scene coverage and can be retried
without rerunning labelers.

### 18.11 Empty, loading, and error states

The UI distinguishes:

- no LabelSet has been generated;
- a LabelSet is currently running;
- a ready LabelSet contains no occurrence of one candidate;
- the selected dataset does not support a source/key;
- a key is mostly not observable;
- statistics or search materialization failed.

None of these states is rendered as an empty successful value distribution.
The last ready generation remains readable while a replacement is running.

### 18.12 Console acceptance tests

Focused Playwright coverage verifies:

- Ontology shows every Section 15.1 key and candidate, including zero-count
  values;
- Overview ratios use the displayed numerator and denominator;
- clicking a chart segment opens the equivalent Search query;
- AND/OR search returns scene identities and opens the correct matched time;
- Reasoning Labels precede ODD Labels on desktop and mobile;
- Current time labels follow playback seeking;
- `none`, unavailable, not observable, and ambiguous remain distinct;
- long keys and candidate values do not overflow;
- no horizontal page overflow, browser error, or failed API request;
- LabelSet version remains consistent across Overview, Search, and Scene detail.

Required viewports are at least 1440 by 1000 and 390 by 844.

### 18.13 Optional future Active Learning

Active Learning is not required for the initial Dashboard. A future extension
may consume the ODD LabelSet as a catalog and emit an independent
`SceneSelectionManifest`. It must never convert ODD values into model input
features or per-sample targets.

The initial KITScenes selection unit would be one complete `scene_uid`. A
selection policy may combine underrepresented ODD combinations, event rarity,
perception difficulty, model uncertainty from a separate model-run projection,
and compute/storage budgets.

The training data pipeline would consume only selected scene identities, then
perform normal parsing and sample enumeration from raw scene data. ODD labels
would not enter the loader batch. Existing frozen validation/test scenes and
split groups must be excluded before ranking.

This future work requires a separately reviewed `SceneSelectionManifest`
contract, selection quality metrics, and Console workflow. It does not gate the
Dataset Labeler or Dashboard delivery.

## 19. Quality, Audit, and Governance

### 19.1 Automated validation

Every partition and final LabelSet validates:

1. exact dataset manifest identity;
2. complete expected scene coverage;
3. unique scene, observation, evidence, and event identities;
4. monotonic, bounded intervals;
5. ontology key/value/cardinality conformance;
6. status/value invariants;
7. `none` mutual exclusion;
8. confidence range and finiteness;
9. allowed source and subject scope;
10. evidence references exist and stay within the same LabelSet;
11. event phases are ordered and within the event interval;
12. object labels have actor subjects;
13. camera labels have camera subjects;
14. source artifact hashes match;
15. deterministic reserialization hashes match.

### 19.2 Human audit

The audit set is selected before quality results are viewed and is stratified
by:

- key and value;
- source;
- confidence band;
- status;
- common and rare conditions;
- map/VLM conflicts;
- dataset and camera;
- event boundary cases.

Human review records independent annotations, adjudication, reviewer agreement,
and ontology interpretation questions. Reviewers see source evidence only after
their initial label to reduce anchoring.

The immutable annotation contract is
`odd_human_audit_annotations_v1`. It binds:

- LabelSet ID, audit-manifest SHA-256, and annotation-set ID;
- reviewer count and `draft` or `adjudicated` status;
- per-label evaluation units with scene/key/source/interval, sampling weight,
  prediction, confidence, optional evidence ID, adjudicated reference, and
  reviewer agreement;
- per-event units with matched, spurious, or missed state, predicted/reference
  boundaries, actor-continuity outcome, and sampling weight;
- open and resolved ontology interpretation questions.

Label units may have no predicted evidence but a valid human reference. This is
required to measure false negatives and recall; auditing only emitted evidence
would produce a biased precision-only report. Predicted evidence IDs and
predicted/reference event IDs are unique within an annotation set so no sample
can be counted twice.

`odd_human_audit_results_v1` computes:

- per key/source/value TP, FP, FN, precision, recall, inverse-sampling-weighted
  estimates, and Wilson 95% intervals;
- exact status/value accuracy and status confusion;
- confidence-band accuracy and expected calibration error;
- event precision/recall, onset and offset absolute error, temporal IoU, and
  actor fragmentation/switch error;
- reviewer agreement and unresolved ontology-question counts;
- positive and negative sample sufficiency against the initial 50-example
  target.

Only adjudicated annotations from at least two reviewers produce measured
results. The evaluator always emits `certified=false`; frozen per-family gates,
sample sufficiency, resolved ontology questions, and explicit approval are
separate requirements. Human audit results are post-publication artifacts
bound to an immutable LabelSet, so Dataset Labeler execution remains
independent of manual review.

### 19.3 Initial quality gates

Before a source/key pair becomes certified:

- at least 50 audited valid examples for each material value where corpus
  availability permits;
- at least 50 audited negative/neutral examples for multi-select groups;
- precision and recall with Wilson or bootstrap 95% intervals;
- boundary error for events and temporal states;
- status accuracy for unavailable/not-observable/ambiguous;
- confidence calibration metrics;
- documented failure modes.

Numerical pass thresholds are frozen per label family before the full audit.
Until then, labels are experimental. Sparse values that cannot meet the sample
minimum remain experimental rather than being silently merged or dropped.

### 19.4 Coverage gates

A full LabelSet manifest reports, for every key:

```text
eligible_duration
valid_duration
unavailable_duration
not_observable_duration
ambiguous_duration
attempted_count
successful_count
quality_tier
```

Publication fails on structural incompleteness. Semantic coverage may be below
100% when accurately represented by status; the manifest must not call such a
LabelSet complete without showing the per-key coverage.

### 19.5 Ontology governance

An ontology change requires:

- motivation and examples;
- affected keys/values;
- migration or explicit non-migration decision;
- source and fusion impact;
- Console/statistics impact;
- audit impact;
- a new ontology digest and LabelSet.

VLM prompt edits do not redefine ontology semantics. If a prompt reveals that a
definition is unclear, the ontology is changed first.

## 20. Failure Semantics

| Failure | Required behavior |
|---|---|
| Dataset source channel absent | `unavailable`; continue supported labelers |
| Timestamp gap exceeds bound | `not_observable` for affected interval |
| Map match invalid | Map/route labels unavailable or ambiguous; retain quality |
| Route absent | `odd.route.action` unavailable; actual maneuver may still be valid |
| Camera frame corrupt | QC label valid for corruption; visual semantic labels not observable |
| One camera absent | Camera-scoped unavailable; surround aggregate follows its declared rule |
| VLM transport failure | Explicit failed evidence; no `none` substitution |
| VLM schema failure | Invalid raw response retained; evidence status reflects failure |
| VLM bounded protocol repair | Preserve raw response; record exact repair in exchange, observation provenance, and provider report |
| Map and visual conflict | Preserve both; apply key policy or publish ambiguous |
| Object track discontinuity | End or split actor-scoped event; do not join by row position |
| CAN absent | CAN evidence unavailable; GNSS/INS path continues |
| Partial Flyte partition output | No ready LabelSet publication |
| Duplicate observation/event ID | Fail publication |
| Taxonomy mismatch across partitions | Fail publication |
| Calibration bundle mismatch | Fail resolved-label publication |

## 21. Security, Privacy, and Cost

- Exact geography follows the dataset's existing privacy and Console exposure
  policy. ODD search results must not bypass disabled exact-map access.
- Raw camera clips and VLM responses remain under the same access controls as
  source data.
- Vehicle-camera images are sent only to the configured OpenAI-compatible road
  VLM endpoint. External providers are disabled unless dataset policy explicitly
  permits data transfer. The initial protected Cosmos endpoint is the expected
  backend.
- Bedrock Claude receives no vehicle-camera image, latitude/longitude, street
  name, provider-native map ID, or scene ID. It receives only an ego-local
  semantic map/route render, a privacy-filtered graph summary, and constrained
  candidates.
- Bedrock and OpenAI-compatible request/response artifacts use separate prefixes
  and IAM permissions so routing mistakes fail closed.
- Credentials and endpoint URLs are never stored in LabelSet provenance.
- Request counts, input image counts, failures, latency, model identifiers, and
  estimated cost are published separately for `ORV` and `BMR`.
- Deterministic source labels are generated before VLM scheduling so the VLM is
  not billed for fields that can be resolved without it.
- Re-labeling one source or prompt reuses unaffected evidence through content
  addressing.

## 22. Initial KITScenes Delivery

### 22.1 Phase 0: capability and ontology audit

1. Pin the KITScenes dataset publication manifest and scene inventory.
2. Audit actual availability and quality of map attributes, route semantics,
   absolute timestamps, cameras, and GNSS/INS. Object tracks, detector output,
   and LiDAR are explicitly outside the initial pipeline.
3. Produce corpus distributions for GNSS/INS noise, gaps, speed, acceleration,
   yaw rate, curvature, and map-match quality.
4. Freeze ontology definitions, spatial ROI, temporal cadence, speed epsilon,
   motion/event thresholds, and quality tiers.
5. Build a small human-reviewed golden scene set.

No full VLM run starts before this phase.

### 22.2 Phase 1: deterministic labels

Implement and validate:

- map/route road topology and planned action;
- deterministic ambiguity scores for route action and junction classification;
- GNSS/INS speed, raw speed, motion, and actual maneuver candidates;
- deterministic image quality;
- status and capability records;
- immutable evidence and LabelSet smoke publication.

This phase should already produce useful ODD statistics without Cosmos.

### 22.3 Phase 2: selective model resolvers

1. Build privacy-filtered ego-local map renders and structured summaries for
   three-arm T/Y boundary cases.
2. Validate Bedrock Claude Opus 5 on T/Y golden fixtures, including
   post-response three-arm geometry checks.
3. Define schema-constrained OpenAI-compatible task-specific camera prompts for
   camera-derived road knowledge.
4. Run both providers on separate small stratified KITScenes subsets.
5. Audit per key/value and calibrate confidence separately for `BMR` and `ORV`.
6. Correct ontology, prompt, or deterministic-candidate ambiguities before the
   full run.
7. Run full 0.25 Hz front-center `ORV` coverage plus trigger-only temporal
   Event inference and bounded refinement; run `BMR` only for deterministic
   T/Y ambiguous junction observations.
8. Publish each provider's evidence separately from resolved fusion output.

### 22.4 Phase 3: events and fusion

Implement:

- map/visual control association;
- route-versus-actual maneuver comparison;
- event segmentation and phases;
- traffic, VRU, hazard, workzone, and interaction events supported by the
  audited visual and trajectory capabilities.

Unsupported event families remain unavailable or experimental. Full coverage is
not fabricated from VLM-only still images.

### 22.5 Phase 4: Console Dashboard and evaluation slices

Add:

- complete Ontology catalog with all Section 15 candidates;
- ODD composition statistics with observable coverage;
- scene-presence, duration, and distance ratio modes;
- label/status/source/confidence filters;
- AND/OR scene and interval search;
- event timeline and evidence detail;
- ODD Labels directly below Reasoning Labels on Scene pages;
- route action versus executed maneuver display;
- LabelSet generation state and standalone Flyte launch visibility;
- model ADE/FDE and other metric slicing by LabelSet;
- desktop and mobile playback validation.

### 22.6 KITScenes publication policy

The initial v10 LabelSet binds to the immutable KITScenes v3.1 manifest and
navigation contract v2 artifacts. Creating v3.1 performs one training-free
repack from the audited recovery manifest so lossless provider attributes and
topology reach `scene_navigation.json`; it does not invoke training, evaluation,
Reasoning labeling, or Cosmos. ODD reruns then publish sidecars against v3.1
without another shard repack.

The Dashboard milestone has no training dependency. A later Active Learning
experiment may define and consume a `SceneSelectionManifest` to decide which raw
scenes are included. It must not consume ODD values as model inputs or targets.
The selected scenes are packed through the ordinary dataset pipeline, and the
source LabelSet remains unchanged.

## 23. Expansion to Other Datasets

### 23.1 Minimum adapter tier

A camera + pose dataset can support:

- image QC;
- some VLM ODD/perception labels;
- ego speed/motion when timestamps and metric pose are valid;
- actual trajectory maneuvers with reduced topology confidence.

Map/route labels are unavailable unless a canonical map/route contract exists.

### 23.2 Map-enabled tier

A dataset with Lanelet2, OSM, or another vector map implements a provider
adapter to `NavigationMap`. If a selected route exists, it also emits
`NavigationRoute`. The ODD MapRouteLabeler remains unchanged.

Map formats are not added to the labeler one by one.

### 23.3 Deferred actor-geometry tier

A future, separately scoped actor-geometry pipeline could enable:

- calibrated actor range and scale;
- traffic/VRU density with stronger observability;
- relative motion and interaction events;
- TTC and, after validation, possible near-collision candidates.

This tier is not implemented or scheduled for the initial KITScenes LabelSet.
If later approved, the label schema can remain the same while evidence records
identify the newly introduced geometry source.

### 23.4 CAN-enhanced tier

CAN improves motion-state and strong-response confidence but does not create a
separate taxonomy. Dataset comparisons report which source tier produced each
label so a CAN-rich dataset is not silently treated as equivalent to a
camera-only dataset.

### 23.5 Cross-dataset statistics

Cross-dataset aggregation requires:

- exact ontology version;
- compatible label/source quality tier;
- compatible temporal/spatial definitions;
- compatible status semantics;
- per-dataset coverage shown separately.

The system must not compare raw label percentages when one dataset has 95%
observable coverage and another has 30%.

## 24. Testing Strategy

### 24.1 Contract tests

- JSON/Arrow schema round trip.
- Exact enum and cardinality validation.
- Every ontology key has exactly one acquisition-matrix entry and every matrix
  entry references a real ontology key.
- Status/value and `none` invariants.
- Stable content identities across partitioning.
- Dataset and LabelSet digest checks.
- Provenance completeness.
- Backend/source mapping permits only `BMR -> map_route` and `ORV -> vlm`.

### 24.2 Adapter conformance tests

- Timestamp monotonicity and unit conversion.
- Coordinate-frame transforms.
- Source gap and missing-frame handling.
- Camera role/calibration mapping.
- Stable scene/frame/track identities.
- Capability absence versus interval observability.

### 24.3 Labeler unit tests

- Synthetic map fixtures for every road/junction/route class.
- Synthetic trajectories for speed boundaries, turns, lane changes, stopping,
  reverse, hard brake, and evasive candidates.
- Image fixtures for exposure, blur, black, frozen, dropped, and corrupted
  frames.
- OpenAI-compatible road-VLM parser tests for valid, incomplete, conflicting,
  and excluded outputs.
- Bounded protocol-repair tests prove raw responses remain immutable, repaired
  values remain ontology-valid, repair provenance is complete, and unsupported
  cameras, unsupported-only timestamps, and unsupported values are still
  rejected.
- Bedrock map-resolver fixtures for only the three-arm T/Y boundary; prove
  Route action and all other junction classes never generate a request.
- Post-Bedrock geometry validation rejects unsupported value/primitive pairs.
- Request serialization proves that Bedrock receives no camera bytes or exact
  geography and the road VLM receives no map render.
- CAN normalization and missing-signal tests.

### 24.4 Fusion tests

- Map/VLM agreement and conflict.
- Static map versus temporary visual control.
- Route-planned turn versus actual straight trajectory.
- Several visible signals with one route-relevant control.
- Track discontinuity and actor reassociation rejection.
- `none` versus not-observable.
- Confidence calibration and conflict margin.

### 24.5 Temporal tests

- Half-open interval boundaries.
- Hysteresis and minimum duration.
- Event onset/active/resolution ordering.
- Multiple overlapping events.
- Source gap behavior.
- Determinism under Flyte repartitioning.

### 24.6 End-to-end smoke

The KITScenes smoke set includes:

- straight midblock;
- left and right planned turns;
- actually executed turn;
- junction approach/inside/exit;
- stationary and speed-boundary intervals;
- day/night or low-light where available;
- map/visual disagreement;
- image-quality defect;
- at least one temporal interaction candidate;
- absent capability and not-observable examples.

The smoke run validates artifacts, materialized statistics, Console search,
ontology candidates, Scene-section ordering, and playback before a full VLM
run.

The compiled Flyte graph is also inspected to verify that it contains no
training, checkpoint, MLflow, model-registry, or evaluation node. A synthetic
dataset smoke must publish a LabelSet without any model artifact or training
execution.

## 25. Staged Implementation

### PR 1: Contracts and ontology

- Canonical evidence, label, event, capability, and provenance schemas.
- Machine-readable ontology v1.
- Validators and generated JSON/Arrow schemas.
- Contract and status tests.

### PR 2: Adapter framework and KITScenes adapter

- `DatasetEvidenceAdapter`.
- KITScenes timeline, camera, pose, navigation, and identity integration.
- Capability audit.
- Adapter conformance fixtures.

### PR 3: Deterministic labelers

- Map/route, GNSS/INS, and image-QC labelers.
- Deterministic candidate scores and ambiguity reasons for map/route topology.
- Continuous measurements and temporal smoothing.
- KITScenes deterministic smoke LabelSet.

### PR 4: Selective model-resolver evidence

- OpenAI-compatible ODD temporal clip builder and prompt schema.
- Bedrock Claude privacy-filtered map/route request and geometry validator.
- Provider-specific bounded scheduling, cache, raw response artifacts, and
  parser.
- Stratified audit tooling.

### PR 5: Fusion and events

- Label-specific fusion registry.
- Confidence calibration.
- Event candidate and segmentation state machines.
- Conflict and quality artifacts.

### PR 6: Publication and statistics

- Immutable LabelSet publisher.
- Scene-rooted Parquet schemas and receipts.
- Duration/distance/scene statistics.
- Full validation and ready cutover.

### PR 7: Console

- LabelSet catalog and materialization.
- Complete ontology/candidate catalog.
- Scene-presence, duration, distance, status, and source statistics.
- Structured scene search, event timeline, and provenance.
- ODD Labels directly below Reasoning Labels on the Scene page.
- Standalone Dataset Labeler run state and permitted launch/retry actions.
- Model metric slicing.
- Desktop/mobile Playwright verification.

### PR 8: Full KITScenes run and audit

- Frozen configurations and image/model digests.
- Full evidence and LabelSet publication.
- Human audit and calibration report.
- Coverage, conflict, and cost report.

### Future PR: Optional Active Learning

- Separately reviewed `SceneSelectionManifest`.
- Scene ranking, budget, diversity, and frozen-split safeguards.
- Selection preview and coverage comparison.
- Explicit handoff of scene identities to data preparation without ODD model
  inputs.

## 26. Acceptance Criteria

### 26.1 Contract

- Every label carries value/status, confidence, source, scope, evidence, and
  provenance.
- Every observation/event is owned by one `SceneLabelRecord`; no ODD
  `SampleLabelRecord` exists.
- `none`, unavailable, not observable, and ambiguous are demonstrably distinct.
- Planned route action and actual maneuver are separate keys and derivations.
- Object and camera labels have valid subjects.
- Taxonomy and output schemas are single-sourced and hashed.
- Every ontology key has a reviewed acquisition rule with required evidence,
  backend, scope, fallback, and status gate.
- ODD values are absent from model input and target contracts.

### 26.2 Dataset independence

- KITScenes-specific code exists only in its adapter and provider mappings.
- Shared labelers consume canonical evidence.
- A synthetic second adapter passes the same conformance and labeler tests.
- Missing map, route, track, LiDAR, or CAN channels do not break supported
  labeling.

### 26.3 Reproducibility

- Same immutable inputs and configuration produce byte-equivalent canonical
  records and identical LabelSet identity.
- Repartitioning does not change IDs or labels.
- Every output is traceable to source artifact hashes, code, image, config, and
  Flyte executions.
- Partial or mixed-version output cannot be published as ready.
- OpenAI-compatible and Bedrock model revisions, prompts, decoding settings,
  request digests, and response digests are pinned in provenance.

### 26.4 Quality

- Structural validation passes for 100% of published rows.
- Every key reports valid/unavailable/not-observable/ambiguous coverage.
- Certified labels meet frozen human-audit and calibration gates.
- Experimental labels are visibly marked and never merged into certified
  statistics without filtering.
- Event boundary and actor-continuity errors are measured.
- Bedrock map/route responses pass independent geometry validation.
- Camera-derived model labels pass per-key human audit and calibrated-confidence
  gates before certification.

### 26.5 Product use

- Console Ontology shows every key and candidate value, including zero-count and
  unsupported candidates.
- Console Overview reports scene-presence, duration, and distance ratios with
  explicit numerators, denominators, and observable coverage.
- Console Search can find a scene/time interval by label, value, status, source,
  confidence, and AND/OR combinations.
- Scene detail shows ODD Labels directly below Reasoning Labels and synchronizes
  Current time labels with playback.
- Corpus statistics use correct observable denominators and timeline weighting.
- Console displays LabelSet run/readiness state without exposing partial output
  as ready.
- The Dataset Labeler can run without a training execution, checkpoint, MLflow
  run, or model registry entry and never launches training.
- Model metrics can be projected onto scene/time slices for analysis without
  changing the LabelSet or model inputs.
- The UI distinguishes route intent, actual maneuver, event, and perception
  difficulty.

## 27. Open Decisions to Resolve in the KITScenes Audit

These are empirical configuration decisions, not unresolved architecture:

1. Exact KITScenes support for absolute civil time in the published processing
   contract. Object tracks and LiDAR are not initial dependencies.
2. Lanelet2/OSM attribute coverage beyond the audited `location`, sparse
   `one_way`, and sparse `road_section` fields, especially road class, lane
   count/type, surface, structures, and traffic controls.
3. `stationary_epsilon_kph`, dwell time, derivative smoothing, and source-gap
   thresholds.
4. Curvature, grade, event, and density thresholds.
5. VLM temporal cadence and triggered refinement budget within the frozen
   task-specific camera policy.
6. Initial per-key certification thresholds and audit sample allocation.
7. Whether a future, separately scoped LabelSet should add object-level
   geometry; all such labels remain disabled in the initial implementation.
8. Pinned OpenAI-compatible road-VLM model revision and per-bundle prompt
   versions.
9. Opus 5 topology prompt revisions and any change to the frozen 140-150 degree
   T/Y ambiguity boundary that triggers `BMR`.

Each value is selected from a reproducible audit artifact and frozen before the
full run. None may be selected after inspecting desired model-performance
results.

## 28. References

- `Docs/navigation_input_design.md`, provider-independent navigation and
  KITScenes route contracts.
- `Design/horizon_reasoning_architecture.md`, action-relevant Reasoning Label
  purpose and artifact separation.
- ISO 34503:2023, Road vehicles - Test scenarios for automated driving systems
  - Specification for operational design domain.
- ASAM OpenODD, machine-readable ODD specification concepts.
- ASAM OpenLABEL, scene, object, frame, and stream labeling concepts.
- BSI PAS 1883:2020, operational design domain taxonomy.
- PEGASUS project, six-layer scenario model.
- AWS Bedrock Runtime,
  [Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html).
- OpenAI,
  [Chat Completions API](https://platform.openai.com/docs/api-reference/chat),
  used as the provider-compatible multimodal request boundary.
