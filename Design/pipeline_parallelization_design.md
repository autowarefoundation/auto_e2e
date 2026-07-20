# Design: Dataset-Group-Sharded, Map-Task-Parallel Data Pipeline (#121)

Status: SCOPE UPDATED (2026-07-15) — option-B fan-out remains locked; the active
implementation and execution target is now KitScenes.
Scope: make the AutoE2E data pipeline (ingest → reasoning-label → pack → train →
eval) scale to **all KitScenes train scenes** by fanning each data-prep stage out
across many pods instead of one. Existing L2D and NVIDIA code and design notes
remain in place, but no new L2D/NVIDIA implementation or full run is part of this
milestone.

### 2026-07-15 scope override
This section supersedes the earlier L2D/NVIDIA execution scope elsewhere in the
document:
- **KitScenes is the only active dataset for #121.** Complete its parser contract,
  wire it into the current Flyte workflow, validate scene fan-out, then run the
  full KitScenes train split.
- **L2D is deferred because its full corpus is operationally too large for this
  milestone** (~3 TB / ~100k episodes; the measured plan would spend roughly
  95 hours in repeated ingest batches before training). Keep its code working but
  do not optimize or launch it now.
- **NVIDIA PhysicalAI-AV is also deferred.** Preserve its current parser and
  workflow paths; do not remove or rewrite them while adding KitScenes.
- PR #41 already provides `Model/data_parsing/kit_scenes/`, but the active
  workflow does not reference it. Parser-to-pipeline contract completion is
  therefore the first implementation gate, not a new parser from scratch.
- The immutable source pins for reproducible development/smoke runs are
  `KIT-MRT/KITScenes-Multimodal@6fde0034446669e2ed7235e4c7fe323cd23d599d`
  (tag `v1.0.1`) and KitScenes SDK commit
  `7765cdec5490894266070ab46e23724b58b3da42`. The pinned release is missing one
  of the 534 official train archives. For this milestone, the full pipeline runs
  all 533 available train scenes under the bounded inventory policy in §3.3.1.
- Data prep may fan out across CPU pods. Training remains serial on one GPU;
  a 3–4 day training run is acceptable and DDP is out of scope.

### Decisions locked in review
1. **Distribute data prep, not training.** CPU/memory capacity may be increased
   for ingest/label/pack fan-out. `train_il` remains a single-GPU serial task.
2. **Do NOT co-locate ingest+label+pack** (§3.5-1). Keep them as SEPARATE Flyte
   tasks so each stage retries independently. Rely on **Flyte caching** so a
   re-run skips unchanged ranges (ingest especially).
3. **Flyte caching is mandatory and already present on the L2D fan-out tasks.**
   Extend the same explicit `cache=True` + provenance/version inputs to every
   KitScenes ingest, label, and pack partition so unchanged scenes are skipped.
4. **Do not perform L2D cache cleanup as part of the KitScenes cutover.** Legacy
   L2D artifacts follow the retention flow in §3.4b when L2D work resumes.
5. **Partition size = one KitScenes scene/pod initially.** In the pinned
   `v1.0.1` tree, train archives have p50 3.82 GiB, p95 10.23 GiB, and max
   20.12 GiB. A fixed 10-scenes/pod plan is therefore unsafe. Use measured archive
   bytes/frame counts for any later cost-based grouping. Immediate target: all
   533 archives available from the official 534-scene `train` split at the pinned
   revision.
6. **sample_uid scheme approved:** `l2d-v1-e{episode}-f{frame}`,
   `nv-v1-{clip_uuid}-f{idx}`, and
   `kitscenes-v1-{scene_uuid}-f{frame_idx}`.
7. **KitScenes workflow plumbing is required.** Add `Dataset.KITSCENES` and
   explicit ingest/label/pack branches while retaining the L2D and NVIDIA
   branches unchanged.

---

## 1. Problem statement

### 1.1 Symptom
The L2D measurements below are the evidence that led to option B. They remain
architecturally relevant, but L2D is no longer the corpus selected for this
milestone.

Scaling from 10 → 20 → 50 L2D episodes, every data-prep stage fails in turn with
`OOMKilled (137)`, each at a different point:

| Stage | 10 ep | 20 ep | 50 ep | Failure locus |
|---|---|---|---|---|
| `data_ingest` | OK | OK* | **OOM** | `FlyteDirectory(out_dir)` upload of the whole raw tree buffers in RAM |
| `generate_reasoning_labels` | OK | **OOM** (fixed→OK) | — | 24 concurrent front-clip decoders in one pod |
| `data_processing` (pack) | OK | ? | — | WM-window decode (~300 MB/worker), workers capped at 6 |

\* 20 ep ingest only passes after the hardlink + 48 Gi limit fix.

These are **band-aids**: raising a single pod's memory / lowering its worker count
buys a few more episodes but never reaches 100+. The root cause is architectural:

### 1.2 Root cause
Every data-prep stage is **one Flyte pod** doing **all** the work, with in-pod
`ProcessPoolExecutor` for parallelism. Peak memory and wall-clock therefore scale
with the *total* episode count:
- `data_ingest` (`workflows.py:173`) downloads all requested episodes into one pod
  and returns a single `FlyteDirectory` (`:257`) that both downstream tasks
  re-download whole.
- `generate_reasoning_labels` (`workflows.py:531`) labels all samples in one pod
  (`ProcessPoolExecutor` over `range(n_samples)`, `:649`), ~12 s/teacher-call, so
  100k samples ≈ 15–17 h in one pod, one eviction from losing everything.
- `data_processing` (`workflows.py:286`) packs all samples in one pod, single
  tar writer in the parent (`:449–467`), WM workers capped at 6 by memory.

### 1.3 Original hard blocker: positional `sample_id`
The intended fix was Flyte `map_task`/`@dynamic` fan-out over episode ranges, but
the baseline was unsafe because the reasoning-label ↔ shard JOIN keyed on a
*positional* id that only made sense within a single process that loaded the
exact same episode set:

- Label side: `sample_key = f"s{si:08d}"` where `si` is the index into the
  in-process `_samples` list (`parallel_label.py:86`).
- Pack side: `sample_key = f"s{si:08d}"`, same positional `si`
  (`workflows.py:450`).
- JOIN: `labels_by_id.get(sample_key)` (`workflows.py:461`).
- Cache key: `reasoning_labels_cache/dataset=/teacher=/prompt_version=/{sample_id}.json`
  (`label_cache.py:2–6, 45–46`).

`_samples` is built over the loaded `episodes` subset
(`l2d/dataset.py:_build_sample_index:196–225`, iterating
`sorted(self._episode_ranges.items())`). So `s00000000` in a pod that loaded
episodes [0–9] is a *different physical frame* than in a pod that loaded [10–19].
If two pods label/pack different ranges, the JOIN silently mis-attaches labels and
every cache key collides. L2D/NVIDIA now use global IDs; KitScenes must implement
the same invariant before its first fan-out.

### 1.4 Current blocker: KitScenes parser/workflow contract
PR #41 merged a standalone seven-camera + Lanelet2 parser, but the Flyte path
still only knows L2D and NVIDIA. Before scene fan-out, `KitScenesDataset` must
provide the same pipeline-facing contract used by label and pack:
- partition-independent `sample_uid(idx)` and scene-level
  `split_group_uid(idx)`;
- stable `frame_index`, raw camera-frame access, `get_front_clip(idx)`,
  egomotion/window accessors, geospatial fields, and `projection_spec`;
- construction from an explicit scene-UUID list rooted at one materialized
  partition, without discovering or retaining unrelated scenes;
- raw arrays/images for pack workers rather than only timm-transformed tensors.

The current map heading also reads egomotion channel 2 as yaw, although that
channel is yaw rate. The parser must retain absolute pose yaw separately and use
it for map rasterization before the first smoke run.

---

## 2. Baseline architecture (historical, verified during design)

The diagram below records the single-pod architecture that motivated #121. The
current branch already adds `wf_create_dataset_sharded` for L2D, stage caching,
and multi-partition train/eval. The remaining gap is that this sharded path still
rejects every dataset except L2D and has no KitScenes ingest/label/pack branches.

```
wf_create_dataset(episodes, world_model, reasoning_teacher, prompt_version)   [workflows.py:1515]
  └─ data_ingest(dataset, episodes)  ── 1 pod ──▶ FlyteDirectory(raw)          [:173, returns :257]
       │   L2D: LeRobotDataset(episodes=range(episodes)); copytree→out_dir      [:251-259]
       │   NVIDIA: download clips[:episodes]                                    [:~200]
       │   KitScenes: NOT WIRED — parser exists under Model/data_parsing/kit_scenes/
       ▼
  conditional(reasoning_teacher != "none")                                     [:1524]
    ├─ _pack_with_labels(raw, ...)                                             [:1484]
    │     ├─ generate_reasoning_labels(raw, episodes, teacher, prompt) ─ 1 pod  [:531]
    │     │     builds L2DDataset(episodes) → range(n) → ProcessPool.map        [:649]
    │     │     LabelCache.get_or_compute per sample_key=f"s{si:08d}"           [parallel_label.py:86-96]
    │     │     writes records.jsonl (whole records) + per-sample S3 cache      [:~660]
    │     └─ data_processing(raw, episodes, world_model, reasoning_labels) 1 pod [:286]
    │           builds L2DDataset(episodes) → range(n) → parallel_pack          [:446]
    │           tar member key = f"s{si:08d}.<suffix>"                          [:450]
    │           JOIN reasoning.json by labels_by_id[sample_key]                 [:461-464]
    │           consistency guard: same episodes ⇒ same sample_ids              [:322-332]
    └─ (else) data_processing(...) imitation-only
  ▼ returns ONE FlyteDirectory of shards (train-000000.tar ...)

train_il(shards: List[FlyteDirectory], ...)                                    [:719]
  shard_dirs = [_loader_download_dir(s) for s in shards]                       [:822]
  make_multi_dataset_loader(shard_dirs, ...)   ← ALREADY consumes a LIST       [:826]
```

Key existing capability we exploit: **`train_il` already takes a *list* of shard
dirs** and merges them (`workflows.py:719, 822, 826`; `MergedDatasetLoader`). So
if fan-out produces K shard dirs, training already consumes them with no change.

Per-sample identity already on every sample (this is what the new id uses):
- L2D `L2DSample`: `episode_index: int`, `frame_index: int`
  (`l2d/dataset.py:64-65`, set at `:319-320`; `frame_index = row - ep_start`,
  `:281`, i.e. offset within its own episode — globally stable).
- NVIDIA `_samples`: `(clip_uuid: str, sample_idx: int, ts)`
  (`nvidia_physical_ai/dataset.py:120-124`).
- KitScenes `_samples`: `(scene_id: str, frame_idx: int)`
  (`kit_scenes/dataset.py`). The scene UUID is both the source archive identity
  and the split/fan-out group.

---

## 3. Design

### 3.1 Part A — Global, partition-independent `sample_id` (prerequisite)

Replace positional `f"s{si:08d}"` with an id built from identity the sample
already carries, stable no matter which episodes/clips/scenes a given pod loaded.

Proposed scheme (add a `sample_uid(idx) -> str` method to each parser):
- L2D: `l2d-e{episode_index:06d}-f{frame_index:06d}`
  (both fields exist: `l2d/dataset.py:319-320`).
- NVIDIA: `nv-{clip_uuid}-{sample_idx:06d}`
  (`nvidia_physical_ai/dataset.py:120`).
- KitScenes: `kitscenes-v1-{scene_uuid}-f{frame_idx:06d}`.

Constraint: the uid becomes the WebDataset `__key__` (the part of a tar member
name before the first `.`), so it MUST contain no `.`. Both schemes use only
`-`/hex, so they are safe. `clip_uuid` is a UUID (hex + `-`) — safe.

Rationale:
- `episode_index` is the TRUE lerobot episode index (from the `episode_index`
  column, `l2d/dataset.py:188`), not a subset-relative position.
- `frame_index` is the within-episode offset (`row - ep_start`), independent of
  other episodes.
- So `sample_uid` for a given physical frame is identical whether that frame was
  processed in a full run or in a shard covering only its episode range.

Call-site invariant (already implemented for L2D/NVIDIA):
- `parallel_label` gets `sample_key` from `_DS.sample_uid(si)`.
- `parallel_pack.pack_sample` returns the uid and the parent uses it as the tar
  member prefix instead of a positional `f"s{si:08d}"`.
- Tar member keys become `{sample_uid}.<suffix>`; the loader groups members by the
  WebDataset `__key__` which is the part before the first `.` — **`sample_uid`
  must contain no `.`** (the scheme above uses only `-`, safe). Verify against
  `pre_extracted.py` key grouping.
- Every label/pack artifact uses this uid as its JOIN identity; partition position
  must never enter the key.

**`sample_uid` is a formal identity contract, not a string format** (review pt 2).
Define a typed identity and derive the uid from it, so releases can't collide and
malformed keys are caught at build time:

```python
@dataclass(frozen=True)
class SampleIdentity:
    dataset_namespace: str   # e.g. "<dataset-repo>@<revision>"
    uid_schema_version: str  # "v1"
    group_id: str            # episode index / clip UUID / scene UUID — the SPLIT unit
    frame_id: int
```
- `sample_uid = f"l2d-v1-e{group_id}-f{frame_id:06d}"` (L2D),
  `f"nv-v1-{clip_uuid}-{frame_id:06d}"` (NVIDIA), or
  `f"kitscenes-v1-{scene_uuid}-f{frame_id:06d}"` (KitScenes). Include the
  `uid_schema_version` so a future scheme change is a clean cache/JOIN break, not
  a silent mismatch.
- Validate every uid at generation: `re.fullmatch(r"[A-Za-z0-9_-]+", uid)` and no
  `.`/`/`. Store the raw `episode_index`/`clip_uuid`/`scene_uuid` and source
  revision in the sample's `meta.json` (not only encoded in the tar key) for
  traceability.

**`split_group_uid` — the eval-split unit (review pt 1, the most important fix).**
A SEPARATE id at episode/clip/scene granularity, NOT the per-frame uid:
```python
split_group_uid = f"l2d-e{episode_index:06d}"
# NVIDIA: f"nv-{clip_uuid}"
# KitScenes: f"kitscenes-{scene_uuid}"
split_bucket = blake2b(f"{split_seed}:{split_group_uid}") % 100
```
Rationale: frames within an episode/clip/scene are strongly correlated; a per-frame
`__key__` hash split (the current `pre_extracted._split_bucket`) puts adjacent
frames of the SAME episode into both train and val → evaluation leak, which
silently inflates held-out numbers. Splitting by episode/clip/scene makes train/val
disjoint at the group level. Required invariant + test:
`assert train_group_uids.isdisjoint(val_group_uids)`. → This replaces the
per-`__key__` split in `pre_extracted.py:_split_bucket`; the loader must split on
a `split_group_uid` carried in each sample's meta (add it to the packed
`meta.json`), not on `__key__`.

**Strict decimated JOIN (review pt 2).** Pack remains 10 Hz while Cosmos labels
the deterministic 1 Hz subset. Require exact equality against that expected
subset, not against every packed sample:
```python
assert len(pack_uids) == len(set(pack_uids))   # no dup uids in a shard
assert len(label_uids) == len(set(label_uids))
expected_label_uids = {
    sample_uid for sample_uid, frame_index in packed_identity
    if frame_index % label_stride == 0
}
assert set(label_uids) == expected_label_uids
assert set(label_uids) <= set(pack_uids)
```
Abstain is an explicit label STATE (already modelled — `ReasoningLabelRecord`
carries an abstain/error field), never a missing key for a selected 1 Hz sample.
The other 9 Hz of packed samples intentionally have no `reasoning.json` and are
masked out of the reasoning loss.

### 3.2 Part B — Map-task fan-out per stage

Introduce a **dataset-group partition** as the unit of fan-out. For the active
KitScenes path, a group is one scene UUID and the initial partition is exactly
one scene. L2D ranges and NVIDIA clip lists remain supported representations but
are not implementation targets in this milestone.

**Ingest materializes raw PER PARTITION (option B, chosen 2026-07-13).** Each
partition's `data_ingest` fetches ONLY that partition's raw data from source and
saves it as its OWN `FlyteDirectory`; the label and pack pods for that partition
read that partition's raw dir. This keeps every stage's memory/disk proportional
to `partition` size (not the total corpus, so the whole-corpus ingest OOM is
gone), and crucially avoids re-hitting HuggingFace / re-downloading from source in
every downstream pod — the raw is fetched ONCE per partition, then reused by that
partition's label + pack. Re-try stability is the priority: a failed label/pack
re-reads the already-materialized partition raw (Flyte cache serves the same URI),
rather than re-pulling from HF.
- **KitScenes (active)**: resolve the requested split to an exact scene-UUID
  inventory, then download and extract only each partition's scene archive into
  its own raw `FlyteDirectory`. Downstream tasks open only that materialized
  directory with `scene_ids=partition.group_ids`; they must not re-query or
  download the full gated dataset. The SDK already supports
  `KITScenesDownloader.select_scenes(scene_ids=...)`, and verifies each archive
  against `data/sequence_archives.csv`, so singleton selection/extraction can be
  reused. However, the pinned SDK downloader does **not** expose a Hugging Face
  `revision` and its `hf_hub_download` calls read `main`. The workflow MUST extend
  or wrap it so both the manifest and archive calls pass
  `revision=DatasetSnapshot.source_revision`; using the unmodified downloader
  would silently defeat the source pin. Archive bytes/checksums from that pinned
  manifest enter the deterministic partition plan.
- **L2D**: `data_ingest` (with `group_ids`) builds `L2DDataset(repo_id,
  episodes=partition_eps)` so lerobot pulls only that range from HF, then persists
  it (hardlink copytree, §Phase-0 fix) as the partition's raw `FlyteDirectory`. The
  label/pack pods re-open it with `LeRobotDataset(root=<partition raw>, episodes=…)`,
  which (verified against lerobot v0.5.0 source) takes the cached path and does NOT
  re-hit HF: data is discovered by globbing the parquet that physically exist and
  filtered by `episode_index.isin(episodes)`, and `episode_index` stays GLOBAL, so
  a partition dir holding only its own episodes opens cleanly (no whole-repo meta
  requirement, no off-by-one). INVARIANT (do not break): ingest MUST download the
  videos too (`download_videos=True`, the default) and hardlink `videos/` — lerobot's
  `_check_cached_episodes_sufficient` requires the requested episodes' video files
  to exist on disk, else the offline label/pack pod would try a network re-download
  and fail.
- **NVIDIA**: downloads ONLY the partition's clips via the `physical_ai_av` SDK
  (`workflows.py:200-225`) into the partition's raw dir. Label/pack DISCOVER clips
  from that raw dir (sorted), so the packer and labeler enumerate the same order
  (sample-index JOIN holds). This path is retained but deferred.

**As-built Flyte gap (verified 2026-07-15).** The current
`wf_create_dataset_sharded` is an `@dynamic` Python `for` loop. It creates one
explicit `ingest -> label -> pack` chain per L2D partition; it does not call
`map_task`, rejects non-L2D fan-out, defaults to `max_partitions=512`, and has no
stage-level label concurrency limit. A 533-scene plan cannot pass the current
guard, and raising only that guard would serialize roughly 1,602 task nodes into
one dynamic graph. That is below the observed ~2,000-node Propeller/gRPC failure
region but leaves too little event-size margin and could drive 60 label pods x 6
workers = 360 concurrent teacher calls.

**Target Flyte structure.** Keep the existing task bodies, cache keys, retries,
resources, `group_ids`, and per-partition artifacts, but replace the dynamic loop
with three array nodes. `group_ids` is a `List[List[str]]` of singleton scene-UUID
lists, so the existing generic partition input remains usable:

```python
@dynamic
def wf_create_dataset_sharded(dataset, group_ids, target_cost, world_model, teacher, prompt):
    inventory = resolve_and_validate_inventory(
        dataset, group_ids, max_missing_scenes=1)
    partitions = plan_partitions(inventory, partition_size=1)
    partition_group_ids = [list(p.group_ids) for p in partitions.partitions]

    raws = map_task(data_ingest, concurrency=60)(
        dataset=dataset, episodes=0, group_ids=partition_group_ids)

    labels = None
    if teacher != "none":
        labels = map_task(generate_reasoning_labels, concurrency=5)(
            raw_data=raws, dataset=dataset, episodes=0,
            group_ids=partition_group_ids, label_workers=2, ...)

    shards = map_task(data_processing, concurrency=60)(
        raw_data=raws, reasoning_labels=labels, dataset=dataset, episodes=0,
        group_ids=partition_group_ids, ...)
    return validate_and_publish_manifest(inventory, shards)
```

Each downstream array depends on the completed upstream array, giving an explicit
stage barrier. Ingest and pack can use the namespace-wide 60-pod resource cap,
while label has its own 5-pod cap. Flyte records three mapped nodes instead of
~1,602 explicit dynamic child nodes; failed array elements still retry and cache
independently.

Per-stage detail:

**(1) `data_ingest_range(partition)`** — one pod per partition. Fetches ONLY its
partition's raw data from source (KitScenes: `partition_scene_ids`; L2D: lerobot
`episodes=partition_eps`; NVIDIA: SDK `partition_clips`) and persists it as the
partition's raw `FlyteDirectory`
(hardlink copytree). Memory/disk scale with `partition` size, so the whole-corpus
ingest OOM is gone. Raw is fetched from source ONCE per partition here; downstream
pods reuse this materialized raw (no HF re-hit).

**(2) `generate_reasoning_labels_range(raw, partition)`** — one pod per range.
- Reads its partition's raw dir (no source re-fetch), builds the front-clip parser
  on that slice, labels with the global `sample_uid`.
- Emits one partition-level `records.jsonl`. Cross-run reuse comes from the Flyte
  task cache keyed by immutable inputs and contract versions; no per-sample S3
  cache is reintroduced.
- **Bounded global teacher concurrency (review pt 6).** Total in-flight calls =
  `map_concurrency × label_workers_per_pod` must stay ≤ what the Cosmos endpoint
  (10 replicas) can serve without 429/tail-latency. Set BOTH:
  `map_task(generate_reasoning_labels_range, concurrency=C)` caps concurrent pods,
  and `label_workers_per_pod` caps in-pod parallelism. Start conservative
  (`label_workers_per_pod=2`, `concurrency=5` → ≤10 in-flight ≈ 1/replica); tune
  from measured endpoint batching. NOT "#pods × 12 unbounded" (that = 120 calls on
  10 replicas → retry storm).
- **Teacher retry (review pt 6):** retry only 429/5xx, honour `Retry-After`,
  exponential backoff + jitter, max attempts + max elapsed, fail fast on 4xx.
  A task retry deterministically regenerates its partition-level `records.jsonl`;
  there is no per-uid cache side effect. Currently `openai_compatible.py` has NO
  retry — add it. The >50% abstain guard stays per-range.

**(3) `data_processing_range(raw, labels, partition)`** — one pod per range.
- Reads its partition's raw dir (the same materialized raw the label stage used —
  no source re-fetch), packs it into its OWN shard files, JOINs only its
  partition's labels by uid.
- WM worker cap (6) stays per-pod but now bounds a small range, not the whole set.
- Emits a per-partition shard `FlyteDirectory` + a `ShardPartitionManifest`.

**Combine → a reducer that emits a `DatasetManifest` (review pt 7).** Instead of
returning a bare `List[FlyteDirectory]` (which loses coverage/checksum/split
stats), a final `validate_and_publish_manifest` reducer collects the per-partition
manifests, validates them, and emits ONE `DatasetManifest`:
```json
{"dataset_snapshot": "...", "uid_schema_version": "v1", "shard_schema_version": "v3",
 "partitions": [{"partition_id": "p-000", "shard_uris": ["s3://..."],
                 "sample_count": 1834, "label_count": 184, "sha256": "..."}],
 "expected_source_group_count": 534, "selected_source_group_count": 533,
 "missing_group_ids": ["0aef5c74-debd-67ee-c41a-72bb6c82b221"],
 "internal_split_manifest_digest": "..."}
```
Reducer validations (fail the run if any break): selected-scene partition
coverage is exact with no overlap or gap, the missing set matches the preflight
record and remains within policy, `label_uids == expected_1hz_uids` per partition,
and shard checksums/counts match. Always associate by `partition_id`, never map
output order/list position.
`train_il`/eval take the `DatasetManifest` (or its shard_uris) — the manifest is
the formal artifact; the `List[FlyteDirectory]` train_il already accepts is the
transport underneath.

### 3.3 Partitioning function
`plan_partitions(snapshot, target_cost, max_partitions)` returns a deterministic
`PartitionSpec` list. The active KitScenes policy is deliberately simple:
`partition_size=1`, where the one group is one scene UUID. Set the KitScenes
guard to at least `max_partitions=600`; the current L2D default of 512 rejects the
533-scene available train set.

#### 3.3.1 Source-inventory gate

The official SDK split at commit
`7765cdec5490894266070ab46e23724b58b3da42` lists 534 train scene UUIDs, but the
pinned Hugging Face dataset revision
`6fde0034446669e2ed7235e4c7fe323cd23d599d` (`v1.0.1`) currently contains only
533 train archives:

| Pinned train inventory | Count / size |
|---|---:|
| SDK split UUIDs (expected) | 534 |
| Hugging Face archives (available) | 533 |
| Compressed archive total | 2,439.15 GiB |
| Archive p50 / p95 / max | 3.82 / 10.23 / 20.12 GiB |

The missing archive is
`0aef5c74-debd-67ee-c41a-72bb6c82b221`. It is absent from every Hugging Face
split directory, not merely stored under the wrong split. Therefore the pinned
source cannot yet produce an honest "all 534 train scenes" run.

`resolve_and_validate_inventory` MUST compare the SDK UUID set with the actual
archive UUID set and checksum/size metadata before creating any mapped node. It
accepts a bounded `max_missing_scenes` policy, set to `1` for this run. The known
single missing UUID is recorded in execution metadata and emits a warning; the
remaining 533 archives become the exact selected inventory. The preflight still
fails closed on duplicate, extra, wrong-split, checksum/size mismatch, or more
than one missing archive. This is a complete run over the data actually available
from pinned `v1.0.1`, but reports and manifests MUST say `533/534 available train
scenes`; they must not imply that the absent scene was processed or call the
result a complete official-release benchmark.

#### 3.3.2 Full-train execution schedule

For pinned `v1.0.1`, the complete available-train data preparation is **one
workflow execution**, not nine manual launches:

| Stage | Partitions / tasks | Partition size | Active concurrency | Scheduler waves |
|---|---:|---:|---:|---:|
| inventory preflight | 1 | 534 expected / 533 available | 1 | 1 |
| ingest map | 533 | 1 scene/pod | 60 pods | `ceil(533/60) = 9` |
| reasoning-label map | 533 | 1 scene/pod | 5 pods x 2 workers | `ceil(533/5) = 107` |
| pack map | 533 | 1 scene/pod | 60 pods | `ceil(533/60) = 9` |
| manifest reducer | 1 | all 533 selected results | 1 | 1 |
| `train_il` | 1 | merged manifest | 1 GPU | 1 serial run |

The scheduler waves are internal `map_task` admission waves; operators do not
launch the workflow nine or 107 times. At 60 data-prep pods, requested capacity is
900 vCPU and 3.75 TiB (60 x 15 vCPU / 64 GiB), below the planned 1000-vCPU /
8-TiB namespace quota and the 1152-vCPU EC2 quota. Using 15 rather than 16 vCPU
keeps a pod below the kube-reserved allocatable CPU of a 16-vCPU node, so
Karpenter does not need to place every pod on a 32-vCPU instance. Each pod also
requests 60 GiB ephemeral storage: the default EKS Auto Mode `NodeClass` exposes
about 70 GiB allocatable from its 80-GiB volume, while the largest pinned scene
briefly needs about 40.24 GiB during tar download plus extraction. Label
concurrency is lower
because the 10-replica Cosmos endpoint, not Kubernetes capacity, is its bound.
Training runs exactly once on one GPU after the validated manifest is complete.

The earlier L2D profile of 16 vCPU / 128 GiB / 800 GiB is not reused here. It
cannot schedule on the live default `NodeClass`, and a 128-GiB request also
crosses the allocatable-memory boundary of a nominal 128-GiB instance. Restoring
that profile for future L2D work requires an IaC-managed data-prep `NodeClass`
and `NodePool`; it is not a prerequisite for the one-scene KITScenes run.

The L2D-specific 17-batch launcher is not the KitScenes entry point. If the
installed Flyte version cannot execute the mapped-array design, the fallback is
two stable prep executions of 267 and 266 scenes, followed by one manifest
collection and one training run. Do not use a 512+21 split or change scene
boundaries between retries, because stable partition inputs are required for
cache reuse.

#### 3.3.3 Optional cost-based successor

One scene/pod is the production baseline, not only a smoke setting. If measured
utilization later justifies combining small scenes, use a **cost-based, not
scene-count-based** plan:

```python
estimated_cost = n_frames*decode_cost + n_wm_windows*wm_cost + est_bytes*io_cost
# close the partition when running cost >= target_cost (tuned from measured
# P95 pod memory / time / S3 transfer of the smoke runs)
```
- "all scenes" MUST first resolve to the pinned, explicit scene-UUID list and be
  guarded:
  `assert n_partitions <= max_partitions` unless an explicit `allow_large_fanout`
  override is set.
- `log()` the partition plan (count, sizes, est cost) so fan-out scale is visible.
- The plan is deterministic given `(snapshot, target_cost)` so re-runs reproduce
  identical partitions (required for Flyte cache hits).
- Combining scenes MUST preserve one scene as the identity/split atom and keep
  every partition below measured P95 memory, ephemeral-storage, and wall-time
  limits. It is an optimization after the one-scene run, not a prerequisite.

### 3.4 Label cache: DROP the per-sample S3 cache (decided 2026-07-13)
The reasoning teacher was cached one-JSON-per-sample in
`s3://…/reasoning_labels_cache/…/{sample_uid}.json` (#117). At full L2D
(~100k episodes × ~100 valid samples/ep) that is **~10M tiny objects** — exactly
the small-file failure mode that blows up S3 request-rate on any copy/list and
exhausts inodes on the train node (per the wids/Turing writeup: transfer is
throttled by per-file COPY requests, and object count is capped before inode
exhaustion). The per-sample cache is therefore DROPPED.

This L2D scale estimate is retained as the design rationale. KitScenes uses the
same partition-level `records.jsonl` contract and must not introduce a new
per-sample object cache.

Its content is not lost: `generate_reasoning_labels` already writes a
`records.jsonl` (one full `record_to_json(record)` per line — byte-for-byte the
SAME payload the per-sample JSON stored, just concatenated) as the JOIN
interchange for pack. So the partition's `records.jsonl` (533 files for the
one-scene KitScenes train plan) replaces per-sample objects with NO information
loss. `LabelCache` (put/get per sample) is removed from the labeling path.

Re-label protection now comes from the STAGE cache, not per-sample objects: the
partition set is deterministic (`plan_partitions`), so re-running an unchanged
partition is a Flyte task-cache no-op (§3.4a) and never re-bills Cosmos. The
only thing lost vs per-sample caching is CROSS-partition-boundary sample reuse
(the same physical sample labeled under a different partition split) — which the
option-B design never does (partitions are stable), so it is a non-cost.

The old positional-keyed objects (`s{si:08d}.json`) and any per-sample uid
objects already written are now dead; clean them up under §3.4b.

### 3.4a Flyte caching + provenance (skip unchanged ranges) (review pt 2)
Stages are separate (decision 2) and fan out per range, so a re-run must NOT redo
unchanged work. The current L2D tasks already enable Flyte caching; preserve it
for KitScenes. A generic `cache_version="v1"` alone is insufficient: Flyte's
cache key is (task interface, input literals, cache_version), and
`FlyteDirectory` inputs hash by URI, so a code/spec change that doesn't change
inputs would serve a STALE cached output. Thread an explicit provenance object
through every stage as an input so the cache key reflects the real determinants:
```python
@dataclass(frozen=True)
class DatasetSnapshot:
    dataset: str            # e.g. "KIT-MRT/KITScenes-Multimodal"
    source_revision: str    # HF commit sha — pins the raw data
    uid_schema_version: str # "v1"
    parser_version: str     # bump on parser/enumeration change
    metadata_digest: str    # hash of the resolved group-id list
```
Per stage, include the provenance that actually affects its output:
- `data_ingest_range`: `DatasetSnapshot` (dataset + revision + group list). lerobot
  also caches the HF download on disk; Flyte cache skips the whole task on re-run.
- `generate_reasoning_labels_range`: `DatasetSnapshot` + `teacher_model_revision`
  + `prompt_body_hash` + `prompt_version` + decode params. The per-sample S3 label
  cache is GONE (§3.4); re-label protection is now SOLELY the Flyte task cache —
  an unchanged range is a task-cache no-op (no Cosmos call), a changed
  prompt/model correctly misses. (This is why `prompt_version` alone was fragile —
  it must ride in the cache key alongside the snapshot.)
- `data_processing_range`: `DatasetSnapshot` + `shard_schema_version` +
  `world_model` flag + `geometry_version`.
Bump the relevant field/version on ANY code change to that stage — version bump is
the implementer's responsibility (Flyte won't detect a pure code change).
Cache key = (task signature, input literals, cache_version). Since inputs are the
partition + dataset + flags, ranges are independently cacheable. This is what
makes extending from a smoke scene set to all scenes cheap: only NEW partitions
run.
Caveat: `FlyteDirectory` inputs hash by URI, so upstream re-runs that produce a
new raw dir URI will invalidate downstream cache — acceptable, and why ingest
caching (stable raw URI per range) matters most.

### 3.4b S3 cleanup — retention, not immediate delete (review pt 8)
This is the retained L2D cleanup design only. It is deferred and no legacy L2D
object is changed by the KitScenes milestone.

The cutover leaves unused S3 state (old positional label-cache prefix; orphaned
raw/shard `FlyteDirectory` outputs from superseded single-pod runs, e.g. the
failed 50-ep ingest `raw`, partial packs). Deleting Flyte-managed artifact
prefixes outright can break past executions that still reference them, so use a
retention flow rather than `rm`:
1. Write a manifest of deletion candidates (list → save).
2. Verify the new partition `records.jsonl` files and aggregate manifest cover
   the expected deterministic label subset before retiring the old prefix.
3. Tag legacy objects (e.g. `lifecycle=retire`) rather than deleting.
4. Give a 14–30 day rollback window.
5. Expire via an S3 Lifecycle rule on the tag/prefix (not a manual bulk delete).
6. If the bucket is versioned, a DELETE only adds a delete-marker — also expire
   NONCURRENT versions to actually reclaim space.
- MUST use `--profile autowarefoundation` / us-west-2 and confirm each prefix;
  never touch the Cosmos account. All steps logged, never silent.

### 3.4c Contract stability — what forces a re-run (and what must NOT)
The whole point of caching is "ingest once, never again; pack rarely". That only
holds if the CONTRACTS whose values enter the cache key are STABLE. This section
is the single registry of every cache-invalidating knob; it exists so we do not
accidentally churn a contract and silently re-run the whole corpus.

Accepted re-runs (fine, per user): (i) extending the requested group set (a NEW
partition is a new cache key — only the new ranges run; old ranges stay cached); (ii) an
ingest cache miss cascading to its own pack (the raw URI changes, so that pack
re-runs). We do NOT engineer around these.

The contracts that MUST stay stable (each is a single, centrally-defined constant,
NOT ad-hoc strings scattered in code). A change to any one = intentional,
reviewed, corpus-wide re-run of the stage(s) below it:

| Contract | Defined in | Enters cache key of | Bump ONLY when… |
|---|---|---|---|
| `uid_schema_version` | one module constant | (nothing directly — it shapes uids/keys) | the sample_uid / split_group_uid FORMAT changes |
| `DatasetSnapshot.source_revision` | resolved from HF at plan time | ingest, label, pack | the upstream dataset revision itself changes |
| `parser_version` | one constant per parser | ingest, label, pack | the sample ENUMERATION or per-sample fields change |
| `prompt_body_hash` + `prompt_version` | hash of the actual prompt text | label | the teacher prompt text changes |
| `teacher_model_revision` | teacher config | label | the teacher model/endpoint version changes |
| `shard_schema_version` | one constant | pack | the packed tar member layout changes |
| `geometry_version` | one constant | pack | the calibration/projection encoding changes |
| `world_model` (bool) | run input | pack | (a real config choice, expected) |

Rules that keep this from drifting:
- Each version is a NAMED constant in ONE place (e.g. `CONTRACT_VERSIONS` dict),
  referenced by every task — never an inline literal. A grep shows all of them.
- Bumping a version is a deliberate PR change with a one-line "why" — it is the
  ONLY sanctioned way to invalidate cache, and reviewers can see the blast radius
  (which stages re-run) from the table above.
- **What must NOT invalidate the cache** (else "never again" breaks): the number
  of groups requested in a later run, the Flyte execution id, the wall-clock
  time, pod resource limits (mem/cpu), `num_workers`, `label_workers`,
  `map_concurrency`, or any log/print change. None of these are inputs to the
  cached tasks, so they must NOT be added as task parameters that feed the key.
  (Corollary: resource/worker tuning is a task-decorator/pod concern, kept OUT of
  the cached input signature.)
- Refactors that do not change output BYTES for a given input should NOT bump a
  version. Correctness of "unchanged output" is checked by the per-uid semantic
  test (§6), so a pure refactor can ship without a corpus re-run.

Stability check to run before Phase 2 fan-out: enumerate the `CONTRACT_VERSIONS`,
confirm each is a single constant, and confirm no runtime/tuning knob is in any
cached task's input signature. This is the concrete guard that the contract we
are locking now stays locked.

### 3.4d World-Model pack: dedup frames (decided 2026-07-13)
The measurements and key examples below came from L2D. The same packed contract
applies to KitScenes, with scene UUID replacing episode index and seven camera
views replacing six. No `window_index.json` entry may reference another scene.

**Problem (measured).** With reasoning labels present, `data_processing` forces
`world_model=True`, and a WM sample packs 55 image JPEGs: 6 current cams + 1 map +
(4 history + 4 future) × 6 cams = 48 window frames. Samples are enumerated at
10 Hz (every source row) but the window steps at 1 Hz (stride 10), so **every
physical camera frame is JPEG-encoded and stored ~9×** across a shard (it lands in
8 stride-10 window slots plus its own current-frame member). A WM shard is ~8× an
imitation-only shard (~1.4 GB vs ~175 MB / 1000 samples). This duplication — not
the decode location — is the storage+pack bottleneck.

**Rejected: train-time seek / NVDEC.** Rebuilding the window by lerobot
`delta_timestamps` seek at train time (or GPU NVDEC) removes the storage cost but
reintroduces exactly what pre-extraction (#30) was built to avoid: the WM window
is on the critical path for ALL THREE losses (it produces the planner+reasoning
`visual_history` via `encode_history`, not just the JEPA target — auto_e2e.py:145-168),
so every batch of every epoch re-pays ~48 multi-cam decodes. The per-worker ~300 MB
video footprint OOM-killed 16 workers at 32 Gi during PACK; train_il requests only
16 Gi. It also drags lerobot/PyAV + raw video + a live reasoning JOIN onto the
train node (a large image/loader rebuild) for a recurring per-epoch compute cost.
Also note the label stage decodes a DIFFERENT frame set (front cam × 5 horizons)
than train (6 cams × 8 window steps), so "reuse the label-time JPEGs at train
time" does not apply — they are not even a superset.

**Per-loss Hz (locked with the user):** the sample enumeration stays **10 Hz**
(one sample per source row) — the REACTIVE head needs the dense per-0.1 s
trajectory targets, so we do NOT decimate pack/train samples. The WM window is
**1 Hz** internally (stride-10 offsets), and dedup removes the resulting
cross-sample frame duplication WITHOUT touching sample density. The Cosmos
labeler is decimated to **1 Hz** separately (below) — reasoning is a 1 Hz concern.

**Chosen: dedup at the ENCODE layer, keep train-time decode-free.** Each distinct
`(episode,row,cam)` 256² frame is JPEG-encoded ONCE per shard into a
content-addressed frame pool keyed by a stable, dot-free `frame_id`
(`e{ep}-r{row}-c{cam}`); each sample carries a tiny `window_index.json` mapping
its history/future `(step,view) → frame_id`. `cam_*.jpg` (the reactive current
frame) / `map.jpg / ego.npy / meta.json / calib.json / reasoning.json` stay
byte-for-byte per sample, so `sample_uid`, `split_group_uid` bucketing, and the
reasoning JOIN are untouched.

**Pool lives as a sibling `pool/` DIRECTORY next to the `*.tar` shards, NOT inside
them.** This is the load-bearing detail: the loader builds its stream from
`glob("*.tar")` and WebDataset's `split_by_worker` shards THAT list across workers;
a `pool/` dir does not match `*.tar`, so it is never part of the sharded stream and
every worker reaches any `frame_id` by path regardless of which tar it owns.
Putting the pool inside a shard would hit the exact `split_by_worker` invisibility
that the double-split data-loss fix warns about (a sample's future frame can live in
a tar a different worker holds). The loader (`pre_extracted.py`) replaces
`_decode_window`'s hist_/fut_ regex with: parse `window_index.json` → look up each
`frame_id` in `pool/` → `_decode_image` → stack into the SAME
`history_frames [T,V,3,H,W]` / `future_frames [F,V,3,H,W]` tensors `auto_e2e.py`
already consumes. AutoE2E, all three losses, MergedDatasetLoader, and the JOIN are
unchanged. Result: storage ~8× → ~1.1× of imitation-only, zero train-time video
decode, all three losses intact, 10 Hz density preserved. Scope: `parallel_pack.py`,
the parent pack loop in `workflows.py`, and `pre_extracted.py`.

**Cosmos labeling decimated to 1 Hz (separate from pack).** Reasoning horizons are
0/1/2/3/4 s, so the teacher only needs the 1 Hz subset. The labeler enumerates a
STABLE uid-derived subset (label iff `frame_index % 10 == 0`, a function of the
sample's identity — NOT its positional index — so it is partition-independent), so
Cosmos is called ~10× less. The packer still packs ALL 10 Hz samples; the 9/10
without a matching `reasoning.json` decode to a fully-MASKED (abstained) reasoning
target and contribute nothing to the reasoning loss (the masking path already
exists and `data_processing` does not fail when most samples lack a label). Net:
reactive + JEPA train on all 10 Hz samples; reasoning trains on the 1 Hz labeled
subset; teacher cost drops ~10×.

**Boundary safety — no cross-episode/scene reference, INCLUDING the reactive
history (locked requirement).** Every backward/forward reach must stay inside the
sample's own episode:
- egomotion history/future: `extract_egomotion` slices a window taken from
  `_get_vehicle_states_window(ep_start, ep_end)`, i.e. clamped to the episode.
- WM window: enumeration excludes episode-edge frames (margins 64/64 ≥ WM reach 40)
  and lerobot's `_get_query_indices` clamps every delta to the current episode
  `[dataset_from_index, dataset_to_index)` — so no future/history frame crosses a
  clip boundary.
- dedup frame_ids are `(episode,row,cam)`, so a `window_index` can only reference
  rows of the SAME episode — the pool never lets a window borrow a neighbour clip's
  frame.
A test asserts every `window_index` frame_id shares the sample's episode. Within one
L2D episode (a continuous 10 Hz clip) the +4 s target can still be a large-but-valid
appearance change — the intended JEPA difficulty, distinct from cross-clip
contamination.

For KitScenes, use a stable scene-qualified pool key such as
`kitscenes-{scene_uuid}-r{row}-c{cam}` and assert that every referenced key shares
the sample's scene UUID. This is the active full-run invariant.

**Must validate on GPU before full run:** (i) byte-equality — `history_frames`/
`future_frames` rebuilt from a deduped shard equal the current WM-shard tensors for
the same samples; (ii) a full-loss `train_il` step (WM+reasoning on) yields finite
non-None traj/jepa/reason; (iii) measured deduped shard size ≈ imitation baseline;
(iv) the pool key namespace does not leak into the camera-key regex (`num_views`
unchanged in the manifest); (v) `num_workers>0` still hides the gather+decode.

### 3.5 Open design questions — RESOLVED in review
1. **Ingest↔pack coupling → SEPARATE tasks + PER-PARTITION raw materialization
   (option B, chosen 2026-07-13).** Keep ingest/label/pack as distinct tasks
   (stage-level retry). Each partition's ingest fetches ONLY that partition's raw
   from source and saves it as the partition's `FlyteDirectory`; that partition's
   label + pack read it. Raw is pulled from HF/SDK ONCE per partition (not
   re-fetched in every downstream pod), and a retry reads the already-materialized
   partition raw (Flyte cache serves the same URI) rather than re-hitting HF —
   re-try stability is the priority. Per-partition sizing keeps memory bounded, so
   the whole-corpus ingest OOM is gone.
2. **Eval multi-dir → eval over ALL shard dirs.** Change `_select_shard_dir` →
   consume the full `List[FlyteDirectory]`. This fixes internal group-holdout
   coverage across partitions; the official fixed-window benchmark remains the
   separate §3.6 path.
3. **Partition size → one KitScenes scene/pod to start**, tunable only from
   measured archive bytes/frame cost; this run targets all 533 available train
   scenes.
4. **Kueue quota.** Data-prep pods are CPU/mem (not GPU). Confirm the CPU/mem
   ClusterQueue admits up to 60 concurrent map-task pods; raise the live namespace
   quota before the smoke run. Training remains on one GPU.
5. **`@dynamic` + `map_task`.** Partition count depends on the resolved scene
   inventory → `@dynamic` computes partitions then maps. Confirm the installed Flyte
   version supports the nesting during Phase 2 (validate on a 2-partition run
   first).
6. **Shard count vs DataLoader workers.** More partitions ⇒ more shard files ⇒
   `num_workers>0` parallelism (capped by shard count) works better — aligns with
   P0. Also pack multiple smaller shards per partition if a partition yields few.

### 3.6 KITScenes benchmark objective and split policy

Producing a KITScenes E2E benchmark result is a first-class output of #121, not an
informal ADE/FDE check after training. The benchmark report must be reproducible
from an immutable dataset revision, SDK revision, scene/window manifest, packed
manifest, checkpoint SHA-256, evaluator version, and declared input modality.

#### 3.6.1 Official split contract

The dataset paper, Appendix G.1, defines the following geographic split:

| Official split | Scenes | Hours | Path | Released data | #121 use |
|---|---:|---:|---:|---|---|
| `train` | 534 | 3.00 | 81.0 km | all | primary map-conditioned training |
| `val` | 117 | 0.60 | 14.5 km | all | paper-protocol benchmark only |
| `overlap-train-val` | 23 | 0.13 | 3.3 km | all | paper-protocol benchmark only |
| `test` | 206 | 1.17 | 30.1 km | no maps | optional sensor-only training |
| `test-e2e` | 127 | 0.76 | 33.1 km | no maps, geo-pose, or post-keyframe future | future held-out submission |
| **total** | **1,007** | **5.66** | **162.0 km** | | |

The official split construction excludes train scenes within 100 m of test
scenarios and within 70 m of validation scenarios. Preserve these supplied split
IDs exactly; do not reshuffle official `val`, `overlap-train-val`, `test`, or
`test-e2e` scenes into the primary training set.

The pinned SDK is an alpha release and explicitly says the current dataset
`v1.0.1` is not recommended for final benchmark reporting. Pins make development
experiments reproducible, but a result on this release must be labelled
`development`, not an official final benchmark submission.

#### 3.6.2 E2E protocol and source discrepancy

The paper's Section 4.4 and Appendix H.4 specify:

- 200 non-overlapping samples drawn from
  `val union overlap-train-val`;
- one 9-second window per sample: 4 seconds of past observation and up to
  5 seconds of future trajectory, at 10 Hz;
- 3 seconds (30 future steps) as the headline horizon and 5 seconds (50 steps)
  as the long-horizon protocol;
- ADE and FDE, plus drivable-surface survival, collision-free rate, centerline
  distance, and Multi-Maneuver Score (MMS);
- MMS compares against the best of at least three human-annotated admissible
  5-second maneuvers per scene.

The map-grounded metrics are not aliases for ADE/FDE. Drivable-surface survival
checks the predicted ego footprint against the union of drivable Lanelet2
polygons through the horizon. Collision-free rate checks the footprint against
the LiDAR-derived occupancy layer and logged dynamic-agent boxes. Centerline
distance is mean lateral offset to the nearest drivable centerline. These must use
the official map/occupancy geometry and ego footprint.

The website's published baseline table is explicitly camera-based. Released HD
maps may be used by the evaluator to compute safety metrics, but that does not
make an HD map an allowed model input. Report the current map-conditioned
AutoE2E result as a separate declared input track; do not compare it directly
with the camera-only table as if the inputs were equivalent.

There is a current upstream inconsistency:

- the paper says the public 200-window evaluation comes from
  `val union overlap-train-val` and reserves the 127-scene `test-e2e` split for a
  future leaderboard;
- the KITScenes benchmark website says the 200 windows come from `test-e2e`,
  while also saying that the community leaderboard is "coming soon";
- the pinned SDK publishes the split scene lists but no exact 200-window
  keyframe manifest or released submission evaluator.

Until upstream publishes the manifest/evaluator, keep two tracks separate:

1. **Paper-reproduction development track:** package `val` plus
   `overlap-train-val`, then score the exact 200 anchors if/when their official
   manifest is obtained. A locally selected 200-window manifest is useful for
   regression testing but must be labelled `paper-protocol approximation`.
2. **Official held-out track:** run `test-e2e` only under the rules and manifest
   shipped with the future challenge, then submit predictions to its evaluator.
   The released challenge definition supersedes this interim interpretation.

Do not merge samples from these two tracks or publish either approximation as a
leaderboard score.

#### 3.6.3 Train/eval strategy

Use four distinct data roles:

1. **Model development:** freeze an explicit, scene-level
   `train-dev-split-v1.json` inside the 533 available official train scenes.
   Target about 10% (53 scenes) for internal holdout, assigning geographic scene components
   together where pose metadata permits. Never split frames from one scene across
   train and holdout. This replaces the runtime 10-bucket hash as the experiment
   selection contract.
2. **Final primary training:** after architecture and hyperparameters are fixed,
   retrain once on all 533 archives available from the official train split at
   the pinned revision. The internal holdout returns to training here.
3. **Benchmark evaluation:** never train on the 117 `val` or 23
   `overlap-train-val` scenes. Use them only for the fixed paper-protocol windows;
   do not substitute the current `val_fraction=0.1` loader split for this
   benchmark.
4. **Future held-out evaluation:** do not locally score or tune on `test-e2e`.
   Generate predictions and submit them when the official evaluator exists.

Appendix H.4 permits E2E training on `train union test` (over 100 km), because
`test` retains sensors and trajectories. However, `test` withholds maps, while
the current AutoE2E model consumes an HD-map tensor. Therefore the primary
map-conditioned run uses `train` only. A separately declared sensor-only/no-map
model may later use all 740 `train union test` scenes; it must not be mixed into
the primary model silently.

#### 3.6.4 Dataset-prep and evaluation executions

| Purpose | Source scenes | Scene partitions | Labels | 60-pod data-prep waves | Top-level executions |
|---|---:|---:|---|---:|---:|
| primary final training | 533/534 available `train` | 533 | yes: 107 waves at label concurrency 5 | 9 ingest + 9 pack | 1 prep/full-run |
| paper benchmark inputs | 117 `val` + 23 overlap = 140 max | up to 140 | no | 3 ingest + 3 pack | 1 prep + 1 eval |
| optional sensor-only extension | 206 `test` | 206 | training-dependent | 4 ingest + 4 pack | deferred |
| future held-out submission | 127 `test-e2e` | up to 127 | no targets | 3 ingest/inference waves | 1 submission run |

The exact paper benchmark may touch fewer than 140 scenes once the 200-window
manifest is published; fan out only its unique scene UUIDs then. Evaluation never
calls Cosmos, so it has no 107-wave label stage.

#### 3.6.5 Current evaluator gaps and definition of done

The current `_run_evaluation` is an internal regression evaluator, not the
KITScenes benchmark:

- it selects a group-hash `val_fraction` from the training shards instead of a
  fixed official scene/window manifest;
- it integrates all 64 predicted control steps (about 6.4 seconds) and reports one
  ADE/FDE pair, rather than exact 30-step/3-second and 50-step/5-second horizons;
- the KitScenes parser and AutoE2E training contract use 64 ego-history steps
  (6.4 seconds), while the benchmark permits only 4 seconds of past observation.
  A benchmark-only adapter must left-pad/mask to the model's fixed tensor width
  without reading pre-window data. Training and internal validation retain the
  AutoE2E 64-step contract; this horizon predates the L2D parser and is not a
  dataset-specific assumption;
- it does not compute drivable-surface survival, collision-free rate, centerline
  distance, or MMS;
- no published benchmark manifest/evaluator currently exposes the exact
  multi-maneuver references needed to reproduce MMS, so an approximation must
  omit MMS rather than invent labels;
- the current model requires `map_input`, but held-out `test-e2e` supplies no map.
  An official submission therefore needs a declared protocol-compliant no-map
  configuration rather than a hidden zero-map substitution.

Milestone completion requires: (a) all 533 available train scenes processed after
an inventory preflight that records the one allowed missing UUID; (b) one final
available-train checkpoint; (c) an upstream
or explicitly approximate 200-window manifest; (d) 3-second and 5-second metrics
with the full safety suite; and (e) a versioned report that states release
status, map/input usage, every artifact digest, and whether the score is
paper-reproduction, approximation, or official held-out.

---

## 4. Risks
- **JOIN correctness** is the highest risk: the whole scheme rests on the global
  uid being byte-identical between the label pod and the pack pod for the same
  physical frame. Mitigation: a unit test that builds the parser over two
  different scene subsets that overlap, and asserts `sample_uid` matches for the
  shared frames.
- **Control-plane/event-size pressure** from 533 scenes × 3 stages. Mitigation:
  represent each stage as one mapped array node, log/validate its exact
  cardinality, and retain the two-by-267 explicit-graph fallback.
- **Archive-size and transfer volume** are larger than the original estimate:
  train is 2,439.15 GiB compressed, with 10.23 GiB p95 and 20.12 GiB max per
  scene. Mitigation: keep one scene per partition, measure extract expansion and
  source throttling in the 10-scene smoke, and use archive bytes in any later
  cost plan.
- **Published inventory is incomplete:** the official train list has 534 UUIDs
  while pinned `v1.0.1` has 533 archives. Mitigation: permit only the exact
  one-scene deficit through `max_missing_scenes=1`, record the UUID and counts in
  the manifest, and fail on any additional discrepancy. Report the result as a
  533/534 development run, not a complete official-release benchmark.
- **Gated-source authentication or source drift** can make an ingest retry
  non-reproducible. Mitigation: pin both dataset and SDK commits, validate access
  before launch, and include source revision/checksum in `DatasetSnapshot`.
- **Benchmark specification drift:** the paper and website currently disagree on
  the source split for the 200 windows, and the official evaluator is not
  released. Mitigation: separate paper-reproduction, approximation, and official
  held-out tracks in artifact metadata; never compare them as one score.
- **Cross-dataset merge fairness** (separate, tracked): weighted interleaver
  (already scoped) — not required for Milestone 1 (single dataset, many partitions).
- **Eval leakage** if a development split is per-frame: split by
  `split_group_uid` (episode/clip/scene), NOT per-`__key__`, and freeze an explicit
  scene manifest. Official benchmark scenes never enter training. Mitigation:
  `train_group_uids.isdisjoint(eval_group_uids)` tests at both internal and
  official split boundaries.

---

## 5. Phased plan (KitScenes priority, 2026-07-15)
- **Phase 0 (done, retained):** L2D work established partition-independent IDs,
  option-B partition artifacts, bounded uploads, retries, and multi-directory
  train/eval. These foundations remain shared infrastructure.
- **Phase 1 — complete the KitScenes parser contract:** add stable sample/split
  IDs, raw frame/window access, front clips, egomotion/geospatial outputs, and a
  projection spec; fix absolute-yaw map rasterization. Cover these with synthetic
  scene tests that do not require the gated corpus.
- **Phase 2 — wire KitScenes into Flyte:** add `Dataset.KITSCENES`, scene-scoped
  ingest, explicit label/pack branches, three stage-level mapped arrays,
  deterministic scene planning, immutable source provenance, and the pinned
  SDK/runtime dependencies in the data-prep image. Raise the KitScenes
  `max_partitions` guard from 512 to at least 600. L2D and NVIDIA branches remain
  intact.
- **Phase 3 — operational smoke:** apply and verify the planned Flyte namespace
  quota (1000 CPU / 8 TiB; the handoff notes it is committed but not yet live),
  and verify Flyte storage auth remains `accesskey` after the Helm/Terraform
  operation. Rebuild/register images and workflows, then run one scene. Expand
  to a 10-scene test to validate fan-out, cache hits, manifest coverage, bounded
  Cosmos concurrency, source throughput, extract expansion, and multi-directory
  train/eval input.
- **Phase 4 — available-inventory full run:** preflight the pinned dataset with
  `max_missing_scenes=1`, record the known missing UUID, and launch 533 one-scene
  partitions in one mapped workflow, with at most 60 active data-prep pods,
  validate the aggregate manifest, then run serial single-GPU `train_il` once.
  A 3–4 day training runtime is acceptable.
- **Phase 5 — benchmark:** package up to 140 `val`/`overlap-train-val` scenes
  without teacher labels, implement the 3-second/5-second metric suite and
  protocol-compliant four-second input adapter, and publish a versioned
  paper-reproduction or clearly labelled approximation report. Add the official
  `test-e2e` submission path only when its manifest/evaluator is released.
- **Deferred:** do not launch or further adapt full L2D or NVIDIA PhysicalAI-AV
  in this milestone. Revisit them only after the KitScenes full run is complete.

## 6. Test plan (revised per review)
Do NOT byte-diff tar shards (member order / mtime / impl differences make
semantically-identical shards differ). Compare per-uid, semantically:
```python
assert old_rec.keys() == new_rec.keys()
assert content_hash(old_rec["cam_0.jpg"]) == content_hash(new_rec["cam_0.jpg"])
assert old_rec["reasoning.json"] == new_rec["reasoning.json"]
```
Required tests:
- same KitScenes frame from two DIFFERENT scene subsets → identical `sample_uid`.
- every uid matches `[A-Za-z0-9_-]+`, no `.`/`/`.
- `plan_partitions` deterministic; partitions cover all groups with no overlap/gap.
- the 534-UUID split against pinned `v1.0.1` passes only with
  `max_missing_scenes=1`, records exactly the known missing UUID, and fails with
  `max_missing_scenes=0` or any second discrepancy.
- mocked Hub calls prove both `sequence_archives.csv` and every scene archive use
  the exact `DatasetSnapshot.source_revision`, never implicit `main`.
- a 533-scene, size-1 plan has 533 partitions, passes `max_partitions>=600`, and
  computes 9 waves at concurrency 60.
- no scene appears in both internal train/holdout, or in final train and official
  benchmark sets (`isdisjoint`).
- changing `source_revision` / `prompt_body_hash` / `parser_version` → cache MISS.
- forcing ONE partition to OOM retries only that partition (stage isolation).
- mapped ingest/pack never exceed 60 active pods; teacher 429/backoff tests prove
  global calls stay at or below 5 pods x 2 workers.
- `set(label_uids) == expected_1hz_uids` and every label joins one packed sample.
- one-scene direct vs sharded KitScenes outputs are SEMANTICALLY identical
  (per-uid, above).
- benchmark input construction reads exactly four seconds of past data and never
  reads a pre-window frame when filling the model's 64-step input tensor.
- metric fixtures pin 30-step and 50-step ADE/FDE plus drivable survival,
  collision-free rate, centerline distance, and MMS behavior.
- **Contract stability (§3.4c):** every version is a single named constant in
  `CONTRACT_VERSIONS` (no inline literals — assert by grep/import); NO
  runtime/tuning knob (episodes count, num_workers, label_workers, map_concurrency,
  resource limits, execution id) appears in any cached task's input signature — so
  tuning never invalidates the cache and "ingest once, never again" holds.

---

## 7. What we are NOT doing (and why)
- Not raising single-pod memory further — that is the band-aid this design
  replaces.
- Not DDP in this milestone — training is explicitly single-GPU and may run for
  3–4 days.
- Not running or optimizing L2D/NVIDIA — preserve those paths, but spend #121
  implementation and operational capacity on KitScenes only.
- Not a full `DatasetAdapter` protocol refactor. KitScenes is the third dataset,
  but focused explicit branches are acceptable as long as all three satisfy the
  same identity/split/snapshot/manifest contracts.
- Not changing the model or losses to obtain fan-out. A protocol-compliant no-map
  model/configuration for a future `test-e2e` submission is a separately gated
  benchmark requirement, not a hidden pipeline substitution.

---

## 8. Sources (accessed 2026-07-15)

- [KITScenes Multimodal E2E Driving benchmark](https://kitscenes.com/benchmarks/multimodal-e2e-driving)
- [KITScenes paper](https://arxiv.org/html/2606.02956), especially Section 4.4,
  Appendix G.1, and Appendix H.4
- [KITScenes SDK at the pinned alpha commit](https://github.com/KIT-MRT/kitscenes/tree/7765cdec5490894266070ab46e23724b58b3da42)
- [Official generated split lists](https://github.com/KIT-MRT/kitscenes/tree/7765cdec5490894266070ab46e23724b58b3da42/kitscenes/split/generated_splits/default_geo_split_v1_0)
- [KITScenes Multimodal pinned dataset tree](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal/tree/6fde0034446669e2ed7235e4c7fe323cd23d599d)
