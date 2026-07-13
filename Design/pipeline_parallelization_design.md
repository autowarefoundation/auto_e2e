# Design: Episode-Sharded, Map-Task-Parallel Data Pipeline (#121)

Status: DECISIONS LOCKED (2026-07-13 review) — ready to implement Phase 1.
Scope: make the AutoE2E data pipeline (ingest → reasoning-label → pack → train →
eval) scale to ALL episodes/clips of **L2D and NVIDIA** by fanning each data-prep
stage out across many pods instead of one, so memory/time stop being a function
of total episode count. KitScenes is OUT of scope.

### Decisions locked in review
1. **GPU capacity is not a hard blocker** — g6e can scale to ~10 nodes each where
   quota/spec limits it. So GPU-side scaling (DDP, Kueue quota) is unblocked when
   we reach it; but data-prep (CPU/mem) is the current bottleneck, not GPU.
2. **Do NOT co-locate ingest+label+pack** (§3.5-1). Keep them as SEPARATE Flyte
   tasks so each stage retries independently. Rely on **Flyte caching** so a
   re-run skips unchanged ranges (ingest especially).
3. **Flyte caching must be added** — it is NOT used today (no `cache=True` in
   `workflows.py`). Add `cache=True` + a `cache_version` to `data_ingest`,
   `generate_reasoning_labels`, `data_processing` so unchanged (inputs, version)
   ranges are skipped on re-run.
4. **Cache migration = option (a):** discard the old positional-keyed reasoning
   labels and re-label (cheap). **Delete the now-unused S3 directories** left by
   the old scheme (old positional cache prefix + orphaned shard/raw dirs from
   superseded runs) as part of the cutover.
5. **Partition size = 10 episodes/pod to start.** Final target: **ALL episodes of
   L2D AND all clips of NVIDIA.**
6. **sample_uid scheme approved:** `l2d-e{episode}-f{frame}`,
   `nv-{clip_uuid}-{idx}`.
7. **KitScenes is OUT of scope** — only L2D + NVIDIA. Both are already wired into
   the `Dataset` enum + `data_ingest`/`data_processing`, so no new dataset
   plumbing is needed (unlike KitScenes, which would have required it).

---

## 1. Problem statement

### 1.1 Symptom
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

### 1.3 The one hard blocker: positional `sample_id`
The fix is Flyte `map_task`/`@dynamic` fan-out over episode ranges — but that is
**unsafe today** because the reasoning-label ↔ shard JOIN keys on a *positional*
id that only makes sense within a single process that loaded the exact same
episode set:

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
every cache key collides. **This must be fixed before any fan-out.**

---

## 2. Current architecture (verified, file:line)

```
wf_create_dataset(episodes, world_model, reasoning_teacher, prompt_version)   [workflows.py:1515]
  └─ data_ingest(dataset, episodes)  ── 1 pod ──▶ FlyteDirectory(raw)          [:173, returns :257]
       │   L2D: LeRobotDataset(episodes=range(episodes)); copytree→out_dir      [:251-259]
       │   NVIDIA: download clips[:episodes]                                    [:~200]
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

---

## 3. Design

### 3.1 Part A — Global, partition-independent `sample_id` (prerequisite)

Replace positional `f"s{si:08d}"` with an id built from identity the sample
already carries, stable no matter which episodes/clips a given pod loaded.

Proposed scheme (add a `sample_uid(idx) -> str` method to each parser):
- L2D: `l2d-e{episode_index:06d}-f{frame_index:06d}`
  (both fields exist: `l2d/dataset.py:319-320`).
- NVIDIA: `nv-{clip_uuid}-{sample_idx:06d}`
  (`nvidia_physical_ai/dataset.py:120`).

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

Call sites to change:
- `parallel_label.py:86`: `sample_key = _DS.sample_uid(si)` (worker holds `_DS`).
- Pack: the worker (`parallel_pack.pack_sample`) must RETURN the uid; the parent
  (`workflows.py:449-450`) uses it as the tar member prefix instead of
  `f"s{si:08d}"`.
- Tar member keys become `{sample_uid}.<suffix>`; the loader groups members by the
  WebDataset `__key__` which is the part before the first `.` — **`sample_uid`
  must contain no `.`** (the scheme above uses only `-`, safe). Verify against
  `pre_extracted.py` key grouping.
- Cache key `label_cache._key` already takes a `sample_id` string; feeding it the
  uid needs no signature change, but the KEY CONTENT changes → see migration 3.4.

**`sample_uid` is a formal identity contract, not a string format** (review pt 2).
Define a typed identity and derive the uid from it, so releases can't collide and
malformed keys are caught at build time:

```python
@dataclass(frozen=True)
class SampleIdentity:
    dataset_namespace: str   # e.g. "yaak-ai/L2D@<revision>"
    uid_schema_version: str  # "v1"
    group_id: str            # episode index (L2D) / clip_uuid (NVIDIA) — the SPLIT unit
    frame_id: int
```
- `sample_uid = f"l2d-v1-e{group_id}-f{frame_id:06d}"` (L2D),
  `f"nv-v1-{clip_uuid}-{frame_id:06d}"` (NVIDIA). Include the `uid_schema_version`
  so a future scheme change is a clean cache/JOIN break, not a silent mismatch.
- Validate every uid at generation: `re.fullmatch(r"[A-Za-z0-9_-]+", uid)` and no
  `.`/`/`. Store the raw `episode_index`/`clip_uuid`/`revision` in the sample's
  `meta.json` (not only encoded in the tar key) for traceability.

**`split_group_uid` — the eval-split unit (review pt 1, the most important fix).**
A SEPARATE id at episode/clip granularity, NOT the per-frame uid:
```python
split_group_uid = f"l2d-e{episode_index:06d}"   # NVIDIA: f"nv-{clip_uuid}"
split_bucket = blake2b(f"{split_seed}:{split_group_uid}") % 100
```
Rationale: L2D frames within an episode are strongly correlated; a per-frame
`__key__` hash split (the current `pre_extracted._split_bucket`) puts adjacent
frames of the SAME episode into both train and val → evaluation leak, which
silently inflates held-out numbers. Splitting by episode/clip makes train/val
disjoint at the group level. Required invariant + test:
`assert train_group_uids.isdisjoint(val_group_uids)`. → This replaces the
per-`__key__` split in `pre_extracted.py:_split_bucket`; the loader must split on
a `split_group_uid` carried in each sample's meta (add it to the packed
`meta.json`), not on `__key__`.

**Strict JOIN (review pt 2).** The consistency guard (`workflows.py:322-332`)
"label set covers pack set" is too weak. Require EXACT equality:
```python
assert len(pack_uids) == len(set(pack_uids))   # no dup uids in a shard
assert len(label_uids) == len(set(label_uids))
assert set(pack_uids) == set(label_uids)        # per partition
```
Abstain is an explicit label STATE (already modelled — `ReasoningLabelRecord`
carries an abstain/error field), never a missing key, so a labelled partition has
one record per packed sample.

### 3.2 Part B — Map-task fan-out per stage

Introduce an **episode-range partition** as the unit of fan-out. A partition is a
contiguous `(start_ep, end_ep)` (L2D) or a clip-uuid sublist (NVIDIA). Choose
partition size so ONE pod comfortably handles it (e.g. 10 episodes ≈ the known-good
single-pod size).

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
  (sample-index JOIN holds). (NVIDIA clip-uuid fan-out is Phase 4; L2D validated first.)

New Flyte structure (`@dynamic` to size partitions at run time, then `map_task`):

```
@dynamic
def wf_create_dataset_sharded(dataset, episodes, target_cost, world_model, teacher, prompt):
    partitions = plan_partitions(dataset, episodes, target_cost)   # list[PartitionSpec] (episode/clip ids)
    # 1) INGEST fan-out: each pod fetches ONLY its partition's raw → its own FlyteDirectory
    raws = map_task(data_ingest_range)(partition=partitions, dataset=..., ...)
    # 2) LABEL fan-out: each pod reads its partition's raw, labels it
    label_manifests = map_task(generate_reasoning_labels_range, concurrency=C)(
        raw=raws, partition=partitions, ...)                       # → per-partition records.jsonl (+ S3 cache)
    # 3) PACK fan-out: each pod reads its partition's raw + JOINs its labels → its OWN shards
    shard_manifests = map_task(data_processing_range)(
        raw=raws, labels=label_manifests, partition=partitions, ...)
    return validate_and_publish_manifest(shard_manifests)          # DatasetManifest (train_il consumes it)
```

Per-stage detail:

**(1) `data_ingest_range(partition)`** — one pod per partition. Fetches ONLY its
partition's raw data from source (L2D: lerobot `episodes=partition_eps`; NVIDIA:
SDK `partition_clips`) and persists it as the partition's raw `FlyteDirectory`
(hardlink copytree). Memory/disk scale with `partition` size, so the whole-corpus
ingest OOM is gone. Raw is fetched from source ONCE per partition here; downstream
pods reuse this materialized raw (no HF re-hit).

**(2) `generate_reasoning_labels_range(raw, partition)`** — one pod per range.
- Reads its partition's raw dir (no source re-fetch), builds the front-clip parser
  on that slice, labels with the global `sample_uid`.
- Writes to the SAME S3 label cache prefix — cache hits across runs still work
  because the uid is global. Emits a per-range `records.jsonl`.
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
  Per-uid cache write is idempotent (`label_cache.put` overwrites the same key).
  Currently `openai_compatible.py` has NO retry — add it. The >50% abstain guard
  stays per-range.

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
                 "sample_count": 1834, "label_count": 1834, "sha256": "..."}],
 "train_groups": 90000, "val_groups": 10000}
```
Reducer validations (fail the run if any break): partition coverage is exact (no
missing/overlapping episodes), no group appears in >1 partition, no duplicate uids
across shards, `label_uids == pack_uids` per partition, shard checksums/counts
match. Always associate by `partition_id`, never map output order/list position.
`train_il`/eval take the `DatasetManifest` (or its shard_uris) — the manifest is
the formal artifact; the `List[FlyteDirectory]` train_il already accepts is the
transport underneath.

### 3.3 Partitioning function
`plan_partitions(snapshot, target_cost, max_partitions)` returns a deterministic
`PartitionSpec` list. Start with fixed `partition_size=10` episodes for smoke
tests, but the PRODUCTION plan is **cost-based, not episode-count-based**
(review pt 5): L2D is ~100,000 episodes / ~19M frames (~190 frames/ep, uneven), so
a fixed 10-ep unit would create ~10,000 partitions × 3 stages ≈ 30,000 mapped
executions — crushing the Flyte control plane + Kubernetes scheduler and spawning
huge numbers of tiny S3 objects. Instead accumulate consecutive episodes/clips
into a partition until an estimated cost threshold is hit:
```python
estimated_cost = n_frames*decode_cost + n_wm_windows*wm_cost + est_bytes*io_cost
# close the partition when running cost >= target_cost (tuned from measured
# P95 pod memory / time / S3 transfer of the smoke runs)
```
- `episodes=0` (= all) MUST first resolve the true count and is guarded:
  `assert n_partitions <= max_partitions` unless an explicit `allow_large_fanout`
  override is set. NEVER silently fan out 10k pods.
- `log()` the partition plan (count, sizes, est cost) so fan-out scale is visible.
- The plan is deterministic given `(snapshot, target_cost)` so re-runs reproduce
  identical partitions (required for Flyte cache hits).

### 3.4 Label cache: DROP the per-sample S3 cache (decided 2026-07-13)
The reasoning teacher was cached one-JSON-per-sample in
`s3://…/reasoning_labels_cache/…/{sample_uid}.json` (#117). At full L2D
(~100k episodes × ~100 valid samples/ep) that is **~10M tiny objects** — exactly
the small-file failure mode that blows up S3 request-rate on any copy/list and
exhausts inodes on the train node (per the wids/Turing writeup: transfer is
throttled by per-file COPY requests, and object count is capped before inode
exhaustion). The per-sample cache is therefore DROPPED.

Its content is not lost: `generate_reasoning_labels` already writes a
`records.jsonl` (one full `record_to_json(record)` per line — byte-for-byte the
SAME payload the per-sample JSON stored, just concatenated) as the JOIN
interchange for pack. So the partition's `records.jsonl` (a handful of files,
one per partition) fully replaces the ~10M per-sample objects with NO information
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
unchanged work. Add Flyte task caching — BUT a generic `cache_version="v1"` alone
is insufficient: Flyte's cache key is (task interface, input literals,
cache_version), and `FlyteDirectory` inputs hash by URI, so a code/spec change
that doesn't change inputs would serve a STALE cached output. Thread an explicit
provenance object through every stage as an input so the cache key reflects the
real determinants:
```python
@dataclass(frozen=True)
class DatasetSnapshot:
    dataset: str            # "yaak-ai/L2D"
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
makes "extend from 20 → 50 → all episodes" cheap: only the NEW ranges run.
Caveat: `FlyteDirectory` inputs hash by URI, so upstream re-runs that produce a
new raw dir URI will invalidate downstream cache — acceptable, and why ingest
caching (stable raw URI per range) matters most.

### 3.4b S3 cleanup — retention, not immediate delete (review pt 8)
The cutover leaves unused S3 state (old positional label-cache prefix; orphaned
raw/shard `FlyteDirectory` outputs from superseded single-pod runs, e.g. the
failed 50-ep ingest `raw`, partial packs). Deleting Flyte-managed artifact
prefixes outright can break past executions that still reference them, so use a
retention flow rather than `rm`:
1. Write a manifest of deletion candidates (list → save).
2. Verify the NEW uid cache covers the same samples before retiring the old prefix.
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

Accepted re-runs (fine, per user): (i) extending the episode set (a NEW partition
is a new cache key — only the new ranges run; old ranges stay cached); (ii) an
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
  of episodes requested in a later run, the Flyte execution id, the wall-clock
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
   consume the full `List[FlyteDirectory]` (the held-out `val` split already makes
   this a proper generalization measure across all partitions).
3. **Partition size → 10 episodes/pod to start**, tunable; final target is ALL
   L2D episodes + all NVIDIA clips.
4. **Kueue quota.** Data-prep pods are CPU/mem (not GPU). Confirm the CPU/mem
   ClusterQueue admits N concurrent map-task pods; raise if needed. GPU nodes can
   scale to ~10 (decision 1) for the later DDP phase.
5. **`@dynamic` + `map_task`.** Partition count depends on the runtime `episodes`
   input → `@dynamic` computes partitions then maps. Confirm the installed Flyte
   version supports the nesting during Phase 2 (validate on a 2-partition run
   first).
6. **Shard count vs DataLoader workers.** More partitions ⇒ more shard files ⇒
   `num_workers>0` parallelism (capped by shard count) works better — aligns with
   P0. Also pack multiple smaller shards per partition if a partition yields few.

---

## 4. Risks
- **JOIN correctness** is the highest risk: the whole scheme rests on the global
  uid being byte-identical between the label pod and the pack pod for the same
  physical frame. Mitigation: a unit test that builds the parser over two
  different episode subsets that overlap, and asserts `sample_uid` matches for the
  shared frames.
- **Silent giant fan-out** if `episodes=0` resolves to 100k. Mitigation: require
  an explicit episode cap; `log()` the partition plan; fail if #partitions > a
  sane bound without an override.
- **Cross-dataset merge fairness** (separate, tracked): weighted interleaver
  (already scoped) — not required for Milestone 1 (single dataset, many partitions).
- **Eval leakage** if the split is per-frame (fixed): split by `split_group_uid`
  (episode/clip), NOT per-`__key__` — correlated intra-episode frames must not
  straddle train/val (§3.1). Mitigation: `train_group_uids.isdisjoint(val_group_uids)`
  test. Map-task ordering is otherwise safe: the group-hash split is
  order-independent and associates by `partition_id`, not list position.

---

## 5. Phased plan (revised per review)
- **Phase 0 (done):** P0 single-pod fixes — num_workers + /dev/shm + the webdataset
  double-split data-loss fix + ingest hardlink + label mem. Independently correct.
- **Phase 1 — data contracts + global sample_id (no fan-out):** add the four data
  contracts — `SampleIdentity`/`sample_uid`, `split_group_uid`, and (stubs for)
  `DatasetSnapshot`/`PartitionSpec`; swap the 3 uid call sites
  (`parallel_label.py:86`, pack worker return, `workflows.py:450`); switch the
  loader split from per-`__key__` to `split_group_uid` (`pre_extracted._split_bucket`);
  make the JOIN EXACT (`set(pack)==set(label)`); UID-format validation. Verify the
  single-pod pipeline is unchanged by **semantic (per-uid) comparison, NOT tar
  byte-diff** (§6).
- **Phase 2 — per-partition ingest + label/pack fan-out + caching + teacher
  concurrency:** `@dynamic plan_partitions` (cost-based, guarded) → `map_task` over
  partitions for ALL THREE stages: `data_ingest_range` materializes each
  partition's raw `FlyteDirectory` (fetched once from source), then label + pack
  read it. Add `cache=True` + `DatasetSnapshot`/prompt-hash/schema-version
  provenance (§3.4a); bound teacher concurrency via `map_task(concurrency=)` +
  in-pod workers + retry/backoff (§3.2-2). Validate on a 2-partition run first,
  then 20–50 episodes: no stage OOMs; each partition's raw is fetched from HF/SDK
  exactly once.
- **Phase 3 — DatasetManifest reducer + group-level eval split + multi-dir
  train/eval:** add `validate_and_publish_manifest` (coverage/no-overlap/uid-dup/
  label==pack/checksum); `_select_shard_dir` → consume all shard dirs; eval on the
  disjoint group-level `val`.
- **Phase 4 — full-scale run + S3 retention cleanup:** run ALL L2D episodes + all
  NVIDIA clips → train → held-out eval → report ADE/FDE; retire old S3 state via
  the retention/Lifecycle flow (§3.4b).
- **Phase 5 — (only if needed) DDP:** multi-GPU (Kueue GPU quota→N, g6e→~10 nodes,
  `find_unused_parameters=True`, WebDataset `split_by_node`) only if single-GPU
  wall-clock is the bottleneck at full data scale.

## 6. Test plan (revised per review)
Do NOT byte-diff tar shards (member order / mtime / impl differences make
semantically-identical shards differ). Compare per-uid, semantically:
```python
assert old_rec.keys() == new_rec.keys()
assert content_hash(old_rec["cam_0.jpg"]) == content_hash(new_rec["cam_0.jpg"])
assert old_rec["reasoning.json"] == new_rec["reasoning.json"]
```
Required tests:
- same physical frame from two DIFFERENT episode subsets → identical `sample_uid`.
- every uid matches `[A-Za-z0-9_-]+`, no `.`/`/`.
- `plan_partitions` deterministic; partitions cover all groups with no overlap/gap.
- a group never appears in both train and val (`isdisjoint`).
- changing `source_revision` / `prompt_body_hash` / `parser_version` → cache MISS.
- forcing ONE partition to OOM retries only that partition (stage isolation).
- teacher 429 → global in-flight stays ≤ the configured cap.
- `set(pack_uids) == set(label_uids)` after pack.
- 10-ep old vs new pipeline are SEMANTICALLY identical (per-uid, above).
- **Contract stability (§3.4c):** every version is a single named constant in
  `CONTRACT_VERSIONS` (no inline literals — assert by grep/import); NO
  runtime/tuning knob (episodes count, num_workers, label_workers, map_concurrency,
  resource limits, execution id) appears in any cached task's input signature — so
  tuning never invalidates the cache and "ingest once, never again" holds.

---

## 7. What we are NOT doing (and why)
- Not raising single-pod memory further — that is the band-aid this design
  replaces.
- Not DDP in this milestone — training is not the current bottleneck at 10–50
  episodes; data-prep is. DDP is Phase 5, only if needed.
- Not a full `DatasetAdapter` protocol refactor — only 2 datasets (L2D + NVIDIA),
  both already wired, so the `if dataset == …` branches are acceptable; the four
  data CONTRACTS (SampleIdentity/split_group/DatasetSnapshot/DatasetManifest) give
  the reproducibility benefit without the interface churn. Revisit if a 3rd
  dataset is added.
- Not changing the model or losses — this is purely a data-pipeline scaling design.
