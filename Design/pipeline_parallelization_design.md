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

New/changed Flyte structure (using `@dynamic` to compute partitions at run time,
then `map_task` over them):

```
@dynamic
def wf_create_dataset_sharded(dataset, episodes, partition_size, world_model, teacher, prompt):
    partitions = make_partitions(dataset, episodes, partition_size)   # list[(start,end)] or list[list[clip]]
    # 1) INGEST fan-out: each pod ingests only its range → its own raw FlyteDirectory
    raws = map_task(data_ingest_range)(partition=partitions, dataset=..., ...)
    # 2) LABEL fan-out: each pod labels only its raw range → per-range records.jsonl
    #    (writes to the SAME S3 cache; global sample_uid keeps keys correct)
    label_dirs = map_task(generate_reasoning_labels_range)(raw=raws, partition=partitions, ...)
    # 3) PACK fan-out: each pod packs only its raw range + joins its label range
    #    → its OWN shard dir (train-*.tar). Emits K shard dirs.
    shard_dirs = map_task(data_processing_range)(raw=raws, labels=label_dirs, partition=partitions, ...)
    return shard_dirs   # List[FlyteDirectory] — train_il already consumes a list
```

Per-stage detail:

**(1) `data_ingest_range(partition)`** — one pod per range, from the START
(review pt 3/4: do NOT keep a single giant raw dir that every pack pod
re-downloads). Each pod ingests ONLY its slice → per-range raw `FlyteDirectory`.
- L2D: `LeRobotDataset(episodes=list(range(start,end)))` — lerobot fetches only
  the requested episodes (`l2d/dataset.py:157`), so memory/disk scale with
  `partition_size`, not total episodes → the 50-ep ingest OOM disappears.
- Keep the hardlink (`os.link`) copytree fix. The pack pod for the same partition
  downloads only its own small raw slice, not the whole corpus (K× blow-up
  avoided).

**(2) `generate_reasoning_labels_range(raw, partition)`** — one pod per range.
- Builds the parser on its range only; labels with global `sample_uid`.
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
- Packs its range into its OWN shard files, JOINs only its range's labels by uid.
- WM worker cap (6) stays per-pod but now bounds a small range, not the whole set.
- Emits a per-range shard `FlyteDirectory`.

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

### 3.4 Migration / backward-compat of the label cache
Changing `sample_id` from `s{si:08d}` to the global uid changes every cache KEY,
so the ~1000 already-cached L2D labels under the old positional keys become
unreachable (cache miss → re-bill Cosmos for episodes 0–9).
Options (decide in review):
- (a) Accept the one-time re-label of the already-cached episodes (cheap: ~1000
  samples ≈ minutes; simplest, cleanest going forward).
- (b) Write a one-shot migration that re-keys existing cache objects from
  positional → uid (needs the old→new mapping, which requires re-enumerating the
  exact old episode set; brittle).
Recommendation: **(a)** — the cache is an optimization, the teacher is cheap at
this scale, and (a) avoids carrying a fragile remap. Bump the `prompt_version`?
No — same prompt; just let the new keys populate.

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
  + `prompt_body_hash` + `prompt_version` + decode params. Combined with the
  per-sample S3 label cache, an unchanged range is a task-cache no-op; a changed
  prompt/model correctly misses. (This is why `prompt_version` alone was fragile.)
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

### 3.5 Open design questions — RESOLVED in review
1. **Ingest↔pack coupling → SEPARATE (do NOT co-locate).** Keep ingest, label,
   pack as distinct Flyte tasks so each retries independently. Raw round-trips
   through S3 per range; Flyte caching (§3.4a) makes the re-download cheap/skipped.
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
- **Phase 2 — full fan-out (ingest+label+pack) + caching + teacher concurrency:**
  `@dynamic` `plan_partitions` (cost-based, guarded) + `map_task` over ranges for
  ALL THREE stages (ingest per-range from the start — no single giant raw dir);
  add `cache=True` + `DatasetSnapshot`/prompt-hash/schema-version provenance
  (§3.4a); bound teacher concurrency via `map_task(concurrency=)` + in-pod workers
  + retry/backoff (§3.2-2). Validate on a 2-partition run first, then 20–50
  episodes: no stage OOMs.
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
