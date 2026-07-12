# Design: Episode-Sharded, Map-Task-Parallel Data Pipeline (#121)

Status: DECISIONS LOCKED (2026-07-13 review) — ready to implement Phase 1.
Scope: make the AutoE2E data pipeline (ingest → reasoning-label → pack → train →
eval) scale to ALL episodes of **L2D and KitScenes** by fanning each data-prep
stage out across many pods instead of one, so memory/time stop being a function
of total episode count.

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
   L2D AND KitScenes** (NVIDIA is not the second dataset — KitScenes is).
6. **sample_uid scheme approved:** `l2d-e{episode}-f{frame}`, KitScenes
   `kit-{scene_id}-f{frame}`, (NVIDIA `nv-{clip_uuid}-{idx}` kept for parity).

### Added scope surfaced in review
- **KitScenes is not wired into the pipeline yet.** `KitScenesDataset` exists
  (`Model/data_parsing/kit_scenes/dataset.py`, identity `(scene_id, frame_idx)`,
  `:64,70`), but the `Dataset` enum (`workflows.py:70-72`) and
  `data_ingest`/`data_processing` only handle L2D + NVIDIA. Wiring KitScenes
  (enum entry, ingest branch, parser build in label/pack, geometry) is a
  prerequisite for "L2D + KitScenes all episodes" and is added to the plan
  (Phase 2.5).

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
- KitScenes: `kit-{scene_id}-f{frame_idx:06d}`
  (identity `(scene_id, frame_idx)`, `kit_scenes/dataset.py:64,70`).
- NVIDIA: `nv-{clip_uuid}-{sample_idx:06d}`
  (`nvidia_physical_ai/dataset.py:120`; kept for parity though NVIDIA is not a
  target dataset for this milestone).

Constraint: the uid becomes the WebDataset `__key__` (the part of a tar member
name before the first `.`), so it MUST contain no `.`. All three schemes use only
`-`/hex, so they are safe. `scene_id`/`clip_uuid` are UUIDs (hex + `-`) — safe.

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

Consistency guard (`workflows.py:322-332`) currently asserts "same episodes ⇒
same sample_ids"; with a global uid it becomes "the label set covers the pack
set" — reframe as a uid-membership check, not positional-count equality.

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

**(1) `data_ingest_range(partition)`** — one pod per range.
- L2D: `LeRobotDataset(episodes=list(range(start,end)))` — lerobot already fetches
  only requested episodes (`l2d/dataset.py:157`), so a range pod downloads only its
  slice. Memory/disk now scale with `partition_size`, not total episodes → the
  50-ep ingest OOM disappears.
- Keep the hardlink (`os.link`) copytree fix.
- Returns a per-range `FlyteDirectory`. NOTE: raw is still uploaded to S3 per
  range; the pack pod for the same range re-downloads only its slice.
- Open question (3.5): can we skip the raw round-trip by co-locating ingest+pack
  in one range task? (fewer moving parts, less S3 I/O).

**(2) `generate_reasoning_labels_range(raw, partition)`** — one pod per range.
- Builds the parser on its range only; labels with global `sample_uid`.
- Writes to the SAME S3 label cache prefix (unchanged) — cache hits across runs
  still work because the uid is global. Emits a per-range `records.jsonl`.
- `label_workers` per pod can stay modest (12) because each pod handles few
  samples; total concurrency = (#pods × 12), which is how we actually parallelize
  the ~12 s teacher calls across the 10 Cosmos replicas.
- Robustness: add teacher retry/backoff (currently none, `openai_compatible.py`),
  and the >50% abstain guard stays per-range.

**(3) `data_processing_range(raw, labels, partition)`** — one pod per range.
- Packs its range into its OWN shard files, JOINs only its range's labels by uid.
- WM worker cap (6) stays per-pod but now bounds a small range, not the whole set.
- Emits a per-range shard `FlyteDirectory`.

**Combine:** `@dynamic` returns `List[FlyteDirectory]` (K shard dirs). `train_il`
consumes the list unchanged. Eval likewise (may need `_select_shard_dir` →
multi-dir; see 3.5).

### 3.3 Partitioning function
`make_partitions(dataset, episodes, partition_size)`:
- L2D: `[(i, min(i+partition_size, episodes)) for i in range(0, episodes, partition_size)]`.
  With `episodes=0` (= all), first resolve the true episode count (HF meta:
  100,000 for L2D — so "all" needs an explicit cap or a max-episodes arg; do NOT
  silently fan out 10k pods).
- NVIDIA: chunk the discovered clip-uuid list into sublists of `partition_size`.
- Emit `log()` of partition count + size so a huge fan-out is visible, never
  silent.

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

### 3.4a Flyte caching (skip unchanged ranges)
Because the stages are kept separate (decision 2) and fan out per range, a re-run
(e.g. after a downstream failure, or extending the episode set) should NOT redo
work whose inputs didn't change. Add Flyte task caching:
- `data_ingest` (per range): `@task(cache=True, cache_version="v1")`. A given
  `(dataset, partition)` re-ingests only when the code/version changes. lerobot
  already caches the HF download on disk; Flyte caching skips the whole task.
- `generate_reasoning_labels` (per range): `cache=True`. Combined with the
  per-sample S3 label cache, an unchanged range is a no-op (task cache hit) and
  even a changed range only calls the teacher for uncached uids.
- `data_processing` (per range): `cache=True`, `cache_version` bumped whenever the
  shard schema changes.
Cache key = (task signature, input literals, cache_version). Since inputs are the
partition + dataset + flags, ranges are independently cacheable. This is what
makes "extend from 20 → 50 → all episodes" cheap: only the NEW ranges run.
Caveat: `FlyteDirectory` inputs hash by URI, so upstream re-runs that produce a
new raw dir URI will invalidate downstream cache — acceptable, and why ingest
caching (stable raw URI per range) matters most.

### 3.4b S3 cleanup (decision 4)
The cutover leaves unused S3 state that must be deleted:
- Old positional-keyed reasoning-label cache prefix
  (`reasoning_labels_cache/dataset=/teacher=/prompt_version=/s00000000.json` …) —
  superseded by the global-uid keys. Delete after the new keys are populated.
- Orphaned raw/shard `FlyteDirectory` outputs from superseded single-pod runs
  under `s3://…-artifacts-…/` (e.g. the failed 50-ep ingest `raw`, partial
  packs). Enumerate and delete the runs that the new pipeline replaces.
- MUST use the correct account/profile (`--profile autowarefoundation`,
  us-west-2) and confirm each prefix before `rm`; never touch the Cosmos account.
Do the deletion as an explicit, logged step (list → confirm → delete), not silently.

### 3.5 Open design questions — RESOLVED in review
1. **Ingest↔pack coupling → SEPARATE (do NOT co-locate).** Keep ingest, label,
   pack as distinct Flyte tasks so each retries independently. Raw round-trips
   through S3 per range; Flyte caching (§3.4a) makes the re-download cheap/skipped.
2. **Eval multi-dir → eval over ALL shard dirs.** Change `_select_shard_dir` →
   consume the full `List[FlyteDirectory]` (the held-out `val` split already makes
   this a proper generalization measure across all partitions).
3. **Partition size → 10 episodes/pod to start**, tunable; final target is ALL
   L2D + KitScenes episodes.
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
- **Non-determinism** from map-task ordering: shard dirs may complete in any
  order, but WebDataset shuffles anyway and the held-out split is a per-sample
  `__key__` hash (order-independent) — so no train/val leakage.

---

## 5. Phased plan
- **Phase 0 (done):** P0 single-pod fixes — num_workers + /dev/shm + the webdataset
  double-split data-loss fix + ingest hardlink + label mem. Lets moderate scale
  run today and is independently correct.
- **Phase 1 — global sample_id (no fan-out):** add `sample_uid` to L2D + KitScenes
  (+ NVIDIA) parsers; swap the 3 call sites (`parallel_label.py:86`, pack worker
  return, `workflows.py:450`); reframe the consistency guard (`workflows.py:322`)
  to uid-membership; unit test for cross-subset uid stability + JOIN. Verify the
  existing single-pod pipeline still produces identical shards (byte-diff the
  reasoning.json JOIN on the 10-ep shard, modulo the new key strings).
- **Phase 2 — Flyte caching + label/pack fan-out:** add `cache=True`/`cache_version`
  to the three data-prep tasks (§3.4a); `@dynamic` computes partitions +
  `map_task` over ranges for label + pack (ingest still single for now). Validate
  on a 2-partition run first, then 20–50 episodes: no stage OOMs, K shard dirs
  train correctly.
- **Phase 2.5 — wire KitScenes into the pipeline:** add the `Dataset` enum entry,
  the `data_ingest` KitScenes branch, and the parser build in label/pack + its
  geometry/calib. Then both L2D and KitScenes flow through the sharded pipeline.
- **Phase 3 — ingest fan-out + eval multi-dir + S3 cleanup + partition tuning:**
  fan out ingest per range for full episode-count independence; eval over all
  shard dirs (`_select_shard_dir` → plural); delete superseded S3 dirs + old
  positional cache prefix (§3.4b); choose partition size.
- **Phase 4 — full-scale run + (separate) DDP:** run ALL L2D + KitScenes episodes
  through the sharded pipeline → train → held-out eval → report ADE/FDE. DDP
  multi-GPU training (Kueue GPU quota→N, g6e→~10 nodes, find_unused_parameters,
  split_by_node) only if single-GPU wall-clock becomes the bottleneck at that
  data scale.

---

## 6. What we are NOT doing (and why)
- Not raising single-pod memory further — that is the band-aid this design
  replaces.
- Not DDP in this milestone — training is not the current bottleneck at 10–50
  episodes; data-prep is. DDP is Phase 4.
- Not changing the model or losses — this is purely a data-pipeline scaling design.
