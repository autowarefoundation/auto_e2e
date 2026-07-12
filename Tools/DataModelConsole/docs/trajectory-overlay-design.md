# Design Doc — AutoE2E DataModelConsole: Per-Model Trajectory Overlays + GPS Map / ODD Geo-Stats

Status: Design (Research Phase — breaking schema changes permitted per project CLAUDE.md)
Owning code: `Tools/DataModelConsole` (Go API + Next.js) and `Platform/pipelines/workflows.py` (Flyte)
Author: design pass grounded in `docs/.traj_brief.json` + source read of `Model/model_components/*`, `Model/evaluation/metrics.py`, `Model/data_parsing/l2d/egomotion.py`, packer, and the console store; revised against four adversarial reviews (adas, infra-dynamo, flyte-feasibility, cost-storage-frontend).

---

## Decisions locked by the user (2026-07-13)

These OVERRIDE any contradicting recommendation later in this doc; the body below is annotated where a prior recommendation was reversed.

1. **Playback representation = VECTOR-FIRST (accepted as recommended).** Store raw `(64,2)` control + `v0`, draw client-side; baking demoted to optional MP4 export. (§7, §0.3–4)
2. **Predicted trajectory ON the geographic (GPS) map = IN SCOPE** — REVERSES the doc's default (was "out of scope / driven-path only"). Requires an explicit error budget: yaw-sign convention (§10), lat/lon float precision, ego-heading→map-bearing conversion, and pseudo-geometry. See §9-bis (predicted-on-map) added below. (Open Question 5 resolved: YES.)
3. **GPS packing = FULL RE-PACK to a new dataset version `v2.1`** — REVERSES the doc's preferred "decode-free in-place backfill into v2.0". Consequence (MANDATORY): overlays (`PRED#`), the playback index (`IDX#`), and label-search resolution (`ResolveSampleShards`) MUST ALL move to `v2.1`, and `sample_id = s{si:08d}` enumeration MUST be asserted byte-stable across the re-pack (same parser + episode order) so keys don't shift. (§4b, §consistency, Open Question 4 resolved: v2.1.)
4. **Commit this doc; DEFER implementation** until an explicit go-ahead. (Open Question: commit-only.)

---

## 0. TL;DR of the big decisions (read this first)

1. **Store the RAW control prediction, not integrated XY.** Overlays persist `(64,2) [accel_x, curvature] + v0` — exactly the `ego_future` representation the index already carries — and the client integrates it with the *same* integrator + clamps used for the GT plan. This makes prediction and GT rendered identically, keeps any future fix to `integrate_trajectory` / the yaw-sign / the curvature clamp a **pure render change with zero GPU recompute**, and shrinks the payload. (adas #2, #9)
2. **Playback artifact = one gzip blob per (model, shard), not per-sample edges.** Key `PRED#{run_id}#{dataset}#{version}#{shard}/META`, mirroring `ShardIndex`: one DynamoDB GetItem loads every sample's overlay for a whole shard → smooth scrubbing. This is the brief's own recommended Option A; the draft's per-sample inline-`t` edge is rejected (it turns episode playback into ~1000 GetItems / paginated ~1 MB reads). (infra-dynamo #1, brief Option A)
3. **Recommendation: VECTOR-FIRST for all three sources AND both views.** The one thing baked frames appeared to buy — camera projection under L2D's `geometry_type='pseudo'` — is **model-independent** (it's a function of ego pose/geometry, not weights). So we precompute a per-sample camera-projection artifact **once**, and the client draws *any* model's polyline in camera space from the same tiny vector blob. This removes baking's sole advantage and saves ~3–4 orders of magnitude of storage. (cost-frontend #2, #6)
4. **PR#74 baking is DEMOTED to an optional offline export** (thumbnails / shareable clips), never the playback path. The user leaned pre-rendered; §7 presents the honest storage/flexibility math and states exactly when baking is the right call (it is not, for the interactive per-model picker). If ever used, it emits **one MP4 per scene**, not 64 loose JPEGs.
5. **Model identity is `run_id`, not the MLflow registry version** (the version is a moving "latest" pointer). All keys use `run_id`; `mlflow_version` is a stored attribute, and a tiny `VER#{version}→run_id` alias item resolves the picker. (infra-dynamo #4)
6. **No model-side code change is required for determinism.** `generator=` already threads `AutoE2E.forward(**kwargs) → Reactive_E2E(**kwargs) → FlowMatchingPlanner.forward(generator=…)` (verified in source). The draft's "gap #3 / Phase-2 model edit / top risk" is deleted. (flyte #1)

---

## 1. Goal & Scope

Give the console a **Scene view** in which a user picks a **Model** and sees **that model's predicted trajectory** overlaid on the scene's camera + BEV frames, with **very smooth playback**. This must work for **three scene sources**, all of which resolve to the same `(dataset, version, shard, sampleKey)` tuple space (grounded: all three are WebDataset shard samples, differing only in which sample keys are selected):

- **(a) Leaked training set** — the shards the model trained on. Because `training.val_fraction` defaults to `0.0`, by default *every* sample is training-leaked. Train-vs-eval is a per-sample `blake2b(__key__, digest_size=8) mod 10` bucket, **not** a separate shard set.
- **(b) Eval set** — the held-out `val` bucket (only non-empty when the run used `val_fraction > 0`). **Honesty requirement (adas #7):** this split is a per-frame hash over the *same episodes*, so held-out frames are temporally/spatially adjacent to training frames from the same drive (`splits.py` geographic/episode holdout is unwired). The UI MUST label this "in-distribution / near-duplicate hold-out," not "generalization."
- **(c) Reasoning (Action-Relevance) label search** — scenes from the existing `LBL#` index (`GET /api/v1/scenes/search`), then overlaid with the picked model's trajectory.

Also in scope (new requirement):
- **Pack GPS** `lat/lon` (L2D raw cols 3–4: `hp_loc_latitude`, `hp_loc_longitude`; the packer today keeps only the derived 4-ch ego `[speed, accel_x, yaw_rate, curvature]` and never surfaces lat/lon out of the loader).
- **Map view**: draw the driven path on a real map.
- **ODD geo-statistics**: "where was this data collected."

**In scope (user decision 2, 2026-07-13):** overlaying the *predicted* trajectory on the real geographic (GPS) map, in addition to the camera/BEV views. This requires the error budget in §9-bis (yaw-sign, float precision, heading→bearing, pseudo-geometry).

**Out of scope:** on-demand (runtime) inference in the Go API — infeasible (checkpoint is ~509 MiB `best.pt`, loadable only via Python `AutoE2E(**_model_kwargs(config))`; Go cannot load `.pt`). All inference is **heavy Flyte-side precompute**, which the user explicitly accepts.

---

## 2. Current-State Summary

### Verified facts (source-read)
- **Forward contract:** `AutoE2E(...)(camera_tiles, map_input, visual_history, egomotion_history, projection=…, geometry_type=…, mode="infer", **kwargs)` returns a bare `[B,128]` = `(64,2)` = `[accel_x (m/s²), curvature (1/m)]` control at 10 Hz over 6.4 s — **NOT XY**. `mode="infer"` returns the bare tensor (no aux dict). Reasoning branch does not change the output shape.
- **`generator` already threads end-to-end (flyte #1, verified):** `auto_e2e.py:94–97,191` forwards `**kwargs`; `reactive_e2e.py:104–106,166–170` forwards `**kwargs` into `self.TrajectoryPlanner(...)`; `flow_matching_planner.py:318–320,348–350` consumes `generator=` at `torch.randn(B, self.trajectory_dim, …, generator=generator)`. `BezierPlanner.forward` accepts/ignores `**kwargs`. **Zero model edits needed.**
- **Integration:** `metrics.py::integrate_trajectory(accel, curvature, v0, theta0=0.0, dt=0.1)` → `(T,2)` `[x_forward, y_left]` metres, ego frame. Loop: `theta += curvature[t]·v·dt; x += v·cos θ·dt; y += v·sin θ·dt`. `v0 = ego_history.reshape(64,4)[-1,0]` (speed channel, last history step).
- **Egomotion channels (verified `egomotion.py`):** derived signals are `[speed, accel_x, yaw_rate, curvature]` (channel 3 = curvature, **not** yaw angle — the README wording is stale; cite code). `_derive_signals` builds `speed = raw/3.6` (km/h→m/s), `heading = unwrap(radians(...))`, `yaw_rate = diff(heading)/dt`, `curvature = yaw_rate/max(speed, 0.5)`, then `clip(curvature, ±0.5)`. GT is thus **speed-floored (0.5 m/s) and curvature-clamped (±0.5 rad/m)**; the model prediction is **not**.
- **FlowMatchingPlanner is stochastic** (`torch.randn(..., generator=generator)`); Bezier is deterministic and ignores the generator.
- **MLflow linkage:** one registered model `auto-e2e-driving-policy`; a "Model ID" = a registry **version** whose durable pointer is `run_id`. Latest observed: **v30 → run_id `b457606594204ac88e3e1a0fe09075f5`**. Checkpoint: `s3://auto-e2e-platform-artifacts-381491877296/mlflow/8/{run_id}/artifacts/model/best.pt` (~509 MiB) + sibling `config.yaml` (1951 B). `.pt = {model_state_dict, config, epoch}`; rebuild via `AutoE2E(**_model_kwargs(config))`. Run params link model→dataset (`data/dataset`), Flyte exec ids, eval metrics (`eval/ade`, `eval/fde`, `eval/gate_pass`). `training.val_fraction` is **in `config.yaml` and `metadata.json` but NOT a logged MLflow param** (flyte #11).
- **DynamoDB single-table `auto-e2e-console`** (`store/keys.go`): `pk` HASH + `sk` RANGE (String), plus a `gsi1` that **appears only in a doc comment** — no Go code sets `gsi1pk/gsi1sk`, no `IndexName` query, and no `CreateTable`/Terraform in-repo. The index is created out-of-band; **its KeySchema and ProjectionType are unverified.** Existing items:
  - `IDX#{dataset}#{version}#{shard}` / `META` → gzip `ShardIndex` (playback source: `fps`, per-sample `members[suffix]→{offset,size}`, `ego_now[4]`, `ego_history`, `ego_future[128]`, `has_reasoning`).
  - `STATS#{dataset}#{version}#{promptVersion}` / `META` → reasoning stats blob.
  - `LBL#{dataset}#{promptVersion}#{field}#{value}` / `SCENE#{sampleID}` → scene-by-label index.
- **Playback mechanism:** frontend fetches each JPEG via S3 byte-range GET (`StreamTarMemberRange`, `MaxRangeBytes = 32 MiB`) using `members[suffix].{offset,size}` from `ShardIndex`. **A "scene" sample is one 10 Hz frame; an episode plays consecutive samples** (per-shard `frame_idx == global sample idx`). `ego_future` is the per-frame GT the BEV already integrates+draws — the model overlay is its direct analogue.
- **Go API** is a READ-ONLY MLflow/Flyte proxy + S3/Dynamo reader. `MLflowModelVersion` normalization **drops the `source` artifact URI** and exposes no `model-versions/search`, so version→`run_id`→checkpoint mapping is impossible today. Artifacts bucket configured but `mlflow/` prefix unused.
- **`ResolveSampleShards`** resolves label-search sampleIDs against the console's **published** dataset version (`resolveVersion`) — a version-coordinate landmine (§ below).
- **Packer** (`parallel_pack.pack_sample`): writes `ego.npy = concat(ego_history[256], trajectory_target[128])` float32, `cam_i.jpg`, optional `map.jpg`, WM windows, `meta.json`, `calib.json`. `sample_key = f"s{si:08d}"`, 1000 samples/shard, `train-{idx:06d}.tar`. GPS never enters the sample dict.
- **`gps_to_map.py`** renders GPS waypoints on an ego-centric OSM/`osmnx` BEV tile (L2D palette), with equirectangular lat/lon→m helper.

### Concrete GAPS (revised)
1. No standalone single-sample inference / overlay entry point (only `_run_evaluation`, batched, discards integrated XY).
2. No Flyte-free checkpoint-load helper.
3. ~~No seed threading~~ **RESOLVED in code** — only a **fixed-seed *convention*** is missing (which seed(s), where recorded). (flyte #1)
4. No per-sample **camera-projection artifact** for L2D pseudo-geometry (the new model-independent artifact that replaces baking).
5. No model dimension in the key space; `gsi1` unverified/unused.
6. `val_fraction` not a logged param — but **readable now from `config.yaml`** (no re-run needed). (flyte #11)
7. GPS dropped by loader+packer.
8. No geo-stats keying.
9. **Yaw-sign / clamp render contract undefined** (adas #1, #9) — see §10.

---

## 3. Architecture Overview

```
                     ┌──────────────────────── Flyte (GPU, us-west-2, ONE warm L40S) ─────────────────────┐
 MLflow registry     │  wf_precompute_overlays(run_id, dataset, version, source, seed_set)                │
 auto-e2e-driving-   │    resolve run_id (once) ─► GET best.pt ONCE per coarse subtask                     │
   policy v→run_id   │    coarse @task per shard-BATCH (amortize 509 MiB load; NOT map over tiny units)    │
 config.yaml ───────►│      load_policy() ─► for each seed in seed_set:                                     │
   (val_fraction)    │        predict batched ─► [128] control  (NO integration; store raw)                │
 shards (datasets    │      write ONE gzip blob per shard: sampleKey→{accel_curv,v0,seed}                  │
   bucket) ─────────►│      write per-sample CAMERA-PROJECTION artifact (model-independent, once)          │
                     │      write artifacts FIRST → flip OVLSET# status building→ready LAST                 │
                     │  data_processing (MOD): pack gps.npy sidecar; + full-episode GPS path artifact       │
                     │  wf_geo_stats: scan gps → GEO# summary (inline) + geojson (S3 pointer)               │
                     └──────┬───────────────────────────────────────────────────┬──────────────────────────┘
                            │ S3 PUT (idempotent deterministic keys)              │ Put/BatchWrite (chunk 25)
                            ▼                                                     ▼
   S3 auto-e2e-platform-artifacts-381491877296            DynamoDB auto-e2e-console (single table)
     overlays/schema=v1/run_id=…/…/shard=…/proj.f32          PRED#{run_id}#{ds}#{ver}#{shard} / META  (gzip blob)
     overlays_manifest/…/manifest.json                       MODEL#{run_id} / META           (run profile)
   S3 datasets bucket: shards …/gps.npy (sidecar)            VER#{version} / META            (version→run_id)
     + episode_gps/{ds}/{ver}/{episode}.f64  (full path)     SCENELIST#{run_id}#{ds}#{ver}#{shard} / META (sparse)
                            │                                 GEO#{ds}#{ver} / META           (ODD summary)
                            └────────────────► Go API (READ-ONLY++) ◄───────────────────────────┘
                                  new endpoints: models-for-scene, overlay-blob (per shard),
                                  cam-projection, geo-stats, gps-path  (Dynamo + S3; NO inference)
                                                        │
                                                        ▼
                               Next.js: model-picker → two-layer canvas vector overlay (BEV + camera);
                               Map view (driven GPS path); ODD geo-stats page
```

Principle (`bp-relational-modeling`, materialized-index pattern): **playback never runs the model and never scans S3 per frame** — one GetItem loads a shard's overlay blob; the camera-projection artifact is one GET per shard; vectors integrate/draw client-side.

---

## 4. Flyte Pipeline Design

### 4a. `wf_precompute_overlays`

**Location:** `Platform/pipelines/workflows.py`, sibling to `train_il` / `data_processing` / `_run_evaluation`.

**Reusable, Flyte-free helper (fills gap #2).**
```python
# Platform/pipelines/inference.py  (importable, no Flyte deps)
def load_policy(ckpt_path: str, device: str) -> tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = AutoE2E(**_model_kwargs(cfg)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg

def predict_control(model, batch, cfg, *, seed: int, sampler="euler", num_steps=10) -> np.ndarray:
    """Return RAW (B,64,2) [accel_x, curvature] control — NOT integrated XY."""
    dev = batch["egomotion_history"].device
    gen = torch.Generator(device=dev).manual_seed(seed)
    with torch.no_grad():
        pred = model(batch["visual_tiles"], batch["map_input"],
                     batch["visual_history"], batch["egomotion_history"],
                     projection=batch.get("projection"),
                     geometry_type=cfg["model"].get("geometry_type", "pseudo"),
                     mode="infer", generator=gen)          # [B,128]; generator no-op for Bezier
    return pred.reshape(pred.shape[0], 64, 2).cpu().numpy()
```
> **Do NOT integrate here.** We persist raw control + `v0` and integrate on the client with the shared, fixable integrator (see §10). This is the single most important change vs the draft (adas #2). `cfg_geom`/`batch["projection"]` keys MUST be confirmed against `make_pre_extracted_loader` output before merge (flyte #12a); if the loader omits `projection`/`geometry_type`, thread them from `calib.json` / `cfg`.

**Coarse subtasking to amortize the 509 MiB load (flyte #2, #3).** Flyte `map_task` subtasks are independent executions on fresh pods — **not** a warm Ray actor pool; a fine-grained map would re-download `best.pt` and re-instantiate `AutoE2E` per subtask. Since the platform runs **exactly one warm g6e.4xlarge (1× L40S, do-not-disrupt)** and scale-up is Karpenter+Kueue against a single ODCR, parallelism is near-serial anyway. Decision: **one coarse `@task` per shard (or per small shard-batch), loading the checkpoint ONCE and streaming all that shard's samples through batched forward.** No `map_task` fan-out over tiny units; the Ray-actor citation from the draft is removed as non-transferable.

```python
@dataclass
class OverlayShardJob:
    run_id: str                 # DURABLE model identity (resolved once by the workflow)
    dataset: str                # "l2d" | "nvidia_av"
    version: str                # published shard version — MUST equal IDX#/search version (§ consistency)
    shard: str                  # "train-000000.tar"
    sample_keys_hash: str       # blake2b of the sorted key subset (NOT the list — caching guidance)
    key_bucket_spec: str        # how to re-derive keys inside the task (e.g. "split:eval:vf=0.2" | "search:<token>")
    split_role: str             # "train" | "eval" | "search"
    seeds: tuple[int, ...] = (0,)   # seed-fan; single-sample FlowMatching is misleading (adas #4)
    sampler: str = "euler"
    num_steps: int = 10
    overlay_schema: str = "v1"

@task(requests=Resources(gpu="1", mem="24Gi"), cache=True,
      cache_version="{OVERLAY_CODE_VERSION}",   # injected at build time (see caching)
      cache_serialize=True)
def precompute_overlay_shard(job: OverlayShardJob) -> OverlayShardResult: ...

@workflow
def wf_precompute_overlays(model_version: int, dataset: str, version: str,
                           source: str = "eval",            # "train"|"eval"|"search"
                           prompt_version: str = "", field: str = "", value: str = "",
                           seeds: tuple[int, ...] = (0,)) -> OverlaySetManifest: ...
```

**Inside the task:**
1. Workflow resolves `run_id` once (MLflow `model-versions/get`) and reads `val_fraction` from `config.yaml` (no re-run needed — flyte #11). Download `best.pt` once; `load_policy`.
2. Build `make_pre_extracted_loader` over the shard, `dataset.select(...)` to the keys re-derived from `key_bucket_spec` (identical decode path for all three sources).
3. **Idempotency/resume:** HEAD the target S3 blob(s); skip if present and `overlay_schema`+seed set match (mirrors reasoning labeler `n_hit`/`n_computed`).
4. `predict_control` in **batches** (start `batch_size=32`, halve on OOM), overlapping CPU decode with GPU forward. For a seed-fan, run each seed; store all.
5. Emit per shard:
   - **ONE gzip overlay blob** (§6) mapping `sampleKey → {ac: (64,2) accel_curv, v0, per-seed variants}`.
   - **ONE per-sample camera-projection artifact** `proj.f32` per shard (model-independent; written once, reused by every model — see §5/§8).
6. **Write artifacts FIRST, index LAST:** PUT S3 blobs, then `PutItem` the `PRED#…/META` (gzip payload, one call per shard, paralleling `PutShardIndex`/`PutStats`), `BatchWriteItem` the sparse `SCENELIST#` rows (chunks of 25 + `UnprocessedItems` retry, like `PutSceneLabels`), and finally flip `OVLSET#` `building→ready`.

**Caching (flyte #4, infra #10).** `cache_version` is a **static string fixed at registration**; Flyte does not hash your source. A **build-time step MUST inject the git SHA / module hash** into `cache_version` (e.g. via env-substituted registration), or a planner/integration change silently serves stale overlays. Inputs are **scalars + `sample_keys_hash`** (never a `list[str]` of up to 1000 keys, never a `FlyteDirectory`) so the catalog key stays tight. `cache_serialize=True` prevents double-billing the single GPU on concurrent "compute" clicks. Since we store **raw control**, a later fix to the *integrator/yaw-sign/clamp* is a client change and does **not** bump `cache_version` at all.

**Determinism reproducibility (adas #13).** For byte-identical re-runs behind a `cache_version` bump, set `torch.use_deterministic_algorithms(True)` + `torch.backends.cudnn.deterministic=True; benchmark=False` in the task (Swin backbone atomics/autotune otherwise vary). If we accept "numerically-close, not bit-identical," document it and skip — but then don't claim byte-reproducibility.

**Seed policy (adas #4).** A single FlowMatching draw is one sample from the predictive distribution and can sit in the tail — misleading for a comparison tool. Default `seeds=(0,)` but **the overlay blob carries the seed(s)** and the UI must label "single sample, seed 0." Provide a `seeds=(0,1,2,3)` fan option; the client can then draw the median path + envelope. Bezier ignores seeds (store once).

**Resolving the three sources → keys (gaps #1, #6; leak facts):**
- **train/eval:** read `val_fraction` from `config.yaml`; per sample `blake2b(key, digest_size=8) mod 10`; `bucket < round(val_fraction*10)` ⇒ `eval` else `train`. If `val_fraction==0`, `eval` is empty and all samples are `train`. Also cheaply **log `val_fraction` as an MLflow param** going forward (small `train_il`/`_run_evaluation` change) so the picker doesn't parse an artifact per model.
- **search:** workflow calls the same resolution the Go API uses (`QueryScenesByLabel → ResolveSampleShards`) for `(dataset, prompt_version, field, value)` → `(shard, sampleKey)`, groups by shard. `split_role="search"`.

### Version-coordinate consistency (infra-dynamo #3) — MANDATORY invariant

Overlay `PRED#` keys, the playback `IDX#`, and `ResolveSampleShards`' published version **MUST all be the same `version` string**. **User decision 3: GPS is a full re-pack to `v2.1`** (§4b), so ALL of overlays (`PRED#`), the playback index (`IDX#`), and label-search resolution MUST run at **`v2.1`** — publish `v2.1` shards, rebuild `IDX#` at `v2.1`, point `resolveVersion`/search at `v2.1`, and run `wf_precompute_overlays` with `version="v2.1"`. **MANDATORY: assert `sample_id = s{si:08d}` enumeration is byte-stable across the re-pack** (identical parser + episode order + episode count); if the enumeration shifts, every `SCENE#…`/`PRED#…` key silently mis-maps. A single shared key-builder is the source of truth for the composite key so P4 GetItems never mis-key (infra #14). Reasoning labels are keyed by `prompt_version` (not dataset version), so they are unaffected by the v2.0→v2.1 move.

### 4b. Data-gen change: PACK lat/lon (gap #7)

**Loader plumbing is new (flyte #8, #10).** `L2DDataset.__getitem__` builds egomotion via `extract_egomotion` and **never surfaces lat/lon** today. Add `sample["gps_latlon"]` by slicing raw `vehicle_states[:, 3:5]` (`hp_loc_latitude`, `hp_loc_longitude`) aligned to the **same `sample_idx`** `extract_egomotion` chooses. Note the window is `history [idx-64:idx]` + `future [idx+1:idx+65]` = **64+64**, not 65 — the draft's "(65,2)" was ad-hoc.

**Two distinct GPS products (flyte #8):**
1. **Per-sample sidecar `gps.npy`** for the overlay/scene view: the *current-frame* absolute position + the sample's future window `[lat, lon]`, shape `(1+64, 2)` aligned to `idx` and `idx+1..idx+64`. This is enough for per-scene "where am I + where am I going."
2. **Per-episode full path artifact** for the Map view driven route: the *whole episode's* lat/lon, written once per episode as `episode_gps/{dataset}/{version}/{episode_id}.f64`. A 6.4 s future window cannot draw a driven route; the map needs the full drive.

**Precision (adas #8, cost #9): store lat/lon as float64.** float32's 24-bit mantissa gives ~7 significant digits; `48.8566xx` spends 2–3 before the decimal, leaving ~1–2 m of quantization jitter — visible when animating an ego marker. Store `gps.npy`/episode path as **float64** (or int32 fixed-point ENU offset from a per-episode float64 origin). This is the one field where the `ego.npy`-style float32 is wrong.

```python
# parallel_pack.pack_sample, after ego.npy:
gps = sample.get("gps_latlon")           # (65,2) float64 [lat, lon]; loader provides
if gps is not None:
    members["gps.npy"] = np.asarray(gps, np.float64).tobytes()  # + npy header
```
NVIDIA has no GPS ⇒ member simply absent (`ShardIndex.has_gps=False`).

**Backfill = FULL re-pack to `v2.1` (user decision 3).** A decode-free in-place append into `v2.0` would be cheaper (GPS needs no video decode — it comes from `_get_vehicle_states_window`), but the user chose a clean full `data_processing` re-pack producing a NEW version `v2.1`. Implications to budget:
- The re-pack re-decodes + re-encodes all 7 L2D cameras per sample (PyAV decode + JPEG re-encode dominate cost) — a heavy Flyte run; give a wall-clock/cost estimate before triggering.
- **Everything downstream moves to `v2.1`** (see §consistency): publish `v2.1` shards, rebuild `IDX#` at `v2.1`, repoint `resolveVersion`/search, and run overlays at `v2.1`. `v2.0` can be retired after cutover (S3 lifecycle).
- **The re-pack MUST preserve `sample_id = s{si:08d}` enumeration byte-for-byte** (same parser, episode list, episode order, samples/shard) so existing reasoning-label sample_ids (which are prompt_version-keyed and version-agnostic) still line up with `v2.1` frames. Add a test asserting v2.0 vs v2.1 key sets are identical for the same episodes.
- Opportunity: since we re-pack anyway, also drop in any other additive members (e.g. the per-sample `proj.f32` inputs) in the same pass.

**Index (flyte #10):** populating `ShardIndex.has_gps` + a denormalized `IndexSample.gps_now[2]` (float64) requires **new Go npy-decode logic** in `BuildShardIndex` (parse npy header + little-endian floats), analogous to the existing `ego_now/ego_future` extraction — and a rebuild of affected `IDX#` items. This is real Go work, not a free side effect; scope it. `gps_now` is used only for cheap ODD reads (see §9 caveat about not scanning gzip blobs).

---

## 5. S3 Layout

Bucket `auto-e2e-platform-artifacts-381491877296` (us-west-2). Keys are a pure function of identity so the Go API builds them without a lookup. `schema=v1` sits high so a render/kernel change bumps the S3 path and `cache_version` together.

**Prefix entropy (infra #7).** In one precompute job `run_id/dataset/version/split` are constant, so entropy only appears far-right at `shard=`. To avoid cold-prefix `503 SlowDown` on a large parallel write, put a high-entropy bucket **early**: `overlays/b={blake2b(shard)%16}/schema=v1/run_id=…/…`. (With the single-GPU near-serial reality this is a lesser concern, but cheap insurance.)

**Overlay control + projection (vector-first, per shard):**
```
overlays/b={0..15}/schema=v1/run_id={run_id}/dataset={l2d}/version={v2.0}/split={train|eval|search}/shard={train-000000}/
    control.f32shard         # header + per-sample directory (sampleKey→offset) + (64,2) accel_curv, v0, seed
                             # ALSO stored gzipped inline in the PRED# Dynamo item (§6); this S3 copy is the audit/large fallback
    proj.f32                 # MODEL-INDEPENDENT per-sample camera projection (written ONCE, reused by all models)
```
The **`control.f32shard` has a header offset directory** `sampleKey→{offset,size}` so one GET + in-memory parse serves the shard; seeking one sample never downloads the whole blob (infra #9). The vector key **includes `seed=` in the payload** and the schema segment, so a re-run at a different seed never silently overwrites (infra #9).

`proj.f32` (cost-frontend #2, the decisive artifact): per sample, the parameters mapping ego-frame XY → camera pixels (the same pseudo-projection PR#74 would bake). Because it's a function of ego pose/geometry only, it is stored **once per (dataset, version, shard)** and reused by every model. The client draws any model's polyline in camera space from `control → integrate → proj`.

**Per-run manifest (lineage/audit):**
```
overlays_manifest/schema=v1/run_id={run_id}/dataset={l2d}/version={v2.0}/split={eval}/manifest.json
    {mlflow_version, run_id, dataset, version, split, n_samples, seeds, sampler, num_steps,
     overlay_code_version, cuda_deterministic:bool, created_at, status:"ready"}
```

**Optional baked export (demoted, §7):** if ever produced,
```
overlays_export/schema=v1/run_id=…/…/scene={s00000042}/clip.mp4    # ONE mp4 per scene, NOT 64 JPEGs
```

**GPS** lives in the datasets bucket: per-sample `gps.npy` inside the shard tar, and per-episode `episode_gps/{dataset}/{version}/{episode_id}.f64`.

**Lifecycle (infra #8).** Add an **S3 lifecycle policy** on `overlays/` and `overlays_export/` keyed by `schema=`/`run_id=` so retiring an `overlay_schema` or an experimental run actually reclaims bytes — Dynamo TTL only deletes pointers, not the S3 objects.

---

## 6. DynamoDB Schema Additions

Single table `auto-e2e-console`, prefix-namespaced. **Playback is a per-shard gzip blob (Option A), not per-sample edges.**

| # | Access pattern | Query |
|---|---|---|
| P1 | which models have a ready overlay for **scene Y** | see decision below (`SCENELIST#` or `gsi1` — gated on GSI verification) |
| P2 | **model X's** overlays for a whole **shard** (playback) | base GetItem `pk=PRED#{run_id}#{ds}#{ver}#{shard}`, `sk=META` → gzip blob |
| P3 | all shards **model X** was computed on | base Query `pk=OVLSET#…` / list manifests |
| P4 | reasoning-search scenes + overlay for **model X** | existing `LBL#` Query → group by shard → P2 per shard |

### New item types

**(1) Per-shard overlay blob — the playback artifact (brief Option A; infra #1):**
```
pk      = PRED#{run_id}#{dataset}#{version}#{shard}      e.g. PRED#b457…#l2d#v2.0#train-000000.tar
sk      = META
attrs:
  payload   B   gzip JSON { "s00000042": {"ac":[[a,c]…64], "v0":12.3, "seed":0}, … }  # mirrors ShardIndex
  mlflow_version N 30
  seeds     L   [0]           # or [0,1,2,3] for a fan
  overlay_schema S "v1"
  status    S   "ready"
```
One GetItem loads every sample's overlay for the shard → smooth scrubbing, exactly like `IDX#`. Payload is **raw `(64,2)` control + `v0`** (adas #2), so the client integrates prediction and GT with the identical integrator/clamps and any integrator fix needs no recompute. **Do not gzip the trajectory floats individually** — gzip the whole JSON payload (high-entropy floats don't compress alone; cost-frontend #8), consistent with how `ShardIndex` is stored. Size: 1000 samples × ~520 B raw ≈ 0.5 MB → the same `>400 KB` **gzip trick** used for `ShardIndex` (l2d 1.7 MB→77 KB) keeps it under the 400 KB item cap; if a fan of 4 seeds overflows, split per-seed pk suffix `…#{shard}#seed{n}`.

**(2) Model run profile — run-level metadata (infra #5):**
```
pk = MODEL#{run_id}   sk = META
attrs: mlflow_version N, model_name S, eval_ade N, eval_fde N, eval_gate_pass N,
       dataset S, train_execution_id S, val_fraction N, created_at S
```
`eval_ade`/`model_name` are **run-level, not scene-level** — stored once here, never denormalized onto thousands of edges (the draft duplicated them per edge).

**(3) Version alias (infra #4):**
```
pk = VER#{mlflow_version}   sk = META   attrs: run_id S
```
Picker maps a chosen registry version → durable `run_id`.

**(4) "Which models for this scene" list — sparse (P1).**
Decision depends on **verifying `gsi1`** (adas #6, infra #2): the index exists only in a comment; its KeySchema/ProjectionType are unknown. **Action: pin down and IaC `gsi1` first** with `ProjectionType=INCLUDE(mlflow_version, model_name, eval_ade, split_role)` **before** relying on it. Until verified, use an explicit base-table item instead of the GSI:
```
pk = SCENELIST#{run_id}#{dataset}#{version}#{shard}   sk = SCENE#{sampleID}
```
plus, for the scene→models direction, a sparse per-scene fanout row written only when `ready`:
```
pk = SCENEMODELS#{dataset}#{version}#{shard}#{sampleID}   sk = MODEL#{run_id}
attrs: mlflow_version N, model_name S, eval_ade N, split_role S, has_frames BOOL
```
P1 = base Query `pk=SCENEMODELS#…` → returns exactly the models with a playable overlay (no client filtering). If/when `gsi1` is verified with adequate projection, collapse `SCENEMODELS#` into a GSI-inverse of `PRED#`/`SCENELIST#` to avoid the extra write; do **not** build on the GSI speculatively.

**(5) Overlay-set status singleton (write-then-index gate; flyte #5):**
```
pk = OVLSET#{run_id}#{dataset}#{version}#{split}   sk = META
attrs: status S ("building"|"ready"|"deleted"), n_samples N, seeds L, manifest_key S,
       overlay_schema S, created_at S
```
The Go API checks `status=ready` before advertising. **TTL/lifecycle coherence (flyte #5):** do **NOT** TTL a `ready` overlay while its Flyte catalog entry still reports "cached" — that yields "advertised but bytes 404." Rule: retiring an overlay MUST (a) flip `OVLSET#` to `deleted`, (b) delete the S3 objects via lifecycle/explicit delete, and (c) purge the Flyte Datacatalog entry (or bump `overlay_schema` so the cache key changes). Never TTL just the pointer.

**(6) ODD geo-stats (§9):**
```
pk = GEO#{dataset}#{version}   sk = META
attrs: summary S (inline JSON: bbox, per-region counts, total),   # SMALL
       geojson_key S (S3 pointer to full point/heatmap set),      # avoids 400KB breach
       n_samples N, computed_at S
```
Full point sets / fine geohash bins would breach 400 KB (infra #11, cost #10) — store them in S3, keep only the summary inline.

### Hot-partition (adas #5, infra #12)
`PRED#{run_id}#…#{shard}` spreads writes across shards naturally (the shard is in the pk), so bulk `PutItem` is **not** concentrated on one partition — this is a further reason to prefer the per-shard-blob shape over the draft's `MODEL#{ver}`-keyed per-scene edges (all of which hashed to one partition, ~1000 WCU ceiling regardless of sort key; sort-key pagination fixes reads, not writes). `SCENEMODELS#…#{sampleID}` also spreads by scene. On-demand + adaptive capacity absorbs the rest; size the burst for GSI write amplification if `gsi1` is later used (infra #12).

---

## 7. Pre-rendered frames vs vector overlay — the decision, with math

**Recommendation: VECTOR-FIRST for all three sources and both views. Demote PR#74 baking to an optional offline export.** The user leaned pre-rendered; here is the honest math and the reason the lean does not survive it.

**Storage (cost-frontend #1, self-consistent numbers):**
- **Vectors:** `(64,2)` float32 + `v0` ≈ 520 B/sample/model, gzipped in one per-shard Dynamo item. 50 k samples × 10 models ≈ **~260 MB total**, plus one model-independent `proj.f32` shared across all models.
- **Baked (front-camera only):**
  - If playback = consecutive samples (matching the existing engine): **1 baked JPEG/sample/model** → 50 shards × 1000 × ~60 KB ≈ **~3 GB/model** → ~30 GB @ 10 models.
  - If each sample is a 64-step future clip: **64 frames/sample** → **~190 GB/model** → **~1.9 TB @ 10 models**, and it abandons the windowed `/blob` engine.
  The draft's "10 models × 1 k scenes × 20 MB = 200 GB" matches neither layout nor real shard counts.

**The decisive fact (cost-frontend #2, adas #3, flyte #7):** baking's *only* claimed advantage was camera projection under L2D `geometry_type='pseudo'`. But that projection is a function of **ego pose/geometry, not model weights** — it is model-independent. Precompute it **once per sample** (`proj.f32`), and the client draws *any* model's polyline in camera space. Baking therefore buys nothing that a per-sample projection blob + vectors don't, at ~3–4 orders more storage. And under pseudo-geometry the projection is a heuristic approximation **either way** — baking just freezes the same error into un-auditable pixels.

**Flexibility:** vectors toggle/compare 2–3 models on one canvas at ~0.5 KB each; baked frames make multi-model comparison physically impossible (can't composite two videos) and make a single toggle a multi-MB re-download (cost-frontend #5). The feature centers on *picking* (and comparing) models — vector-first is the correct default.

**Requirement (a) coverage (cost-frontend #3):** vectors are cheap enough to precompute for **all three sources including the leaked-train set**, so the train/eval/search UX is uniform. The draft's "bake eval+search only" silently dropped train-set camera overlays.

**When baking IS right (stated for honesty):** (i) a shareable/exportable clip for a doc or Discord where no interactivity is needed; (ii) a static thumbnail. In those cases emit **one MP4 per scene** (H.264 is ~10–50× smaller than a JPEG sequence and the console can `<video>`-play it), scoped and TTL'd — never as the interactive playback path, and never as loose per-frame JPEGs (which lose the windowed-prefetch smoothness and multiply S3 request cost; cost-frontend #4, infra #6).

This honors the user's intent (heavy Flyte precompute, "smooth playback") while spending bytes and flexibility correctly. If the user still wants baked-as-default, Open Question 1 flags it.

---

## 8. Frontend Design (Next.js)

### Scene view: model-picker + vector overlay playback
- **Model-picker** populated by P1 (`/scenes/{…}/models`) — only models with a `ready` overlay for the current scene, labeled `model_name` + `eval_ade` + the honest split tag ("train-leaked" / "near-duplicate hold-out").
- **Playback** reuses the existing per-frame JPEG byte-range path from `ShardIndex.members`. On model select, one GetItem loads `PRED#…#{shard}` (all overlays for the shard) and one GET loads `proj.f32`.

**Rendering (vector, both views):**
- **Two-layer canvas:** static frame layer (existing `<img>`/bitmap) + one transparent overlay `<canvas>`. Toggling/adding a model = `clearRect` + re-stroke the thin top layer only.
- **Client integrates raw control → XY** with the shared integrator (§10), applying the **same speed floor (0.5 m/s) and curvature clamp (±0.5 rad/m)** as GT so a wild prediction degrades gracefully and pred/GT are clamped identically (adas #9). GT (`ego_future`) and prediction are drawn by the exact same code path.
- **BEV view:** well-defined `meters_per_pixel` map (`metrics.py::offroad_rate` convention: forward `+x`→up/decreasing row, left `+y`→left/decreasing col). Metrically sound.
- **Camera view:** draw the integrated polyline through `proj.f32` (the model-independent projection). Under pseudo-geometry this is approximate but **auditable and correctable** (unlike baked pixels), and identical for every model.
- **Binary payload:** `fetch(url).then(r=>r.arrayBuffer())` → `Float32Array`, `subarray` views (zero-copy, no `JSON.parse`, no per-frame GC).
- **Frame-locked sync (cost-frontend #7):** there is **no `<video>` element** — playback is a manual JPEG-swap loop over tar byte-ranges, so `requestVideoFrameCallback` does **not** apply. The primary mechanism is: the overlay is indexed by the **same integer sample ordinal** that drives the image swap; the draw is triggered off the frame-advance step, not a wall-clock rAF, so image and overlay cannot drift under buffering stalls. `OffscreenCanvas`/worker only if profiling shows multi-cam main-thread jank.
- **Seed labeling (adas #4):** if `seeds=(0,)`, badge "single sample (seed 0)"; if a fan, draw median + envelope.

### Map view: driven GPS path
Per scene and per episode: fetch the **per-episode `episode_gps/*.f64`** (full driven route) and draw the `[lat,lon]` polyline on MapLibre/Leaflet vector tiles; animate an ego marker locked to the same frame clock. `gps_to_map.py` is used only for a static L2D-styled thumbnail (Overpass/`osmnx` is slow + needs internet — render offline in Flyte if used).

### ODD geo-stats page
Reads `GET /datasets/{name}/{version}/geo-stats` → inline summary (bbox, per-region counts) for KPIs + the S3 `geojson_key` for a heatmap/cluster map ("where was this data collected"). Follows the `dataviz` skill conventions.

---

## 9. GPS / Map / ODD

- **Packing (§4b):** per-sample `gps.npy` **float64** sidecar (current + 64-future window, aligned to `extract_egomotion`'s `sample_idx`) + per-episode `episode_gps/*.f64` full route. L2D loader slices raw `vehicle_states[:, 3:5]`; NVIDIA omits. **Full re-pack to `v2.1` (user decision 3)**; `ShardIndex.has_gps` + `IndexSample.gps_now[2]` (float64) via new Go npy-decode.
- **Geo-aggregation:** a Flyte `wf_geo_stats` task reads `IndexSample.gps_now` **but must not re-inflate every multi-MB gzip `IDX#` blob just to read one lat/lon** (cost-frontend #10). Instead, during overlay/index precompute write a compact per-shard `gps_now[]` array (or centroid/bbox) into a small item so aggregation is O(shards) small reads. Output: geohash-prefix bin counts, bbox, per-region counts (offline reverse-geocode via cached OSM) → `GEO#{dataset}#{version}/META` (inline summary + S3 `geojson_key`).

---

## 9-bis. Predicted trajectory ON the GPS map (user decision 2 — in scope)

The predicted ego-frame trajectory is overlaid on the geographic map in addition to camera/BEV. This is the highest-error-risk render path; it needs an explicit **error budget** and is gated behind a validation harness before being trusted.

**Placement math.** The integrated prediction is ego-frame `(x_forward, y_left)` metres (§10). To place it on the map we need, at the current sample: the absolute origin `(lat0, lon0)` (from `gps.npy`) and the **absolute heading** `ψ0` (map bearing, from-north). Then each ego-frame point rotates into ENU and offsets from the origin:
```
east  = x·sin(ψ0) + y·cos(ψ0)         # depends on the exact heading convention — MUST be pinned
north = x·cos(ψ0) − y·sin(ψ0)
lat = lat0 + north/R·(180/π);  lon = lon0 + east/(R·cos(lat0))·(180/π)
```

**Error budget (each term must be resolved, not assumed):**
1. **Yaw-sign / heading convention (dominant risk).** The BEV overlay uses ego-relative `+y=left` and may already be mirrored (§10). Placing on the map additionally needs the **absolute** compass bearing `ψ0`. L2D `heading` is compass (CW-from-north, degrees); `integrate_trajectory`'s internal `θ` is math CCW starting at 0. Mixing the two flips or rotates the whole predicted path on the map. **Action:** validate on a known straight + known-turn clip that the predicted path lies ON the driven GPS path when the model predicts near-GT; only then trust turns.
2. **Float precision.** `gps.npy` is float64 (§4b) precisely so the map origin doesn't jitter; do NOT downcast for this path.
3. **`v0` / one-frame gap (§10).** ~0.2 s scale offset — small but visible against real roads; note it.
4. **Pseudo-geometry.** L2D has no calibration; the *shape* is metric (unicycle integration is calibration-free), so map placement is actually **less** sensitive to pseudo-geometry than the camera-pixel projection is — the map path depends only on integration + origin + bearing, not on camera intrinsics. This is a point in favor of the map view.

**Reuse:** placement is a pure client transform over the SAME raw `(64,2)+v0` control blob + the per-sample `gps.npy` origin/heading — **no extra Flyte artifact**. GT (`ego_future`) places on the map by the identical transform, so pred-vs-GT-vs-driven-path can be compared on one map. Acceptance: the harness in §10 must pass for the map path specifically (a mirror is far more obvious on a real road than in BEV).

## 10. Determinism, Coordinate Contract, Versioning, Cache-Invalidation

### Coordinate & clamp contract (adas #1, #9, #10, #11) — resolve BEFORE trusting any overlay
- **Yaw-sign mirror is a real hazard.** `_derive_signals` builds `yaw_rate = diff(heading)/dt` from L2D's **compass heading (clockwise-from-north)**, while `integrate_trajectory` uses math-positive CCW (`y = v·sin θ`, `θ += curvature·v·dt`). A physical **right** turn (heading increasing) yields `curvature>0 → +θ → +y`, which the BEV convention (`+y→left`) renders as a **left** turn. This is invisible in ADE/FDE (GT and pred integrate identically) but **flips every rendered overlay left/right**. **Action:** verify the sign against a known turning clip and, if mirrored, apply the correction **in the shared client integrator** (so GT and pred flip together). Because we store raw control, this fix is a render-only change — no GPU recompute.
- **Clamp parity:** the client integrator MUST apply the same **speed floor 0.5 m/s** and **curvature clamp ±0.5 rad/m** that GT uses (`egomotion.py`), so a mispredicted curvature spike doesn't draw a loop and pred/GT are treated identically.
- **`v0` staleness (adas #10):** `v0 = ego_history[-1, 0]` is speed at `idx-1`; the future begins at `idx+1` (~0.2 s / one-frame gap). Harmless for ADE (GT shares it) but a small absolute scale offset when laid on the real scene — note it in the contract.
- **Channel truth (adas #11):** reference `egomotion.py` (`[speed, accel_x, yaw_rate, curvature]`), not the stale README ("yaw angle"). `v0 = channel 0` is unaffected.

### Versioning axes
- `overlay_schema` (render/kernel contract) — in S3 path + `cache_version`.
- `run_id` (durable model identity; never the moving registry version).
- shard `version` (dataset packing) — **one value across `PRED#`/`IDX#`/search** (§consistency).
- Seeds (in the blob + manifest + cache key).

### Cache-invalidation
- `cache_version` gets a **build-time git SHA / module hash** (flyte #4) — static strings don't auto-hash source.
- A render/integrator fix bumps nothing on the GPU side (client-only) — a direct benefit of storing raw control.
- A kernel change bumps `overlay_schema` → new S3 prefix + new cache key + lifecycle reclaims the old prefix.
- **Determinism flags** (`torch.use_deterministic_algorithms(True)`, cuDNN deterministic) for byte-reproducible re-runs, or explicitly document "numerically close" (adas #13).

### Storage-cost summary
- Vectors (all sources, both views): **~260 MB Dynamo** total + shared `proj.f32` — trivial.
- Optional MP4 export: bounded by scope + S3 lifecycle TTL; never on the playback path.

---

## 11. Phasing, Risks

**Phase 0 — console-only linkage (no Flyte):**
- Fix `MLflowModelVersion` to keep `source`; add `model-versions/search` proxy; surface `run_id` + read `val_fraction` from `config.yaml` (no re-run). Write `VER#`/`MODEL#` seeds.
- Frontend model-picker skeleton + BEV two-layer canvas against a mocked control blob + shared integrator (with yaw-sign verification harness).

**Phase 1 — GPS (Flyte data-gen → console):**
- L2D loader `gps_latlon` (new plumbing); decode-free `gps.npy` backfill into v2.0; per-episode path artifact; Go npy-decode for `has_gps`/`gps_now`; Map view; `GEO#` geo-stats page.

**Phase 2 — vector overlays (Flyte GPU + Dynamo + API):**
- `load_policy`/`predict_control` helper (no model edit needed); coarse per-shard task; `proj.f32` projection artifact; `PRED#`/`OVLSET#`/`SCENEMODELS#` items; new API endpoints; BEV + camera multi-model overlay/toggle/compare.

**Phase 3 — optional MP4 export (PR#74):**
- Move `Tools/trajectory_visualization/runner.py` under `/Tools`; wrap to emit **one MP4 per scene**; expose as a "download clip," scoped + TTL'd. **Not required for the feature.**

**Risks:**
- **Yaw-sign mirror** (§10) — must be verified before any overlay is trusted; blocks Phase 2 acceptance.
- **`gsi1` unverified** — do not build P1 on it; IaC + verify projection first, else use `SCENEMODELS#`.
- **Version-coordinate drift** (`v2.0` vs `v2.1`) — enforce one version; assert `sample_id` byte-stability across any repack.
- **Single warm GPU** — a full-leaked-train × all-versions vector backfill is near-serial; scope which runs/sources before triggering (Open Question 3). Give a wall-clock estimate before backfilling.
- **PR#74 output format** — the tool emits an MP4 + manifest and is absent from this checkout; its single-sample vs batched API and projection source are **unverified**. Phase 3 assumes MP4 output (aligns with its actual format), not a JPEG-sequence rewrite.
- **Default `val_fraction=0`** — many models have no eval set; UI must say "train-leaked (no held-out eval)."
- **TTL vs catalog** coherence (§6 rule) — deletion must purge S3 + catalog + flip status together.

---

## 12. Design decisions & rejected alternatives

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Overlay representation | **Raw `(64,2) accel/curvature + v0`** | Integrated XY blob (`traj.f32`/inline `t`) | XY bakes the integrator + yaw-sign + clamp into the artifact → any fix forces full GPU recompute; raw keeps fixes client-side and reuses the shipped BEV integrator (adas #2). |
| Playback storage shape | **One gzip blob per (model, shard)** `PRED#…/META` | Per-sample inline-`t` edges keyed `MODEL#{ver}`/`SCENE#…` | Per-sample edges turn episode playback into ~1000 GetItems / paginated ~1 MB reads and concentrate writes on one partition; the per-shard blob = one GetItem, mirrors `ShardIndex`, spreads writes by shard (brief Option A; infra #1, #5, adas #5). |
| Camera overlay | **Model-independent `proj.f32` + client vector draw** | Baked per-model front-camera frames | Projection is a function of ego geometry, not weights; baking buys nothing but ~3–4 orders more storage and kills multi-model compare (cost-frontend #2, adas #3). |
| Baked frames | **Optional MP4 export only** | Baked as default deliverable | Storage math + flexibility both favor vectors; baking can't composite/toggle models; if exported, MP4 ≫ JPEG-sequence (cost-frontend #1,4,5). |
| Model identity key | **`run_id`** (+`mlflow_version` attr, `VER#` alias) | `MODEL#{mlflow_version}` | Registry version is a moving pointer; `run_id` is durable/content-addressable (infra #4). |
| Run metadata | **Single `MODEL#{run_id}/META` profile** | `eval_ade`/`model_name` on every edge | Run-level, not scene-level; per-edge copies are pure duplication + rewrite-all on change (infra #5). |
| P1 "models for scene" | **`SCENEMODELS#` base item now; GSI later only if verified** | Build on `gsi1` immediately | `gsi1` exists only in a comment; KeySchema/Projection unknown; a `KEYS_ONLY` GSI silently un-indexes inline reads (adas #6, infra #2). |
| GPS precision | **float64** (or int32 ENU offset) | float32 | float32 → ~1–2 m jitter visible on the driven-path map (adas #8, cost #9). |
| GPS for map | **Per-episode full-path artifact** | Per-sample 6.4 s future window | A future window can't draw a driven route (flyte #8). |
| GPS backfill | **Full re-pack to v2.1 (USER DECISION 3)** | Decode-free in-place into v2.0 | User chose a clean versioned re-pack over the cheaper in-place append; cost = re-decode 7 cams/sample + move all of `PRED#`/`IDX#`/search to v2.1 + assert sample_id byte-stability (§4b, §consistency). The in-place option remains the cheaper fallback if the re-pack proves too slow. |
| Flyte fan-out | **Coarse per-shard task, load ckpt once** | `map_task` over tiny units ("actor pool") | Flyte subtasks aren't a warm Ray actor pool; fine fan-out re-downloads 509 MiB per subtask, and there's one warm GPU anyway (flyte #2, #3). |
| Determinism | **Reuse existing `generator` threading** | New model-side seed plumbing | Already threaded `AutoE2E→Reactive→FlowMatching` (verified; flyte #1). |
| GEO# storage | **Inline summary + S3 geojson pointer** | Inline full point set/geohash bins | Full data breaches 400 KB; coarse binning loses map resolution (infra #11, cost #10). |
| Inline float compression | **Gzip the whole payload JSON (like ShardIndex)** | Gzip the `(64,2)` floats alone | High-entropy IEEE floats don't compress alone; gzip the aggregate as `ShardIndex` does (cost-frontend #8). |

---

## 13. Open questions for the user

RESOLVED 2026-07-13 (see "Decisions locked" at top):
- ~~Q1 Baked-as-default?~~ → **Vector-first accepted** as the playback path; baking = optional MP4 export.
- ~~Q4 GPS backfill mechanism?~~ → **Full re-pack to v2.1** (not in-place into v2.0).
- ~~Q5 Predicted-trajectory-on-map?~~ → **YES, in scope** (§9-bis error budget added).

STILL OPEN:
1. **Seed policy:** single seed (badge "one sample") vs a 4-seed fan with median+envelope for FlowMatching. Fan ≈ 4× GPU + blob size; still cheap on storage.
2. **Backfill scope on one GPU:** which model versions and which sources get precomputed? (All registered versions × leaked-train is near-serial on the single warm L40S — likely long. Recommend: latest N versions, all three sources for vectors.)
3. **`gsi1` ownership:** may we IaC the table/GSI (currently created out-of-band) so P1 can use a verified inverse index instead of the `SCENEMODELS#` base item?
4. **`POST …:compute` in the UI:** self-serve GPU-spend trigger vs ops-only? (Self-serve on a single warm GPU risks queueing/cost.)

---

Key source anchors: forward + generator threading (`Model/model_components/auto_e2e.py:94-97,191`, `reactive_e2e.py:104-106,166-170`, `trajectory_planning/flow_matching_planner.py:318-320,348-350`, `bezier_planner.py:107`); integrate + BEV convention (`Model/evaluation/metrics.py:20-49,255`); egomotion channels/clamps/GPS cols (`Model/data_parsing/l2d/egomotion.py:5-6,39-79`); packer (`Model/data_processing/reasoning_label_generation/parallel_pack.py`, `Platform/pipelines/workflows.py::data_processing`/`_run_evaluation`); console store (`Tools/DataModelConsole/api/internal/store/keys.go`, `model/types.go` `ShardIndex`/`IndexSample`, `service/s3.go::StreamTarMemberRange`, `service/reasoning_stats.go::ResolveSampleShards`/`resolveVersion`); map (`Model/data_parsing/map_rendering/gps_to_map.py`). Brief: Option A per-shard gzip `PRED#` blob (recommended shape), inference/planner facts, MLflow/checkpoint linkage, DynamoDB 400 KB + `gsi1`-unused, playback byte-range mechanism.