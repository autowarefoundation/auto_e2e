# Design Doc — AutoE2E DataModelConsole: Per-Model Trajectory Overlays + GPS Map / ODD Geo-Stats

Status: Implemented on `feat/trajectory-overlay-console`; production rollout pending
Owning code: `Tools/DataModelConsole` (Go API + Next.js) and `Platform/pipelines/workflows.py` (Flyte)
Author: design pass grounded in `docs/.traj_brief.json` + source read of `Model/model_components/*`, `Model/evaluation/metrics.py`, `Model/data_parsing/l2d/egomotion.py`, packer, and the console store; revised against four adversarial reviews (adas, infra-dynamo, flyte-feasibility, cost-storage-frontend) and a second-round external review (verdicts P0.1–P0.5, P1.6–P1.15).

Implementation snapshot (2026-07-16):
- v2.1 geospatial packing/publication, canonical AOVL overlays, Console API, and
  the BEV/camera/map UI are implemented.
- `wf_create_publish_and_precompute_overlays` owns the complete data path:
  sharded dataset creation -> immutable publication -> GPU overlay precompute.
- `Platform/buildspec-launch-overlay.yml` launches that workflow from the
  VPC-local CodeBuild project `auto-e2e-platform-overlay-launch`.
- PR #74's offline report boundary is implemented as
  `wf_export_trajectory_report`: it consumes the canonical AOVL, its v2.1
  shard, and both immutable publication manifests; it emits per-scene MP4,
  thumbnail, metrics, and verified model/dataset provenance.
- The publication workflow exposes the immutable manifest key and digest.
  Reasoning stats, direct sample lookups, and scene-search rows are populated
  by an explicit digest-pinned `batch/v1 Job` after that workflow succeeds;
  normal Console deployment never starts this scan.
- Production KITScenes v2.2 receipts contain 404 non-empty scene shards across
  three calibrated pinhole rigs; 129 empty partitions carry a pseudo rig but
  publish no shard. Publication schema v2 stores each unique projection at
  `rig/{sha256}.json` and binds every `shard_entry` to its exact rig.
- Production activation still requires the rollout sequence in section 11.
  Exact routes remain disabled until authenticated, non-spoofable role
  propagation is deployed.

---

## Decisions locked by the user (2026-07-13)

These OVERRIDE any contradicting recommendation later in this doc; the body below is annotated where a prior recommendation was reversed.

> **Two locked decisions are amended by the round-2 external review (flagged inline and to the user):**
> - **P1.6 SUPERSEDES the round-2 gsi1 decision.** With canonical per-shard overlays (P0.2), the model↔scene relation collapses to model↔SHARD, so the scene×model `gsi1` inverse index and `SCENELIST#` fanout are no longer needed for P1. Whether to still provision `gsi1` for *future* inverse lookups is **deferred to the user**.
> - **P0.1 revises the "zero model change for determinism" claim.** A small model-side change *is* required: an `initial_noise=` kwarg on `FlowMatchingPlanner.forward` for per-sample, batch-invariant noise. Stored overlays were always deterministic-as-stored; the fix targets recompute batch-invariance + cross-run reproducibility.

1. **Playback representation = VECTOR-FIRST (accepted as recommended).** Store raw `(64,2)` control + `v0`, draw client-side; baking demoted to optional MP4 export. (§7, §0.3–4)
2. **Predicted trajectory ON the geographic (GPS) map = IN SCOPE** — REVERSES the doc's default (was "out of scope / driven-path only"). Requires an explicit error budget: yaw-sign convention (§10), lat/lon float precision, ego-heading→map-bearing conversion, and pseudo-geometry. See §9-bis (predicted-on-map). (Open Question 5 resolved: YES.)
3. **GPS packing = FULL RE-PACK to a new dataset version `v2.1`** — REVERSES the doc's preferred "decode-free in-place backfill into v2.0". Consequence (MANDATORY): overlays, the playback index (`IDX#`), and label-search resolution (`ResolveSampleShards`) MUST ALL move to `v2.1`. Sample identity is now the content-addressed `sample_uid` (P1.7); `legacy_sample_id = s{si:08d}` is retained via a migration manifest so keys don't silently shift. (§4b, §consistency, Open Question 4 resolved: v2.1.)
4. **Implementation go-ahead was received after this design pass.** The
   implementation status and remaining production gates are recorded in
   sections 2 and 11.

---

## 0. TL;DR of the big decisions (read this first)

1. **Store the RAW control prediction, not integrated XY.** Overlays persist `(64,2) [accel_x, curvature] + v0` — exactly the `ego_future` representation the index already carries — and the client integrates it with the *same* reference integrator used for the GT plan (PY↔TS golden-tested). This keeps any future fix to `integrate_trajectory` / the yaw-sign a **pure render change with zero GPU recompute**, and shrinks the payload. The client **defaults to raw prediction** (no GT-style clamping); a "display-limited" mode is a separate, labeled toggle (P1.9). (adas #2, #9)
2. **Overlays are CANONICAL per `(model_artifact_id, dataset_version, shard)`.** ONE inference over **ALL** samples in a shard (train ∪ eval = all samples; search ⊂ all). **train / eval / search are display-time FILTERS, not separate compute or storage.** `split`/`source` appears in **no** physical artifact — not in the S3 key, not in the Dynamo key, not in the manifest. This removes redundant inference, the overwrite bug (train→eval→search subsets used to clobber each other), per-source manifests, and makes a scene belonging to multiple splits a non-issue. (P0.2 — the single most important change)
3. **S3 is the sole overlay body; DynamoDB is a pointer only.** The overlay body is one **binary** `overlay.bin.gz` per (model, shard) in S3 (§6 layout). DynamoDB stores only `s3_key / sha256 / byte_size / sample_count / overlay_schema / status / created_at`. There is **no gzip-JSON overlay item in Dynamo** — that both risked the hard 400 KB item cap (gzip ratio is content/seed-count dependent, esp. seed fans) and contradicted the frontend plan (`arrayBuffer → Float32Array`, no `JSON.parse`). One S3 GET per (model, shard) → held in memory → smooth scrubbing. (P0.3)
4. **Camera projection is a v2.1-repack artifact and a per-RIG CONSTANT — not per-sample, not per-model.** For a fixed camera rig, ego-frame-XY→camera-pixels depends only on the rig's fixed intrinsics/extrinsics, not on ego pose or model weights. PR#74's `project_BEV_to_CameraView` confirms this with a fixed `P = K[R|t]` input. It is generated **inside the v2.1 repack** (which already holds pose/calibration), NOT in the per-model overlay task. Store each unique rig once by content digest, bind each shard to its rig, and drop any per-sample `proj.f32`. (P1.12, cost-frontend #2)
5. **Overlay identity is `model_artifact_id = sha256(best.pt bytes)`** — content-addressable, dedupes identical checkpoints, detects content change. The registry coordinate `MODELVER#{registered_model_name}#{model_version} → {run_id, artifact_uri, checkpoint_sha256}` resolves a picked registry version to the artifact. `run_id` is kept only as a **provenance attribute** (it is a lineage id, NOT content-addressable). MLflow version *numbers* are immutable; it is *aliases* (e.g. "latest"/"champion") that move. (P1.8)
6. **A small model-side change IS required for determinism (corrects the earlier claim).** Per-batch `generator=` does **not** give per-sample-stable noise: `torch.randn(B,dim,generator=gen)` draws in batch order, so batch size / sample order / OOM-triggered batch-size halving / retry re-splits all shift a given sample's noise. FIX: add `initial_noise=` to `FlowMatchingPlanner.forward` (falls back to `torch.randn` when unset) and feed per-sample noise `z0 = noise_from(hash64(model_artifact_id, dataset_manifest_digest, sample_uid, base_seed))`. **Nuance:** a stored overlay is deterministic *as stored*; this fix buys batch-invariance on **recompute** and cross-run reproducibility, not correctness of an already-written blob. (P0.1)

---

## 1. Goal & Scope

Give the console a **Scene view** in which a user picks a **Model** and sees **that model's predicted trajectory** overlaid on the scene's camera + BEV frames, with **very smooth playback**. This must work for **three scene sources**, all of which resolve to the same `(dataset, version, shard, sample_uid)` tuple space (grounded: all three are WebDataset shard samples, differing only in which sample keys are selected). **Because overlays are canonical per (model, dataset, shard), all three sources read the SAME overlay body — they differ only in which samples the UI shows:**

- **(a) Leaked training set** — the shards the model trained on. Because `training.val_fraction` defaults to `0.0`, by default *every* sample is training-leaked. When a hold-out is configured, train-vs-eval uses `blake2b(split_group_uid, digest_size=8) mod 10`, where `split_group_uid` is an episode or clip — not a separate shard set — so it remains a pure **display-time filter** over the canonical overlay.
- **(b) Eval set** — the held-out `val` bucket (only non-empty when the run used `val_fraction > 0`). The split is episode/clip-level, so temporally adjacent frames never straddle train and eval. The UI labels it "episode/clip hold-out"; it is a stronger generalization check than the superseded per-frame split, while still remaining an in-dataset evaluation rather than a claim of cross-dataset ODD generalization. Again this is a display-time filter, not a separate overlay.
- **(c) Reasoning (Action-Relevance) label search** — scenes from the existing `LBL#` index (`GET /api/v1/scenes/search`), then overlaid with the picked model's trajectory — a subset of the canonical shard overlay.

Also in scope (new requirement):
- **Pack GPS** `lat/lon` (L2D raw cols 3–4: `hp_loc_latitude`, `hp_loc_longitude`) **plus raw vehicle `heading`** (L2D `heading` column) into an explicit per-sample `pose_current` (§4b). The packer today keeps only the derived 4-ch ego `[speed, accel_x, yaw_rate, curvature]` and never surfaces lat/lon or absolute heading out of the loader.
- **Map view**: draw the driven path on a real map.
- **ODD geo-statistics**: "where was this data collected."

**In scope (user decision 2):** overlaying the *predicted* trajectory on the real geographic (GPS) map, in addition to the camera/BEV views. This requires the error budget in §9-bis (yaw-sign, float precision, heading→bearing, pseudo-geometry).

**Out of scope:** on-demand (runtime) inference in the Go API — infeasible (checkpoint is ~509 MiB `best.pt`, loadable only via Python `AutoE2E(**_model_kwargs(config))`; Go cannot load `.pt`). All inference is **heavy Flyte-side precompute**, which the user explicitly accepts. The Console is **read-only** and never triggers GPU compute (ops-only).

---

## 2. Implementation Status and Pre-Implementation Baseline

### Implemented contract
- Packed samples use stable `sample_uid` and `split_group_uid`; v2.1 shards
  carry portable pose/GPS members, a shard-bound content-addressed rig
  projection, embedded reasoning labels, and dataset-level privacy-filtered
  geo products.
- Dataset publication copies every body with an S3 conditional write, writes a
  hidden publication lock, and writes `shards/manifest.json` last as the public
  readiness gate. `wf_publish_and_precompute_overlays` passes that manifest's
  SHA-256 directly to overlay precompute.
- Overlay request identity includes checkpoint bytes, dataset manifest,
  preprocessing contract, inference source, task image digest, sampler,
  seeds, and schema versions. The checkpoint-derived inference-step count is
  added to the cache identity.
- Overlay bodies and manifests use `If-None-Match: *`. DynamoDB model,
  pointer, and initial `OVLSET` records use create-only conditions. The final
  `building -> ready` transition is conditional on the request and dataset
  identity. A compatible retry reuses existing objects; a conflicting retry
  fails instead of replacing them.
- The preparation token carries the immutable request coordinate through every
  Flyte child task. An already-`ready` set stays ready during retries and is
  never reset to `building`.
- Console readers require pointer and ready-gate agreement on dataset manifest
  and cache identity before exposing an overlay. They then constrain the S3 key
  to the canonical schema/model/dataset/version/shard prefix and verify body
  size, SHA-256, and gzip framing.
- Registration and launch resolve every task image to an ECR digest. Tasks
  recompute the preprocessing/inference digests from their running source and
  reject launcher values or an image digest that do not describe that runtime.

The facts and gaps below describe the baseline used to derive the design. They
are retained for rationale; the gap list is no longer the rollout checklist.

### Verified facts (source-read)
- **Forward contract:** `AutoE2E(...)(camera_tiles, map_input, visual_history, egomotion_history, projection=…, geometry_type=…, mode="infer", **kwargs)` returns a bare `[B,128]` = `(64,2)` = `[accel_x (m/s²), curvature (1/m)]` control at 10 Hz over 6.4 s — **NOT XY**. `mode="infer"` returns the bare tensor (no aux dict). Reasoning branch does not change the output shape.
- **`generator` threads end-to-end, but is NOT per-sample stable (P0.1):** `auto_e2e.py:94–97,191` forwards `**kwargs`; `reactive_e2e.py:104–106,166–170` forwards `**kwargs` into `self.TrajectoryPlanner(...)`; `flow_matching_planner.py:318–320,348–350` consumes `generator=` at `torch.randn(B, self.trajectory_dim, …, generator=generator)`. Because the draw is `randn(B, …)` in batch order, the noise a given sample receives depends on batch size / position / retries. A **new `initial_noise=` kwarg** (fed per-sample noise) is required for batch-invariant recompute (§10). `BezierPlanner.forward` accepts/ignores `**kwargs` and `initial_noise`.
- **Integration:** `metrics.py::integrate_trajectory(accel, curvature, v0, theta0=0.0, dt=0.1)` → `(T,2)` `[x_forward, y_left]` metres, ego frame. Loop: `theta += curvature[t]·v·dt; x += v·cos θ·dt; y += v·sin θ·dt`. `v0 = ego_history.reshape(64,4)[-1,0]` (speed channel, last history step). **No clamp lives here** — see P1.9 below.
- **Egomotion channels (verified `egomotion.py`):** derived signals are `[speed, accel_x, yaw_rate, curvature]` (channel 3 = curvature, **not** yaw angle — the README wording is stale; cite code). `_derive_signals` builds `speed = raw/3.6` (km/h→m/s), `heading = unwrap(radians(...))`, `yaw_rate = diff(heading)/dt`, `curvature = yaw_rate/max(speed, 0.5)`, then `clip(curvature, ±0.5)`. **The 0.5 m/s floor and ±0.5 rad/m clamp are properties of GT curvature DERIVATION, not of the integrator** (P1.9). The model prediction is not derived this way.
- **FlowMatchingPlanner is stochastic** (`torch.randn(..., generator=generator)`); Bezier is deterministic and ignores the generator / `initial_noise`.
- **MLflow linkage:** one registered model `auto-e2e-driving-policy`; a "Model ID" = a registry **version** (an immutable version *number*; aliases like "latest" move). Durable lineage pointer is `run_id`; **content identity is `sha256(best.pt)`** (P1.8). Latest observed: **v30 → run_id `b457606594204ac88e3e1a0fe09075f5`**. Checkpoint: `s3://auto-e2e-platform-artifacts-381491877296/mlflow/8/{run_id}/artifacts/model/best.pt` (~509 MiB) + sibling `config.yaml` (1951 B). `.pt = {model_state_dict, config, epoch}`; rebuild via `AutoE2E(**_model_kwargs(config))`. Run params link model→dataset (`data/dataset`), Flyte exec ids, eval metrics (`eval/ade`, `eval/fde`, `eval/gate_pass`). `training.val_fraction` is **in `config.yaml` and `metadata.json` but NOT a logged MLflow param** (flyte #11).
- **DynamoDB single-table `auto-e2e-console`** (`store/keys.go`): `pk` HASH + `sk` RANGE (String), plus a `gsi1` that **appears only in a doc comment** — no Go code sets `gsi1pk/gsi1sk`, no `IndexName` query, and no `CreateTable`/Terraform in-repo. **With P1.6 the design no longer needs `gsi1` for P1.** Existing items:
  - `IDX#{dataset}#{version}#{shard}` / `META` → gzip `ShardIndex` (playback source: `fps`, per-sample `members[suffix]→{offset,size}`, `ego_now[4]`, `ego_history`, `ego_future[128]`, `has_reasoning`).
  - `STATS#{dataset}#{version}#{promptVersion}` / `META` → reasoning stats blob.
  - `LBL#{dataset}#{promptVersion}#{field}#{value}` / `SCENE#{sampleID}` → scene-by-label index (**keyed by legacy `s{si:08d}` today; P1.7 migration blast-radius**).
- **Playback mechanism:** frontend fetches each JPEG via S3 byte-range GET (`StreamTarMemberRange`, `MaxRangeBytes = 32 MiB`) using `members[suffix].{offset,size}` from `ShardIndex`. **A "scene" sample is one 10 Hz frame; an episode plays consecutive samples** (per-shard `frame_idx == global sample idx`). `ego_future` is the per-frame GT the BEV already integrates+draws — the model overlay is its direct analogue.
- **Go API** is a READ-ONLY MLflow/Flyte proxy + S3/Dynamo reader. `MLflowModelVersion` normalization **drops the `source` artifact URI** and exposes no `model-versions/search`, so version→`run_id`→checkpoint→`sha256` mapping is impossible today. Artifacts bucket configured but `mlflow/` prefix unused.
- **`ResolveSampleShards`** resolves label-search sampleIDs against the console's **published** dataset version (`resolveVersion`) — a version-coordinate landmine (§ below).
- **Packer** (`parallel_pack.pack_sample`): writes `ego.npy = concat(ego_history[256], trajectory_target[128])` float32, `cam_i.jpg`, optional `map.jpg`, WM windows, `meta.json`, `calib.json`. `sample_key = f"s{si:08d}"`, 1000 samples/shard, `train-{idx:06d}.tar`. GPS and absolute heading never enter the sample dict.
- **`gps_to_map.py`** renders GPS waypoints on an ego-centric OSM/`osmnx` BEV tile (L2D palette), with equirectangular lat/lon→m helper.

### Original gaps (resolved by this implementation)
1. No standalone single-sample inference / overlay entry point (only `_run_evaluation`, batched, discards integrated XY).
2. No Flyte-free checkpoint-load helper (and no `sha256(best.pt)` computed at registration).
3. **Per-sample noise plumbing missing** — `initial_noise=` kwarg on `FlowMatchingPlanner.forward` needed for batch-invariant recompute (P0.1); the fixed-seed convention (base_seed, hash inputs) is also undefined.
4. **Camera-projection responsibility misplaced (resolved in the implementation)** — PR#74 confirmed that calibrated projection uses fixed rig inputs, so v2.1 publishes one rig-level projection and no per-sample projection payload (P1.12).
5. No model dimension in the key space; `gsi1` unused (and now unnecessary for P1).
6. `val_fraction` not a logged param — but **readable now from `config.yaml`** (no re-run needed). (flyte #11)
7. GPS + absolute heading dropped by loader+packer.
8. No geo-stats keying (now produced during the v2.1 repack, P1.14).
9. **Yaw-sign / display-mode render contract undefined** (adas #1, #9; P1.9) — see §10.
10. **`sample_uid` (content-addressed sample identity) does not exist** — enumeration is the fragile global `s{si:08d}` (P1.7).

---

## 3. Architecture Overview

```
                     ┌──────────────────────── Flyte (GPU, us-west-2, ONE warm L40S) ─────────────────────┐
 MLflow registry     │  wf_create_publish_and_precompute_overlays(model_version, ..., version=v2.1)       │
                     │    wf_create_dataset_sharded -> wf_publish_dataset_snapshot                         │
                     │      manifest SHA-256 wired directly -> wf_precompute_overlays                      │
 auto-e2e-driving-   │    resolve MODELVER → {run_id, artifact_uri}; GET best.pt ONCE; sha256 → model_artifact_id │
   policy ──────────►│    coarse @task per shard  (amortize 509 MiB load; NOT map over tiny units)         │
 config.yaml ───────►│      load_policy() ─►                                                                │
   (val_fraction)    │        predict batched over ALL samples in shard (train∪eval=all; search⊂all)        │
 v2.1 shards ───────►│          initial_noise from hash64(model_artifact_id,ds_manifest,sample_uid,seed)    │
   (datasets bucket) │          ─► [128] control  (NO integration; store RAW)                               │
                     │      conditional write ONE overlay.bin.gz per (model, shard) ──► S3                  │
                     │      create-only pointer + projected model attrs ──► Dynamo (SHARD×MODEL item)        │
                     │      write manifest FIRST → conditional OVLSET building→ready LAST                    │
                     │  data_processing v2.1 REPACK (MOD): pack pose_current; episode GPS path;              │
                     │      rig projection params (per-rig CONSTANT, confirmed by PR#74); geo/* stats        │
                     └──────┬───────────────────────────────────────────────────┬──────────────────────────┘
                            │ S3 PUT (idempotent deterministic keys)              │ Put/BatchWrite (chunk 25)
                            ▼                                                     ▼
   S3 auto-e2e-platform-artifacts-381491877296            DynamoDB auto-e2e-console (single table)
     overlays/schema=v1/model={artifact_id}/dataset={l2d}/    SHARD#{ds}#v2.1#{shard} / MODEL#{artifact_id}   (POINTER + attrs)
       version=v2.1/shard={train-000000.tar}/overlay.bin.gz       MODEL#{artifact_id} / META        (run profile)
     overlays_manifest/…/manifest.json                        MODELVER#{regname}#{ver} / META    (→ run_id, uri, sha256)
   S3 datasets bucket (v2.1 repack):                          OVLSET#{artifact_id}#{ds}#v2.1 / META (status)
     shards …/pose.npy  (per-sample pose_current)             GEO#{ds}#v2.1 / META               (summary + S3 ptr)
     geo/episode_paths/*, geo/sample_pose.parquet,
     geo/summary.json, geo/heatmap.fgb, rig/{rig_sha256}.json
                            │                                 (NO overlay body in Dynamo; NO split in any key)
                            └────────────────► Go API (READ-ONLY++) ◄───────────────────────────┘
                                  new endpoints: models-for-shard, overlay-pointer (per model+shard),
                                  rig-projection, geo-stats, gps-path  (Dynamo + S3; NO inference)
                                                        │
                                                        ▼
                               Next.js: model-picker → two-layer canvas vector overlay (BEV + camera);
                               display-mode toggle (raw default / display-limited); Map view; ODD geo-stats
```

Principle (`bp-relational-modeling`, materialized-index pattern): **playback never runs the model and never scans S3 per frame** — one Dynamo GetItem yields the pointer, one S3 GET loads a shard's `overlay.bin.gz` into memory; vectors integrate/draw client-side. train/eval/search are display filters over that one body.

---

## 4. Flyte Pipeline Design

### 4a. `wf_precompute_overlays`

**Location:** `Platform/pipelines/workflows.py`, sibling to `train_il` / `data_processing` / `_run_evaluation`.

**Reusable, Flyte-free helper (fills gaps #2, #3).**
```python
# Platform/pipelines/inference.py  (importable, no Flyte deps)
def load_policy(ckpt_path: str, device: str) -> tuple[nn.Module, dict, str]:
    raw = open(ckpt_path, "rb").read()
    model_artifact_id = hashlib.sha256(raw).hexdigest()      # CONTENT identity (P1.8)
    ckpt = torch.load(io.BytesIO(raw), map_location=device)
    cfg = ckpt["config"]
    model = AutoE2E(**_model_kwargs(cfg)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, model_artifact_id

def noise_from(model_artifact_id: str, ds_manifest_digest: str,
               sample_uid: str, base_seed: int, shape, device) -> torch.Tensor:
    """Per-sample, batch-INVARIANT initial noise (P0.1)."""
    h = hash64(model_artifact_id, ds_manifest_digest, sample_uid, base_seed)  # stable 64-bit
    g = torch.Generator(device=device).manual_seed(h)
    return torch.randn(shape, generator=g, device=device)

def predict_control(model, batch, cfg, *, sample_uids, model_artifact_id,
                    ds_manifest_digest, base_seed, sampler="euler", num_steps=10) -> np.ndarray:
    """Return RAW (B,64,2) [accel_x, curvature] control — NOT integrated XY."""
    dev = batch["egomotion_history"].device
    z0 = torch.stack([noise_from(model_artifact_id, ds_manifest_digest, uid, base_seed,
                                 (model.trajectory_dim,), dev) for uid in sample_uids])  # (B,dim)
    with torch.no_grad():
        pred = model(batch["visual_tiles"], batch["map_input"],
                     batch["visual_history"], batch["egomotion_history"],
                     projection=batch.get("projection"),
                     geometry_type=cfg["model"].get("geometry_type", "pseudo"),
                     mode="infer", initial_noise=z0)          # NEW kwarg; Bezier ignores it
    return pred.reshape(pred.shape[0], 64, 2).cpu().numpy()
```
> **Model-side change (P0.1):** add `initial_noise: Optional[Tensor]=None` to `FlowMatchingPlanner.forward`; when set, use it as `z0` instead of `torch.randn(B,dim,generator=…)`; when unset, fall back to today's `torch.randn`. This is the **only** model edit, and it makes recompute batch-invariant (a given `sample_uid` gets the same noise regardless of batch size / order / retry). It does NOT change an already-stored overlay — those are deterministic-as-stored.

> **Do NOT integrate here.** We persist raw control + `v0` and integrate on the client with the shared, fixable integrator (§10). This is the single most important representation change vs the draft (adas #2). `batch["projection"]`/`geometry_type` keys MUST be confirmed against `make_pre_extracted_loader` output before merge (flyte #12a); if the loader omits them, thread from `calib.json` / `cfg` — but note (P1.12) camera projection is a v2.1-repack rig artifact, not a per-overlay input.

**Canonical, split-free inference (P0.2).** For each shard, run the model over **ALL 1000 samples once**. There is no `split`/`source` in the task inputs, S3 keys, Dynamo keys, or manifest. train (all when `val_fraction=0`), eval (val bucket), and search (label subset) are computed **downstream at display time** by the Go API / frontend, never by re-running inference on a subset. This removes the overwrite bug (subsets used to clobber each other under the old `PRED#…` key that lacked `split`), removes per-source manifests, and makes multi-split membership a non-issue.

**Coarse subtasking to amortize the 509 MiB load (flyte #2, #3).** Flyte `map_task` subtasks are independent executions on fresh pods — **not** a warm Ray actor pool; a fine-grained map would re-download `best.pt` and re-instantiate `AutoE2E` per subtask. Since the platform runs **exactly one warm g6e.4xlarge (1× L40S, do-not-disrupt)** and scale-up is Karpenter+Kueue against a single ODCR, parallelism is near-serial anyway. Decision: **one coarse `@task` per shard, loading the checkpoint ONCE and streaming all that shard's samples through batched forward.** No `map_task` fan-out over tiny units; the Ray-actor citation from the draft is removed as non-transferable.

```python
@dataclass
class OverlayShardJob:
    model_artifact_id: str      # sha256(best.pt) — CONTENT identity (P1.8)
    registered_model_name: str  # "auto-e2e-driving-policy"
    model_version: int          # registry version number (immutable)
    run_id: str                 # provenance attr only (NOT identity)
    artifact_uri: str           # s3://…/best.pt
    dataset: str                # "l2d" | "nvidia_av"
    version: str                # "v2.1" — MUST equal IDX#/search version (§ consistency)
    dataset_manifest_digest: str # v2.1 manifest sha (immutable) — noise + cache input (P1.11)
    shard: str                  # "train-000000.tar"
    base_seed: int = 0          # per-sample noise seed (recorded); NOT a batch generator seed
    sampler: str = "euler"
    num_steps: int = 10
    overlay_schema: str = "v1"  # overlay BINARY schema (P1.11)

@task(requests=Resources(gpu="1", mem="16Gi"), cache=True,
      cache_version=OVERLAY_CACHE_VERSION,
      cache_serialize=True)
def precompute_overlay_shard(job: OverlayShardJob) -> OverlayShardResult: ...

@workflow
def wf_precompute_overlays(model_version: int, dataset: str, version: str = "v2.1",
                           base_seed: int = 0) -> OverlaySetManifest: ...
```
Note the absence of any `source`/`split`/`prompt_version`/`field`/`value` inputs — overlays are canonical.

**Inside the task:**
1. Workflow resolves the registry coordinate once: `MODELVER#{registered_model_name}#{model_version}` → `{run_id, artifact_uri, checkpoint_sha256}`; download `best.pt` once; `load_policy` also (re)computes `sha256` and asserts it equals `checkpoint_sha256`. Read `val_fraction` from `config.yaml` **only** to inform display-time filtering metadata (not to gate compute).
2. Build `make_pre_extracted_loader` over the **whole** shard (all samples; identical decode path for all downstream sources).
3. **Idempotency/resume:** HEAD the target `overlay.bin.gz`; skip only when all
   object identity metadata matches. Otherwise fail. A missing body is created
   with `If-None-Match: *`, so concurrent retries cannot overwrite each other.
4. `predict_control` in **batches** (start `batch_size=32`, halve on OOM), overlapping CPU decode with GPU forward. Per-sample `initial_noise` (from `sample_uid`) means an OOM-triggered batch-size halving does **not** change any sample's noise.
5. Emit per shard:
   - **ONE binary `overlay.bin.gz`** (§6 layout) covering all samples: directory `sample_uid → offset`, `controls float32[N, seeds, 64, 2]`, `v0 float32[N]`.
6. **Write artifacts FIRST, readiness LAST:** conditionally create the S3
   `overlay.bin.gz`, conditionally create the
   `SHARD#{ds}#{ver}#{shard}` / `MODEL#{model_artifact_id}` pointer, then
   conditionally create the immutable audit manifest. Finally transition
   `OVLSET#{model_artifact_id}#{ds}#{ver}` from `building` to `ready`, guarded
   by request identity, dataset manifest, and artifact bucket. A retry of an
   already-ready identical set is a no-op; it never writes `building` over
   `ready`. No `SCENELIST#`/`gsi1` writes (P1.6).

**Caching (flyte #4, infra #10, P1.11).** Flyte's static
`OVERLAY_CACHE_VERSION` versions the AOVL/inference/noise contracts, while task
inputs and the publication-layer `request_identity`/`cache_identity` carry the
runtime content identity. They are deliberately narrower than the repo-wide
git SHA. The runtime cache identity is:
```
cache_identity = hash(model_artifact_sha256, dataset_manifest_digest, preprocessing_contract_digest,
                      model_inference_code_digest, sampler, num_steps,
                      noise_policy_version, overlay_binary_schema)
```
The current coarse task receives one partition `FlyteDirectory` plus scalar
identity inputs and the resolved checkpoint. `cache_serialize=True` prevents
duplicate execution for an identical Flyte cache key; conditional S3/Dynamo
writes provide the authoritative cross-execution race guard. Since we store
**raw control**, a later fix to the integrator / yaw-sign / display mode is a
client change and does **not** bump `cache_identity`.

**Reproducibility, two-tier (P1.10).** Drop any "byte-identical" claim. Instead:
- **Same-environment reproducibility (identical):** pin `container_image_digest`, `gpu_model`, `cuda`, `cudnn`, `torch`; with per-sample batch-independent noise this yields identical outputs.
- **Cross-environment (numerically close, no bitwise guarantee):** different GPU/driver/cuDNN may differ in the last ULPs.
Record in the manifest: `container_image_digest, torch/cuda/cudnn versions, gpu_model, model_artifact_sha256, dataset_manifest_sha256, inference_contract_version, noise_policy_version, output_sha256`. `torch.use_deterministic_algorithms(True)` + `cudnn.deterministic=True; benchmark=False` are set to make same-env runs stable, but we do not promise cross-env bit-equality.

**Seed policy (adas #4, P0.1).** `base_seed=0` by default (per-sample noise from `hash64(model_artifact_id, dataset_manifest_digest, sample_uid, base_seed)`; the seed is recorded in the binary + manifest). A single FlowMatching draw is one sample from the predictive distribution and can sit in the tail — the UI MUST label "single sample (base_seed 0)." A seed-fan (`seeds=(0,1,2,3)`) is supported by the binary layout's `seed_count` dimension; the client can draw the median path + envelope. Bezier ignores noise (store `seed_count=1`).

**Resolving the three sources at DISPLAY time (not compute; gaps #1, #6; leak facts):**
- **train/eval:** the packer records `split_group_uid` and its stable `blake2b(..., digest_size=8) mod 10` bucket in every sample's `meta.json`. The Go index exposes `split_bucket`; the frontend combines it with the selected model's `val_fraction`, using `bucket < round(val_fraction*10)` ⇒ `eval` else `train`. If `val_fraction==0`, `eval` is empty and all samples are `train`. These are filters over the one canonical overlay. `val_fraction` is logged as an MLflow param.
- **search:** the existing `QueryScenesByLabel → ResolveSampleShards` path returns `(shard, sample_uid)` for `(dataset, prompt_version, field, value)`; the UI shows exactly those samples from the canonical shard overlay. No `split_role`, no separate compute.

### Version-coordinate consistency (infra-dynamo #3) — MANDATORY invariant

Overlay `SHARD#`/`MODEL#` pointers, the playback `IDX#`, and `ResolveSampleShards`' published version **MUST all be the same `version` string = `v2.1`** (P0.5). Publish `v2.1` shards, rebuild `IDX#` at `v2.1`, point `resolveVersion`/search at `v2.1`, run `wf_precompute_overlays(version="v2.1")`. **Sample identity is the content-addressed `sample_uid`** (P1.7), so a shard-enumeration shift no longer silently re-points scenes; the migration manifest maps `legacy_sample_id (s{si:08d}) → sample_uid → shard/member`. A single shared key-builder is the source of truth for composite keys. Reasoning labels are keyed by `prompt_version` (not dataset version) but by `legacy_sample_id` today — see the P1.7 migration blast-radius below.

### 4b. Data-gen change: v2.1 REPACK — pack `pose_current`, GPS path, rig projection, geo stats (gaps #4, #7, #8)

**This is a FULL re-pack to `v2.1` (user decision 3, P0.5). There is no decode-free in-place backfill into v2.0 anywhere in this design.** The repack re-decodes + re-encodes all 7 L2D cameras per sample (PyAV decode + JPEG re-encode dominate cost) — a heavy Flyte run; give a wall-clock/cost estimate before triggering.

**Loader plumbing is new (flyte #8, #10).** `L2DDataset.__getitem__` builds egomotion via `extract_egomotion` and **never surfaces lat/lon or absolute heading** today. Add both, aligned to the **same `sample_idx`** `extract_egomotion` chooses. Note the window is `history [idx-64:idx]` + `future [idx+1:idx+65]` = **64+64**, not 65.

**Explicit per-sample `pose_current` (P0.4).** Map placement needs the **absolute heading**, which must come from the **RAW vehicle-state heading** (L2D `heading` column), NOT derived from the GPS point sequence (unstable at low speed / GPS jitter). Store:
```python
# pose_current, packed as pose.npy (structured / fixed layout), float64 except accuracy:
{
  "latitude_deg":            f64,   # raw hp_loc_latitude at idx
  "longitude_deg":           f64,   # raw hp_loc_longitude at idx
  "heading_deg_cw_from_north": f64, # raw vehicle heading (compass), NOT from GPS deltas
  "timestamp_ns":            i64,
  "gps_accuracy_m":          f32,   # if available; else NaN
}
```

**Two distinct GPS products (flyte #8):**
1. **Per-sample `pose.npy`** (above) + the sample's future window `[lat, lon]`, shape `(1+64, 2)` float64 aligned to `idx` and `idx+1..idx+64`, for per-scene "where am I + where am I going" and for §9-bis map placement of the predicted path.
2. **Per-episode full path artifact** for the Map view driven route: the *whole episode's* `[lat,lon,heading,timestamp_ns]`, written once per episode as `geo/episode_paths/{dataset}/{version}/{episode_id}.f64`. A 6.4 s future window cannot draw a driven route.

**Precision (adas #8, cost #9): store lat/lon + heading as float64.** float32's 24-bit mantissa gives ~7 significant digits; `48.8566xx` spends 2–3 before the decimal, leaving ~1–2 m of quantization jitter — visible when animating an ego marker. Store `pose.npy` / episode path as **float64** (or int32 fixed-point ENU offset from a per-episode float64 origin). This is the one field where the `ego.npy`-style float32 is wrong. Do NOT downcast for the §9-bis map path.

```python
# parallel_pack.pack_sample, after ego.npy (v2.1 repack):
pose = sample.get("pose_current")          # dict above; loader provides from RAW heading + GPS
if pose is not None:
    members["pose.npy"] = pack_pose_f64(pose)          # + npy header
gps_fut = sample.get("gps_future")         # (1+64,2) float64 [lat,lon]
if gps_fut is not None:
    members["gps.npy"] = np.asarray(gps_fut, np.float64).tobytes()
```
NVIDIA has no GPS ⇒ members simply absent (`ShardIndex.has_gps=False`).

**Camera-projection is a v2.1-repack RIG artifact and a CONSTANT (P1.12).** Move projection generation OUT of the per-model overlay task (race/duplication-prone, wrong responsibility) INTO the v2.1 repack, which already holds pose/calibration. For a fixed camera rig, ego-frame-trajectory→camera-pixels depends only on the rig's fixed intrinsics/extrinsics, NOT on ego pose. PR#74's implementation takes one fixed `P = K[R|t]` matrix and confirms that no sample state enters the projection. Store each unique rig at `rig/{sha256}.json`, put its `{key, sha256}` descriptor on every `shard_entry`, and **drop the per-sample `proj.f32` entirely**. A dataset-wide `rig/projection.json` is invalid when scenes were recorded with different calibrated rigs.

**Geo stats emitted during the repack (P1.14).** The repack already scans GPS, so emit in the same pass (no round-trip through DynamoDB for the heavy data):
```
geo/episode_paths/{ds}/{version}/{episode}.f64      # full driven routes
geo/sample_pose.parquet                             # per-sample pose columns for aggregation
geo/summary.json                                    # bbox, per-region counts, total
geo/heatmap.fgb  (FlatGeobuf)  |  geo/heatmap.geojson.gz
```
DynamoDB then gets only the serving summary + S3 pointer (§9, §6).

**Index (flyte #10):** populating `ShardIndex.has_gps` + a denormalized `IndexSample.gps_now[2]` / `heading_now` (float64) requires **new Go npy-decode logic** in `BuildShardIndex` (parse npy header + little-endian floats), analogous to the existing `ego_now/ego_future` extraction — and a rebuild of affected `IDX#` items at `v2.1`. This is real Go work, not a free side effect; scope it. `gps_now` feeds only cheap serving reads (the heavy geo aggregation is done in the repack, P1.14 — never re-inflate gzip `IDX#` blobs to read one lat/lon).

**Sample identity — `sample_uid` (P1.7).** Since v2.1 is already a breaking repack, introduce a content-addressed sample id and stop trusting the fragile global enumeration:
```
sample_uid = hash(dataset_name, source_episode_id, source_frame_idx_or_timestamp_ns)
```
Keep `legacy_sample_id = s{si:08d}` for compat and emit a migration manifest `legacy_sample_id → sample_uid → shard/member`. A 1-item order shift on repack no longer re-points every downstream scene.

> **FLAG — reasoning-subsystem migration blast radius (bigger than the review implies; state to user).** The SHIPPED reasoning-label cache (`reasoning_labels_cache/dataset=/teacher=/prompt_version=/{sample_id}.json`) and the live `LBL#…/SCENE#{sampleID}` DynamoDB index are keyed by `s{si:08d}`. Adopting `sample_uid` means either **re-keying those** or **maintaining the `legacy_sample_id ↔ sample_uid` mapping** at every read. Recommendation: adopt `sample_uid` for robustness, but treat the reasoning-cache + `LBL#` re-key as its **own scoped work item**, gated behind the migration manifest. This is item (4) flagged to the user in §13.

---

## 5. S3 Layout

Bucket `auto-e2e-platform-artifacts-381491877296` (us-west-2). Keys are a pure function of identity so the Go API builds them without a lookup. `schema=v1` sits high so a binary-format change bumps the S3 path and cache identity together. **No `split=`/`source=` segment anywhere** (P0.2). **No `b={hash(shard)%16}` prefix bucketing** (P1.13): S3 auto-scales to 3,500 PUT / 5,500 GET per prefix and the write is near-serial on one GPU; keep a simple human-readable prefix and add sharding only if measured 503s appear (revisit if backfill later parallelizes under Kueue).

**Overlay body (binary, canonical per model+shard):**
```
overlays/schema=v1/model={model_artifact_id}/dataset={l2d}/version={v2.1}/shard={train-000000.tar}/
    overlay.bin.gz          # gzipped binary; the SOLE overlay body (P0.3). No split segment.
```

**`overlay.bin.gz` binary layout (P0.3)** — after gunzip (little-endian):
```
offset  type            field
0       char[4]         magic          = "AOVL"
4       uint16          format_version = 1            (== overlay_schema "v1")
6       uint16          flags          (bit0: bezier/deterministic → seed_count meaningless)
8       uint32          sample_count   = N
12      uint16          seed_count     = S            (1 for base_seed only; 4 for a fan)
14      uint16          horizon        = 64
16      uint16          dims           = 2            ([accel_x, curvature])
18      uint16          reserved       = 0
20      int64[S]        base_seeds                    (the recorded per-sample noise seeds)
...     Directory[N]:   { uint64 sample_uid_hash; uint32 sample_index }   (sorted by sample_uid_hash)
...     float32[N,S,64,2]  controls                   (accel_x, curvature)
...     float32[N]         v0                         (speed at idx-1, m/s)
```
The **directory maps `sample_uid → row**, so a search subset or a scrub to one sample indexes directly in-memory after a single GET; there is no per-sample S3 request. Frontend: `fetch(url)` (served `Content-Encoding: gzip`, browser inflates, or gunzip in a worker) → `ArrayBuffer` → `Float32Array` subarray views (zero-copy, no `JSON.parse`, no per-frame GC). This binary layout is exactly what the frontend already assumed (P0.3 contradiction resolved).

**Per-run manifest (lineage/audit; no split):**
```
overlays_manifest/schema=v1/model={model_artifact_id}/dataset={l2d}/version={v2.1}/manifest.json
    { registered_model_name, model_version, run_id, model_artifact_sha256,
      dataset, version, dataset_manifest_sha256, n_shards, n_samples, seeds, sampler, num_steps,
      inference_contract_version, noise_policy_version, overlay_binary_schema,
      container_image_digest, torch_version, cuda_version, cudnn_version, gpu_model,
      output_sha256, created_at, status:"ready" }
```

**Rig projection (content-addressed per rig, shard-bound — P1.12):**
```
rig/{rig_sha256}.json               # intrinsics/extrinsics per camera; one object per unique rig.
shards/manifest.json                # each shard_entry carries rig {key, sha256}.
                                     # PR#74 confirms no per-sample proj.f32 is required.
```

**Geo (produced by the v2.1 repack — P1.14):**
```
geo/episode_paths/{ds}/{version}/{episode}.f64
geo/sample_pose.parquet
geo/summary.json
geo/heatmap.fgb | geo/heatmap.geojson.gz
```

**GPS per-sample** lives in the datasets bucket inside the v2.1 shard tar: `pose.npy` + `gps.npy` members.

**Optional baked export (demoted, §7):** if ever produced,
```
overlays_export/schema=v1/model={model_artifact_id}/…/scene={sample_uid}/clip.mp4    # ONE mp4 per scene, NOT 64 JPEGs
```

**Lifecycle (infra #8).** Add an **S3 lifecycle policy** on `overlays/` and `overlays_export/` keyed by `schema=`/`model=` so retiring an `overlay_schema` or an experimental model actually reclaims bytes — Dynamo TTL only deletes pointers, not S3 objects.

---

## 6. DynamoDB Schema Additions

Single table `auto-e2e-console`, prefix-namespaced. **The overlay BODY is in S3; DynamoDB holds only pointers + small metadata (P0.3).** The model↔scene relation collapses to model↔SHARD (P0.2, P1.6), so **no `gsi1` and no `SCENELIST#` fanout are required for P1**.

| # | Access pattern | Query |
|---|---|---|
| P1 | which models have a ready overlay for **scene Y** | resolve scene→shard (existing), then base `Query pk=SHARD#{ds}#{ver}#{shard}` → all `sk=MODEL#{model_artifact_id}` items |
| P2 | **model X's** overlay for a **shard** (playback) | base GetItem `pk=SHARD#{ds}#{ver}#{shard}`, `sk=MODEL#{model_artifact_id}` → S3 pointer → 1 S3 GET of `overlay.bin.gz` |
| P3 | all shards **model X** was computed on | base Query `pk=OVLSET#{model_artifact_id}#{ds}#{ver}` + manifest, or `Query gsi` deferred |
| P4 | reasoning-search scenes + overlay for **model X** | existing `LBL#` Query → group by shard → P2 per shard; filter rows by `sample_uid` in-memory |

> **P1.6 — supersedes the user's round-2 gsi1 decision (flagged to user).** With canonical per-shard overlays, "which models for this scene" is ~50 shards × 10 models ≈ **500 items**, not 50k scenes × 10 = 500k edges. P1 is a single base-table `Query pk=SHARD#…`. This **removes** the scene×model `gsi1` inverse index and the `SCENELIST#` fanout entirely. *"This supersedes the earlier decision to IaC + use `gsi1` for P1 (that decision was made under the per-scene-edge design); `gsi1` is no longer required for P1. Whether to still provision `gsi1` for future inverse lookups is deferred to the user."* A coverage bitmap / `sample_uid`-set digest on the SHARD×MODEL item is only needed IF partial inference is ever introduced — not now, since train∪eval=all samples.

### New item types

**(1) SHARD×MODEL overlay pointer — serves P1 + P2 (P0.2, P0.3, P1.6):**
```
pk = SHARD#{dataset}#{version}#{shard}       e.g. SHARD#l2d#v2.1#train-000000.tar
sk = MODEL#{model_artifact_id}               e.g. MODEL#3f9a…(sha256 of best.pt)
attrs (POINTER ONLY — no overlay body):
  s3_key         S   overlays/schema=v1/model=3f9a…/dataset=l2d/version=v2.1/shard=train-000000.tar/overlay.bin.gz
  sha256         S   <sha256 of overlay.bin.gz>
  byte_size      N   <bytes>
  sample_count   N   1000
  overlay_schema S   "v1"
  dataset_manifest_sha256 S
  cache_identity S
  status         S   "ready"
  created_at     S   <iso8601>
  # small projected model attrs for the picker (avoid a 2nd GetItem):
  registered_model_name S, model_version N, run_id S (provenance), model_name S,
  eval_ade N, eval_fde N, val_fraction N
```
P1 = `Query pk=SHARD#{ds}#{ver}#{shard}` → every model with a `ready` overlay for that shard, with picker attrs inline. P2 = GetItem the specific `sk` → follow `s3_key` → one S3 GET. Writes spread by shard in the pk (no hot partition; §hot-partition).

**(2) Model run profile — run-level metadata (infra #5):**
```
pk = MODEL#{model_artifact_id}   sk = META
attrs: registered_model_name S, model_version N, run_id S (provenance),
       model_name S, eval_ade N, eval_fde N, eval_gate_pass N,
       dataset S, train_execution_id S, val_fraction N, created_at S
```
`eval_ade`/`model_name` are **run-level, not scene-level** — stored once here (and lightly projected onto the SHARD×MODEL pointer), never denormalized per scene.

**(3) Registry coordinate → artifact (P1.8):**
```
pk = MODELVER#{registered_model_name}#{model_version}   sk = META
attrs: run_id S, artifact_uri S, checkpoint_sha256 S   # checkpoint_sha256 == model_artifact_id
```
Picker maps a chosen (registered model, version number) → `run_id` + `artifact_uri` + content id. A bare `VER#{version}` is rejected: it collides if a second registered model appears. MLflow version *numbers* are immutable; aliases move — so the picker resolves an alias to a number first, then to this item.

**(4) Overlay-set status singleton (write-then-index gate; flyte #5) — no split:**
```
pk = OVLSET#{model_artifact_id}#{dataset}#{version}   sk = META
attrs: status S ("building"|"ready"|"deleted"), n_shards N, n_samples N, seeds L,
       manifest_key S, overlay_schema S, dataset_manifest_sha256 S,
       request_identity S, cache_identity S, artifacts_bucket S, created_at S
```
The Go API checks `status=ready` and pointer/gate identity agreement before
advertising. The writer creates `building` only when the key is absent and
conditionally publishes `ready`; an identical retry preserves the original
`created_at`, while a conflicting request fails. **TTL/lifecycle coherence
(flyte #5):** never TTL a `ready` overlay while its Flyte catalog entry still
reports "cached" (→ "advertised but bytes 404"). Retiring an overlay MUST (a)
flip `OVLSET#`/the SHARD×MODEL pointers to `deleted`, (b) delete S3 objects via
lifecycle/explicit delete, and (c) purge the Flyte Datacatalog entry (or bump
`overlay_schema`). Never TTL just the pointer.

**(5) ODD geo-stats — serving summary + pointer only (§9, P1.14):**
```
pk = GEO#{dataset}#{version}   sk = META
attrs: summary S (inline JSON: bbox, per-region counts, total),   # SMALL
       geojson_key S  (S3 pointer to geo/heatmap.* + geo/summary.json),
       n_samples N, computed_at S
```
Heavy geo data (point sets, heatmap, parquet) is produced by the v2.1 repack into S3 (P1.14); Dynamo holds only the serving summary + pointer, well under 400 KB.

### Hot-partition (adas #5, infra #12)
`SHARD#{dataset}#{version}#{shard}` spreads writes across shards naturally (the shard is in the pk), so bulk `PutItem` is **not** concentrated on one partition. With only ~500 pointer items total (50 shards × 10 models) and one near-serial GPU writer, write pressure is trivial. On-demand + adaptive capacity absorbs the rest. No `gsi1` write amplification to size (P1.6 dropped it).

---

## 7. Pre-rendered frames vs vector overlay — the decision, with math

**Recommendation: VECTOR-FIRST for all three sources and both views. Demote PR#74 baking to an optional offline export.** The user leaned pre-rendered; here is the honest math and the reason the lean does not survive it.

**Storage (cost-frontend #1, canonical/split-free numbers):**
- **Vectors (canonical, split-free):** `(64,2)` float32 + `v0` ≈ 520 B/sample/model in the binary body. Because there is **one** overlay per (model, shard) covering all samples (not one per split), 50 k samples × 10 models ≈ **~260 MB total** across S3, plus one small object per unique rig.
- **Baked (front-camera only):**
  - If playback = consecutive samples (matching the existing engine): **1 baked JPEG/sample/model** → 50 shards × 1000 × ~60 KB ≈ **~3 GB/model** → ~30 GB @ 10 models.
  - If each sample is a 64-step future clip: **64 frames/sample** → **~190 GB/model** → **~1.9 TB @ 10 models**, and it abandons the windowed `/blob` engine.
  The draft's "10 models × 1 k scenes × 20 MB = 200 GB" matches neither layout nor real shard counts.

**The decisive fact (cost-frontend #2, adas #3, flyte #7, P1.12):** baking's *only* claimed advantage was camera projection under L2D `geometry_type='pseudo'`. But calibrated projection is a function of the **fixed camera rig**, not model weights, and PR#74 confirms it is a **per-rig constant**. Store each rig once and resolve it through the active shard (§5), and the client draws *any* model's polyline in camera space. Baking therefore buys nothing that vectors + a rig projection don't, at ~3–4 orders more storage. And under pseudo-geometry the projection is a heuristic approximation **either way** — baking just freezes the same error into un-auditable pixels.

**Flexibility:** vectors toggle/compare 2–3 models on one canvas at ~0.5 KB each; baked frames make multi-model comparison physically impossible (can't composite two videos) and make a single toggle a multi-MB re-download (cost-frontend #5). The feature centers on *picking* (and comparing) models — vector-first is the correct default.

**Requirement (a) coverage (cost-frontend #3):** because overlays are canonical over ALL samples in a shard, the train/eval/search UX is uniform for free — the same body serves every display filter, including the leaked-train set. The draft's "bake eval+search only" silently dropped train-set camera overlays.

**When baking IS right (stated for honesty):** (i) a shareable/exportable clip for a doc or Discord where no interactivity is needed; (ii) a static thumbnail. In those cases emit **one MP4 per scene** (H.264 ~10–50× smaller than a JPEG sequence; `<video>`-playable), scoped and TTL'd — never as the interactive playback path, and never as loose per-frame JPEGs (cost-frontend #4, infra #6).

This honors the user's intent (heavy Flyte precompute, "smooth playback") while spending bytes and flexibility correctly.

---

## 8. Frontend Design (Next.js)

### Scene view: model-picker + vector overlay playback
- **Model-picker** populated by P1 (`/scenes/{…}/models` → resolve scene→shard → `Query pk=SHARD#…`) — only models with a `ready` overlay for the current scene's shard, labeled `model_name` + `eval_ade` + the honest split tag ("training set" / "episode/clip hold-out"). The split tag is computed **at display time** from `val_fraction` + the packed episode/clip-level `split_bucket`, not from any per-split artifact.
- **Playback** reuses the existing per-frame JPEG byte-range path from `ShardIndex.members`. On model select, one GetItem yields the SHARD×MODEL pointer → one S3 GET loads `overlay.bin.gz` (all samples for the shard) → held in memory.

**Rendering (vector, both views):**
- **Two-layer canvas:** static frame layer (existing `<img>`/bitmap) + one transparent overlay `<canvas>`. Toggling/adding a model = `clearRect` + re-stroke the thin top layer only.
- **Client integrates raw control → XY** with the **reference Python integrator ported to TS, guarded by PY↔TS golden tests** (P1.9). The client **DEFAULTS to raw prediction** (no clamping). A **"display-limited" mode** is an explicit, labeled toggle that applies GT-style processing for visualization only — it is NOT the default and the label states it post-processes/hides model error. There is **no "clamp parity" claim**: the 0.5 m/s floor / ±0.5 rad/m clamp are GT *derivation* properties, not integrator behavior, so applying them to a prediction is display post-processing, not "identical treatment." For any eval-metric comparison, use the exact eval-code processing path (not the display toggle).
- **BEV view:** well-defined `meters_per_pixel` map (`metrics.py::offroad_rate` convention: forward `+x`→up/decreasing row, left `+y`→left/decreasing col). Metrically sound.
- **Camera view:** draw the integrated polyline through the active shard's **rig projection** (`rig/{sha256}.json` via `/datasets/{name}/shards/{shard}/rig-projection` — P1.12). Under pseudo-geometry this is approximate but **auditable and correctable** (unlike baked pixels), and identical for every model using that shard.
- **Binary payload:** `fetch(url).then(r=>r.arrayBuffer())` → parse the `AOVL` header → `Float32Array` `subarray` views over `controls`/`v0` (zero-copy, no `JSON.parse`, no per-frame GC). Directory maps `sample_uid → row`.
- **Frame-locked sync (cost-frontend #7):** there is **no `<video>` element** — playback is a manual JPEG-swap loop over tar byte-ranges, so `requestVideoFrameCallback` does **not** apply. The overlay is indexed by the same integer sample ordinal (→ `sample_uid`) that drives the image swap; the draw is triggered off the frame-advance step, not a wall-clock rAF, so image and overlay cannot drift under buffering stalls. `OffscreenCanvas`/worker only if profiling shows multi-cam main-thread jank.
- **Seed labeling (adas #4):** if `seed_count=1`, badge "single sample (base_seed 0)"; if a fan, draw median + envelope.

### Map view: driven GPS path (+ predicted path, §9-bis)
Per scene and per episode: fetch the **per-episode `geo/episode_paths/*.f64`** (full driven route) and draw the `[lat,lon]` polyline on MapLibre/Leaflet vector tiles; animate an ego marker locked to the same frame clock. The predicted path is placed via the §9-bis transform over the same raw control blob + `pose_current`. `gps_to_map.py` is used only for a static L2D-styled thumbnail (Overpass/`osmnx` is slow + needs internet — render offline in Flyte if used). See §9-ter for privacy constraints on exact-route rendering.

### ODD geo-stats page
Reads `GET /datasets/{name}/{version}/geo-stats` → inline summary (bbox, per-region counts) for KPIs + the S3 `geojson_key` for a heatmap/cluster map ("where was this data collected"). Follows the `dataviz` skill conventions. Subject to §9-ter (min zoom, k-anonymity suppression).

---

## 9. GPS / Map / ODD

- **Packing (§4b):** per-sample `pose.npy` (**explicit `pose_current`: lat/lon/heading_cw_from_north/timestamp_ns/accuracy, from RAW vehicle heading**, P0.4) + per-sample `gps.npy` future window + per-episode `geo/episode_paths/*.f64` full route. All float64. L2D loader slices raw `vehicle_states[:, 3:5]` for lat/lon and the raw `heading` column for absolute bearing; NVIDIA omits. **Full re-pack to `v2.1`** (P0.5); `ShardIndex.has_gps` + `IndexSample.gps_now[2]`/`heading_now` (float64) via new Go npy-decode.
- **Geo-aggregation is produced during the v2.1 repack (P1.14), not via DynamoDB round-trips.** The repack scans GPS once and emits `geo/episode_paths/*`, `geo/sample_pose.parquet`, `geo/summary.json`, `geo/heatmap.fgb|geojson.gz`. Reverse-geocode to per-region counts offline via cached OSM. DynamoDB `GEO#{dataset}#{version}/META` gets ONLY the serving summary (bbox, per-region counts) + the S3 `geojson_key`. Never re-inflate multi-MB gzip `IDX#` blobs to read one lat/lon.

---

## 9-bis. Predicted trajectory ON the GPS map (user decision 2 — in scope)

The predicted ego-frame trajectory is overlaid on the geographic map in addition to camera/BEV. This is the highest-error-risk render path; it needs an explicit **error budget** and is gated behind a validation harness before being trusted.

**Placement math (CORRECTED — P0.4).** The integrated prediction is ego-frame `(x_forward, y_left)` metres (§10). To place it on the map we need, at the current sample: the absolute origin `(lat0, lon0)` and the **absolute heading** `ψ0` (compass bearing, CW-from-north) — both from the explicit `pose_current` (§4b), taken from the RAW vehicle heading, NOT from GPS point deltas (which are unstable at low speed / under jitter). For x=forward, y=left, ψ=bearing CW-from-north, the correct ENU is:
```
east  = x·sin(ψ0) − y·cos(ψ0)
north = x·cos(ψ0) + y·sin(ψ0)
lat = lat0 + north/R·(180/π);  lon = lon0 + east/(R·cos(lat0))·(180/π)
```
**Sanity check at ψ0 = 0 (heading due north):** a point purely to the left (`y_left>0`, `x=0`) must move **WEST** (`east<0`). With the corrected formula `east = −y·cos(0) = −y < 0` ✓. The draft's `east = x·sinψ + y·cosψ` moved it EAST — it implicitly treated `y` as RIGHT. Fixed.

**Error budget (each term must be resolved, not assumed):**
1. **Yaw-sign / heading convention (dominant risk).** The BEV overlay uses ego-relative `+y=left` and may already be mirrored (§10). Placing on the map additionally needs the **absolute** compass bearing `ψ0`. L2D `heading` is compass (CW-from-north, degrees); `integrate_trajectory`'s internal `θ` is math CCW starting at 0. **The corrected ENU formula is NECESSARY but NOT SUFFICIENT (P0.4):** it composes with the L2D yaw-sign mirror (§10) and the heading *source*. Trusting the corrected formula alone gives false confidence. **Action:** validate map placement **JOINTLY** on (a) a known straight clip and (b) a known left AND right turn — the predicted path must lie ON the driven GPS path when the model predicts near-GT; only then trust turns. A mirror is far more obvious on a real road than in BEV.
2. **Float precision.** `pose_current` and the path are float64 (§4b) precisely so the map origin doesn't jitter; do NOT downcast for this path.
3. **`v0` / one-frame gap (§10).** ~0.2 s scale offset — small but visible against real roads; note it.
4. **Pseudo-geometry.** L2D has no calibration; the *shape* is metric (unicycle integration is calibration-free), so map placement is actually **less** sensitive to pseudo-geometry than the camera-pixel projection is — the map path depends only on integration + origin + bearing, not on camera intrinsics. A point in favor of the map view.

**Reuse:** placement is a pure client transform over the SAME raw `(64,2)+v0` control blob + `pose_current` — **no extra Flyte artifact**. GT (`ego_future`) places on the map by the identical transform, so pred-vs-GT-vs-driven-path can be compared on one map. Acceptance: the joint harness (§10) must pass for the map path specifically.

## 9-ter. GPS privacy (P1.15)

Exact GPS traces are personal-location data; the ODD map and any per-episode route expose where a vehicle drove. Requirements:
- **AuthN/AuthZ on the exact routes:** the `gps-path` / `episode_paths` endpoints require authenticated console access and are authorized per role; raw float64 routes are not public.
- **Min zoom / coarsening:** the ODD map only renders points above a minimum zoom, and below that shows coarsened aggregates (geohash bins), never raw points.
- **Endpoint fuzzing:** exclude or fuzz trip **start/end** points (home/depot inference risk).
- **k-anonymity suppression:** suppress geohash cells whose sample count is below a minimum threshold (`k`), both in `geo/heatmap.*` and the served summary.
- **Auditing:** log dataset-export and screen-access to exact routes.
- **Map-tile ToS:** document the tile provider's Terms of Service + required attribution (MapLibre/OSM); keep attribution visible.

Current production gate: privacy-filtered heatmaps are available without exact
coordinates, but exact sample poses, tar ranges containing exact pose/GPS
members, and episode paths are denied while `EXACT_GEO_ENABLED=false`. Do not
enable it behind the current unauthenticated CloudFront distribution:
viewer-supplied role headers are ignored, and the API only accepts roles from
a principal established by signature-validating authentication middleware.

---

## 10. Determinism, Coordinate Contract, Versioning, Cache-Invalidation

### Coordinate & display contract (adas #1, #9, #10, #11; P1.9) — resolve BEFORE trusting any overlay
- **Yaw-sign mirror is a real hazard.** `_derive_signals` builds `yaw_rate = diff(heading)/dt` from L2D's **compass heading (clockwise-from-north)**, while `integrate_trajectory` uses math-positive CCW (`y = v·sin θ`, `θ += curvature·v·dt`). A physical **right** turn (heading increasing) yields `curvature>0 → +θ → +y`, which the BEV convention (`+y→left`) renders as a **left** turn. This is invisible in ADE/FDE (GT and pred integrate identically) but **flips every rendered overlay left/right**. **Action:** verify the sign against a known turning clip and, if mirrored, apply the correction **in the shared client integrator** (so GT and pred flip together). Because we store raw control, this fix is a render-only change — no GPU recompute. It composes with the §9-bis map ENU fix; validate jointly (P0.4).
- **Display mode, NOT "clamp parity" (P1.9).** The 0.5 m/s floor and ±0.5 rad/m clamp live in GT curvature **derivation** (`curvature = yaw_rate/max(speed,0.5)`; `clip(±0.5)`), **not** in `integrate_trajectory`. Clamping a model's **predicted** curvature at render time "to match GT" is display post-processing that **hides model errors**, not identical treatment. Therefore:
  - **Default render = RAW prediction** (no floor, no clamp).
  - **"Display-limited" mode** is an explicit, labeled toggle that applies the GT-style processing for readability only; the label states it suppresses model error.
  - **Eval-metric comparison** uses the exact eval-code processing path (not the display toggle).
  - The reference integrator is ported Python→TS with **PY↔TS golden tests** so pred and GT are integrated by provably identical code.
- **`v0` staleness (adas #10):** `v0 = ego_history[-1, 0]` is speed at `idx-1`; the future begins at `idx+1` (~0.2 s / one-frame gap). Harmless for ADE (GT shares it) but a small absolute scale offset when laid on the real scene — note it in the contract.
- **Channel truth (adas #11):** reference `egomotion.py` (`[speed, accel_x, yaw_rate, curvature]`), not the stale README ("yaw angle"). `v0 = channel 0` is unaffected.

### Determinism / noise contract (P0.1)
- Per-sample initial noise `z0 = noise_from(hash64(model_artifact_id, dataset_manifest_digest, sample_uid, base_seed))`, fed via the new `initial_noise=` kwarg on `FlowMatchingPlanner.forward`. This is **batch-invariant**: a sample's noise is independent of batch size / order / OOM-halving / retry.
- **Nuance (state honestly):** a stored `overlay.bin.gz` is deterministic *as stored*; this fix targets **recompute batch-invariance + cross-run reproducibility**, not the correctness of an already-written blob. Bezier ignores noise entirely.
- `noise_policy_version` is recorded in the manifest + cache identity.

### Versioning axes
- `overlay_schema` / `overlay_binary_schema` (binary + render contract) — in S3 path + cache identity.
- `model_artifact_id = sha256(best.pt)` (durable, content-addressable identity; NOT `run_id`, NOT the moving alias).
- shard `version` = **`v2.1`** (dataset packing) — one value across `SHARD#`/`IDX#`/search (§consistency).
- `dataset_manifest_digest` — v2.1 is **immutable**; its manifest digest / S3 object-version pins the data (P1.11).
- Seeds / `noise_policy_version` (in the binary + manifest + cache identity).

### Cache-invalidation (P1.11)
- Cache identity = `hash(model_artifact_sha256, dataset_manifest_digest, preprocessing_contract_digest, model_inference_code_digest, sampler, num_steps, noise_policy_version, overlay_binary_schema)` — **NOT the repo-wide git SHA** (a Next.js edit must not invalidate the GPU cache).
- A render/integrator/display-mode fix bumps nothing on the GPU side (client-only) — a direct benefit of storing raw control.
- A binary-schema change bumps `overlay_binary_schema` → new S3 prefix + new cache identity + lifecycle reclaims the old prefix.

### Reproducibility (two-tier — P1.10)
- **Same-environment:** pinned container digest + GPU/CUDA/cuDNN/torch + batch-independent noise → identical outputs. Set `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True; benchmark=False`.
- **Cross-environment:** numerically close, **no bitwise guarantee**. Do not claim byte-identical.
- Manifest records: `container_image_digest, torch/cuda/cudnn versions, gpu_model, model_artifact_sha256, dataset_manifest_sha256, inference_contract_version, noise_policy_version, output_sha256`.

### Storage-cost summary
- Vectors (canonical, all sources/views): **~260 MB S3 body** total + pointers in Dynamo + one small object per unique rig — trivial.
- Optional MP4 export: bounded by scope + S3 lifecycle TTL; never on the playback path.

---

## 11. Implementation and Rollout

### Completed implementation

- **Phase 0:** immutable MLflow checkpoint resolution, model picker, AOVL parser,
  shared trajectory math, raw/display-limited rendering contract, and golden
  tests.
- **Phase 1:** v2.1 stable identities, portable pose/GPS members, episode paths,
  rig projection, geo aggregation, embedded reasoning-label migration,
  immutable dataset publication, and Console v2.1 search/index support.
- **Phase 2:** per-sample deterministic noise, canonical GPU overlay
  precompute, write-once S3/Dynamo publication, ready-gated API endpoints, and
  BEV/camera/map overlays.
- **Operations:** digest-pinned registration, runtime contract digest checks,
  `wf_publish_and_precompute_overlays`,
  `wf_create_publish_and_precompute_overlays`, and the VPC-local
  `auto-e2e-platform-overlay-launch` CodeBuild project.
- **Optional export:** PR #74's renderer/report concept is adapted to consume
  canonical AOVL controls and `sample_uid` joins. The CPU-only
  `wf_export_trajectory_report` returns a self-contained `FlyteDirectory`
  without re-running checkpoint inference.

### Remaining production rollout

1. Run local Python, Go, frontend, Playwright, and Terraform validation.
2. On the configured GPU EC2 host, import/serialize with FlyteKit 1.14.9 and run
   model/GPU tests.
3. Apply the reviewed Platform Terraform plan, build images, and register
   digest-pinned workflows.
4. Launch `EPISODES=1`, validate manifest/pointer/gate ordering and Console
   playback, then estimate the full-run duration and cost.
5. Launch `EPISODES=0` only after the smoke succeeds; monitor the Flyte
   execution to the ready gate.
6. Build the digest-pinned Console API image, then use the successful
   workflow's `manifest_key` and `manifest_sha256` outputs to run the guarded
   reasoning-materialization Job to completion.
7. Roll out Console API/web images and verify desktop/mobile layout, browser
   console, canvas pixels, model switching, and privacy-filtered geo output.
8. Validate straight, left-turn, and right-turn map placement before treating
   geographic predictions as trustworthy.
9. Keep exact geography disabled until authenticated edge role propagation is
   implemented and verified.

### Original phase breakdown

**Phase 0 — console-only linkage (no Flyte):**
- Fix `MLflowModelVersion` to keep `source`; add `model-versions/search` proxy; compute + store `model_artifact_id = sha256(best.pt)`; write `MODELVER#`/`MODEL#` seeds; surface `val_fraction` from `config.yaml`.
- Frontend model-picker skeleton + BEV two-layer canvas against a mocked binary overlay + the TS integrator with PY↔TS golden tests (raw default + display-limited toggle) + yaw-sign verification harness.

**Phase 1 — v2.1 FULL RE-PACK (Flyte data-gen → console):**
- Full `data_processing` re-pack to **`v2.1`** (NOT a decode-free in-place backfill): L2D loader `pose_current` + `gps_future` (from RAW heading + lat/lon); per-episode path artifact; **rig projection generated here** (per-rig constant confirmed against PR#74); **geo stats emitted here** (`geo/*`); introduce `sample_uid` + `legacy_sample_id` migration manifest; Go npy-decode for `has_gps`/`gps_now`/`heading_now`; rebuild `IDX#` at v2.1; repoint `resolveVersion`/search to v2.1; Map view (with §9-ter privacy) + `GEO#` geo-stats page.
- **Reasoning-subsystem migration** (its own work item): re-key or map the shipped reasoning-label cache + `LBL#…/SCENE#{sampleID}` from `legacy_sample_id` to `sample_uid` (P1.7 blast radius).

**Phase 2 — canonical vector overlays (Flyte GPU + Dynamo + API):**
- Add `initial_noise=` to `FlowMatchingPlanner.forward` (P0.1) + `load_policy`/`predict_control`/`noise_from` helpers; coarse per-shard task computing the **canonical, split-free** overlay over ALL samples; write binary `overlay.bin.gz` to S3 + SHARD×MODEL pointer + `OVLSET#` status; new API endpoints (read-only; NO compute trigger — ops-only); BEV + camera + map multi-model overlay/toggle/compare with display-mode toggle. **No `gsi1`/`SCENELIST#`** (P1.6).

**Phase 3 — optional MP4 export (PR#74, implemented):**
- `Tools/trajectory_visualization` reads the canonical AOVL and matching v2.1
  shard, verifies both immutable publication manifests, emits **one MP4 per
  scene** plus thumbnail/metrics/manifest, and is wrapped by
  `wf_export_trajectory_report`. It deliberately does not reuse the PR's legacy
  checkpoint inference, `v0=0`, synthetic calibration, or eval-only loader.

**Risks:**
- **Yaw-sign mirror + map ENU** (§10, §9-bis) — the corrected ENU formula is necessary-not-sufficient; must be verified JOINTLY (straight + left/right turn) before any overlay or map path is trusted; blocks Phase 2 acceptance.
- **`sample_uid` migration blast radius** (P1.7) — the shipped reasoning-label cache + `LBL#` index are keyed by `s{si:08d}`; adopting `sample_uid` is a scoped migration, not free. Confirm adopt-now vs defer with the user.
- **Rig projection contract (resolved).** PR#74 `project_BEV_to_CameraView` consumes only a fixed `P = K[R|t]`; publication schema v2 therefore writes one content-addressed artifact per unique rig, binds it to each shard, and stores no per-sample projection. KITScenes v2.2 production receipts prove a dataset-wide singleton is incorrect.
- **Version-coordinate drift** — enforce one version (`v2.1`) across `SHARD#`/`IDX#`/search; v2.1 immutable + manifest-digest pinned.
- **Single warm GPU** — backfill is near-serial. Scope: **latest N model versions × all samples per shard** (canonical). Give a wall-clock estimate before triggering; ops launches it (not the UI).
- **PR#74 optional export boundary.** It emits one MP4 + manifest per invocation, but its `[right, forward]` integrator, `v0=0` placeholder, eval-only loader, and legacy checkpoint schema do not satisfy the canonical overlay contract. Reuse the MP4/manifest concept only after adapting it to AOVL controls and `sample_uid`; do not merge its inference path into Phase 2.
- **Default `val_fraction=0`** — many models have no eval set; UI must say "train-leaked (no held-out eval)" (a display filter, not a missing overlay).
- **TTL vs catalog** coherence (§6 rule) — deletion must purge S3 + catalog + flip status together.
- **GPS privacy** (§9-ter) — exact routes are personal data; enforce authz, coarsening, k-anonymity, endpoint fuzzing, attribution.

---

## 12. Design decisions & rejected alternatives

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Overlay scope | **Canonical per `(model_artifact_id, dataset_version, shard)`; ONE inference over ALL samples** | Per-`split`/`source` overlays | The old `PRED#…` key lacked `split`, so train→eval→search subsets of a shard OVERWROTE each other; and re-running inference per subset was redundant. train/eval/search are display-time filters over one body (P0.2). |
| Overlay body storage | **S3 binary `overlay.bin.gz` (sole body); Dynamo = pointer only** | Gzip-JSON payload in the Dynamo item | 400 KB is a hard item cap incl. attr names; gzip ratio is content/seed-count dependent (seed fans overflow), and the JSON-in-Dynamo item contradicted the frontend's `arrayBuffer→Float32Array, no JSON.parse`. Binary + pointer resolves both (P0.3). |
| Overlay representation | **Raw `(64,2) accel/curvature + v0`** | Integrated XY blob | XY bakes the integrator + yaw-sign into the artifact → any fix forces full GPU recompute; raw keeps fixes client-side and reuses the reference integrator (adas #2). |
| Overlay identity key | **`model_artifact_id = sha256(best.pt)`** (+`MODELVER#` registry coordinate; `run_id` provenance attr) | `run_id`; bare `VER#{version}` | `run_id` is a lineage id, not content-addressable; version *numbers* are immutable but a bare `VER#` collides across registered models. sha256 dedupes identical checkpoints + detects content change (P1.8). |
| P1 "models for scene" | **`SHARD#{ds}#{ver}#{shard}` / `MODEL#{artifact_id}` base-table query** | scene×model `gsi1` inverse index + `SCENELIST#` fanout | Canonical per-shard overlays collapse model↔scene to model↔SHARD (~500 items vs 500k edges). **SUPERSEDES the round-2 gsi1 decision; gsi1 for future inverse lookups is deferred to the user** (P1.6). |
| Determinism / noise | **`initial_noise=` kwarg + per-sample `hash64(model_artifact_id, ds_manifest, sample_uid, base_seed)`** | Rely on per-batch `generator=` ("zero model change") | `randn(B,dim,gen)` draws in batch order → not batch-invariant on recompute/retry. Small model edit is required; corrects the earlier "zero model change" claim (P0.1). |
| Sample identity | **`sample_uid = hash(dataset, episode_id, frame_idx|ts_ns)`** (+`legacy_sample_id` + migration manifest) | Fragile global `s{si:08d}` enumeration | A 1-item repack shift re-points every downstream scene; content-addressed uid is stable. FLAG: reasoning-cache + `LBL#` re-key is its own work item (P1.7). |
| Camera projection | **v2.1-repack RIG artifact, content-addressed at `rig/{sha256}.json` and bound from each shard** | One dataset-wide rig; per-sample `proj.f32`; baked per-model frames | PR#74 confirms projection uses fixed `P = K[R|t]`; ego pose and model weights do not enter it. KITScenes has multiple calibrated rigs, so the shard selects the correct per-rig constant without duplicating it per sample or model (P1.12, cost-frontend #2). |
| Render/clamp contract | **Raw prediction default + explicit "display-limited" toggle; PY↔TS golden integrator** | "Clamp parity" (clamp pred to match GT) | The floor/clamp live in GT *derivation*, not the integrator; clamping predictions hides model error rather than treating them identically (P1.9). |
| Reproducibility | **Two-tier (same-env identical / cross-env numerically close)** | "Byte-identical re-runs" | Cross-env cuDNN/driver differences preclude bitwise guarantees; record full env + digests instead (P1.10). |
| Cache identity | **Narrow `hash(model sha, ds_manifest, preprocess, infer-code, sampler, steps, noise_policy, binary_schema)`** | Repo-wide git SHA | Git SHA invalidates the GPU cache on unrelated (e.g. Next.js) changes (P1.11). |
| S3 prefixing | **Simple human-readable prefix; shard later only if 503s appear** | `b={hash(shard)%16}` bucketing | S3 auto-scales to 3,500 PUT/5,500 GET per prefix and the write is near-serial on one GPU (P1.13). |
| Geo stats production | **Emitted during the v2.1 repack (`geo/*` in S3); Dynamo = summary + pointer** | Aggregate via DynamoDB `IDX#` scans; full point set inline | Full data breaches 400 KB and re-inflating gzip `IDX#` blobs for one lat/lon is wasteful; the repack already scans GPS (P1.14, infra #11). |
| GPS precision + pose | **float64 lat/lon + explicit `pose_current` heading from RAW vehicle heading** | float32; heading derived from GPS deltas | float32 → ~1–2 m jitter on the map; GPS-delta heading is unstable at low speed/jitter (P0.4, adas #8). |
| Map ENU formula | **`east=x·sinψ−y·cosψ`, `north=x·cosψ+y·sinψ`** | `east=x·sinψ+y·cosψ`, `north=x·cosψ−y·sinψ` | The rejected form treats `y` as RIGHT; at ψ=0 a left point must move WEST. Necessary-not-sufficient: compose with yaw-sign + heading source, validate jointly (P0.4). |
| GPS for map | **Per-episode full-path artifact** | Per-sample 6.4 s future window | A future window can't draw a driven route (flyte #8). |
| GPS backfill | **Full re-pack to v2.1 (USER DECISION 3)** | Decode-free in-place into v2.0 | User chose a clean versioned re-pack; the whole doc is normalized to v2.1 (S3, Dynamo, manifest, diagram, phasing) — no v2.0 anywhere (P0.5). |
| Flyte fan-out | **Coarse per-shard task, load ckpt once** | `map_task` over tiny units ("actor pool") | Flyte subtasks aren't a warm Ray actor pool; fine fan-out re-downloads 509 MiB per subtask, and there's one warm GPU anyway (flyte #2, #3). |
| Baked frames | **Optional MP4 export only** | Baked as default deliverable | Vectors + rig projection give multi-model compare at ~3–4 orders less storage; if exported, MP4 ≫ JPEG-sequence (cost-frontend #1,4,5). |

---

## 13. Resolved Decisions and Remaining Gates

The implementation go-ahead resolved the four original design questions:

1. `initial_noise=` was added and is covered by batch-order/batch-size
   reproducibility tests.
2. The corrected ENU transform is implemented, but straight/left/right
   real-data validation remains a production acceptance gate.
3. P1 uses the base `SHARD#...` query. No overlay `gsi1` or `SCENELIST#`
   fanout was added.
4. v2.1 adopts stable `sample_uid`; embedded reasoning labels, stats, and scene
   indexes are versioned and joined by that identity. Legacy per-sample cache
   assumptions were removed from the Console path.

Two operational decisions remain intentionally conservative:

- A canonical `(model artifact, dataset, version, overlay schema)` coordinate
  is write-once. A different seed set, runtime contract, or inference-step
  count at that same coordinate is a conflict, not an overwrite. Publish a new
  dataset/schema coordinate when intentionally changing the canonical result.
- Exact geography stays off until authentication verifies viewers and supplies
  the API's trusted principal context; the current unauthenticated distribution
  cannot safely authorize raw routes.
