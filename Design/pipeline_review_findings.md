I've verified the key line numbers and code facts against the actual source. All the CONFIRMED findings check out. Here is the consolidated report.

---

# AutoE2E pipeline review — consolidated findings

## Dedup summary
No true duplicates. Several findings share a file but are distinct root causes with distinct fixes:
- `prompt_builder.py` has 3 independent bugs (lines 100, 103, 139) — keep separate.
- `label_cache.py` has 2 (lines 34, 82) — keep separate.
- `pre_extracted.py` has 2 (lines 74, 118) — keep separate.
- **Related pair (fix together):** `label_cache.py:34` (teacher-model not in key) and `parallel_label.py:79` (dataset-version/positional key) are two facets of one class — *the reasoning-label cache key under-specifies its inputs, so stale labels are served silently.* Distinct triggers, but the right fix is one hardened `cache_prefix`.

No CONFIRMED finding is a false positive. The single UNCERTAIN finding is a real latent weakness but its stated OOM-kill scenario is overstated (details below).

---

## CONFIRMED (ranked by severity)

### HIGH

**1. fp16 downcast before normalization defeats the fp32 perspective-divide guard → nan grads freeze AMP training**
`Model/model_components/view_fusion/projection.py:179`
The fp32 divide result is cast back to fp16 via `.to(projected.dtype)` *before* the `/ wh` normalize (line 181). The value cast is an unbounded pixel coord `numerator/depth`; for a point near a camera focal plane (depth just above `_DEPTH_EPS`, so `valid_depth` stays True) the fp32 quotient exceeds fp16 max (65504) → `inf` → `uv_norm=inf` flows unmasked into `grid_sample` → nan grads → GradScaler skips every `optimizer.step()`. Only bites under `float16` autocast (`amp=True`); bf16 unaffected.
**Fix:** keep `uv_norm` in fp32 (drop the `.to(...)`, do `/ wh` in fp32), or clamp the quotient to an fp16-safe range before cast, e.g. `uv = (projected[...,:2].float() / depth_safe.float()).clamp(-3.0e4, 3.0e4).to(projected.dtype)`. Note: "normalize by wh then cast" alone is insufficient — a normalized coord can still exceed fp16 max.

**2. Partial `reasoning.json` coverage crashes collation or silently drops supervision**
`Model/data_parsing/pre_extracted.py:118`
`reasoning__*` keys are added only when a sample carries `reasoning.json`, but the packer deliberately supports mixed shards (`workflows.py:410-411` "packed without reasoning.json … imitation-only"). `make_pre_extracted_loader` uses `default_collate` with no `collate_fn`: a batch whose first sample is labeled + a later one unlabeled → `KeyError('reasoning__source_weight')`; first sample unlabeled → all reasoning keys silently dropped for the whole batch. `shuffle=1000` makes it nondeterministic; the `batch_size=1` training guard (`workflows.py:763-770`) can't detect partial coverage.
**Fix:** add a `collate_fn` that fills missing-key samples with fully-masked (zero `source_weight`/IGNORE) reasoning tensors before `default_collate`; or enforce all-or-nothing at pack time (raise when `labels_by_id` non-empty but not every packed `sample_id` matches).

**3. Out-of-vocab multi-label values silently produce a WEIGHTED all-zero (false-negative) target row**
`Model/data_processing/reasoning_label_generation/prompt_builder.py:100`
Single-label groups coerce OOV → `None` → IGNORE_INDEX (masked, safe). Multi-label groups (`hazard_event`, `cause`) coerce OOV → empty list → `targets` emits an all-zero `[C]` row (even the abstain class `no_hazard`/`unknown_hazard` = 0), fed to BCE/ASL with nonzero `source_weight = 0.5×confidence`. This actively trains "no hazard AND every hazard class false" on frames where a hazard was truly present. Silent: not abstained, passes `validate_record`, cached, reported "computed." Directly violates the R9 "never silently turned into all-zero labels" guarantee. Reachable on the real `openai_compatible` path (VLMs emit near-miss labels like `pedestrian` vs vocab `pedestrian_crossing`).
**Fix:** in `_coerce_horizon` MULTI branch, backfill the group's abstain label when the coerced list is empty, mirroring the single-label None→IGNORE masking so multi-label groups are never an all-zero row.

**4. Teacher model (`COSMOS_TEACHER_MODEL`) is absent from the cache key — only the provider string is**
`Model/data_processing/reasoning_label_generation/label_cache.py:34`
`cache_prefix` keys on `teacher=<provider>` = bare `"openai_compatible"`; the actual model is resolved separately into `teacher_kwargs['model']` and never reaches `LabelCache`. Repointing `COSMOS_TEACHER_MODEL` to a different model with the same `prompt_version` yields a byte-identical prefix → every sample is a stale hit → the new model is never called and the artifact ships the old model's labels. `schema.teacher_model` being a distinct provenance field proves the model is a real discriminator the key drops. Caching is ON by default. (The reviewer's secondary claim that `meta.json` misrepresents the model is slightly off — `meta.json` records only `teacher`/`prompt_version`; the cached records' `teacher_model` is accurate for the *old* labels. Primary stale-cache defect is real.)
**Fix:** thread the resolved model into `LabelCache` and fold into `cache_prefix`, e.g. `teacher={teacher}:{model}` or a `/model=<model>` component. *Fix together with #8.*

### MEDIUM

**5. NVIDIA map branch is fed ImageNet-normalized black (nonzero constant), not zeros; loader's zero-fallback is dead for NVIDIA**
`Model/data_parsing/pre_extracted.py:74`
`NvidiaAVDataset` sets `map_tile = torch.zeros_like(frame)` (non-None), so `pack_sample` writes a black `map.jpg` into every NVIDIA sample. The loader takes the `"map.jpg" in sample` branch, decodes + `Normalize` → per-channel constant `[-2.118,-2.036,-1.804]`, not zeros. The intended zero-fallback (lines 76-78) is dead for NVIDIA, `has_map` is wrongly True, and `test_missing_map_yields_zeros` (asserts abs max 0) is contradicted. In merged L2D+NVIDIA training a constant contaminates the shared map encoder. (Severity qualifier: the "degrades encoder" impact is somewhat speculative — a constant through residual-gated fusion behaves like a learned bias — but every factual claim is confirmed.)
**Fix:** have `NvidiaAVDataset` omit `map_tile` (don't set the key / set None) so `pack_sample`'s `sample.get("map_tile")` is None, no `map.jpg` is written, loader hits the zero-fallback, and `has_map` stays False.

**6. Missing/non-numeric confidence defaults to 0.0, zeroing `source_weight` and dropping the whole horizon's supervision**
`Model/data_processing/reasoning_label_generation/prompt_builder.py:103`
`_coerce_horizon` defaults a missing or non-numeric (`"high"`) confidence to 0.0. `source_weight = 0.5×confidence = 0` → `_weighted_mean` normalizes it away → the fully-valid horizon contributes nothing to any reasoning-loss term, yet the record passes `validate_record`, is not abstained, and is cached as good. Conflates "confidence 0" with "confidence unspecified." Triggers exactly on the real `openai_compatible` path (mock teacher always emits numeric confidence, so tests miss it); the prompt example even shows `"confidence": 0.0`.
**Fix:** default missing/non-numeric confidence to a non-zero neutral (e.g. 1.0) so a validated label is weighted by provenance; alternatively treat missing/non-numeric confidence as a parse failure and abstain.

**7. `get()` increments both hits and misses on a deserialization failure; broad `except` masks systematic decode bugs**
`Model/data_processing/reasoning_label_generation/label_cache.py:82`
`self.hits += 1` (line 82) runs before `record_from_json(payload)` (line 83). If deserialize raises (missing required field after a schema change, missing `horizon_sec`, non-dict payload from a partial write), the `except` bumps `self.misses += 1` — one `get()` increments both, breaking `hits+misses == num_calls`. The broad `except` converts genuine decode bugs into silent misses; because the key omits `schema_version`, a schema change under an unchanged `prompt_version` keeps the same prefix and every old entry silently fails to deserialize. (Qualifiers: metrics are currently read only by the unit test, not any live reporting path; and `get_or_compute` rewrites via `put()`, so it's a one-time re-bill after a schema change, not a permanent 100% miss.)
**Fix:** deserialize first, then count the hit — `record = record_from_json(payload); self.hits += 1; return record` — so a decode failure is a single miss, not a double-count.

**8. `sample_id` is a bare positional index and `dataset_version` is absent from the key → re-versioning collides with stale labels**
`Model/data_processing/reasoning_label_generation/parallel_label.py:79`
The per-sample key is `f"s{si:08d}"` (enumeration position) with no content hash, no episode/frame identity, no `dataset_version`; the L2D loader is opened with no `revision` pin. If `yaak-ai/L2D` is re-versioned upstream (frames inserted/reordered) under the same `repo_id`/`teacher`/`prompt_version`, position `si` maps to different pixels but the key is identical → cache hits serve OLD labels, JOINed into freshly-packed NEW frames — silent mislabeling, no invalidation. (One inaccuracy in the finding: the "non-prefix episode subset" trigger is *not* reachable — `episodes` is an int → `range(episodes)` is always a prefix and enumeration-stable. The upstream-re-versioning branch is the genuine bug.)
**Fix:** pin the source and put dataset content identity in the key — pass `revision=` to `LeRobotDataset` and fold the resolved HF commit/hash into `cache_prefix` (`dataset={name}@{revision}`). *Fix together with #4.*

### LOW

**9. Exact multiple of `samples_per_shard` produces a spurious empty trailing `.tar` and over-counts `manifest['shards']` by 1**
`Platform/pipelines/workflows.py:418`
`open_new_shard()` fires eagerly whenever `sample_count % samples_per_shard == 0`, including right after the final sample of a full shard. A dataset packing to an exact multiple (1000, 2000, …) opens `train-00000N.tar`, writes nothing, and the post-loop `close()` flushes a 0-member tar; `shard_idx` is already incremented so `manifest['shards']` is off by one and an empty object ships to S3. Loader survives (`empty_check=False`) but wastes a worker and the manifest count is wrong.
**Fix:** open shards lazily — rotate at the top of each iteration before writing (`if sample_count % samples_per_shard == 0: open_new_shard()` then add+increment); or after the loop, if the current tar has zero members, close+delete it and decrement `shard_idx` before the manifest write.

**10. `parse_clip_response` accepts >5 horizons and blindly positional-assigns `horizon_sec`, ignoring the model's stated order; docstring says "exactly"**
`Model/data_processing/reasoning_label_generation/prompt_builder.py:139`
Guard is `len(horizons_raw) < NUM_HORIZONS`, so >5 entries are accepted and truncated to the first 5; the docstring (line 127) promises None unless "exactly" 5. `_coerce_horizon` stamps `horizon_sec` positionally and never reads the entry's own `horizon_sec` (which the prompt asks the model to emit). If the model returns horizons reversed or with an extra leading entry, each label is stamped with the wrong second, and `validate_record`'s ordering check is tautological (seconds were force-set) so it can't catch it.
**Note:** severity is "low" only because it needs an LLM ordering failure to trigger; when it does trigger the mislabeling is silent and undetectable (e.g. "now" horizon supervised with the +4s prediction), so treat as a low-with-a-sharp-edge — worth fixing alongside the other teacher-parse hardening (#3, #6).
**Fix:** change the guard to `!= NUM_HORIZONS` and, per entry, require `entry.get("horizon_sec") == sec` (return None on mismatch) instead of blindly positional-assigning.

**11. Empty history window raises `IndexError`, violating the documented `ValueError` contract**
`Model/data_processing/reasoning_label_generation/clip_builder.py:55`
`build_temporal_front_clip` guards the future-frame count but never checks `history_frames` has ≥1 time-step. A `[0,V,3,H,W]` tensor passes the `ndim==5` and future-length checks, then `history_frames[-1, FRONT_VIEW_INDEX]` throws torch `IndexError` where the docstring promises `ValueError`. Frame indexing/ordering/shape checks are all correct — no wrong frames are ever emitted. Only reachable from tests (production uses `get_front_clip`/`load_front_clip`; `wm_num_frames >= 1` keeps history non-empty).
**Fix:** after the ndim check, add `if history_frames.shape[0] < 1: raise ValueError(f"need >=1 history frame for the current horizon, got {history_frames.shape[0]}")`.

---

## UNCERTAIN

**U1. `pool.map` eager-submit + in-order drain buffers all completed pack results in the parent (unbounded by the 6-worker cap)**
`Platform/pipelines/workflows.py:401` — severity medium if real
**Verdict: real mechanism, overstated failure — do not treat as a reproducible OOM bug.** The code-level claim is correct: `Executor.map` with default `buffersize=None` submits all N tasks up front and drains strictly in input order, so any task finishing ahead of the pop frontier keeps its full members dict (~1MB of JPEG/npy bytes per WM sample) alive; worst-case parent buffering is O(n_samples), not bounded by the worker cap, and the worker-count comment (lines 384-389) genuinely omits this. **But** the parent (single-threaded `tarfile.addfile` to local ephemeral storage, ~ms/sample) drains ~60× faster than 6 workers produce (~0.6s/result at ~3.6s/sample decode). An ordinary straggler buffers only MB-scale (a 100s front stall ≈ ~140 results ≈ 140MB). Reaching the 30-32Gi limit needs one early worker to hang ~4-5 hours while the other 5 finish the entire job — a degenerate hang, not the "longer clip / cold read / GC pause" the finding posits. So: a legitimate latent robustness weakness with an incomplete safety comment, not a concretely reproducible OOM on the #33 full-episode run.
**Fix (defensive, cheap):** bound in-flight work while keeping in-order drain — `pool.map(pack_sample, idx_list, buffersize=4*pack_workers)` (supported on the 3.14 runtime here), or an equivalent sliding-window `submit()`/`as_completed()` that still consumes in sample order for the shard-boundary logic.

---

## False positives
None among the CONFIRMED set. Two findings carry factual over-reach in their *rationale* (already corrected inline above) but the core defect stands in each:
- #4 (`label_cache:34`): the "`meta.json` misrepresents the model" sub-claim is wrong; the stale-cache defect is real.
- #8 (`parallel_label:79`): the "non-prefix episode subset" trigger is unreachable (episodes is always a prefix); the upstream-re-versioning trigger is real.

## Suggested fix order
1. `projection.py:179` (#1) — silently freezes all AMP training; one-line-scope fix.
2. `pre_extracted.py:118` (#2) — nondeterministic crashes / lost supervision.
3. `prompt_builder.py:100/103/139` (#3, #6, #10) — batch these; all are teacher-response coercion hardening and share `_coerce_horizon`/`parse_clip_response`.
4. Cache-key hardening `label_cache.py:34` + `parallel_label.py:79` (#4, #8) — one hardened `cache_prefix`.
5. `pre_extracted.py:74` (#5), `label_cache.py:82` (#7).
6. `workflows.py:418` (#9), `clip_builder.py:55` (#11), and U1's defensive `buffersize` cap.