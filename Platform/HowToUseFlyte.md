# How to Use Flyte

The platform registers two workflows. Everything a model needs, from raw data
to registered and evaluated checkpoints, runs through them. The design is in
`Design/single_flyte_pipeline.md`.

| Workflow | Run it when | Reaches Hugging Face |
|---|---|---|
| `wf_acquire_raw_snapshot` | once per dataset revision, to copy upstream data into an S3 raw snapshot | yes, and nothing else does |
| `wf_autoe2e` | for every dataset build, training run, multi-stage step, or evaluation | no |

Both are launched with `Platform/scripts/launch_autoe2e.py` and a JSON run
file under `Platform/scripts/runs/` (gitignored, because it names account
buckets). The launcher prints the typed inputs without `--execute`, so a run
file can be checked before it starts anything.

## 1. Acquire raw data once

A run file for KITScenes train adopts the July Full Run downloads and labels
from S3 instead of the Hub:

```json
{
  "workflow": "wf_acquire_raw_snapshot",
  "inputs": {
    "dataset": "kitscenes",
    "revision": "6fde0034446669e2ed7235e4c7fe323cd23d599d",
    "snapshot_id": "kitscenes-train-20261021",
    "split": "train",
    "datasets_bucket": "<datasets bucket>",
    "recovery_manifest": {"uri": "s3://<checkpoints bucket>/recovery-manifests/24afc76f….json", "version_id": "…", "sha256": "…"},
    "adopt_labels": {"teacher": "openai_compatible", "teacher_model": "nvidia/Cosmos3-Nano", "prompt_version": "action_relevant_reasoning_v3_temporal_front256", "label_stride": 10}
  }
}
```

Without `recovery_manifest`, the workflow downloads the listed `units` (or
the whole SDK split) from the Hub. For nuPlan, pass
`nuplan_source_manifest` instead; the existing receipt-based acquisition runs
and its manifest is returned.

The output is a `ManifestRef` (URI, version id, SHA-256). That reference is
the `source_manifest` of every later `DatasetSpec`.

## 2. Run the pipeline

```json
{
  "workflow": "wf_autoe2e",
  "dataset": {
    "dataset": "nuplan",
    "source_manifest": {"uri": "…", "version_id": "…", "sha256": "…"},
    "pack_version": "<current_pack_version('nuplan')>",
    "image_size": 1344
  },
  "mix": [{"dataset": "nuscenes", "source_manifest": {"uri": "…", "version_id": "…", "sha256": "…"}, "pack_version": "<current_pack_version('nuscenes')>", "image_size": 1344}],
  "early_mix": [{"dataset": "simbev", "source_manifest": {"uri": "…", "version_id": "…", "sha256": "…"}, "pack_version": "<current_pack_version('simbev')>", "image_size": 1344}],
  "training": {"stage": 1, "gpu_profile": "ray_1", "steps": 8, "checkpoint_interval_steps": 4},
  "evaluation": ["geo_val"],
  "runtime": {"datasets_bucket": "…", "checkpoints_bucket": "…", "tracking_url": "https://<mlflow>"}
}
```

`training` takes the `TrainingSpec` fields of
`Platform/pipelines/autoe2e_training.py` (`stage`, `gpu_profile`, `steps`,
`learning_rate`, `per_rank_batch_size`, `checkpoint_interval_steps`, `seed`).
`mix` lists the BEV datasets trained with `dataset`, `early_mix` those used in
the first half of stage 1 only.

`pack_version` must equal `current_pack_version(dataset)` from
`Platform/pipelines/autoe2e_adapters.py`; it changes whenever a pack contract
changes, so stale data is never reused.

What the run does, and what it skips:

| Step | Skipped when |
|---|---|
| plan and resolve the dataset | — |
| label units, pack, publish, commit | the same `DatasetSpec` is already published |
| label one unit | its label set is already committed |
| train on the profile's queue | `training.steps` is 0 |
| register one LoggedModel per distinct checkpoint | the checkpoint is already registered |
| evaluate each target | — |
| verify lineage | — |

GPU profiles: `ray_1` (one validation GPU, smoke only) and `ray_8` (one
8-GPU p5en node). The profile is the only way to pick a queue. Every
training run also starts a one-GPU task that validates each checkpoint as it
lands; the best and the final checkpoint are registered.

## 3. Multi-stage training

Set `parent` to the previous stage's model version, for example
`{"model_version": "auto-e2e-driving-policy/68"}`, and pick the next
`training.stage`. A bare checkpoint (`{"checkpoint_uri": "s3://…"}`) is
registered as an imported model first, so every stage has a model page.

## 4. Evaluate an existing model

Set `training.steps` to 0, `parent` to the model version, and list the
evaluation targets. The metrics attach to that model; no new version is made.

## 5. When something fails

| Message | Meaning |
|---|---|
| `pack_version … is not this code's …` | regenerate the run file's `pack_version` |
| `label set … is being produced by execution …` | another execution holds that unit's claim; wait, or delete the claim object if that execution is dead |
| `… changed` / `… differs` | a stored artifact no longer matches its recorded digest; investigate before rerunning, nothing is rebuilt silently |
| `expected one active auto-e2e-p5en-capacity-block` | buy or wait for a Capacity Block (`Platform/scripts/p5en_capacity_block.py`) |
