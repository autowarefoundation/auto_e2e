# Design: Distributed BEV Multi-stage Training

Status: IMPLEMENTATION IN PROGRESS (2026-08-08)

This document defines the production experiment that combines the Reactive
BEV auxiliary objectives with the validated Ray/KubeRay distributed runtime.
It supersedes neither
[BEVSegmentationAuxiliaryLoss.md](BEVSegmentationAuxiliaryLoss.md) nor
[distributed_imitation_training.md](distributed_imitation_training.md).
Those documents define the model objective and the generic DDP platform. This
document defines their integration and the exact nuPlan -> L2D -> KITScenes
program.

## 1. Current facts

The following statements were verified before implementation:

1. The BEV branch contains a working single-GPU Reactive-only program:
   nuPlan full multi-task training, L2D continuation without BEV loss, a
   four-cell retention report, and optimizer-free KITScenes evaluation.
2. The distributed worktree contains a validated synthetic Ray Train DDP path:
   Flyte -> RayJob -> Kueue -> KubeRay -> one GPU per worker. Four-rank and
   eight-rank cluster smokes completed with parameter consensus and durable S3
   checkpoints.
3. The distributed worktree implementation is not committed to its named
   branch. It is imported into this integration worktree as a reviewable
   snapshot.
4. The distributed path has not yet trained the Reactive model on packed
   nuPlan or L2D samples.
5. The Platform cluster has an eight-instance `g6e.4xlarge` capacity
   reservation, but currently has no running GPU nodes. Karpenter starts the
   nodes only after Kueue admits a RayJob.
6. No nuPlan raw or packed dataset is currently present in the Platform
   dataset bucket.
7. The existing L2D `v2.0` shards predate the Reactive Route and XY target
   contract and cannot be used for this program.
8. The existing development KITScenes manifest contains two samples. It is a
   pipeline smoke manifest, not a benchmark result.

## 2. Locked experiment

The experiment is:

```text
Stage A: nuPlan
  Camera + Map + Route
    -> Reactive model
    -> Trajectory XY imitation
    -> BEV semantic segmentation
    -> Route reconstruction

Stage B: L2D
  Camera + OSM Map + waypoint Route
    -> Stage A weights
    -> new optimizer and scheduler
    -> Trajectory XY imitation
    -> Route reconstruction
    -> no BEV head execution and no BEV head update

Benchmark: KITScenes
  frozen Stage A and Stage B checkpoints
    -> fixed manifest
    -> optimizer-free trajectory inference
    -> ADE/FDE at the declared horizons
```

World Model and Reasoning remain disabled. The Reactive GRU planner is the
only planner under test. The common BEV geometry is `450 x 300` at
`0.4 m/px`.

## 3. Data readiness gates

### 3.1 nuPlan

nuPlan download requires the operator to accept the dataset Terms of Use.
The workflow MUST receive already authorized, immutable assets:

- nuPlan DB files;
- map root and exact map version;
- sensor blob root containing the eight required cameras;
- dataset source revision.

The workflow MUST NOT download nuPlan from an unauthenticated public endpoint
or accept mutable "latest" paths. Packing MUST reject missing cameras,
incomplete trajectory horizons, incomplete BEV supervision, or a rejection
fraction above the declared threshold.

### 3.2 L2D

L2D MUST be repacked with the current Reactive schema. The pack input MUST
include a pinned OSM snapshot with source digest, source revision, attribution,
and adapter version. Runtime OSM network access is forbidden.

The Route target MUST come from the provided L2D waypoints after map matching.
The future executed trajectory MUST NOT be used to construct the Route input
or Route reconstruction target.

### 3.3 KITScenes

The two-sample development manifest is used only for an end-to-end smoke. A
benchmark run requires:

- a frozen KITScenes v3.3 corpus digest;
- a frozen sample UID manifest;
- no overlap with training or checkpoint selection;
- the same manifest for Stage A, Stage B, and all external baselines.

No benchmark number may be published from the development manifest.

## 4. DDP data contract

Each packed manifest MUST list:

- immutable tar shard names and SHA-256 digests;
- exact sample count per tar shard;
- dataset and schema versions;
- camera count and geometry contract;
- Reactive target coverage flags;
- source revision and partition identity.

All ranks independently derive the same deterministic shard assignment. Tar
shards are assigned with longest-processing-time balancing by sample count,
with rank ID as the deterministic tie-breaker. A rank owns complete tar files;
WebDataset only splits those files among that rank's DataLoader workers.

The run fails before model construction when:

- the number of non-empty tar shards is smaller than the DDP world size;
- a shard digest or manifest contract differs;
- one rank receives no train data;
- the requested dataset does not match the training stage.

Each rank stages only its assigned tar files from S3 into node-local storage.
Training workers never depend on the Ray head pod filesystem.

## 5. Equal-step training

DDP ranks MUST execute the same number of backward collectives and optimizer
steps. The runner therefore uses a fixed optimizer-step count per epoch:

```text
steps_per_epoch =
  ceil(estimated_train_samples / global_effective_batch)
```

The value is computed once from the immutable manifests unless explicitly
overridden for a smoke. Rank-local loaders restart when exhausted. The report
records loader restarts, consumed samples, assigned samples, and optimizer
steps per rank. Unequal rank evidence fails the run.

For the initial eight-rank program:

| Setting | Value |
| --- | ---: |
| Workers | 8 |
| GPUs | 8 |
| GPUs per worker | 1 |
| Per-rank micro-batch | 1 |
| Gradient accumulation | 1 |
| Global effective batch | 8 |
| Precision | bf16 on L40S, with fp32 smoke fallback |
| DDP unused-parameter detection | enabled until branch gradients are audited |

The global batch differs from the single-GPU baseline. It is recorded as an
explicit experiment parameter. Learning rate is not silently scaled.

The production Stage A task preserves the existing pretrained SwinV2
initialization. Its public `timm` weights are cached while the immutable
training image is built. Workers MUST NOT download initialization weights at
runtime. The synthetic canary sets `is_pretrained=false`.

Gradient accumulation, when enabled, uses `DDP.no_sync()` for non-final
micro-steps. A finite-loss vote occurs before backward, and a finite-gradient
vote occurs before optimizer update so one bad rank cannot leave peers blocked
in a collective.

## 6. Validation and checkpoint selection

Validation uses the unwrapped model so ranks can consume different numbers of
validation batches without triggering DDP buffer broadcasts. Each rank emits
trajectory error sums and counts. Global ADE/FDE are computed with
`all_reduce`.

Rank 0 writes the checkpoint. Every rank calls `ray.train.report` with the same
global metrics. A valid checkpoint contains:

- model, optimizer, and scheduler states;
- completed epoch and optimizer-step count;
- model/config/dataset digests;
- world size, global batch, precision, and rank assignment digest;
- Stage A parent digest for Stage B;
- validation metrics used for selection.

Stage B loads only Stage A model weights and creates a new optimizer and
scheduler. It MUST reject Stage A optimizer state restoration.

Ray worker-group recovery may resume the same stage checkpoint. It MUST reject
a changed dataset, world size, objective, geometry, or shard assignment.

## 7. Flyte program

The production workflow is:

```text
pack_nuplan_reactive_dataset
  -> validate_nuplan_manifest
  -> train_reactive_stage_a_ray_8

pack_l2d_reactive_dataset
  -> validate_l2d_manifest
  -> train_reactive_stage_b_ray_8(stage_a_checkpoint)

stage_a_checkpoint + stage_b_checkpoint
  -> retention_matrix
  -> kitscenes_development_smoke
  -> fixed_kitscenes_benchmark
```

Flyte owns cross-stage dependencies. Each training stage creates one
ephemeral RayJob. Kueue admits the full worker group before execution, and
KubeRay deletes the transient cluster after completion.

Account IDs, bucket names, reservation IDs, placement groups, and image URIs
MUST be supplied by environment or deployment configuration. They MUST NOT be
embedded in model or workflow source.

## 8. Execution gates

The program advances only through these gates:

1. **Unit gate**: shard assignment, fixed-step cycling, global reductions,
   Stage B weight-only loading, and checkpoint metadata pass locally.
2. **Synthetic GPU gate**: two-rank Reactive Stage A and Stage B complete on
   real GPUs with parameter consensus.
3. **nuPlan mini-overfit gate**: a small real nuPlan subset completes at least
   two epochs; trajectory, BEV, Route, and total losses are finite, and total
   loss decreases from its initial value.
4. **L2D mini gate**: Stage B loads the exact Stage A checkpoint, leaves BEV
   parameters byte-identical, and produces finite trajectory and Route losses.
5. **Eight-rank gate**: all ranks use distinct hosts, equal optimizer steps,
   matching model state, and a durable checkpoint.
6. **Development benchmark gate**: the two-sample KITScenes manifest verifies
   the inference and artifact path only.
7. **Benchmark gate**: the frozen full manifest runs without optimizer access
   and emits checkpoint, dataset, and prediction digests.

Failure at a gate stops the later and more expensive stages.

## 9. Immediate blocker

Implementation and synthetic cluster verification can proceed without nuPlan.
The real-data gates cannot start until an authorized nuPlan mini asset root or
immutable S3 prefix is supplied. Loss reduction on real nuPlan data has not
yet been measured.
