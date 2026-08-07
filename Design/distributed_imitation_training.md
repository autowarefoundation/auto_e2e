# Design: Ray-based Distributed Imitation Training

Status: DECIDED; PLATFORM PATH VALIDATED, MODEL INTEGRATION PENDING (2026-08-08)

Scope: reduce AutoE2E KITScenes imitation-training wall time by moving
`train_il` from one GPU to Ray-managed synchronous data-parallel training. Data
preparation remains distributed as defined in
[pipeline_parallelization_design.md](pipeline_parallelization_design.md).

This is a follow-up milestone. The earlier pipeline design explicitly kept DDP
out of scope while the dataset path was being stabilized; it is not evidence
that single-GPU training should remain the target architecture.

## 1. Decision

Use Ray Train `TorchTrainer` to run PyTorch `DistributedDataParallel` (DDP)
with a fixed world size selected from a reviewed topology allowlist. Flyte
remains the outer workflow orchestrator. Its Ray plugin creates one ephemeral
KubeRay `RayJob` per training task, and Kueue admits the complete Ray cluster.

The first parity target and the production capacity ceiling are:

| Setting | Value |
|---|---|
| Dataset and objective | KITScenes, existing offline imitation objectives |
| Parity topology | 4 `g6e.4xlarge` nodes, 4 GPUs total |
| Hard scale ceiling (v1) | 8 `g6e.4xlarge` nodes, 8 GPUs total |
| Placement | One Availability Zone and one EC2 cluster placement group |
| Capacity | Placement-group-backed targeted ODCR; 4-node and 8-node batch allocation both validated |
| GPU per node | 1 NVIDIA L40S 48 GiB |
| Processes | 1 DDP rank per node and GPU |
| Backend | NCCL for tensors, Gloo control group for metadata |
| Ray cluster | 1 CPU head pod plus 2, 4, or 8 one-GPU worker pods |
| Per-rank micro-batch | 1 |
| Gradient accumulation | 1 at world size 4 |
| Global effective batch | 4 for parity; 8 in the scale experiment |
| Precision | fp32 initially; bf16 is a separate measured optimization |
| Orchestration | Flyte Ray task -> KubeRay `RayJob` -> Ray Train `TorchTrainer` |
| Scheduling | Kueue admission of the transient head and fixed worker group |
| Recovery | Ray worker-group restart from S3 checkpoint; Flyte retry for driver/RayJob failure |

A two-node run is the integration canary. A single `g6e.12xlarge` with four
GPUs is a useful fallback benchmark, but it does not replace the four-node
acceptance run because the requested capability is multi-node training. The
four-node run is the optimizer-equivalent quality gate. The eight-node run is
a separate large-batch experiment and becomes a production candidate only
after quality and scaling gates pass.

Four nodes are the required multi-node acceptance topology. Eight nodes are a
hard operational ceiling, not a launch prerequisite: if AWS cannot allocate
eight matching instances together in the selected placement group, record the
capacity error and defer the eight-node experiment without blocking four-node
acceptance. World sizes above eight require a new design review.

Do not use Ray or KubeRay autoscaling in the first version. Each registered
Flyte task has a fixed Ray worker replica count, and `TorchTrainer.num_workers`
must match it for the entire run. Ray is adopted for its distributed runtime,
checkpoint-aware worker recovery, and future rollout/Tune expansion, not to
change the DDP optimization algorithm.

Ray Train does not itself make this model tensor-, pipeline-, or
optimizer-state-parallel. Adding FSDP or DeepSpeed remains a separate model
architecture decision if the model stops fitting on one GPU.

The platform decision is no longer provisional. Section 17.6 records successful
four-node, checkpoint-resume, Flyte-to-Ray, and eight-node cluster smokes. This
does not yet approve distributed `train_il` as the production default: full
KITScenes data sharding, quality parity, performance, worker-failure, and OOM
gates remain open.

## 2. Why this is the right form of distributed imitation learning

The current workload is offline imitation learning, also described as behavior
cloning:

1. demonstrations are packed before training;
2. the model consumes independent training samples from immutable WebDataset
   shards;
3. trajectory, reasoning, JEPA, route, and rollout-aligned terms are
   differentiable supervised objectives;
4. there is no simulator or online actor collecting new experience during an
   optimizer step.

Consequently, the distributed algorithm does not need an imitation-specific
parameter server or actor-learner system. Every rank can train on a different
demonstration micro-batch, and DDP can average gradients before all ranks apply
the same optimizer update.

Online methods such as DAgger would require distributed rollout, oracle
labeling, and dataset mutation. That is a different system and is out of scope.

## 3. Current-state evidence

### 3.1 Training is one GPU

`Platform/pipelines/workflows.py::train_il` is a normal Flyte container task
requesting one GPU. It constructs one `AutoE2E`, one AdamW optimizer, and one
training loop in the task process.

The full KITScenes workflow defaults are:

- `batch_size=1`;
- `grad_accum_steps=4`;
- `num_workers=4`;
- reasoning and World Model enabled;
- fp32 because the current fp16 path overflows.

The measured model sizes from the current tree, constructed without pretrained
weights, are:

| Configuration | Parameters | Raw fp32 parameter bytes |
|---|---:|---:|
| Reactive baseline | 91,348,342 | 348.5 MiB |
| World Model + Reasoning | 131,654,383 total / 104,077,765 trainable | 502.2 MiB |

The full model and a micro-batch already fit on one 48 GiB L40S. The immediate
problem is throughput, not parameter capacity.

### 3.2 The loader is deliberately single-rank today

`Model/data_parsing/pre_extracted.py::make_pre_extracted_loader` passes
`nodesplitter=wds.single_node_only`. WebDataset 1.0.2 raises when this splitter
observes `WORLD_SIZE > 1`. Replacing it blindly with `split_by_node` is also
incorrect for this repository:

- each FlyteDirectory is normally one KITScenes scene;
- an individual scene directory may contain too few tar files for every rank;
- `MergedDatasetLoader` opens scene directories independently;
- rank-local scene counts and batch counts would be unequal.

DDP requires the same sequence of gradient collectives on every rank. Unequal
loader exhaustion would hang or would require DDP join semantics that leave
optimizer and checkpoint behavior harder to reason about.

The distributed design therefore assigns complete scene partitions to ranks
before child WebDataset loaders are opened. WebDataset continues to split tar
files only among DataLoader workers within that rank.

### 3.3 Platform path is connected

The platform prototype now connects the complete control path:

| Area | Validated state |
|---|---|
| Flyte | 1.16.7 with the `ray` backend plugin enabled and mapped |
| KubeRay | Operator and CRDs installed at 1.4.2 |
| Kueue | 0.18.1 admits `ray.io/rayjob` in `auto-e2e-development` |
| Queue capacity | 8 GPU, 40 CPU, and 160 GiB with separate CPU and GPU flavors |
| Karpenter | `gpu-training` ceiling of 8 `g6e.4xlarge` nodes / 128 vCPU |
| EKS Auto Mode | Dedicated NodeClass selects the ODCR and cluster placement group |
| Capacity | Targeted 4-node and 8-node ODCRs allocated as complete batches |
| Runtime | PyTorch 2.4.1, Ray 2.46.0, Flytekit/plugin 1.16.24 |
| Cleanup | Transient RayClusters deleted by `shutdownAfterJobFinishes` plus TTL |

A Flyte-generated RayJob has been admitted by Kueue, created one CPU head and
four one-GPU workers, returned a Flyte file output, and deleted its RayCluster.
The same infrastructure also completed a direct eight-worker RayJob. The
remaining work is model/data integration rather than proving that these
control planes can form a multi-node DDP cluster.

## 4. Goals and non-goals

### 4.1 Goals

1. Reduce full KITScenes wall time without changing the imitation objective.
2. Preserve global effective batch size, learning rate, split, and sampling
   policy for the four-node parity experiment.
3. Ensure every rank performs the same number of backward and optimizer steps.
4. Keep the existing immutable checkpoint, resume, validation, and model
   selection contracts while exposing checkpoints through Ray Train.
5. Make rank ownership of samples explicit and auditable.
6. Keep one-GPU training available as a fallback and parity baseline.
7. Fail the complete worker group when any rank has a non-finite value or loses
   data/collective synchronization.
8. Automatically restart the complete training worker group after a transient
   worker/GPU/node failure and resume from the latest durable checkpoint.
9. Keep Flyte as the owner of workflow dependencies, typed inputs/outputs,
   caching, and the final task retry boundary.
10. Allow one bounded checkpoint replay after a first CUDA OOM, then terminate
    if the same configuration and checkpoint lineage encounter OOM again.

### 4.2 Non-goals

- FSDP, tensor parallelism, or pipeline parallelism.
- Migrating preprocessing, Flyte DAG ownership, or model serving to Ray.
- A persistent shared RayCluster.
- Ray Tune in the first distributed-training release.
- Dynamic scale-up or scale-down during an epoch.
- Multi-dataset distributed training.
- Offline-RL distribution.
- Spot capacity in the initial production path.
- Changing objective weights, precision, optimizer, or learning-rate policy in
  the DDP parity run.
- Automatically changing batch size, precision, or model configuration after
  an out-of-memory failure.

## 5. Alternatives and Ray trade-offs

| Option | Decision | Reason |
|---|---|---|
| Ray Train + PyTorch DDP | Adopt | Keeps native synchronous DDP while adding a managed worker group, durable checkpoint reporting, and restart hooks |
| Flyte Ray plugin + KubeRay RayJob | Adopt | Preserves Flyte DAG ownership and creates an isolated, queue-managed RayCluster per training task |
| Kubeflow PyTorchJob | Defer for this path | Simpler for DDP alone, but provides less direct leverage for future Ray Tune, distributed rollout, and heterogeneous Ray actors |
| FSDP | Defer | Useful for parameter/optimizer memory pressure, which is not the current bottleneck; adds sharded state and checkpoint complexity |
| DeepSpeed | Do not add now | Duplicates DDP/FSDP capabilities and adds a new runtime/configuration surface |
| Horovod | Do not add | Adds a dependency without a material advantage over native NCCL DDP here |
| Asynchronous parameter server | Reject | Changes optimizer semantics and reproducibility for no demonstrated need |
| Independent one-GPU experiments | Keep for sweeps only | Improves experiment throughput, not time-to-one-trained-model |
| One four-GPU node | Benchmark/fallback | Simpler networking, but does not validate multi-node operation and requires a new instance shape |

### 5.1 Why Ray is adopted

Native PyTorch DDP is the training algorithm; Ray is not required to average
gradients. Ray is adopted as the runtime inside one Flyte task because it adds:

- complete worker-group restart with latest reported checkpoint handoff;
- a uniform API for fixed multi-node worker creation and rank context;
- an extension path for future distributed rollout actors and Ray Tune without
  moving workflow DAG ownership away from Flyte;
- one transient, isolated RayCluster per training attempt rather than a shared
  long-lived cluster.

The accepted costs are an additional KubeRay operator/CRD lifecycle, a CPU head
pod, a stricter Flyte/Ray/KubeRay compatibility matrix, and a longer debugging
chain across Flyte, Kueue, KubeRay, Ray, and DDP. Ray also does not make an
invalid checkpoint, a deterministic CUDA OOM, or incorrect data sharding
recoverable. Application code still owns checkpoint completeness and resume
validation.

Flyte and Ray overlap only if both are allowed to own the application DAG.
This design prevents that overlap: Flyte owns cross-task orchestration and the
outer retry, while Ray owns only the distributed execution and worker recovery
inside the training task. The recovery fault-injection gate in section 17.3 is
a condition of adoption; passing a successful DDP run alone is insufficient.

## 6. Target architecture

```text
Flyte workflow
  |
  +-- build_training_dataset_index (small manifest, CPU)
  |
  +-- train_il_ray_{2,4,8}
        |
        +-- Flyte Ray task plugin creates one transient RayJob
              |
              +-- Kueue holds RayJob suspended until all pod sets fit
                    |
                    +-- KubeRay creates one transient RayCluster
                          |
                          +-- head pod, CPU pool, no GPU
                          |
                          +-- N fixed worker pods admitted together
                                |
                                +-- one AZ and EC2 cluster placement group
                                      |
                                      +-- worker/rank 0, g6e.4xlarge, L40S
                                      +-- ...
                                      +-- worker/rank N-1, g6e.4xlarge, L40S

Ray head/driver:
  constructs TorchTrainer with num_workers=N
  owns Ray run state and Flyte task return value
  uses S3 RunConfig.storage_path

Ray Train:
  creates one GPU training actor per worker pod
  initializes PyTorch DDP/NCCL
  restarts all training workers after a recoverable worker failure
  supplies the latest reported checkpoint to restarted workers

Each rank:
  assigned train/validation scene partitions only
    -> local FlyteDirectory download/cache
    -> WebDataset + DataLoader workers
    -> AutoE2E replica
    -> forward/backward
    -> NCCL all-reduce

Rank 0 only:
  MLflow run ownership
  Ray checkpoint payload creation and canonical model-artifact pointer updates
  checkpoint selection and early-stopping decision

On task completion:
  head returns Flyte TrainOutput
  KubeRay deletes the transient RayCluster after a short debug TTL
```

### 6.1 Control-plane ownership

| Concern | Owner | Explicitly not owned by |
|---|---|---|
| Workflow DAG, typed inputs/outputs, cache | Flyte | Ray |
| Final task status and outer retry | Flyte | KubeRay |
| Queue priority and whole-workload admission | Kueue | Flyte, Ray autoscaler |
| RayCluster creation and deletion | KubeRay RayJob controller | Training code |
| Training actor placement inside the admitted cluster | Ray | Flyte |
| Worker-group restart and latest reported checkpoint handoff | Ray Train | KubeRay |
| Gradient synchronization and optimizer semantics | PyTorch DDP | Ray Core |
| EC2 node provisioning and replacement | Karpenter/EKS Auto Mode | Ray |
| Durable checkpoint and model artifacts | S3, indexed by Ray/Flyte execution identity | Pod-local disk |

There is no independent long-lived Ray scheduler outside a Flyte task. A Flyte
attempt owns exactly one RayJob, and a RayJob owns exactly one transient
RayCluster. This boundary prevents Flyte and Ray from both trying to schedule
the same application-level DAG.

## 7. Flyte and Ray task boundary

Keep task topology static at registration time. Do not make `world_size` a
normal workflow input because the Flyte Ray task configuration determines the
number of Ray worker pods. Retain explicit wrappers for the reviewed topology
allowlist `2`, `4`, and `8`.

The Flytekit 1.x shape is:

```python
from flytekit import Resources, task
from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig
from ray import train
from ray.train.torch import TorchTrainer

RAY_4 = RayJobConfig(
    head_node_config=HeadNodeConfig(
        requests=Resources(cpu="2", mem="8Gi"),
        limits=Resources(cpu="2", mem="8Gi"),
    ),
    worker_node_config=[
        WorkerNodeConfig(
            group_name="gpu-workers",
            replicas=4,
            min_replicas=4,
            max_replicas=4,
            requests=Resources(cpu="4", mem="16Gi", gpu="1"),
            limits=Resources(cpu="4", mem="16Gi", gpu="1"),
        )
    ],
    enable_autoscaling=False,
    shutdown_after_job_finishes=True,
    ttl_seconds_after_finished=300,
)

@task(
    task_config=RAY_4,
    retries=1,
    labels={"kueue.x-k8s.io/queue-name": "training"},
    ...,
)
def train_il_ray_4(...) -> TrainOutput:
    trainer = TorchTrainer(
        train_loop_per_worker=_train_il_worker,
        train_loop_config={...},
        scaling_config=train.ScalingConfig(
            num_workers=4,
            use_gpu=True,
            resources_per_worker={"CPU": 4, "GPU": 1},
        ),
        run_config=train.RunConfig(
            name=<stable-flyte-execution-id>,
            storage_path=<s3-ray-training-root>,
            failure_config=train.FailureConfig(max_failures=2),
        ),
    )
    return _result_to_flyte_output(trainer.fit())
```

`RayJobConfig` worker replicas, `min_replicas`, `max_replicas`, and
`TorchTrainer` workers must be the same integer. The worker pod template carries
the GPU taint toleration, placement-group node selector, and hostname
anti-affinity. The head template explicitly selects a CPU pool and requests no
GPU. Production dependencies are baked into one image; do not install them
through `runtime_env`.

Use `flytekitplugins-ray==1.16.24` with the existing Flytekit generation. A
migration to the newer `flyteplugins-ray` API is separate from this milestone.
The generated RayJob must carry `kueue.x-k8s.io/queue-name`, use fixed replicas,
set `shutdownAfterJobFinishes=true`, and disable autoscaling.

Ray 2.46.0 requires `RAY_TRAIN_V2_ENABLED=1` for the constructor-based driver
recovery contract used here. Set it explicitly in the task image/environment
and test it as part of the pinned matrix; do not use the deprecated
`TorchTrainer.restore()` path.

Derive the Ray run key from Flyte workflow execution ID plus node ID, excluding
the task-attempt number. Flyte retries therefore create new RayJobs but resolve
the same Ray checkpoint and MLflow namespaces. In particular, every attempt
must construct `RunConfig` with the same `(storage_path, name)` pair so Ray
Train can recover the persisted run state and latest checkpoint automatically.

## 8. DDP process contract

Ray Train owns the default PyTorch process group lifecycle. Every training
worker must:

1. read world, local, and node ranks from `ray.train.get_context()`;
2. obtain the assigned CUDA device through Ray Train before constructing CUDA
   state;
3. let `TorchTrainer` initialize the NCCL process group;
4. create one auxiliary Gloo group for object metadata and long validation
   coordination after the default group is ready;
5. construct the model with the same seed and configuration;
6. wrap it through `ray.train.torch.prepare_model`, which produces DDP for a
   multi-worker TorchTrainer;
7. destroy only the auxiliary group in `finally`; Ray owns teardown of the
   default group.

Do not call `ray.train.torch.prepare_data_loader` for the outer scene loader.
This repository performs deterministic scene ownership before opening child
WebDataset loaders; adding Ray's default distributed sampler would shard the
input a second time.

The initial DDP settings are:

- `device_ids=[local_rank]`;
- `output_device=local_rank`;
- `gradient_as_bucket_view=True`;
- `find_unused_parameters=True` during integration.

Pass these through `prepare_model`'s tested DDP configuration surface; do not
wrap the model a second time after Ray Train has prepared it.

Run with `TORCH_DISTRIBUTED_DEBUG=DETAIL` in smoke tests. If a full-objective
backward proves that every trainable parameter participates on every batch,
switch to `find_unused_parameters=False`. Do not enable `static_graph=True`
until sparse reasoning/route batches and the diagnostic `autograd.grad` calls
have also been tested.

Checkpoint model state from `ddp_model.module.state_dict()` so one-GPU
inference and existing evaluation do not acquire a `module.` key prefix.

## 9. Global batch and optimizer equivalence

Define:

```text
global_effective_batch =
    per_rank_micro_batch * grad_accum_steps * world_size
```

The matched first experiment is:

```text
single GPU: 1 * 4 * 1 = 4
four GPUs:  1 * 1 * 4 = 4
```

Keep `lr=1e-4`, AdamW, weight decay, clipping, scheduler, and all objective
weights unchanged. This isolates distribution as the experimental variable.

Pure data parallelism cannot preserve global batch 4 with more than four ranks
when each rank must receive at least one sample. The supported batch contracts
are therefore:

| World size | Per-rank batch | Accumulation | Global batch | Purpose |
|---:|---:|---:|---:|---|
| 1 | 1 | 4 | 4 | Baseline |
| 2 | 1 | 2 | 4 | Integration canary |
| 4 | 1 | 1 | 4 | Quality parity and first production candidate |
| 8 | 1 | 1 | 8 | Maximum large-batch scale experiment |

Add `target_global_batch_size` as the user-facing contract. Derive accumulation
from it and fail unless:

```text
target_global_batch_size % (per_rank_micro_batch * world_size) == 0
```

Do not silently let the global batch grow with world size. For the eight-node
experiment, record the changed batch as part of the experiment identity and
calibrate learning rate before production use. Compare at least the fixed
parity learning rate, square-root scaling, and linear scaling from the
world-size-4 reference; select by the predeclared validation gate rather than
assuming linear scaling is correct.

When accumulation is greater than one, wrap the first `N-1` forward/backward
micro-steps in `ddp_model.no_sync()`. The final micro-step performs one
all-reduce for the complete accumulation window. All ranks must enter and leave
each window on the same step.

Clip gradients only after synchronization and AMP unscale. Every rank performs
the optimizer and scheduler step so optimizer state remains identical.

## 10. Distributed data contract

### 10.1 Training dataset index

Add a small immutable `TrainingDatasetIndex` artifact after packing and
navigation-quality audit. One record per partition contains at least:

- input list index;
- dataset, dataset version, source revision, and partition ID;
- FlyteDirectory URI and manifest SHA-256;
- unique sample count;
- effective training exposure count after navigation repeat policy;
- `split_group_uid` and train/validation membership;
- camera/geometry contract;
- reasoning, World Model, route, and GPS capability flags;
- tar count and packed byte count.

The index has a schema version and a digest over its canonical JSON. The digest
becomes part of the checkpoint data fingerprint.

This avoids making every rank download every scene merely to inspect
`manifest.json`.

### 10.2 Rank assignment

Rank 0 validates the index and computes a deterministic longest-processing-time
assignment:

1. sort train partitions by descending effective exposure count, then
   partition ID;
2. assign each partition to the rank with the lowest current exposure sum,
   breaking ties by rank;
3. use a separate deterministic assignment for validation;
4. broadcast the complete assignment over the Gloo control group;
5. persist assignment and digest in MLflow/checkpoint metadata.

Scene groups remain indivisible. A train/validation scene cannot cross ranks or
splits.

Only the owning rank calls `FlyteDirectory.download()` for a partition. On a
single multi-GPU pod, rank 0 may stage all assigned directories once to shared
ephemeral storage before a barrier. On multi-node, each pod stages only its own
partitions.

### 10.3 WebDataset behavior

Add an explicit rank-owned mode to `make_pre_extracted_loader`.

In this mode, use a module-level identity node splitter because outer code has
already assigned disjoint inputs. Do not use:

- `single_node_only`, which intentionally raises at world size greater than one;
- `split_by_node`, which would split every small scene tar list a second time.

Keep WebDataset's default `split_by_worker` exactly once for DataLoader worker
parallelism. Include rank and epoch in each shuffle seed:

```text
seed = training_seed + epoch * 1_000_003 + rank * 10_007
```

For an intra-epoch resume, rebuild the deterministic iterator for that epoch
and skip exactly the checkpointed rank-local exposure count before yielding the
next batch. The skipped samples perform no forward, backward, metric, or
optimizer work. Persist and verify the identity of the next sample on every
rank so a resume cannot silently replay or omit an optimizer input.

The first distributed scope keeps `batch_size=1`, so scene-boundary partial
batches do not exist. Supporting a larger per-rank batch requires a rebatching
or explicit `drop_last` contract and is deferred.

### 10.4 Equal step count without dropping scenes

Let `E_r` be the effective number of assigned training exposures on rank `r`.
For per-rank batch `B` and accumulation `A`:

```text
optimizer_steps_per_epoch = max_r(ceil(E_r / (B * A)))
rank_exposures_per_epoch = optimizer_steps_per_epoch * B * A
```

Each rank iterates all of its assigned exposures, then deterministically cycles
from the start only to fill its padding tail. No unique scene is dropped.
Longest-processing-time assignment keeps padding small.

Record:

- unique and effective exposure count per rank;
- padded exposure count per rank;
- padding ratio;
- optimizer and micro-step counts.

The initial gate is padding ratio <= 2%. A larger ratio fails before training
and requires a finer partition plan; it must not be hidden with excessive
sample repetition.

Do not use DDP `join()` for normal training. Although it prevents collective
hangs for uneven inputs, ranks that finish early stop normal optimizer work and
complicate optimizer/checkpoint equivalence.

## 11. Losses, metrics, and failure propagation

Local Python means are not valid global metrics. Track each loss as a weighted
sum and count, then `all_reduce(SUM)` before calculating the epoch result.
Eligibility-dependent route/rollout terms use their actual eligible sample
counts, preserving the current metric contract.

Use distributed reductions for:

- total, trajectory, JEPA, reasoning, route, and rollout losses;
- route and rollout eligibility counts;
- samples and optimizer steps;
- non-finite/error flags;
- maximum rank wall time.

Global throughput is:

```text
sum(rank_sample_count) / max(rank_training_wall_seconds)
```

Log per-rank input and timing summaries as one rank-0 JSON artifact. This is
needed to distinguish compute, input, and straggler bottlenecks.

Any rank detecting an invalid batch, non-finite loss/gradient, or data contract
violation sets a distributed failure flag and raises the worker group. No rank
may continue alone.

## 12. Validation and checkpoint selection

Validation is distributed by scene but does not use the DDP wrapper:

1. every rank evaluates its assigned validation scenes with
   `ddp_model.module` under `torch.no_grad()`;
2. uneven validation counts are safe because validation performs no DDP
   forward/backward collectives;
3. scalar metric partials are reduced;
4. sample identities and rollout-selector records are merged on rank 0;
5. rank 0 verifies global sample count/digest and computes the canonical
   scene-balanced selector;
6. rank 0 broadcasts scheduler metric, best-checkpoint decisions, and
   early-stop state;
7. every rank applies the same scheduler step and stop decision.

For small canaries, Gloo `gather_object` is sufficient. Full KITScenes selector
records use one immutable compressed JSONL partial per rank and epoch in S3;
rank 0 merges those artifacts. This avoids large pickled Python objects in a
collective and leaves auditable evidence.

Only rank 0:

- creates/reopens the MLflow run;
- updates best pointers;
- registers model versions;
- writes `metadata.json`;
- creates the checkpoint directory reported to Ray Train.

Training workers do not return Flyte outputs. The Ray head/driver converts
`TorchTrainer.fit()` results into the single `TrainOutput` returned by the
Flyte task.

## 13. Checkpoint and resume

Bump the checkpoint schema for distributed state. Create a checkpoint at every
epoch boundary and often enough during an epoch to keep the measured recovery
point objective at or below 15 minutes. The step interval is a pinned training
parameter derived from a checkpoint-overhead canary, not an unbounded
wall-clock timer inside one collective.

Every checkpoint stores:

- unwrapped model state;
- optimizer, scheduler, and scaler state;
- epoch, micro-step, and global optimizer-step count;
- world size, RayJob/worker topology, backend, and precision;
- per-rank RNG state;
- training index and rank-assignment digests;
- per-rank exposure, padding, loader cursor, and accumulation-window state;
- global batch derivation;
- metric history and selector state;
- existing model/data/objective fingerprints.

Gather RNG states to rank 0 before serialization. On resume, rank 0 loads and
validates the checkpoint, broadcasts the common state, and each rank restores
its own RNG and loader cursor. Resume only from a completed optimizer-step
boundary; never serialize a partially accumulated gradient window.

Rank 0 writes the checkpoint into a temporary local directory and constructs
`Checkpoint.from_directory(...)`. Every rank must then call
`ray.train.report()` at the same logical cadence because the call is Ray
Train's synchronization barrier. Rank 0 passes the checkpoint payload and
non-zero ranks pass `checkpoint=None`; they do not upload duplicate replicated
state. All ranks enter the preceding state-gather collectives before that
report call. Configure:

```python
train.RunConfig(
    name=<stable-flyte-execution-id>,
    storage_path="s3://<bucket>/ray-train/",
    failure_config=train.FailureConfig(max_failures=2),
    checkpoint_config=train.CheckpointConfig(num_to_keep=3),
)
```

The retained recovery checkpoints and the existing immutable model-selection
checkpoints have different lifecycles. Ray may prune old recovery checkpoints;
it must not delete canonical best/epoch artifacts referenced by MLflow or the
model registry.

Checkpoint schema v1 resume rules:

- world size must match;
- per-rank micro-batch, accumulation, and global batch must match;
- partition/index and assignment digests must match;
- resume starts after the checkpoint's last completed optimizer step;
- one-GPU checkpoints may initialize a new DDP run as pretrained weights, but
  may not resume optimizer/scheduler history as if they were the same run;
- DDP checkpoints remain consumable by one-GPU inference because model keys are
  unwrapped.

### 13.1 Recovery hierarchy

Use four explicit failure classes:

1. **Training worker/GPU/node failure:** Ray Train restarts the complete
   training worker group, up to `max_failures=2`, and populates
   `ray.train.get_checkpoint()` with the latest reported S3 checkpoint.
2. **Ray head, driver, RayJob, or cluster failure:** the Flyte task retry creates
   a new RayJob. It reconstructs `RunConfig` with the same S3 `storage_path`
   and stable Ray run `name`; Ray Train loads the persisted run state and
   supplies its latest checkpoint to the new worker group.
3. **First CUDA OOM:** atomically create an OOM marker that authorizes one
   replay, raise a worker error, and let Ray replace the complete worker group.
   The new group resumes from the latest checkpoint that predates the OOM. Do
   not create a checkpoint from the failed step.
4. **Deterministic or repeated failure:** data-contract violations, non-finite
   gradients, or a second OOM with the same configuration/checkpoint lineage
   terminate the run. They are not repaired by changing hyperparameters
   automatically.

The first OOM replay is a bounded recovery probe for a transient allocator or
worker-state failure; it is not a claim that Ray fixes OOM. If the same
configuration reaches the same OOM lineage again, replaying the checkpoint
cannot make forward progress and must terminate.

The persisted Ray run state under the stable `(storage_path, name)` pair is the
recovery source of truth across both worker restarts and Flyte task attempts.
A Flyte retry must not generate a new Ray run name. A restarted group reopens
the same MLflow run and must not create conflicting immutable S3 keys.

Ray Train's worker failure budget does not cover job-driver crashes. Keep the
outer Flyte retry budget at one initially and the KubeRay RayJob backoff at
zero, so there is one owner at each retry layer. Raise either budget only after
fault-injection tests prove idempotency.

Catch `torch.cuda.OutOfMemoryError` and conditionally write an OOM marker
containing the configuration digest, source checkpoint URI, optimizer step,
sample identity, peak memory, and replay state. The first write changes the
state from absent to `replay_authorized` and raises a recoverable worker error.
The restarted group atomically claims that replay before loading the source
checkpoint. A matching OOM changes the marker to `terminal`; when the control
group remains responsive, return that coordinated terminal status to the
driver instead of raising another recoverable worker error. If the collective
is already broken, startup preflight reads the terminal marker and fails before
GPU training restarts. All marker transitions use S3 conditional writes so
concurrent ranks cannot grant multiple replays.

Do not attempt to checkpoint after OOM. Because the current per-rank batch is
already one, a repeated OOM requires an explicit change such as bf16,
activation checkpointing, or FSDP.

## 14. Infrastructure changes

### 14.1 Flyte

1. Enable `ray` in the active
   `Platform/helm-values/flyte-core-eks.yaml`.
2. Set `ray: ray` in the default task-type mapping.
3. Install the same `flytekit` and `flytekitplugins-ray` versions in
   registration and training environments.
4. Configure the Flyte Ray backend plugin to use a cluster-internal service,
   not an internet-facing Ray dashboard or `LoadBalancer`.
5. Ensure Flyte task labels propagate
   `kueue.x-k8s.io/queue-name: training` to the generated RayJob.
6. Register and launch a Flyte-generated two-worker Ray task before model work.
7. Verify Flyte output collection and RayCluster deletion, not only a
   hand-written RayJob.
8. Give Flyte control-plane service accounts a real storage role. In this
   cluster, `flyteadmin`, `flytepropeller`, and `datacatalog` use
   `auto-e2e-platform-s3-access`; a dangling `iam-role-flyte` annotation blocks
   workflows before task creation.
9. Define task-wide environment variables such as `RAY_TRAIN_V2_ENABLED` once.
   Flyte merges task environment into head and worker templates, and duplicate
   entries make the generated RayJob fail Kubernetes validation.

### 14.2 KubeRay

1. Install the KubeRay operator and RayJob/RayCluster CRDs through IaC.
2. Pin the operator version together with the Ray image and Flyte backend
   compatibility matrix.
3. Use a RayJob-created transient RayCluster; do not target an existing shared
   RayCluster. Kueue-managed RayJobs do not support that mode.
4. Set `shutdownAfterJobFinishes: true` and a short
   `ttlSecondsAfterFinished` for log inspection.
5. Keep one fixed worker group with
   `replicas == minReplicas == maxReplicas == N`; disable KubeRay and Ray
   autoscaling for the first release.
6. Put the Ray head on the CPU pool with no GPU request. Put each worker pod on
   the GPU pool with exactly one GPU.
7. Keep the dashboard cluster-internal and grant the Ray service account only
   the S3, MLflow, and dataset permissions required by the job.
8. Include `wget` in the training image. KubeRay 1.4.2 injects readiness
   commands that require it; an otherwise healthy Ray runtime remains
   unready when the binary is absent.

### 14.3 Kueue

1. Install KubeRay CRDs before validating Kueue's RayJob integration.
2. Create the LocalQueue in `auto-e2e-development`.
3. Put `kueue.x-k8s.io/queue-name` on the generated RayJob; use a namespace
   default only as a fallback.
4. Make the ClusterQueue namespace selector include the Flyte namespace.
5. Raise nominal quota to 8 GPU, at least 40 CPU, and 160 GiB. This includes
   eight current worker requests plus Ray head/submitter overhead.
6. Use an unconstrained CPU/memory ResourceFlavor and the `g6e-l40s` GPU
   ResourceFlavor. The CPU/memory flavor carries no node labels; the GPU flavor
   carries the GPU-pool/placement labels and applies only to pod sets requesting
   `nvidia.com/gpu`. The head therefore consumes CPU/memory quota without
   inheriting the GPU flavor or occupying a reserved GPU node.
7. Let Kueue exclusively control `RayJob.spec.suspend`; do not mutate it from
   Flyte or an admission webhook after creation.
8. Enable all-or-nothing pod readiness/requeue behavior so a partially started
   Ray cluster does not hold GPUs indefinitely.
9. Keep research and production priorities explicit.

### 14.4 Karpenter, placement, and EC2 capacity

Use an EC2 cluster placement group, not a spread placement group. A cluster
placement group keeps the instances in one Availability Zone and places them
in the same high-bisection-bandwidth network segment. A spread placement group
optimizes failure isolation instead and is limited to seven running instances
per Availability Zone, so it cannot hold the eight-node target. This is the
strongest relevant EC2 placement control, but it does not promise that all
instances share one physical rack.

Create the complete capacity for each tested topology as one batch and use one
instance type. Reserve four instances for the required acceptance run. For the
optional eight-node run, request all eight together rather than incrementally
adding four nodes; cancel or replace the four-instance benchmark reservation as
appropriate for the experiment window. AWS recommends this launch shape and
warns that adding capacity later has a higher `InsufficientInstanceCapacity`
risk:

```bash
aws ec2 create-placement-group \
  --group-name auto-e2e-training-pg \
  --strategy cluster

aws ec2 create-capacity-reservation \
  --instance-type g6e.4xlarge \
  --instance-platform Linux/UNIX \
  --availability-zone-id usw2-azN \
  --instance-count "$INSTANCE_COUNT" \
  --placement-group-arn "$PLACEMENT_GROUP_ARN"
```

Set `INSTANCE_COUNT` to `4` for parity acceptance or `8` for the scale
experiment.

An `InsufficientInstanceCapacity` response for the eight-instance request is a
valid reason to defer only the scale experiment. Preserve the request
parameters and AWS error as capacity evidence; do not weaken the one-AZ,
one-instance-type, or placement-group constraints just to claim an eight-node
run.

Replace the `default` EKS Auto Mode `NodeClass` reference with a dedicated
class. The selectors below are the required shape; IaC supplies the actual IAM
role, subnet, security group, reservation, and placement-group identifiers:

```yaml
apiVersion: eks.amazonaws.com/v1
kind: NodeClass
metadata:
  name: auto-e2e-gpu-training
spec:
  role: <eks-auto-node-role>
  subnetSelectorTerms:
    - id: <private-subnet-in-selected-az>
  securityGroupSelectorTerms:
    - id: <eks-node-security-group>
  capacityReservationSelectorTerms:
    - id: <placement-group-backed-odcr-id>
  placementGroupSelector:
    name: auto-e2e-training-pg
```

The `gpu-training` NodePool must reference this class and remain pinned to the
same AZ as the ODCR. EKS warns that a cluster placement group is pinned to the
AZ of its first instance; allowing multiple AZs can make parallel initial
launches race and fail. Ray GPU worker pods must also select
`eks.amazonaws.com/placement-group-id: <pg-id>` so consolidation cannot move
them to nodes outside the placement group.

Set the NodePool aggregate limits to the eight-instance ceiling:

```yaml
spec:
  template:
    spec:
      nodeClassRef:
        group: eks.amazonaws.com
        kind: NodeClass
        name: auto-e2e-gpu-training
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: [reserved]
        - key: topology.kubernetes.io/zone
          operator: In
          values: [<selected-az>]
  limits:
    cpu: "128"
    memory: 1024Gi
    nodes: "8"
    nvidia.com/gpu: "8"
```

Infrastructure requirements are:

1. Allow 8 `g6e.4xlarge` nodes, 8 GPU, and at least 128 vCPU in
   `gpu-training`.
2. Set `karpenter.sh/capacity-type: reserved`. A targeted ODCR can appear in
   NodeClass status while an `on-demand` NodePool still launches normal
   On-Demand instances and leaves the reservation unused.
3. Keep all ranks in the placement group's one AZ and selected subnet set.
4. Verify all `N` reservations are active before admitting an `N`-node job.
5. Preserve the GPU taint/toleration and `karpenter.sh/do-not-disrupt`.
6. Add hostname spread/anti-affinity so one-rank-per-node remains true if the
   allowed instance list changes.
7. Verify node security-group self-traffic for rendezvous and NCCL.
8. Keep the Ray head and KubeRay operator on non-GPU pools.
9. Keep the eight-instance production ODCR active between jobs so the complete
   training topology remains available. GPU nodes may scale to zero, but the
   unused reservation remains billable. Treat cancellation as an explicit
   capacity and cost decision; an active reservation also prevents
   placement-group deletion.

The initial `g6e.4xlarge` run uses NCCL over standard enhanced networking. The
AWS EFA-supported-instance table starts G6e support at `g6e.8xlarge`;
`g6e.4xlarge` is not EFA-capable. If profiling shows all-reduce exceeds 20% of
step time, benchmark an EFA-capable shape such as `g6e.8xlarge` as a separate
topology and cost experiment. Do not describe EFA as an option that can be
enabled on the selected `g6e.4xlarge` nodes.

### 14.5 Storage and input

Eight GPUs can consume samples much faster than one. Before each scale step:

- measure DataLoader wait time and GPU utilization;
- verify rank-local staging fits ephemeral storage;
- verify aggregate S3 download and local read throughput;
- configure Ray `RunConfig.storage_path` to an S3 prefix accessible from every
  worker and the head;
- keep checkpoints and dataset shards out of the Ray object store;
- size the Ray object store explicitly so it cannot starve DataLoader memory;
- retain `pin_memory=True`, asynchronous host-to-device copies, and tuned
  DataLoader workers;
- avoid complete copies of the corpus on every node.

If input wait exceeds 15% of training step time, improve shard staging/cache or
decode parallelism before advancing to the next topology.

### 14.6 Capacity cost envelope

AWS Pricing APIs returned the following `us-west-2` Linux rates on 2026-08-08.
These figures assume eight `g6e.4xlarge` instances for 730 hours per month.
They exclude tax, EBS, S3, transfer, and the CPU Ray head. All-Upfront values
are amortized monthly; actual payment timing differs.

| Purchase model | Effective monthly cost | Discount from On-Demand |
|---|---:|---:|
| On-Demand / uncovered ODCR | $17,544.76 | 0% |
| Compute Savings Plan, 1y No Upfront | $13,263.81 | 24.4% |
| Compute Savings Plan, 1y All Upfront | $12,379.57 | 29.4% |
| EC2 Instance Savings Plan, 1y No Upfront | $11,053.19 | 37.0% |
| EC2 Instance Savings Plan, 1y All Upfront | $10,316.30 | 41.2% |
| Compute Savings Plan, 3y No Upfront | $9,095.22 | 48.2% |
| Compute Savings Plan, 3y All Upfront | $8,253.03 | 53.0% |
| EC2 Instance Savings Plan, 3y No Upfront | $7,579.33 | 56.8% |
| EC2 Instance Savings Plan, 3y All Upfront | $6,596.81 | 62.4% |

The On-Demand rate is `$3.00424` per instance-hour, or `$24.03392` per hour
for eight nodes. AWS publishes the same applicable EC2 Instance Savings Plan
rate for `USW2-BoxUsage` and `USW2-UnusedBox`, so a matching commitment can
discount both used and idle ODCR hours. It does not remove the commitment or
idle-capacity risk.

Do not buy an eight-node commitment solely because eight is the technical
ceiling. If four nodes are the sustained baseline and eight nodes are
occasional burst capacity, cover the measured baseline with a Savings Plan and
leave the extra four nodes On-Demand. Prefer an EC2 Instance Savings Plan only
when the `g6e` family and `us-west-2` commitment are acceptable; otherwise use
the more flexible Compute Savings Plan.

## 15. Dependency matrix

Pin one tested set in root requirements, image, registration, and launch
buildspecs. The conservative first matrix preserves the current training
runtime:

| Dependency | Initial pin |
|---|---|
| PyTorch | 2.4.1 |
| torchvision | 0.19.1 |
| CUDA base | 12.1 |
| Flytekit | 1.16.24 |
| flytekitplugins-ray | 1.16.24 |
| Ray | 2.46.0 |
| Ray Train API | V2 (`RAY_TRAIN_V2_ENABLED=1`) |
| KubeRay operator | 1.4.2 |
| WebDataset | 1.0.2 |
| Kueue | 0.18.1 |

Ray 2.46.0 and KubeRay 1.4.2 are the initial compatibility candidates reflected
by the current Flyte and KubeRay examples, not assumed production compatibility.
PR 1 must prove that the pinned Flyte backend emits a RayJob API version served
by the pinned KubeRay CRD and that Kueue admits it. Do not combine a Ray,
PyTorch, bf16, or DDP-behavior upgrade in one acceptance experiment.

## 16. Observability

Add the following MLflow parameters/tags:

- `distributed/enabled`;
- `distributed/runtime=ray_train`;
- `distributed/backend`;
- `distributed/world_size`;
- `distributed/ray_version`;
- `distributed/kuberay_version`;
- `distributed/ray_job_name`;
- `distributed/ray_cluster_name`;
- `distributed/ray_run_name`;
- `distributed/ray_worker_replicas`;
- `distributed/flyte_task_attempt`;
- `distributed/instance_type`;
- `distributed/availability_zone`;
- `distributed/placement_group_id`;
- `distributed/capacity_reservation_id`;
- `distributed/global_batch_size`;
- `distributed/per_rank_batch_size`;
- `distributed/grad_accum_steps`;
- `distributed/training_index_digest`;
- `distributed/assignment_digest`;
- `distributed/padding_ratio`;
- `distributed/checkpoint_schema`;
- `distributed/checkpoint_uri`;
- `distributed/recovery_point_steps`;
- `distributed/precision`.

Add epoch metrics/artifacts:

- global samples/s and optimizer steps/s;
- min/mean/max per-rank samples/s;
- min/mean/max data wait and step time;
- maximum allocated/reserved CUDA memory per rank;
- all-reduce time or communication fraction from profiler canaries;
- per-rank unique/effective/padded exposure counts;
- RayJob queue wait, cluster startup, and worker registration time;
- Ray worker-group restart count and last failure class;
- Flyte task attempt and restored checkpoint step;
- Ray object-store used/spilled bytes;
- validation merge identity and partial artifact digests.

Do not log independently from every rank into the same MLflow run. Persist Ray
head, worker, KubeRay operator, and Kueue events under the Flyte execution ID so
one incident can be traced across all control layers.

## 17. Verification and acceptance

### 17.1 Unit and CPU tests

1. Partition assignment is deterministic and disjoint.
2. Every train partition is assigned exactly once before padding.
3. Train and validation scene groups remain disjoint.
4. Every rank gets the same optimizer-step count.
5. Padding stays deterministic and is included in the digest.
6. Rank-owned loader mode does not apply a second node split.
7. Global weighted metric reduction matches a single-process reference.
8. Rank-0 checkpoint keys have no `module.` prefix.
9. Resume rejects world-size, global-batch, index, and assignment mismatch.
10. Non-rank-0 code cannot write MLflow or canonical model-artifact pointers.
11. Flyte serialization produces a RayJob with one CPU head, exactly `N`
    one-GPU workers, fixed replicas, autoscaling disabled, and cleanup enabled.
12. `RayJobConfig` replicas and `TorchTrainer.num_workers` mismatch fails before
    cluster submission.
13. A reported checkpoint round-trips model, optimizer, scheduler, per-rank
    RNG, loader cursor, and stable run identity.
14. The first OOM marker permits one checkpoint replay; a matching second
    marker is terminal and cannot trigger configuration mutation.
15. Reconstructing `TorchTrainer` under Ray Train V2 with the same
    `(storage_path, name)` restores the latest checkpoint without
    `TorchTrainer.restore()`.

### 17.2 Cluster smoke progression

1. Flyte-generated CPU-only RayJob queues, starts one head and workers, returns
   output, and deletes its RayCluster.
2. Two GPU workers, synthetic tensors, Ray Train DDP/NCCL all-reduce.
3. Two workers, one packed scene per rank, one forward/backward/step.
4. Two workers, uneven scene sizes, deterministic padding and two epochs.
5. Four workers, small KITScenes subset, full objective and validation merge.
6. Four workers, periodic checkpoint and same-process resume.
7. Four workers, kill one training process; Ray restarts all workers and resumes.
8. Four workers, delete one worker pod/node; replacement workers resume.
9. Delete the Ray head; one Flyte task retry creates a new RayJob and resumes.
10. Inject CUDA OOM; one checkpoint replay occurs and a repeated matching OOM
    terminates without a retry loop.
11. Full four-node KITScenes parity run.
12. Eight workers, the same full dataset, explicit global batch 8.

At each topology, run `nccl-tests` or an equivalent all-reduce benchmark before
the model canary. Record bandwidth, latency, placement-group membership, and
the AZ as platform evidence. Do not advance when Kueue cannot admit the whole
group or when EC2 cannot satisfy the placement-group capacity.

### 17.3 Recovery gate

Automatic recovery is accepted only if:

- a recoverable worker or node failure restarts the complete DDP worker group;
- the restored optimizer step equals the latest committed S3 checkpoint;
- measured lost work is at most the 15-minute recovery point objective;
- model, optimizer, scheduler, RNG, data cursor, and assignment digests restore;
- the same MLflow run and immutable checkpoint namespace are reused;
- no duplicate sample is applied across the resumed optimizer-step boundary;
- the replacement group retains the same world size and placement constraints;
- RayCluster resources are deleted after success or terminal failure;
- a first OOM replays exactly once from the preceding checkpoint;
- a repeated matching OOM and data-contract failures terminate without
  repeating GPU training.

### 17.4 Quality parity gate

Compare a one-GPU baseline and four-GPU DDP run with the same:

- dataset/index and validation split;
- initial model state;
- global batch size 4;
- optimizer and learning rate;
- objective and sampling policy;
- epoch count and checkpoint selector.

Exact bit equality is not required across different distributed reduction
orders. Accept DDP only if:

- every expected validation sample appears exactly once;
- no train/validation group leakage is introduced;
- loss curves show no systematic divergence;
- selected-checkpoint ADE/FDE and composite components remain within the
  predeclared experiment tolerance;
- every objective branch has non-zero, finite gradient evidence.

The tolerance must be declared before looking at the result. A suggested smoke
tolerance is 1% relative for deterministic aggregate losses and 2% relative
for ADE/FDE; the full comparison should use the repository's paired validation
records rather than only scalar thresholds.

### 17.5 Performance and topology-selection gate

Measure end-to-end epoch time, not only model step time.

For any candidate world size `N`:

```text
scaling_efficiency_N =
    throughput_N / (N * throughput_1)

incremental_efficiency_N_from_M =
    (throughput_N / throughput_M) / (N / M)
```

Evaluate in the order `2 -> 4 -> 8`. Attempt eight-node capacity after the
required four-node gate, but select the smallest topology that meets the
wall-time objective; eight nodes are a ceiling, not an assumption that more
ranks will be faster. If the eight-node reservation is unavailable, retain the
four-node result and mark the scale gate deferred with the capacity evidence.

Four-node acceptance:

- at least 2.5x end-to-end training speedup;
- at least 62.5% four-GPU scaling efficiency;
- report RayJob queue/startup overhead separately from steady-state epoch time;
- padding ratio <= 2%;
- input wait <= 15% of training step time;
- no rank straggler more than 20% slower than rank median;
- no extra immutable checkpoint or MLflow run on retry.

Eight-node acceptance additionally requires:

- the declared large-batch quality gate passes after learning-rate calibration;
- throughput is higher than the preceding accepted topology;
- both total and incremental scaling efficiency are reported;
- the projected wall-time reduction justifies the additional GPU-hours;
- communication fraction and input wait remain below their declared gates.

If a topology misses the gate, stop scaling and profile in this order:

1. JPEG/frame-pool input and local storage;
2. Python/DataLoader imbalance;
3. per-rank scene assignment;
4. NCCL all-reduce;
5. validation serialization/merge;
6. only then an EFA-capable instance shape, bf16, or a larger topology.

### 17.6 Platform validation evidence

The following smokes ran in the Platform deployment account, region
`us-west-2`, EKS context `auto-e2e-platform`, on 2026-08-08. The immutable
training image digest was:

```text
<account>.dkr.ecr.us-west-2.amazonaws.com/auto-e2e/training
  @sha256:2b82243b1f5d7207197c2a5196dc7e8614ce25dcc6d15938a730dcbf661099b4
```

The common placement was:

| Item | Value |
|---|---|
| Availability Zone | `us-west-2b` |
| Placement group | `auto-e2e-distributed-training-pg` |
| Instance shape | `g6e.4xlarge`, one L40S per node |
| Node capacity type | `reserved` |
| DDP backend | NCCL 2.20.5 |

Four-node direct RayJob `auto-e2e-ddp-smoke-4-084633`:

```json
{
  "world_size": 4,
  "hostname_count": 4,
  "steps": 4,
  "initial_global_loss": 0.4306890070438385,
  "final_global_loss": 0.42428308725357056,
  "maximum_parameter_delta": 0.0,
  "parameter_update_norm": 0.016307178884744644,
  "elapsed_seconds": 2.3099124580000137
}
```

All four GPU nodes carried the expected placement-group, reservation, and
`reserved` capacity labels. Kueue admitted the CPU head, four GPU workers, and
submitter as one workload. Ray wrote the 1,103,278,848-byte checkpoint:

```text
s3://<checkpoint-bucket>/ray-train/
  auto-e2e-ddp-smoke-4-084633/
  checkpoint_2026-08-08_09-00-26.264142/checkpoint.pt
```

The four-instance reservation was cancelled after the four-node and recovery
tests.

Checkpoint continuation used a new RayJob with the same
`(storage_path, run_name)` and requested six total steps. Ray logged that it
found the previous run snapshot and exposed its latest checkpoint through
`ray.train.get_checkpoint()`. The second worker group ran only steps 4 through
5 and reported:

```json
{
  "world_size": 4,
  "steps": 6,
  "initial_global_loss": 0.42262864112854004,
  "final_global_loss": 0.42122191190719604,
  "maximum_parameter_delta": 0.0,
  "parameter_update_norm": 0.007697849068790674,
  "elapsed_seconds": 1.4639309320000393
}
```

`checkpoint_manager_snapshot.json` retained both the step-4 and step-6
checkpoint records. This validates driver/RayJob-level checkpoint continuation;
worker and node fault injection is still required for the complete recovery
gate.

Flyte execution `ray-ddp-flyte-4-0929`, entity version
`0J8oNEpuyfs6Bc45kr1MZA`, validated:

```text
Flyte -> Ray backend plugin -> RayJob -> Kueue -> KubeRay
      -> Ray Train -> PyTorch DDP/NCCL -> S3 checkpoint -> Flyte output
```

Its generated RayJob `ray-ddp-flyte-4-0929-n0-0` succeeded with four workers,
zero parameter divergence, loss `0.4306890070 -> 0.4242830873`, and a durable
checkpoint under run name `ray-ddp-flyte-4-0929-ray-ddp-smoke-4`. The
RayCluster was deleted after the configured TTL.

An eight-instance batch ODCR was successfully created in the same placement
group. Direct RayJob `auto-e2e-ddp-smoke-8-0941` completed on eight distinct
nodes:

```json
{
  "world_size": 8,
  "hostname_count": 8,
  "steps": 4,
  "initial_global_loss": 0.4279312789440155,
  "final_global_loss": 0.42261189222335815,
  "maximum_parameter_delta": 0.0,
  "parameter_update_norm": 0.016332970932126045,
  "elapsed_seconds": 2.498050099000011
}
```

Kueue admitted one 2-CPU/8-Gi head, eight 4-CPU/16-Gi GPU workers, and one
submitter together. Every worker node carried the same placement group and
capacity reservation IDs. This proves the v1 ceiling is allocatable and that
an eight-rank NCCL ring can train and checkpoint. It does not prove that the
eight-node global-batch-8 configuration has acceptable KITScenes quality or
cost efficiency.

After evidence collection, the RayCluster was removed, all eight GPU instances
reached `terminated`, and the smoke-test ODCR reached `cancelled`. A persistent
reservation was then created for the production capacity path. It is an
unlimited, targeted ODCR for eight `g6e.4xlarge` instances in `us-west-2b`,
attached to `auto-e2e-distributed-training-pg`. It remains active while GPU
nodes scale to zero between jobs.

The smokes exposed four operational requirements now incorporated above:

1. Targeted ODCR consumption requires NodePool capacity type `reserved`.
2. The KubeRay probe path requires `wget` in the image.
3. Flyte task environment must not duplicate PodTemplate environment keys.
4. Flyte control-plane service accounts need a valid S3 role and trust policy.

Remaining evidence before production enablement:

- full KITScenes index, rank-owned WebDataset loading, and validation merge;
- matched one-GPU/four-GPU quality and end-to-end throughput results;
- `nccl-tests` bandwidth/latency results, not only training collectives;
- process, pod, node, and head fault injection;
- bounded first-OOM replay and terminal second-OOM behavior;
- clean process-group teardown without NCCL/TCPStore exit warnings;
- decision on `find_unused_parameters=False` after all objective branches run.

## 18. Rollout plan

### PR 1: Platform and RayJob smoke

- Align dependency pins.
- Install KubeRay and enable the active Flyte Ray plugin.
- Connect Flyte-generated RayJobs in the execution namespace to Kueue.
- Add distinct CPU-head and GPU-worker pod templates/resource flavors.
- Increase two-node canary capacity/quota.
- Run RayCluster cleanup, synthetic two-rank NCCL, and Flyte-output smoke tests.

### PR 2: Ray Train DDP core

- Extract `_train_il_impl`.
- Add `TorchTrainer`, Ray context, process-group lifecycle, and DDP preparation.
- Make all side effects rank-aware.
- Add global batch derivation and `no_sync`.
- Add distributed metric reductions and failure propagation.

### PR 3: Dataset index and rank-owned loading

- Add `TrainingDatasetIndex`.
- Add deterministic weighted partition assignment.
- Add rank-owned WebDataset mode and exact step equalization.
- Persist assignment and padding evidence.

### PR 4: Distributed validation and automatic recovery

- Add validation partials and canonical rank-0 merge.
- Add periodic Ray Train S3 checkpoints, per-rank RNG/cursor state, and
  fixed-world-size resume validation.
- Add Ray worker recovery and Flyte retry recovery with fault injection.
- Add terminal OOM classification.
- Preserve existing checkpoint selection, early stopping, and registry roles.

### PR 5: Four-node parity acceptance

- Create the cluster placement group and four-instance ODCR through IaC.
- Add the dedicated EKS Auto Mode `NodeClass` and an eight-node hard ceiling
  to the GPU NodePool.
- Raise capacity/quota from two to four.
- Run the matched one-GPU/four-GPU experiment.
- Record quality, throughput, utilization, padding, and NCCL evidence.
- Make the DDP workflow the full-run default only after all gates pass.

### PR 6: Placement-group scale evaluation

- Attempt an eight-instance placement-group-backed ODCR through IaC.
- Raise Kueue quota to eight workers.
- Benchmark `2`, `4`, and `8` ranks in the same placement group.
- Calibrate the eight-node large-batch run and select the smallest topology
  that meets quality, wall-time, and GPU-hour gates.
- If eight matching instances cannot be reserved together, retain the
  four-node production candidate, record the AWS capacity failure, and defer
  the eight-node benchmark.

## 19. Risks

| Risk | Mitigation |
|---|---|
| Loader exhaustion hangs DDP | Fixed per-rank step count; no normal-path `join()` |
| Global batch silently changes | Derive accumulation from explicit target and fail on non-divisibility |
| Scene imbalance causes duplication | LPT assignment, <=2% padding gate, persisted counts |
| Sparse branches produce unused parameters | Start with `find_unused_parameters=True`; test before optimization |
| Input cannot feed eight GPUs | Rank-local staging, input timing, stop scaling when input wait exceeds gate |
| Validation records are too large for collectives | Immutable per-rank compressed partials, rank-0 merge |
| Duplicate MLflow/checkpoint writes | Rank-0-only side effects and tests |
| Resume is not reproducible | Checkpoint per-rank RNG plus world/index/assignment contracts |
| Flyte, KubeRay, and Ray retry the same failure | One bounded owner per retry layer; Ray handles workers and Flyte handles the driver/RayJob |
| Partial Kueue admission wastes GPUs | Kueue-managed RayJob suspension, all-or-nothing readiness, and adequate quota |
| Ray head consumes a GPU node | Separate CPU pod template and unconstrained CPU ResourceFlavor |
| One node disappears mid-collective | Explicit timeouts, whole-group restart, and periodic S3 checkpoint resume |
| Ray head or driver disappears | Flyte task retry creates a new transient RayJob with the same durable Ray run identity |
| OOM loops through the recovery budget | One atomically claimed checkpoint replay, then a terminal marker keyed by configuration and checkpoint lineage |
| Ray checkpoint exists only on local disk | Mandatory S3 `RunConfig.storage_path` and stable run identity across Flyte attempts |
| Ray object store competes with DataLoader | Explicit object-store sizing; do not place dataset/checkpoints in it |
| RayCluster leaks after task completion | `shutdownAfterJobFinishes=true`, TTL, and cleanup acceptance test |
| Eight nodes are unavailable together | Request a placement-group ODCR for all eight together; defer the scale experiment with AWS capacity evidence if unavailable |
| Placement group is pinned to the wrong AZ | Pin NodeClass subnets and NodePool AZ before the first launch |
| Multi-node network limits scaling | Cluster placement group and NCCL profiling; separately test an EFA-capable shape only with evidence |
| Large-batch training changes quality | Keep world-size-4 parity gate; calibrate LR and quality separately at 8 |
| Platform and runtime package drift | One pinned dependency matrix in every build path |

## 20. References

Official documentation, accessed 2026-08-08:

1. [PyTorch 2.4 DistributedDataParallel](https://docs.pytorch.org/docs/2.4/generated/torch.nn.parallel.DistributedDataParallel.html)
   describes one-process-per-GPU DDP, NCCL guidance, `no_sync`, and uneven-input
   join behavior.
2. [Ray Train TorchTrainer](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html)
   defines the PyTorch training worker group, `ScalingConfig.num_workers`, and
   per-worker GPU execution.
3. [Ray Train fault tolerance](https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html)
   documents complete worker-group restart, `FailureConfig.max_failures`,
   latest-checkpoint restoration, and the separate driver-failure boundary.
4. [Ray Train checkpoints](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html)
   documents rank-0 checkpoint reporting for replicated DDP and checkpoint
   retention.
5. [Ray Train persistent storage](https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html)
   recommends cloud object storage such as S3 for multi-node checkpoints and
   configures it through `RunConfig.storage_path`.
6. [Flyte Ray integration](https://www.union.ai/docs/v2/flyte/integrations/ray/)
   documents transient KubeRay clusters, RayJob task configuration, fixed
   worker replicas, cleanup, and the Flyte Ray plugin.
7. [KubeRay RayJob quickstart](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayjob-quick-start.html)
   defines the RayJob-created RayCluster, head/worker pod groups, job
   submission, retry, and cleanup fields.
8. [Kueue RayJob guide](https://kueue.sigs.k8s.io/docs/tasks/run/rayjobs/)
   documents RayJob queue labels, Kueue control of `spec.suspend`, transient
   cluster requirements, and KubeRay compatibility.
9. [PyTorch FSDP](https://docs.pytorch.org/docs/2.4/fsdp.html) defines FSDP as
   parameter sharding, which is not required for the current model capacity.
10. [WebDataset multi-node training](https://rom1504.github.io/webdataset/multinode/)
   describes node/worker splitting and the equal-batch requirement for DDP.
11. [AWS G6e instances](https://aws.amazon.com/ec2/instance-types/g6e/) lists
    one L40S/48 GiB and 20 Gbps networking for `g6e.4xlarge`, and four L40S
    GPUs for `g6e.12xlarge`.
12. [EFA on Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/node-efa.html)
    describes EFA/NCCL requirements and the benefit for communication-heavy
    distributed training.
13. [EC2 placement strategies](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/placement-strategies.html)
    defines cluster placement groups as one-AZ, high-bisection-bandwidth
    network placement and recommends one launch request with one instance type.
14. [Capacity Reservations in placement groups](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/cr-cpg.html)
    documents placement-group-backed ODCR creation and its capacity rules.
15. [EKS Auto Mode NodeClass](https://docs.aws.amazon.com/eks/latest/userguide/create-node-class.html)
    documents `capacityReservationSelectorTerms`, `placementGroupSelector`,
    cluster-placement-group AZ pinning, and the spread-group seven-instance
    limit.
16. [EFA-supported instance types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
    lists `g6e.8xlarge` and larger G6e shapes, but not `g6e.4xlarge`.
17. [Capacity Reservation billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-reservations-pricing-billing.html)
    describes used and unused reservation charges and eligible billing
    discounts.
18. [AWS Savings Plans pricing](https://aws.amazon.com/savingsplans/pricing/)
    describes Compute and EC2 Instance Savings Plan commitment trade-offs.
