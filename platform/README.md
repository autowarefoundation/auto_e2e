# ADAS Development Platform

EKS-based MLOps platform for end-to-end autonomous driving model development.
All training and inference runs as containers on Kubernetes.

## Architecture Overview

```
Developer (Mac)
    │
    ├── Model/ code changes → git push → CI (lint + unit test)
    │
    └── platform/ infra changes → terraform apply (--profile autowarefoundation)
                                        │
                ┌───────────────────────▼────────────────────────────┐
                │              EKS Cluster (us-east-1)               │
                │                                                    │
                │  ┌──────────────────────────────────────────────┐ │
                │  │  System / Control (CPU managed nodegroup)     │ │
                │  │                                               │ │
                │  │  Flyte       Kueue       MLflow    LakeFS    │ │
                │  │  (pipelines) (GPU queue) (exps)   (data ver) │ │
                │  │                                               │ │
                │  │  Prometheus + Grafana + DCGM    Kubecost     │ │
                │  └──────────────────────────────────────────────┘ │
                │                                                    │
                │  ┌──────────────────────────────────────────────┐ │
                │  │  GPU Pool (Karpenter; 1 warm g6e, do-not-disrupt) │ │
                │  │                                               │ │
                │  │  g6e.xlarge ── g6e.2xlarge ── (future: p5)   │ │
                │  │       │              │                        │ │
                │  │  PyTorchJob     PyTorchJob (multi-node DDP)  │ │
                │  │  Eval Jobs      KServe + Triton              │ │
                │  └──────────────────────────────────────────────┘ │
                │                                                    │
                │  ┌──────────────────────────────────────────────┐ │
                │  │  Simulation Pool (scale-to-zero, future)      │ │
                │  │  g5.xlarge (CARLA server + client)            │ │
                │  └──────────────────────────────────────────────┘ │
                │                                                    │
                └────────────────────────┬───────────────────────────┘
                                         │
                ┌────────────────────────▼───────────────────────────┐
                │                 Data Layer (S3)                     │
                │                                                    │
                │  s3://datasets/        Raw + processed datasets    │
                │  s3://checkpoints/     Model checkpoints           │
                │  s3://artifacts/       Metrics, logs, sim results  │
                │                                                    │
                │  LakeFS: branch per experiment for data lineage    │
                │  Mountpoint for S3 CSI: direct Pod mount (read)    │
                └────────────────────────────────────────────────────┘
```

## Data Pipeline (Flyte)

OSS datasets arrive as raw video + sensor logs. The platform converts them into
a training-ready format: pre-extracted JPEG frames + egomotion parquet + manifest.

```
Raw Dataset (HF / S3)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Flyte Data Pipeline                                        │
│                                                             │
│  1. Ingest        HF download / SDK fetch / S3 copy         │
│  2. Extract       Video → JPEG frames (per camera, 256x256) │
│  3. Normalize     Egomotion resampling (→10Hz), calibration  │
│  4. Index         Build manifest, assign train/val split     │
│  5. Version       LakeFS commit (dataset state snapshot)     │
│                                                             │
│  Parallelism: Flyte map_task per episode/clip               │
│  Compute: CPU nodes (c6i), Karpenter-scaled                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Training-Ready Format (S3, unified across all datasets):
    s3://datasets/{name}/{version}/
    ├── manifest.json
    ├── splits/train.json, val.json
    ├── frames/{sample_id}/cam_0.jpg ... cam_6.jpg
    ├── egomotion/episodes.parquet
    └── metadata/camera_params.json, dataset_info.json
```

### Datasets

| Dataset | Source | Cameras | Egomotion | Map | Status |
|---------|--------|---------|-----------|-----|--------|
| L2D | HuggingFace (yaak-ai/L2D) | 7 (6 surround + BEV map) | CAN bus 10Hz | BEV render included | Parser ready |
| NVIDIA PhysicalAI | HuggingFace (gated, SDK) | 7 | Pose-derived 10Hz | None | Parser + DL script ready |
| KIT Scenes | TBD | 6-9 | Pose-derived 10Hz | Lanelet2 → rasterize | PR #41 draft |

## Training Pipeline (Flyte + Kubeflow Training Operator)

```
Training-Ready Data (S3)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Flyte Training Pipeline                                    │
│                                                             │
│  1. Select data    LakeFS branch + manifest → subset        │
│  2. Launch job     PyTorchJob (Training Operator via Kueue) │
│  3. Monitor        Poll job status, stream metrics to MLflow│
│  4. Collect        Checkpoint → S3, final metrics → MLflow  │
│                                                             │
│  Compute: GPU nodes (g6e), Karpenter-scaled                 │
│  Distribution: DDP (single/multi-node), future FSDP         │
│  Queue: Kueue (priority, fair-sharing, preemption)          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Checkpoint (S3) + Experiment Record (MLflow)
```

## Evaluation Pipeline (Flyte)

```
Checkpoint (S3)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Flyte Evaluation Pipeline                                  │
│                                                             │
│  1. Open-loop     ADE/FDE at 1s/2s/3s/6.4s, Comfort        │
│  2. Gate          Compare vs baseline + previous best       │
│  3. Promote       Pass → MLflow Model Registry (Staging)    │
│  4. Closed-loop   (future) CARLA scenario suite             │
│  5. Release       Pass all gates → Production               │
│                                                             │
│  Compute: GPU node (inference), CPU node (metrics compute)  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Model Registry (MLflow): None → Staging → Production
```

## Closed-Loop Simulation (Future)

```
Model (from Registry)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Flyte Simulation Pipeline                                  │
│                                                             │
│  1. Provision     CARLA server Pod (GPU, headless)           │
│  2. Load          Model into client Pod (KServe/Triton)     │
│  3. Execute       ScenarioRunner: N scenarios in parallel   │
│  4. Collect       Route completion, collision, comfort       │
│  5. Report        Aggregate → MLflow + Grafana dashboard    │
│                                                             │
│  Compute: Simulation NodePool (g5.xlarge, scale-to-zero)    │
│  Orchestration: 1 CARLA server + N parallel scenario jobs   │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
platform/
├── infra/                          Terraform (IaC)
│   ├── modules/
│   │   ├── vpc/                    VPC, subnets, NAT
│   │   ├── eks/                    EKS cluster + managed nodegroup
│   │   ├── karpenter/              Karpenter controller + NodePool definitions
│   │   ├── gpu-operator/           NVIDIA GPU Operator Helm release
│   │   ├── storage/                S3 buckets + Pod Identity + Mountpoint CSI
│   │   ├── ecr/                    Container registries
│   │   ├── flyte/                  Flyte backend (Helm)
│   │   ├── mlflow/                 MLflow server (Helm, RDS Postgres + S3)
│   │   ├── lakefs/                 LakeFS server (Helm)
│   │   ├── kueue/                  Kueue ClusterQueue + LocalQueue
│   │   ├── training-operator/      Kubeflow Training Operator
│   │   └── observability/          Prometheus + Grafana + DCGM + Kubecost
│   ├── environments/
│   │   └── dev/                    Dev environment tfvars
│   └── main.tf
│
├── pipelines/                      Flyte workflow code (Python)
│   ├── data_ingest/                Raw → training-ready
│   │   ├── tasks.py                Typed tasks (ingest, extract, normalize, index)
│   │   └── workflow.py             DAG definition
│   ├── training/                   Launch + monitor PyTorchJob
│   │   ├── tasks.py
│   │   └── workflow.py
│   ├── evaluation/                 Open-loop metrics + gate
│   │   ├── tasks.py
│   │   └── workflow.py
│   ├── simulation/                 CARLA closed-loop (future)
│   │   └── workflow.py
│   └── end_to_end.py              Master workflow (data → train → eval → sim)
│
├── docker/                         Container images
│   ├── training/
│   │   └── Dockerfile              PyTorch + auto_e2e + training deps
│   ├── data-prep/
│   │   └── Dockerfile              ffmpeg + torchcodec + parsers
│   ├── eval/
│   │   └── Dockerfile              Model + metrics computation
│   └── carla/
│       └── Dockerfile              CARLA client (future)
│
├── helm-values/                    K8s addon Helm overrides
│   ├── flyte.yaml
│   ├── kueue.yaml
│   ├── karpenter.yaml
│   ├── mlflow.yaml
│   ├── lakefs.yaml
│   └── gpu-operator.yaml
│
├── k8s/                            Additional K8s manifests
│   ├── karpenter-nodepools/        GPU/CPU/Sim NodePool CRDs
│   ├── kueue-config/               ClusterQueue, LocalQueue, ResourceFlavor
│   └── pytorchjob-templates/       Reusable PyTorchJob specs
│
└── README.md                       (this file)
```

## Implementation Phases

### Phase 1: Foundation (EKS + GPU + Container Registry) — DONE

Goal: `train.py` runs as a container on EKS with GPU. Region us-west-2
(us-east-1 was fully g6e capacity-starved across all sizes).

- [x] Terraform backend (S3 + DynamoDB state lock; bucket in us-east-1)
- [x] VPC (Private Subnets x3 AZ + Public Subnets x3 + single NAT Gateway)
- [x] EKS Auto Mode cluster (managed Karpenter built-in, OIDC, access entry)
- [x] GPU NodePool (g6e.4xlarge, on-demand only, ODCR AZ, nvidia.com/gpu taint)
- [x] One warm GPU node via do-not-disrupt pause Deployment (NOT scale-to-zero;
      g6e capacity is too scarce to re-acquire on demand, so the node + ODCR
      are held)
- [x] S3 buckets (datasets, checkpoints, artifacts) + Pod Identity
- [x] ECR repositories (training, data-prep, eval)
- [x] Training Dockerfile (pytorch/pytorch CUDA runtime base) → built/pushed via Finch
- [x] Verified: smoke-test Pod on g6e — device=cuda, amp=bf16, peak VRAM ~4GB

(CloudFront + Cognito UI exposure moved into Phase 2 alongside the UIs that
need it — Flyte/MLflow consoles.)

### Phase 2: Queue + Orchestration + Tracking — see PHASE2.md

Goal: a team launches training from a UI, jobs queue on the GPU, results are
tracked. Detailed, reviewed design in `platform/PHASE2.md`.

Stack: Flyte (UI + orchestration + registry-driven hyperparameter sweep) →
Kueue (GPU quota, 2-tier research/production priority) → Kubeflow Training
Operator v1 (PyTorchJob) → warm g6e. MLflow (RDS-backed) for experiment
tracking + Model Registry. UIs exposed via internal ALB → CloudFront → Cognito.

- [x] StorageClass + RDS Postgres (db.r6g.large; flyteadmin + mlflow DBs) + Pod Identity associations
- [x] Kubeflow Training Operator v1.9.3 + Kueue 0.18.1 (kubeflow.org/pytorchjob)
- [x] Kueue objects: ResourceFlavor/ClusterQueue/LocalQueue + 2 WorkloadPriorityClass
- [x] MLflow (server-proxied S3 artifacts) + minimal MLflow logging in train.py
- [x] Flyte (flyte-binary) + kfpytorch plugin; LaunchPlan enums from registries; sweep
- [ ] Internal ALB → CloudFront + Cognito for Flyte/MLflow UIs
- [x] Verify: UI launch → Kueue admit → PyTorchJob on g6e → MLflow run + artifact

### Phase 3: Data Pipeline (Flyte + LakeFS)

Goal: Raw OSS datasets are automatically converted to training-ready format.

- [x] Flyte backend on EKS (Helm) — deployed in Phase 2
- [ ] LakeFS on EKS (Helm, S3-backed)
- [x] Data prep Dockerfile (ffmpeg, torchcodec, parsers)
- [x] Flyte data_ingest workflow (L2D: HF → JPEG extract → S3)
- [x] Flyte data_ingest workflow (nvidia: SDK → extract → S3)
- [x] Unified DataLoader that reads from pre-extracted format
- [x] Verify: Flyte pipeline produces training-ready data, training job reads it

(Experiment Management / MLflow was pulled forward into Phase 2 — it is needed
the moment the first UI-launched training run produces results to compare.)

### Phase 4: Evaluation Pipeline (Flyte + KServe)

Goal: Every checkpoint is automatically evaluated with open-loop metrics.

- [ ] Evaluation Dockerfile (model + metrics code)
- [ ] Flyte evaluation workflow (load checkpoint → val set → ADE/FDE/Comfort)
- [ ] KServe + Triton for GPU inference (batch eval)
- [ ] Gate logic: metrics must improve over previous best to promote
- [ ] Verify: Flyte auto-evaluates after training, promotes to MLflow Staging

### Phase 5: Closed-Loop Simulation (CARLA)

Goal: Models are tested in simulated driving scenarios before production.

- [ ] CARLA Dockerfile (server, headless GPU)
- [ ] Simulation NodePool (Karpenter, g5.xlarge, scale-to-zero)
- [ ] Flyte simulation workflow (provision → run scenarios → collect)
- [ ] ScenarioRunner integration (parallel scenario execution)
- [ ] Metrics: route completion, collision rate, comfort
- [ ] Verify: model runs closed-loop in CARLA, results feed back to MLflow

### Phase 6: CI/CD Integration

Goal: Code changes automatically trigger the full pipeline.

- [ ] GitHub Actions: on PR merge → build images → push ECR
- [ ] Flyte trigger: new image → end-to-end pipeline (data → train → eval → sim)
- [ ] Notification: Slack/Discord on pipeline completion or failure
- [ ] Dashboard: Grafana with GPU cost, queue depth, pipeline status

## Observability

| Component | Tool | Metrics |
|-----------|------|---------|
| GPU | DCGM Exporter + Prometheus | Utilization, memory, temperature, power |
| K8s | kube-prometheus-stack | Pod CPU/mem, node status, scheduling latency |
| Cost | Kubecost | Per-team GPU hours, Spot savings, idle waste |
| Pipelines | Flyte UI | Workflow status, duration, failure rate |
| Experiments | MLflow UI | Loss curves, metric comparison, model lineage |
| Data | LakeFS UI | Dataset branches, commit history, diff |

## Network & Security

```
Internet
    │
    ▼
CloudFront (WAF + Cognito auth)
    │
    ▼
ALB (internal, Private Subnet only)      ← インターネット非公開
    │
    ▼
EKS Pods (Private Subnet)
    │
    ▼ (outbound only)
NAT Gateway → Internet
```

- 全 EC2/Pod は Private Subnet に配置。インターネットへの outbound は NAT Gateway 経由
- ALB はインターネットに直接晒さない。CloudFront → internal ALB の構成
- CloudFront に Cognito (or IAM Identity Center) 認証を付けて全内部ツール UI を保護
- WAF は CloudFront に付与

| Internal Tool | Access |
|---|---|
| MLflow UI | CloudFront → ALB → mlflow-server Pod |
| Flyte UI | CloudFront → ALB → flyte-console Pod |
| Grafana | CloudFront → ALB → grafana Pod |
| LakeFS UI | CloudFront → ALB → lakefs Pod |

## EKS Configuration

- **EKS Auto Mode**: Managed Karpenter built-in. GPU nodes provisioned by
  declaring a NodePool CRD only.
- **GPU NodePool**: g6e.4xlarge (L40S 48GB), on-demand only (no spot), pinned to
  the ODCR AZ, taint `nvidia.com/gpu:NoSchedule`, label `workload-type=gpu-training`.
- **Built-in NodePools** (general-purpose / system): Auto Mode default, host the
  control-plane pods (Flyte, MLflow, Kueue, operators, Prometheus). CPU only.
- **Simulation NodePool** (future): g5.xlarge for CARLA.
- The NVIDIA device plugin is NOT installed — Auto Mode's Bottlerocket-NVIDIA AMI
  exposes `nvidia.com/gpu` out of the box (verified on the live node).

## GPU Reservation Strategy

- g6e on-demand capacity in us-east-1 was fully exhausted across all sizes/AZs;
  us-west-2 was too. The only capacity obtainable was a **g6e.4xlarge ODCR in
  us-west-2b** — hence the region and instance-size choice.
- One On-Demand Capacity Reservation (ODCR) holds the GPU capacity. The GPU
  NodePool is pinned to that AZ; a warm node sits on the reservation and is kept
  alive via `karpenter.sh/do-not-disrupt` (NOT scaled to zero — re-acquiring g6e
  capacity is not guaranteed).
- Spot is explicitly NOT used for training (no mid-run interruption).
- Converting the ODCR to a Zonal Reserved Instance for billing discount is a
  later cost optimization; the ODCR already guarantees the capacity.

## AWS Account & Authentication

| Purpose | AWS Profile | Account | Notes |
|---------|-------------|---------|-------|
| EC2 dev (model code) | (default) | `<DEV_ACCOUNT_ID>` | g6e instance, SSM |
| Platform (EKS, MLOps) | `--profile autowarefoundation` | `<ACCOUNT_ID>` | Terraform, kubectl |

All `aws` / `terraform` / `eksctl` commands for platform work MUST use
`--profile autowarefoundation`. Real account IDs live in `.env` (see repo root
`.env.example`), never committed.
