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
                │  │  GPU Pool (Karpenter, scale-to-zero)          │ │
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

## Training Pipeline (Flyte + Kubeflow Trainer)

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
│   │   ├── storage/                S3 buckets + IRSA + Mountpoint CSI
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

### Phase 1: Foundation (EKS + GPU + Container Registry)

Goal: `train.py` runs as a container on EKS with GPU.

- [ ] Terraform backend (S3 + DynamoDB state lock)
- [ ] VPC (Private Subnets x3 AZ + Public Subnets x3 + NAT Gateway)
- [ ] EKS Auto Mode cluster (managed Karpenter 内蔵, Private endpoint)
- [ ] NodePool 定義 (g6e.xlarge/2xlarge, Private Subnet, AZ 制約)
- [ ] S3 buckets (datasets, checkpoints, artifacts) + IRSA
- [ ] ECR repositories (training, data-prep, eval)
- [ ] CloudFront + internal ALB + Cognito 認証 (UI アクセス基盤)
- [ ] Training Dockerfile (PyTorch + auto_e2e) → ECR push
- [ ] Verify: Pod submit → Karpenter が g6e 起動 → train.py --smoke-test pass → node 回収

### Phase 2: Job Scheduling (Kueue + Training Operator)

Goal: Team members submit training jobs to a queue; GPU nodes auto-scale.

- [ ] Kubeflow Training Operator (PyTorchJob CRD)
- [ ] Kueue (ClusterQueue, LocalQueue, ResourceFlavor for GPU)
- [ ] Multi-node DDP PyTorchJob template
- [ ] S3 Mountpoint CSI for dataset access from Pods
- [ ] Verify: 2-node DDP training on L2D subset

### Phase 3: Data Pipeline (Flyte + LakeFS)

Goal: Raw OSS datasets are automatically converted to training-ready format.

- [ ] Flyte backend on EKS (Helm)
- [ ] LakeFS on EKS (Helm, S3-backed)
- [ ] Data prep Dockerfile (ffmpeg, torchcodec, parsers)
- [ ] Flyte data_ingest workflow (L2D: HF → JPEG extract → S3)
- [ ] Flyte data_ingest workflow (nvidia: SDK → extract → S3)
- [ ] Unified DataLoader that reads from pre-extracted format
- [ ] Verify: Flyte pipeline produces training-ready data, training job reads it

### Phase 4: Experiment Management (MLflow)

Goal: All experiments are tracked, comparable, and reproducible.

- [ ] MLflow server on EKS (Helm, RDS Postgres + S3 artifact store)
- [ ] Training container logs to MLflow (metrics, params, checkpoint artifact)
- [ ] MLflow Model Registry (lifecycle: None → Staging → Production)
- [ ] Verify: experiment comparison across runs in MLflow UI

### Phase 5: Evaluation Pipeline (Flyte + KServe)

Goal: Every checkpoint is automatically evaluated with open-loop metrics.

- [ ] Evaluation Dockerfile (model + metrics code)
- [ ] Flyte evaluation workflow (load checkpoint → val set → ADE/FDE/Comfort)
- [ ] KServe + Triton for GPU inference (batch eval)
- [ ] Gate logic: metrics must improve over previous best to promote
- [ ] Verify: Flyte auto-evaluates after training, promotes to MLflow Staging

### Phase 6: Closed-Loop Simulation (CARLA)

Goal: Models are tested in simulated driving scenarios before production.

- [ ] CARLA Dockerfile (server, headless GPU)
- [ ] Simulation NodePool (Karpenter, g5.xlarge, scale-to-zero)
- [ ] Flyte simulation workflow (provision → run scenarios → collect)
- [ ] ScenarioRunner integration (parallel scenario execution)
- [ ] Metrics: route completion, collision rate, comfort
- [ ] Verify: model runs closed-loop in CARLA, results feed back to MLflow

### Phase 7: CI/CD Integration

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

- **EKS Auto Mode**: Managed Karpenter 内蔵。NodePool 定義だけで GPU ノードが自動プロビジョン
- **GPU NodePool**: g6e.xlarge / g6e.2xlarge (L40S)。Private Subnet, AZ 制約付き
- **CPU NodePool**: system Pods (Flyte, MLflow, Prometheus 等) 用。Auto Mode default
- **Simulation NodePool** (将来): g5.xlarge (CARLA 用)

## GPU Reservation Strategy

- g6e はキャパシティが逼迫しやすいため、Zonal Reserved Instance で確保する方針
- RI 購入は運用確認後 (Phase 1 完了後に AZ/サイズを確定)
- Karpenter NodePool は RI と同じ AZ に制約をかける (購入後に設定追加)
- RI 未購入の間は On-Demand + Spot fallback で稼働

## AWS Account & Authentication

| Purpose | AWS Profile | Account | Notes |
|---------|-------------|---------|-------|
| EC2 dev (model code) | (default) | 833707099141 | g6e instance, SSM |
| Platform (EKS, MLOps) | `--profile autowarefoundation` | `<ACCOUNT_ID>` | Terraform, kubectl |

All `aws` / `terraform` / `eksctl` commands for platform work MUST use `--profile autowarefoundation`.
