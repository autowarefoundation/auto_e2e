# AutoE2E MLOps Platform

EKS Auto Mode-based MLOps platform for autonomous driving model training, evaluation, and refinement. Fully IaC-managed and portable across AWS accounts.

## UI Access

| Service | URL |
|---------|-----|
| MLflow (Experiment Tracking) | https://d33520viyb0smg.cloudfront.net/ |
| Flyte Console (Pipeline Orchestration) | https://d1fk8c95f6ice9.cloudfront.net/ |

---

## Architecture

```
                     ┌─────────────────────────┐
                     │      CloudFront         │
                     │   (VPC Origin, HTTP)     │
                     └────────────┬────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │   Internal ALB (port 80) │
                     │   Private Subnet Only    │
                     └────────────┬────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│                    EKS Auto Mode (us-west-2)                       │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  System Nodes (Auto Mode general-purpose / system pools)    │   │
│  │                                                             │   │
│  │  MLflow        Flyte         Kueue        Training Op       │   │
│  │  (tracking)    (pipelines)   (GPU queue)  (PyTorchJob)      │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  GPU Pool (Karpenter NodePool: g6e.4xlarge, L40S 48GB)      │   │
│  │                                                             │   │
│  │  PyTorchJob (IL Training, AMP bf16)                         │   │
│  │  Eval Jobs (Open-Loop metrics)                              │   │
│  │  Offline RL (IQL refinement)                                │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
└───────────────────────────────┬────────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────────┐
│                          Data Layer                                 │
│                                                                    │
│  S3 datasets bucket     → WebDataset shards (.tar)                 │
│  S3 artifacts bucket    → MLflow artifacts + checkpoints           │
│  RDS PostgreSQL         → MLflow DB + Flyte DB                     │
└────────────────────────────────────────────────────────────────────┘
```

## Design Decisions

### Why EKS Auto Mode

| Comparison | Auto Mode | Standard + Karpenter |
|------|-----------|---------------------|
| CNI | Built-in eBPF (no addon needed) | vpc-cni addon management required |
| Karpenter | Built-in (CRD apply only) | IAM role + Helm + IRSA setup required |
| LB Controller | Built-in (TGB CRD) | Helm install + IAM setup required |
| GPU driver | Included in Bottlerocket AMI | GPU Operator or AMI management |
| Ops overhead | Minimal | Medium |

**Note**: Mixing Auto Mode + Managed Node Groups causes CNI conflicts (vpc-cni addon conflicts with Auto Mode built-in CNI). This platform uses **pure Auto Mode** — no Managed NGs.

### Why Not CARLA

Originally planned CARLA Closed-Loop Simulation for Phase 5, but abandoned due to:

1. **Vulkan dependency**: CARLA requires Vulkan ICD, not available in Bottlerocket AMI
2. **Incompatible with EKS Auto Mode**: nvidia-container-toolkit + Vulkan not supported on Bottlerocket
3. **EC2 standalone unstable**: Worked on g5.xlarge but communication with EKS pods was unreliable
4. **Adding Managed NG causes CNI conflicts**: vpc-cni addon makes Auto Mode nodes NotReady

**Alternative**: Offline RL (IQL) — no simulator needed, learns from recorded data. NAVSIM (2D replay closed-loop) planned for future.

### GPU Capacity Strategy (ODCR)

g6e.4xlarge is difficult to acquire on-demand. Secured via On-Demand Capacity Reservation (ODCR):

```bash
# Training GPU (g6e.4xlarge, us-west-2b)
aws ec2 create-capacity-reservation \
  --instance-type g6e.4xlarge \
  --instance-platform Linux/UNIX \
  --availability-zone us-west-2b \
  --instance-count 1 \
  --end-date-type unlimited
```

NodePool is pinned to ODCR AZ. Spot is not used (training interruption risk).

### Flyte S3 Authentication Constraint

Flyte's internal storage library (stow/minio-go) uses AWS SDK v1 — does not support Pod Identity or IRSA. Solution:

- Terraform creates IAM User + Access Key for Flyte S3 access
- Post-apply patches Flyte configmap to use accesskey auth
- Must re-patch after every `terraform apply` (Helm overwrites configmap)

---

## Pipeline (All Verified E2E)

### Data Ingest

```
HuggingFace Dataset → IngestAdapter → WebDataset (.tar shards) → S3
```

- **IngestAdapter protocol**: Supports L2D / NVIDIA Physical AI
- Decomposes each episode into JPEG frames + egomotion, packs into WebDataset shards
- Output: `s3://auto-e2e-platform-datasets-{account}/{name}/{version}/shards/train-000000.tar`

### IL Training (Imitation Learning)

```
S3 Shards → PyTorchJob (Kueue managed) → GPU g6e.4xlarge → MLflow
```

- **Kueue**: GPU quota management, priority-based admission
- **Training Operator**: PyTorchJob CRD → Pod with `nvidia.com/gpu` request
- **Model**: VisionPilot (SwinV2 Tiny + concat fusion)
- **Output**: Checkpoint (S3) + metrics (MLflow)
- `runPolicy.suspend: true` enables Kueue admission control

### Open-Loop Evaluation

```
Checkpoint → Inference → ADE/FDE + Comfort metrics → Gate Check
```

- **Metrics**: ADE (Average Displacement Error), FDE (Final Displacement Error)
- **Comfort**: Jerk, Lateral Acceleration
- **Gate**: ADE < 2.0m, FDE < 4.0m → PASS to proceed
- Runs on GPU or CPU

### Offline RL (IQL)

```
WebDataset Shards → IQL (Implicit Q-Learning) → Refined Policy → MLflow
```

- **No simulator needed**: Learns Q-function from expert demonstrations
- **Method**: Expectile regression (V) + Advantage-weighted regression (policy)
- **Parameters**: τ=0.7, β=3.0, γ=0.99
- Input: same WebDataset shards as IL Training

### Full Pipeline

```
Data Ingest → IL Training → Evaluation (Gate) → Offline RL → Final Eval
```

All stages log experiments/runs/artifacts to MLflow.

---

## Infrastructure (Terraform)

All resources managed by Terraform under `Platform/infra/`. No hardcoded account IDs.

### Modules

| Module | Description |
|--------|------|
| `vpc` | VPC, Private/Public Subnets x3 AZ, NAT Gateway |
| `eks` | EKS Auto Mode, Cluster IAM, Node IAM, OIDC Provider, Pod Identity Agent |
| `storage` | S3 buckets (datasets, artifacts), Pod Identity associations |
| `rds` | PostgreSQL (db.t4g.micro), MLflow DB + Flyte DB |
| `mlflow` | Helm release (S3 artifacts, RDS backend) |
| `flyte` | Helm release (flyte-core), IAM User for S3 access |
| `kueue` | Helm release, ResourceFlavor/ClusterQueue/LocalQueue |
| `training-operator` | Kubeflow Training Operator v1.9.3 |
| `codebuild` | Docker image build + Flyte workflow registration (VPC) |


### Deploy

```bash
cd Platform/infra
cp environments/dev/secrets.auto.tfvars.example environments/dev/secrets.auto.tfvars
# Edit secrets.auto.tfvars with actual values

terraform init
terraform apply -var-file=environments/dev/terraform.tfvars \
               -var-file=environments/dev/secrets.auto.tfvars

# Post-apply (kubeconfig + K8s resources)
aws eks update-kubeconfig --name auto-e2e-platform --region us-west-2 --profile autowarefoundation

# GPU NodePool
kubectl apply -f Platform/k8s/gpu-nodepool.yaml

# Kueue config
kubectl apply -f Platform/k8s/kueue-config.yaml

# Flyte S3 patch (required after every terraform apply)
./Platform/infra/post-apply-phase2.sh
```

### Cross-Account Migration

1. Set `hf_token` in `secrets.auto.tfvars`
2. Create S3 backend bucket in new account
3. `terraform init -backend-config=...` to switch backend
4. `terraform apply` — all resources created in new account
5. ODCR created manually (depends on AZ/instance-type availability)

---

## Directory Structure

```
Platform/
├── infra/                          Terraform
│   ├── modules/
│   │   ├── vpc/
│   │   ├── eks/                    EKS Auto Mode (no Managed NG)
│   │   ├── storage/                S3 + Pod Identity
│   │   ├── rds/                    PostgreSQL
│   │   ├── mlflow/                 Helm release
│   │   ├── flyte/                  Helm release + IAM User
│   │   ├── kueue/                  Helm release
│   │   ├── training-operator/      kubectl apply (kustomize)
│   │   ├── codebuild/              Docker build
│   │   └── ui-exposure/            CloudFront + ALB + Cognito
│   ├── environments/dev/
│   ├── main.tf
│   ├── variables.tf
│   └── post-apply-phase2.sh
│
├── pipelines/                      Flyte workflows
│   ├── data_ingest/
│   │   ├── workflow.py
│   │   └── adapters/               L2D, NVIDIA adapters
│   ├── training/workflow.py
│   ├── evaluation/workflow.py
│   └── full_pipeline.py            Master pipeline
│
├── docker/
│   ├── training/Dockerfile         PyTorch + timm + webdataset + mlflow-skinny
│   └── data-prep/Dockerfile        lerobot + flytekit + ffmpeg
│
├── helm-values/
│   ├── mlflow.yaml
│   └── flyte.yaml
│
├── k8s/                            Post-apply K8s manifests
│   └── (GPU NodePool, Kueue config, TGB)
│
└── README.md                       (this file)
```

---

## Security & Network

- **All workloads**: Private Subnets (no internet reachable)
- **Outbound**: NAT Gateway only
- **ALB/NLB**: Internal only (not internet-facing)
- **UI access**: CloudFront → VPC Origin → Internal ALB/NLB → Pod
- **Auth**: Cognito User Pool (deployed on Flyte Console via Lambda@Edge). Credentials
  are shared with Core Contributors only — ask Ryota Yamada. Never stored in git.
- **SG design**:
  -   - CloudFront VPC Origin ENI SG → ALB/NLB SG (port 80)
  -   - ALB/NLB SG → EKS Cluster SG (service ports)

---

## Cost (dev estimate)

| Resource | Monthly (USD) |
|----------|-----------|
| EKS Auto Mode cluster | $73 |
| g6e.4xlarge ODCR (1 node, 24h) | ~$1,300 |
| System nodes (3x c6a.large) | ~$180 |
| RDS (db.t4g.micro) | ~$15 |
| NAT Gateway | ~$35 |
| S3 + CloudFront | ~$5 |
| **Total** | **~$1,600/mo** |

GPU nodes scale to zero when idle. ODCR holds capacity reservation.

---

## AWS Account & Authentication

| Purpose | AWS Profile | Notes |
|------|-------------|-------|
| Platform (EKS, MLOps) | `--profile autowarefoundation` | Terraform, kubectl |

All commands require `--profile autowarefoundation`. Account IDs managed via env vars/tfvars (never hardcoded).
