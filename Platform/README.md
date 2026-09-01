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
│  │  GPU Pools                                                   │   │
│  │  g6.2xlarge: inference, validation, and canaries             │   │
│  │  p5en.48xlarge Capacity Block: 4/8-GPU Ray training          │   │
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

### GPU Capacity Strategy

Two targeted `g6.2xlarge` ODCR slots are retained for inference,
validation, and two-rank canaries. Large nuPlan and multi-dataset training
uses one `p5en.48xlarge` Capacity Block purchased for the run window.
Capacity purchase is an operator action performed before job submission.
Kubernetes, Karpenter, Kueue, Flyte, CodeBuild, and task Pods never call
`PurchaseCapacityBlock`; they only consume an already purchased tagged block.
Do not add purchase triggers for Pending Pods, queued workloads, retries, or
autoscaling events.

Verify that the retained reservation is targeted, tagged, and has exactly two
available slots:

```bash
aws ec2 describe-capacity-reservations \
  --profile autowarefoundation \
  --region us-west-2 \
  --filters \
    Name=tag:Name,Values=auto-e2e-gpu-validation \
    Name=instance-type,Values=g6.2xlarge \
  --query 'CapacityReservations[].{Id:CapacityReservationId,Type:InstanceType,Total:TotalInstanceCount,Available:AvailableInstanceCount,Match:InstanceMatchCriteria,State:State}'
```

If an existing two-slot reservation is missing the selector tags, attach them
before applying the NodeClass:

```bash
aws ec2 create-tags \
  --profile autowarefoundation \
  --region us-west-2 \
  --resources cr-xxxxxxxxxxxxxxxxx \
  --tags \
    Key=Name,Value=auto-e2e-gpu-validation \
    Key=purpose,Value=inference-validation \
    Key=retention,Value=keep-until-user-cancels
```

The p5en Ray topology uses whole-GPU allocation rather than MIG:

| Workload | Worker Pods | GPUs per Pod | Ray Train workers |
|----------|-------------|--------------|-------------------|
| Validation | 2 | 1 | 2 |
| Four-GPU training | 1 | 4 | 4 |
| Eight-GPU training | 1 | 8 | 8 |

Keeping all workers for one trainer in a single Pod preserves intra-instance
GPU communication. Two independent four-GPU jobs can share the same p5en
node when Kueue admits both workloads. The p5en queue is reserved for these
four- and eight-GPU Ray jobs. Legacy single-GPU training and evaluation use
the retained g6 validation pool.

The p5en queue uses `BestEffortFIFO`. Priority influences admission among
workloads that fit, but a four-GPU job may be admitted while a higher-priority
eight-GPU job is blocked on the full-node quota. This avoids leaving half of
the node idle, at the cost of possible delay for an eight-GPU job. An admitted
training workload is never preempted to make room for a later workload.

Search for a block without purchasing it:

```bash
python3 Platform/scripts/p5en_capacity_block.py \
  --profile autowarefoundation \
  --region us-west-2 \
  search \
  --duration-hours 24 \
  --start-after 2026-09-02T00:00:00Z \
  --end-before 2026-09-09T23:59:59Z
```

The platform CLI accepts Capacity Blocks from one through 14 days. This keeps
the purchased window below the EKS Auto Mode NodePool lifetime limit.

The CLI requests offerings across all Availability Zones, then keeps only
`us-west-2a` and `us-west-2c`, where this VPC has private subnets. This
prevents purchasing a block that the cluster cannot use.

The purchase command defaults to the AWS DryRun path. The maximum fee is an
independent spend ceiling in addition to the exact expected fee:

```bash
python3 Platform/scripts/p5en_capacity_block.py \
  --profile autowarefoundation \
  --region us-west-2 \
  purchase \
  --offering-id cb-xxxxxxxxxxxxxxxxx \
  --expected-upfront-fee 0000.0000 \
  --max-upfront-fee 1500.0000 \
  --confirm "PURCHASE cb-xxxxxxxxxxxxxxxxx 0000.0000" \
  --duration-hours 24 \
  --start-after 2026-09-02T00:00:00Z \
  --end-before 2026-09-09T23:59:59Z
```

Repeat the same command with `--execute` only after reviewing the selected
Availability Zone, start time, end time, and upfront fee. The execute path
performs AWS DryRun again before sending the real purchase request.

Wait until the purchased reservation is active before launching Flyte:

```bash
python3 Platform/scripts/p5en_capacity_block.py \
  --profile autowarefoundation \
  --region us-west-2 \
  wait-ready
```

`wait-ready` refuses to return a block with less than 22 hours remaining.
It accepts a consumed active reservation so a second independent four-GPU
job can use the other half of an already running p5en node.
The nuPlan launcher reads the active reservation EndDate and passes it into
the workflow. The p5en training task has a 20-hour timeout and rejects any
start with less than 22 hours remaining. Production tasks save an S3
checkpoint every 128 optimizer steps, so a Flyte retry in the same execution
can resume from the latest persisted optimizer step. Use a longer Capacity
Block when the planned training cannot finish inside one 20-hour task window.
The sequential two-stage 8-GPU workflow needs at least 42 hours remaining:
20 hours for each task and a 2-hour safety margin before each task starts.
Purchase a 48-hour or longer block for that workflow.

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
S3 Shards → Ray Train (Kueue managed) → p5en.48xlarge → MLflow
```

- **Kueue**: GPU quota management, priority-based admission
- **Training Operator**: PyTorchJob CRD → Pod with `nvidia.com/gpu` request
- **Model**: AutoE2E (SwinV2 Tiny + BEV fusion)
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

# GPU NodeClasses and NodePools
ACCOUNT=$(aws sts get-caller-identity \
  --profile autowarefoundation \
  --query Account \
  --output text)
sed "s/REPLACE_WITH_AWS_ACCOUNT_ID/${ACCOUNT}/g" \
  ../k8s/karpenter-nodepools/gpu-nodeclass.yaml | kubectl apply -f -
kubectl apply -f ../k8s/karpenter-nodepools/gpu-nodepool.yaml

# Kueue config
kubectl apply -f ../k8s/kueue-config/kueue-objects.yaml

# Flyte S3 patch (required after every terraform apply)
AWS_PROFILE=autowarefoundation ./post-apply-phase2.sh
```

### Cross-Account Migration

1. Set `hf_token` in `secrets.auto.tfvars`
2. Create S3 backend bucket in new account
3. `terraform init -backend-config=...` to switch backend
4. `terraform apply` — all resources created in new account
5. Retain the two-slot g6 validation ODCR
6. Search and explicitly purchase a p5en Capacity Block for each large run

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
| g6.2xlarge validation ODCR (2 slots) | Region price |
| p5en Capacity Block | Offering-specific upfront fee |
| System nodes (3x c6a.large) | ~$180 |
| RDS (db.t4g.micro) | ~$15 |
| NAT Gateway | ~$35 |
| S3 + CloudFront | ~$5 |
| **Fixed subtotal** | **~$308/mo plus storage and GPU capacity** |

GPU nodes scale to zero when idle. The validation ODCR remains active, while
p5en capacity exists only for the purchased block window.

---

## AWS Account & Authentication

| Purpose | AWS Profile | Notes |
|------|-------------|-------|
| Platform (EKS, MLOps) | `--profile autowarefoundation` | Terraform, kubectl |

All commands require `--profile autowarefoundation`. Account IDs managed via env vars/tfvars (never hardcoded).
