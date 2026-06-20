# AutoE2E MLOps Platform

EKS Auto Mode ベースの MLOps プラットフォーム。自律走行モデルの学習・評価・改善を再現可能な IaC で管理し、任意の AWS アカウントに移行可能。

## UI Access

| Service | URL |
|---------|-----|
| MLflow (実験管理) | https://d3t4qye59n0rhq.cloudfront.net/ |
| Flyte Console (パイプライン) | https://d3t4qye59n0rhq.cloudfront.net/console/ |

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

### EKS Auto Mode を選んだ理由

| 比較 | Auto Mode | Standard + Karpenter |
|------|-----------|---------------------|
| CNI | Built-in eBPF (addon不要) | vpc-cni addon管理必要 |
| Karpenter | 組み込み (CRD apply のみ) | IAM role + Helm + IRSA 設定必要 |
| LB Controller | 組み込み (TGB CRD) | Helm install + IAM 設定必要 |
| GPU driver | Bottlerocket AMI に含まれる | GPU Operator or AMI 管理 |
| 運用負荷 | 最小 | 中 |

**注意**: Auto Mode + Managed Node Group の混在は CNI 衝突を起こす (vpc-cni addon と Auto Mode 内蔵 CNI が競合)。本プラットフォームは **Auto Mode 純粋構成** とし、Managed NG は使用しない。

### CARLA を採用しなかった理由

当初 Phase 5 で CARLA Closed-Loop Simulation を計画したが断念:

1. **Vulkan 依存**: CARLA は Vulkan ICD が必要。Bottlerocket AMI には含まれない
2. **EKS上で動かせない**: nvidia-container-toolkit + Vulkan の組み合わせが Auto Mode Bottlerocket と非互換
3. **EC2 standalone も不安定**: g5.xlarge で動作したが EKS pods との通信が不安定
4. **Managed NG を追加すると CNI 衝突**: vpc-cni addon を入れると Auto Mode ノードが NotReady

**代替**: Offline RL (IQL) — シミュレータ不要、recorded data のみで policy 改善。将来的に NAVSIM (2D replay closed-loop) を統合予定。

### GPU 確保戦略 (ODCR)

g6e.4xlarge は On-Demand 確保が困難。On-Demand Capacity Reservation (ODCR) で確保:

```bash
# Training GPU (g6e.4xlarge, us-west-2b)
aws ec2 create-capacity-reservation \
  --instance-type g6e.4xlarge \
  --instance-platform Linux/UNIX \
  --availability-zone us-west-2b \
  --instance-count 1 \
  --end-date-type unlimited
```

NodePool は ODCR の AZ にピン留め。Spot は使用しない (学習中断リスク)。

### Flyte S3 認証の制約

Flyte の内部ストレージライブラリ (stow/minio-go) は AWS SDK v1 ベースで、**Pod Identity も IRSA も非対応**。解決:

- Terraform が IAM User (`auto-e2e-platform-flyte-s3`) + Access Key を作成
- post-apply で Flyte configmap を `auth-type: accesskey` に patch
- `terraform apply` 後に毎回 patch が必要 (Helm が configmap を上書きするため)

---

## Pipeline (全パイプラインE2E動作確認済み)

### Phase 1: Data Ingest

```
HuggingFace Dataset → IngestAdapter → WebDataset (.tar shards) → S3
```

- **IngestAdapter protocol**: L2D / NVIDIA Physical AI 対応
- 各エピソードを JPEG + egomotion に分解し WebDataset shard に pack
- 出力: `s3://auto-e2e-platform-datasets-{account}/l2d/v1.0/shards/train-000000.tar`

### Phase 2: IL Training (Imitation Learning)

```
S3 Shards → PyTorchJob (Kueue managed) → GPU g6e.4xlarge → MLflow
```

- **Kueue**: GPU quota 管理、priority-based admission
- **Training Operator**: PyTorchJob CRD → Pod with `nvidia.com/gpu` request
- **Model**: VisionPilot (SwinV2 Tiny + concat fusion)
- **Output**: Checkpoint (S3) + metrics (MLflow)
- `runPolicy.suspend: true` で Kueue が admission control

### Phase 3: Open-Loop Evaluation

```
Checkpoint → Inference → ADE/FDE + Comfort metrics → Gate Check
```

- **Metrics**: ADE (Average Displacement Error), FDE (Final Displacement Error)
- **Comfort**: Jerk, Lateral Acceleration
- **Gate**: ADE < 2.0m, FDE < 4.0m → PASS で次段階へ
- GPU or CPU で実行可能

### Phase 4: Offline RL (IQL)

```
WebDataset Shards → IQL (Implicit Q-Learning) → Refined Policy → MLflow
```

- **シミュレータ不要**: Expert demonstrations から Q-function を学習
- **手法**: Expectile regression (V) + Advantage-weighted regression (π)
- **Parameters**: τ=0.7, β=3.0, γ=0.99
- 入力は IL Training と同じ WebDataset shards

### Full Pipeline (E2E)

```
Data Ingest → IL Training → Evaluation (Gate) → Offline RL → Final Eval
```

全ステージが MLflow に experiment/run/artifact を記録。

---

## Infrastructure (Terraform)

全リソースは `platform/infra/` で Terraform 管理。Account ID はハードコードしない。

### Modules

| Module | 内容 |
|--------|------|
| `vpc` | VPC, Private/Public Subnets x3 AZ, NAT Gateway |
| `eks` | EKS Auto Mode, Cluster IAM, Node IAM, OIDC Provider, Pod Identity Agent |
| `storage` | S3 buckets (datasets, artifacts), Pod Identity associations |
| `rds` | PostgreSQL (db.t4g.micro), MLflow DB + Flyte DB |
| `mlflow` | Helm release (S3 artifacts, RDS backend) |
| `flyte` | Helm release (flyte-core), IAM User for S3 access |
| `kueue` | Helm release, ResourceFlavor/ClusterQueue/LocalQueue |
| `training-operator` | Kubeflow Training Operator v1.9.3 |
| `codebuild` | Docker image build (training, data-prep) |
| `ui-exposure` | CloudFront + VPC Origin + Internal ALB + Cognito + Target Groups |

### Deploy

```bash
cd platform/infra
cp environments/dev/secrets.auto.tfvars.example environments/dev/secrets.auto.tfvars
# Edit secrets.auto.tfvars with actual values

terraform init
terraform apply -var-file=environments/dev/terraform.tfvars \
               -var-file=environments/dev/secrets.auto.tfvars

# Post-apply (kubeconfig + K8s resources)
aws eks update-kubeconfig --name auto-e2e-platform --region us-west-2 --profile autowarefoundation

# GPU NodePool
kubectl apply -f platform/k8s/gpu-nodepool.yaml

# Kueue config
kubectl apply -f platform/k8s/kueue-config.yaml

# Flyte S3 patch (required after every terraform apply)
./platform/infra/post-apply-phase2.sh
```

### Cross-Account 移行

1. `secrets.auto.tfvars` の `hf_token` を設定
2. S3 backend bucket を新アカウントに作成
3. `terraform init -backend-config=...` で backend 切り替え
4. `terraform apply` — 全リソースが新アカウントに作成される
5. ODCR は手動作成 (AZ/instance-type に依存)

---

## Directory Structure

```
platform/
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

- **全ワークロード**: Private Subnet (インターネット非到達)
- **Outbound**: NAT Gateway 経由のみ
- **ALB**: Internal (internet-facing ではない)
- **UI アクセス**: CloudFront → VPC Origin → Internal ALB → Pod
- **認証**: Cognito User Pool (CloudFront Lambda@Edge)
- **SG 設計**:
  - CloudFront VPC Origin ENI SG → ALB SG (port 80)
  - ALB SG → EKS Cluster SG (ports 5000, 8080, 8088)

---

## Cost (dev 環境概算)

| リソース | 月額 (USD) |
|----------|-----------|
| EKS Auto Mode cluster | $73 |
| g6e.4xlarge ODCR (1 node, 24h) | ~$1,300 |
| System nodes (3x c6a.large) | ~$180 |
| RDS (db.t4g.micro) | ~$15 |
| NAT Gateway | ~$35 |
| S3 + CloudFront | ~$5 |
| **合計** | **~$1,600/mo** |

GPU ノードを使わない時間帯は NodePool の `limits` で制御。ODCR は capacity 確保のため常時保持。

---

## AWS Account & Authentication

| 用途 | AWS Profile | Notes |
|------|-------------|-------|
| Platform (EKS, MLOps) | `--profile autowarefoundation` | Terraform, kubectl |

全コマンドに `--profile autowarefoundation` 必須。Account ID は環境変数 or tfvars で管理（ハードコード禁止）。
