# DataModelConsole — Phase 1 Design Document

## 1. Overview

Autonomous Driving Data & Model Intelligence Platform の UI コンソール。
既存 Platform (Flyte + MLflow + EKS) を拡張し、S3 上の走行データ・学習データ・
モデル・評価結果を Scene 中心に統合的に閲覧できる基盤を提供する。

Phase 1 では S3 にすでに存在するデータ (WebDataset shards, reasoning label cache,
MLflow artifacts, Flyte execution metadata) を **読み取り専用** で可視化することに集中する。

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CloudFront                            │
│  (Cognito auth, WAF, custom domain)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTPS only
┌───────────────────────────▼─────────────────────────────────┐
│  Internal ALB (port 443 → 3000 frontend, /api → 8080 api)  │
│  SG: inbound from CloudFront prefix-list ONLY               │
└─────────┬──────────────────────────────────┬────────────────┘
          │                                  │
┌─────────▼──────────┐          ┌────────────▼───────────────┐
│ Next.js Frontend   │          │ Go API Server              │
│ (SSR + static)     │          │ (chi router, S3 SDK,       │
│ Port 3000          │          │  MLflow proxy, Flyte proxy)│
│                    │          │ Port 8080                  │
└────────────────────┘          └────────────────────────────┘
          │                                  │
          │         ┌────────────────────────┼──────────────┐
          │         │                        │              │
          ▼         ▼                        ▼              ▼
    ┌──────────┐ ┌──────────┐     ┌──────────────┐  ┌──────────┐
    │ S3       │ │ MLflow   │     │ Flyte Admin  │  │ RDS      │
    │ Buckets  │ │ (in-EKS) │     │ (in-EKS)     │  │ Postgres │
    └──────────┘ └──────────┘     └──────────────┘  └──────────┘
```

## 3. Data Sources (Phase 1 — Read-Only)

### 3.1 S3 Datasets Bucket (`auto-e2e-platform-datasets-<ACCOUNT_ID>`)

| Prefix | Contents | Console Use |
|--------|----------|-------------|
| `l2d/v1.0/shards/` | WebDataset .tar (cam_0..6.jpg, ego.npy, meta.json) | Scene browser, camera viewer |
| `nvidia_av/v1.0/shards/` | WebDataset .tar (7 cams, ego.npy, meta.json) | Scene browser |
| `reasoning_labels_cache/dataset=*/teacher=*/prompt_version=*/*.json` | Per-sample reasoning labels (v2 schema) | Reasoning label viewer, taxonomy stats |
| `mock/v1.0/shards/` | Test shards | Dev/QA only |

Shard naming: `{episode_prefix}_{frame_idx:06d}.{member}`
- L2D: `ep{episode_id}_{frame_idx}`
- NVIDIA: `{clip_hash}_{frame_idx}`

### 3.2 S3 Artifacts Bucket (`auto-e2e-platform-artifacts-<ACCOUNT_ID>`)

| Prefix | Contents | Console Use |
|--------|----------|-------------|
| `auto-e2e/development/*/` | Flyte task raw outputs (FlyteDirectory) | Run outputs, shard locations |
| `mlflow/{experiment_id}/{run_id}/artifacts/` | Model checkpoints (epoch_N.pt), configs | Model registry viewer |

### 3.3 MLflow (in-cluster HTTP)

- Experiments: `imitation-learning`, `offline-rl`
- Params: `data/*`, `model/*`, `train/*`, `rl/*`, `ctx/*`
- Metrics: `train/loss`, `eval/ade`, `eval/fde`, `eval/gate_pass`
- Model Registry: `auto-e2e-driving-policy`

### 3.4 Flyte Admin (in-cluster gRPC)

- Project: `auto-e2e`, Domain: `development`
- Workflows: wf_create_dataset, wf_train_il, wf_full_pipeline, etc.
- Executions: status, inputs, outputs, node details

## 4. Phase 1 Feature Set

### 4.1 Home / Dashboard
- KPI cards: total samples (L2D + NVIDIA), reasoning labels count, MLflow runs, latest eval ADE/FDE
- Recent Flyte executions (last 10)
- Storage usage per bucket/prefix

### 4.2 Datasets
- Dataset list (l2d, nvidia_av) with shard count, sample count
- Shard browser: list tar members, preview camera images
- Sample detail: 7-camera grid, ego signal plot, meta.json

### 4.3 Reasoning Labels
- Browse by dataset/teacher/prompt_version
- Label statistics: taxonomy distribution (per-group histograms)
- Single label inspector: 5-horizon view with evidence text

### 4.4 Models (MLflow Proxy)
- Experiment list with run counts
- Run detail: params, metrics, loss curve
- Model Registry: versions, stages, lineage

### 4.5 Runs (Flyte Proxy)
- Execution list with status, duration, workflow name
- Execution detail: DAG, node status, inputs/outputs
- Link to Flyte Console for full interaction

### 4.6 Scenes (Phase 1 — Derived from Shards)
- Scene = one sample in a shard (episode + frame_idx)
- Scene detail: camera mosaic + ego signals + reasoning label (if exists)
- Basic search: by dataset, episode, frame range

## 5. Domain Model (Phase 1 Subset)

```
Dataset (l2d | nvidia_av)
  └── Version (v1.0)
       └── Shard (train-000000.tar)
            └── Sample (ep0_000064)
                 ├── CameraFrame[] (cam_0..6.jpg)
                 ├── EgomotionHistory (ego.npy: float32[384])
                 ├── Metadata (meta.json: {episode_id, frame_idx})
                 └── ReasoningLabel? (from cache, 5 horizons)

MLflowExperiment
  └── MLflowRun
       ├── Params
       ├── Metrics[]
       └── Artifacts[] (checkpoints)

FlyteExecution
  └── Node[]
       ├── Status
       ├── Inputs
       └── Outputs
```

## 6. API Endpoints (Go)

### Datasets
- `GET /api/v1/datasets` — list datasets
- `GET /api/v1/datasets/{name}/versions` — list versions
- `GET /api/v1/datasets/{name}/versions/{ver}/shards` — list shards
- `GET /api/v1/datasets/{name}/versions/{ver}/shards/{shard}/samples` — list samples
- `GET /api/v1/datasets/{name}/versions/{ver}/shards/{shard}/samples/{key}/image/{cam}` — presigned URL

### Reasoning Labels
- `GET /api/v1/reasoning-labels/stats` — aggregate counts per dataset/teacher
- `GET /api/v1/reasoning-labels/{dataset}/{sample_id}` — single label

### Models (MLflow proxy)
- `GET /api/v1/mlflow/experiments` — list experiments
- `GET /api/v1/mlflow/experiments/{id}/runs` — list runs
- `GET /api/v1/mlflow/runs/{id}` — run detail (params, metrics)
- `GET /api/v1/mlflow/models` — registered models

### Runs (Flyte proxy)
- `GET /api/v1/flyte/executions` — list executions
- `GET /api/v1/flyte/executions/{id}` — execution detail

### System
- `GET /healthz` — liveness (always 200 while the process is up)
- `GET /readyz` — readiness; gates on S3 (HeadBucket) ONLY. MLflow/Flyte are
  intentionally excluded so a proxied-dependency outage degrades those tabs
  gracefully instead of pulling the whole pod out of the load balancer.

## 7. Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Frontend | Next.js 15 (App Router), TypeScript, Tailwind CSS, shadcn/ui | SSR for SEO-free internal tool, rich component library |
| API | Go 1.23, chi router, AWS SDK v2 | Low latency, small container, native S3 streaming |
| Auth | CloudFront + Cognito (reuse existing auth-edge Lambda) | Consistent with Flyte/MLflow access |
| Infra | EKS Auto Mode (existing cluster), Terraform | Already provisioned |
| Observability | Structured JSON logs, /healthz, /readyz | K8s native |

## 8. Deployment

- 2 Deployments: `console-api` (Go, port 8080), `console-web` (Next.js, port 3000)
- 1 internal ALB with path-based routing (/api → api, /* → web)
- Security Group: inbound TCP 443 from AWS CloudFront prefix-list only
- CloudFront distribution with VPC Origin (reuse `cloudfront` TF module)
- Cognito auth via existing `auth-edge` Lambda@Edge
- Namespace: `console` (new)

## 9. Roadmap

| Phase | Timeline | Scope |
|-------|----------|-------|
| **1** | Month 1-2 | Read-only S3 browser, MLflow/Flyte proxy, basic Scene view |
| **2** | Month 3-4 | Search (structured filters), Scene auto-segmentation, ODD tags |
| **3** | Month 4-5 | Model comparison, evaluation drill-down, trajectory visualization |
| **4** | Month 5-6 | Dataset Composer, Issue tracking, AI copilot (natural language search) |

## 10. Security

- No direct S3 bucket access from browser (all via API presigned URLs, short-lived)
- CloudFront-only ALB access (SG + prefix-list)
- Pod Identity for API server (read-only S3 policy)
- Cognito JWT validation at CloudFront edge
- No PII stored in Console's own state (stateless API)
