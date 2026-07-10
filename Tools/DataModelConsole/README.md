# DataModelConsole

Autonomous Driving Data & Model Intelligence Platform — Phase 1

## Quick Start (Local Development)

### API Server (Go)
```bash
cd api
go run .
# Listens on :8080
```

### Frontend (Next.js)
```bash
cd web
npm install
npm run dev
# Listens on :3000, proxies /api to localhost:8080
```

### Environment Variables
Copy `.env.example` to `.env` in the respective directories and configure.

## Architecture

- **Frontend**: Next.js 15 (App Router, TypeScript, Tailwind, shadcn/ui)
- **API**: Go 1.23 (chi router, AWS SDK v2, S3 streaming)
- **Auth**: CloudFront + Cognito (Lambda@Edge)
- **Infra**: EKS Auto Mode, internal ALB, CloudFront VPC Origin

## Deployment

1. Build images: `aws codebuild start-build --project-name auto-e2e-platform-console`
2. Apply K8s manifests: `kubectl apply -f deploy/k8s/`
3. Terraform (CloudFront + SG): `cd deploy/terraform && terraform apply`

## Data Sources (Read-Only)

| Source | What | Access |
|--------|------|--------|
| S3 datasets bucket | WebDataset shards (L2D, NVIDIA) | Pod Identity |
| S3 datasets bucket | Reasoning label cache (per-sample JSON) | Pod Identity |
| S3 artifacts bucket | MLflow checkpoints, Flyte outputs | Pod Identity |
| MLflow (in-cluster) | Experiments, runs, metrics, model registry | HTTP proxy |
| Flyte Admin (in-cluster) | Executions, workflows, node status | HTTP proxy |

## Directory Structure

```
Tools/DataModelConsole/
├── api/          # Go API server
├── web/          # Next.js frontend
├── deploy/
│   ├── docker/   # Dockerfiles
│   ├── k8s/      # Kubernetes manifests
│   └── terraform/# CloudFront, SG, IAM
├── docs/         # Design documents
└── README.md
```
