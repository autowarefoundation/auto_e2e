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
- **API**: Go 1.25 (chi router, AWS SDK v2, S3 streaming)
- **Auth**: CloudFront + Cognito (Lambda@Edge)
- **Infra**: EKS Auto Mode, internal ALB, CloudFront VPC Origin

## Deployment

Production infrastructure is owned only by `Platform/infra-console`. The
`deploy/terraform` root is retired.

1. Build digest-pinned images with
   `Tools/DataModelConsole/deploy/buildspec.yml`.
2. Source the generated `console-images.env`.
3. Deploy with `Tools/DataModelConsole/deploy/apply.sh`; do not apply the
   manifest directory directly.

Reasoning stats and scene-search rows are materialized only after
`wf_publish_full_run_overlays` succeeds. Take its `manifest_key` and
`manifest_sha256` outputs and launch the separately guarded one-shot Job:

```bash
source console-images.env
export EXPECTED_AWS_ACCOUNT_ID=<PLATFORM_ACCOUNT_ID>
export PUBLISHED_DATASET=kitscenes
export DATASET_VERSION=v2.1
export MANIFEST_KEY=kitscenes/v2.1/shards/manifest.json
export MANIFEST_SHA256=<FLYTE_OUTPUT>
export CONFIRM_PRODUCTION_MATERIALIZATION=yes
Tools/DataModelConsole/deploy/run-reasoning-materialization.sh
```

The launcher verifies the AWS account, EKS context, API image digest, and the
SHA-256 of the published S3 manifest before creating the Job. It is
intentionally not part of `apply.sh`.

## Data Sources (Read-Only)

| Source | What | Access |
|--------|------|--------|
| S3 datasets bucket | WebDataset shards (L2D, NVIDIA) | Pod Identity |
| S3 datasets bucket | Published shards with embedded reasoning JSON | Pod Identity |
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
│   ├── jobs/     # Explicit one-shot production Job templates
│   └── terraform/# CloudFront, SG, IAM
├── docs/         # Design documents
└── README.md
```
