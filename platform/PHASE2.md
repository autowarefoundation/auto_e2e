# AutoE2E Phase 2 Platform Design

Status: APPROVED FOR IMPLEMENTATION. Account `<ACCOUNT_ID>` (platform account,
`--profile autowarefoundation`), region us-west-2, cluster `auto-e2e-platform`
(EKS Auto Mode). Reviewed against current repo state (`platform/infra`,
`platform/k8s`, `Model/training/train.py`).

Phase 2 goal: a team launches training jobs (dataset + backbone + fusion-mode +
hyperparams) from a UI, jobs are queued on the GPU, and results are tracked.

Stack: Flyte (UI + orchestration + sweep) → Kueue (GPU queue/quota) → Kubeflow
Training Operator v1 (PyTorchJob) → warm g6e node. MLflow for experiment tracking
+ Model Registry. Ray is NOT used.

## User-confirmed decisions (override research defaults)

1. **Kueue: 2-tier priority** — `research-low` (default, sweeps) vs
   `production-high` (preempts research on the single GPU). Adopted in §5.5.
2. **Namespaces: per-component separation** — `flyte`, `mlflow`, `kueue-system`,
   `kubeflow` (training-operator), `auto-e2e-training` (training pods). §4.
3. **RDS: provision via Terraform, sized large** — `db.r6g.large` (not the
   research-default t4g.medium), Postgres 16.x, one instance hosting both
   `flyteadmin` and `mlflow` databases. §3.1.
4. **UI exposure: CloudFront + Cognito** — full authenticated internet exposure
   in Phase 2. Since EKS Auto Mode ALB has NO native OIDC, the chain is
   internal ALB → CloudFront (OAC) → Cognito (via CloudFront, e.g. Lambda@Edge
   or hosted UI). §3.3 + §12.

## 0. Key corrections from adversarial verification (load-bearing)

1. **cert-manager NOT required.** Kueue (`internalCertManagement.enable: true`),
   Training Operator v1.9.3, and Flyte (`flyte-pod-webhook` self-signed init)
   all self-generate webhook TLS. No cert-manager install.
2. **Training Operator v1 (`kubeflow.org/v1 PyTorchJob`), pinned v1.9.3** — NOT
   Trainer v2 (`TrainJob`, alpha). Flyte's kfpytorch backend builds v1.
3. **No NVIDIA device plugin / GPU Operator install.** Auto Mode's Bottlerocket
   AMI bundles driver+toolkit+device plugin; `nvidia.com/gpu` already allocatable
   (proven on the warm g6e node). DCGM-exporter is a separate, deferred install.
4. **Flyte → Kueue queue-name label gap is CLOSED** (flyte PR #6421 + flytekit
   PR #3243, merged 2025-06-25). flytekit 1.16.23 `@task(labels=...)` lands the
   label on the PyTorchJob object. Add a namespace-default LocalQueue safety net.
5. **EKS Auto Mode ALB has no OIDC auth-action and no default StorageClass.**
   Must create a StorageClass before any PVC chart; auth lives in CloudFront.
6. **pause-pod headroom confirmed**: gpu-node-keeper requests cpu 10m / mem 16Mi
   — negligible vs the 2 vCPU / 8 GiB Kueue headroom. nominalQuota stands.

## 1. Component inventory

| Component | Artifact (repo + version) | Namespace | Provides |
|---|---|---|---|
| Kueue | Helm OCI `oci://registry.k8s.io/kueue/charts/kueue` v0.18.1 | `kueue-system` | GPU quota/queue admission. CRDs ResourceFlavor/ClusterQueue/LocalQueue/WorkloadPriorityClass (`kueue.x-k8s.io/v1beta2`). |
| Kubeflow Training Operator v1 | kustomize `github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.9.3` (no maintained v1 Helm chart) | `kubeflow` | `kubeflow.org/v1 PyTorchJob` CRD + controller. |
| Flyte | Helm `flyteorg/flyte-binary` (chart v0.1.10, appVersion 1.16.0 — confirm tag via `helm search repo flyteorg/flyte-binary --versions`) | `flyte` | UI + control/data plane single binary; renders LaunchPlan forms; emits PyTorchJob via kfpytorch plugin. |
| MLflow | Helm `community-charts/mlflow` (chart 1.8.5, MLflow 3.13.0 — verify keys via `helm show values`) | `mlflow` | Tracking + Model Registry; RDS backend, S3 server-proxied artifacts. |
| cert-manager | NOT INSTALLED | — | n/a |
| RDS Postgres | Terraform `aws_db_instance`, Postgres 16.x, **db.r6g.large**, single-AZ, 100 GB gp3, private subnets | (AWS) | DBs `flyteadmin` + `mlflow`. |
| EBS StorageClass | manifest `auto-ebs-sc`, provisioner `ebs.csi.eks.amazonaws.com`, default | (cluster) | Default SC so PVCs bind. |
| flytekit + plugins | `flytekit==1.16.23`, `flytekitplugins-kfpytorch==1.16.23`, `mlflow-skinny` (in training image) | (image) | SDK, PyTorch task config, registration. |

## 2. Install-order DAG

```
Phase 1 (DONE): EKS Auto Mode + GPU NodePool (taint nvidia.com/gpu:NoSchedule,
                label workload-type=gpu-training) + warm g6e + OIDC + s3-access IRSA
                                   │
   (0) StorageClass auto-ebs-sc    (1a) RDS Postgres        (1b) IRSA roles
       (before any PVC chart)          (flyteadmin+mlflow DBs)    (flyte-backend, mlflow-server)
                                   │
   (2) Training Operator v1  ──→  (3) Kueue (Helm, framework kubeflow.org/pytorchjob)
       (CRD must exist first)          │
                                   (5) Kueue objects: ResourceFlavor + ClusterQueue
                                       + WorkloadPriorityClass + LocalQueue
   (4) MLflow (Helm) — independent leaf (needs RDS + mlflow IRSA)
                                   │
   (6) Flyte (flyte-binary Helm) — LAST. depends_on RDS + backend IRSA +
       Training Operator CRD + Kueue objects. Enables `pytorch` task plugin.
                                   │
   (7) Internal ALB Ingress (flyteconsole + MLflow) → (8) CloudFront + Cognito
```

Terraform mapping: (0) `kubernetes_manifest`; (1a) `aws_db_instance`; (1b)
`aws_iam_role` x2; (2) `null_resource` kustomize apply; (3) `helm_release.kueue`;
(4) `helm_release.mlflow`; (5) `kubernetes_manifest` x N; (6)
`helm_release.flyte_binary`; (7) `kubernetes_manifest` Ingress; (8)
`aws_cloudfront_distribution` + `aws_cognito_user_pool`.
depends_on: `(1a),(1b),(0)→(4)`; `(2)→(3)→(5)`; `(1a),(1b),(2),(5)→(6)`; `(6)→(7)→(8)`.

## 3. AWS resources to add (Terraform)

### 3.1 RDS Postgres (`modules/rds`)
- `aws_db_subnet_group` over `module.vpc.private_subnet_ids`.
- `aws_security_group` allowing 5432 from the EKS cluster SG only.
- `aws_db_instance`: Postgres 16.x, **db.r6g.large**, 100 GB gp3,
  `multi_az=false` (Phase 2), `storage_encrypted=true`,
  `deletion_protection=true`, `backup_retention_period=7`; master creds via
  `random_password` → Secrets Manager.
- Two DBs `flyteadmin` + `mlflow` (second created via one-shot job/psql; each app
  runs its own migrations).
- Creds → K8s Secrets `flyte-db-pass` (ns `flyte`), `mlflow-db-secret` (ns `mlflow`).

### 3.2 IRSA roles (extend `modules/storage`)
- (a) `auto-e2e-platform-flyte-backend` — trust `system:serviceaccount:flyte:flyte-backend-flyte-binary`; S3 R/W on `artifacts` bucket (prefixes `flyte/metadata`, `flyte/raw`).
- (b) `auto-e2e-platform-mlflow-server` — trust `system:serviceaccount:mlflow:mlflow`; S3 R/W on `artifacts` bucket prefix `mlflow/`.
- (c) training pods — reuse existing `auto-e2e-platform-s3-access` (R/W datasets/checkpoints/artifacts). Tighten trust from `*:*` to `auto-e2e-training:*` + Flyte project-domain namespaces post-Phase-2.

### 3.3 UI exposure: internal ALB → CloudFront → Cognito
- `IngressClassParams` `internal-alb` (`scheme: internal`, private subnets, shared `group.name`); `IngressClass` `internal-alb`.
- Ingress for flyteconsole (HTTP 8088 + gRPC 8089, `backend-protocol-version: GRPC` on gRPC path) and MLflow (HTTP 5000).
- **CloudFront** distribution fronting the internal ALB (via VPC origin / PrivateLink), with **Cognito** auth (hosted UI + Lambda@Edge or CloudFront auth). WAF on CloudFront.
- Phase 2 bring-up order: port-forward → internal ALB → CloudFront+Cognito (do not block first end-to-end run on the CDN layer).

## 4. Namespace layout

| Namespace | Contents |
|---|---|
| `flyte` | flyte-binary pod (admin/propeller/console/webhook) + SA `flyte-backend-flyte-binary` (IRSA → backend role) |
| `kueue-system` | Kueue controller + webhook (ClusterQueue/ResourceFlavor/WorkloadPriorityClass are cluster-scoped) |
| `kubeflow` | Training Operator v1 controller (CPU only) |
| `mlflow` | MLflow server + SA `mlflow` (IRSA → mlflow-server role); bundled Postgres disabled |
| `auto-e2e-training` | PyTorchJobs + training pods; `LocalQueue gpu-queue`; `training-sa` (IRSA → reused s3-access) |

LocalQueue is namespaced → must co-locate with the PyTorchJob. Control-plane pods
stay on the un-tainted `general-purpose` Auto Mode NodePool (no GPU toleration),
never on the warm g6e.

## 5. Kueue config (1x g6e.4xlarge = 16 vCPU / 128 GiB / 1 L40S)

### 5.1 ResourceFlavor
```yaml
apiVersion: kueue.x-k8s.io/v1beta2
kind: ResourceFlavor
metadata: { name: g6e-l40s }
spec:
  nodeLabels:
    eks.amazonaws.com/instance-gpu-name: l40s
    node.kubernetes.io/instance-type: g6e.4xlarge
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
```

### 5.2 ClusterQueue (+ preemption)
```yaml
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata: { name: gpu-cq }
spec:
  namespaceSelector:
    matchLabels: { kubernetes.io/metadata.name: auto-e2e-training }
  resourceGroups:
    - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
      flavors:
        - name: g6e-l40s
          resources:
            - { name: "cpu", nominalQuota: "14" }
            - { name: "memory", nominalQuota: "120Gi" }
            - { name: "nvidia.com/gpu", nominalQuota: "1" }
  preemption:
    withinClusterQueue: LowerPriority
    reclaimWithinCohort: Never
```

### 5.3 LocalQueue
```yaml
apiVersion: kueue.x-k8s.io/v1beta2
kind: LocalQueue
metadata: { name: gpu-queue, namespace: auto-e2e-training }
spec: { clusterQueue: gpu-cq }
```

### 5.4 Safety net
flytekit 1.16.23 propagates the queue-name label, but also label the
`auto-e2e-training` namespace for Kueue's default-LocalQueue, or set
`manageJobsWithoutQueueName: true` scoped via `managedJobsNamespaceSelector` to
`auto-e2e-training`. Validate exact 0.18.1 keys on-cluster.

### 5.5 Two-tier WorkloadPriorityClass
```yaml
apiVersion: kueue.x-k8s.io/v1beta2
kind: WorkloadPriorityClass
metadata: { name: research-low }
value: 1000
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: WorkloadPriorityClass
metadata: { name: production-high }
value: 10000
```
Attach via label `kueue.x-k8s.io/priority-class`. With `withinClusterQueue:
LowerPriority`, production evicts a running research job for the single GPU.

### 5.6 Kueue Configuration (framework enablement)
```yaml
apiVersion: config.kueue.x-k8s.io/v1beta2
kind: Configuration
integrations:
  frameworks:
    - "batch/job"
    - "kubeflow.org/pytorchjob"
```

## 6. PyTorchJob template (single-node single-GPU)

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: train-swin-concat
  namespace: auto-e2e-training
  labels:
    kueue.x-k8s.io/queue-name: gpu-queue
    kueue.x-k8s.io/priority-class: research-low
spec:
  runPolicy: { cleanPodPolicy: None, ttlSecondsAfterFinished: 86400 }
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        metadata:
          annotations: { karpenter.sh/do-not-disrupt: "true" }
        spec:
          serviceAccountName: training-sa
          nodeSelector: { workload-type: gpu-training }
          tolerations:
            - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
          containers:
            - name: pytorch
              image: <ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/auto-e2e/training:latest
              imagePullPolicy: Always
              command: ["python", "Model/training/train.py"]
              args: ["--backbone=swin_v2_tiny","--fusion-mode=concat","--batch-size=8","--epochs=20","--amp","--save-dir=/tmp/ckpt"]
              env:
                - { name: MLFLOW_TRACKING_URI, value: "http://mlflow.mlflow.svc.cluster.local:5000" }
                - { name: AWS_DEFAULT_REGION, value: "us-west-2" }
              ports: [{ containerPort: 23456, name: pytorchjob-port }]
              resources:
                requests: { cpu: "6", memory: 40Gi, nvidia.com/gpu: 1 }
                limits: { nvidia.com/gpu: 1 }
```

## 7. Flyte (chart, plugin, @task→PyTorchJob, enum dropdowns, sweep)

- Chart `flyteorg/flyte-binary`. Enable the `pytorch` task plugin in inline config
  (`enabled-plugins: [container, sidecar, k8s-array, pytorch]`,
  `default-for-task-types: [..., pytorch: pytorch]`), wire RDS + S3 + backend SA IRSA.
- flytepropeller renders+applies a `kubeflow.org/v1 PyTorchJob` into
  `auto-e2e-training`; Kueue webhook suspends → admits on GPU quota; Training
  Operator materializes pods.
- Enum dropdowns: build `Enum` from `BACKBONE_REGISTRY`/`FUSION_REGISTRY` (str
  members). Materialized at registration → re-register when a registry key changes.
- Sweep: `@dynamic` loop over combos (robust); try `map_task` on 1.16.23 and fall
  back. Kueue serializes at 1-GPU quota.
- Skeleton `workflow.py` in §7.5 of the research doc (kfpytorch `PyTorch` task with
  Kueue labels + do-not-disrupt annotation).

## 8. MLflow (chart, RDS, S3, train.py changes, registry)

- community-charts/mlflow 1.8.5: RDS backend (`existingDatabaseSecret:
  mlflow-db-secret`), **server-proxied artifacts** (`proxiedArtifactStorage: true`,
  `--serve-artifacts`) on S3 `artifacts` bucket prefix `mlflow/`, SA `mlflow` IRSA.
  Bundled Postgres disabled.
- Server-proxied = only the MLflow SA needs S3; training pods need only
  `MLFLOW_TRACKING_URI`.
- **train.py minimal additive changes** (all gated, default-off, smoke-test stays
  byte-identical):
  - `--dataset` arg (maps to `--repo-id` when not overridden).
  - MLflow block: lazy `import mlflow`, active only when `MLFLOW_TRACKING_URI` set
    AND not `--smoke-test`; `start_run` + `log_params` + per-step `log_metric` +
    `log_artifact(ckpt)`; `--register-model` → `mlflow.pytorch.log_model(...,
    registered_model_name="auto_e2e")`.
  - Optional DDP scaffolding gated on `LOCAL_RANK` (off for single-GPU).
  - Add `mlflow-skinny`, `flytekit==1.16.23`, `flytekitplugins-kfpytorch==1.16.23`
    to the training image.
- Model Registry: use **aliases** (`staging`, `champion`), NOT deprecated stages.

## 9. End-to-end sequences

Single run: Flyte UI form → flyteadmin (RDS) → flytepropeller → PyTorchJob (labeled)
→ Kueue suspend/admit on GPU quota → Training Operator → Master pod on warm g6e →
train.py builds from registries, logs to MLflow, checkpoint to S3 → optional model
register → PyTorchJob Succeeded → GPU freed → green in console.

Sweep: Flyte fans out N combos → N PyTorchJobs → all queue on `gpu-queue` → Kueue
admits one at a time (production-high preempts research-low) → N MLflow runs →
compare in MLflow UI. Raise parallelism later by bumping ClusterQueue GPU quota +
letting Karpenter scale.

## 10. Implementation TODO (ordered)

1. StorageClass `auto-ebs-sc` (gp3, default, encrypted, WaitForFirstConsumer).
2. `modules/rds`: subnet group, SG (5432 from cluster SG), `db.r6g.large` Postgres
   16.x; DBs `flyteadmin`+`mlflow`; creds → Secrets Manager.
3. IRSA: add `flyte-backend` + `mlflow-server`; reuse `s3-access` for training.
4. Namespaces `flyte`/`kueue-system`/`kubeflow`/`mlflow`/`auto-e2e-training`; label
   `auto-e2e-training`; Secrets `flyte-db-pass`,`mlflow-db-secret`; SA `training-sa`.
5. Training Operator v1.9.3 (kustomize via null_resource).
6. Kueue 0.18.1 (Helm) with `kubeflow.org/pytorchjob` framework.
7. Kueue objects: ResourceFlavor/ClusterQueue/LocalQueue/2x WorkloadPriorityClass.
8. MLflow 1.8.5 (Helm): RDS + server-proxied S3 + IRSA.
9. Flyte flyte-binary (Helm): RDS + S3 + backend IRSA + pytorch plugin.
10. train.py changes (--dataset, MLflow block, --register-model, DDP scaffold);
    extend training image deps. Keep smoke-test + 92-test suite green (test on EC2).
11. workflow.py: dynamic enums, `train_one` kfpytorch task w/ Kueue labels,
    `train_single`, `sweep` via `@dynamic`.
12. Build/push image; `pyflyte register`; smoke run → 1-epoch real run; verify
    PyTorchJob → Kueue admit → pod on g6e → MLflow run+artifact.
13. Verify Flyte→Kueue label on the PyTorchJob object; enable safety net if absent.
14. Internal ALB Ingress (flyteconsole + MLflow).
15. CloudFront + Cognito in front of the internal ALB; WAF.
16. (Deferred) DCGM-exporter PoC under Bottlerocket SELinux.

## 11. Resolved / remaining decisions

Resolved by user: 2-tier priority (§5.5), per-component namespaces (§4), RDS
db.r6g.large single instance (§3.1), CloudFront+Cognito UI exposure (§3.3/§12).
pause-pod headroom confirmed (§0.6).

Remaining to confirm during implementation:
1. Flyte stays on 1.x line (recommended — everything verified targets it).
2. `map_task` vs `@dynamic` for sweep (use `@dynamic`, try `map_task` on-cluster).
3. GPU concurrency: keep `nvidia.com/gpu=1` (serial sweeps, cost control) vs allow
   burst to multiple g6e nodes by raising quota.
4. In-container entrypoint: `train_one` shells `python Model/training/train.py ...`
   vs imports `run_training` directly.

## 12. CloudFront + Cognito notes (Phase 2 §3.3 detail)

EKS Auto Mode ALB cannot do OIDC. Therefore:
- internal ALB terminates inside the VPC (private subnets only).
- CloudFront fronts it via a VPC origin (CloudFront → private ALB through
  PrivateLink/VPC origin), so the ALB is never internet-facing.
- Authentication via Cognito user pool: either CloudFront + Lambda@Edge validating
  Cognito tokens, or Cognito hosted UI redirect. WAF attached to CloudFront.
- This satisfies "ALB not exposed to internet; access only via CloudFront" and
  "Cognito auth" from the platform-wide network policy.
