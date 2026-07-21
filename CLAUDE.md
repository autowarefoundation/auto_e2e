# AutoE2E Project Context

## Repository

- 本家 (Owner): `autowarefoundation/auto_e2e` — riita は Autoware Foundation 側の Owner
- Fork: `riita10069/auto_e2e`
- Remote `origin`: autowarefoundation (本家), Remote `fork`: riita10069 (fork、push 用に残置)
- local `main` は `origin/main` (autowarefoundation) を追従

## EC2 開発環境

- Instance ID: `i-0e73ec8d6a7766395`
- Type: g6e.xlarge (NVIDIA L40S, 45GB VRAM, 4 vCPU, 32GB RAM)
- Region: us-east-1 / AZ: us-east-1d
- AMI: `ami-0b9c99b766a895d68` (Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.10, Ubuntu 24.04)
- Subnet: `subnet-01a03fafd1a02e56b` Private (NAT Gateway `nat-0c9117060e68b41f8` 経由、インターネット非公開)
- Security Group: `sg-0b7bf9f8af00fe089` (Inbound 全閉、Outbound 全開。接続は SSM のみで Inbound 不要)
- 接続: SSM Session Manager (`aws ssm start-session --target i-0e73ec8d6a7766395 --region us-east-1`)
- Python venv: `/opt/pytorch/bin/activate` (DLAMI 同梱。torch 2.10.0+cu130, bf16 対応。`.bashrc`/`.profile` で自動 activate 済み)
- リポジトリ: `/home/ubuntu/auto_e2e`
- IAM Role: `access_ssm_role_handson` (SSM + S3 ReadOnly)
- 導入済みツール: git, tmux, docker, make, gh, node v22, claude (Claude Code), timm, pytest

旧 g4dn 環境 (`i-091734cd9d85a78ce`, T4 16GB) は g6e へ置き換え。BEV 高解像度学習は T4 16GB では回らないため (bev 450x300 は batch 4 で 20GB 超)。

## AWS アカウント使い分け

2つの AWS アカウントを用途で使い分ける。

| 用途 | AWS Profile | Account ID | 備考 |
|---|---|---|---|
| EC2 開発環境 (g6e 単体GPU, モデル開発) | (default, profile 指定なし) | `833707099141` | `user/admin`。sync-to-ec2, SSM 接続 |
| Platform (EKS, MLOps 基盤) | `--profile autowarefoundation` | `381491877296` | `user/bedrock-engineer`。Terraform, EKS, ECR |

- EC2 上でのモデル開発・テスト: profile 指定なし (既存の g6e インスタンス)
- Platform インフラ構築 (Terraform, EKS, Karpenter 等): 必ず `--profile autowarefoundation` を付ける
- MUST: 2アカウントを混同しない。特に破壊的操作は profile を明示確認してから実行

## Cosmos3-Nano ラベリングエンドポイント (別クラスタ・絶対に壊さない)

MUST: 以下の Cosmos クラスタを絶対に削除・破壊しない。`terraform destroy` /
`eks delete-cluster` / `delete nodegroup` / `delete ingress` 等を私の判断で実行しない。
これは reasoning ラベリングの teacher (Cosmos3-Nano vLLM) を提供する高コスト GPU 資産で、
ユーザーの明示承認なしに触ってはいけない。

- 所属: **default profile / account `833707099141` / region `us-west-2`** (Platform とは別アカウント)。
  Terraform 管理リポジトリは `/Users/riita/code/github.com/riita10069/cosmos3-reasoner-nim-endpoint-terraform`
  (実 ID・状態は同リポジトリ `.secrets/ENV.md` が正)。
- クラスタ名: `cosmos3-vllm-poc` (EKS 1.34)、GPU node group `gpu-cosmos` (g6e.2xlarge, min1/max10)。
  namespace `cosmos` に Deployment `cosmos3-nano-vllm` (vLLM)。
- Flyte から使うエンドポイント: K8s Secret `auto-e2e-development/cosmos-teacher` の
  `COSMOS_TEACHER_BASE_URL` (us-west-2 の internet-facing ALB :8000)。SG `/32` allowlist で
  Platform NAT EIP からのみ到達可。
- MUST: 存在確認は必ず **default profile + `--region us-west-2`** で行う
  (`AWS_PROFILE=default aws eks describe-cluster --name cosmos3-vllm-poc --region us-west-2`)。
  `--profile autowarefoundation` や別リージョンで見ると「無い」ように見えるが、それは
  探す場所を間違えているだけ。「見つからない=壊れている」と絶対に即断しない。まず
  profile/region を疑い、ENV.md を確認する。

## Platform (ADAS MLOps)

- Account: `381491877296` (`--profile autowarefoundation`)
- Region: `us-west-2` (us-east-1 は g6e キャパ全滅のため移行)
- EKS Auto Mode クラスタ: `auto-e2e-platform`
- GPU: g6e.4xlarge (L40S) primary / burst と g6e.8xlarge (L40S) Smoke の ODCR 確保済み
- EKS 上で Training と Inference をコンテナで実行する構成
- コンテナ必須: 学習も推論も Docker image を ECR に push して K8s Job/Deployment で実行
- Terraform で IaC 管理 (`platform/infra/`)
- Flyte でパイプラインオーケストレーション (`platform/pipelines/`)

主要コンポーネント:
- EKS Auto Mode + Managed Karpenter (GPU は warm node 1台を do-not-disrupt で維持。scale-to-zero ではない)
- device plugin は入れない (Auto Mode の Bottlerocket-NVIDIA AMI が nvidia.com/gpu を提供)
- Kueue (GPU ジョブキュー、2段階優先度 research/production)
- Kubeflow Training Operator v1 (PyTorchJob)
- Flyte (データ前処理 → 学習 → 評価 → closed-loop sim、UI から起動 + sweep)
- MLflow (実験管理 + Model Registry、RDS backend)
- LakeFS (データバージョニング on S3)
- KServe + Triton (推論評価)
- CARLA (将来 closed-loop sim)
- Prometheus + DCGM Exporter + Grafana (GPU 監視)
- Kubecost (コスト可視化)

## Platform as-built リソース一覧 (現状スナップショット)

MUST: Platform のインフラを変更したら、このセクションを必ず更新する。次のセッション
が「今何が動いているか」を最初に把握する唯一の信頼できる場所。実 ID はここに書く
(CLAUDE.md はグローバル gitignore 対象なので git には入らない)。

最終更新: 2026-07-21 (KITScenes v42 FullSet trajectory overlay 公開・本番検証)

| リソース | 値 | 確認コマンド (要 `--profile autowarefoundation --region us-west-2`) |
|---|---|---|
| EKS クラスタ | `auto-e2e-platform` (Auto Mode) | `aws eks describe-cluster --name auto-e2e-platform` |
| Region / AZ | us-west-2 / GPU は us-west-2b (primary) と us-west-2c (burst) | — |
| GPU ノード | primary `gpu-training`: g6e.4xlarge (L40S 44.7GB), us-west-2b warm・最大1台、現 node `i-04988fd4fa8818e94` は legacy recovery Training `aqcl4nk6zz7jfsgzt79b` が使用中。concurrent Training 用 `gpu-burst`: g6e.4xlarge, us-west-2c, On-Demand、最大1 node、現在0 node。corrected Full Training 用 `gpu-smoke`: g6e.8xlarge (同じ L40S 44.7GB), us-west-2b, On-Demand、最大1 node、現 node `i-0e50da0d552756f35` は `avp4n785pdfdfnjgzd7t` が使用中 | `kubectl get nodes -L workload-type -l 'workload-type in (gpu-training,gpu-burst,gpu-smoke)'` |
| ODCR | primary `cr-0907753c65fac8fa6` (g6e.4xlarge @ us-west-2b)、burst `cr-006a717f30ff05884` (g6e.4xlarge @ us-west-2c)、Smoke `cr-0ca944fbb0a12f664` (g6e.8xlarge @ us-west-2b)、すべてopen/unlimited | `aws ec2 describe-capacity-reservations --capacity-reservation-ids cr-0907753c65fac8fa6 cr-006a717f30ff05884 cr-0ca944fbb0a12f664` |
| Concurrent GPU CR Fleet | `crf-05e90b46e60de071e`。fulfilled 0のまま通常On-Demand nodeを確保できたためcancelled | `aws ec2 describe-capacity-reservation-fleets --capacity-reservation-fleet-ids crf-05e90b46e60de071e` |
| S3 datasets | `auto-e2e-platform-datasets-381491877296` | `aws s3 ls` |
| S3 checkpoints | `auto-e2e-platform-checkpoints-381491877296` (versioned) | — |
| S3 artifacts | `auto-e2e-platform-artifacts-381491877296` | — |
| ECR training | `381491877296.dkr.ecr.us-west-2.amazonaws.com/auto-e2e/training` | `aws ecr describe-images --repository-name auto-e2e/training` |
| ECR data-prep / eval | 同上 `/auto-e2e/data-prep`, `/auto-e2e/eval` | — |
| Terraform state | s3://auto-e2e-platform-tfstate (us-east-1) + DynamoDB `auto-e2e-platform-tflock` | — |
| Pod Identity (S3) | role `auto-e2e-platform-s3-access` + associations: training-sa, flyte, mlflow | `aws eks list-pod-identity-associations --cluster-name auto-e2e-platform` |
| RDS Postgres | `auto-e2e-platform-pg.cteeaycygoyj.us-west-2.rds.amazonaws.com` db.r6g.large | `aws rds describe-db-instances --db-instance-identifier auto-e2e-platform-pg` |
| Kueue | v0.18.1, ClusterQueue `gpu-cq`, LocalQueue `gpu-queue` | `kubectl get clusterqueues; kubectl get localqueues -A` |
| Training Operator | v1.9.3, namespace `kubeflow` | `kubectl get pods -n kubeflow` |
| MLflow | v3.13, namespace `mlflow`, RDS backend, S3 proxied artifacts | `kubectl get pods -n mlflow` |
| Flyte | v2.0.23 binary, namespace `flyte`。development project quota は requests/limits とも 1000 CPU / 8Ti。`flyte-propeller-config` / `flyte-admin-base-config` / `datacatalog-config` の storage auth は Terraform の `flyte_storage_accesskey_patch` で `accesskey` に統一。task role `flyte-user-role` は datasets bucket と checkpoints bucket の immutable Get/Put および bucket-level `ListBucket` を持つ。VPC-local trajectory overlay launcher は CodeBuild `auto-e2e-platform-overlay-launch` | `kubectl get pods -n flyte; kubectl get resourcequota project-quota -n auto-e2e-development; aws codebuild batch-get-projects --names auto-e2e-platform-overlay-launch` |
| DynamoDB (DataModelConsole) | table `auto-e2e-console` (PAY_PER_REQUEST, pk/sk + GSI `gsi1`)。Console の shard-index キャッシュ / reasoning 統計 / label→scene 検索インデックス。`Tools/DataModelConsole` 用 | `aws dynamodb describe-table --table-name auto-e2e-console` |
| Console (DataModelConsole) | namespace `console`。Deployment `console-api` (Go, :8080) + `console-web` (Next.js, :3000) 各 replica 2。`console-api@sha256:e9d76cadc65e610bd0d64e7549bd78f87e50c351316354e726dbcf27886463f6` / `console-web@sha256:93f4c3c493bf4dd2620dfb68716fb0c2eeacad676030a47d4043d06dacf8c003` を 2/2 Ready で稼働。CodeBuild #95 `2a0efd86-96ab-481f-9ac1-eba79f6d8257`、source commit `68b2c72dc20f241f14e6f72506ba565eaaea97c8`。exact geography は `EXACT_GEO_ENABLED=false` | `kubectl -n console get deploy,ingress` |
| Console ALB | 内部 ALB `k8s-console-consolei-ca46c3828b` (HTTP:80, internal, 3 internal-elb subnet)。frontend SG `sg-08a5e36f2af12c836` は CloudFront-VPCOrigins-Service-SG (`sg-09f2833c5e06b2da1`) からのみ :80 許可 | `kubectl -n console get ingress console-ingress` |
| Console CloudFront | `E3U5B9CYT93SXP` → https://d2itskdqq39tx1.cloudfront.net (default cert, VPC origin → 内部 ALB, CachingDisabled)。viewer は HTTPS、CF→ALB は http-only。KITScenes FullSet playback / reasoning を desktop・mobile で本番検証済み (2026-07-17) | `aws cloudfront get-distribution --id E3U5B9CYT93SXP` |
| Console Pod Identity | role `auto-e2e-platform-console-api` (S3 RO datasets/artifacts + DynamoDB `auto-e2e-console`)。SA `console/console-api` に紐付け | `aws eks list-pod-identity-associations --cluster-name auto-e2e-platform` |
| Console infra Terraform | `Platform/infra-console/` (**独立 state** key `infra-console/terraform.tfstate`、既存 Platform を壊さない)。2-phase apply (SG+IAM → k8s Ingress → CloudFront)。認証 (Cognito/Lambda@Edge/ACM) は現状なし | README 参照 |

kubeconfig 取得: `aws eks update-kubeconfig --name auto-e2e-platform --region us-west-2 --profile autowarefoundation`

クラスタ稼働確認の定番:
```bash
kubectl get nodes -l workload-type=gpu-training        # warm g6e が Ready か
kubectl get nodepools                                  # gpu-training NodePool
kubectl get pods -l app=gpu-node-keeper                # warm node 維持の pause Pod
```

Phase 2 完了 (PHASE2.md §1-13): Kueue / Training Operator / Flyte / MLflow / RDS
がデプロイ済み。PyTorchJob smoke-test で end-to-end 動作確認済み (Kueue admit →
Training Operator → GPU pod → cuda bf16 完了)。

Phase 3 完了: Data Pipeline 実装済み + E2E 動作確認済み。
- IngestAdapter protocol + L2D/NVIDIA adapters
- WebDataset shard packing + PreExtractedDataset (DataLoader)
- data-prep image: ECR push 成功 (Python 3.12 + torch CPU + lerobot --no-deps + av)
- CodeBuild: training + data-prep 両イメージビルド自動化
- L2D 実データ ingest: 7カメラ PyAV decode → shard → S3 → GPU training 成功

#121 KITScenes full-run:
- data-prep image digest: `sha256:e486a9985a4d3daf223596d93a13c6bf4a40fe79349e14f9fa7c63aa7bfbf148`
- image build: `auto-e2e-platform-build-images:1cb332e7-06c8-43e6-8b90-63d92946e3ff`
- Flyte registration: `auto-e2e-platform-flyte-register:3596c1d4-f1fd-4396-8d10-3cabebe5e1d3`
- 10-scene smoke: `at5t76tjc8ljdxphbvrz` SUCCEEDED、32/32 Pods、eviction 0。checkpoint SHA-256
  `d611b1bdb40ef28b59bfb1890523bb04b2513bf569b01202754cc52929940129`、
  ADE 8.22296 / FDE 19.33854 (1 epoch smoke の quality gate は FAIL)
- 533/534 available-scene full-run: `a5nk56zsbcg7g6npjfpt`、CodeBuild launcher
  `auto-e2e-platform-flyte-register:b7cf9748-d433-4797-aeea-626e6e3c376a`。
  1 scene/pod、ingest/pack concurrency 60、label concurrency 5 x 2 workers、3 epochs
- full-run 中に DataCatalog の cache metadata S3 write が IAM auth で 403 になる不具合を検出。
  commit `f715acf` で IaC patch 対象へ DataCatalog を追加し、targeted replacement apply 済み。
  3 ConfigMap の `accesskey` と新規 cache metadata write を確認
- full-run `a5nk56zsbcg7g6npjfpt` は raw 533 scene と Cosmos label 4,598件
  (404 non-empty / 129 empty) まで完了。Training は404 partition分の persistent
  DataLoader workerを同時保持して16 GiB OOMKilled。旧packはmap座標系修正前のため再利用しない
- recovery manifest は
  `s3://auto-e2e-platform-checkpoints-381491877296/recovery-manifests/24afc76fc07d74023fa2acc3ee17868dcae5e306be09cbf7350032b636304a12.json`
  (VersionId `zpW2cAkwaFhEFOcUs7m5KI7uW.0KV33A`)。Ingest/Cosmosを呼ばず、
  geometry v2で533 sceneをrepackしてからtrain/evalする
- recovery Training execution `aqcl4nk6zz7jfsgzt79b` は MLflow run
  `35494304ba8a44369a85a10bfb86aabd` で継続中。epoch 8 checkpoint
  まで保存済み。best pointer は epoch 6
  (`7ac460edee09bd27077c813e84a6d2dae29450ae84938ad2acf9da0fdd7491be`)、
  ADE 7.37362 / FDE 23.61565。この execution は dataset policy 導入前の
  contract/image なので checkpoint の意味を新 policy で再解釈しない。
- KITScenes dataset policy 修正版は commit `edc9f808fa09`。
  image build `auto-e2e-platform-build-images:31baafff-9006-491e-b7b3-1a07dcb72e70`、
  tag `kitscenes-policy-edc9f808fa09`。training/eval/offline digest は
  `sha256:b0e3042b4ccd839f5a0114dfed37d4a49a313949ccf9c1f6c888feac142230d4`、
  data-prep digest は
  `sha256:59361f5de1b2f9f83b1389a2bfa2cec9b5f72b8bb75b62775aaebf8fd10d6017`。
  Flyte registration `auto-e2e-platform-flyte-register:99083301-2c33-47c0-9ae0-3aefe5df8954`,
  version `kZHXXWt_VkGC3iSjQRHNVQ`。
- corrected KITScenes recovery execution `a82m78t5w7qqfgmw22gs` は launcher
  `auto-e2e-platform-flyte-register:746ed7ec-fbec-4535-bc8b-bba69ae4463d`。
  audited recovery manifest から 533 partition graph を生成済みで、ingest /
  Cosmos teacher は未使用。Training pod は上記 immutable training digest で
  3台目GPUへ配置されたが、execution は `ABORTED`。自動再起動しない。
- dataset-policy 修正版の最初の 10-scene / 1-epoch Training Smoke は CodeBuild
  `auto-e2e-platform-flyte-register:23549299-aedc-4e20-8af0-bdf4f8f43c47`
  / Flyte `ajjgz774t4fqjjsjnj5z`。全量 frozen split を10-scene subsetへ
  強制したため、optimizer step前に partition count mismatch で FAILED。
- subset split 修正版は commit `cea84f7abd2396ab588036db5ebfab05edb70fe1`。
  image build `auto-e2e-platform-build-images:b8995c07-9ed6-4518-bf72-355bf33720bf`、
  tag `kitscenes-subset-cea84f7`。training/eval/offline digest は
  `sha256:b48fd049ff6d404410c28e2021800a4df5a84c0efe6eabcc65f1110d76a891ff`、
  data-prep digest は
  `sha256:e131217ddecfb2f123f5277e6dbe736a63802e580e463189c204f1820785f087`。
  Flyte registration `auto-e2e-platform-flyte-register:297b4d4e-a528-4452-8b54-7f472bf5fd4a`,
  version `kdidihA_FC_mcsDjY9l9pA`。
- fixed 10-scene Smoke は launcher
  `auto-e2e-platform-flyte-register:0f572b99-ad7c-4ab7-a5d4-8c1889760246` /
  Flyte `ajp786rtzx4wrf6sr48j` で SUCCEEDED (8m50s, restart 0)。
  10 partition中9 non-empty / 1 empty、708 samplesから deterministic holdout
  1 scene / 12 samplesを選択。validation sample digest は
  `f7e7e7bb6c28994f1042eb2eaa528a5548046131aeddfcae9f3410904d6d9e2a`。
  planner / world-model / reasoning の初回 grad norm は
  `17.7015 / 0.1031 / 0.5192`。epoch 1 loss は total `0.82770`、
  trajectory `0.20875`、JEPA `0.61178`、reasoning `0.14338`。
  MLflow run `9ffc075aaf1840a19bed1bf6fe78ffba` / model v39、checkpoint SHA-256
  `408fbce5148d0fdc77ca337bff3c2aedeff8997aa684eafb5b53afeab7c24e9b`、
  ADE 13.28822 / FDE 24.66942 (1 epoch Smoke の quality gate は FAIL)。
- corrected Full Training は launcher
  `auto-e2e-platform-flyte-register:454a1352-3feb-4674-9595-7e01b30c5a8c` /
  Flyte `avp4n785pdfdfnjgzd7t`。上記 audited recovery manifest と immutable
  image `kitscenes-subset-cea84f7` を使い、ingest / Cosmos teacherを再実行せず
  10 epochs、batch 1、grad accumulation 4、early stopping patience 3で起動。
  533/533 artifact materialization後、404 non-empty / 129 empty、
  42,667 samples、frozen validation 40 scenes / 3,820 samplesのfull corpus
  identityを検証済み。MLflow run `6535d0814d54475ab30f924311bb8ada`。
  `gpu-smoke` node `i-0e50da0d552756f35` 上でepoch 1を実行中 (restart 0)。
  初回 grad norm は planner `45.8614` / world-model `0.1095`。
  reasoningは最初の4 micro-batchに疎ラベルが無く `0.0` だったが、同一imageの
  10-scene Smokeでreasoning `0.5192`を確認済み。初回checkpointは未保存。
- checkpoint-only KITScenes retrospective evaluator
  `wf_evaluate_kitscenes_benchmark` を登録済み。checkpoint は `FlyteFile`、
  fixed-manifest shards は `List[FlyteDirectory]` で受け、同じ MLflow run の
  checkpoint epoch step へ3秒/5秒 ADE/FDEを追記する。Eval image tag
  `kitscenes-benchmark-0dc7680`、digest
  `sha256:8025a2d2ff0056856ed4f2e0b8183742658ff8e644b9c53828c6e9cabb21e724`。
  epoch 2 development smoke `at8hnqwzn9qgtkxtfmc4` は2 sampleで SUCCEEDED
  (ADE/FDE: 3s 1.31068/3.47087、5s 2.64854/5.50996)。これは公式 benchmark
  score ではない。epoch 3 verification `a7s9q9mbm77gh6k5bhm5` も2 sampleで
  SUCCEEDED (ADE/FDE: 3s 0.83018/2.32483、5s 2.51766/9.11954)。
  最新 Flyte registration build
  `c98ddbfe-9c87-4650-b78c-c2d5491fd33d`、version
  `ab2bvu1k8rMH2PfWRBEACg`。

Trajectory overlay production smoke:
- immutable image tag `trajectory-a7637cafc311`。eval/training/offline digest
  `sha256:7388f7a55031e5a65179a06e5ea2aed2fcb086a49b8c1f184e3999c23c5e16cf`、
  data-prep digest
  `sha256:c0c6cd044dc74243f4602552e39f61b3a06eee0c19aa9bfe134e873e41356fea`
- CodeBuild image build `d518503b-6c77-4353-b23f-fdc8fbfac51c`、Flyte registration
  `be3a1708-2480-44b5-9a97-4290281ee5d9`、VPC launcher
  `edd1ec79-ef12-43f5-a7c4-32ba166ba3b8`
- Flyte execution `a8h69js4kjdkhdq9g8v4` SUCCEEDED (217.6s)。
  published dataset `kitscenes-smoke-8aec8355b116/v2.1`、MLflow model v34 /
  checkpoint `d611b1bdb40ef28b59bfb1890523bb04b2513bf569b01202754cc52929940129`
- Dynamo `OVLSET` ready。overlay 1,069 bytes、SHA-256
  `524337c1...`。CloudFront 経由で model v34 と binary を取得し、
  desktop/mobile playback、7 camera canvas、3 overlay canvas を確認済み
- production coordinate `kitscenes/v2.2` の初回 execution
  `a457pf9mdbx748dm5j8b` は batch 32 / workers 4 で host memory OOM。
  batch 16 / workers 0 に下げた `abbbcwljggnwkchd5ms5` は SUCCEEDED。
  旧 recovery run `aqcl4nk6zz7jfsgzt79b` の epoch 5 checkpoint
  (`139e363e5034bbe64fda323bf850645e702d3c40f1c1d14ce218b3fefbb117df`、
  VersionId `S.D8gDjTpJKe9jsVlnAkfL5vgoPYha2V`) と eval image digest
  `sha256:9aeea0a6781973d97e8574760c3af15ada18139f522c7634e96ec64d8e40368f`
  を固定。MLflow model version `38`、ADE 7.53002 / FDE 20.83869
- FullSet overlay は404 shards / 42,667 samples / 20,904,379 bytes。
  DynamoDB `OVLSET` は `ready`。request identity
  `16a159f84ef28172119ea721c09a7c2421cb4dfa616f6a1d0bc3d5661a6b34ab`、
  cache identity
  `acfba7db3077c0879fa8cdac1cbe7dbf4f49d66108d95a1adadcd33271765557`。
  manifest key は
  `overlays_manifest/schema=v1/model=139e363e5034bbe64fda323bf850645e702d3c40f1c1d14ce218b3fefbb117df/dataset=kitscenes/version=v2.2/manifest.json`、
  SHA-256 は `90ac3bb6f4a77d6fa3679695f6125eda0fc5d7078f503104a59fa3e3ee00fb66`
- reasoning materialization Job `console-reasoning-materialization-rhj4m` は
  2026-07-17 15:40 JST に SUCCEEDED (restart 0)。DynamoDB
  `RINV5#kitscenes#v2.2` generation
  `103fea73e7bb8fc35c7ca6df2646360d8305db2cc12a8adef04921c6ebc1e893`
  に4,598 records / 32,579 scene rowsをpublish。dataset manifest SHA-256
  `2e2ea5722e993b505b03fc466e38a61da331253e06d08eb68cd46811dff8d798`
  と一致
- CloudFront 本番で dataset `kitscenes/v2.2` のみ、model v38、7 camera、
  BEV / camera trajectory、reasoning stats・search・single-label APIを検証。
  Playwright 1440x1000 / 390x844 で document横overflowなし、mobile filmstripは
  内部scroll、`/blob` request 0、HTTP / console / page error 0
- PR #74 style trajectory overlayを source commit `68b2c72` で本番deploy。
  front-center 3 sceneで紫Ground Truth / 緑PredictionのCanvas画素を検出し、
  desktop/mobileともHTTP / console / page error 0、横overflow 0を確認
- PR #136 は PR #135 の merge commit `c352db7` を取り込み、head
  `9d12b478381a62271ed377dfb8fea3487039416e`。GitHub CI 2件と DCO 成功後、
  merge commit `bcd05346af9ea0876e1d417448c75fb2d20e3da3` で main へ統合済み。
  ground-plane projection と Flyte task resolver の修正を含む。
- v42 overlay image build
  `auto-e2e-platform-build-images:35a5240f-457b-4390-b7bb-a756b037d094`、
  tag `trajectory-9d12b478381`。training/eval/offline digest
  `sha256:e66e5a5c80999c2ae7cc795050048ac4be91426ba79884d7b5f6b44f4295927a`、
  data-prep digest
  `sha256:8cbf525ba93064fe98a8d0ce6b4eb2261a73b010b9b3f11e33b0aed933966dc5`。
  Flyte registration
  `auto-e2e-platform-flyte-register:563e639e-1777-469b-86e7-e6fcb54a5c89`、
  version `uzZrVdYAJHKSpcwHm9qd1A`。
- MLflow model v42 は Full Run `avp4n785pdfdfnjgzd7t` / run
  `6535d0814d54475ab30f924311bb8ada` の epoch 8 checkpoint。
  SHA-256 は
  `2d7be6e2abb35824f275662f91b2025f1eb71d5887858337d173d22f3f4586bd`、
  validation ADE 3.51344 / FDE 10.18399。
- canonical v42 launcher
  `auto-e2e-platform-overlay-launch:0a171551-b373-41bd-a84d-244413328271` /
  Flyte `aqk92zpqstrp4t9bv6mq` は SUCCEEDED (約3時間18分、batch 16 /
  workers 0、GPU restart 0)。404 shards / 42,667 samples /
  20,963,627 bytesを生成し、DynamoDB `OVLSET` は `ready`。
  request identity
  `141c15e3921dec3e5dabc298671f502cd14668a7672b1601607f031fa990b57a`、
  cache identity
  `ad38d4e4b5ce8f9f57c8218145b2d118d5b745766e49dafab08a63f991b7886d`。
  manifest key は
  `overlays_manifest/schema=v1/model=2d7be6e2abb35824f275662f91b2025f1eb71d5887858337d173d22f3f4586bd/dataset=kitscenes/version=v2.2/manifest.json`、
  manifest SHA-256 は
  `d504bd1a0c76bcf212f3ba0e9df99437bf0731339fa5f84e3f72d8fd900ad049`、
  output SHA-256 は
  `9b9a207f7832f92da57bec840ac4613b8d9961a46c5596f8c0144a29cf9423b0`。
- CloudFront 本番で v42/v38 catalog と v42 binaryを確認。straight / left /
  right 3 sceneを Playwright 1440x1000 / 390x844 で検証し、v42選択、
  7/7 camera、front-centerの紫Ground Truth / 緑Prediction、BEV path、
  mobile filmstrip、横overflow 0、`/blob` request 0、HTTP / console /
  page error 0を確認。

残り: CloudFront + Cognito UI exposure (Phase 2§14-15, deferred), DCGM-exporter。

## 機密情報・アカウント固有値の扱い

- MUST: Account ID / ODCR ID / instance-id 等のアカウント固有値・キャパ依存値は git に commit しない。
- これらはリポジトリ直下の `.env` (gitignore 済み) に集約する。テンプレートは `.env.example` (git 管理)。
- 使うとき: `set -a; source .env; set +a` で環境変数として読み込む。
- Terraform へは `TF_VAR_odcr_id` 環境変数、または gitignore された `platform/infra/environments/<env>/secrets.auto.tfvars` 経由で渡す。`terraform.tfvars` には非機密値のみ。
- platform スクリプト (bootstrap.sh / post-apply.sh) はアカウントIDをハードコードせず `aws sts get-caller-identity` で動的取得、profile/region 等は `${AWS_PROFILE:-...}` で環境変数上書き可能にする。
- ドキュメント (PHASE2.md 等 git 管理対象) では Account ID を `<ACCOUNT_ID>` プレースホルダにする。
- 注: この CLAUDE.md 自体はグローバル gitignore 対象なので、ここに実値を書いても git には入らない。実値はあくまで `.env` を正とする。

## ローカル → EC2 同期

```bash
./sync-to-ec2.sh i-0e73ec8d6a7766395
```

- 引数で instance-id を指定 (省略時は g6e `i-0e73ec8d6a7766395` がデフォルト)
- S3 (`cdk-hnb659fds-assets-833707099141-us-east-1`) 経由で tar.gz を転送
- `COPYFILE_DISABLE=1` で macOS の `._*` ファイルを除外
- EC2 側は `/home/ubuntu/auto_e2e` を毎回全削除 → 展開

## EC2 でのコマンド実行 (SSM)

```bash
aws ssm send-command --region us-east-1 \
  --instance-ids i-0e73ec8d6a7766395 \
  --document-name "AWS-RunShellScript" \
  --parameters commands='["#!/bin/bash","source /opt/pytorch/bin/activate","cd /home/ubuntu/auto_e2e","make test"]' \
  --timeout-seconds 300 \
  --query 'Command.CommandId' --output text
```

結果取得:
```bash
aws ssm get-command-invocation --region us-east-1 \
  --command-id <COMMAND_ID> --instance-id i-0e73ec8d6a7766395 \
  --query '{Status:Status,Output:StandardOutputContent,Error:StandardErrorContent}' --output json
```

注意: SSM の `commands` 各要素は dash で実行される。`echo` 等の引数に丸括弧 `()` を含めると `syntax error near unexpected token (` で落ちるため、括弧は使わない。

## Makefile

これらは Macの上では動かないので、基本的に動作確認するときは、毎回 Sync してからEC2上で動かす。

- `make setup` — pip install torch timm pytest
- `make test` — pytest 実行 (92テスト, ~3分 on GPU)
- `make benchmark` — 速度ベンチマーク (18設定, ~10-15分)

## 学習経路 (Flyte のみ)

MUST: 学習は Platform の Flyte task 経由のみ。スタンドアロンのオンライン学習 (`Model/training/train.py`,
`train_offline_rl.py`) は廃止・削除済み。ローカルで直接学習ループを回す入口は無い。

- IL 学習: `Platform/pipelines/workflows.py` の `train_il` task (`wf_train_il` workflow)。
  pre-extracted WebDataset shard を読み、AutoE2E を forward → imitation loss (+ 任意で
  reasoning loss / world-model JEPA loss) → backward → step。
- データ前処理: `data_processing` task が raw → shard (WebDataset .tar) を生成。
- Reasoning ラベル (#98): `data_processing(reasoning_teacher=mock|cached|openai_compatible)` で
  各サンプルに `reasoning.json` をオフライン付与 (teacher は model-agnostic、実運用は
  openai_compatible を Cosmos3-Nano vLLM エンドポイントへ)。`train_il(enable_reasoning=True,
  reasoning_mode=pooled_latent|horizon_cross_attention)` で HorizonReasoningLoss を合算。
  ラベルの無い shard では reasoning loss はスキップ (zero-init なので軌跡は不変)。
- 学習コードは model_components を import するが、teacher は絶対に import しない
  (teacher は data_processing 配下、オフライン専用)。

## 開発フェーズ方針 (Research Phase)

このプロジェクトは現在完全に Research Phase にある。したがって:

- MUST NOT: 後方互換性を気にして設計を歪めない。互換ラッパー・deprecation shim・旧 API 温存は不要。
- 既存の taxonomy / schema / モジュール / 公開 ABI は必要なら破壊的に作り替えてよい (壊して OK)。
  例: reasoning の 3軸 taxonomy (maneuver/edge_case/weather_env) は v2 の compositional
  action-relevant ontology に置き換えてよく、旧 taxonomy を残す義務はない。
- 判断基準は「互換性」ではなく「設計として正しいか / research として検証したい仮説に合うか」。
- ただし壊す場合も、壊した理由をコミットメッセージに1行で書く (なぜその設計に変えたか)。
- テストは壊れた API を守るためでなく、新しい設計の正しさを担保するために書き直す。

## ブランチ運用

- `main` は origin/main (autowarefoundation 本家) に追従
- feature ブランチで作業 → origin (本家) に直接 push → origin/main へ PR (riita は Owner)
- fork (riita10069) はバックアップ用に残置
- コミットは限界まで小さい単位に分割し、できるだけコミット数が多くなるようにして。
MUST: コミットは数行のまとまった変更ごとに毎回コミット。とにかく小さくコミットを打って。
MUST: 1ファイル作成/変更するたびに即コミット。複数ファイルをまとめてコミットしない。
理由: 開発の経緯と「なぜその変更をしたか」をコミットメッセージでたどれるようにする。
また、問題があったときに任意の変更単位で切り戻し (revert) できるようにするため。
コミットメッセージには「何をしたか」だけでなく「なぜそうしたか」を1行で書く。
例: `feat(platform): add StorageClass for EKS Auto Mode (no built-in default)`
悪い例: `add files` / `update` / `WIP` — これらは禁止。
- `--signoff` を忘れずに (`git rebase HEAD~N --signoff`)が必須。 git push --force-with-lease origin <branch-name> でリリース (origin = autowarefoundation 本家)
- MUST: Co-Authored-By に Claude を絶対に入れない。コミットメッセージに AI の痕跡を残さないこと。

## PR 作成手順

コミットまではMac上でするが、
必ず、Sync して持ってった EC2 上からPRを作成する。絶対に Mac からPushしない。
理由：あなたはよく、未テストのコードをプッシュする癖があるため。絶対にEC2 に持っていき、テストをしてからPush

プッシュ後は、Mac 上の gh コマンドを使ってPR作成。プッシュだけEC2からすること。

## 既知の問題

- テスト ~3分は backbone ロードが支配的 (#24)
- 学習ループ (#14): Flyte `train_il` に実装 (LR scheduler / checkpoint / TensorBoard 等は今後)
- FutureState の Feature Reconstruction Loss 未接続 (#13)。`train_il` でも FutureState 出力は loss に未寄与
- FrozenBackbone は Swin V1 features_only。FutureState 出力 [B,256,H,W] と次元が合わず #13 実装には射影層が必要
- データセット: L2D パーサ実装済み (#16, Model/data_parsing/l2d)。KitScenes は PR #41 (draft) で進行中
- 解決済み: embed_dim は 256 に変更済み (#23)、TrajectoryImitationLoss 実装済み (#15)

## Owner

- Muhammad Zain Khawaja (Autoware Foundation President)
- Discord の Robotaxi WG チャンネルでコミュニケーション
- 謙虚な姿勢で提案する。Issue は学術論文を引用して丁寧に。

## コミュニケーション方針

- Issue: 英語、謙虚に、論文引用つき
- Discord: 英語、簡潔に
- Zenn ブログ: 日本語、初心者向け、比喩禁止、具体的な数値例で説明
- 文章スタイル: 絵文字・太字(**) ・チェックマーク等の装飾を使わない。自然な技術文章にする (AI っぽさを避ける)

## GitHub への投稿ルール (gh コメント/Issue/PR)

- MUST: gh コマンドで Issue/PR/コメントを作成・編集する前に、必ず投稿する文面の全文を私に提示し、承認を得ること。無断で gh による投稿をしない。
- 私が「投稿していい」と明示的に承認したら、Claude が gh で投稿してよい。
- 投稿前に必ず確認する項目: 投稿先 (Issue 番号 / PR 番号)、先頭メンションの要否。メンションは推測で付けない。
- 英語文面と日本語訳をセットで提示する。
