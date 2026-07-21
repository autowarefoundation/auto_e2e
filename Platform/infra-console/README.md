# infra-console

DataModelConsole-dedicated infrastructure, kept in **isolated terraform state**
(`infra-console/terraform.tfstate`, separate from the main Platform root) so a
console apply can never plan a change to existing EKS / RDS / Flyte / MLflow.
This is the sole Terraform root that owns the production Console resources.
`Tools/DataModelConsole/deploy/terraform` is retired and can only detach legacy
local state without destroying AWS resources.

CloudFront serves HTTPS with its default `*.cloudfront.net` cert; the internal
ALB is plain HTTP. Exact geography remains disabled until authenticated role
propagation is implemented.

## What it creates

- `aws_security_group.console_alb` — attached to the internal ALB by the k8s
  Ingress. Its ingress rule is the gate that enforces "CloudFront only".
- `aws_cloudfront_vpc_origin` + `aws_cloudfront_distribution` — CloudFront reaches
  the internal ALB through a VPC origin (an AWS-managed ENI inside the VPC), not
  from public edge IPs.
- `aws_iam_role.console_api` + Pod Identity association — S3 read-only
  (datasets + artifacts) and DynamoDB (`auto-e2e-console` + `gsi1`) for the API.
  The explicit post-publication reasoning-materialization Job reuses this
  identity; it does not require a second role or a Terraform change.

## Why two phases

The managed `CloudFront-VPCOrigins-Service-SG` (the source the ALB SG must trust)
does not exist until a VPC origin is created, and the VPC origin needs the ALB
ARN, and the ALB is created by the k8s Ingress controller — which needs the SG
id. Chicken-and-egg, so `deployment_phase` must be explicitly set on every
plan/apply. A no-argument apply cannot silently return to bootstrap. Once the
locked resources exist, `prevent_destroy` also blocks a mistaken downgrade.

Set the account-specific values from the ignored repository `.env` or shell,
never in a tracked tfvars file:

```bash
cd Platform/infra-console
export AWS_PROFILE=autowarefoundation
export TF_VAR_expected_aws_account_id="<ACCOUNT_ID>"
export TF_VAR_vpc_id="<VPC_ID>"
terraform init
```

### Phase 1 — SG + IAM (before k8s)

```bash
terraform apply -var='deployment_phase=bootstrap'
terraform output -raw console_alb_sg_id
```

Bootstrap is only for initial ALB creation. It temporarily admits HTTP from the
VPC CIDR because the CloudFront VPC-origin service SG does not exist yet.

### Deploy k8s (creates the internal ALB)

Substitute the SG id into the Ingress and apply the manifests (see
`Tools/DataModelConsole/deploy/`), then read the ALB the controller created:

```bash
kubectl -n console get ingress console-ingress \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

### Phase 2 — CloudFront + lock the SG

```bash
ALB_DNS=$(kubectl -n console get ingress console-ingress \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
ALB_ARN=$(aws elbv2 describe-load-balancers --region us-west-2 \
  --query "LoadBalancers[?DNSName=='${ALB_DNS}'].LoadBalancerArn" --output text)

terraform apply \
  -var='deployment_phase=locked' \
  -var="alb_arn=${ALB_ARN}" \
  -var="alb_dns=${ALB_DNS}"
terraform output -raw cloudfront_url
```

Phase 2 also swaps the SG's bootstrap VPC-CIDR rule for the CloudFront-only rule,
so the end state is: nothing but CloudFront's VPC-origin ENIs can reach the ALB.
Every later plan/apply must continue to pass `deployment_phase=locked` and the
same ALB ARN/DNS, either explicitly or through an ignored local
`*.auto.tfvars` file.

The ALB SG, CloudFront-only ingress rule, VPC origin, distribution, Console API
IAM role, and Pod Identity association use `prevent_destroy`. Removing that
protection is a deliberate migration requiring a reviewed code change; do not
use bootstrap or `terraform destroy` as an update path.
