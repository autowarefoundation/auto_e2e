# infra-console

DataModelConsole-dedicated infrastructure, kept in **isolated terraform state**
(`infra-console/terraform.tfstate`, separate from the main Platform root) so a
console apply can never plan a change to existing EKS / RDS / Flyte / MLflow.

Scope for the current goal: get the **current dashboard reachable, CloudFront-only**.
No Cognito / Lambda@Edge / ACM — CloudFront serves HTTPS with its default
`*.cloudfront.net` cert; the internal ALB is plain HTTP. The trajectory-overlay
work is deferred.

## What it creates

- `aws_security_group.console_alb` — attached to the internal ALB by the k8s
  Ingress. Its ingress rule is the gate that enforces "CloudFront only".
- `aws_cloudfront_vpc_origin` + `aws_cloudfront_distribution` — CloudFront reaches
  the internal ALB through a VPC origin (an AWS-managed ENI inside the VPC), not
  from public edge IPs.
- `aws_iam_role.console_api` + Pod Identity association — S3 read-only
  (datasets + artifacts) and DynamoDB (`auto-e2e-console` + `gsi1`) for the API.

## Why two phases

The managed `CloudFront-VPCOrigins-Service-SG` (the source the ALB SG must trust)
does not exist until a VPC origin is created, and the VPC origin needs the ALB
ARN, and the ALB is created by the k8s Ingress controller — which needs the SG
id. Chicken-and-egg, so we split on `alb_arn`:

### Phase 1 — SG + IAM (before k8s)

```bash
cd Platform/infra-console
export AWS_PROFILE=autowarefoundation
terraform init
terraform apply            # alb_arn defaults to "" → SG (VPC-CIDR bootstrap rule) + IAM only
terraform output -raw console_alb_sg_id
```

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

terraform apply -var="alb_arn=${ALB_ARN}" -var="alb_dns=${ALB_DNS}"
terraform output -raw cloudfront_url
```

Phase 2 also swaps the SG's bootstrap VPC-CIDR rule for the CloudFront-only rule,
so the end state is: nothing but CloudFront's VPC-origin ENIs can reach the ALB.
