#!/usr/bin/env bash
set -euo pipefail

# Resolve K8s manifest placeholders and apply, in dependency order.
# Usage: ./apply.sh
#
# Requires: kubectl configured for the auto-e2e-platform cluster, and the
# following resolvable from env or terraform output:
#   CONSOLE_ALB_SG_ID      SG that restricts the ALB to CloudFront's managed
#                          VPC-origin ENIs (terraform output console_alb_sg_id)
#   CONSOLE_ORIGIN         CloudFront console origin, e.g. https://dXXXX.cloudfront.net
#                          (known only after infra Phase 2; pass "" on the first
#                          apply — CORS is off the same-origin /api path anyway)
#   AWS_ACCOUNT_ID         auto-detected via STS if unset
#
# No ACM_CERT_ARN: the internal ALB listens on HTTP:80 (CloudFront terminates
# viewer TLS and reaches the ALB over http-only through its VPC origin).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
K8S_DIR="${SCRIPT_DIR}/k8s"

AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_PREFIX="${ECR_PREFIX:-${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com}"

# SG attached to the ALB via the Ingress security-groups annotation (Auto Mode
# has no IngressClassParams.securityGroups). Admits HTTP:80 only from CloudFront's
# managed VPC-origin ENIs.
: "${CONSOLE_ALB_SG_ID:?Set CONSOLE_ALB_SG_ID (terraform output console_alb_sg_id)}"
# CloudFront origin is only known after infra Phase 2; default to empty on the
# first apply. The frontend uses same-origin /api so CORS is not on the hot path.
CONSOLE_ORIGIN="${CONSOLE_ORIGIN:-}"
# Private-subnet (internal-elb) CIDRs where the internal ALB ENIs live. NOT the
# whole VPC CIDR — under VPC CNI that would match every pod and make the
# NetworkPolicy a no-op. The cluster has THREE internal-elb subnets (one per
# AZ) and the ALB may land an ENI in any of them, so all three are required.
: "${ALB_SUBNET_CIDR_A:?Set ALB_SUBNET_CIDR_A (first internal-elb subnet CIDR)}"
: "${ALB_SUBNET_CIDR_B:?Set ALB_SUBNET_CIDR_B (second internal-elb subnet CIDR)}"
: "${ALB_SUBNET_CIDR_C:?Set ALB_SUBNET_CIDR_C (third internal-elb subnet CIDR)}"

export ECR_PREFIX CONSOLE_ALB_SG_ID CONSOLE_ORIGIN
export ALB_SUBNET_CIDR_A ALB_SUBNET_CIDR_B ALB_SUBNET_CIDR_C

echo "Deploying DataModelConsole to EKS..."
echo "  ECR_PREFIX:         ${ECR_PREFIX}"
echo "  CONSOLE_ALB_SG_ID:  ${CONSOLE_ALB_SG_ID}"
echo "  CONSOLE_ORIGIN:     ${CONSOLE_ORIGIN:-(unset; same-origin /api)}"
echo "  ALB_SUBNET_CIDRs:   ${ALB_SUBNET_CIDR_A}, ${ALB_SUBNET_CIDR_B}, ${ALB_SUBNET_CIDR_C}"

SUBST_VARS='${ECR_PREFIX} ${CONSOLE_ALB_SG_ID} ${CONSOLE_ORIGIN} ${ALB_SUBNET_CIDR_A} ${ALB_SUBNET_CIDR_B} ${ALB_SUBNET_CIDR_C}'

# Namespace first, then config/identity, then workloads, then network + policy.
kubectl apply -f "${K8S_DIR}/namespace.yaml"
for f in configmap.yaml serviceaccount.yaml \
         api-deployment.yaml web-deployment.yaml \
         services.yaml pdb.yaml networkpolicy.yaml ingress.yaml; do
    echo "  Applying ${f}..."
    envsubst "${SUBST_VARS}" < "${K8S_DIR}/${f}" | kubectl apply -f -
done

echo "Waiting for rollout..."
kubectl -n console rollout status deployment/console-api --timeout=180s
kubectl -n console rollout status deployment/console-web --timeout=180s
echo "DataModelConsole deployed. ALB SG ${CONSOLE_ALB_SG_ID} restricts ingress to the CloudFront managed VPC-origin ENIs (CloudFront-VPCOrigins-Service-SG) only."
