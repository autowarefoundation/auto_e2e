#!/usr/bin/env bash
set -euo pipefail

# Resolve K8s manifest placeholders and apply, in dependency order.
# Usage: ./apply.sh
#
# Requires: kubectl configured for the auto-e2e-platform cluster, and the
# following resolvable from env or terraform output:
#   ACM_CERT_ARN          HTTPS listener cert for the internal ALB
#   CONSOLE_ALB_SG_ID      SG that restricts the ALB to the CloudFront prefix list
#                          (terraform output console_alb_sg_id)
#   CONSOLE_ORIGIN         CloudFront console origin, e.g. https://dXXXX.cloudfront.net
#   AWS_ACCOUNT_ID         auto-detected via STS if unset
#   CONSOLE_S3_ROLE_ARN    informational only (S3 access is via Pod Identity)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
K8S_DIR="${SCRIPT_DIR}/k8s"

AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_PREFIX="${ECR_PREFIX:-${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com}"

: "${ACM_CERT_ARN:?Set ACM_CERT_ARN for the ALB HTTPS listener}"
: "${CONSOLE_ALB_SG_ID:?Set CONSOLE_ALB_SG_ID (terraform output console_alb_sg_id; SG admits HTTPS only from the CloudFront managed VPC-origin ENIs)}"
: "${CONSOLE_ORIGIN:?Set CONSOLE_ORIGIN (CloudFront console origin URL)}"
# Private-subnet CIDRs where the internal ALB ENIs live (terraform
# private-subnet outputs). NOT the whole VPC CIDR — under VPC CNI that would
# match every pod and make the NetworkPolicy a no-op.
: "${ALB_SUBNET_CIDR_A:?Set ALB_SUBNET_CIDR_A (first private-subnet CIDR)}"
: "${ALB_SUBNET_CIDR_B:?Set ALB_SUBNET_CIDR_B (second private-subnet CIDR)}"

export ECR_PREFIX ACM_CERT_ARN CONSOLE_ALB_SG_ID CONSOLE_ORIGIN
export ALB_SUBNET_CIDR_A ALB_SUBNET_CIDR_B

echo "Deploying DataModelConsole to EKS..."
echo "  ECR_PREFIX:         ${ECR_PREFIX}"
echo "  ACM_CERT_ARN:       ${ACM_CERT_ARN}"
echo "  CONSOLE_ALB_SG_ID:  ${CONSOLE_ALB_SG_ID}"
echo "  CONSOLE_ORIGIN:     ${CONSOLE_ORIGIN}"
echo "  ALB_SUBNET_CIDRs:   ${ALB_SUBNET_CIDR_A}, ${ALB_SUBNET_CIDR_B}"

SUBST_VARS='${ECR_PREFIX} ${ACM_CERT_ARN} ${CONSOLE_ALB_SG_ID} ${CONSOLE_ORIGIN} ${ALB_SUBNET_CIDR_A} ${ALB_SUBNET_CIDR_B}'

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
