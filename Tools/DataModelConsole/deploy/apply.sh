#!/usr/bin/env bash
set -euo pipefail

# Resolve K8s manifest placeholders, pin images by digest, and apply in
# dependency order.
#
# Required environment:
#   EXPECTED_AWS_ACCOUNT_ID Known Platform account ID; never inferred from the
#                           active credentials used for the deployment
#   CONSOLE_INFRA_PHASE     bootstrap or locked, matching the canonical
#                           Platform/infra-console Terraform phase
#   CONSOLE_ALB_SG_ID      SG that restricts the ALB to CloudFront's managed
#                          VPC-origin ENIs (terraform output console_alb_sg_id)
#   CONSOLE_API_IMAGE      Full ECR digest URI for console-api
#   CONSOLE_WEB_IMAGE      Full ECR digest URI for console-web
#   CONSOLE_ORIGIN         CloudFront console origin, e.g. https://dXXXX.cloudfront.net
#                          (required in locked phase; unset in bootstrap)
#
# No ACM_CERT_ARN: the internal ALB listens on HTTP:80 (CloudFront terminates
# viewer TLS and reaches the ALB over http-only through its VPC origin).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
K8S_DIR="${SCRIPT_DIR}/k8s"

REQUIRED_AWS_REGION="us-west-2"
REQUIRED_CLUSTER_NAME="auto-e2e-platform"
AWS_PROFILE="${AWS_PROFILE:-autowarefoundation}"
AWS_REGION="${AWS_REGION:-${REQUIRED_AWS_REGION}}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

for command_name in aws kubectl envsubst; do
    command -v "${command_name}" >/dev/null 2>&1 ||
        die "Required command not found: ${command_name}"
done

: "${EXPECTED_AWS_ACCOUNT_ID:?Set EXPECTED_AWS_ACCOUNT_ID to the known Platform account ID}"
: "${CONSOLE_INFRA_PHASE:?Set CONSOLE_INFRA_PHASE to bootstrap or locked}"
: "${CONSOLE_API_IMAGE:?Set CONSOLE_API_IMAGE to the full ECR image digest URI}"
: "${CONSOLE_WEB_IMAGE:?Set CONSOLE_WEB_IMAGE to the full ECR image digest URI}"

# SG attached to the ALB via the Ingress security-groups annotation (Auto Mode
# has no IngressClassParams.securityGroups). Admits HTTP:80 only from CloudFront's
# managed VPC-origin ENIs.
: "${CONSOLE_ALB_SG_ID:?Set CONSOLE_ALB_SG_ID (terraform output console_alb_sg_id)}"
# Private-subnet (internal-elb) CIDRs where the internal ALB ENIs live. NOT the
# whole VPC CIDR — under VPC CNI that would match every pod and make the
# NetworkPolicy a no-op. The cluster has THREE internal-elb subnets (one per
# AZ) and the ALB may land an ENI in any of them, so all three are required.
: "${ALB_SUBNET_CIDR_A:?Set ALB_SUBNET_CIDR_A (first internal-elb subnet CIDR)}"
: "${ALB_SUBNET_CIDR_B:?Set ALB_SUBNET_CIDR_B (second internal-elb subnet CIDR)}"
: "${ALB_SUBNET_CIDR_C:?Set ALB_SUBNET_CIDR_C (third internal-elb subnet CIDR)}"

[[ "${EXPECTED_AWS_ACCOUNT_ID}" =~ ^[0-9]{12}$ ]] ||
    die "EXPECTED_AWS_ACCOUNT_ID must be a 12-digit AWS account ID"
[[ "${AWS_REGION}" == "${REQUIRED_AWS_REGION}" ]] ||
    die "AWS_REGION must be ${REQUIRED_AWS_REGION}, got ${AWS_REGION}"
[[ "${CONSOLE_ALB_SG_ID}" =~ ^sg-[0-9a-f]+$ ]] ||
    die "CONSOLE_ALB_SG_ID must be a valid security group ID"

case "${CONSOLE_INFRA_PHASE}" in
    bootstrap)
        [[ -z "${CONSOLE_ORIGIN:-}" ]] ||
            die "CONSOLE_ORIGIN must be unset during bootstrap"
        CONSOLE_ORIGIN=""
        ;;
    locked)
        : "${CONSOLE_ORIGIN:?Set CONSOLE_ORIGIN to the CloudFront URL in locked phase}"
        [[ "${CONSOLE_ORIGIN}" =~ ^https://[a-z0-9]+\.cloudfront\.net$ ]] ||
            die "CONSOLE_ORIGIN must be an https://*.cloudfront.net URL"
        ;;
    *)
        die "CONSOLE_INFRA_PHASE must be bootstrap or locked"
        ;;
esac

validate_image() {
    local image="$1"
    local repository="$2"
    local expected_prefix
    local digest

    expected_prefix="${EXPECTED_AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/auto-e2e/${repository}@sha256:"
    [[ "${image}" == "${expected_prefix}"* ]] ||
        die "${repository} image must use the expected account, region, repository, and @sha256 digest"
    digest="${image#"${expected_prefix}"}"
    [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] ||
        die "${repository} image digest must contain 64 lowercase hexadecimal characters"
}

validate_image "${CONSOLE_API_IMAGE}" "console-api"
validate_image "${CONSOLE_WEB_IMAGE}" "console-web"

actual_account_id="$(
    aws --profile "${AWS_PROFILE}" --region "${AWS_REGION}" \
        sts get-caller-identity --query Account --output text
)"
[[ "${actual_account_id}" == "${EXPECTED_AWS_ACCOUNT_ID}" ]] ||
    die "AWS account mismatch: expected ${EXPECTED_AWS_ACCOUNT_ID}, got ${actual_account_id}"

expected_cluster_endpoint="$(
    aws --profile "${AWS_PROFILE}" --region "${AWS_REGION}" \
        eks describe-cluster --name "${REQUIRED_CLUSTER_NAME}" \
        --query 'cluster.endpoint' --output text
)"
active_cluster_endpoint="$(
    kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}'
)"
[[ -n "${active_cluster_endpoint}" ]] ||
    die "No active kubectl cluster is configured"
[[ "${active_cluster_endpoint}" == "${expected_cluster_endpoint}" ]] ||
    die "kubectl context does not target ${REQUIRED_CLUSTER_NAME} in the expected account and region"

export CONSOLE_ALB_SG_ID CONSOLE_ORIGIN
export ALB_SUBNET_CIDR_A ALB_SUBNET_CIDR_B ALB_SUBNET_CIDR_C

echo "Deploying DataModelConsole to EKS..."
echo "  AWS account:        ${actual_account_id}"
echo "  AWS region:         ${AWS_REGION}"
echo "  EKS cluster:        ${REQUIRED_CLUSTER_NAME}"
echo "  Infrastructure:     ${CONSOLE_INFRA_PHASE}"
echo "  API image:          ${CONSOLE_API_IMAGE}"
echo "  Web image:          ${CONSOLE_WEB_IMAGE}"
echo "  CONSOLE_ALB_SG_ID:  ${CONSOLE_ALB_SG_ID}"
echo "  CONSOLE_ORIGIN:     ${CONSOLE_ORIGIN:-(unset; same-origin /api)}"
echo "  ALB_SUBNET_CIDRs:   ${ALB_SUBNET_CIDR_A}, ${ALB_SUBNET_CIDR_B}, ${ALB_SUBNET_CIDR_C}"

# envsubst expects literal variable references in its allowlist.
# shellcheck disable=SC2016
SUBST_VARS='${CONSOLE_ALB_SG_ID} ${CONSOLE_ORIGIN} ${ALB_SUBNET_CIDR_A} ${ALB_SUBNET_CIDR_B} ${ALB_SUBNET_CIDR_C}'

apply_manifest() {
    local manifest_name="$1"

    echo "  Applying ${manifest_name}..."
    envsubst "${SUBST_VARS}" <"${K8S_DIR}/${manifest_name}" |
        kubectl apply -f -
}

apply_deployment() {
    local manifest_name="$1"
    local container_name="$2"
    local image="$3"

    echo "  Applying ${manifest_name} with immutable image ${image}..."
    kubectl set image --local -f "${K8S_DIR}/${manifest_name}" \
        "${container_name}=${image}" -o yaml |
        kubectl apply -f -
}

# Namespace first, then config/identity, then network + policy. Web is rolled
# out before API because it can read both the legacy array and paginated API
# envelopes; once every Web pod is compatible, the API contract can change
# without a mixed-replica outage.
kubectl apply -f "${K8S_DIR}/namespace.yaml"
for manifest_name in configmap.yaml serviceaccount.yaml; do
    apply_manifest "${manifest_name}"
done
for manifest_name in services.yaml pdb.yaml networkpolicy.yaml ingress.yaml; do
    apply_manifest "${manifest_name}"
done

echo "Rolling out backward-compatible Web..."
apply_deployment "web-deployment.yaml" "web" "${CONSOLE_WEB_IMAGE}"
kubectl -n console rollout status deployment/console-web --timeout=180s

echo "Rolling out API after Web compatibility is established..."
apply_deployment "api-deployment.yaml" "api" "${CONSOLE_API_IMAGE}"
kubectl -n console rollout status deployment/console-api --timeout=180s
echo "DataModelConsole deployed in ${CONSOLE_INFRA_PHASE} phase with digest-pinned images."
