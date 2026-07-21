#!/usr/bin/env bash
set -euo pipefail

# Launch the production one-shot materializer after the Flyte publication and
# overlay workflow has succeeded. This script is intentionally separate from
# apply.sh: a normal Console deployment must never start a dataset scan.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_TEMPLATE="${SCRIPT_DIR}/jobs/reasoning-materialization-job.yaml"

REQUIRED_AWS_REGION="us-west-2"
REQUIRED_CLUSTER_NAME="auto-e2e-platform"
AWS_PROFILE="${AWS_PROFILE:-autowarefoundation}"
AWS_REGION="${AWS_REGION:-${REQUIRED_AWS_REGION}}"
DRY_RUN="${DRY_RUN:-false}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

for command_name in aws kubectl envsubst; do
    command -v "${command_name}" >/dev/null 2>&1 ||
        die "Required command not found: ${command_name}"
done
if ! command -v sha256sum >/dev/null 2>&1 &&
    ! command -v shasum >/dev/null 2>&1; then
    die "Required command not found: sha256sum or shasum"
fi

: "${EXPECTED_AWS_ACCOUNT_ID:?Set EXPECTED_AWS_ACCOUNT_ID to the known Platform account ID}"
: "${CONSOLE_API_IMAGE:?Set CONSOLE_API_IMAGE to a digest-pinned console-api image}"
: "${PUBLISHED_DATASET:?Set PUBLISHED_DATASET from the successful Flyte execution}"
: "${DATASET_VERSION:?Set DATASET_VERSION from the successful Flyte execution}"
: "${MANIFEST_KEY:?Set MANIFEST_KEY from the successful Flyte execution output}"
: "${MANIFEST_SHA256:?Set MANIFEST_SHA256 from the successful Flyte execution output}"

[[ "${EXPECTED_AWS_ACCOUNT_ID}" =~ ^[0-9]{12}$ ]] ||
    die "EXPECTED_AWS_ACCOUNT_ID must be a 12-digit AWS account ID"
[[ "${AWS_REGION}" == "${REQUIRED_AWS_REGION}" ]] ||
    die "AWS_REGION must be ${REQUIRED_AWS_REGION}, got ${AWS_REGION}"
[[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "false" ]] ||
    die "DRY_RUN must be true or false"
if [[ "${DRY_RUN}" == "false" ]]; then
    [[ "${CONFIRM_PRODUCTION_MATERIALIZATION:-}" == "yes" ]] ||
        die "Set CONFIRM_PRODUCTION_MATERIALIZATION=yes to create the production Job"
fi

if [[ "${PUBLISHED_DATASET}" != "kitscenes" &&
    "${PUBLISHED_DATASET}" != "l2d" &&
    "${PUBLISHED_DATASET}" != "nvidia_av" &&
    ! "${PUBLISHED_DATASET}" =~ ^kitscenes-smoke-[0-9a-f]{12}$ ]]; then
    die "PUBLISHED_DATASET is not exposed by DataModelConsole"
fi
[[ "${DATASET_VERSION}" =~ ^v[0-9]+(\.[0-9]+)*$ ]] ||
    die "DATASET_VERSION is invalid"
[[ "${MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] ||
    die "MANIFEST_SHA256 must be 64 lowercase hexadecimal characters"

expected_manifest_key="${PUBLISHED_DATASET}/${DATASET_VERSION}/shards/manifest.json"
[[ "${MANIFEST_KEY}" == "${expected_manifest_key}" ]] ||
    die "MANIFEST_KEY does not match the requested dataset and version"

expected_image_prefix="${EXPECTED_AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/auto-e2e/console-api@sha256:"
[[ "${CONSOLE_API_IMAGE}" == "${expected_image_prefix}"* ]] ||
    die "CONSOLE_API_IMAGE must use the expected account, region, repository, and digest"
image_digest="${CONSOLE_API_IMAGE#"${expected_image_prefix}"}"
[[ "${image_digest}" =~ ^[0-9a-f]{64}$ ]] ||
    die "CONSOLE_API_IMAGE has an invalid digest"

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
    die "kubectl context does not target ${REQUIRED_CLUSTER_NAME}"

datasets_bucket="$(
    kubectl -n console get configmap console-config \
        -o jsonpath='{.data.datasets-bucket}'
)"
[[ "${datasets_bucket}" == "auto-e2e-platform-datasets-${EXPECTED_AWS_ACCOUNT_ID}" ]] ||
    die "console-config datasets bucket does not match the expected account"
kubectl -n console get serviceaccount console-api >/dev/null

manifest_file="$(mktemp)"
trap 'rm -f "${manifest_file}"' EXIT
aws --profile "${AWS_PROFILE}" --region "${AWS_REGION}" \
    s3api get-object \
    --bucket "${datasets_bucket}" \
    --key "${MANIFEST_KEY}" \
    "${manifest_file}" >/dev/null
if command -v sha256sum >/dev/null 2>&1; then
    actual_manifest_sha256="$(sha256sum "${manifest_file}" | awk '{print $1}')"
else
    actual_manifest_sha256="$(shasum -a 256 "${manifest_file}" | awk '{print $1}')"
fi
[[ "${actual_manifest_sha256}" == "${MANIFEST_SHA256}" ]] ||
    die "Flyte manifest digest does not match the published S3 bytes"

export CONSOLE_API_IMAGE PUBLISHED_DATASET DATASET_VERSION MANIFEST_SHA256
# envsubst expects literal variable references in its allowlist.
# shellcheck disable=SC2016
subst_vars='${CONSOLE_API_IMAGE} ${PUBLISHED_DATASET} ${DATASET_VERSION} ${MANIFEST_SHA256}'

if [[ "${DRY_RUN}" == "true" ]]; then
    envsubst "${subst_vars}" <"${JOB_TEMPLATE}" |
        kubectl create --dry-run=client -o yaml -f -
    exit 0
fi

echo "Creating reasoning materialization Job for ${MANIFEST_KEY}..."
envsubst "${subst_vars}" <"${JOB_TEMPLATE}" |
    kubectl create -f -
echo "Monitor with: kubectl -n console get jobs,pods -l app.kubernetes.io/name=console-reasoning-materializer"
