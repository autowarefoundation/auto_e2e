#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILDSPEC="${SCRIPT_DIR}/buildspec.yml"
APPLY_SCRIPT="${SCRIPT_DIR}/apply.sh"
MATERIALIZE_SCRIPT="${SCRIPT_DIR}/run-reasoning-materialization.sh"
MATERIALIZE_JOB="${SCRIPT_DIR}/jobs/reasoning-materialization-job.yaml"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

ruby -ryaml -e '
document = YAML.load_file(ARGV.fetch(0))
abort "buildspec must be a mapping" unless document.is_a?(Hash)
abort "unexpected buildspec version" unless document["version"] == 0.2
abort "missing buildspec phases" unless document["phases"].is_a?(Hash)
' "${BUILDSPEC}"

# The image placeholder is intentionally passed literally to Ruby.
# shellcheck disable=SC2016
ruby -ryaml -e '
document = YAML.load_file(ARGV.fetch(0))
abort "materializer must be a batch/v1 Job" unless
  document["apiVersion"] == "batch/v1" && document["kind"] == "Job"
spec = document.fetch("spec")
pod = spec.fetch("template").fetch("spec")
container = pod.fetch("containers").fetch(0)
abort "materializer retries must be disabled" unless spec["backoffLimit"] == 0
abort "materializer deadline is missing" unless
  spec["activeDeadlineSeconds"].is_a?(Integer)
abort "materializer must use console-api identity" unless
  pod["serviceAccountName"] == "console-api"
abort "materializer must not restart" unless pod["restartPolicy"] == "Never"
abort "materializer image is not parameterized" unless
  container["image"] == "${CONSOLE_API_IMAGE}"
abort "materializer command is missing" unless
  container.fetch("args").include?("materialize-reasoning")
' "${MATERIALIZE_JOB}"

bash -n "${MATERIALIZE_SCRIPT}"

if grep -Eq '(^|[^0-9])[0-9]{12}([^0-9]|$)' "${BUILDSPEC}"; then
    fail "buildspec contains a hardcoded AWS account ID"
fi
if grep -Eiq '(^|[^[:alnum:]_-])latest([^[:alnum:]_-]|$)' "${BUILDSPEC}"; then
    fail "buildspec contains a mutable latest reference"
fi

# These are source-code snippets to match verbatim, not shell expressions.
# shellcheck disable=SC2016
required_buildspec_literals=(
    'CODEBUILD_RESOLVED_SOURCE_VERSION'
    'CODEBUILD_BUILD_NUMBER'
    'sts get-caller-identity'
    'imageDetails[0].imageDigest'
    'CONSOLE_API_IMAGE="${ECR_REPO_PREFIX}/console-api@${API_DIGEST}"'
    'CONSOLE_WEB_IMAGE="${ECR_REPO_PREFIX}/console-web@${WEB_DIGEST}"'
    'console-images.env'
)
for literal in "${required_buildspec_literals[@]}"; do
    grep -Fq -- "${literal}" "${BUILDSPEC}" ||
        fail "buildspec is missing required contract: ${literal}"
done

# shellcheck disable=SC2016
required_apply_literals=(
    'validate_image "${CONSOLE_API_IMAGE}" "console-api"'
    'validate_image "${CONSOLE_WEB_IMAGE}" "console-web"'
    'kubectl set image --local'
    '@sha256:'
)
for literal in "${required_apply_literals[@]}"; do
    grep -Fq -- "${literal}" "${APPLY_SCRIPT}" ||
        fail "apply.sh is missing required contract: ${literal}"
done

# The one-shot data scan is deliberately excluded from normal deployments.
if grep -Fq 'reasoning-materialization-job.yaml' "${APPLY_SCRIPT}"; then
    fail "apply.sh must not launch the reasoning materializer"
fi

# shellcheck disable=SC2016
required_materializer_literals=(
    'CONFIRM_PRODUCTION_MATERIALIZATION'
    'Flyte manifest digest does not match the published S3 bytes'
    'kubectl create -f -'
    'serviceAccountName: console-api'
    'backoffLimit: 0'
    'activeDeadlineSeconds:'
    'restartPolicy: Never'
    '${CONSOLE_API_IMAGE}'
    '--manifest-sha256'
)
for literal in "${required_materializer_literals[@]}"; do
    grep -Fq -- "${literal}" "${MATERIALIZE_SCRIPT}" "${MATERIALIZE_JOB}" ||
        fail "materialization launcher is missing required contract: ${literal}"
done

echo "Build and materialization manifests satisfy the deployment contracts."
