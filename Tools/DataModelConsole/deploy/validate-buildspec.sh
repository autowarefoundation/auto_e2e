#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILDSPEC="${SCRIPT_DIR}/buildspec.yml"
APPLY_SCRIPT="${SCRIPT_DIR}/apply.sh"

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

echo "buildspec.yml satisfies the digest-pinned apply.sh contract."
