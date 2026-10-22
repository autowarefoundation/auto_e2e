#!/bin/bash
# Copy one run's reviewable outputs (rollout.asl, videos, metrics, configs) to
# S3 so they can be inspected off the GPU host.
#
#   RESULTS_S3_URI=s3://<bucket>/closed-loop-runs \
#       bash experiment/closed-loop/publish_results.sh <run name>
# The bucket is account-specific and comes from the environment, not the repo.
set -euo pipefail

RUN_NAME="${1:?usage: publish_results.sh <run name>}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/closed-loop-runs}"
: "${RESULTS_S3_URI:?set RESULTS_S3_URI, e.g. s3://<bucket>/closed-loop-runs}"
LOG_DIR="${RUNS_ROOT}/${RUN_NAME}"
[[ -f "${LOG_DIR}/experiment.json" ]] || { echo "no run at ${LOG_DIR}" >&2; exit 1; }

# Container logs and Prometheus data are large and only matter on the host.
aws s3 sync --only-show-errors "${LOG_DIR}" "${RESULTS_S3_URI}/${RUN_NAME}" \
    --exclude "prometheus/*" --exclude "txt-logs/*" --exclude "*/rclog-all*"
echo "Published ${RUN_NAME} to ${RESULTS_S3_URI}/${RUN_NAME}"
