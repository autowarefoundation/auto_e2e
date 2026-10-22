#!/bin/bash
# Pack one run's reviewable outputs (rollout.asl, videos, metrics, configs)
# into a tarball and upload it with a presigned PUT URL, because the GPU host's
# instance role can only read S3.
#
#   bash experiment/closed-loop/publish_results.sh <run name>              # pack only
#   RESULTS_PUT_URL=<presigned PUT url> \
#       bash experiment/closed-loop/publish_results.sh <run name>          # pack + upload
set -euo pipefail

RUN_NAME="${1:?usage: publish_results.sh <run name>}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/closed-loop-runs}"
LOG_DIR="${RUNS_ROOT}/${RUN_NAME}"
ARCHIVE="${RUNS_ROOT}/${RUN_NAME}.tar.gz"
[[ -f "${LOG_DIR}/experiment.json" ]] || { echo "no run at ${LOG_DIR}" >&2; exit 1; }

# Container logs, Prometheus data and the raw rclog are large and only matter
# on the host; rollout.asl carries the messages.
tar czf "${ARCHIVE}" -C "${RUNS_ROOT}" \
    --exclude "${RUN_NAME}/prometheus" --exclude "${RUN_NAME}/txt-logs" \
    --exclude "*.rclog" --exclude "rclog-all*" \
    "${RUN_NAME}"
echo "Packed ${ARCHIVE} ($(du -h "${ARCHIVE}" | cut -f1))"

if [[ -n "${RESULTS_PUT_URL:-}" ]]; then
    curl -fsS --retry 3 -X PUT -T "${ARCHIVE}" "${RESULTS_PUT_URL}"
    echo "Uploaded ${RUN_NAME}.tar.gz"
fi
