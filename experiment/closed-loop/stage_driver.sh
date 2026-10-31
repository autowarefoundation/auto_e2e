#!/bin/bash
# Place the AutoE2E model code and checkpoint where the driver container
# mounts them ($ALPASIM_ROOT/data/drivers -> /mnt/drivers).
#
# The checkpoint lives in the Platform account, which the GPU host cannot read
# directly. Pass a presigned URL created on a machine that can:
#   CHECKPOINT_URL=<presigned url> bash experiment/closed-loop/stage_driver.sh
# A checkpoint already staged with the pinned SHA-256 is kept.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
# shellcheck source=config.env
source "${HERE}/config.env"

DRIVER_DIR="${ALPASIM_ROOT}/data/drivers/autoe2e"
CHECKPOINT="${DRIVER_DIR}/checkpoint.pt"
mkdir -p "${DRIVER_DIR}"

checkpoint_ok() {
    [[ -f "${CHECKPOINT}" ]] &&
        echo "${CHECKPOINT_SHA256}  ${CHECKPOINT}" | sha256sum --check --status
}

if ! checkpoint_ok; then
    : "${CHECKPOINT_URL:?set CHECKPOINT_URL to a presigned URL of ${CHECKPOINT_S3_KEY}}"
    curl -fsSL --retry 3 -o "${CHECKPOINT}.part" "${CHECKPOINT_URL}"
    mv "${CHECKPOINT}.part" "${CHECKPOINT}"
    checkpoint_ok || { echo "checkpoint SHA-256 mismatch" >&2; exit 1; }
fi

# The driver imports AutoE2E from here (AUTOE2E_MODEL_ROOT in the driver config).
rsync -a --delete --exclude tests --exclude __pycache__ \
    "${REPO_ROOT}/Model/" "${DRIVER_DIR}/Model/"
echo "Staged ${CHECKPOINT} (${CHECKPOINT_SHA256:0:12}) and Model/ at $(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
