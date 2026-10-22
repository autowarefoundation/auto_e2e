#!/bin/bash
# Run AutoE2E closed loop in AlpaSim (docker compose) over the scenes in
# scenes.txt and keep each run's rollouts, videos and metrics under
# $RUNS_ROOT/<run name>.
#
#   bash experiment/closed-loop/run_scenes.sh                 # nuPlan Epoch 5
#   CHECKPOINT=MOCK bash experiment/closed-loop/run_scenes.sh # plumbing only
#   SCENES="clipgt-..." bash experiment/closed-loop/run_scenes.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
# shellcheck source=config.env
source "${HERE}/config.env"
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

# MOCK drives straight without a model; anything else is a path in the container.
CHECKPOINT="${CHECKPOINT:-/mnt/drivers/autoe2e/checkpoint.pt}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/closed-loop-runs}"
RUN_NAME="${RUN_NAME:-$(date -u +%Y%m%dT%H%M%SZ)-$([[ ${CHECKPOINT} == MOCK ]] && echo mock || echo ${CHECKPOINT_SHA256:0:12})}"
LOG_DIR="${RUNS_ROOT}/${RUN_NAME}"

if [[ -z "${SCENES:-}" ]]; then
    SCENES="$(grep -vE '^\s*(#|$)' "${HERE}/scenes.txt" | paste -sd, -)"
fi
# The wizard downloads missing scenes from the gated NuRec dataset. The token
# of an account with access is kept apart from the host's default HF token.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HOME}/.config/closed-loop/hf_token}"
if [[ -z "${HF_TOKEN:-}" && -f "${HF_TOKEN_FILE}" ]]; then
    HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
fi
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN or write it to ${HF_TOKEN_FILE}}"

mkdir -p "${LOG_DIR}"
cat > "${LOG_DIR}/experiment.json" <<EOF
{
  "alpasim_commit": "${ALPASIM_COMMIT}",
  "auto_e2e_commit": "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)",
  "checkpoint": "${CHECKPOINT}",
  "checkpoint_sha256": "$([[ ${CHECKPOINT} == MOCK ]] && echo null || echo "${CHECKPOINT_SHA256}")",
  "driver": "${DRIVER_CONFIG}",
  "topology": "${TOPOLOGY}",
  "base_image": "${ALPASIM_BASE_IMAGE}",
  "scenes": "${SCENES}"
}
EOF

cd "${ALPASIM_ROOT}"
uv run --project src/wizard alpasim_wizard \
    deploy=local "topology=${TOPOLOGY}" "driver=${DRIVER_CONFIG}" \
    "driver.model.checkpoint_path=${CHECKPOINT}" \
    "defines.base_image=${ALPASIM_BASE_IMAGE}" \
    "wizard.log_dir=${LOG_DIR}" \
    "scenes.scene_ids=[${SCENES}]" \
    "hydra.searchpath=[file://${ALPASIM_ROOT}/plugins/alpasim_driver/alpasim_autoe2e/configs]" \
    "$@" 2>&1 | tee "${LOG_DIR}/wizard.log"

echo "Run ${RUN_NAME}: $(find "${LOG_DIR}/rollouts" -name rollout.asl 2>/dev/null | wc -l) rollout.asl, $(find "${LOG_DIR}" -name '*.mp4' 2>/dev/null | wc -l) mp4 in ${LOG_DIR}"
