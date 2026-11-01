#!/usr/bin/env bash
# Whole experiment on one H100 (sm90) instance: setup, nuPlan to RDS-HQ, WSM render, Sim-Auto, comparison.
# 480p is the resolution NVIDIA times; 720p is closer to the 1024 px front camera AutoE2E reads.
# 480p with seed 1 separates seed-to-seed variation from the effect of the ego shift.
# Usage: run.sh   (run from the directory holding these scripts)
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/cosmos-nano-sim}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${ROOT}/wsm-venv/bin/python

ROOT=${ROOT} bash "${HERE}/setup.sh"
"${PY}" -I "${HERE}/nuplan_to_rdshq.py" --out "${ROOT}/scenes"
SCENE=$(ls -d "${ROOT}"/scenes/nuplan_*)
for RUN in 480:0 720:0 480:1; do
  RES=${RUN%:*} SEED=${RUN#*:} TAG=${RUN%:*}p_seed${RUN#*:}
  "${PY}" -I "${HERE}/render_and_request.py" --root "${ROOT}" --scene "${SCENE}" \
    --captions "${HERE}/captions_vegas_wynn.json" --resolution "${RES}" --seed "${SEED}"
  ROOT=${ROOT} bash "${HERE}/generate.sh" "${SCENE}/requests_${TAG}.jsonl" "${ROOT}/outputs/nuplan_${TAG}"
done
"${PY}" -I "${HERE}/compare.py" --scene "${SCENE}" --generated "${ROOT}/outputs/nuplan_480p_seed0" \
  --other-seed "${ROOT}/outputs/nuplan_480p_seed1" --out "${ROOT}/outputs/compare_480p"
"${PY}" -I "${HERE}/compare.py" --scene "${SCENE}" --generated "${ROOT}/outputs/nuplan_720p_seed0" \
  --out "${ROOT}/outputs/compare_720p"
