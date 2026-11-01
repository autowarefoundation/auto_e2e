#!/usr/bin/env bash
# Whole experiment on one H100 (sm90) instance: setup, nuPlan to RDS-HQ, WSM render, Sim-Auto, comparison.
# 480p is the resolution NVIDIA times; 720p is closer to the 1024 px front camera AutoE2E reads.
# Usage: run.sh   (run from the directory holding these scripts)
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/cosmos-nano-sim}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${ROOT}/wsm-venv/bin/python

ROOT=${ROOT} bash "${HERE}/setup.sh"
"${PY}" -I "${HERE}/nuplan_to_rdshq.py" --out "${ROOT}/scenes"
SCENE=$(ls -d "${ROOT}"/scenes/nuplan_*)
for RES in 480 720; do
  "${PY}" -I "${HERE}/render_and_request.py" --root "${ROOT}" --scene "${SCENE}" \
    --captions "${HERE}/captions_vegas_wynn.json" --resolution "${RES}"
  ROOT=${ROOT} bash "${HERE}/generate.sh" "${SCENE}/requests_${RES}p.jsonl" "${ROOT}/outputs/nuplan_${RES}p"
  "${PY}" -I "${HERE}/compare.py" --scene "${SCENE}" --generated "${ROOT}/outputs/nuplan_${RES}p" \
    --out "${ROOT}/outputs/compare_${RES}p"
done
