#!/usr/bin/env bash
# Whole experiment on one H100 (sm90) instance: setup, nuPlan to RDS-HQ, WSM render, Sim-Auto, comparison.
# Six views at 720p do not fit in 80 GB with this CLI, so the clips run at 480p, with seed 0 and seed 1
# to separate seed-to-seed variation from the effect of the ego shift.
# Usage: run.sh   (run from the directory holding these scripts)
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/cosmos-nano-sim}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${ROOT}/wsm-venv/bin/python

ROOT=${ROOT} bash "${HERE}/setup.sh"
"${PY}" -I "${HERE}/nuplan_to_rdshq.py" --out "${ROOT}/scenes"
SCENE=$(ls -d "${ROOT}"/scenes/nuplan_*)
for SEED in 0 1; do
  "${PY}" -I "${HERE}/render_and_request.py" --root "${ROOT}" --scene "${SCENE}" \
    --captions "${HERE}/captions_vegas_wynn.json" --seed "${SEED}"
  ROOT=${ROOT} bash "${HERE}/generate.sh" "${SCENE}/requests_480p_seed${SEED}.jsonl" "${ROOT}/outputs/nuplan_480p_seed${SEED}"
done
"${PY}" -I "${HERE}/compare.py" --scene "${SCENE}" --generated "${ROOT}/outputs/nuplan_480p_seed0" \
  --other-seed "${ROOT}/outputs/nuplan_480p_seed1" --out "${ROOT}/outputs/compare_480p"
