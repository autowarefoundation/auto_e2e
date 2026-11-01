#!/usr/bin/env bash
# Whole experiment on one H100 (sm90) instance: setup, nuPlan to RDS-HQ, WSM render, Sim-Auto, comparison.
# Usage: run.sh   (run from the directory holding these scripts)
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/cosmos-nano-sim}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${ROOT}/wsm-venv/bin/python

ROOT=${ROOT} bash "${HERE}/setup.sh"
"${PY}" -I "${HERE}/nuplan_to_rdshq.py" --out "${ROOT}/scenes"
SCENE=$(ls -d "${ROOT}"/scenes/nuplan_*)
"${PY}" -I "${HERE}/render_and_request.py" --root "${ROOT}" --scene "${SCENE}" --captions "${HERE}/captions_vegas_wynn.json"
ROOT=${ROOT} bash "${HERE}/generate.sh" "${SCENE}/requests.jsonl" "${ROOT}/outputs/nuplan"
"${PY}" -I "${HERE}/compare.py" --scene "${SCENE}" --generated "${ROOT}/outputs/nuplan" --out "${ROOT}/outputs/compare"
