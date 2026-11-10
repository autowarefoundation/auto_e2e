#!/usr/bin/env bash
# Closed-loop-style check after run.sh: a 20 s rollout of the same window in which the ego drifts
# 3 m to the left over 6 s and stays there, next to the unshifted rollout. A drift this large is where
# NuRec renders broke down in experiment/closed-loop; the WSM controls are rebuilt from the map instead.
# Usage: run_drift.sh   (after setup.sh, from the directory holding these scripts)
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/cosmos-nano-sim}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${ROOT}/wsm-venv/bin/python
START_US=1620859210179724   # the window run.sh picks, so the first frames match

"${PY}" -I "${HERE}/nuplan_to_rdshq.py" --out "${ROOT}/scenes_drift" --start-us "${START_US}" \
  --frames 601 --ramp-s 6 --shifts 0:0,3.0:0
SCENE=$(ls -d "${ROOT}"/scenes_drift/nuplan_*)
"${PY}" -I "${HERE}/render_and_request.py" --root "${ROOT}" --scene "${SCENE}" --captions "${HERE}/captions_vegas_wynn.json"
ROOT=${ROOT} bash "${HERE}/generate.sh" "${SCENE}/requests_480p_seed0.jsonl" "${ROOT}/outputs/drift_480p_seed0"
"${PY}" -I "${HERE}/compare.py" --scene "${SCENE}" --generated "${ROOT}/outputs/drift_480p_seed0" \
  --times 0,5,10,15,20 --out "${ROOT}/outputs/compare_drift"
bash "${HERE}/make_videos.sh" "${SCENE}" "${ROOT}/outputs/drift_480p_seed0" "${ROOT}/outputs/videos_drift" 20
