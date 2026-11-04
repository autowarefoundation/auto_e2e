#!/usr/bin/env bash
# Run Sim-Auto on NVIDIA's shared sample with fewer views, frames or a lower resolution,
# to check that the model runs on this GPU and to measure speed and memory.
# Usage: sample_run.sh <num_views> <resolution 480|720> <num_frames (4k+1)>
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/cosmos-nano-sim}
VIEWS=${1:-1}
RES=${2:-480}
FRAMES=${3:-21}
SAMPLE=${ROOT}/auto-example/assets/sample_data
NAME=sample_v${VIEWS}_${RES}p_f${FRAMES}

# Front camera first, so one view means the front-wide camera. RGB only: the lidar block is dropped.
python3 - "${SAMPLE}/input-sim-auto.jsonl" "${SAMPLE}/${NAME}.jsonl" "${VIEWS}" "${RES}" "${FRAMES}" <<'EOF'
import json, sys
src, dst, views, res, frames = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
request = json.loads(open(src).read())
request.pop("lidar")
order = sorted(request["multiview"]["views"], key=lambda v: v["camera_key"] != "camera_front_wide_120fov")
request["multiview"]["views"] = order[:views]
request.update(resolution=res, num_frames=frames, num_video_frames_per_chunk=frames)
open(dst, "w").write(json.dumps(request) + "\n")
EOF

ROOT=${ROOT} bash "$(dirname "$0")/generate.sh" "${SAMPLE}/${NAME}.jsonl" "${ROOT}/outputs/${NAME}"
