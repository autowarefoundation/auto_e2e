#!/usr/bin/env bash
# Side-by-side videos of the recording and every shifted clip, for the PR.
# Usage: make_videos.sh <scene_dir> <generated_dir> <out_dir> [seconds, default 4]
#   front.mp4, rear.mp4: rows = recording (10 Hz), then each clip in name order (0, 0.5 m, 1.0 m, 1.0 m + 3 deg);
#                        columns = left, centre, right camera of that side.
#   blend_front.mp4:     50% WSM control / 50% generation, rows = unshifted and 1.0 m + 3 deg clips.
set -euo pipefail

SCENE=$1 GEN=$2 OUT=$3 SECONDS_=${4:-4}
mkdir -p "${OUT}"
mapfile -t CLIPS < <(ls -d "${GEN}"/nuplan_* | sort)
TILE="scale=416:240,fps=30,setsar=1"

grid() {  # grid <out.mp4> <camera...>
  local out=$1; shift
  local inputs=() filter="" rows="" n=0 row
  for cam in "$@"; do inputs+=(-framerate 10 -pattern_type glob -i "${SCENE}/gt/${cam}/*.jpg"); done
  for clip in "${CLIPS[@]}"; do for cam in "$@"; do inputs+=(-i "$(ls "${clip}"/vision_view*_"${cam}".mp4)"); done; done
  for row in $(seq 0 ${#CLIPS[@]}); do
    for col in $(seq 0 $(($# - 1))); do filter+="[${n}:v]${TILE}[t${n}];"; n=$((n + 1)); done
    filter+="$(for i in $(seq $((n - $#)) $((n - 1))); do printf '[t%d]' "$i"; done)hstack=$#[r${row}];"
    rows+="[r${row}]"
  done
  ffmpeg -loglevel error -y "${inputs[@]}" -filter_complex "${filter}${rows}vstack=$((${#CLIPS[@]} + 1))" \
    -t "${SECONDS_}" -c:v libx264 -crf 28 -pix_fmt yuv420p "${out}"
}

grid "${OUT}/front.mp4" camera_cross_left_120fov camera_front_wide_120fov camera_cross_right_120fov
grid "${OUT}/rear.mp4" camera_rear_left_70fov camera_rear_tele_30fov camera_rear_right_70fov

inputs=() filter="" n=0
for clip in "${CLIPS[0]}" "${CLIPS[-1]}"; do
  for cam in camera_cross_left_120fov camera_front_wide_120fov camera_cross_right_120fov; do
    inputs+=(-i "$(ls "${clip}"/vision_view*_"${cam}".mp4)" -i "$(ls "${clip}"/control_view*_"${cam}".mp4)")
    filter+="[$((n * 2)):v]${TILE}[v${n}];[$((n * 2 + 1)):v]${TILE}[c${n}];[v${n}][c${n}]blend=all_mode=average[b${n}];"
    n=$((n + 1))
  done
done
ffmpeg -loglevel error -y "${inputs[@]}" -filter_complex \
  "${filter}[b0][b1][b2]hstack=3[r0];[b3][b4][b5]hstack=3[r1];[r0][r1]vstack=2" \
  -t "${SECONDS_}" -c:v libx264 -crf 28 -pix_fmt yuv420p "${OUT}/blend_front.mp4"
ls -la "${OUT}"
