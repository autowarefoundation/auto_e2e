#!/usr/bin/env bash
# Run Sim-Auto on one request file on a single GPU, logging GPU memory and wall time.
# Usage: generate.sh <requests.jsonl> <output_dir>
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/cosmos-nano-sim}
# The framework shells out to uv to fetch the Wan2.2 VAE from Hugging Face on first use.
export PATH="${ROOT}/bin:${PATH}" HF_HOME="${ROOT}/hf" UV_CACHE_DIR="${ROOT}/uv-cache"
# 6 views at 720p ran out of memory on an 80 GB H100 with 4.5 GiB reserved but unused; let segments grow.
export PYTORCH_ALLOC_CONF=expandable_segments:True
REQUESTS=$(realpath "$1")
OUT=$(realpath -m "$2")
mkdir -p "${OUT}"

cd "${ROOT}/cosmos-framework"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "${OUT}/gpu_mem_mib.log" &
SMI=$!
START=$(date +%s)
COSMOS_TRAINING=0 COSMOS_INTERNAL=0 .venv/bin/torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  -i "${REQUESTS}" -o "${OUT}" \
  --checkpoint-path "${ROOT}/checkpoints/Cosmos3-Nano-Sim-Auto" \
  --no-guardrails --no-use-torch-compile --no-diffusion-cache 2>&1 | tee "${OUT}/inference.log"
END=$(date +%s)
kill "${SMI}"
echo "requests=$(wc -l < "${REQUESTS}") wall_s=$((END - START)) peak_gpu_mib=$(sort -n "${OUT}/gpu_mem_mib.log" | tail -1)" | tee "${OUT}/summary.txt"
