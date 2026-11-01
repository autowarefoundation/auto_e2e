#!/usr/bin/env bash
# Prepare one GPU EC2 instance for the Cosmos3-Nano-Sim-Auto experiment.
# Everything lives under ROOT (the instance-store NVMe), so the root disk stays untouched.
# Re-running is safe: finished steps are skipped.
set -euo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/cosmos-nano-sim}
FRAMEWORK_COMMIT=55f0554d0bf167603432041c794a8babbac759b9   # NVIDIA/cosmos-framework, 2026-11-01
MODEL_REVISION=e289a50a4049614eaa62c9edc10b1e8d6cc1a6c1      # nvidia/Cosmos3-Nano-Sim-Auto
SAMPLE_REVISION=main                                         # shared sample in Cosmos3-Nano-Transfer-Auto

sudo mkdir -p "${ROOT}" && sudo chown "$(id -u):$(id -g)" "${ROOT}"
cd "${ROOT}"
export PATH="${ROOT}/bin:${PATH}" HF_HOME="${ROOT}/hf" UV_CACHE_DIR="${ROOT}/uv-cache"

# Renderer runtime (EGL + H.264) from the WSM cookbook install section.
sudo apt-get update -q
sudo apt-get install -y -q ffmpeg libgl1 libegl1 libx11-dev libgl-dev libegl-dev build-essential python3-dev

UV_VERSION=0.11.29   # cosmos-framework pyproject requires exactly this uv
if [ "$("${ROOT}/bin/uv" --version 2>/dev/null | cut -d' ' -f2)" != "${UV_VERSION}" ]; then
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR="${ROOT}/bin" UV_NO_MODIFY_PATH=1 sh
fi

if [ ! -d cosmos-framework ]; then
  git clone -q https://github.com/NVIDIA/cosmos-framework.git cosmos-framework
fi
git -C cosmos-framework fetch -q origin "${FRAMEWORK_COMMIT}"
git -C cosmos-framework checkout -q "${FRAMEWORK_COMMIT}"

# Inference environment as in docs/inference_auto.md, without the guardrail extra (run with --no-guardrails).
(cd cosmos-framework && uv sync --locked --python 3.13 --group cu130-train --extra train)

# WSM4 renderer in its own environment, as the cookbook asks.
WSM_DIR=cosmos-framework/.agents/skills/cookbooks/cosmos3/simulation/auto/wsm
if [ ! -x wsm-venv/bin/cosmos-auto-wm ]; then
  uv venv -q --python 3.12 wsm-venv
  uv pip install -q --python wsm-venv/bin/python "./${WSM_DIR}"
fi
# nuplan_to_rdshq.py reads the nuPlan archives and map in the same environment.
uv pip install -q --python wsm-venv/bin/python s3fs pyogrio geopandas shapely pyproj opencv-python-headless

HF=cosmos-framework/.venv/bin/hf
"${HF}" download nvidia/Cosmos3-Nano-Sim-Auto --revision "${MODEL_REVISION}" \
  --local-dir checkpoints/Cosmos3-Nano-Sim-Auto
"${HF}" download nvidia/Cosmos3-Nano-Transfer-Auto --revision "${SAMPLE_REVISION}" \
  --include 'assets/sample_data/*' --local-dir auto-example
(cd auto-example/assets/sample_data && sha256sum -c --quiet SHA256SUMS)

cosmos-framework/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
wsm-venv/bin/python -c 'import moderngl; c = moderngl.create_context(standalone=True, backend="egl", require=330); print(c.info["GL_RENDERER"]); c.release()'
du -sh "${ROOT}"/* | sort -h
