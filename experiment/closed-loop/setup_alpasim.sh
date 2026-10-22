#!/bin/bash
# Prepare a GPU host for the closed-loop experiment: install the host tools
# AlpaSim's wizard needs, check out the pinned AlpaSim commit, add the AutoE2E
# driver plugin as a workspace member and build the driver image.
#
# Run from anywhere inside this repository on the GPU host:
#   bash experiment/closed-loop/setup_alpasim.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
# shellcheck source=config.env
source "${HERE}/config.env"

# uv >= 0.9.17 parses AlpaSim's relative exclude-newer; cargo builds utils_rs.
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv self update || true
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

if [[ ! -d "${ALPASIM_ROOT}/.git" ]]; then
    # The LFS sample artifacts are test data; scenes come from Hugging Face.
    GIT_LFS_SKIP_SMUDGE=1 git clone "${ALPASIM_REPO}" "${ALPASIM_ROOT}"
fi
git -C "${ALPASIM_ROOT}" fetch --quiet origin "${ALPASIM_COMMIT}"
git -C "${ALPASIM_ROOT}" checkout --quiet --force "${ALPASIM_COMMIT}"

# Docker cannot follow symlinks out of the build context, so copy the plugin.
rm -rf "${ALPASIM_ROOT}/plugins/alpasim_driver"
cp -r "${REPO_ROOT}/Model/plugins/alpasim_driver" "${ALPASIM_ROOT}/plugins/"

# Register the plugin as an extra and a workspace source, as for transfuser.
python3 - "${ALPASIM_ROOT}/pyproject.toml" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path).read()
if 'autoe2e = ["alpasim_autoe2e"]' not in text:
    text = text.replace(
        'transfuser = ["alpasim_transfuser"]',
        'transfuser = ["alpasim_transfuser"]\nautoe2e = ["alpasim_autoe2e"]',
    )
if "alpasim_autoe2e = { workspace = true }" not in text:
    text = text.replace(
        "alpasim_transfuser = { workspace = true }",
        "alpasim_transfuser = { workspace = true }\nalpasim_autoe2e = { workspace = true }",
    )
assert re.search(r'^autoe2e = \["alpasim_autoe2e"\]$', text, re.M), "extra not added"
assert "alpasim_autoe2e = { workspace = true }" in text, "source not added"
open(path, "w").write(text)
PY
if ! grep -q -- "--extra autoe2e" "${ALPASIM_ROOT}/Dockerfile"; then
    sed -i 's/uv sync --extra all --extra recipes/uv sync --extra all --extra recipes --extra autoe2e/' \
        "${ALPASIM_ROOT}/Dockerfile"
fi
grep -q -- "--extra autoe2e" "${ALPASIM_ROOT}/Dockerfile"

# Host side: protos, utils_rs and the wizard. The script must be sourced.
(cd "${ALPASIM_ROOT}" && source ./setup_local_env.sh)

docker build -t "${ALPASIM_BASE_IMAGE}" "${ALPASIM_ROOT}"
docker pull nvcr.io/nvidia/nre/nre-ga:26.04

# The plugin must be discoverable inside the image, or driver=autoe2e fails late.
docker run --rm "${ALPASIM_BASE_IMAGE}" uv run alpasim-info | tee /dev/stderr | grep -q autoe2e
echo "AlpaSim ${ALPASIM_COMMIT:0:12} ready at ${ALPASIM_ROOT}, image ${ALPASIM_BASE_IMAGE}"
