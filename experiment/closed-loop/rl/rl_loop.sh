#!/bin/bash
# Closed-loop GRPO on one scene: roll out the policy in AlpaSim, take one GRPO
# step on the planner from that rollout (grpo_step.py), and repeat with the new
# planner. The last iteration only rolls out, to evaluate the final policy.
#
#   bash experiment/closed-loop/rl/rl_loop.sh
#   ITERATIONS=5 SCENE=clipgt-... bash experiment/closed-loop/rl/rl_loop.sh
#
# Checkpoints go to $ALPASIM_ROOT/data/drivers/autoe2e/rl/<tag>/ (only the
# newest is kept); rollouts to $RUNS_ROOT/<tag>-iter-<n>/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOSED_LOOP="$(dirname "${HERE}")"
REPO_ROOT="$(cd "${CLOSED_LOOP}/../.." && pwd)"
# shellcheck source=../config.env
source "${CLOSED_LOOP}/config.env"

ITERATIONS="${ITERATIONS:-3}"
SCENE="${SCENE:-$(grep -vE '^\s*(#|$)' "${CLOSED_LOOP}/scenes.txt" | head -1)}"
# 7 s on the log, then 4 s (about 40 decisions) in closed loop: the imitation
# policy leaves the 3 m bound within 2-3 s (#221), so later steps are truncated.
SIM_STEPS="${SIM_STEPS:-110}"
TAG="${TAG:-grpo-$(date -u +%Y%m%dT%H%M%SZ)}"
RUNS_ROOT="${RUNS_ROOT:-${HOME}/closed-loop-runs}"
DRIVERS="${ALPASIM_ROOT}/data/drivers"
RL_DIR="autoe2e/rl/${TAG}"
REFERENCE="autoe2e/checkpoint.pt"
mkdir -p "${DRIVERS}/${RL_DIR}"

# The driver runs this branch's plugin and model code.
rsync -a --delete --exclude __pycache__ \
    "${REPO_ROOT}/Model/plugins/alpasim_driver/" "${ALPASIM_ROOT}/plugins/alpasim_driver/"
bash "${CLOSED_LOOP}/stage_driver.sh"

checkpoint="${REFERENCE}"
for ((iteration = 0; iteration <= ITERATIONS; iteration++)); do
    run="${TAG}-iter-${iteration}"
    RUN_NAME="${run}" SCENES="${SCENE}" CHECKPOINT="/mnt/drivers/${checkpoint}" \
        CHECKPOINT_SHA256="$(sha256sum "${DRIVERS}/${checkpoint}" | cut -d' ' -f1)" \
        bash "${CLOSED_LOOP}/run_scenes.sh" "runtime.simulation_config.n_sim_steps=${SIM_STEPS}"
    (cd "${ALPASIM_ROOT}" && uv run --no-sync python "${CLOSED_LOOP}/summarize_run.py" "${RUNS_ROOT}/${run}")
    ((iteration == ITERATIONS)) && break

    next="${RL_DIR}/iter-$((iteration + 1)).pt"
    docker run --rm --gpus all \
        -v "${RUNS_ROOT}/${run}:/rundir:ro" \
        -v "${DRIVERS}:/mnt/drivers" \
        -v "${ALPASIM_ROOT}/data/nre-artifacts:/mnt/nre-data:ro" \
        -v "${ALPASIM_ROOT}/src:/repo/src:ro" \
        -v "${REPO_ROOT}:/src:ro" \
        -e AUTOE2E_MODEL_ROOT=/src/Model \
        -e PYTHONPATH=/src/Model/plugins/alpasim_driver:/src/Model/plugins/alpasim_driver/tools \
        "${ALPASIM_BASE_IMAGE}" bash -lc "cd /repo && uv run --no-sync python \
            /src/experiment/closed-loop/rl/grpo_step.py /rundir /mnt/drivers/${checkpoint} \
            --reference /mnt/drivers/${REFERENCE} --output /mnt/drivers/${next}" \
        | tee "${RUNS_ROOT}/${run}/grpo_step.log"
    # Keep the newest planner only; each step's JSON stays next to it.
    [[ "${checkpoint}" == "${REFERENCE}" ]] || rm -f "${DRIVERS}/${checkpoint}"
    checkpoint="${next}"
done
echo "GRPO ${TAG}: ${ITERATIONS} steps on ${SCENE}; final checkpoint ${DRIVERS}/${checkpoint}"
