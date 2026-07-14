#!/bin/bash
# 17-batch full-episode data-prep + IL train + eval for L2D (#121).
#
# Design: the 100k L2D episodes cannot fit in one Flyte execution — the
# per-partition graph nodes would exceed the Propeller gRPC 4MB event limit
# (empirical safe max ~2000 nodes). Instead we launch 17 back-to-back
# executions of wf_create_dataset_sharded, each covering ~6k episodes with
# 60 concurrent partitions (~180 graph nodes = 4% of the ceiling). After
# every batch completes, we collect its shard-directory outputs. Once all
# 17 are done, a single wf_train_il execution over the merged shard list
# does IL train + eval.
#
# Idempotence: per-batch marker files under $STATE_DIR mark success so
# re-running the script skips completed batches. Retries: up to 3 attempts
# per batch (transient Karpenter / Cosmos / Flyte SYSTEM errors).
#
# Runtime: expected ~11 h data-prep + ~2 h train + eval = ~13 h total.
# Run from EC2 inside tmux — DO NOT run from CodeBuild (8 h cap).
#
# Prereqs:
#   * EC2 with pyflyte + flyte config accessible
#   * Flyte project/domain: auto-e2e/development
#   * Image + register run for the current commit already done (via CodeBuild)

set -euo pipefail

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
FLYTE_CONFIG="${FLYTE_CONFIG:-$HOME/.flyte/config.yaml}"
FLYTE_PROJECT="${FLYTE_PROJECT:-auto-e2e}"
FLYTE_DOMAIN="${FLYTE_DOMAIN:-development}"
WORKFLOW_MODULE="${WORKFLOW_MODULE:-Platform/pipelines/workflows.py}"

TOTAL_EPISODES="${TOTAL_EPISODES:-100000}"     # L2D total on main branch
NUM_BATCHES="${NUM_BATCHES:-17}"               # 100k / 6k ≈ 17
PARTITION_SIZE="${PARTITION_SIZE:-100}"        # 60 concurrent partitions/batch

DATASET="${DATASET:-yaak-ai/L2D}"
REASONING_TEACHER="${REASONING_TEACHER:-openai_compatible}"
BACKBONE="${BACKBONE:-swin_v2_tiny}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LR="${LR:-1e-4}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_PARTITIONS="${MAX_PARTITIONS:-2000}"

STATE_DIR="${STATE_DIR:-$HOME/.auto-e2e-fullrun}"
mkdir -p "$STATE_DIR"

# ----------------------------------------------------------------------
# Compute batch ranges
# ----------------------------------------------------------------------
# Even split: batch i covers [i * BATCH_SIZE_EPS, (i+1) * BATCH_SIZE_EPS)
# clipped to TOTAL_EPISODES on the last batch.
BATCH_SIZE_EPS=$(( (TOTAL_EPISODES + NUM_BATCHES - 1) / NUM_BATCHES ))
echo "Config: $TOTAL_EPISODES eps split into $NUM_BATCHES batches of ~$BATCH_SIZE_EPS eps each"
echo "        Each batch fans out at partition_size=$PARTITION_SIZE"

# ----------------------------------------------------------------------
# run_batch <i>
# ----------------------------------------------------------------------
run_batch() {
  local idx="$1"
  local start_ep=$((idx * BATCH_SIZE_EPS))
  local end_ep=$(((idx + 1) * BATCH_SIZE_EPS))
  if [ "$end_ep" -gt "$TOTAL_EPISODES" ]; then
    end_ep="$TOTAL_EPISODES"
  fi
  local marker="$STATE_DIR/batch_${idx}.done"
  local exec_id_file="$STATE_DIR/batch_${idx}.exec"

  if [ -f "$marker" ]; then
    echo "[batch $idx] already SUCCEEDED (exec $(cat "$exec_id_file" 2>/dev/null || echo '?')). Skipping."
    return 0
  fi

  echo "[batch $idx] eps [$start_ep, $end_ep) -- $((end_ep - start_ep)) episodes"

  local attempt
  for attempt in 1 2 3; do
    echo "[batch $idx] attempt $attempt/3"
    # `--wait` blocks until terminal state; stream execution ID once known.
    if pyflyte --config "$FLYTE_CONFIG" run --remote --wait \
        --project "$FLYTE_PROJECT" --domain "$FLYTE_DOMAIN" \
        "$WORKFLOW_MODULE" wf_create_dataset_sharded \
        --dataset "$DATASET" \
        --episodes 0 \
        --start_ep "$start_ep" \
        --end_ep "$end_ep" \
        --partition_size "$PARTITION_SIZE" \
        --reasoning_teacher "$REASONING_TEACHER" \
        --max_partitions "$MAX_PARTITIONS" \
        --world_model false \
        2>&1 | tee "$STATE_DIR/batch_${idx}.log"; then
      # Extract execution ID from log — pyflyte prints "Execution ..." lines.
      grep -oE 'a[a-z0-9]{20}' "$STATE_DIR/batch_${idx}.log" | head -1 > "$exec_id_file" || true
      touch "$marker"
      echo "[batch $idx] SUCCEEDED"
      return 0
    fi
    echo "[batch $idx] FAILED on attempt $attempt (retrying after 60s)"
    sleep 60
  done

  echo "[batch $idx] EXHAUSTED retries. Manual intervention needed."
  return 1
}

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
echo "=== 17-batch full-run start: $(date -Iseconds) ==="
for i in $(seq 0 $((NUM_BATCHES - 1))); do
  run_batch "$i"
done
echo "=== all $NUM_BATCHES batches complete: $(date -Iseconds) ==="

# ----------------------------------------------------------------------
# Collect shard dirs + launch training
# ----------------------------------------------------------------------
echo ""
echo "=== Collecting shard dirs from $NUM_BATCHES batch executions ==="
"$(dirname "$0")/collect_shards.py" \
  --state-dir "$STATE_DIR" \
  --project "$FLYTE_PROJECT" --domain "$FLYTE_DOMAIN" \
  --output "$STATE_DIR/all_shards.json"

echo ""
echo "=== Launching wf_train_il on merged shard list ==="
pyflyte --config "$FLYTE_CONFIG" run --remote --wait \
  --project "$FLYTE_PROJECT" --domain "$FLYTE_DOMAIN" \
  "$WORKFLOW_MODULE" wf_train_il \
  --inputs-file "$STATE_DIR/all_shards.json" \
  --dataset "$DATASET" \
  --backbone "$BACKBONE" \
  --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" --lr "$LR" \
  --enable_reasoning true --reasoning_mode pooled_latent \
  --enable_world_model true \
  --val_fraction "$VAL_FRACTION" --num_workers "$NUM_WORKERS" \
  2>&1 | tee "$STATE_DIR/train.log"

echo "=== full-run complete: $(date -Iseconds) ==="
