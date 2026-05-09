#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

# Three independent jobs:
#   1) Fixed-interval placement (k=4), StateCache at layer 2
#   2) Loss-triggered placement (k=4), StateCache at layer 6
#   3) Loss-triggered placement (k=4), StateCache at layer 16
#
# Each runs train_babilong_ablation (training + inline eval 0k–16k).
#
# Usage from COS484-BAM repo root on Neuronic login:
#   bash jobs/submit_neuronic_interval_l2_loss_l6_loss_l16.sh

mkdir -p logs/neuronic checkpoints metrics

SBATCH_SCRIPT="jobs/neuronic_babilong_ablation.sbatch"

submit_run() {
  local run_id="$1"
  shift
  echo "Submitting ${run_id}"
  sbatch "$SBATCH_SCRIPT" "$run_id" "$@"
}

COMMON_ARGS=(
  --n-train 50
  --n-eval 50
  --epochs 20
  --eval-lengths 0k,1k,2k,4k,8k,16k
  --d-attn 256
  --max-seq-len 16384
  --gate on
  --causal-mask on
  --loss-mode focused
)

submit_run interval_k4_layer2 \
  "${COMMON_ARGS[@]}" \
  --placement interval \
  --top-k 4 \
  --cache-layer-idx 2 \
  --output checkpoints/cache_babilong_interval_k4_layer2.pt \
  --metrics-output metrics/interval_k4_layer2.json

submit_run loss_k4_layer6 \
  "${COMMON_ARGS[@]}" \
  --placement loss \
  --top-k 4 \
  --cache-layer-idx 6 \
  --output checkpoints/cache_babilong_loss_k4_layer6.pt \
  --metrics-output metrics/loss_k4_layer6.json

submit_run loss_k4_layer16 \
  "${COMMON_ARGS[@]}" \
  --placement loss \
  --top-k 4 \
  --cache-layer-idx 16 \
  --output checkpoints/cache_babilong_loss_k4_layer16.pt \
  --metrics-output metrics/loss_k4_layer16.json

echo "Submitted interval_k4_layer2, loss_k4_layer6, loss_k4_layer16."
