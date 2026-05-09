#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

# Fixed-interval cache placement at injection layer 2, with TOP_K ∈ {6, 16}.
# Each job runs `train_babilong_ablation` (training + inline multi-length eval).
#
# Notation (matches docs/ablations_writeup.md): L2 = layer 2; L6/L16 here = six vs
# sixteen cache writes per example (not layer indices).
#
# If you meant fixed-interval k=4 at injection layers 2, 6, and 16 instead, use:
#   bash jobs/submit_neuronic_interval_k4_layers_2_6_16.sh
#
# Usage from COS484-BAM repo root on Neuronic login:
#   bash jobs/submit_neuronic_interval_layer2_k6_k16.sh

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
  --cache-layer-idx 2
  --d-attn 256
  --max-seq-len 16384
  --gate on
  --causal-mask on
  --loss-mode focused
)

submit_run interval_k6_layer2 \
  "${COMMON_ARGS[@]}" \
  --placement interval \
  --top-k 6 \
  --output checkpoints/cache_babilong_interval_k6_layer2.pt \
  --metrics-output metrics/interval_k6_layer2.json

submit_run interval_k16_layer2 \
  "${COMMON_ARGS[@]}" \
  --placement interval \
  --top-k 16 \
  --output checkpoints/cache_babilong_interval_k16_layer2.pt \
  --metrics-output metrics/interval_k16_layer2.json

echo "Done. Logs under logs/neuronic/; metrics: metrics/interval_k6_layer2*.json(l); metrics/interval_k16_layer2*.json(l)"
