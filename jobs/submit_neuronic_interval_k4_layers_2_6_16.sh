#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

# Fixed-interval placement with k=4; vary StateCache injection layer (2, 6, 16).
# Each job: train + inline eval (0k–16k). Compare to `interval_k4_layer62` in submit_neuronic_ablations.sh.
#
# Usage from COS484-BAM repo root:
#   bash jobs/submit_neuronic_interval_k4_layers_2_6_16.sh

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

for layer in 2 6 16; do
  submit_run "interval_k4_layer${layer}" \
    "${COMMON_ARGS[@]}" \
    --placement interval \
    --top-k 4 \
    --cache-layer-idx "${layer}" \
    --output "checkpoints/cache_babilong_interval_k4_layer${layer}.pt" \
    --metrics-output "metrics/interval_k4_layer${layer}.json"
done

echo "Done."
