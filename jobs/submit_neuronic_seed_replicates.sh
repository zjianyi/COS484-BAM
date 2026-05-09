#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

# Submit reviewer-facing seed replicates for the core BABILong claims.
# Run from the COS484-BAM project root on the login node.

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
  --cache-layer-idx -2
  --d-attn 256
  --max-entries 64
  --max-seq-len 16384
  --gate on
  --causal-mask on
  --loss-mode focused
)

for seed in 41 42 43; do
  submit_run "loss_k4_seed${seed}" \
    "${COMMON_ARGS[@]}" \
    --placement loss \
    --top-k 4 \
    --seed "$seed" \
    --output "checkpoints/cache_babilong_loss_k4_seed${seed}.pt" \
    --metrics-output "metrics/loss_k4_seed${seed}.json"

  submit_run "random_k4_seed${seed}" \
    "${COMMON_ARGS[@]}" \
    --placement random \
    --top-k 4 \
    --seed "$seed" \
    --output "checkpoints/cache_babilong_random_k4_seed${seed}.pt" \
    --metrics-output "metrics/random_k4_seed${seed}.json"
done
