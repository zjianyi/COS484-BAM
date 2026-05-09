#!/usr/bin/env bash
set -euo pipefail

# Submit optional extra-compute BABILong sweeps as separate Neuronic jobs.
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
  --placement loss
  --top-k 4
  --n-train 50
  --n-eval 50
  --epochs 20
  --eval-lengths 0k,1k,2k,4k,8k,16k
  --cache-layer-idx -2
  --max-seq-len 16384
  --gate on
  --causal-mask on
  --loss-mode focused
)

for d_attn in 64 128 256 512; do
  submit_run "loss_k4_d${d_attn}" \
    "${COMMON_ARGS[@]}" \
    --d-attn "$d_attn" \
    --output "checkpoints/cache_babilong_loss_k4_d${d_attn}.pt" \
    --metrics-output "metrics/loss_k4_d${d_attn}.json"
done

for max_entries in 1 2 4 8 64; do
  submit_run "loss_k4_entries${max_entries}" \
    "${COMMON_ARGS[@]}" \
    --max-entries "$max_entries" \
    --output "checkpoints/cache_babilong_loss_k4_entries${max_entries}.pt" \
    --metrics-output "metrics/loss_k4_entries${max_entries}.json"
done

for n_train in 10 25 50; do
  submit_run "loss_k4_n${n_train}" \
    "${COMMON_ARGS[@]}" \
    --n-train "$n_train" \
    --output "checkpoints/cache_babilong_loss_k4_n${n_train}.pt" \
    --metrics-output "metrics/loss_k4_n${n_train}.json"
done

for epochs in 5 10 20 40; do
  submit_run "loss_k4_ep${epochs}" \
    "${COMMON_ARGS[@]}" \
    --epochs "$epochs" \
    --output "checkpoints/cache_babilong_loss_k4_ep${epochs}.pt" \
    --metrics-output "metrics/loss_k4_ep${epochs}.json"
done
