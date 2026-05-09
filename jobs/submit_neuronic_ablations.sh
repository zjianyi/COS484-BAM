#!/usr/bin/env bash
set -euo pipefail

# Submit one independent Neuronic job per planned BABILong ablation.
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
  --max-seq-len 16384
  --gate on
  --causal-mask on
  --loss-mode focused
)

submit_run regex_layer62 \
  "${COMMON_ARGS[@]}" \
  --placement regex \
  --output checkpoints/cache_babilong_regex_layer62.pt \
  --metrics-output metrics/regex_layer62.json

submit_run loss_k4_layer62 \
  "${COMMON_ARGS[@]}" \
  --placement loss \
  --top-k 4 \
  --output checkpoints/cache_babilong_loss_k4_layer62.pt \
  --metrics-output metrics/loss_k4_layer62.json

submit_run random_k4_layer62 \
  "${COMMON_ARGS[@]}" \
  --placement random \
  --top-k 4 \
  --output checkpoints/cache_babilong_random_k4_layer62.pt \
  --metrics-output metrics/random_k4_layer62.json

submit_run interval_k4_layer62 \
  "${COMMON_ARGS[@]}" \
  --placement interval \
  --top-k 4 \
  --output checkpoints/cache_babilong_interval_k4_layer62.pt \
  --metrics-output metrics/interval_k4_layer62.json

submit_run loss_k1_layer62 \
  "${COMMON_ARGS[@]}" \
  --placement loss \
  --top-k 1 \
  --output checkpoints/cache_babilong_loss_k1_layer62.pt \
  --metrics-output metrics/loss_k1_layer62.json

submit_run loss_k8_layer62 \
  "${COMMON_ARGS[@]}" \
  --placement loss \
  --top-k 8 \
  --output checkpoints/cache_babilong_loss_k8_layer62.pt \
  --metrics-output metrics/loss_k8_layer62.json

submit_run loss_k4_train8k_layer62 \
  "${COMMON_ARGS[@]}" \
  --placement loss \
  --top-k 4 \
  --train-lengths 0k,1k,2k,4k,8k \
  --output checkpoints/cache_babilong_loss_k4_train8k_layer62.pt \
  --metrics-output metrics/loss_k4_train8k_layer62.json

submit_run loss_k4_layer32 \
  "${COMMON_ARGS[@]}" \
  --placement loss \
  --top-k 4 \
  --cache-layer-idx 32 \
  --output checkpoints/cache_babilong_loss_k4_layer32.pt \
  --metrics-output metrics/loss_k4_layer32.json

submit_run loss_k4_nogate \
  "${COMMON_ARGS[@]}" \
  --placement loss \
  --top-k 4 \
  --gate off \
  --output checkpoints/cache_babilong_loss_k4_nogate.pt \
  --metrics-output metrics/loss_k4_nogate.json

submit_run loss_k4_nocausal \
  "${COMMON_ARGS[@]}" \
  --placement loss \
  --top-k 4 \
  --causal-mask off \
  --output checkpoints/cache_babilong_loss_k4_nocausal.pt \
  --metrics-output metrics/loss_k4_nocausal.json

submit_run loss_k4_fullloss \
  "${COMMON_ARGS[@]}" \
  --placement loss \
  --top-k 4 \
  --loss-mode full \
  --output checkpoints/cache_babilong_loss_k4_fullloss.pt \
  --metrics-output metrics/loss_k4_fullloss.json

submit_run regex_cap4_layer62 \
  "${COMMON_ARGS[@]}" \
  --placement regex \
  --regex-cap-k 4 \
  --output checkpoints/cache_babilong_regex_cap4_layer62.pt \
  --metrics-output metrics/regex_cap4_layer62.json

