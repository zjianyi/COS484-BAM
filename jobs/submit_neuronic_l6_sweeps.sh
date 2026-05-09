#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

# Follow-up sweeps for the strongest zero-shot layer found so far: L6.
#
# Baseline already run:
#   loss_k4_layer6: loss placement, k=4, train qa1-qa3 on 0k/1k/2k, d_attn=256, 20 epochs.
#
# New jobs launched here:
#   1) k sweep at L6: k=8,16 (compare to existing k=4 baseline)
#   2) length-supervision sweep: train on 0k,2k,16k
#   3) task-supervision sweep: train on qa1-qa10
#   4) attention-dimension sweep: d_attn=64,128 (compare to existing d_attn=256 baseline)
#   5) pre-QKV route gate: feature-wise gate before W_Q/W_K/W_V projection
#
# Usage from COS484-BAM repo root on Neuronic login:
#   bash jobs/submit_neuronic_l6_sweeps.sh

mkdir -p logs/neuronic checkpoints metrics/l6-sweeps

SBATCH_SCRIPT="jobs/neuronic_babilong_ablation.sbatch"

submit_run() {
  local run_id="$1"
  shift
  echo "Submitting ${run_id}"
  sbatch "$SBATCH_SCRIPT" "$run_id" "$@"
}

submit_route_gate_run() {
  local run_id="$1"
  shift
  echo "Submitting ${run_id}"
  BAM_TRAIN_MODULE=bam.train_babilong_route_gate_ablation sbatch "$SBATCH_SCRIPT" "$run_id" "$@"
}

COMMON_ARGS=(
  --placement loss
  --top-k 4
  --n-train 50
  --n-eval 50
  --epochs 20
  --train-tasks qa1,qa2,qa3
  --train-lengths 0k,1k,2k
  --eval-tasks qa1,qa2,qa3
  --eval-lengths 0k,1k,2k,4k,8k,16k
  --cache-layer-idx 6
  --d-attn 256
  --max-seq-len 16384
  --gate on
  --causal-mask on
  --loss-mode focused
)

for top_k in 8 16; do
  submit_run "l6_loss_k${top_k}" \
    "${COMMON_ARGS[@]}" \
    --top-k "$top_k" \
    --output "checkpoints/cache_babilong_l6_loss_k${top_k}.pt" \
    --metrics-output "metrics/l6-sweeps/l6_loss_k${top_k}.json"
done

submit_run l6_loss_k4_train_0k2k16k \
  "${COMMON_ARGS[@]}" \
  --train-lengths 0k,2k,16k \
  --output checkpoints/cache_babilong_l6_loss_k4_train_0k2k16k.pt \
  --metrics-output metrics/l6-sweeps/l6_loss_k4_train_0k2k16k.json

submit_run l6_loss_k4_train_qa1_qa10 \
  "${COMMON_ARGS[@]}" \
  --train-tasks qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10 \
  --output checkpoints/cache_babilong_l6_loss_k4_train_qa1_qa10.pt \
  --metrics-output metrics/l6-sweeps/l6_loss_k4_train_qa1_qa10.json

submit_route_gate_run l6_loss_k4_routegate_vector \
  "${COMMON_ARGS[@]}" \
  --output checkpoints/cache_babilong_l6_loss_k4_routegate_vector.pt \
  --metrics-output metrics/l6-sweeps/l6_loss_k4_routegate_vector.json

for d_attn in 64 128; do
  submit_run "l6_loss_k4_d${d_attn}" \
    "${COMMON_ARGS[@]}" \
    --d-attn "$d_attn" \
    --output "checkpoints/cache_babilong_l6_loss_k4_d${d_attn}.pt" \
    --metrics-output "metrics/l6-sweeps/l6_loss_k4_d${d_attn}.json"
done

echo "Submitted L6 sweeps. Existing baseline for comparison: metrics/layer-experiments/loss_k4_layer6.json"
