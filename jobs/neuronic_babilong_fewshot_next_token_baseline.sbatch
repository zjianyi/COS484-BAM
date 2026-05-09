#!/bin/bash
#SBATCH --job-name=bam-fs-nt
#SBATCH --output=logs/neuronic/%x-%j.out
#SBATCH --error=logs/neuronic/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR"

mkdir -p logs/neuronic metrics/fewshot_babilong
mkdir -p "/scratch/$USER/hf" "/scratch/$USER/bam_runs/$SLURM_JOB_ID" "/scratch/$USER/torchinductor"

export HF_HOME="/scratch/$USER/hf"
export HF_DATASETS_CACHE="/scratch/$USER/hf/datasets"
export TRANSFORMERS_CACHE="/scratch/$USER/hf/transformers"
export HF_HUB_CACHE="/scratch/$USER/hf/hub"
export TORCHINDUCTOR_CACHE_DIR="/scratch/$USER/torchinductor"
export TMPDIR="/scratch/$USER/bam_runs/$SLURM_JOB_ID/tmp"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR"

PYTHON_CMD=(python)
if command -v pixi >/dev/null 2>&1 && [ -f "pixi.toml" ]; then
  PYTHON_CMD=(pixi run python)
elif [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

RUN_ID="fewshot_next_token_qa1_qa3"

echo "run_id=$RUN_ID job_id=$SLURM_JOB_ID node=$(hostname)"
nvidia-smi || true

"${PYTHON_CMD[@]}" -m bam.eval_babilongv2 \
  --mamba-model tiiuae/falcon-mamba-7b-instruct \
  --tasks qa1,qa2,qa3 \
  --context-lengths 0k,1k,2k,4k,8k,16k \
  --limit 50 \
  --batch-size 1 \
  --babilong-scoring next_token \
  --few-shot-examples 2 \
  --output "metrics/fewshot_babilong/no_cache_baseline_next_token_qa1_qa3.json" \
  --cells-output "metrics/fewshot_babilong/no_cache_baseline_next_token_qa1_qa3.cells.jsonl" \
  > "logs/neuronic/${RUN_ID}_${SLURM_JOB_ID}.log" 2>&1

rm -rf "/scratch/$USER/bam_runs/$SLURM_JOB_ID"
