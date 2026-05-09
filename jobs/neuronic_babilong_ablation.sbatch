#!/bin/bash
#SBATCH --job-name=bam-ablation
#SBATCH --output=logs/neuronic/%x-%j.out
#SBATCH --error=logs/neuronic/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch $0 RUN_ID [train_babilong_ablation args...]"
  exit 2
fi

RUN_ID="$1"
shift

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR"

mkdir -p logs/neuronic checkpoints metrics
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

echo "run_id=$RUN_ID"
echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "scratch_dir=/scratch/$USER/bam_runs/$SLURM_JOB_ID"
echo "train_module=${BAM_TRAIN_MODULE:-bam.train_babilong_ablation}"
nvidia-smi || true

"${PYTHON_CMD[@]}" -m "${BAM_TRAIN_MODULE:-bam.train_babilong_ablation}" \
  --run-id "$RUN_ID" \
  --cells-output "metrics/${RUN_ID}.cells.jsonl" \
  "$@" \
  > "logs/neuronic/${RUN_ID}_${SLURM_JOB_ID}.log" 2>&1

rm -rf "/scratch/$USER/bam_runs/$SLURM_JOB_ID"
