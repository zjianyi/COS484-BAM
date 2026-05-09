# Ablations v1: Minimal Neuronic Run Set

This is the minimal ablation set to make the StateCache story solid without
launching every optional sweep. Run from the `COS484-BAM` project root on the
Neuronic login node. Do not run training on the login node.

## Baseline

Use the existing reasoning-curves no-cache baseline; do not rerun it unless
you need a reproduction.

```text
../reasoning-curves/runs/recall_eval/merged_final/babilong_tinker_llama_nemotron_falcon_mamba_final.json
```

This baseline covers `tiiuae/falcon-mamba-7b-instruct`, generative BABILong
scoring, `qa1`-`qa10`, `0k`-`16k`, and `limit=50`.

## Setup

`COS484-BAM` includes a `pixi.toml` for the Neuronic runs. Pixi is not lighter
in terms of installed GPU packages, but it is more reproducible than hand-made
conda/venv environments. It creates a Python 3.10 Linux environment and installs
the CUDA 12.8 PyTorch wheel plus `causal-conv1d` and `mamba-ssm`.

If `pixi install` fails with a `requests`/`datasets` conflict involving the
PyTorch CUDA index, make sure you have the latest `pixi.toml`. It sets
`index-strategy = "unsafe-best-match"` so normal packages resolve from PyPI
while PyTorch resolves from `https://download.pytorch.org/whl/cu128`.

If `causal-conv1d` tries to build from source against CUDA 13.1, make sure the
latest `pixi.toml` is present. It installs prebuilt GitHub wheels directly for:

- `causal-conv1d` for Python 3.10, CUDA 12, Torch 2.7, Linux x86_64.
- `mamba-ssm` for Python 3.10, CUDA 12, Torch 2.7, Linux x86_64.

```bash
cd "$HOME/COS484-BAM"
mkdir -p logs/neuronic checkpoints metrics

rm -rf .pixi pixi.lock
pixi install
```

Make sure `.env` contains a HuggingFace token with access to FalconMamba:

```bash
cat > .env <<'EOF'
HF_TOKEN=your_huggingface_token_here
EOF
```

Verify the stack before submitting jobs:

```bash
pixi run python --version
pixi run python -c "import torch; print(torch.__version__, torch.version.cuda)"
pixi run python -m pip show causal-conv1d mamba-ssm transformers
pixi run check-falcon-mamba
```

The Neuronic SLURM wrapper automatically uses `pixi run python` when `pixi` is
available and `pixi.toml` exists. If Pixi is not available on the cluster, fall
back to the `requirements.txt` venv/conda setup from `docs/ABLATION_RUN_PLAN.md`.

## Submit v1 Jobs

These jobs all train on `0k,1k,2k` unless otherwise noted, evaluate on
`0k,1k,2k,4k,8k,16k`, and use `--max-seq-len 16384`.

```bash
sbatch jobs/neuronic_babilong_ablation.sbatch loss_k4_layer62 \
  --placement loss \
  --top-k 4 \
  --train-lengths 0k,1k,2k \
  --eval-lengths 0k,1k,2k,4k,8k,16k \
  --max-seq-len 16384 \
  --cache-layer-idx -2 \
  --output checkpoints/cache_babilong_loss_k4_layer62.pt \
  --metrics-output metrics/loss_k4_layer62.json

sbatch jobs/neuronic_babilong_ablation.sbatch regex_layer62 \
  --placement regex \
  --train-lengths 0k,1k,2k \
  --eval-lengths 0k,1k,2k,4k,8k,16k \
  --max-seq-len 16384 \
  --cache-layer-idx -2 \
  --output checkpoints/cache_babilong_regex_layer62.pt \
  --metrics-output metrics/regex_layer62.json

sbatch jobs/neuronic_babilong_ablation.sbatch random_k4_layer62 \
  --placement random \
  --top-k 4 \
  --train-lengths 0k,1k,2k \
  --eval-lengths 0k,1k,2k,4k,8k,16k \
  --max-seq-len 16384 \
  --cache-layer-idx -2 \
  --output checkpoints/cache_babilong_random_k4_layer62.pt \
  --metrics-output metrics/random_k4_layer62.json

sbatch jobs/neuronic_babilong_ablation.sbatch interval_k4_layer62 \
  --placement interval \
  --top-k 4 \
  --train-lengths 0k,1k,2k \
  --eval-lengths 0k,1k,2k,4k,8k,16k \
  --max-seq-len 16384 \
  --cache-layer-idx -2 \
  --output checkpoints/cache_babilong_interval_k4_layer62.pt \
  --metrics-output metrics/interval_k4_layer62.json

sbatch jobs/neuronic_babilong_ablation.sbatch loss_k4_layer32 \
  --placement loss \
  --top-k 4 \
  --train-lengths 0k,1k,2k \
  --eval-lengths 0k,1k,2k,4k,8k,16k \
  --max-seq-len 16384 \
  --cache-layer-idx 32 \
  --output checkpoints/cache_babilong_loss_k4_layer32.pt \
  --metrics-output metrics/loss_k4_layer32.json

sbatch jobs/neuronic_babilong_ablation.sbatch loss_k4_nogate \
  --placement loss \
  --top-k 4 \
  --train-lengths 0k,1k,2k \
  --eval-lengths 0k,1k,2k,4k,8k,16k \
  --max-seq-len 16384 \
  --cache-layer-idx -2 \
  --gate off \
  --output checkpoints/cache_babilong_loss_k4_nogate.pt \
  --metrics-output metrics/loss_k4_nogate.json

sbatch jobs/neuronic_babilong_ablation.sbatch loss_k4_train8k_layer62 \
  --placement loss \
  --top-k 4 \
  --train-lengths 0k,1k,2k,4k,8k \
  --eval-lengths 0k,1k,2k,4k,8k,16k \
  --max-seq-len 16384 \
  --cache-layer-idx -2 \
  --output checkpoints/cache_babilong_loss_k4_train8k_layer62.pt \
  --metrics-output metrics/loss_k4_train8k_layer62.json
```

## What Each Job Tests

- `loss_k4_layer62`: main StateCache result.
- `regex_layer62`: rule-based placement reference.
- `random_k4_layer62`: whether learned memory helps when placement is arbitrary.
- `interval_k4_layer62`: whether simple position sampling is enough.
- `loss_k4_layer32`: whether late-layer injection matters.
- `loss_k4_nogate`: whether the learned gate matters.
- `loss_k4_train8k_layer62`: whether training on longer contexts changes 8k/16k performance.

## Monitor

```bash
squeue -u "$USER"
tail -f logs/neuronic/loss_k4_layer62_<jobid>.log
tail -f logs/neuronic/bam-ablation-<jobid>.out
```

## Outputs

Each job writes:

```text
checkpoints/<run_id>.pt
metrics/<run_id>.json
logs/neuronic/<run_id>_<jobid>.log
logs/neuronic/bam-ablation-<jobid>.out
```

Use the `metrics/*.json` files for tables. They include config, runtime,
cache-count statistics, and per-task/per-length eval metrics.
