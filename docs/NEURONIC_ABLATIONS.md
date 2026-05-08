# Running BAM Ablations on the Neuronic Cluster

This document summarizes how to run ablations for this codebase on
`neuronic.cs.princeton.edu`.

## Cluster Rules That Matter

Access uses your Princeton OIT LDAP NetID and password:

```bash
ssh <netid>@neuronic.cs.princeton.edu
```

Use the login node only for setup, light testing, and job submission. Do not run
FalconMamba training or evaluation directly on the login node.

Each compute node has:

- 2 x Intel Xeon Gold 5320 CPUs
- 512 GB RAM
- 8 x NVIDIA L40 GPUs
- 3.5 TB local SSD scratch at `/scratch`
- NFS-mounted home and project storage
- 10 Gbps Ethernet, no Infiniband

Use `/scratch/$USER` for data/model staging inside jobs:

```bash
mkdir -p /scratch/$USER
```

`/scratch` is temporary, not backed up, and routinely purged. Copy important
logs/checkpoints back to your home or project directory before the job exits.

## Repo Overview

The active BABILong experiments are:

- `bam/train_babilong.py`: regex-triggered StateCache placement.
- `bam/train_babilong_loss.py`: loss-triggered StateCache placement.
- `bam/eval_babilong.py`: generate-based baseline without a trained cache.
- `bam/cache.py`: StateCache module.

Legacy OpenMath scripts exist, but the current result path is BABILong.

Main default settings:

- Model: `tiiuae/falcon-mamba-7b-instruct`
- Training tasks: `qa1`, `qa2`, `qa3`
- Training lengths: `0k`, `1k`, `2k`
- Eval lengths: `0k`, `1k`, `2k`, `4k`
- Examples: 50 per task/length combo
- Epochs: 20
- Max sequence length: 1536
- Cache layer: `CACHE_LAYER_IDX = -2`, which resolves to layer 62 of 64
- Cache width: `D_ATTN = 256`
- Cache entries: `MAX_ENTRIES = 64`
- Loss-triggered cache positions: `TOP_K = 4`

## One-Time Setup

From the login node:

```bash
git clone <repo-url> COS484-BAM
cd COS484-BAM
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Create `.env` in the project root:

```bash
HF_TOKEN=hf_...
```

You must accept the FalconMamba model license on HuggingFace before the model
download will work.

For cluster jobs, keep HuggingFace and dataset caches on `/scratch` to avoid
hammering NFS:

```bash
export HF_HOME=/scratch/$USER/hf
export HF_DATASETS_CACHE=/scratch/$USER/hf/datasets
export TRANSFORMERS_CACHE=/scratch/$USER/hf/transformers
```

The first job on a node may spend time downloading model and dataset files.

## Minimal Smoke Test

Submit this only as a compute job, not on the login node:

```bash
python -m bam.eval_babilong --tasks qa1 --lengths 0k --n_examples 2 --no_adapter
```

This verifies CUDA, model download, HuggingFace auth, and BABILong loading.

## Batch Job Template

Neuronic scheduler details are not included in the cluster description above.
If the cluster uses Slurm, the following is a starting point. Confirm partition,
account, wall time, and GPU flags with `sinfo`, `squeue`, or local docs.

Save as `jobs/run_babilong_loss_k4.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=bam-loss-k4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

cd "$HOME/COS484-BAM"
source .venv/bin/activate

mkdir -p logs checkpoints /scratch/$USER/hf /scratch/$USER/bam_runs/$SLURM_JOB_ID

export HF_HOME=/scratch/$USER/hf
export HF_DATASETS_CACHE=/scratch/$USER/hf/datasets
export TRANSFORMERS_CACHE=/scratch/$USER/hf/transformers
export TORCHINDUCTOR_CACHE_DIR=/scratch/$USER/torchinductor

python -m bam.train_babilong_loss \
  > logs/babilong_loss_k4_${SLURM_JOB_ID}.log 2>&1

cp bam/cache_babilong_loss.pt checkpoints/cache_babilong_loss_k4_${SLURM_JOB_ID}.pt
cp bam/cache_babilong_loss.ep*.pt checkpoints/ 2>/dev/null || true
```

Submit:

```bash
mkdir -p logs jobs checkpoints
sbatch jobs/run_babilong_loss_k4.sbatch
```

Monitor:

```bash
squeue -u $USER
tail -f logs/bam-loss-k4-<jobid>.out
tail -f logs/babilong_loss_k4_<jobid>.log
```

## Baseline Runs

Run the no-cache generate baseline:

```bash
python -m bam.eval_babilong \
  --tasks qa1 qa2 qa3 \
  --lengths 0k 1k 2k 4k \
  --n_examples 50 \
  > logs/baseline_generate_${SLURM_JOB_ID}.log 2>&1
```

Run the two current cache placements:

```bash
python -m bam.train_babilong \
  > logs/regex_cache_${SLURM_JOB_ID}.log 2>&1

python -m bam.train_babilong_loss \
  > logs/loss_cache_k4_${SLURM_JOB_ID}.log 2>&1
```

Each training script performs inline eval after training and prints a table like:

```text
=== BABILong Cache Eval (...) ===
task    metric         0k     1k     2k     4k
qa1     no_cache     ...
qa1     cache        ...
```

The saved cache checkpoint path is controlled by `CACHE_OUT_PATH`.

## Recommended Ablation Plan

Run these in separate branches, copied scripts, or small patches. Since the
current scripts use top-level constants instead of CLI flags, change the
constants before submitting each job and encode the setting in `CACHE_OUT_PATH`
and the log filename.

### 1. Placement Policy

Goal: test whether selective placement matters.

Existing conditions:

- Regex placement: `bam/train_babilong.py`
- Loss-triggered placement: `bam/train_babilong_loss.py`

Missing condition:

- Random placement: create a variant of `bam/train_babilong_loss.py` where
  `probe_cache_positions(...)` is replaced by random passage positions.

Keep `TOP_K = 4`, `CACHE_LAYER_IDX = -2`, and all training/eval settings fixed.

Suggested output names:

```python
CACHE_OUT_PATH = "bam/cache_babilong_random_k4.pt"
```

### 2. Layer Index

Goal: validate that late-layer injection is better than mid-layer injection.

Edit in `bam/train_babilong.py` or `bam/train_babilong_loss.py`:

```python
CACHE_LAYER_IDX = 32
CACHE_OUT_PATH = "bam/cache_babilong_loss_layer32.pt"
```

Compare against:

```python
CACHE_LAYER_IDX = -2
CACHE_OUT_PATH = "bam/cache_babilong_loss_layer62.pt"
```

The expected interpretation is that layer 62 gives the cache a shorter frozen
backward path than layer 32.

### 3. K Ablation

Goal: measure how many loss-triggered cache writes are needed.

Edit `bam/train_babilong_loss.py`:

```python
TOP_K = 1
CACHE_OUT_PATH = "bam/cache_babilong_loss_k1.pt"
```

Then run:

```python
TOP_K = 4
CACHE_OUT_PATH = "bam/cache_babilong_loss_k4.pt"
```

And:

```python
TOP_K = 8
CACHE_OUT_PATH = "bam/cache_babilong_loss_k8.pt"
```

Keep `MAX_ENTRIES = 64` so the K sweep is not confounded by eviction.

### 4. Gate Ablation

Goal: test whether learned gating helps.

The gate is used in:

- `bam/train_babilong.py`
- `bam/train_babilong_loss.py`
- `bam/cache.py`

For the training scripts, replace:

```python
gate = torch.sigmoid(cache.W_gate(...))
d = cache.W_out(out) * gate
```

with:

```python
d = cache.W_out(out)
```

For loss-triggered eval in `bam/train_babilong_loss.py`, replace:

```python
gt = torch.sigmoid(cache.W_gate(...))
h[0, c_pos[any_av]] = h[0, c_pos[any_av]] + cache.W_out(ot).mul(gt).to(h.dtype)
```

with:

```python
h[0, c_pos[any_av]] = h[0, c_pos[any_av]] + cache.W_out(ot).to(h.dtype)
```

Use:

```python
CACHE_OUT_PATH = "bam/cache_babilong_loss_k4_nogate.pt"
```

Do not compare a no-gate checkpoint using a gated eval path.

## Result Tracking

Create a simple TSV row per job:

```text
run_id	placement	layer	top_k	gate	task	length	no_cache	cache	log_path	ckpt_path
```

Record:

- Git commit hash: `git rev-parse HEAD`
- Job ID
- Changed constants
- Final eval table
- Checkpoint path
- Whether the run completed all epochs

Useful commands:

```bash
grep -n "num_layers\\|cache_layer_idx\\|Precomputing\\|epoch .* done\\|BABILong Cache Eval\\|Total time" logs/*.log
grep -A20 "BABILong Cache Eval" logs/*.log
```

## Practical Notes

- One GPU should be enough for the current scripts. Do not request all 8 GPUs
  unless the code is changed for distributed training.
- The scripts freeze FalconMamba and train only `StateCache`, but still load the
  7B backbone, so expect large VRAM use.
- Use `/scratch/$USER` for HuggingFace caches and temporary run files.
- Copy final checkpoints and logs back to persistent storage before job exit.
- Avoid many simultaneous first-time model downloads from NFS-backed locations.
- Keep one ablation variable changed at a time.
- Keep `N_TRAIN`, `N_EVAL`, tasks, lengths, seed, and eval protocol fixed across
  the main comparison table.

## Suggested Run Order

1. Smoke test: `qa1`, `0k`, 2 examples, no adapter.
2. Baseline generate eval.
3. Reproduce regex placement.
4. Reproduce loss-triggered `TOP_K = 4`.
5. Run `TOP_K = 1` and `TOP_K = 8`.
6. Run layer 32 vs layer 62.
7. Run random placement.
8. Run no-gate.

Do not start large sweeps until steps 1-4 reproduce sane numbers.
