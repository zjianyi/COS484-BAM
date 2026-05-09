# BAM: Bolt-on Associative Memory for FalconMamba 7B

COS484 Final Project, Princeton University, Spring 2026.

StateCache is a 4.2M-parameter cross-attention memory module that bolts onto a frozen FalconMamba 7B backbone. It intercepts the model's hidden state at layer 62, writes compressed snapshots at selected `[CACHE]` token positions, and retrieves relevant entries at query time — all without modifying any backbone weights. The overall pipeline remains O(n) because the cache cross-attention operates over k=4 entries (constant), not the full sequence.

See `writeup.md` for the full project notes and results.

---

## Setup

**Requirements:** Python 3.10+, CUDA 12.8, single GPU with ≥40 GB VRAM (tested on H100).

Recommended for Neuronic:

```bash
pixi install
pixi run check-falcon-mamba
```

Fallback without Pixi:

```bash
pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**HuggingFace token:** Create a `.env` file in the project root:
```
HF_TOKEN=hf_...
```
FalconMamba 7B Instruct requires accepting the model license on HuggingFace first.

---

## Project structure

```
bam/
  cache.py                 # StateCache module (core architecture)
  train_babilong_ablation.py # Parameterized BABILong cache ablation runner
  train_babilong.py        # Legacy/reference regex-triggered training script
  train_babilong_loss.py   # Legacy/reference loss-triggered training script
  eval_babilongv2.py       # Recommended paper-style BABILong baseline eval
  eval_babilong.py         # Legacy chat-template generate eval
  loss_probe.py            # Diagnostic: per-token loss on facts vs distractors
  generator.py             # BAMGenerator (autoregressive inference with cache)
  __init__.py

jobs/
  neuronic_babilong_ablation.sbatch      # Single Neuronic ablation job wrapper
  submit_neuronic_ablations.sh           # Main ablation batch submitter
  submit_neuronic_optional_sweeps.sh     # Extra width/data/epoch sweeps
  submit_neuronic_seed_replicates.sh     # Seed replicate submitter

writeup.md                 # Project notes, all results, paper framing
requirements.txt
```

**Legacy (Phase 1, OpenMath — did not pan out):**
```
train.py / evaluate.py     # OpenMath SFT training and eval
bam/train_cache.py         # Cache training for math reasoning
bam/evaluate_bam.py        # BAM eval on OpenMath
bam/probe_cache.py         # Cache diagnostics
bam/inspect_model.py       # Model inspection utilities
```

---

## Running experiments

All scripts are run as Python modules from the project root.

### 1. Existing BABILong baseline

Use the already-completed reasoning-curves FalconMamba no-cache baseline with
approximate first-token scoring:

`../reasoning-curves/runs/recall_eval/merged_final/babilong_tinker_llama_nemotron_falcon_mamba_final.json`

For comparing against StateCache, use:

`../reasoning-curves/runs/recall_eval/merged_final/babilong_approx_next_token_prefix_eval.json`

This file evaluates saved greedy generations by their first generated
token/prefix, which is the closest baseline to the StateCache eval's
`argmax(logits[-1])` next-token decision. It covers
`tiiuae/falcon-mamba-7b-instruct`, `qa1`-`qa10`, `0k`-`16k`, and `limit=50`.
No Neuronic baseline rerun is needed unless you want a local reproduction inside
this repo. The full generated-answer baseline file is still useful as a
standard BABILong-style reference, but the approximate first-token file is the
right comparison for the cache ablations.

### 2. Train StateCache / run ablations

Use the parameterized ablation runner for new runs. It trains a StateCache checkpoint and then runs inline BABILong eval.

```bash
python -m bam.train_babilong_ablation \
  --placement loss \
  --top-k 4 \
  --cache-layer-idx -2 \
  --train-lengths 0k,1k,2k \
  --eval-lengths 0k,1k,2k,4k,8k,16k \
  --max-seq-len 16384 \
  --output checkpoints/cache_babilong_loss_k4_layer62.pt \
  --metrics-output metrics/loss_k4_layer62.json \
  > logs/loss_k4_layer62.log 2>&1
```

Common placement policies:

- `--placement regex`: insert `[CACHE]` after entity-movement fact sentences.
- `--placement loss`: insert `[CACHE]` at top-K highest-loss token positions.
- `--placement random`: random K-position control.
- `--placement interval`: evenly spaced K-position control.

### 3. Neuronic cluster jobs

On Neuronic, do not run training on the login node. Submit separate SLURM jobs:

```bash
mkdir -p logs/neuronic checkpoints metrics
bash jobs/submit_neuronic_ablations.sh
```

Optional extra sweeps and seed replicates:

```bash
bash jobs/submit_neuronic_optional_sweeps.sh
bash jobs/submit_neuronic_seed_replicates.sh
```

The Neuronic wrapper stores HF/dataset/Torch caches on `/scratch/$USER` and writes checkpoints/metrics back to `checkpoints/` and `metrics/`.

### 4. Diagnostics

Verifies that entity-movement sentences have higher in-context loss than distractors, validating the loss-triggered placement hypothesis.

```bash
python -m bam.loss_probe
```

The older chat-template baseline remains available as a diagnostic, but should not be the reported BABILong baseline:

```bash
python -m bam.eval_babilong --tasks qa1 qa2 qa3 --lengths 0k 1k 2k 4k --n_examples 50
```

---

## Key results

| task | length | baseline | regex cache | loss-triggered |
|------|--------|----------|-------------|----------------|
| qa1  | 0k     | 0.32     | 0.24        | 0.14           |
| qa1  | 1k     | 0.02     | 0.22        | 0.20           |
| qa1  | 4k     | 0.02     | 0.18        | 0.14           |
| qa2  | 1k     | 0.00     | 0.18        | 0.18           |
| qa2  | 4k     | 0.00     | 0.24        | 0.18           |
| qa3  | 1k     | 0.00     | 0.14        | 0.18           |
| qa3  | 4k     | 0.02     | 0.18        | 0.14           |
| **avg** |     | **0.033**| **0.175**   | **0.165**      |

FalconMamba 7B collapses from 32% → 2% as context grows from 0k to 1k on qa1. StateCache recovers 14–24% uniformly across all lengths including 4k (unseen during training). Loss-triggered placement matches regex placement without any domain-specific rules.

---

## Ablations to run (TODO)

- [ ] **Baseline** — use the existing reasoning-curves FalconMamba BABILong baseline.
- [ ] **Placement controls** — regex, loss-triggered, random K=4, and fixed-interval K=4.
- [ ] **Layer index** — layer 32 vs 62. Formally validates the gradient path argument on BABILong.
- [ ] **K ablation** — top-1, 4, 8 for loss-triggered. How many cache tokens are needed?
- [ ] **Training length** — train through `2k` vs train through `8k`, both evaluated through `16k`.
- [ ] **Architecture ablations** — no gate, no causal mask, full-loss instead of focused answer loss.
- [ ] **Capacity and robustness** — `D_ATTN`, `MAX_ENTRIES`, data size, epoch count, and seed replicates.

See `docs/ABLATION_RUN_PLAN.md` for Neuronic job scripts and exact commands.

---

## Previous GPU setup (Lambda Cloud)

Experiments were run on a Lambda Cloud H100 instance (`lambda-gpu-1xh100`). Model weights and adapter are at `/home/ubuntu/final_project/` on that instance. Checkpoints saved as `bam/cache_babilong.pt` (regex) and `bam/cache_babilong_loss.pt` (loss-triggered).
