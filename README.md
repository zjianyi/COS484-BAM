# BAM: Bolt-on Associative Memory for FalconMamba 7B

COS484 Final Project, Princeton University, Spring 2026.

StateCache is a 4.2M-parameter cross-attention memory module that bolts onto a frozen FalconMamba 7B backbone. It intercepts the model's hidden state at layer 62, writes compressed snapshots at selected `[CACHE]` token positions, and retrieves relevant entries at query time — all without modifying any backbone weights. The overall pipeline remains O(n) because the cache cross-attention operates over k=4 entries (constant), not the full sequence.

See `writeup.md` for the full project notes and results.

---

## Setup

**Requirements:** Python 3.10+, CUDA 12.8, single GPU with ≥40 GB VRAM (tested on H100).

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
  train_babilong.py        # Training: regex-triggered placement (main result)
  train_babilong_loss.py   # Training: loss-triggered placement (ablation)
  eval_babilong.py         # Generate-based baseline eval (no cache)
  loss_probe.py            # Diagnostic: per-token loss on facts vs distractors
  generator.py             # BAMGenerator (autoregressive inference with cache)
  __init__.py

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

### 1. Generate-based baseline (no cache)

Evaluates FalconMamba 7B + SFT adapter on BABILong qa1/qa2/qa3 at 0k–4k context using standard generation. This is the comparison baseline for the paper.

```bash
python -m bam.eval_babilong --tasks qa1 qa2 qa3 --lengths 0k 1k 2k 4k --n_examples 50
```

### 2. Train StateCache — regex-triggered placement

Inserts `[CACHE]` after entity-movement fact sentences ("X went to Y"). Main result.

```bash
python -m bam.train_babilong > babilong_train.log 2>&1
```

### 3. Train StateCache — loss-triggered placement

Inserts `[CACHE]` at top-K highest-loss token positions (model's own surprise signal). No task-specific rules — the ablation showing the approach generalizes.

```bash
python -m bam.train_babilong_loss > babilong_train_loss.log 2>&1
```

### 4. Loss probe diagnostic

Verifies that entity-movement sentences have higher in-context loss than distractors, validating the loss-triggered placement hypothesis.

```bash
python -m bam.loss_probe
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

- [ ] **Random placement** — k=4 random positions instead of regex/loss. Tests whether placement matters or just the existence of the cache.
- [ ] **Layer index** — layer 32 vs 62. Formally validates the gradient path argument on BABILong.
- [ ] **K ablation** — top-1, 4, 8 for loss-triggered. How many cache tokens are needed?
- [ ] **No gate** — W_gate fixed to 1. Does learned gating contribute?

---

## GPU setup (Lambda Cloud)

Experiments were run on a Lambda Cloud H100 instance (`lambda-gpu-1xh100`). Model weights and adapter are at `/home/ubuntu/final_project/` on that instance. Checkpoints saved as `bam/cache_babilong.pt` (regex) and `bam/cache_babilong_loss.pt` (loss-triggered).
