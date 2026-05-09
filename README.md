# BAM: Bolt-on Associative Memory for FalconMamba 7B

COS484 Final Project, Princeton University, Spring 2026.

StateCache is a 4.2M-parameter cross-attention memory module that bolts onto a frozen FalconMamba 7B backbone. It intercepts the model's hidden state at layer 62, writes compressed snapshots at selected `[CACHE]` token positions, and retrieves relevant entries at query time — all without modifying any backbone weights. The overall pipeline remains O(n) because the cache cross-attention operates over k=4 entries (constant), not the full sequence.

See `writeup.md` for historical project notes and early Phase 2 numbers. **Current BABILong reporting** (protocols, prompts, and LaTeX/PDF tables) is summarized below and in `docs/`.

---

## BABILong evaluation protocols (read this before comparing numbers)

**What `[CACHE]` means.** Special tokens (`[CACHE]`) are **inserted into the prompt text**. They mark fixed positions where **StateCache** can write/read compressed snapshots in the hidden state. This is **not** HuggingFace KV-cache or disk cache.

**Prompts differ across protocols.** You cannot mix rows from different tables without reading the caption:

| Protocol | Script | Few-shot? | `[CACHE]` in prompt? | Main metric to report |
|----------|--------|-----------|----------------------|----------------------|
| Plain no-cache baseline | `eval_babilongv2.py` | Optional (`--few-shot-examples`; default `0`; use `>0` for paper few-shot + **`--babilong-scoring next_token`** for candidate log-probs) | **No** | Next-token accuracy vs candidate answers |
| Inline ablation (train + eval) | `train_babilong_ablation.py` | **No** | **Yes** (placement policy) | `cache_acc` with StateCache; `baseline_acc` = scaffold-only control |
| Paper-style + cache scaffold | `eval_babilong_paper_scaffold_checkpoint.py` | **Yes** (paper BABILong layout) | **Yes** | `cache_acc` |
| Legacy chat generate | `eval_babilong.py` | Chat template | No `[CACHE]` in user text (vocab may still be resized) | Diagnostic only |

**`cache_acc` vs `baseline_acc` (inline JSON metrics).**

- **`cache_acc`:** `[CACHE]` tokens present **and** StateCache applied — the trained module’s accuracy.
- **`baseline_acc`:** **Same tokenized prompt** (same `[CACHE]` placements), but **no** StateCache delta — *scaffold-only*. This checks whether the backbone can still answer with extra tokens in the string; it is **not** comparable to `eval_babilongv2` and often collapses toward 0%. Do **not** treat it as the “no cache” baseline for external comparison.

**Fair baselines.**

- **vs plain Falcon-Mamba (no `[CACHE]`):** use `eval_babilongv2` with `--few-shot-examples 0` (zero-shot) or match few-shot count to your scaffold eval; compare to **`metrics/zero_shot_baseline/`** or reasoning-curves JSON as documented below. For **few-shot with true candidate logit scoring** (same `top_logprobs` protocol as zero-shot, not prefix-on-generation), run `--babilong-scoring next_token --few-shot-examples 2` — Slurm: **`jobs/neuronic_babilong_fewshot_next_token_baseline.sbatch`** → `metrics/fewshot_babilong/no_cache_baseline_next_token_qa1_qa3.json`.
- **vs StateCache under the same prompt:** compare **`cache_acc`** to **`baseline_acc`** from the **same** run (matched scaffold).

**Few-shot vs zero-shot tail behavior.** Few-shot paper prompts prepend instructions + examples, so **total sequence length** is much larger than the short plain prompt at the same nominal BABILong length (e.g. `16k`). Long-context accuracy can drop sharply for few-shot while zero-shot stays moderate — same distractor passage, **different total tokens and prefix**.

---

## Results tables and metrics layout

**Paper-ready PDFs** (also `.tex` fragments for Overleaf) live under `docs/`:

| File | What it shows |
|------|----------------|
| `docs/ablation_results_table.tex` | **Zero-shot only:** plain no-cache baseline (`metrics/zero_shot_baseline/…`, `eval_babilongv2`) + StateCache `cache_acc` from root `metrics/<run_id>.json`. Do **not** mix with few-shot cells. |
| `docs/ablation_cache_scaffold_zero_shot_table.tex` | **Zero-shot + `[CACHE]` scaffold:** scaffold-only (`metrics/scaffold_baselines/…`) + same `cache_acc` as main zero-shot StateCache rows. |
| `docs/fewshot_babilong_results_table.tex` | **Few-shot only:** `metrics/fewshot_babilong/` (`no_cache_baseline.json` + per-run `cache_acc`). Different protocol than zero-shot table. |

Table numbers were filled from local `metrics/**/*.json` when building those docs (QA1–QA3 × six lengths; rounded to one decimal); those paths are not in git—keep your Neuronic rsync or regenerate.

**Suggested `metrics/` layout (convention):** this directory is **gitignored**—nothing under `metrics/` is committed; regenerate or copy from your cluster run.


- `metrics/fewshot_babilong/` — outputs from `eval_babilong_paper_scaffold_checkpoint.py`
- `metrics/zero_shot_baseline/` or `metrics/no_fewshot_baseline/` — `eval_babilongv2` plain baseline JSON
- `metrics/scaffold_baselines/` — optional dedicated scaffold-only reruns (`eval_babilong_scaffold_baseline.py`)
- Top-level `metrics/*.json` — often copies of Neuronic inline ablation runs (`train_babilong_ablation.py`)

Cluster workflow details: `docs/NEURONIC_RUNBOOK.md`.

**Run catalog (each ablation, metrics paths, good vs bad):** `docs/ablations_writeup.md`.

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

writeup.md                 # Historical notes + early Phase 2 tables
docs/                      # LaTeX/PDF result tables, NEURONIC_RUNBOOK, ablation plans
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

The three primary protocols (expanded table above) are:

- **Reasoning-curves no-cache baseline** — `eval_babilongv2.py`: no `[CACHE]` tokens, no StateCache. Use this for **external** comparison to “plain” Falcon-Mamba.
- **Inline StateCache scaffold eval** — `train_babilong_ablation.py`: `[CACHE]` in passage + read site; **`cache_acc`** vs **`baseline_acc`** (scaffold-only). Same prompt shape for both numbers inside one JSON.
- **Paper-style StateCache scaffold eval** — `eval_babilong_paper_scaffold_checkpoint.py`: reasoning-curves-style few-shot layout + `[CACHE]`; outputs used in **`metrics/fewshot_babilong/`** and `docs/fewshot_babilong_results_table.*`.

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

To evaluate an existing checkpoint on additional BABILong tasks without
retraining, use:

```bash
python -m bam.eval_babilong_ablation_checkpoint \
  --checkpoint checkpoints/cache_babilong_loss_k4_layer62.pt \
  --metrics-output metrics/loss_k4_layer62.json \
  --append-metrics \
  --eval-tasks qa4,qa5,qa6,qa7,qa8,qa9,qa10 \
  --eval-lengths 0k,1k,2k,4k,8k,16k \
  --n-eval 50
```

To evaluate the same checkpoint with the more comparable paper-style scaffold:

```bash
python -m bam.eval_babilong_paper_scaffold_checkpoint \
  --checkpoint checkpoints/cache_babilong_loss_k4_layer62.pt \
  --metrics-output metrics/paper_scaffold/loss_k4_layer62.json \
  --cells-output metrics/paper_scaffold/loss_k4_layer62.cells.jsonl \
  --eval-tasks qa1,qa2,qa3 \
  --eval-lengths 0k,1k,2k,4k,8k,16k \
  --n-eval 50
```

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
