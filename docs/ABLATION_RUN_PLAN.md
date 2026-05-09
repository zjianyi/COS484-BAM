# BAM Ablation Run Plan

This is the recommended set of ablations to run for the BABILong StateCache
result. The current reference run is 450 training examples, 20 epochs, training
on `qa1`/`qa2`/`qa3` across `0k`/`1k`/`2k`, and eval across
`0k`/`1k`/`2k`/`4k`/`8k`/`16k`.

Time estimates below assume the model and datasets are already downloaded. Add
10-30 minutes for first-time HuggingFace downloads or slow shared storage.
Expect L40 runs to be roughly 2-4x slower than H100 runs.

## Parameterized Runner

Most ablations can now be launched with:

```bash
python -m bam.train_babilong_ablation \
  --placement loss \
  --top-k 4 \
  --cache-layer-idx -2 \
  --gate on \
  --causal-mask on \
  --loss-mode focused \
  --d-attn 256 \
  --train-lengths 0k,1k,2k \
  --eval-lengths 0k,1k,2k,4k,8k,16k \
  --max-seq-len 16384 \
  --n-train 50 \
  --n-eval 50 \
  --epochs 20 \
  --output bam/cache_babilong_loss_k4_layer62.pt \
  > logs/loss_k4_layer62.log 2>&1
```

Run commands from the project root. Create `logs/` first if needed:

```bash
mkdir -p logs bam/metrics
```

## Neuronic Cluster Submission

Do not run training on the Neuronic login node. Use it only to prepare the
environment and submit jobs.

The repo includes two helper scripts:

- `jobs/neuronic_babilong_ablation.sbatch`: reusable single-run SLURM job.
- `jobs/submit_neuronic_ablations.sh`: submits one separate job for each
  planned ablation, so no single job has a long wall time.
- `jobs/submit_neuronic_optional_sweeps.sh`: submits the extra-compute width,
  data-size, and epoch sweeps as separate jobs.
- `jobs/submit_neuronic_seed_replicates.sh`: submits three-seed replicates for
  the loss-triggered and random-placement comparison.

Before submitting, set up the environment on the login node:

```bash
cd "$HOME/COS484-BAM"
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p logs/neuronic checkpoints metrics
```

Make sure your HuggingFace token is available, either in `.env` or the job
environment:

```bash
cat > .env <<'EOF'
HF_TOKEN=your_huggingface_token_here
EOF
```

The Neuronic job wrapper stores HuggingFace, dataset, and TorchInductor caches
under `/scratch/$USER`:

```bash
HF_HOME=/scratch/$USER/hf
HF_DATASETS_CACHE=/scratch/$USER/hf/datasets
TRANSFORMERS_CACHE=/scratch/$USER/hf/transformers
TORCHINDUCTOR_CACHE_DIR=/scratch/$USER/torchinductor
```

Check SLURM settings before the first run. The template requests one L40 GPU,
8 CPUs, 96GB RAM, and 4 hours. Standard jobs train only through `2k`, evaluate
through `16k`, and use `--max-seq-len 16384`; shorter examples keep their actual
length:

```bash
sinfo
squeue -u "$USER"
```

Submit all planned ablations as separate jobs:

```bash
bash jobs/submit_neuronic_ablations.sh
```

Submit the optional extra-compute sweeps separately:

```bash
bash jobs/submit_neuronic_optional_sweeps.sh
```

Submit the seed replicates separately:

```bash
bash jobs/submit_neuronic_seed_replicates.sh
```

Submit one run manually:

```bash
sbatch jobs/neuronic_babilong_ablation.sbatch loss_k4_layer62 \
  --placement loss \
  --top-k 4 \
  --cache-layer-idx -2 \
  --output checkpoints/cache_babilong_loss_k4_layer62.pt \
  --metrics-output metrics/loss_k4_layer62.json
```

Monitor jobs and logs:

```bash
squeue -u "$USER"
tail -f logs/neuronic/loss_k4_layer62_<jobid>.log
tail -f logs/neuronic/bam-ablation-<jobid>.out
```

The checkpoint and metrics outputs are written back to NFS under
`checkpoints/` and `metrics/`. Temporary per-job files under
`/scratch/$USER/bam_runs/$SLURM_JOB_ID` are removed at the end of each job. The
HuggingFace cache under `/scratch/$USER/hf` is intentionally kept so later jobs
on the same node can reuse downloads.

## Run First

1. Smoke test
  - Purpose: verify CUDA, HuggingFace auth, BABILong loading, and model load.
  - Command:
    ```bash
    python -m bam.eval_babilong --tasks qa1 --lengths 0k --n_examples 2 --no_adapter
    ```
  - H100 estimate: 5-10 minutes.
  - L40 estimate: 10-20 minutes.
2. Existing paper-style no-cache baseline
  - Purpose: use the already-completed reasoning-curves FalconMamba baseline.
  - Source:
    `../reasoning-curves/runs/recall_eval/merged_final/babilong_tinker_llama_nemotron_falcon_mamba_final.json`
  - Covers `tiiuae/falcon-mamba-7b-instruct`, generative scoring, `qa1`-`qa10`,
    `0k`-`16k`, and `limit=50`.
  - No Neuronic job is needed unless you want to reproduce the baseline locally.
3. Regex-triggered cache reference
  - Purpose: reproduce the main rule-based placement result.
  - Command:
    ```bash
    python -m bam.train_babilong_ablation \
      --placement regex \
      --cache-layer-idx -2 \
      --output bam/cache_babilong_regex_layer62.pt \
      --metrics-output bam/metrics/regex_layer62.json \
      > logs/regex_layer62.log 2>&1
    ```
  - H100 estimate: 20-30 minutes.
  - L40 estimate: 45-90 minutes.
4. Loss-triggered cache reference
  - Purpose: reproduce the domain-agnostic placement result.
  - Command:
    ```bash
    python -m bam.train_babilong_ablation \
      --placement loss \
      --top-k 4 \
      --cache-layer-idx -2 \
      --output bam/cache_babilong_loss_k4_layer62.pt \
      --metrics-output bam/metrics/loss_k4_layer62.json \
      > logs/loss_k4_layer62.log 2>&1
    ```
  - H100 estimate: 25-40 minutes, including loss-position precompute.
  - L40 estimate: 1-2 hours.

## Main Ablations

1. Random placement, `K = 4`
  - Question: does placement matter, or is any small set of cache tokens enough?
  - Command:
    ```bash
    python -m bam.train_babilong_ablation \
      --placement random \
      --top-k 4 \
      --cache-layer-idx -2 \
      --output bam/cache_babilong_random_k4_layer62.pt \
      --metrics-output bam/metrics/random_k4_layer62.json \
      > logs/random_k4_layer62.log 2>&1
    ```
  - Keep fixed: tasks, lengths, epochs, layer 62, cache width, eval protocol.
  - H100 estimate: 20-35 minutes.
  - L40 estimate: 45-90 minutes.
  - Priority: highest. This is the cleanest control for the placement claim.
2. Fixed-interval placement, about 4 cache tokens
  - Question: does a simple position-only policy match selective placement?
  - Command:
    ```bash
    python -m bam.train_babilong_ablation \
      --placement interval \
      --top-k 4 \
      --cache-layer-idx -2 \
      --output bam/cache_babilong_interval_k4_layer62.pt \
      --metrics-output bam/metrics/interval_k4_layer62.json \
      > logs/interval_k4_layer62.log 2>&1
    ```
  - Keep fixed: total cache count near the regex/loss-triggered setting.
  - H100 estimate: 20-35 minutes.
  - L40 estimate: 45-90 minutes.
  - Priority: high. This distinguishes content-aware placement from regular sampling.
3. Loss-triggered `TOP_K = 1`
  - Question: is one memory slot enough?
  - Command:
    ```bash
    python -m bam.train_babilong_ablation \
      --placement loss \
      --top-k 1 \
      --cache-layer-idx -2 \
      --output bam/cache_babilong_loss_k1_layer62.pt \
      --metrics-output bam/metrics/loss_k1_layer62.json \
      > logs/loss_k1_layer62.log 2>&1
    ```
  - H100 estimate: 20-35 minutes.
  - L40 estimate: 45-90 minutes.
  - Priority: high.
4. Loss-triggered `TOP_K = 8`
  - Question: does performance improve with more cache entries, or does it saturate?
  - Command:
    ```bash
    python -m bam.train_babilong_ablation \
      --placement loss \
      --top-k 8 \
      --cache-layer-idx -2 \
      --output bam/cache_babilong_loss_k8_layer62.pt \
      --metrics-output bam/metrics/loss_k8_layer62.json \
      > logs/loss_k8_layer62.log 2>&1
    ```
  - H100 estimate: 25-45 minutes.
  - L40 estimate: 1-2 hours.
  - Priority: high.
5. Train through `8k`
  - Question: how much of the long-context gain comes from extrapolating beyond
    the trained lengths versus seeing longer lengths during cache training?
  - Command:
    ```bash
    python -m bam.train_babilong_ablation \
      --placement loss \
      --top-k 4 \
      --cache-layer-idx -2 \
      --train-lengths 0k,1k,2k,4k,8k \
      --eval-lengths 0k,1k,2k,4k,8k,16k \
      --max-seq-len 16384 \
      --output bam/cache_babilong_loss_k4_train8k_layer62.pt \
      --metrics-output bam/metrics/loss_k4_train8k_layer62.json \
      > logs/loss_k4_train8k_layer62.log 2>&1
    ```
  - Keep fixed: loss-triggered `TOP_K = 4`, layer 62, gate on, causal mask on,
    focused loss.
  - Neuronic: included in `jobs/submit_neuronic_ablations.sh` as
    `loss_k4_train8k_layer62`.
  - H100 estimate: 45-90 minutes.
  - L40 estimate: 2-4 hours.
  - Priority: high if the final story emphasizes 8k/16k generalization.
6. Layer 32 cache injection
  - Question: does the late-layer gradient-path argument hold on BABILong?
  - Command:
    ```bash
    python -m bam.train_babilong_ablation \
      --placement loss \
      --top-k 4 \
      --cache-layer-idx 32 \
      --output bam/cache_babilong_loss_k4_layer32.pt \
      --metrics-output bam/metrics/loss_k4_layer32.json \
      > logs/loss_k4_layer32.log 2>&1
    ```
  - Recommended placement: loss-triggered `TOP_K = 4` first; regex second if time allows.
  - H100 estimate: 25-45 minutes.
  - L40 estimate: 1-2 hours.
  - Priority: high, because it directly tests a key design claim.
7. No gate
  - Question: does the learned gate matter, or is `W_out(attn)` enough?
  - Command:
    ```bash
    python -m bam.train_babilong_ablation \
      --placement loss \
      --top-k 4 \
      --cache-layer-idx -2 \
      --gate off \
      --output bam/cache_babilong_loss_k4_nogate.pt \
      --metrics-output bam/metrics/loss_k4_nogate.json \
      > logs/loss_k4_nogate.log 2>&1
    ```
  - Keep fixed: loss-triggered `TOP_K = 4`, layer 62.
  - H100 estimate: 25-40 minutes.
  - L40 estimate: 1-2 hours.
  - Priority: medium.
8. No causal cache mask
  - Question: is causal cache attention important under the current focused-loss setup?
  - Command:
    ```bash
    python -m bam.train_babilong_ablation \
      --placement loss \
      --top-k 4 \
      --cache-layer-idx -2 \
      --causal-mask off \
      --output bam/cache_babilong_loss_k4_nocausal.pt \
      --metrics-output bam/metrics/loss_k4_nocausal.json \
      > logs/loss_k4_nocausal.log 2>&1
    ```
  - Caveat: this no longer matches causal LM semantics, so present it as a diagnostic rather than a proposed method.
  - H100 estimate: 25-40 minutes.
  - L40 estimate: 1-2 hours.
  - Priority: medium-low.
9. Full loss instead of focused answer loss
  - Question: is focused answer-token CE necessary?
  - Command:
    ```bash
    python -m bam.train_babilong_ablation \
      --placement loss \
      --top-k 4 \
      --cache-layer-idx -2 \
      --loss-mode full \
      --output bam/cache_babilong_loss_k4_fullloss.pt \
      --metrics-output bam/metrics/loss_k4_fullloss.json \
      > logs/loss_k4_fullloss.log 2>&1
    ```
  - Caveat: this may train the cache to preserve general LM behavior instead of retrieval.
  - H100 estimate: 25-45 minutes.
  - L40 estimate: 1-2 hours.
  - Priority: medium-low.
10. Seed replicates for the central comparison
  - Question: is the loss-triggered gain over random placement stable, or seed-sensitive?
  - Command:
    ```bash
    bash jobs/submit_neuronic_seed_replicates.sh
    ```
  - Recommended reporting: mean and standard deviation for loss-triggered K=4
    and random K=4 over seeds 41/42/43.
  - Priority: high if presenting this as a research result rather than a demo.

## If There Is Extra Compute

1. Cache width sweep
  - Settings: `D_ATTN = 64`, `128`, `256`, optionally `512`.
  - Commands:
    ```bash
    for D in 64 128 256 512; do
      python -m bam.train_babilong_ablation \
        --placement loss \
        --top-k 4 \
        --d-attn "$D" \
        --output "bam/cache_babilong_loss_k4_d${D}.pt" \
        --metrics-output "bam/metrics/loss_k4_d${D}.json" \
        > "logs/loss_k4_d${D}.log" 2>&1
    done
    ```
  - Question: how much projection capacity is needed?
  - H100 estimate per run: 20-45 minutes.
  - L40 estimate per run: 45 minutes-2 hours.
2. Cache entry capacity sweep
  - Settings: `MAX_ENTRIES = 1`, `2`, `4`, `8`, `64`.
  - Commands:
    ```bash
    for M in 1 2 4 8 64; do
      python -m bam.train_babilong_ablation \
        --placement loss \
        --top-k 4 \
        --max-entries "$M" \
        --output "bam/cache_babilong_loss_k4_entries${M}.pt" \
        --metrics-output "bam/metrics/loss_k4_entries${M}.json" \
        > "logs/loss_k4_entries${M}.log" 2>&1
    done
    ```
  - Question: how many previous cache entries must remain addressable?
  - H100 estimate per run: 20-45 minutes.
  - L40 estimate per run: 45 minutes-2 hours.
3. Training data size sweep
  - Settings: 10, 25, 50 examples per task/length.
  - Commands:
    ```bash
    for N in 10 25 50; do
      python -m bam.train_babilong_ablation \
        --placement loss \
        --top-k 4 \
        --n-train "$N" \
        --output "bam/cache_babilong_loss_k4_n${N}.pt" \
        --metrics-output "bam/metrics/loss_k4_n${N}.json" \
        > "logs/loss_k4_n${N}.log" 2>&1
    done
    ```
  - Question: does StateCache learn from very small data?
  - H100 estimate per run: 10-30 minutes.
  - L40 estimate per run: 25-90 minutes.
4. Epoch sweep
  - Settings: 5, 10, 20, 40 epochs.
  - Commands:
    ```bash
    for E in 5 10 20 40; do
      python -m bam.train_babilong_ablation \
        --placement loss \
        --top-k 4 \
        --epochs "$E" \
        --output "bam/cache_babilong_loss_k4_ep${E}.pt" \
        --metrics-output "bam/metrics/loss_k4_ep${E}.json" \
        > "logs/loss_k4_ep${E}.log" 2>&1
    done
    ```
  - Question: is the current run undertrained or overtrained?
  - H100 estimate: roughly linear in epoch count, so 5 epochs is about 25% of the reference training time and 40 epochs is about 2x.
  - L40 estimate: roughly 2-4x H100.
5. Regex vs loss-triggered under identical cache count
  - Question: is regex better because it picks facts, or because it inserts a different number of cache tokens?
  - Command:
    ```bash
    python -m bam.train_babilong_ablation \
      --placement regex \
      --regex-cap-k 4 \
      --cache-layer-idx -2 \
      --output bam/cache_babilong_regex_cap4_layer62.pt \
      --metrics-output bam/metrics/regex_cap4_layer62.json \
      > logs/regex_cap4_layer62.log 2>&1
    ```
  - H100 estimate per run: 20-35 minutes.
  - L40 estimate per run: 45-90 minutes.
6. Evaluation length extension
  - This is now included in the standard ablation jobs:
    `--eval-lengths 0k,1k,2k,4k,8k,16k --max-seq-len 16384`.
  - Question: does the cache result stay flat beyond the trained `0k`-`2k`
    lengths?
  - H100 estimate: depends mostly on number and length of eval examples; budget 30-90 minutes.
  - L40 estimate: 1-3 hours.

## Suggested Order

1. Smoke test.
2. Confirm/use existing reasoning-curves no-cache baseline.
3. Reproduce regex and loss-triggered references.
4. Random placement.
5. Fixed-interval placement.
6. Loss-triggered `TOP_K = 1` and `TOP_K = 8`.
7. Train-through-8k ablation if long-context generalization is central.
8. Layer 32.
9. No gate.
10. No causal mask and full-loss diagnostics only if time remains.
11. Seed replicates for loss-triggered vs random if the result will be reported
    as more than a course-project demo.

## Reporting Checklist

For each run, record:

- Git commit hash.
- Changed constants or code path.
- Placement policy.
- `CACHE_LAYER_IDX`.
- Number of cache tokens per example, mean and range if available.
- Training wall time.
- Peak GPU memory if available from scheduler logs or `nvidia-smi`.
- Parameter count and cache attention size (`D_ATTN`, `MAX_ENTRIES`, `TOP_K`).
- Final eval table.
- Checkpoint path.
- Log path.
- Whether the model/dataset cache was already warm.

## Sufficiency Check

The current plan covers the README TODOs:

- Random placement.
- Layer index 32 vs 62.
- K ablation for loss-triggered placement.
- No gate.

It also covers the method claims in `writeup.md` and
`docs/ARCHITECTURE_DEEP_DIVE.md`:

- Selective placement: regex/loss/random/interval and regex capped to K=4.
- Constant-size memory: K sweep and `MAX_ENTRIES` sweep.
- Bottleneck capacity: `D_ATTN` sweep.
- Late-layer training claim: layer 32 vs 62.
- Causal retrieval mechanism: no-causal diagnostic.
- Focused-loss claim: full-loss diagnostic.
- Data efficiency: training data size sweep.
- Long-context supervision: train-through-8k ablation versus train-through-2k
  reference.
- Optimization robustness: epoch sweep and seed replicates.
- Protocol fidelity: the reported no-cache baseline uses the paper-style
  BABILong eval in `bam.eval_babilongv2`.

For a short course project, the high-priority set is sufficient: existing
reasoning-curves baseline, regex, loss K=4, random K=4, interval K=4, K=1/8,
layer 32, no gate, and train-through-8k. For a paper-style claim, add seed
replicates, `D_ATTN`, `MAX_ENTRIES`, and data-size sweeps, then report mean/std
and runtime or memory overhead.

Note: `train_babilong_ablation.py` still logs `no_cache` rows during inline eval
as a debugging sanity check, but they are not part of the planned baseline set.

