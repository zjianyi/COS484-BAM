# autoresearch (phase 2: BAM cache)

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `bam-apr21`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `architecture.md` — what the BAM bolt-on cache is and why it exists.
   - `bam/cache.py` — `StateCache` module (attention math for inference). Do not modify.
   - `bam/generator.py` — `BAMGenerator` for autoregressive inference (hooks + Mamba state). Do not modify.
   - `bam/evaluate_bam.py` — fixed constants, held-out evaluation, metric reporting. Do not modify.
   - `bam/train_cache.py` — the file you modify. Cache-module training loop, custom forward, learning rate, data subset size, training schedule, cache layer index, everything about how we train the bolt-on cache.
4. **Verify data exists**: Check that `SFT_OpenMath_data/annotated/qwen3_235b/annotated_samples.jsonl` contains the [CACHE]-annotated reasoning traces. If not, tell the human to run `python generate_data.py`.
5. **Verify phase-1 adapter exists**: Check that `./adapter` contains a saved phase-1 LoRA adapter. Phase 2 cannot run without it. If not, tell the human to run `python train.py` first.
6. **Initialize bam/results.tsv**: Create `bam/results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Context

Phase 1 produced a LoRA-tuned Falcon Mamba 7B Instruct that emits `[CACHE]` tokens at reasoning-step boundaries. That adapter is **frozen**. Phase 2 trains a small bolt-on cross-attention module (`StateCache`) bolted onto exactly one midpoint layer. At every `[CACHE]` position we write the layer-L hidden state into a bounded ring buffer (≤32 entries); on the token immediately following a `[CACHE]`, we read from the ring buffer and add the result to the layer-L+1 input. Only the cache module's weights (`W_Q`, `W_K`, `W_V`, `W_out`, `gate`) are trained — ~6 MB total. The base model, the LoRA adapter, and the resized `embed_tokens` / `lm_head` are all frozen.

This is a capability task, not a formatting task. The phase-1 model already places `[CACHE]` correctly. Phase 2's job is to make those `[CACHE]` tokens actually *help* on held-out math problems. The model should:
- Solve more held-out problems correctly than phase 1 did (math_accuracy should strictly go up)
- Not regress [CACHE] placement quality (cache_f1 and cache_count_mae should stay roughly where phase 1 left them — this is teacher-forced and driven by the frozen SFT weights, so small drift is fine)
- Actually exercise the cache module (`cache_writes_avg` should be > 0; writing zero entries means the module is dead weight)

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 30 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `python -m bam.train_cache`.

**What you CAN do:**
- Modify `bam/train_cache.py` — this is the only file you edit. Everything is fair game: learning rate, learning rate schedule, batch size, gradient accumulation steps, number of training examples, number of epochs, warmup ratio, weight decay, `D_ATTN`, `MAX_ENTRIES`, `CACHE_LAYER_IDX`, `USE_CHECKPOINT`, label-masking strategy, per-example weighting, which positions contribute to the loss.

**What you CANNOT do:**
- Modify `bam/evaluate_bam.py`, `bam/cache.py`, or `bam/generator.py`. They are read-only. `evaluate_bam.py` loads the adapter + cache checkpoint, generates solutions for 30 held-out math problems with `BAMGenerator`, and reports four metrics. `cache.py` and `generator.py` define the inference-time attention and hook wiring — changing them would make your results incomparable to earlier rows.
- Modify the phase-1 files (`train.py`, `evaluate.py`) or re-train the SFT adapter. Phase 2 layers on top of a fixed phase-1 adapter.
- Install new packages or add dependencies.
- Modify the evaluation harness or the held-out problems.
- Change the base model. We always use `tiiuae/falcon-mamba-7b-instruct`.

**The goal is a balance of four metrics:**
- `math_accuracy`: percentage of held-out problems solved correctly under BAM-aware free generation. Higher is better. **This is the primary metric.**
- `cache_f1`: F1 score of [CACHE] token placement vs oracle boundaries, teacher-forced against the frozen SFT model. Should stay within ±0.03 of the phase-1 baseline (the cache module doesn't change placement — this is a sanity check).
- `cache_count_mae`: mean absolute error between number of [CACHE] tokens produced and number of oracle step boundaries. Lower is better. Should stay roughly where phase 1 left it.
- `cache_writes_avg`: average number of ring-buffer writes per held-out problem. Expected around 10-11 (matches `avg_cache_tokens`). If this is 0, the module is not firing and the run is invalid regardless of math_accuracy.

The primary metric to optimize is `math_accuracy`. But if `cache_writes_avg` is 0, or `cache_f1` drops by more than 0.03 absolute from the phase-1 baseline, the run is a failure regardless of math_accuracy.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but it should not blow up dramatically. The cache module itself is tiny; VRAM is dominated by the frozen backbone's activations during the custom forward.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.

**The first run**: Your very first run should always be to establish the baseline. Run `bam/train_cache.py` as-is with the default hyperparameters (1000 examples, 2 epochs, lr=5e-4, d_attn=256, max_entries=32, cache_layer_idx=None (=L//2)).

## Output format

Once the script finishes it prints a summary like this:

```
---
cache_f1:          0.9680
math_accuracy:     0.2000
cache_count_mae:   3.0000
cache_writes_avg:  10.9000
training_seconds:  1720.3
total_seconds:     2080.1
peak_vram_mb:      48060.2
num_examples:      1000
num_epochs:        2
learning_rate:     0.0005
d_attn:            256
max_entries:       32
cache_layer_idx:   32
stopped_by_budget: False
```

You can extract the key metrics from the log file:

```
grep "^cache_f1:\|^math_accuracy:\|^cache_count_mae:\|^cache_writes_avg:\|^peak_vram_mb:" bam_run.log
```

## Logging results

When an experiment is done, log it to `bam/results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns (same schema as phase-1 `results.tsv`):

```
commit	cache_f1	math_accuracy	cache_count_mae	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. cache_f1 achieved (e.g. 0.968000) — use 0.000000 for crashes
3. math_accuracy achieved (e.g. 0.200000) — use 0.000000 for crashes
4. cache_count_mae (e.g. 3.000000) — use 0.000000 for crashes
5. peak memory in GB, round to .1f — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried — **include `cache_writes_avg` in the description** so we can tell dead runs from real ones (e.g. `lr=1e-3 d_attn=128 writes=10.6`)

Example:

```
commit	cache_f1	math_accuracy	cache_count_mae	memory_gb	status	description
a1b2c3d	0.968000	0.200000	3.000000	47.0	keep	baseline (1000ex 2ep lr=5e-4 d_attn=256 entries=32 L//2 writes=10.9)
b2c3d4e	0.970000	0.300000	3.100000	47.2	keep	lr=1e-3 (accuracy up, placement unchanged, writes=10.9)
c3d4e5f	0.965000	0.133000	3.000000	47.0	discard	d_attn=128 (accuracy dropped, writes=10.9)
d4e5f6g	0.968000	0.000000	3.000000	47.0	discard	cache_layer_idx=8 too early (writes=10.9 but acc regressed to 0)
e5f6g7h	0.968000	0.000000	3.000000	47.0	discard	gate init high + lr=5e-3 (writes=10.9 but acc=0, unstable)
f6g7h8i	0.968000	0.233000	3.000000	47.0	discard	max_entries=8 (writes=8.0, ring saturated, small gain over baseline)
g7h8i9j	0.000000	0.000000	0.000000	0.0	crash	d_attn=1024 (OOM during custom forward)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/bam-apr21`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `bam/train_cache.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python -m bam.train_cache > bam_run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^cache_f1:\|^math_accuracy:\|^cache_count_mae:\|^cache_writes_avg:\|^peak_vram_mb:" bam_run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 bam_run.log` to read the Python stack trace and attempt a fix.
7. Record the results in `bam/results.tsv` (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If math_accuracy improved AND cache_writes_avg > 0 AND cache_f1 stayed within 0.03 of the phase-1 baseline, you "advance" the branch, keeping the git commit
9. If math_accuracy is equal or worse, or cache_writes_avg is 0, or cache_f1 dropped too much, you git reset back to where you started

## Experiment ideas (suggested order)

The agent should try these roughly in order, but use judgment:

1. **Baseline**: 1000 examples, 2 epochs, lr=5e-4, d_attn=256, max_entries=32, cache_layer_idx=None. Establish numbers.
2. **Learning rate sweep**: Try 1e-4, 3e-4, 5e-4, 1e-3, 3e-3. The cache module is trained from scratch, so LR is the highest-leverage knob.
3. **Epoch sweep**: Try 1, 3, 4 epochs on the best LR from step 2. Watch for overfitting (training loss down but math_accuracy down).
4. **`CACHE_LAYER_IDX` sweep**: Try `L//2 - 8`, `L//2 - 4`, `L//2`, `L//2 + 4`, `L//2 + 8` (where `L=64`). Earlier layers have less semantic signal; later layers give the suffix less room to use the injection.
5. **`D_ATTN` sweep**: Try 64, 128, 256, 512, 1024. Larger = more capacity but more VRAM.
6. **`MAX_ENTRIES` sweep**: Try 8, 16, 32, 64. The dataset average is ~11, so 8 forces the ring to evict; 64 never evicts. This tests whether the ring constraint is helping or hurting at training time.
7. **Data size sweep**: Try 500, 2000, 5000 examples. Cache is small; more data may or may not help.
8. **Warmup ratio**: Try 0.0, 0.03, 0.1, 0.2.
9. **Gate initialization**: Try initializing `gate` to small positive (e.g. 0.5, 1.0) instead of zero, so the cache contributes from step 1. Also try a scalar multiplier on `W_out`.
10. **Position weighting**: Mask the LM loss to only the `K` tokens after each `[CACHE]` (instead of the full sequence), so all gradient signal flows through the cache. Sweep `K` in {8, 16, 32, 64}.
11. **Loss temperature / label smoothing**: Try label smoothing 0.0, 0.05, 0.1.
12. **Two-layer cache**: Bolt the cache onto two layers (e.g. `L//2` and `L//2 + 8`) instead of one. This breaks the strict single-layer linearity story but is still bounded, and may help if one layer is too little signal. Document the mild linearity cost.

After exhausting these, get creative. Try combinations of the best settings. Try things not on this list.

**Timeout**: Each experiment should take ~30 minutes. If a run exceeds 45 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes, use your judgment. Typos and easy fixes: fix and re-run. Fundamental issues (OOM): log crash, revert, move on. Notable failure mode to watch for: if `bam_forward` sees a sequence with zero `[CACHE]` tokens it returns a zero-loss no-op — this is intentional, not a crash.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read `architecture.md`, look at actual model outputs, inspect `cache_writes_avg` trends, try combining previous near-misses. The loop runs until the human interrupts you.
