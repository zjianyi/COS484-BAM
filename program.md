# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr19`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `evaluate.py` — fixed constants, data loading, [CACHE] placement quality metrics, held-out evaluation. Do not modify.
   - `train.py` — the file you modify. LoRA config, learning rate, data subset size, training schedule, tokenizer setup, everything about how we SFT.
4. **Verify data exists**: Check that `SFT_OpenMath/data/annotated/qwen3_235b/annotated_samples.jsonl` contains the [CACHE]-annotated reasoning traces. If not, tell the human to run `python generate_data.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Context

We are training Falcon Mamba 7B Instruct (`tiiuae/falcon-mamba-7b-instruct`) to emit a special `[CACHE]` token at reasoning-step boundaries during chain-of-thought math problem solving. The model already knows how to reason — we are teaching it one new behavior: produce `[CACHE]` tokens at the right moments.

This is a formatting/behavior task, not a capability task. The model should:
- Still solve math problems correctly (accuracy should stay above 90% of baseline)
- Produce `[CACHE]` tokens at step conclusions (not mid-computation)
- Produce a reasonable number of `[CACHE]` tokens per problem (average is 10-11, not 0 and not 50)

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 20 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: LoRA rank, LoRA alpha, target modules, learning rate, learning rate schedule, batch size, gradient accumulation steps, number of training examples, number of epochs, warmup ratio, weight decay, max sequence length, data sampling strategy, the chat template formatting.

**What you CANNOT do:**
- Modify `evaluate.py`. It is read-only. It contains the fixed evaluation: loads the trained model, generates solutions for 30 held-out math problems, and reports three metrics.
- Install new packages or add dependencies.
- Modify the evaluation harness or the held-out problems.
- Change the base model. We always use `tiiuae/falcon-mamba-7b-instruct`.

**The goal is a balance of three metrics:**
- `cache_f1`: F1 score of [CACHE] token placement vs oracle boundaries. Higher is better.
- `math_accuracy`: percentage of held-out problems solved correctly. Higher is better. Must stay above 0.9 × baseline.
- `cache_count_mae`: mean absolute error between number of [CACHE] tokens produced and number of oracle step boundaries. Lower is better.

The primary metric to optimize is `cache_f1`. But if `math_accuracy` drops below 90% of baseline, the run is a failure regardless of `cache_f1`.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.

**The first run**: Your very first run should always be to establish the baseline. Run `train.py` as-is with the default hyperparameters on 500 examples for 3 epoch.

## Output format

Once the script finishes it prints a summary like this:

```
---
cache_f1:         0.7200
math_accuracy:    0.8333
cache_count_mae:  1.4000
training_seconds: 1180.3
total_seconds:    1250.1
peak_vram_mb:     45060.2
num_examples:     500
num_epochs:       1
learning_rate:    0.0002
lora_rank:        16
```

You can extract the key metrics from the log file:

```
grep "^cache_f1:\|^math_accuracy:\|^cache_count_mae:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns:

```
commit	cache_f1	math_accuracy	cache_count_mae	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. cache_f1 achieved (e.g. 0.720000) — use 0.000000 for crashes
3. math_accuracy achieved (e.g. 0.833333) — use 0.000000 for crashes
4. cache_count_mae (e.g. 1.400000) — use 0.000000 for crashes
5. peak memory in GB, round to .1f — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	cache_f1	math_accuracy	cache_count_mae	memory_gb	status	description
a1b2c3d	0.720000	0.833333	1.400000	44.0	keep	baseline (500 examples 1 epoch lr=2e-4 rank=16)
b2c3d4e	0.780000	0.800000	1.100000	44.2	keep	increase lr to 5e-4
c3d4e5f	0.650000	0.900000	2.300000	44.0	discard	decrease lr to 5e-5 (underfitting)
d4e5f6g	0.810000	0.766667	0.800000	44.0	discard	3 epochs (accuracy dropped below 90% baseline)
e5f6g7h	0.000000	0.000000	0.000000	0.0	crash	rank=64 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr19`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^cache_f1:\|^math_accuracy:\|^cache_count_mae:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If cache_f1 improved AND math_accuracy >= 0.9 × baseline_accuracy, you "advance" the branch, keeping the git commit
9. If cache_f1 is equal or worse, or math_accuracy dropped too much, you git reset back to where you started

## Experiment ideas (suggested order)

The agent should try these roughly in order, but use judgment:

1. **Baseline**: 500 examples, 1 epoch, lr=2e-4, rank=16. Establish numbers.
2. **Learning rate sweep**: Try 1e-4, 3e-4, 5e-4, 1e-3. This is usually the highest-leverage knob.
3. **Epoch sweep**: Try 2 and 3 epochs on the best LR from step 2. Watch for accuracy degradation.
4. **Data size sweep**: Try 1000, 2000, 5000 examples at the best LR and epoch count.
5. **LoRA rank sweep**: Try 8, 32, 64. Higher rank = more capacity but more VRAM and overfitting risk.
6. **Target modules**: Try adding/removing `x_proj`, `embeddings` from LoRA targets.
7. **LoRA alpha sweep**: Try alpha = rank (standard), alpha = 2*rank, alpha = rank/2.
8. **Max sequence length**: Try 1024, 1536, 2048, 4096. Longer = more context but slower.
9. **Warmup ratio**: Try 0.0, 0.05, 0.1, 0.2.
10. **Mixed data**: Add some examples WITHOUT [CACHE] tokens (plain CoT) to prevent the model from over-caching. Try 80/20 and 60/40 ratios.

After exhausting these, get creative. Try combinations of the best settings. Try things not on this list.

**Timeout**: Each experiment should take ~20 minutes. If a run exceeds 30 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes, use your judgment. Typos and easy fixes: fix and re-run. Fundamental issues (OOM): log crash, revert, move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the data, look at actual model outputs, try combining previous near-misses. The loop runs until the human interrupts you.
