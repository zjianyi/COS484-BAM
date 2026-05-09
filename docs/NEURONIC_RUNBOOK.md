# Neuronic Runbook

This note is meant to be enough context to resume work on Neuronic after losing
chat history. It covers cluster rules, storage, and the COS484-BAM workflow.

## Access

Neuronic access is restricted to users sponsored by a faculty member in a SEAS
department. The cluster uses Princeton OIT LDAP authentication.

Log in with your OIT NetID and password:

```bash
ssh bw7520@neuronic.cs.princeton.edu
```

The login node is only for compiling, job submission, file transfer, and limited
testing. Do not run training or evaluation directly on the login node. Long or
GPU-heavy login-node processes may be killed without notice.

## Hardware

Neuronic has 33 Lenovo ThinkSystem SR670 V2 nodes. One is the login node; the
others are compute nodes.

Each compute node has:

- 2 x Intel Xeon Gold 5320 26-core CPUs
- 512 GB RAM
- 8 x NVIDIA L40 GPUs
- 3.5 TB local SSD scratch at `/scratch`
- 10 Gbps Ethernet uplink

The cluster does not use InfiniBand.

## Storage Rules

There are three relevant storage classes:

- Home/project NFS storage: persistent, network-mounted, safer for source code,
  checkpoints, metrics, and final outputs.
- Local scratch at `/scratch`: fast per-node SSD, good for caches and temporary
  input/intermediate files.
- Research-funded project NFS spaces: may also be mounted if provisioned by CS.

`/scratch` is not backed up and is routinely purged. Always copy important
outputs back to project storage before a job exits. By convention:

```bash
mkdir -p /scratch/$USER
```

For this project, Slurm wrappers put HuggingFace, dataset, TorchInductor, and
temporary files under `/scratch/$USER`, then write checkpoints and metrics back
to the repo.

## Project Location

Remote repo path used in this project:

```bash
cd /u/bw7520/COS484-BAM
```

Use this path for `rsync`/`scp` targets and helper scripts. **`~/COS484-BAM` can be wrong** on Neuronic if your home or symlink layout differs from what you expect—prefer **`/u/bw7520/COS484-BAM`**.

Local repo path:

```bash
cd "/Users/brianwang/Documents/uni/spring26/cos484/final project/COS484-BAM"
```

If `git pull` does not show expected files, check branch and latest commits:

```bash
git status --short --branch
git remote -v
git log --oneline --decorate -5
git fetch origin
git pull --ff-only origin bwang/ablations
```

If the remote branch is not up to date, push from local first:

```bash
git push origin bwang/ablations
```

For urgent script updates, `rsync` is often faster than debugging Git:

```bash
rsync -avz bam/eval_babilong_paper_scaffold_checkpoint.py bw7520@neuronic:/u/bw7520/COS484-BAM/bam/
rsync -avz jobs/neuronic_babilong_paper_scaffold_eval.sbatch bw7520@neuronic:/u/bw7520/COS484-BAM/jobs/
```

## Environment

The repo uses `pixi` on Linux when available. The Slurm scripts automatically
prefer:

```bash
pixi run python
```

if `pixi.toml` exists. Otherwise they fall back to `python` or `.venv`.

Do not run training directly on the login node. Use `sbatch`.

## Main Slurm Wrappers

Train StateCache and run the original inline scaffold eval:

```bash
jobs/neuronic_babilong_ablation.sbatch
```

Evaluate a saved checkpoint on the original inline scaffold, without retraining:

```bash
jobs/neuronic_babilong_eval_checkpoint.sbatch
```

Evaluate the base model on the original scaffold without StateCache:

```bash
jobs/neuronic_babilong_scaffold_baseline.sbatch
```

Evaluate a saved checkpoint with the few-shot BABILong paper-style scaffold:

```bash
jobs/neuronic_babilong_paper_scaffold_eval.sbatch
```

## Evaluation Protocols

There are two main result protocols to keep separate.

Original StateCache scaffold:

```text
passage with inserted [CACHE] tokens
question
[CACHE]
```

This is what `train_babilong_ablation.py` uses for training and initial inline
evaluation. The `baseline_acc` in these JSON files is a scaffold-only sanity
check: same prompt with `[CACHE]` tokens, but no StateCache delta.

Few-shot BABILong scaffold:

```text
instructions
2 solved examples from the same task's 0k split
<context>
passage with inserted [CACHE] tokens
</context>
QUESTION: question
Answer: [CACHE]
```

This is closer to the reasoning-curves BABILong prompt while still giving
StateCache explicit write/read sites. It is eval-only for already trained
checkpoints.

## Common Job Commands

Main loss-selected ablation:

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
```

Layer 32 ablation:

```bash
sbatch jobs/neuronic_babilong_ablation.sbatch loss_k4_layer32 \
  --placement loss \
  --top-k 4 \
  --train-lengths 0k,1k,2k \
  --eval-lengths 0k,1k,2k,4k,8k,16k \
  --max-seq-len 16384 \
  --cache-layer-idx 32 \
  --output checkpoints/cache_babilong_loss_k4_layer32.pt \
  --metrics-output metrics/loss_k4_layer32.json
```

Few-shot BABILong eval for one checkpoint:

```bash
mkdir -p metrics/fewshot_babilong logs/neuronic

sbatch jobs/neuronic_babilong_paper_scaffold_eval.sbatch fewshot_babilong_loss_k4_layer32 \
  --checkpoint checkpoints/cache_babilong_loss_k4_layer32.pt \
  --metrics-output metrics/fewshot_babilong/loss_k4_layer32.json \
  --cells-output metrics/fewshot_babilong/loss_k4_layer32.cells.jsonl \
  --eval-tasks qa1,qa2,qa3 \
  --eval-lengths 0k,1k,2k,4k,8k,16k \
  --n-eval 50
```

Few-shot BABILong eval for the main nine checkpoints:

```bash
mkdir -p metrics/fewshot_babilong logs/neuronic

for run in \
  loss_k1_layer62 \
  loss_k8_layer62 \
  loss_k4_train8k_layer62 \
  loss_k4_nogate \
  loss_k4_layer32 \
  interval_k4_layer62 \
  random_k4_layer62 \
  regex_layer62 \
  loss_k4_layer62
do
  sbatch jobs/neuronic_babilong_paper_scaffold_eval.sbatch "fewshot_babilong_${run}" \
    --checkpoint "checkpoints/cache_babilong_${run}.pt" \
    --metrics-output "metrics/fewshot_babilong/${run}.json" \
    --cells-output "metrics/fewshot_babilong/${run}.cells.jsonl" \
    --eval-tasks qa1,qa2,qa3 \
    --eval-lengths 0k,1k,2k,4k,8k,16k \
    --n-eval 50
done
```

## Monitoring Jobs

Current running or pending jobs:

```bash
squeue -u "$USER"
```

Historical job state:

```bash
sacct -j JOBID --format=JobID,JobName%35,State,ExitCode,Elapsed,Start,End
```

Multiple jobs:

```bash
sacct -j 2989145,2989146,2989147 \
  --format=JobID,JobName%35,State,ExitCode,Elapsed,Start,End
```

Clean success means:

```text
State=COMPLETED
ExitCode=0:0
```

Find failed or cancelled jobs from today:

```bash
sacct -u "$USER" \
  --starttime 2026-05-09T00:00 \
  --state=FAILED,CANCELLED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL,PREEMPTED \
  --format=JobID,JobName%35,State,ExitCode,Elapsed,Start,End
```

## Logs and Outputs

Slurm wrapper stdout/stderr:

```text
logs/neuronic/%x-%j.out
logs/neuronic/%x-%j.err
```

Per-run Python logs:

```text
logs/neuronic/${RUN_ID}_${SLURM_JOB_ID}.log
```

Training/eval checkpoints:

```text
checkpoints/cache_babilong_*.pt
```

Aggregate metrics:

```text
metrics/*.json
metrics/fewshot_babilong/*.json
metrics/scaffold_baselines/*.json
```

Per-cell/per-example eval traces:

```text
metrics/*.cells.jsonl
metrics/fewshot_babilong/*.cells.jsonl
metrics/scaffold_baselines/*.cells.jsonl
```

For result tables, prefer metrics JSONs as the source of truth. Use logs for
debugging and `.cells.jsonl` files for error analysis.

Useful log search:

```bash
rg -n "run_id=|num_layers|cache_layer_idx|epoch .* done|W_out_norm|BABILong Cache Eval|Total time|Traceback|Error|CUDA|OOM" logs/neuronic
```

## Pulling Results Locally

From local machine:

```bash
cd "/Users/brianwang/Documents/uni/spring26/cos484/final project/COS484-BAM"

mkdir -p metrics/neuronic_ablations metrics/fewshot_babilong metrics/scaffold_baselines

rsync -avz bw7520@neuronic:/u/bw7520/COS484-BAM/metrics/*.json metrics/neuronic_ablations/
rsync -avz bw7520@neuronic:/u/bw7520/COS484-BAM/metrics/fewshot_babilong/ metrics/fewshot_babilong/
rsync -avz bw7520@neuronic:/u/bw7520/COS484-BAM/metrics/scaffold_baselines/ metrics/scaffold_baselines/
```

Check JSON validity:

```bash
for f in metrics/**/*.json; do
  python -m json.tool "$f" >/dev/null && echo "OK $f" || echo "BAD $f"
done
```

## Rough Runtime Expectations

Previous train+eval jobs on `qa1-qa3`, six context lengths, 50 examples:

- interval/random controls: about 1 hour
- main loss runs: about 1 hour 20 minutes
- layer 32: about 1 hour 45 minutes
- train through 8k: about 3.5 hours

Few-shot eval-only jobs on `qa1-qa3` should usually fit within the 4 hour
walltime and are expected to be roughly 30-90 minutes, depending on checkpoint
and cache density.

## Common Pitfalls

- Running compute on the login node. Always use `sbatch`.
- Forgetting to push local commits before `git pull` on Neuronic.
- Pulling the wrong branch. This project work has used `bwang/ablations`.
- Reusing the same `--metrics-output` for two concurrent jobs.
- Comparing scaffold-only `baseline_acc` to the reasoning-curves no-cache
  baseline. They are different protocols.
- Leaving important outputs only on `/scratch`. Scratch is temporary and purged.
