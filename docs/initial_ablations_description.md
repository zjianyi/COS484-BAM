# Initial Ablations Description

This document describes the initial Neuronic ablation set for the StateCache
BABILong experiments. All cache runs use a frozen `tiiuae/falcon-mamba-7b-instruct`
backbone and train only the StateCache module.

Unless otherwise noted, runs train on `qa1,qa2,qa3` at `0k,1k,2k`, evaluate on
`0k,1k,2k,4k,8k,16k`, and use `--max-seq-len 16384`.

## Baseline

### FalconMamba No-Cache Baseline

Source:

```text
../reasoning-curves/runs/recall_eval/merged_final/babilong_approx_next_token_prefix_eval.json
```

This is the baseline for comparison against StateCache. It uses saved greedy
FalconMamba generations and scores the first generated token/prefix against the
gold answer. This is closest to the StateCache eval, which uses the greedy
next-token decision at the final query position.

## Initial Cache Runs

### 1. `loss_k4_layer62`

Main StateCache run.

- Placement: loss-triggered
- Cache writes: `K=4`
- Injection layer: layer 62 (`--cache-layer-idx -2`)
- Gate: on
- Causal cache mask: on
- Training lengths: `0k,1k,2k`

Question: does a small learned memory module improve FalconMamba recall when
cache write positions are selected by frozen-model token loss?

This is the main method result.

### 2. `regex_layer62`

Rule-based placement reference.

- Placement: regex after BABILong entity-movement facts
- Injection layer: layer 62
- Gate: on
- Causal cache mask: on

Question: how well does StateCache work when cache writes are placed by a
task-specific fact detector?

This is a strong but less general reference because the placement rule uses
BABILong-specific structure.

### 3. `random_k4_layer62`

Random placement control.

- Placement: random passage positions
- Cache writes: `K=4`
- Injection layer: layer 62

Question: does placement matter, or does any small set of memory writes help?

If loss-triggered placement beats random placement, it supports the claim that
selective memory placement matters.

### 4. `interval_k4_layer62`

Fixed-interval placement control.

- Placement: evenly spaced passage positions
- Cache writes: `K=4`
- Injection layer: layer 62

Question: is content-aware placement better than a simple position-only policy?

This distinguishes loss-triggered placement from regular sampling through the
context.

### 5. `loss_k4_layer32`

Layer index ablation.

- Placement: loss-triggered
- Cache writes: `K=4`
- Injection layer: layer 32

Question: does late-layer injection matter?

The hypothesis is that layer 62 works better because the cache has a much
shorter frozen backward path to the answer logits.

### 6. `loss_k4_nogate`

Gate ablation.

- Placement: loss-triggered
- Cache writes: `K=4`
- Injection layer: layer 62
- Gate: off

Question: does the learned output gate matter, or is an ungated cache residual
enough?

If performance drops without the gate, the gate is an important stabilizing or
selection mechanism.

### 7. `loss_k4_train8k_layer62`

Training-length ablation.

- Placement: loss-triggered
- Cache writes: `K=4`
- Injection layer: layer 62
- Training lengths: `0k,1k,2k,4k,8k`
- Eval lengths: `0k,1k,2k,4k,8k,16k`

Question: how much long-context performance comes from extrapolating beyond
training lengths, versus seeing longer contexts during cache training?

Compare this against `loss_k4_layer62`, which trains only through `2k`.

K Sweep

### 8. `loss_k1_layer62`

- Placement: loss-triggered
- Cache writes: `K=1`
- Injection layer: layer 62

Question: is a single memory slot enough?

### 9. `loss_k8_layer62`

- Placement: loss-triggered
- Cache writes: `K=8`
- Injection layer: layer 62

Question: does more memory improve performance, or does the method saturate by
`K=4`?

Together, `K=1`, `K=4`, and `K=8` test the cache-size tradeoff.

## Reporting Guidance

Use the no-cache baseline from reasoning-curves as the external FalconMamba
reference. Use the cache runs to compare architectural choices under the same
StateCache protocol.

The most important table/plot should include:

- FalconMamba no-cache baseline
- `loss_k4_layer62`
- `regex_layer62`
- `random_k4_layer62`
- `interval_k4_layer62`
- `loss_k4_layer32`
- `loss_k4_nogate`
- `loss_k4_train8k_layer62`

Report accuracy by context length, especially whether training through `2k`
generalizes to `4k,8k,16k`.
