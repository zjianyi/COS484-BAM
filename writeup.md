# BAM StateCache — Project Notes

**COS484 Final Project, Spring 2025**

---

## What we built

StateCache: a 4.2M-param cross-attention memory module bolted onto frozen FalconMamba 7B (layer 62 of 64). Trainable weights: W_Q, W_K, W_V (d_model→256), W_out (256→d_model), W_gate (d_model→1), all zero-initialized. At each `[CACHE]` token, the module reads from earlier cache entries (causal mask) and adds a gated residual delta to the hidden state before it continues through the remaining frozen layer.

---

## Phase 1: OpenMath (all failed)

Goal was to push math_accuracy above 0.10 baseline on OpenMath held-out (indices 4970–5000).

| experiment | config | result | why it failed |
|---|---|---|---|
| v1–v5 | LM loss, layer_idx=32 | math_accuracy=0 | 32 frozen layers in backward path → degenerate gradient |
| v4–v15 | REINFORCE, layer_idx=32, OpenMath | reward ~0% | model generates gibberish on OpenMath; no valid rollouts |
| v15+ | REINFORCE, layer_idx=62, GSM8K | reward ~0% | model never emits `[CACHE]` tokens on GSM8K → zero gradient |
| v16 | LM loss, layer_idx=62, OpenMath | ~0.10 (no improvement) | pivoted before full eval |

**Key insight from failure:** `layer_idx=32` puts 32 frozen Mamba blocks in the backward path. The Jacobian is fixed and degenerate → useless gradient to cache params. `layer_idx=62` fixes this (only 1 frozen block). The LM loss + layer_idx=62 combination was never tried until v16, by which point we pivoted.

---

## Phase 2: BABILong recall (works)

Pivoted to entity-location tracking (qa1/qa2/qa3) across 0k–4k context with Wikipedia distractors.

**Baseline (generate, no cache):** FalconMamba collapses from 32% → 2% at 1k context on qa1. qa2/qa3 near-zero even at 0k.

**Training:** 450 examples (50 × 3 tasks × 3 lengths), 20 epochs, ~17 min on H100. Focused CE loss on answer token at final `[CACHE]` position.

### Results

| task | length | baseline | regex cache | loss-triggered cache |
|------|--------|----------|-------------|----------------------|
| qa1 | 0k | 0.32 | 0.24 | 0.14 |
| qa1 | 1k | 0.02 | 0.22 | 0.20 |
| qa1 | 2k | 0.00 | 0.18 | 0.14 |
| qa1 | 4k | 0.02 | 0.18 | 0.14 |
| qa2 | 0k | 0.04 | 0.12 | 0.20 |
| qa2 | 1k | 0.00 | 0.18 | 0.18 |
| qa2 | 2k | 0.00 | 0.14 | 0.20 |
| qa2 | 4k | 0.00 | 0.24 | 0.18 |
| qa3 | 0k | 0.00 | 0.14 | 0.18 |
| qa3 | 1k | 0.00 | 0.14 | 0.18 |
| qa3 | 2k | 0.02 | 0.16 | 0.14 |
| qa3 | 4k | 0.02 | 0.18 | 0.14 |
| **avg** | | **0.033** | **0.175** | **0.165** |

### Cache placement strategies

**Regex-triggered (v8):** Insert `[CACHE]` after entity-movement sentences ("X went to Y"). ~4 tokens at all context lengths including 4k. Requires domain-specific pattern.

**Loss-triggered:** Insert `[CACHE]` after top-K tokens by per-token loss (model's own surprise signal). No domain knowledge. Precomputed once before training since backbone is frozen. Achieves same accuracy (avg 16.5% vs 17.5%). In-context loss confirms signal: fact sentences avg loss 3.15 vs distractors 2.58 (ratio 1.22).

### Key design decisions that matter

- **Layer 62 not 32** — only 1 frozen block in backward path
- **W_out zero-init** — delta starts at 0, stable training from identity
- **Causal cache mask** — each `[CACHE]` only reads from earlier entries
- **Focused loss** — CE only on the answer token, not the whole sequence
- **Selective placement** — ~4 `[CACHE]` tokens regardless of context length → O(16) cache attention, Mamba handles distractors linearly

---

## Thoughts

**The architectural claim:** Existing hybrid SSM+attention models (Jamba, Zamba, Mamba-2-Hybrid) interleave full attention layers into Mamba, making overall complexity O(n²) — they sacrifice the property that motivated using Mamba in the first place. StateCache is different: cross-attention over k=4 constant-size memory is O(16) regardless of sequence length. The overall pipeline is still O(n). This is not "attention on an SSM" in the Jamba sense; it is a bounded external memory that preserves linear time end-to-end.

Augmenting sequential models with attention (Bahdanau 2014, NTMs) is old. But doing so in a way that keeps the SSM's linear-time property intact — via selective placement that bounds k independently of n — is the specific contribution. Mamba's architecture creates the problem (fixed recurrent state, no KV cache to fall back on) and also makes the solution non-trivial (intercepting hidden states at a specific layer, routing gradients through a mostly-frozen backbone).

**The empirical contribution:** FalconMamba 7B's recall collapse on BABILong (32% → 2% at 1k context) has not been reported before. StateCache recovers 14–24% uniformly across 0k–4k with 0.06% of the model's parameters. The loss-triggered placement result shows the approach does not require task-specific engineering.
