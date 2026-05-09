# BABILong StateCache — ablation runs write-up

COS484 BAM project. **Model:** `tiiuae/falcon-mamba-7b-instruct`. **Backbone:** frozen; **only StateCache** trains (except noted eval-only jobs).

This note catalogs **what we ran**, **where results live**, and **how to interpret** strong vs weak outcomes. For protocol definitions (`cache_acc` vs plain baseline, `[CACHE]` in prompt), see **`README.md`** (*BABILong evaluation protocols*). Summarized LaTeX/PDF tables: **`docs/ablation_results_table.*`**, **`docs/ablation_cache_scaffold_zero_shot_table.*`**, **`docs/fewshot_babilong_results_table.*`**.

---

## Shared setup (unless a row says otherwise)

| Setting | Value |
|--------|--------|
| Train tasks | `qa1`, `qa2`, `qa3` |
| Train lengths | `0k,1k,2k` (except **train up to 8k**: adds `4k,8k`) |
| Eval tasks | `qa1`–`qa3` |
| Eval lengths | `0k`–`16k` |
| Examples per cell | 50 train / 50 eval per task×length |
| Max sequence length | 16384 |
| Default cache layer | L62 (`--cache-layer-idx -2` on 64-layer stack) |
| Default placement | loss-selected |
| Default top-k | 4 |
| Gate / causal mask / loss mode | on / on / **focused** (except **no gate**, **full loss**) |

**Metric JSON naming:** Neuronic runs usually mirror **`metrics/<run_id>.json`** at repo root or under **`metrics/fewshot_babilong/`** / **`metrics/scaffold_baselines/`** after rsync.

---

## Eval-only runs (no StateCache training)

These establish baselines or diagnostics; they do **not** produce a `.pt` checkpoint.

### Plain zero-shot no-cache (`eval_babilongv2`)

- **Command pattern:** `pixi run python -m bam.eval_babilongv2 --mamba-model … --few-shot-examples 0 --babilong-scoring next_token …`
- **Prompt:** Short BABILong string — **no `[CACHE]`** tokens.
- **Purpose:** Fair comparison of Falcon-Mamba without intervention; row **“No cache, zero-shot”** in `ablation_results_table` (~**24.9%** avg in our table).
- **Why it isn’t “bad”:** This is the intended external baseline; numbers are middling because BABILong is hard and scoring is exact-match next-token.

### Few-shot no-cache (reasoning-curves import)

- **Source:** Approximate next-token / prefix scoring from **`reasoning-curves`** merged eval (imported into **`metrics/fewshot_babilong/no_cache_baseline.json`**).
- **Prompt:** Paper-style few-shot BABILong — **no `[CACHE]`**, **with** few-shot block.
- **Row:** “No cache, few-shot” (~**24.4%** avg); strong at **0k**, **bad tail at 8k–16k** because total prompt length explodes (prefix + long passage).

### Scaffold-only baseline (`eval_babilong_scaffold_baseline`)

- **Example:** `sbatch jobs/neuronic_babilong_scaffold_baseline.sbatch scaffold_loss_k4_layer62 --placement loss --top-k 4 …`
- **Prompt:** **Same `[CACHE]` insertion policy** as loss-k4 ablations, **no** StateCache forward delta.
- **Output:** e.g. **`metrics/scaffold_baselines/scaffold_loss_k4_layer62.json`**
- **Result:** ~**0%** everywhere — **expected**, not a bug. The frozen LM is not trained to interpret arbitrary `[CACHE]` slots; without StateCache the scaffold hurts more than it helps under our scorer.
- **Role:** Matches inline **`baseline_acc`** semantics; see **`ablation_cache_scaffold_zero_shot_table`**.

### Few-shot + scaffold, checkpoint eval only (`eval_babilong_paper_scaffold_checkpoint`)

- **Purpose:** Re-score **trained** checkpoints under paper few-shot layout **with** `[CACHE]` (no retraining).
- **Outputs:** **`metrics/fewshot_babilong/<run_id>.json`**

---

## StateCache training runs — catalog and interpretation

Each row is one trained checkpoint + inline eval (`train_babilong_ablation`) and/or a separate few-shot scaffold eval. **“ZS avg”** = zero-shot table average (**`cache_acc`**, scaffold prompt). **“FS avg”** = few-shot table average where we ran paper scaffold eval.

### Placement policy & capacity (loss-selected \(k\), L62)

| Run ID | \(k\) | ZS avg | FS avg | What we tested | Good / bad |
|--------|------|--------|--------|----------------|------------|
| `loss_k1_layer62` | 1 | 17.4 | 24.8 | Minimal memory slots | **Moderate.** One slot often insufficient vs \(k{=}4\). |
| `loss_k4_layer62` | 4 | 15.3 | 26.1 | **Main** loss baseline at L62 | **Mid-pack ZS** — surprisingly below \(k{=}8\) on average in ZS table; placement noise / optimization. **Good FS** relative to many controls. |
| `loss_k8_layer62` | 8 | 19.0 | 26.9 | More write tokens | **Better ZS avg than \(k{=}4\)** in our numbers — extra slots help some regimes; diminishing returns vs complexity. |
| `loss_k16_layer62` | 16 | 16.3 | 23.8 | Wide \(k\) | **Mixed / slightly worse FS avg** — many inserts may dilute signal or crowd the passage representation. |
| `loss_k32_layer62` | 32 | 17.2 | 28.3 | Very wide \(k\) | **FS improves** vs \(k{=}16\) in our table; ZS avg still modest — tradeoff between capacity and interference. |
| `loss_k64_layer62` | 64 | 18.8 | 26.8 | Maximum \(k\) sweep | **Similar story** — not clearly better than mid-\(k\) on ZS avg; risk of noisy placement. |

**Takeaway:** Performance is **not monotonic in \(k\)**. Too few slots (\(k{=}1\)) under-equips the cache; very large \(k\) adds tokens and may hurt unless the policy stays selective.

### Placement controls (\(k{=}4\), L62)

| Run ID | Policy | ZS avg | FS avg | Interpretation |
|--------|--------|--------|--------|----------------|
| `interval_k4_layer62` | Uniform spacing | 15.2 | 26.8 | **Content-agnostic** — beats random slightly on FS; loss still wins when facts are buried. |
| `random_k4_layer62` | Random positions | 17.9 | 20.6 | **Weak FS** — placement matters; random is a tough control. |
| `regex_layer62` | Regex after entity-movement facts | 18.4 | 24.3 | **Domain-informed** — competitive but needs BABILong-specific rules; loss-triggered aims for generality. |

### Layer index (\(k{=}4\), loss, L62 unless noted)

| Run ID | Layer | ZS avg | FS avg | Interpretation |
|--------|-------|--------|--------|----------------|
| `loss_k4_layer2` | **L2** | **44.1** | **45.9** | **Best overall** in both tables. Early injection gives StateCache **many** downstream Mamba layers to transform representations before the answer head — aligns with “short gradient path from supervision” intuition inverted: **early edit propagates**. (Also: verify layer index semantics match intended depth in code.) |
| `loss_k4_layer32` | L32 | 33.8 | 35.4 | **Strong mid** — mid-layer works better than L62 for raw accuracy in our sweep. |
| `loss_k4_layer62` | L62 (default) | 15.3 | 26.1 | **Training default** for backward-path story — but **empirically weaker** than early layers here for **accuracy**, suggesting the original “late layer only” hypothesis is **not what maximizes BABILong cache_acc** under this setup. |

### Architecture / training variants (\(k{=}4\), L62)

| Run ID | Variant | ZS avg | FS avg | Interpretation |
|--------|---------|--------|--------|----------------|
| `loss_k4_nogate` | Gate **off** | 14.9 | 28.1 | **Worse ZS** than gated baseline — gate helps **stabilize** residual injection. |
| `loss_k4_full_loss_layer62` | **`full`** loss mode | 17.7 | 17.9 | **Poor**, especially FS — training CE over **all** tokens fights the **focused** answer objective; cache gets diffuse signal. |
| `loss_k4_train8k_layer62` | Train **0k–8k** | 15.9 | **29.6** | **ZS avg slightly lower** than train-to-2k-only on paper — but **FS row excels at 4k–8k** (longer training matches longer eval). |

---

## Cross-cutting themes

1. **`cache_acc` vs plain baseline.** StateCache rows are trained/evaluated **with `[CACHE]`**; plain **`eval_babilongv2`** does not. Compare **ZS “No cache”** row to **`cache_acc`** only when the narrative is “bolt-on value under scaffold”; use **scaffold-only** row for “same prompt, module off.”

2. **Why scaffold-only ~0%.** Not broken — the backbone sees **[CACHE]** tokens without learned correction; exact-match collapses.

3. **Why L2 (and L32) beat L62 on accuracy.** Empirically, **early** injection wins under this protocol — likely **representation rewrite bandwidth** over depth; late-layer hook may be better for **gradient structure during training** but not for **held-out cache_acc**.

4. **Full loss bad.** Supervision spreads across filler tokens; cache does not specialize on answer-critical behavior.

5. **Few-shot vs zero-shot tables differ.** Few-shot adds a **long prefix**; StateCache was **not** trained with that prefix — FS numbers can be **higher or lower** than ZS by row; treat as **different distribution**.

6. **Train-through-8k.** Helps **long-context eval** in few-shot regime where train and eval lengths align better; may trade off short-context ZS avg.

---

## Legacy / pointers

- Older narrative and Phase 2 snapshot tables: **`writeup.md`**, **`docs/initial_ablations_description.md`**, **`docs/ABLATION_RUN_PLAN.md`**.
- Cluster workflow: **`docs/NEURONIC_RUNBOOK.md`**.

---

## Revision note

Update this file when new **`run_id`** JSONs land or captions change. Table averages cited here match **`docs/ablation_results_table.tex`** and **`docs/fewshot_babilong_results_table.tex`** as of their last regeneration from **`metrics/`**.
