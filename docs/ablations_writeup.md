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

### Few-shot no-cache (legacy reasoning-curves import)

- **Source:** **`metrics/fewshot_babilong/no_cache_baseline.json`** (see also **`no_cache_baseline_legacy_reasoning_curves_approx.json`**), imported from approximate prefix-on-generation rescoring.
- **Prompt:** Paper-style few-shot BABILong — **no `[CACHE]`**, **with** few-shot block.
- **Row:** “No cache, few-shot” in **`fewshot_babilong_results_table`** (~**24.4%** avg in the current LaTeX table); strong at **0k**, bad tail at **8k--16k** because total prompt length grows with prefix + passage.

### Few-shot no-cache (`eval_babilongv2`, candidate scoring — diagnostic)

- **Source:** **`metrics/fewshot_babilong/no_cache_baseline_next_token_qa1_qa3.json`** — `eval_babilongv2` with paper few-shot prompts (`--few-shot-examples 2`), **candidate answer-token** scoring (`--babilong-scoring next_token`).
- **Status:** Useful diagnostic, but **not** the current “No cache, few-shot” LaTeX row. Do not mix this candidate-logit baseline with the legacy few-shot table unless the table is regenerated consistently.

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

The new **uniform interval L2** run is listed in the layer table below, because its main conclusion is about injection depth rather than placement at the default L62 hook.

### Layer index (\(k{=}4\), loss, L62 unless noted)

| Run ID | Layer | ZS avg | FS avg | Interpretation |
|--------|-------|--------|--------|----------------|
| `loss_k4_layer2` | **L2** | 44.1 | **45.9** | **Best among rows with few-shot eval** and still near the zero-shot top. Early injection gives StateCache **many** downstream Mamba layers to transform representations before the answer head. (Also: verify layer index semantics match intended depth in code.) |
| `loss_k4_layer6` | **L6** | **47.9** | n/a | **Best zero-shot row in the current table.** Strength is broad across 0k--8k and remains highest at 16k among the layer sweep, suggesting very-early-but-not-input-adjacent injection is a strong operating point. |
| `loss_k4_layer16` | L16 | 42.1 | n/a | **Strong early/mid injection.** Lower than L2/L6 but far above L62, reinforcing that the late default hook is not accuracy-optimal here. |
| `loss_k4_layer32` | L32 | 33.8 | 35.4 | **Strong mid** — mid-layer works better than L62 for raw accuracy in our sweep. |
| `loss_k4_layer62` | L62 (default) | 15.3 | 26.1 | **Training default** for backward-path story — but **empirically weaker** than early layers here for **accuracy**, suggesting the original “late layer only” hypothesis is **not what maximizes BABILong cache_acc** under this setup. |
| `interval_k4_layer2` | L2, uniform | 42.1 | n/a | **Strong content-agnostic early-layer control.** Uniform spacing at L2 is close to loss-selected L16 and only modestly below loss-selected L2, so injection depth can dominate placement policy in zero-shot accuracy. |

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

3. **Why early layers beat L62 on accuracy.** Empirically, **early** injection wins under this protocol: L6 is the current zero-shot leader, L2 is best among rows with few-shot eval, and L16/L32 also clear L62. Likely explanation: more downstream layers can transform the injected residual into answer-useful representations; late-layer hooks may be attractive for gradient locality but are not accuracy-optimal here.

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
