#!/usr/bin/env python3
"""Train StateCache on BABILong entity-tracking task.

No SFT adapter needed. [CACHE] tokens are inserted deterministically
by regex after each fact sentence. The cache learns to store entity-location
snapshots and retrieve the relevant one at query time.

Key architectural change vs. train_cache.py:
  read_pos = cache_pos (inject delta AT [CACHE] positions, not cache_pos+1)
  → logits[cache_pos] predicts the next token (fact continuation or answer)
  → focused loss on ONLY the last [CACHE] → answer token

Usage:
    python -m bam.train_babilong > babilong_train.log 2>&1
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bam.cache import StateCache  # noqa: E402

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MODEL_NAME       = "tiiuae/falcon-mamba-7b-instruct"
CACHE_OUT_PATH   = "bam/cache_babilong.pt"

TRAIN_TASKS      = ["qa1", "qa2", "qa3"]
TRAIN_LENGTHS    = ["0k", "1k", "2k"]
EVAL_TASKS       = ["qa1", "qa2", "qa3"]
EVAL_LENGTHS     = ["0k", "1k", "2k", "4k"]
N_TRAIN          = 50    # first 50 examples per task/length combo (no overlap with eval)
N_EVAL           = 50    # last 50 examples per task/length combo

NUM_EPOCHS       = 20
LEARNING_RATE    = 5e-4  # lower LR to slow W_out_norm growth
WARMUP_RATIO     = 0.05
WEIGHT_DECAY     = 2.0   # stronger regularization
MAX_SEQ_LEN      = 1536
GRAD_ACCUM       = 4

D_ATTN           = 256
MAX_ENTRIES      = 64    # more entries for longer passages
CACHE_LAYER_IDX  = -2    # layer 62 of 64

SEED             = 42
CACHE_TOKEN      = "[CACHE]"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_token() -> str | None:
    env_file = ROOT / ".env"
    if env_file.is_file():
        from dotenv import load_dotenv as _ld
        _ld(env_file)
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def add_cache_token(tokenizer, model) -> int:
    if CACHE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [CACHE_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer.convert_tokens_to_ids(CACHE_TOKEN)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
# Matches BABILong entity-movement facts: "John went to the kitchen."
FACT_RE = re.compile(
    r'\b\w+\s+(?:travelled|went|journeyed|moved|went\s+back)\s+(?:back\s+)?to\s+(?:the\s+)?\w+\.',
    re.IGNORECASE,
)


def annotate_example(input_text: str, question: str, target: str) -> str:
    """Build teacher-forced sequence with [CACHE] tokens.

    Inserts [CACHE] ONLY after entity-movement fact sentences (not distractors),
    then appends: {question}[CACHE]{target}

    This keeps the number of [CACHE] tokens constant (~4-10) regardless of
    context length, preserving Mamba's linear-time advantage for distractors
    while the cache stores only semantically relevant entity-location snapshots.
    """
    q_idx = input_text.rfind(question)
    passage = input_text[:q_idx].strip() if q_idx != -1 else input_text

    sentences = SENT_SPLIT.split(passage)
    annotated_parts = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if FACT_RE.search(s):
            annotated_parts.append(s + f" {CACHE_TOKEN}")
        else:
            annotated_parts.append(s)

    annotated_passage = " ".join(annotated_parts)
    return f"{annotated_passage}\n{question}{CACHE_TOKEN}{target}"


def load_babilong_rows(tasks: list[str], lengths: list[str], n_per_combo: int) -> list[dict]:
    rows: list[dict] = []
    for task in tasks:
        for length in lengths:
            try:
                ds = load_dataset("RMT-team/babilong", length, split=task)
            except Exception as exc:
                print(f"  [skip {task}/{length}: {exc}]", flush=True)
                continue
            subset = list(ds)[:n_per_combo]
            for ex in subset:
                rows.append({
                    "input": ex["input"],
                    "question": ex["question"],
                    "target": ex["target"],
                    "task": task,
                    "length": length,
                })
    print(f"Loaded {len(rows)} BABILong train rows", flush=True)
    return rows


def load_babilong_eval(tasks: list[str], lengths: list[str], n_per_combo: int) -> dict:
    """Returns {(task, length): [rows]}"""
    split: dict[tuple, list] = {}
    for task in tasks:
        for length in lengths:
            try:
                ds = load_dataset("RMT-team/babilong", length, split=task)
            except Exception as exc:
                print(f"  [skip eval {task}/{length}: {exc}]", flush=True)
                continue
            # Use last n_per_combo as held-out
            examples = list(ds)[-n_per_combo:]
            split[(task, length)] = examples
    return split


# ---------------------------------------------------------------------------
# Model plumbing (same as train_cache.py)
# ---------------------------------------------------------------------------
def _get_backbone(model):
    if hasattr(model, "backbone"):
        return model.backbone
    base = getattr(model, "base_model", None)
    if base is not None:
        inner = getattr(base, "model", base)
        if hasattr(inner, "backbone"):
            return inner.backbone
        if hasattr(inner, "model") and hasattr(inner.model, "backbone"):
            return inner.model.backbone
    raise AttributeError("Could not locate `.backbone` on the given model")


def _run_layer(layer, h):
    out = layer(h)
    return out[0] if isinstance(out, tuple) else out


def _run_layer_ckpt(layer, h, use_checkpoint: bool):
    if not use_checkpoint or not h.requires_grad:
        return _run_layer(layer, h)
    return torch.utils.checkpoint.checkpoint(
        lambda x: _run_layer(layer, x), h, use_reentrant=False
    )


# ---------------------------------------------------------------------------
# Forward with BABILong cache injection
# ---------------------------------------------------------------------------
def bam_forward_babilong(
    model,
    cache: StateCache,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    cache_id: int,
    layer_idx: int,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """Focused CE loss for BABILong entity tracking.

    Injects cache delta AT [CACHE] positions (read_pos = cache_pos).
    Loss is computed only on the LAST [CACHE] → answer token prediction:
        CE(logits[last_cache_pos], labels[last_cache_pos + 1])

    This differs from train_cache.py which uses read_pos = cache_pos + 1.
    """
    backbone = _get_backbone(model)
    layers = backbone.layers
    d_attn = cache.W_Q.out_features

    embed = model.get_input_embeddings()
    h = embed(input_ids)

    with torch.no_grad():
        for i in range(layer_idx + 1):
            h = _run_layer(layers[i], h)

    toks = input_ids[0]
    cache_pos = (toks == cache_id).nonzero(as_tuple=True)[0]
    if cache_pos.numel() == 0:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype, requires_grad=True)

    cache_dtype = cache.W_K.weight.dtype

    # Write K, V from all [CACHE] positions (detached — no gradient through writes)
    cache_src = h[0, cache_pos].detach().to(cache_dtype)
    K = cache.W_K(cache_src)    # (n_cache, d_attn)
    V = cache.W_V(cache_src)    # (n_cache, d_attn)

    # Query AT [CACHE] positions (with gradient)
    Q = cache.W_Q(h[0, cache_pos].to(cache_dtype))  # (n_cache, d_attn)

    # Strictly causal mask: each [CACHE] only attends to EARLIER [CACHE]s
    n = cache_pos.numel()
    avail = torch.tril(torch.ones(n, n, device=Q.device, dtype=torch.bool), diagonal=-1)
    any_avail = avail.any(dim=-1)

    scores = (Q @ K.T) / (d_attn ** 0.5)
    scores = scores.masked_fill(~avail, float("-inf"))

    # If no writes are available for any position (e.g. only 1 [CACHE] token),
    # there is no learning signal — skip this example.
    if not any_avail.any():
        return torch.tensor(0.0, device=h.device, dtype=h.dtype, requires_grad=True)

    delta = torch.zeros_like(h[0, cache_pos])
    attn = scores[any_avail].softmax(dim=-1)
    out = attn @ V
    gate = torch.sigmoid(cache.W_gate(h[0, cache_pos][any_avail].to(cache_dtype)))
    d = cache.W_out(out) * gate
    delta[any_avail] = d.to(delta.dtype)

    h_prime = h.clone()
    h_prime[0, cache_pos[any_avail]] = h_prime[0, cache_pos[any_avail]] + delta[any_avail]

    model_dtype = next(model.parameters()).dtype
    h_prime = h_prime.to(model_dtype)
    for i in range(layer_idx + 1, len(layers)):
        h_prime = _run_layer_ckpt(layers[i], h_prime, use_checkpoint)

    h_prime = backbone.norm_f(h_prime.to(model_dtype))
    logits = model.get_output_embeddings()(h_prime)

    # Focused loss: only on LAST [CACHE] → next token (the answer)
    seq_len = labels.shape[1]
    last_cache = cache_pos[-1].item()
    if last_cache + 1 >= seq_len:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype, requires_grad=True)

    focused_labels = labels.clone()
    focused_labels[0, :] = -100
    focused_labels[0, last_cache + 1] = labels[0, last_cache + 1]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = focused_labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# Inline eval (no subprocess — same model, no cache module)
# ---------------------------------------------------------------------------
def run_inline_eval(
    model,
    tokenizer,
    cache: StateCache,
    cache_id: int,
    layer_idx: int,
    eval_split: dict,
    device,
) -> None:
    """Run greedy eval on held-out BABILong examples with cache injection."""
    cache.eval()
    backbone = _get_backbone(model)
    layers = backbone.layers

    results: dict[tuple, tuple[int, int]] = {}

    for (task, length), rows in eval_split.items():
        correct = 0
        for row in rows:
            seq = annotate_example(row["input"], row["question"], row["target"])
            # Build input up to (but not including) the answer
            # The sequence ends with ...{question}[CACHE]{answer}
            # We want the input to be ...{question}[CACHE]
            cache_tok = CACHE_TOKEN
            last_cache_idx = seq.rfind(cache_tok)
            input_text = seq[:last_cache_idx + len(cache_tok)]
            target_text = row["target"].strip().lower()

            input_ids = tokenizer(
                input_text, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(device)

            if input_ids.shape[1] > MAX_SEQ_LEN:
                input_ids = input_ids[:, -MAX_SEQ_LEN:]

            # Forward pass with cache injection
            embed = model.get_input_embeddings()
            with torch.no_grad():
                h = embed(input_ids)
                for i in range(layer_idx + 1):
                    h = _run_layer(layers[i], h)

            toks = input_ids[0]
            c_pos = (toks == cache_id).nonzero(as_tuple=True)[0]

            if c_pos.numel() > 0:
                cache_dtype = cache.W_K.weight.dtype
                cache_src = h[0, c_pos].detach().to(cache_dtype)
                K = cache.W_K(cache_src)
                V = cache.W_V(cache_src)
                Q = cache.W_Q(h[0, c_pos].to(cache_dtype))
                n = c_pos.numel()
                avail = torch.tril(
                    torch.ones(n, n, device=Q.device, dtype=torch.bool), diagonal=-1
                )
                any_avail = avail.any(dim=-1)
                if any_avail.any():
                    scores = (Q @ K.T) / (D_ATTN ** 0.5)
                    scores = scores.masked_fill(~avail, float("-inf"))
                    attn = scores[any_avail].softmax(dim=-1)
                    out = attn @ V
                    gate = torch.sigmoid(cache.W_gate(h[0, c_pos][any_avail].to(cache_dtype)))
                    d = cache.W_out(out) * gate
                    h[0, c_pos[any_avail]] = h[0, c_pos[any_avail]] + d.to(h.dtype)

            model_dtype = next(model.parameters()).dtype
            with torch.no_grad():
                h = h.to(model_dtype)
                for i in range(layer_idx + 1, len(layers)):
                    h = _run_layer(layers[i], h)
                h = backbone.norm_f(h.to(model_dtype))
                logits = model.get_output_embeddings()(h)

            # Greedy pick at last [CACHE] position (delta injected here)
            next_token_id = logits[0, -1, :].argmax(dim=-1).item()
            pred = tokenizer.decode([next_token_id], skip_special_tokens=True).strip().lower()

            # Match against first subtoken(s) of target (handles "hallway"→["hall","way"])
            target_subtoks = [
                tokenizer.decode([i], skip_special_tokens=True).strip().lower()
                for i in tokenizer.encode(target_text, add_special_tokens=False)[:2]
            ]
            ok = pred in target_subtoks or bool(
                re.search(r"\b" + re.escape(target_text) + r"\b", pred)
            )
            if ok:
                correct += 1

        results[(task, length)] = (correct, len(rows))

    # No-cache baseline with identical protocol (same input, same subtoken match, no delta)
    baseline_results: dict[tuple, tuple[int, int]] = {}
    for (task, length), rows in eval_split.items():
        correct = 0
        for row in rows:
            seq = annotate_example(row["input"], row["question"], row["target"])
            last_cache_idx = seq.rfind(CACHE_TOKEN)
            input_text = seq[:last_cache_idx + len(CACHE_TOKEN)]
            target_text = row["target"].strip().lower()
            input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
            if input_ids.shape[1] > MAX_SEQ_LEN:
                input_ids = input_ids[:, -MAX_SEQ_LEN:]
            with torch.no_grad():
                out = model(input_ids)
            next_token_id = out.logits[0, -1, :].argmax(dim=-1).item()
            pred = tokenizer.decode([next_token_id], skip_special_tokens=True).strip().lower()
            target_subtoks = [
                tokenizer.decode([i], skip_special_tokens=True).strip().lower()
                for i in tokenizer.encode(target_text, add_special_tokens=False)[:2]
            ]
            if pred in target_subtoks or bool(re.search(r"\b" + re.escape(target_text) + r"\b", pred)):
                correct += 1
        baseline_results[(task, length)] = (correct, len(rows))

    print("\n=== BABILong Cache Eval (same-protocol baseline vs cache) ===")
    all_lengths = sorted({l for _, l in results.keys()})
    print(f"{'task':<6}  {'metric':<10}" + "".join(f"  {l:>5s}" for l in all_lengths))
    for task in EVAL_TASKS:
        for label, res in [("no_cache", baseline_results), ("cache", results)]:
            row_str = f"{task:<6}  {label:<10}"
            for length in all_lengths:
                key = (task, length)
                if key in res:
                    c, n = res[key]
                    row_str += f"  {c/n:.3f}"
                else:
                    row_str += "     -"
            print(row_str)

    cache.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    t_start = time.time()
    hf_token = _load_token()

    print(f"Loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=hf_token,
    )
    cache_id = add_cache_token(tokenizer, model)
    model.eval()
    model.requires_grad_(False)

    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    layer_idx = num_layers + CACHE_LAYER_IDX if CACHE_LAYER_IDX < 0 else CACHE_LAYER_IDX
    print(f"num_layers={num_layers} hidden_size={hidden_size} cache_layer_idx={layer_idx}", flush=True)

    device = next(model.parameters()).device
    cache = StateCache(d_model=hidden_size, d_attn=D_ATTN, max_entries=MAX_ENTRIES).to(device=device)
    cache.train()
    for p in cache.parameters():
        p.requires_grad = True
    print(f"cache trainable params: {sum(p.numel() for p in cache.parameters()):,}", flush=True)

    train_rows = load_babilong_rows(TRAIN_TASKS, TRAIN_LENGTHS, N_TRAIN)
    eval_split = load_babilong_eval(EVAL_TASKS, EVAL_LENGTHS, N_EVAL)

    total_opt_steps_est = (NUM_EPOCHS * len(train_rows)) // GRAD_ACCUM
    warmup_steps = max(1, int(WARMUP_RATIO * total_opt_steps_est))

    optimizer = torch.optim.AdamW(
        cache.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_opt_steps_est - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    micro_step = 0
    opt_step = 0
    optimizer.zero_grad(set_to_none=True)
    total_loss_sum = 0.0
    total_examples = 0

    for epoch in range(NUM_EPOCHS):
        order = list(range(len(train_rows)))
        random.shuffle(order)
        epoch_loss_sum = 0.0
        epoch_count = 0

        for idx in order:
            row = train_rows[idx]
            seq = annotate_example(row["input"], row["question"], row["target"])

            full_ids = tokenizer(
                seq, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(device)

            if full_ids.shape[1] > MAX_SEQ_LEN:
                full_ids = full_ids[:, -MAX_SEQ_LEN:]

            lab = full_ids.clone()  # loss on answer token (set in bam_forward_babilong)

            elapsed = time.time() - t_start
            loss = bam_forward_babilong(
                model=model, cache=cache, input_ids=full_ids, labels=lab,
                cache_id=cache_id, layer_idx=layer_idx, use_checkpoint=True,
            )

            if loss.item() == 0.0:
                micro_step += 1
                if micro_step % GRAD_ACCUM == 0:
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1
                continue

            (loss / GRAD_ACCUM).backward()
            micro_step += 1
            epoch_loss_sum += loss.item()
            total_loss_sum += loss.item()
            epoch_count += 1
            total_examples += 1

            if micro_step % GRAD_ACCUM == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] = LEARNING_RATE * lr_at(opt_step)
                torch.nn.utils.clip_grad_norm_(cache.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % 20 == 0:
                    avg_loss = total_loss_sum / max(1, total_examples)
                    w_out_norm = cache.W_out.weight.data.norm().item()
                    print(
                        f"epoch {epoch} opt_step {opt_step} "
                        f"avg_loss={avg_loss:.4f} W_out_norm={w_out_norm:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} elapsed={elapsed:.0f}s",
                        flush=True,
                    )

        epoch_avg = epoch_loss_sum / max(1, epoch_count)
        w_out_norm = cache.W_out.weight.data.norm().item()
        print(
            f"epoch {epoch} done: {epoch_count} examples, "
            f"avg_loss={epoch_avg:.4f} W_out_norm={w_out_norm:.4f}",
            flush=True,
        )

        ckpt_path = Path(CACHE_OUT_PATH).with_suffix(f".ep{epoch}.pt")
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in cache.state_dict().items()},
            "config": {
                "d_model": hidden_size, "d_attn": D_ATTN,
                "max_entries": MAX_ENTRIES, "cache_layer_idx": layer_idx,
            },
        }, ckpt_path)

    # Flush remaining grads
    if micro_step % GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(cache.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    out_path = Path(CACHE_OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in cache.state_dict().items()},
        "config": {
            "d_model": hidden_size, "d_attn": D_ATTN,
            "max_entries": MAX_ENTRIES, "cache_layer_idx": layer_idx,
        },
    }, out_path)
    print(f"Saved cache to {out_path}", flush=True)

    # Inline eval (reuses loaded model)
    print(f"\nRunning inline BABILong eval ...", flush=True)
    run_inline_eval(model, tokenizer, cache, cache_id, layer_idx, eval_split, device)

    total_sec = time.time() - t_start
    print(f"\nTotal time: {total_sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
