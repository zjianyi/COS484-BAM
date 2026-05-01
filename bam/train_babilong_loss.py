#!/usr/bin/env python3
"""Train StateCache on BABILong with LOSS-TRIGGERED cache placement.

Instead of a task-specific regex, [CACHE] tokens are inserted after the
top-K highest-loss tokens in the passage — positions where the frozen model
is most surprised. Because this signal comes from the model itself and not
from movement-verb patterns, it generalizes across task types and domains.

Cache positions are precomputed once before training (model is frozen, so the
probe results are deterministic) and reused across epochs.

Usage:
    python -m bam.train_babilong_loss > babilong_train_loss_v1.log 2>&1
"""

from __future__ import annotations

import os
import random
import re
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
MODEL_NAME      = "tiiuae/falcon-mamba-7b-instruct"
CACHE_OUT_PATH  = "bam/cache_babilong_loss.pt"

TRAIN_TASKS     = ["qa1", "qa2", "qa3"]
TRAIN_LENGTHS   = ["0k", "1k", "2k"]
EVAL_TASKS      = ["qa1", "qa2", "qa3"]
EVAL_LENGTHS    = ["0k", "1k", "2k", "4k"]
N_TRAIN         = 50
N_EVAL          = 50

NUM_EPOCHS      = 20
LEARNING_RATE   = 5e-4
WARMUP_RATIO    = 0.05
WEIGHT_DECAY    = 2.0
MAX_SEQ_LEN     = 1536
GRAD_ACCUM      = 4

D_ATTN          = 256
MAX_ENTRIES     = 64
CACHE_LAYER_IDX = -2   # layer 62 of 64

TOP_K           = 4    # number of [CACHE] tokens to insert per passage

SEED            = 42
CACHE_TOKEN     = "[CACHE]"


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
# Loss-triggered cache placement
# ---------------------------------------------------------------------------
def probe_cache_positions(
    model,
    passage_ids: torch.Tensor,
    device: torch.device,
    top_k: int,
) -> list[int]:
    """Return sorted token positions (in passage_ids) with the highest loss.

    Runs a single no-grad forward pass. Loss at position i = NLL of predicting
    passage_ids[i+1] given passage_ids[:i+1]. High loss = model was surprised
    = a good candidate for a cache write.
    """
    if len(passage_ids) < 2:
        return []
    with torch.no_grad():
        logits = model(passage_ids.unsqueeze(0).to(device)).logits[0]  # (T, V)
    log_probs = F.log_softmax(logits, dim=-1)
    targets = passage_ids[1:].to(device)
    token_loss = -log_probs[torch.arange(len(targets)), targets]       # (T-1,)
    k = min(top_k, len(token_loss))
    return sorted(torch.topk(token_loss, k).indices.cpu().tolist())


def build_training_ids(
    passage_ids: torch.Tensor,
    cache_positions: list[int],
    cache_id: int,
    suffix_ids: list[int],
    bos_id: int | None,
) -> torch.Tensor:
    """Assemble the full token-ID sequence for one training example.

    Inserts cache_id after each position in cache_positions, appends suffix_ids
    (question + query [CACHE] + answer), and prepends BOS if provided.
    """
    pos_set = set(cache_positions)
    ids: list[int] = []
    if bos_id is not None:
        ids.append(bos_id)
    for i, tok in enumerate(passage_ids.tolist()):
        ids.append(tok)
        if i in pos_set:
            ids.append(cache_id)
    ids.extend(suffix_ids)
    return torch.tensor(ids, dtype=torch.long)


def get_passage_ids(tokenizer, row: dict, max_passage_tokens: int) -> torch.Tensor:
    text = row["input"]
    q_idx = text.rfind(row["question"])
    passage = text[:q_idx].strip() if q_idx != -1 else text
    ids = tokenizer(passage, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(ids) > max_passage_tokens:
        ids = ids[-max_passage_tokens:]
    return ids


def build_suffix_ids(tokenizer, question: str, cache_id: int, target: str) -> list[int]:
    q_ids = tokenizer(f"\n{question}", return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    t_ids = tokenizer(target.strip(), return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    return q_ids + [cache_id] + t_ids


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_babilong_rows(tasks: list[str], lengths: list[str], n_per_combo: int) -> list[dict]:
    rows: list[dict] = []
    for task in tasks:
        for length in lengths:
            try:
                ds = load_dataset("RMT-team/babilong", length, split=task)
            except Exception as exc:
                print(f"  [skip {task}/{length}: {exc}]", flush=True)
                continue
            for ex in list(ds)[:n_per_combo]:
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
    split: dict[tuple, list] = {}
    for task in tasks:
        for length in lengths:
            try:
                ds = load_dataset("RMT-team/babilong", length, split=task)
            except Exception as exc:
                print(f"  [skip eval {task}/{length}: {exc}]", flush=True)
                continue
            split[(task, length)] = list(ds)[-n_per_combo:]
    return split


# ---------------------------------------------------------------------------
# Model plumbing
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
# Forward with BABILong cache injection  (identical to train_babilong.py)
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
    cache_src = h[0, cache_pos].detach().to(cache_dtype)
    K = cache.W_K(cache_src)
    V = cache.W_V(cache_src)
    Q = cache.W_Q(h[0, cache_pos].to(cache_dtype))

    n = cache_pos.numel()
    avail = torch.tril(torch.ones(n, n, device=Q.device, dtype=torch.bool), diagonal=-1)
    any_avail = avail.any(dim=-1)

    scores = (Q @ K.T) / (d_attn ** 0.5)
    scores = scores.masked_fill(~avail, float("-inf"))

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
# Inline eval with loss-triggered placement
# ---------------------------------------------------------------------------
def run_inline_eval(
    model,
    tokenizer,
    cache: StateCache,
    cache_id: int,
    layer_idx: int,
    eval_split: dict,
    device,
    bos_id: int | None,
) -> None:
    cache.eval()
    backbone = _get_backbone(model)
    layers = backbone.layers
    margin = 64  # tokens reserved for question + query [CACHE]

    results: dict[tuple, tuple[int, int]] = {}
    baseline_results: dict[tuple, tuple[int, int]] = {}

    for (task, length), rows in eval_split.items():
        cache_correct = 0
        baseline_correct = 0

        for row in rows:
            passage_ids = get_passage_ids(tokenizer, row, MAX_SEQ_LEN - margin)
            cache_pos = probe_cache_positions(model, passage_ids, device, TOP_K)
            suffix_ids = build_suffix_ids(tokenizer, row["question"], cache_id, "")
            # Eval input: passage + [CACHE] tokens + question + query [CACHE]
            # (no target appended — we're predicting it)
            q_suffix = build_suffix_ids(tokenizer, row["question"], cache_id, "")
            input_ids_tensor = build_training_ids(
                passage_ids, cache_pos, cache_id, q_suffix, bos_id
            ).unsqueeze(0).to(device)
            if input_ids_tensor.shape[1] > MAX_SEQ_LEN:
                input_ids_tensor = input_ids_tensor[:, -MAX_SEQ_LEN:]

            target_text = row["target"].strip().lower()
            target_subtoks = [
                tokenizer.decode([i], skip_special_tokens=True).strip().lower()
                for i in tokenizer.encode(target_text, add_special_tokens=False)[:2]
            ]

            def pred_matches(pred: str) -> bool:
                pred = pred.strip().lower()
                return pred in target_subtoks or bool(
                    re.search(r"\b" + re.escape(target_text) + r"\b", pred)
                )

            # --- Cache pass ---
            embed = model.get_input_embeddings()
            with torch.no_grad():
                h = embed(input_ids_tensor)
                for i in range(layer_idx + 1):
                    h = _run_layer(layers[i], h)

            toks = input_ids_tensor[0]
            c_pos = (toks == cache_id).nonzero(as_tuple=True)[0]
            if c_pos.numel() > 0:
                cache_dtype = cache.W_K.weight.dtype
                cache_src = h[0, c_pos].detach().to(cache_dtype)
                K_e = cache.W_K(cache_src)
                V_e = cache.W_V(cache_src)
                Q_e = cache.W_Q(h[0, c_pos].to(cache_dtype))
                nc = c_pos.numel()
                av = torch.tril(torch.ones(nc, nc, device=Q_e.device, dtype=torch.bool), diagonal=-1)
                any_av = av.any(dim=-1)
                if any_av.any():
                    sc = (Q_e @ K_e.T) / (D_ATTN ** 0.5)
                    sc = sc.masked_fill(~av, float("-inf"))
                    at = sc[any_av].softmax(dim=-1)
                    ot = at @ V_e
                    gt = torch.sigmoid(cache.W_gate(h[0, c_pos][any_av].to(cache_dtype)))
                    h[0, c_pos[any_av]] = h[0, c_pos[any_av]] + cache.W_out(ot).mul(gt).to(h.dtype)

            model_dtype = next(model.parameters()).dtype
            with torch.no_grad():
                h = h.to(model_dtype)
                for i in range(layer_idx + 1, len(layers)):
                    h = _run_layer(layers[i], h)
                h = backbone.norm_f(h.to(model_dtype))
                logits = model.get_output_embeddings()(h)

            pred = tokenizer.decode(
                [logits[0, -1, :].argmax(dim=-1).item()], skip_special_tokens=True
            )
            if pred_matches(pred):
                cache_correct += 1

            # --- No-cache baseline (same placement, no delta) ---
            with torch.no_grad():
                out = model(input_ids_tensor)
            pred_b = tokenizer.decode(
                [out.logits[0, -1, :].argmax(dim=-1).item()], skip_special_tokens=True
            )
            if pred_matches(pred_b):
                baseline_correct += 1

        results[(task, length)] = (cache_correct, len(rows))
        baseline_results[(task, length)] = (baseline_correct, len(rows))

    print("\n=== BABILong Cache Eval (loss-triggered placement) ===")
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
    bos_id = tokenizer.bos_token_id
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

    # ------------------------------------------------------------------
    # Precompute cache positions for all training examples (model is frozen
    # so probe results are the same every epoch — no need to recompute).
    # ------------------------------------------------------------------
    margin = 64
    print(f"Precomputing top-{TOP_K} loss-triggered cache positions ...", flush=True)
    precomputed: list[tuple[torch.Tensor, list[int], list[int]]] = []
    for i, row in enumerate(train_rows):
        passage_ids = get_passage_ids(tokenizer, row, MAX_SEQ_LEN - margin)
        positions = probe_cache_positions(model, passage_ids, device, TOP_K)
        suffix_ids = build_suffix_ids(tokenizer, row["question"], cache_id, row["target"])
        precomputed.append((passage_ids, positions, suffix_ids))
        if (i + 1) % 50 == 0:
            print(f"  probed {i+1}/{len(train_rows)}", flush=True)

    n_with_cache = sum(1 for _, pos, _ in precomputed if pos)
    print(f"Precompute done: {n_with_cache}/{len(train_rows)} examples have ≥1 cache position", flush=True)

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
            passage_ids, positions, suffix_ids = precomputed[idx]

            if not positions:
                micro_step += 1
                if micro_step % GRAD_ACCUM == 0:
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1
                continue

            seq_ids = build_training_ids(passage_ids, positions, cache_id, suffix_ids, bos_id)
            full_ids = seq_ids.unsqueeze(0).to(device)
            if full_ids.shape[1] > MAX_SEQ_LEN:
                full_ids = full_ids[:, -MAX_SEQ_LEN:]

            lab = full_ids.clone()
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

    print(f"\nRunning inline BABILong eval ...", flush=True)
    run_inline_eval(model, tokenizer, cache, cache_id, layer_idx, eval_split, device, bos_id)

    print(f"\nTotal time: {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
