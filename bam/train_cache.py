#!/usr/bin/env python3
"""Phase-2: train the BAM StateCache module on top of a frozen SFT model.

Supervised LM loss: teacher-forced over OpenMath annotated traces (which
contain explicit [CACHE] tokens).  Injection at layer 62 (second-to-last)
means only ONE frozen Mamba block remains in the backward gradient path,
versus 32 in the earlier experiments that all used layer_idx=32.

Usage:
    python -m bam.train_cache > bam_run.log 2>&1
"""

from __future__ import annotations

import json
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bam.cache import StateCache  # noqa: E402

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MODEL_NAME       = "tiiuae/falcon-mamba-7b-instruct"
SFT_ADAPTER_DIR  = "adapter"
CACHE_OUT_PATH   = "bam/cache_module.pt"
DATA_PATH        = "SFT_OpenMath_data/annotated/qwen3_235b/annotated_samples.jsonl"

NUM_EXAMPLES     = 1000
NUM_EPOCHS       = 5
LEARNING_RATE    = 2e-3
WARMUP_RATIO     = 0.05
WEIGHT_DECAY     = 1.0   # strong regularization: caps W_out_norm ≈ 1/WD ≈ 1.0
MAX_SEQ_LEN      = 2048
GRAD_ACCUM       = 4

D_ATTN           = 256
MAX_ENTRIES      = 32
# Inject at second-to-last layer (-2 → 62 for 64-layer model).
# Only 1 frozen Mamba block in the backward gradient path (vs 32 in all prior runs).
CACHE_LAYER_IDX  = -2

USE_CHECKPOINT   = True
TIME_BUDGET_SEC  = 4 * 3600
SEED             = 42

CACHE_TOKEN      = "[CACHE]"
HELD_OUT_START   = 4970
HELD_OUT_END     = 5000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_token() -> str | None:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.is_file():
        from dotenv import load_dotenv as _ld
        _ld(env_file)
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def add_cache_token(tokenizer, model) -> int:
    if CACHE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [CACHE_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer.convert_tokens_to_ids(CACHE_TOKEN)


def load_train_rows() -> list[dict]:
    rows: list[dict] = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= HELD_OUT_START:
                break
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= NUM_EXAMPLES:
                break
    print(f"Loaded {len(rows)} OpenMath train rows", flush=True)
    return rows


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
# Supervised LM-loss forward with cache injection
# ---------------------------------------------------------------------------
def bam_forward(
    model,
    cache: StateCache,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    cache_id: int,
    layer_idx: int,
    use_checkpoint: bool,
) -> torch.Tensor:
    """CE loss with cache delta injected at layer_idx.

    Layers 0..layer_idx run in no_grad (frozen prefix).
    Cache delta is computed with grad from cache parameters.
    Layers layer_idx+1..N-1 run with grad tracking so delta propagates.
    Returns 0-loss (no_grad) when no [CACHE] tokens are present.
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

    read_pos = cache_pos + 1
    read_pos = read_pos[read_pos < toks.numel()]
    if read_pos.numel() == 0:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype, requires_grad=True)

    cache_dtype = cache.W_K.weight.dtype
    cache_src = h[0, cache_pos].detach().to(cache_dtype)
    K = cache.W_K(cache_src)
    V = cache.W_V(cache_src)

    Q = cache.W_Q(h[0, read_pos].to(cache_dtype))
    avail = cache_pos[None, :] < read_pos[:, None]
    any_avail = avail.any(dim=-1)

    scores = (Q @ K.transpose(0, 1)) / (d_attn ** 0.5)
    scores = scores.masked_fill(~avail, float("-inf"))
    delta = torch.zeros_like(h[0, read_pos])
    if any_avail.any():
        attn = scores[any_avail].softmax(dim=-1)
        out = attn @ V
        gate = torch.sigmoid(cache.W_gate(h[0, read_pos][any_avail].to(cache_dtype)))
        d = cache.W_out(out) * gate
        delta[any_avail] = d.to(delta.dtype)

    h_prime = h.clone()
    if any_avail.any():
        rows_to_update = read_pos[any_avail]
        h_prime[0, rows_to_update] = h_prime[0, rows_to_update] + delta[any_avail]

    model_dtype = next(model.parameters()).dtype
    h_prime = h_prime.to(model_dtype)
    for i in range(layer_idx + 1, len(layers)):
        h_prime = _run_layer_ckpt(layers[i], h_prime, use_checkpoint)

    h_prime = backbone.norm_f(h_prime.to(model_dtype))
    logits = model.get_output_embeddings()(h_prime)

    # Focused loss: compute CE only at the tokens immediately predicted by
    # cache-modified logits (logits[read_pos] predicts labels[read_pos+1]).
    # This avoids 2000x gradient dilution from the full-sequence average.
    seq_len = labels.shape[1]
    # read_pos are the positions where the cache delta is injected.
    # logits[read_pos] is what we want to train — the prediction right after cache.
    target_pos = read_pos[read_pos + 1 < seq_len] + 1   # labels[read_pos+1] = target tokens
    if target_pos.numel() == 0:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype, requires_grad=True)
    focused_labels = labels.clone()
    focused_labels[0, :] = -100
    focused_labels[0, target_pos] = labels[0, target_pos]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = focused_labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    t_start = time.time()
    hf_token = _load_token()

    print(f"Loading tokenizer + base model: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda", token=hf_token,
    )
    cache_id = add_cache_token(tokenizer, model)

    print(f"Loading SFT adapter from {SFT_ADAPTER_DIR}", flush=True)
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_DIR)
    model.eval()
    model.requires_grad_(False)

    config = model.config if hasattr(model, "config") else model.base_model.model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    layer_idx = CACHE_LAYER_IDX if CACHE_LAYER_IDX is not None else num_layers // 2
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx
    print(f"num_hidden_layers={num_layers} hidden_size={hidden_size} cache_layer_idx={layer_idx}", flush=True)

    device = next(model.parameters()).device
    cache = StateCache(d_model=hidden_size, d_attn=D_ATTN, max_entries=MAX_ENTRIES).to(device=device)
    cache.train()
    for p in cache.parameters():
        p.requires_grad = True
    print(f"cache trainable params: {sum(p.numel() for p in cache.parameters()):,}", flush=True)

    rows = load_train_rows()

    total_opt_steps_est = (NUM_EPOCHS * len(rows)) // GRAD_ACCUM
    warmup_steps = max(1, int(WARMUP_RATIO * total_opt_steps_est))

    optimizer = torch.optim.AdamW(
        cache.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_opt_steps_est - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    torch.cuda.reset_peak_memory_stats()

    micro_step = 0
    opt_step = 0
    stopped_by_budget = False
    optimizer.zero_grad(set_to_none=True)
    total_loss_sum = 0.0
    total_examples = 0

    for epoch in range(NUM_EPOCHS):
        order = list(range(len(rows)))
        random.shuffle(order)
        epoch_loss_sum = 0.0
        epoch_count = 0

        for idx in order:
            if time.time() - t_start > TIME_BUDGET_SEC:
                stopped_by_budget = True
                break

            row = rows[idx]

            # Build teacher-forced sequence: prompt + annotated trace
            msgs = [{"role": "user", "content": row["problem"]}]
            prompt_ids = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt"
            ).to(device)
            trace_ids = tokenizer(
                row["annotated_trace"],
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(device)
            eos_tensor = torch.tensor([[tokenizer.eos_token_id]], device=device)
            full_ids = torch.cat([prompt_ids, trace_ids, eos_tensor], dim=1)

            if full_ids.shape[1] > MAX_SEQ_LEN:
                full_ids = full_ids[:, :MAX_SEQ_LEN]

            lab = full_ids.clone()
            lab[0, : prompt_ids.shape[1]] = -100   # mask prompt tokens

            elapsed = time.time() - t_start
            loss = bam_forward(
                model=model, cache=cache, input_ids=full_ids, labels=lab,
                cache_id=cache_id, layer_idx=layer_idx, use_checkpoint=USE_CHECKPOINT,
            )

            if loss.item() == 0.0:
                # No [CACHE] tokens in this example — skip
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
                        f"avg_loss={avg_loss:.4f} "
                        f"W_out_norm={w_out_norm:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} elapsed={elapsed:.0f}s",
                        flush=True,
                    )

        epoch_avg = epoch_loss_sum / max(1, epoch_count)
        w_out_norm = cache.W_out.weight.data.norm().item()
        print(f"epoch {epoch} done: {epoch_count} examples, avg_loss={epoch_avg:.4f} W_out_norm={w_out_norm:.4f}", flush=True)
        # Save per-epoch checkpoint so we can recover the best point
        ckpt_path = Path(CACHE_OUT_PATH).with_suffix(f".ep{epoch}.pt")
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in cache.state_dict().items()},
            "config": {"d_model": hidden_size, "d_attn": D_ATTN,
                       "max_entries": MAX_ENTRIES, "cache_layer_idx": layer_idx},
        }, ckpt_path)
        if stopped_by_budget:
            break

    # flush remaining grads
    if micro_step % GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(cache.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        opt_step += 1

    training_seconds = time.time() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    final_avg_loss = total_loss_sum / max(1, total_examples)
    print(
        f"Training done: {total_examples} examples, avg_loss={final_avg_loss:.4f}, "
        f"{opt_step} opt steps in {training_seconds:.0f}s",
        flush=True,
    )

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

    del model
    torch.cuda.empty_cache()

    print("Running bam/evaluate_bam.py", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "bam.evaluate_bam"],
        stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1,
    )
    eval_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        eval_lines.append(line)
    proc.wait()
    eval_stdout = "".join(eval_lines)

    metrics: dict[str, float] = {}
    for line in eval_stdout.splitlines():
        for key in ("cache_f1", "math_accuracy", "cache_count_mae", "cache_writes_avg"):
            prefix = f"{key}:"
            if line.startswith(prefix):
                try:
                    metrics[key] = float(line[len(prefix):].strip())
                except ValueError:
                    pass

    total_seconds = time.time() - t_start
    print("---")
    print(f"cache_f1:          {metrics.get('cache_f1', 0.0):.4f}")
    print(f"math_accuracy:     {metrics.get('math_accuracy', 0.0):.4f}")
    print(f"cache_count_mae:   {metrics.get('cache_count_mae', 0.0):.4f}")
    print(f"cache_writes_avg:  {metrics.get('cache_writes_avg', 0.0):.4f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"avg_loss:          {final_avg_loss:.4f}")
    print(f"num_examples:      {len(rows)}")
    print(f"num_epochs:        {NUM_EPOCHS}")
    print(f"learning_rate:     {LEARNING_RATE}")
    print(f"d_attn:            {D_ATTN}")
    print(f"cache_layer_idx:   {layer_idx}")
    print(f"stopped_by_budget: {stopped_by_budget}")


if __name__ == "__main__":
    main()
