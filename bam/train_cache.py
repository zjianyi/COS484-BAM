#!/usr/bin/env python3
"""Phase-2: train the BAM StateCache module on top of a frozen SFT model.

The phase-1 LoRA adapter, the base Falcon Mamba weights, and the trained
embed_tokens / lm_head are all frozen. Only the StateCache parameters are
updated (~6MB). Training uses a custom forward pass that splits the
backbone at a midpoint layer, injects retrieved state from the cache at
positions immediately following a [CACHE] token, and backpropagates the
LM loss through the cache module only.

Usage:
    python -m bam.train_cache > bam_run.log 2>&1
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bam.cache import StateCache  # noqa: E402

# ---------------------------------------------------------------------------
# Hyperparameters (the phase-2 autoresearch loop sweeps these)
# ---------------------------------------------------------------------------
MODEL_NAME       = "tiiuae/falcon-mamba-7b-instruct"
SFT_ADAPTER_DIR  = "adapter"
CACHE_OUT_PATH   = "bam/cache_module.pt"
DATA_PATH        = "SFT_OpenMath_data/annotated/qwen3_235b/annotated_samples.jsonl"

NUM_EXAMPLES     = 1000
NUM_EPOCHS       = 2
LEARNING_RATE    = 5e-4
WARMUP_RATIO     = 0.03
WEIGHT_DECAY     = 0.0
MAX_SEQ_LEN      = 2048
BATCH_SIZE       = 1
GRAD_ACCUM       = 8

D_ATTN           = 256
MAX_ENTRIES      = 32
CACHE_LAYER_IDX  = None         # None => num_hidden_layers // 2

USE_CHECKPOINT   = True         # rematerialize post-cache layers in backward

TIME_BUDGET_SEC  = 30 * 60
SEED             = 42

# ---------------------------------------------------------------------------
# Fixed constants (do not sweep)
# ---------------------------------------------------------------------------
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
        load_dotenv(env_file)
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
    if len(rows) < NUM_EXAMPLES:
        print(
            f"warning: requested NUM_EXAMPLES={NUM_EXAMPLES} but only loaded "
            f"{len(rows)} rows from {DATA_PATH}",
            file=sys.stderr,
        )
    return rows


def build_examples(tokenizer, rows: list[dict]) -> list[dict]:
    """Tokenize each row as a user/assistant chat with loss masked on the prompt."""
    out = []
    for row in rows:
        messages = [
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": row["annotated_trace"]},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        prompt_text = tokenizer.apply_chat_template(
            messages[:1], add_generation_prompt=True, tokenize=False
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for i in range(prompt_len):
            labels[i] = -100
        out.append({"input_ids": full_ids, "labels": labels})
    return out


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
    if isinstance(out, tuple):
        return out[0]
    return out


def _run_layer_ckpt(layer, h, use_checkpoint: bool):
    if not use_checkpoint or not h.requires_grad:
        return _run_layer(layer, h)
    return torch.utils.checkpoint.checkpoint(
        lambda x: _run_layer(layer, x), h, use_reentrant=False
    )


def bam_forward(
    model,
    cache: StateCache,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    cache_id: int,
    layer_idx: int,
    use_checkpoint: bool,
) -> torch.Tensor:
    """Custom forward with layer-split + causal-masked cross-attention injection.

    Returns the LM loss (scalar) with gradients flowing only through the
    StateCache parameters.
    """
    backbone = _get_backbone(model)
    layers = backbone.layers
    d_attn = cache.W_Q.out_features

    # 1) embed
    embed = model.get_input_embeddings()
    h = embed(input_ids)  # (1, T, D)

    # 2) frozen prefix (layers 0..layer_idx inclusive)
    with torch.no_grad():
        for i in range(layer_idx + 1):
            h = _run_layer(layers[i], h)

    # 3) identify cache positions and the positions we'll inject into
    toks = input_ids[0]
    cache_pos = (toks == cache_id).nonzero(as_tuple=True)[0]
    if cache_pos.numel() == 0:
        # No [CACHE] tokens in this sequence — nothing for the cache to learn
        # from. Return a zero loss with a stable graph on cache params.
        return (cache.gate.float().sum() * 0.0).to(h.dtype)

    read_pos = cache_pos + 1
    read_pos = read_pos[read_pos < toks.numel()]

    # 4) keys/values from cache positions (detached, matching inference)
    cache_dtype = cache.W_K.weight.dtype
    cache_src = h[0, cache_pos].detach().to(cache_dtype)  # (N, D)
    K = cache.W_K(cache_src)              # (N, d_attn)
    V = cache.W_V(cache_src)              # (N, d_attn)

    if read_pos.numel() == 0:
        # all [CACHE] at last position; no reads
        return (cache.gate.float().sum() * 0.0).to(h.dtype)

    # 5) queries from read positions (gradients flow)
    Q = cache.W_Q(h[0, read_pos].to(cache_dtype))         # (M, d_attn)

    # 6) causal mask: entry j available for read i iff cache_pos[j] < read_pos[i]
    avail = cache_pos[None, :] < read_pos[:, None]  # (M, N) bool
    any_avail = avail.any(dim=-1)                   # (M,)

    # 7) masked cross-attention (only non-empty rows)
    scores = (Q @ K.transpose(0, 1)) / (d_attn ** 0.5)
    scores = scores.masked_fill(~avail, float("-inf"))
    delta = torch.zeros_like(h[0, read_pos])
    if any_avail.any():
        attn = scores[any_avail].softmax(dim=-1)
        out = attn @ V
        d = cache.W_out(out) * torch.sigmoid(cache.gate)
        delta[any_avail] = d.to(delta.dtype)

    # 8) inject delta onto layer-idx output at read positions
    h_prime = h.clone()
    if any_avail.any():
        rows_to_update = read_pos[any_avail]
        h_prime[0, rows_to_update] = h_prime[0, rows_to_update] + delta[any_avail]

    # 9) frozen suffix (layers layer_idx+1..end). Checkpoint to save memory.
    for i in range(layer_idx + 1, len(layers)):
        h_prime = _run_layer_ckpt(layers[i], h_prime, use_checkpoint)

    h_prime = backbone.norm_f(h_prime)

    lm_head = model.get_output_embeddings()
    logits = lm_head(h_prime)  # (1, T, V)

    # 10) standard LM loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def main() -> None:
    set_seed(SEED)
    t_start = time.time()
    hf_token = _load_token()

    print(f"Loading tokenizer + base model: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda",
        token=hf_token,
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
    print(
        f"num_hidden_layers={num_layers} hidden_size={hidden_size} cache_layer_idx={layer_idx}",
        flush=True,
    )

    device = next(model.parameters()).device
    cache = StateCache(
        d_model=hidden_size, d_attn=D_ATTN, max_entries=MAX_ENTRIES
    ).to(device=device, dtype=torch.bfloat16)
    cache.train()
    for p in cache.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in cache.parameters() if p.requires_grad)
    print(f"cache trainable params: {trainable:,}", flush=True)

    print(f"Loading training rows from {DATA_PATH}", flush=True)
    rows = load_train_rows()
    print(f"Tokenizing {len(rows)} examples", flush=True)
    examples = build_examples(tokenizer, rows)

    optimizer = torch.optim.AdamW(
        cache.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_micro_steps = NUM_EPOCHS * len(examples)
    total_opt_steps = max(1, total_micro_steps // GRAD_ACCUM)
    warmup_steps = max(1, int(WARMUP_RATIO * total_opt_steps))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    torch.cuda.reset_peak_memory_stats()
    print(f"Starting training (budget {TIME_BUDGET_SEC}s)", flush=True)

    micro_step = 0
    opt_step = 0
    running_loss = 0.0
    running_n = 0
    stopped_by_budget = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(NUM_EPOCHS):
        order = list(range(len(examples)))
        random.shuffle(order)
        for idx in order:
            if time.time() - t_start > TIME_BUDGET_SEC:
                stopped_by_budget = True
                break

            ex = examples[idx]
            ids = ex["input_ids"]
            lbls = ex["labels"]
            if len(ids) > MAX_SEQ_LEN:
                # keep the end of the trace where the answer lives
                ids = ids[-MAX_SEQ_LEN:]
                lbls = lbls[-MAX_SEQ_LEN:]

            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            labels = torch.tensor([lbls], dtype=torch.long, device=device)

            loss = bam_forward(
                model=model,
                cache=cache,
                input_ids=input_ids,
                labels=labels,
                cache_id=cache_id,
                layer_idx=layer_idx,
                use_checkpoint=USE_CHECKPOINT,
            )
            (loss / GRAD_ACCUM).backward()
            running_loss += float(loss.detach())
            running_n += 1
            micro_step += 1

            if micro_step % GRAD_ACCUM == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] = LEARNING_RATE * lr_at(opt_step)
                torch.nn.utils.clip_grad_norm_(cache.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % 10 == 0:
                    avg = running_loss / max(1, running_n)
                    elapsed = time.time() - t_start
                    print(
                        f"epoch {epoch} opt_step {opt_step}/{total_opt_steps} "
                        f"loss={avg:.4f} lr={pg['lr']:.2e} elapsed={elapsed:.0f}s",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_n = 0

        if stopped_by_budget:
            break

    # flush any remaining grads
    if micro_step % GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(cache.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        opt_step += 1

    training_seconds = time.time() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    out_path = Path(CACHE_OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "state_dict": {k: v.detach().cpu() for k, v in cache.state_dict().items()},
        "config": {
            "d_model": hidden_size,
            "d_attn": D_ATTN,
            "max_entries": MAX_ENTRIES,
            "cache_layer_idx": layer_idx,
        },
    }
    torch.save(state, out_path)
    print(f"Saved cache to {out_path}", flush=True)

    del model
    torch.cuda.empty_cache()

    print("Running bam/evaluate_bam.py", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "bam.evaluate_bam"],
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
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
    print(f"num_examples:      {len(rows)}")
    print(f"num_epochs:        {NUM_EPOCHS}")
    print(f"learning_rate:     {LEARNING_RATE}")
    print(f"d_attn:            {D_ATTN}")
    print(f"max_entries:       {MAX_ENTRIES}")
    print(f"cache_layer_idx:   {layer_idx}")
    print(f"stopped_by_budget: {stopped_by_budget}")


if __name__ == "__main__":
    main()
