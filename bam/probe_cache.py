#!/usr/bin/env python3
"""Diagnostic probe for the BAM StateCache.

Checks three things:
  1. Gate health: is sigmoid(gate) usefully open after training?
  2. Delta magnitude: how much does the cache injection change hidden states?
  3. Cache-on vs cache-off: does disabling the gate change generation outputs?

Usage:
    python -m bam.probe_cache
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bam.cache import StateCache  # noqa: E402
from bam.generator import BAMGenerator  # noqa: E402
from evaluate import (  # noqa: E402
    CACHE_TOKEN, HELD_OUT_START, MODEL_NAME, _load_token, add_cache_token, load_held_out,
)

SFT_ADAPTER_DIR = "adapter"
CACHE_CKPT_PATH = "bam/cache_module.pt"
PROBE_N = 5  # problems to compare cache-on vs cache-off


def main() -> None:
    hf_token = _load_token()
    device = torch.device("cuda")

    # ------------------------------------------------------------------ #
    # 1. W_out scale check (gate removed)
    # ------------------------------------------------------------------ #
    ckpt = torch.load(CACHE_CKPT_PATH, map_location="cpu")
    cfg = ckpt["config"]
    cache = StateCache(d_model=cfg["d_model"], d_attn=cfg["d_attn"], max_entries=cfg["max_entries"])
    cache.load_state_dict(ckpt["state_dict"])
    w_out_norm = cache.W_out.weight.norm().item()
    print(f"[probe 1] W_out weight norm={w_out_norm:.4f} (0=untrained, >0=active)")
    if w_out_norm < 1e-3:
        print("  !! W_out is near-zero — cache not learning")
    else:
        print("  W_out has trained")

    # ------------------------------------------------------------------ #
    # 2. Delta magnitude (requires model)
    # ------------------------------------------------------------------ #
    print("\n[probe 2] Loading model to measure delta magnitude...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda", token=hf_token
    )
    cache_id = add_cache_token(tokenizer, model)
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_DIR)
    model.eval()
    model.requires_grad_(False)

    from bam.train_cache import _get_backbone, _run_layer  # noqa: E402
    backbone = _get_backbone(model)
    layers = backbone.layers
    layer_idx = cfg["cache_layer_idx"]

    cache_module = StateCache(
        d_model=cfg["d_model"], d_attn=cfg["d_attn"], max_entries=cfg["max_entries"]
    ).to(device=device)
    cache_module.load_state_dict(ckpt["state_dict"])
    cache_module.eval()

    rows = load_held_out()
    row = rows[0]
    messages = [{"role": "user", "content": row["problem"]}]
    full_text = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": row["annotated_trace"]}],
        add_generation_prompt=False, tokenize=False,
    )
    ids = tokenizer.encode(full_text, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        embed = model.get_input_embeddings()
        h = embed(input_ids)
        for i in range(layer_idx + 1):
            h = _run_layer(layers[i], h)

    toks = input_ids[0]
    cache_pos = (toks == cache_id).nonzero(as_tuple=True)[0]
    read_pos = cache_pos + 1
    read_pos = read_pos[read_pos < toks.numel()]

    if cache_pos.numel() == 0:
        print("  No [CACHE] tokens in sample — skipping delta probe")
    else:
        cache_dtype = cache_module.W_K.weight.dtype
        cache_src = h[0, cache_pos].detach().to(cache_dtype)
        K = cache_module.W_K(cache_src)
        V = cache_module.W_V(cache_src)

        if read_pos.numel() > 0:
            Q = cache_module.W_Q(h[0, read_pos].to(cache_dtype))
            d_attn = cache_module.W_Q.out_features
            avail = cache_pos[None, :] < read_pos[:, None]
            scores = (Q @ K.T) / (d_attn ** 0.5)
            scores = scores.masked_fill(~avail, float("-inf"))
            any_avail = avail.any(dim=-1)

            deltas = []
            if any_avail.any():
                attn = scores[any_avail].softmax(dim=-1)
                out = attn @ V
                d = cache_module.W_out(out)
                deltas = d

            h_norms = h[0, read_pos].norm(dim=-1)
            if len(deltas) > 0:
                d_norms = deltas.norm(dim=-1)
                rel = (d_norms / h_norms[any_avail]).mean().item()
                print(f"  {cache_pos.numel()} cache writes, {read_pos.numel()} read positions")
                print(f"  delta norm (mean): {d_norms.mean().item():.4f}")
                print(f"  h norm (mean):     {h_norms.mean().item():.4f}")
                print(f"  relative delta:    {rel:.6f}")
                if rel < 1e-4:
                    print("  !! delta is NEGLIGIBLE — injection has no effect")
                elif rel < 0.01:
                    print("  !! delta is very small (<1% of h norm)")
                else:
                    print("  delta magnitude looks non-trivial")
            else:
                print("  No available cache entries for any read position")

    # ------------------------------------------------------------------ #
    # 3. Cache-on vs cache-off generation comparison
    # ------------------------------------------------------------------ #
    print(f"\n[probe 3] Cache-on vs cache-off on {PROBE_N} problems...")

    def run_gen(gate_override: float | None) -> list[list[int]]:
        cm = StateCache(
            d_model=cfg["d_model"], d_attn=cfg["d_attn"], max_entries=cfg["max_entries"]
        ).to(device=device)
        cm.load_state_dict(ckpt["state_dict"])
        cm.eval()
        if gate_override is not None:
            cm.gate.data.fill_(gate_override)

        gen = BAMGenerator(model=model, tokenizer=tokenizer, cache_module=cm,
                           cache_layer_idx=layer_idx, cache_token=CACHE_TOKEN)
        results = []
        try:
            for row in rows[:PROBE_N]:
                msgs = [{"role": "user", "content": row["problem"]}]
                prompt_ids = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt"
                ).to(device)
                new_ids = gen.generate(prompt_input_ids=prompt_ids, max_new_tokens=512)
                results.append(new_ids)
        finally:
            gen.close()
        return results

    with torch.no_grad():
        on_outputs = run_gen(gate_override=None)
        off_outputs = run_gen(gate_override=-10.0)

    n_differ = sum(1 for a, b in zip(on_outputs, off_outputs) if a != b)
    print(f"  {n_differ}/{PROBE_N} outputs differ between cache-on and cache-off")
    for i, (a, b) in enumerate(zip(on_outputs, off_outputs)):
        match = "SAME" if a == b else "DIFFER"
        print(f"  problem {i+1}: {match}  (on_len={len(a)} off_len={len(b)})")

    if n_differ == 0:
        print("  !! ALL outputs identical — cache has ZERO effect on generation")
    elif n_differ == PROBE_N:
        print("  cache affects all outputs — mechanism is active")
    else:
        print("  cache affects some outputs")

    print("\nDONE")


if __name__ == "__main__":
    main()
