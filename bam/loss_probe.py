#!/usr/bin/env python3
"""Diagnostic: does per-token loss spike on entity-fact sentences vs distractors?

Tests the hypothesis that loss-triggered caching would naturally select
fact sentences, enabling learned cache placement without task-specific rules.

Usage:
    python -m bam.loss_probe
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_NAME = "tiiuae/falcon-mamba-7b-instruct"
FACT_RE = re.compile(
    r'\b\w+\s+(?:travelled|went|journeyed|moved|went\s+back)\s+(?:back\s+)?to\s+(?:the\s+)?\w+\.',
    re.IGNORECASE,
)
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
N_EXAMPLES = 5


def _load_token() -> str | None:
    env_file = ROOT / ".env"
    if env_file.is_file():
        from dotenv import load_dotenv as _ld
        _ld(env_file)
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def sentence_losses_in_context(model, tokenizer, passage: str) -> list[tuple[str, float, bool]]:
    """Return (sentence, avg_loss_in_context, is_fact) for each sentence.

    Runs a single forward pass over the full passage so loss at each token
    reflects what the model has read so far — not isolation perplexity.
    """
    sentences = [s.strip() for s in SENT_SPLIT.split(passage) if s.strip()]
    if not sentences:
        return []

    # Tokenize each sentence separately to get per-sentence token spans.
    # Use the full passage so BOS/context is correct.
    full_ids = tokenizer(passage, return_tensors="pt", add_special_tokens=True).input_ids[0]

    # Find token-level span for each sentence via cumulative re-tokenization.
    spans: list[tuple[int, int]] = []
    prefix = ""
    for sent in sentences:
        prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=True).input_ids[0]
        next_prefix = (prefix + " " + sent).lstrip()
        next_ids = tokenizer(next_prefix, return_tensors="pt", add_special_tokens=True).input_ids[0]
        start = len(prefix_ids)
        end = len(next_ids)
        spans.append((start, end))
        prefix = next_prefix

    # Single forward pass.
    device = next(model.parameters()).device
    input_ids = full_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits  # (1, T, V)

    # Per-token NLL: loss at position t predicts token t+1.
    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)  # (T, V)
    token_loss = -log_probs[torch.arange(len(full_ids) - 1), full_ids[1:].to(device)]  # (T-1,)

    results = []
    for sent, (start, end) in zip(sentences, spans):
        # token_loss[i] = loss predicting token i+1, so sentence tokens start..end-1
        # correspond to loss indices start-1..end-2 (shifted by 1).
        loss_start = max(0, start - 1)
        loss_end = max(0, end - 1)
        if loss_end <= loss_start:
            continue
        avg_loss = token_loss[loss_start:loss_end].mean().item()
        is_fact = bool(FACT_RE.search(sent))
        results.append((sent, avg_loss, is_fact))

    return results


def main() -> None:
    hf_token = _load_token()
    print(f"Loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[CACHE]"]})

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=hf_token,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    fact_losses: list[float] = []
    distractor_losses: list[float] = []

    for length in ["0k", "1k"]:
        print(f"\n--- BABILong {length} ---", flush=True)
        ds = load_dataset("RMT-team/babilong", length, split="qa1", trust_remote_code=True)
        examples = list(ds)[:N_EXAMPLES]

        for i, ex in enumerate(examples):
            q_idx = ex["input"].rfind(ex["question"])
            passage = ex["input"][:q_idx].strip() if q_idx != -1 else ex["input"]

            rows = sentence_losses_in_context(model, tokenizer, passage)
            n_facts = sum(1 for _, _, f in rows if f)
            n_dist = sum(1 for _, _, f in rows if not f)
            print(f"  ex{i}: {len(rows)} sentences ({n_facts} facts, {n_dist} distractors)", flush=True)

            for sent, loss, is_fact in rows:
                label = "FACT" if is_fact else "dist"
                print(f"    [{label}] loss={loss:.3f}  {sent[:60]}", flush=True)
                if is_fact:
                    fact_losses.append(loss)
                else:
                    distractor_losses.append(loss)

    print("\n=== Summary ===")
    if fact_losses:
        print(f"Fact sentences:       n={len(fact_losses):4d}  avg_loss={sum(fact_losses)/len(fact_losses):.4f}")
    if distractor_losses:
        print(f"Distractor sentences: n={len(distractor_losses):4d}  avg_loss={sum(distractor_losses)/len(distractor_losses):.4f}")
    if fact_losses and distractor_losses:
        ratio = (sum(fact_losses)/len(fact_losses)) / (sum(distractor_losses)/len(distractor_losses))
        print(f"Ratio (fact/dist):    {ratio:.3f}  {'(facts harder ✓)' if ratio > 1 else '(distractors harder ✗)'}")


if __name__ == "__main__":
    main()
