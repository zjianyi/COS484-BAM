#!/usr/bin/env python3
"""Baseline BABILong eval for FalconMamba 7B Instruct (no cache module).

Tests exact-match accuracy across tasks and context lengths to find where
the model's recall breaks down — the target gap for StateCache to close.

Usage:
    python -m bam.eval_babilong [--tasks qa1 qa2 qa3] [--lengths 0k 1k 2k 4k 8k]
                                [--n_examples 50] [--no_adapter]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate import MODEL_NAME, _load_token  # noqa: E402

SFT_ADAPTER_DIR = "adapter"
MAX_NEW_TOKENS = 32  # BABILong answers are 1-3 words


SYSTEM_PROMPT = (
    "You are a helpful assistant. Read the passage carefully and answer the "
    "question with a single word or short phrase. Do not explain."
)


def exact_match(pred: str, target: str) -> bool:
    """Check if target word/phrase appears in the prediction (case-insensitive).

    BABILong answers are single words (e.g. "kitchen"); the model often says
    "In the kitchen." so substring containment is the right check.
    """
    pred_norm = pred.strip().lower()
    target_norm = target.strip().lower()
    # Require whole-word match to avoid false positives (e.g. "garden" in "gardening")
    return bool(re.search(r"\b" + re.escape(target_norm) + r"\b", pred_norm))


def run_eval(
    model,
    tokenizer,
    tasks: list[str],
    lengths: list[str],
    n_examples: int,
) -> None:
    device = next(model.parameters()).device
    results: dict[str, dict[str, float]] = {}

    for task in tasks:
        results[task] = {}
        for length in lengths:
            try:
                ds = load_dataset("RMT-team/babilong", length, split=task, trust_remote_code=True)
            except Exception as exc:
                print(f"  [skip {task}/{length}: {exc}]", flush=True)
                continue

            examples = list(ds)[:n_examples]
            correct = 0

            for ex in examples:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex["input"]},
                ]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    out_ids = model.generate(
                        input_ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                new_ids = out_ids[0, input_ids.shape[1]:]
                pred_text = tokenizer.decode(new_ids, skip_special_tokens=True)

                if exact_match(pred_text, ex["target"]):
                    correct += 1

            acc = correct / max(1, len(examples))
            results[task][length] = acc
            print(
                f"  {task}/{length:>4s}  acc={acc:.3f}  ({correct}/{len(examples)})",
                flush=True,
            )

    # Summary table
    print("\n=== BABILong Baseline Results ===")
    header = f"{'task':<6}" + "".join(f"  {l:>5s}" for l in lengths)
    print(header)
    for task in tasks:
        row = f"{task:<6}"
        for length in lengths:
            v = results[task].get(length)
            row += f"  {v:.3f}" if v is not None else "     -"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["qa1", "qa2", "qa3"])
    parser.add_argument("--lengths", nargs="+", default=["0k", "1k", "2k", "4k"])
    parser.add_argument("--n_examples", type=int, default=50)
    parser.add_argument("--no_adapter", action="store_true",
                        help="Skip loading the SFT adapter (pure base model)")
    args = parser.parse_args()

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

    # The adapter was saved after [CACHE] was added to the vocab; resize to match.
    tokenizer.add_special_tokens({"additional_special_tokens": ["[CACHE]"]})
    model.resize_token_embeddings(len(tokenizer))

    if not args.no_adapter:
        print(f"Loading SFT adapter from {SFT_ADAPTER_DIR}", flush=True)
        model = PeftModel.from_pretrained(model, SFT_ADAPTER_DIR)

    model.eval()

    print(
        f"Evaluating on tasks={args.tasks} lengths={args.lengths} "
        f"n_examples={args.n_examples}",
        flush=True,
    )
    run_eval(model, tokenizer, args.tasks, args.lengths, args.n_examples)


if __name__ == "__main__":
    main()
