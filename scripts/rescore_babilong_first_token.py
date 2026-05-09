#!/usr/bin/env python3
"""
Rescore or summarize BABILong runs using a strict "first token vs gold" notion.

1) eval_babilongv2 JSON (babilong_scoring=next_token):
   The saved `pred` is already the highest-probability *first token* among candidate
   answer strings (see score_next_token in eval_babilongv2). We report strict and
   normalized string match rates and verify pred == top_logprobs[0].candidate.

2) Reasoning-curves merged JSON (generative generations with generated_token_ids):
   For each example, the model's first generated token id is compared to the token id(s)
   obtained by encoding the gold string the same way candidate_token_ids does (leading
   space variant first). Requires `transformers` for AutoTokenizer.

Run (no PyTorch needed for mode 1):
  python scripts/rescore_babilong_first_token.py --input metrics/zero_shot_baseline/falcon_mamba_next_token_qa1_qa3.json

Run mode 2 on cluster / Linux pixi env where transformers is installed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


def normalize_babilong_answer(text: str) -> str:
    """Mirror bam.eval_babilongv2.normalize_babilong_answer (stdlib only)."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"(?i)^answer\s*:\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" \t\n\r.:;!?\"'`()[]{}")
    return cleaned


def _candidate_first_token_ids(tokenizer: Any, candidate: str) -> set[int]:
    candidate_text = str(candidate).strip()
    ids: set[int] = set()
    for variant in (f" {candidate_text}", candidate_text):
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if encoded:
            ids.add(int(encoded[0]))
    return ids


def _iter_cells(payload: dict[str, Any], architecture: str) -> Iterable[dict[str, Any]]:
    results = payload.get("results") or {}
    arch = results.get(architecture)
    if arch is None:
        raise KeyError(f"No results for architecture {architecture!r}; have {list(results)!r}")
    if not isinstance(arch, list):
        raise TypeError(f"results[{architecture!r}] must be a list")
    yield from arch


def _task_filter(tasks: frozenset[str], task: str) -> bool:
    return task in tasks


def rescore_eval_v2_next_token(
    payload: dict[str, Any],
    *,
    architecture: str,
    tasks: frozenset[str],
) -> dict[str, Any]:
    mismatches_top = 0
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for row in _iter_cells(payload, architecture):
        context_length = str(row.get("context_length", ""))
        task = str(row.get("task", ""))
        if not _task_filter(tasks, task):
            continue
        key = (context_length, task)
        bucket = counts.setdefault(
            key,
            {"n": 0, "strict_ok": 0, "norm_ok": 0, "stored_ok": 0},
        )
        for ex in row.get("examples") or []:
            gold = str(ex.get("gold", "")).strip()
            pred = str(ex.get("pred", "")).strip()
            tops = ex.get("top_logprobs") or []
            if tops and tops[0].get("candidate") != pred:
                mismatches_top += 1
            strict = pred == gold
            norm = normalize_babilong_answer(pred) == normalize_babilong_answer(gold)
            stored = bool(ex.get("correct"))
            bucket["n"] += 1
            bucket["strict_ok"] += int(strict)
            bucket["norm_ok"] += int(norm)
            bucket["stored_ok"] += int(stored)
    return {
        "kind": "eval_babilongv2_next_token",
        "architecture": architecture,
        "tasks": sorted(tasks),
        "pred_vs_top_logprob_mismatch_count": mismatches_top,
        "per_cell": {
            f"{task}/{cl}": {
                "n": v["n"],
                "acc_strict_pred_vs_gold": v["strict_ok"] / v["n"] if v["n"] else 0.0,
                "acc_normalized_pred_vs_gold": v["norm_ok"] / v["n"] if v["n"] else 0.0,
                "acc_stored_correct_flag": v["stored_ok"] / v["n"] if v["n"] else 0.0,
            }
            for (cl, task), v in sorted(counts.items(), key=lambda x: (x[0][1], x[0][0]))
        },
    }


def rescore_merged_generative_first_token_id(
    payload: dict[str, Any],
    *,
    architecture: str,
    tasks: frozenset[str],
    model_name: str | None,
    tokenizer: Any,
) -> dict[str, Any]:
    models = payload.get("models") or {}
    resolved_model = model_name
    if resolved_model is None:
        entry = models.get(architecture)
        if isinstance(entry, str):
            resolved_model = entry
        elif isinstance(entry, dict):
            resolved_model = str(entry.get("model_name", "")) or None
    if not resolved_model:
        raise ValueError(
            "Could not resolve model_name; pass --model-name or ensure payload['models'] entry."
        )

    counts: dict[tuple[str, str], dict[str, int]] = {}
    missing_ids = 0

    for row in _iter_cells(payload, architecture):
        context_length = str(row.get("context_length", ""))
        task = str(row.get("task", ""))
        if not _task_filter(tasks, task):
            continue
        key = (context_length, task)
        bucket = counts.setdefault(key, {"n": 0, "first_token_ok": 0})
        for ex in row.get("examples") or []:
            gold = str(ex.get("gold", "")).strip()
            ids = ex.get("generated_token_ids")
            if not ids:
                missing_ids += 1
                bucket["n"] += 1
                continue
            gen_first = int(ids[0])
            gold_ids = _candidate_first_token_ids(tokenizer, gold)
            ok = gen_first in gold_ids
            bucket["n"] += 1
            bucket["first_token_ok"] += int(ok)

    return {
        "kind": "merged_generative_first_generated_token_id_vs_gold_encoding",
        "architecture": architecture,
        "model_name": resolved_model,
        "tasks": sorted(tasks),
        "examples_missing_generated_token_ids": missing_ids,
        "per_cell": {
            f"{task}/{cl}": {
                "n": v["n"],
                "acc_first_gen_token_matches_gold_candidate_token": (
                    v["first_token_ok"] / v["n"] if v["n"] else 0.0
                ),
            }
            for (cl, task), v in sorted(counts.items(), key=lambda x: (x[0][1], x[0][0]))
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to eval JSON (eval_babilongv2 save or reasoning-curves merged JSON).",
    )
    p.add_argument("--architecture", type=str, default="mamba")
    p.add_argument(
        "--tasks",
        type=str,
        default="qa1,qa2,qa3",
        help="Comma-separated task names to include.",
    )
    p.add_argument(
        "--kind",
        choices=("auto", "eval_v2_next_token", "merged_generative"),
        default="auto",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HF model id for tokenizer (merged_generative); default from payload['models'].",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the summary dict as JSON.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tasks = frozenset(t.strip() for t in args.tasks.split(",") if t.strip())
    payload = json.loads(args.input.read_text())

    kind = args.kind
    if kind == "auto":
        scoring = str(payload.get("scoring", ""))
        bscoring = str(payload.get("babilong_scoring", ""))
        if scoring == "hf_next_token_logprob_or_generated_answer" and bscoring == "next_token":
            kind = "eval_v2_next_token"
        else:
            kind = "merged_generative"

    if kind == "eval_v2_next_token":
        summary = rescore_eval_v2_next_token(
            payload, architecture=args.architecture, tasks=tasks
        )
    elif kind == "merged_generative":
        try:
            from transformers import AutoTokenizer
        except ImportError as err:
            print(
                "merged_generative requires `transformers` (e.g. Linux pixi env on this repo).",
                file=sys.stderr,
            )
            raise SystemExit(1) from err
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name or "tiiuae/falcon-mamba-7b-instruct",
            trust_remote_code=True,
        )
        summary = rescore_merged_generative_first_token_id(
            payload,
            architecture=args.architecture,
            tasks=tasks,
            model_name=args.model_name,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(kind)

    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))
        print(f"[saved] {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
