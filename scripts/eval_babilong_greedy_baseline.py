#!/usr/bin/env python3
"""Plain no-cache BABILong baseline using the inline ablation matcher.

This evaluates Falcon-Mamba without inserted [CACHE] tokens. Unlike
``bam.eval_babilongv2 --babilong-scoring next_token``, this does not restrict
the prediction to BABILong answer candidates. It greedily decodes the full-vocab
next token and applies ``pred_matches`` from ``train_babilong_ablation.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bam.train_babilong_ablation import (
    DEFAULT_MODEL_NAME,
    _load_token,
    get_passage_ids,
    load_babilong_eval,
    parse_csv,
    parse_device_map,
    pred_matches,
    resolve_torch_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run plain no-cache BABILong eval with greedy full-vocab one-token scoring."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--eval-tasks", default="qa1,qa2,qa3")
    parser.add_argument("--eval-lengths", default="0k,1k,2k,4k,8k,16k")
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--output", default="metrics/zero_shot_baseline/falcon_mamba_greedy_pred_matches_qa1_qa3.json")
    parser.add_argument("--cells-output", default=None)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--answer-colon",
        action="store_true",
        help="Append '\\nAnswer:' after the question. Default mirrors inline ablation question suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_eval <= 0:
        raise SystemExit("--n-eval must be positive.")

    tasks = parse_csv(args.eval_tasks)
    lengths = parse_csv(args.eval_lengths)
    output_path = Path(args.output)
    cells_path = (
        Path(args.cells_output)
        if args.cells_output is not None
        else output_path.with_suffix(".cells.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cells_path.parent.mkdir(parents=True, exist_ok=True)
    if cells_path.exists():
        cells_path.unlink()

    t_start = time.time()
    hf_token = _load_token()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=hf_token,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "token": hf_token,
        "local_files_only": args.local_files_only,
    }
    device_map = parse_device_map(args.device_map)
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    resolved_dtype = resolve_torch_dtype(args.torch_dtype)
    if resolved_dtype != "auto":
        model_kwargs["torch_dtype"] = resolved_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()
    model.requires_grad_(False)
    device = next(model.parameters()).device

    eval_split = load_babilong_eval(tasks, lengths, args.n_eval)
    results: dict[str, dict[str, Any]] = {}
    margin = 64

    for task in tasks:
        for length in lengths:
            rows = eval_split.get((task, length), [])
            correct = 0
            examples: list[dict[str, Any]] = []
            for row_index, row in enumerate(rows):
                passage_ids = get_passage_ids(tokenizer, row, args.max_seq_len - margin)
                suffix_text = f"\n{row['question']}"
                if args.answer_colon:
                    suffix_text += "\nAnswer:"
                suffix_ids = tokenizer(
                    suffix_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids[0]
                input_ids = torch.cat([passage_ids, suffix_ids]).unsqueeze(0).to(device)
                if input_ids.shape[1] > args.max_seq_len:
                    input_ids = input_ids[:, -args.max_seq_len :]

                with torch.no_grad():
                    logits = model(input_ids).logits
                pred = tokenizer.decode(
                    [logits[0, -1, :].argmax(dim=-1).item()],
                    skip_special_tokens=True,
                )
                ok = pred_matches(tokenizer, pred, str(row["target"]))
                correct += int(ok)
                examples.append(
                    {
                        "row_index": row_index,
                        "task": task,
                        "context_length": length,
                        "question": str(row["question"]),
                        "gold": str(row["target"]),
                        "pred": pred.strip(),
                        "correct": ok,
                        "input_tokens": int(input_ids.shape[1]),
                    }
                )

            cell = {
                "task": task,
                "context_length": length,
                "correct": correct,
                "total": len(rows),
                "acc": correct / max(1, len(rows)),
                "scoring": "greedy_full_vocab_one_token_pred_matches",
                "prompt": "plain_question_answer_colon" if args.answer_colon else "plain_question_suffix",
                "examples": examples,
            }
            results[f"{task}/{length}"] = {k: v for k, v in cell.items() if k != "examples"}
            with cells_path.open("a") as cells_file:
                cells_file.write(json.dumps(cell) + "\n")
            print(
                f"{task} {length} {correct}/{len(rows)} "
                f"{correct / max(1, len(rows)):.3f}",
                flush=True,
            )

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": args.model_name,
        "tasks": tasks,
        "context_lengths": lengths,
        "n_eval": args.n_eval,
        "max_seq_len": args.max_seq_len,
        "scoring": "greedy_full_vocab_one_token_pred_matches",
        "prompt": "plain_question_answer_colon" if args.answer_colon else "plain_question_suffix",
        "total_seconds": time.time() - t_start,
        "results": results,
        "cells_output": str(cells_path),
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
