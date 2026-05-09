#!/usr/bin/env python3
"""Evaluate the BABILong cache-token scaffold without StateCache.

This reproduces the inline ``baseline_acc`` path from
``train_babilong_ablation.py``: build the same scaffolded inputs with inserted
``[CACHE]`` tokens, then run the frozen model directly without applying any
StateCache delta.
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
    add_cache_token,
    build_suffix_ids,
    build_training_ids,
    get_passage_ids,
    load_babilong_eval,
    parse_csv,
    parse_device_map,
    pred_matches,
    resolve_torch_dtype,
    select_cache_positions,
    set_seed,
    _load_token,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scaffold-only BABILong eval with cache tokens but no StateCache."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--cells-output", default=None)
    parser.add_argument("--output", default=None, help="Accepted for CLI parity; unused.")
    parser.add_argument("--placement", choices=("regex", "loss", "random", "interval"), default="loss")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--regex-cap-k", type=int, default=None)
    parser.add_argument("--cache-layer-idx", type=int, default=-2, help="Accepted for config parity; unused.")
    parser.add_argument("--gate", choices=("on", "off"), default="on", help="Accepted for config parity; unused.")
    parser.add_argument("--causal-mask", choices=("on", "off"), default="on", help="Accepted for config parity; unused.")
    parser.add_argument("--loss-mode", choices=("focused", "full"), default="focused", help="Accepted for config parity; unused.")
    parser.add_argument("--train-tasks", default="qa1,qa2,qa3", help="Accepted for config parity; unused.")
    parser.add_argument("--train-lengths", default="0k,1k,2k", help="Accepted for config parity; unused.")
    parser.add_argument("--eval-tasks", default="qa1,qa2,qa3")
    parser.add_argument("--eval-lengths", default="0k,1k,2k,4k,8k,16k")
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def checkpoint_like_config(args: argparse.Namespace, layer_idx: int | None) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "cache_layer_idx": layer_idx,
        "placement": args.placement,
        "top_k": args.top_k,
        "regex_cap_k": args.regex_cap_k,
        "gate": args.gate,
        "causal_mask": args.causal_mask,
        "loss_mode": args.loss_mode,
        "train_tasks": parse_csv(args.train_tasks),
        "train_lengths": parse_csv(args.train_lengths),
        "eval_tasks": parse_csv(args.eval_tasks),
        "eval_lengths": parse_csv(args.eval_lengths),
        "n_eval": args.n_eval,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "scaffold_only": True,
    }


def main() -> None:
    args = parse_args()
    if args.run_id is None:
        args.run_id = Path(args.metrics_output).stem
    if args.top_k < 0:
        raise SystemExit("--top-k must be non-negative.")
    if args.n_eval <= 0:
        raise SystemExit("--n-eval must be positive.")

    set_seed(args.seed)
    t_start = time.time()
    hf_token = _load_token()

    print(f"Loading {args.model_name} ...", flush=True)
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
    cache_id = add_cache_token(tokenizer, model)
    bos_id = tokenizer.bos_token_id
    model.eval()
    model.requires_grad_(False)

    layer_idx = None
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
        layer_idx = num_layers + args.cache_layer_idx if args.cache_layer_idx < 0 else args.cache_layer_idx
    device = next(model.parameters()).device

    eval_split = load_babilong_eval(
        parse_csv(args.eval_tasks),
        parse_csv(args.eval_lengths),
        args.n_eval,
    )

    results: dict[str, dict[str, Any]] = {}
    cells_output_path = Path(args.cells_output) if args.cells_output else None
    if cells_output_path is not None:
        cells_output_path.parent.mkdir(parents=True, exist_ok=True)
        if cells_output_path.exists():
            cells_output_path.unlink()

    margin = 64
    for (task, length), rows in eval_split.items():
        correct = 0
        examples: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            passage_ids = get_passage_ids(tokenizer, row, args.max_seq_len - margin)
            positions = select_cache_positions(
                placement=args.placement,
                model=model,
                tokenizer=tokenizer,
                row=row,
                passage_ids=passage_ids,
                device=device,
                top_k=args.top_k,
                seed=args.seed + 100_000,
                row_index=row_index,
                regex_cap_k=args.regex_cap_k,
            )
            q_suffix = build_suffix_ids(tokenizer, row["question"], cache_id, "")
            input_ids = build_training_ids(
                passage_ids, positions, cache_id, q_suffix, bos_id
            ).unsqueeze(0).to(device)
            if input_ids.shape[1] > args.max_seq_len:
                input_ids = input_ids[:, -args.max_seq_len:]

            with torch.no_grad():
                out = model(input_ids)
            pred = tokenizer.decode(
                [out.logits[0, -1, :].argmax(dim=-1).item()],
                skip_special_tokens=True,
            )
            ok = pred_matches(tokenizer, pred, str(row["target"]))
            if ok:
                correct += 1

            examples.append(
                {
                    "row_index": row_index,
                    "task": task,
                    "context_length": length,
                    "question": str(row["question"]),
                    "gold": str(row["target"]),
                    "pred": pred.strip(),
                    "correct": ok,
                    "passage_tokens": int(len(passage_ids)),
                    "input_tokens": int(input_ids.shape[1]),
                    "cache_positions": positions,
                    "num_cache_positions": len(positions),
                }
            )

        total = len(rows)
        key = f"{task}/{length}"
        results[key] = {
            "baseline_correct": correct,
            "total": total,
            "baseline_acc": correct / max(1, total),
        }
        print(f"{key}: {correct}/{total} = {correct / max(1, total):.3f}", flush=True)

        if cells_output_path is not None:
            cell = {
                "run_id": args.run_id,
                "placement": args.placement,
                "top_k": args.top_k,
                "cache_layer_idx": layer_idx,
                "task": task,
                "context_length": length,
                "baseline_correct": correct,
                "total": total,
                "baseline_acc": correct / max(1, total),
                "mean_input_tokens": (
                    sum(example["input_tokens"] for example in examples) / max(1, len(examples))
                ),
                "mean_cache_positions": (
                    sum(example["num_cache_positions"] for example in examples) / max(1, len(examples))
                ),
                "examples": examples,
            }
            with cells_output_path.open("a") as cells_file:
                cells_file.write(json.dumps(cell) + "\n")

    metrics_payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": args.run_id,
        "cells_output": args.cells_output,
        "total_seconds": time.time() - t_start,
        "config": checkpoint_like_config(args, layer_idx),
        "eval": results,
    }
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"Saved metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
