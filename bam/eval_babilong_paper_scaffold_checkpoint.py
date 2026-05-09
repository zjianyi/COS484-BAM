#!/usr/bin/env python3
"""Evaluate StateCache checkpoints with a paper-style BABILong prompt scaffold.

This keeps the reasoning-curves-style BABILong prompt shape, inserts cache write
tokens inside the context, appends a final cache read token after ``Answer:``,
and scores the first answer token using candidate next-token log probabilities.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bam.cache import StateCache
from bam.eval_babilongv2 import (
    first_answer_token_ids,
    few_shot_examples,
    load_babilong_split,
    sample_indices,
    score_next_token,
    target_candidates,
)
from bam.train_babilong_ablation import (
    DEFAULT_MODEL_NAME,
    add_cache_token,
    build_training_ids,
    cache_delta,
    get_passage_ids,
    parse_csv,
    parse_device_map,
    resolve_torch_dtype,
    select_cache_positions,
    set_seed,
    _get_backbone,
    _load_token,
    _run_layer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved StateCache checkpoint with a paper-style BABILong scaffold."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-name", default="RMT-team/babilong")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--cells-output", default=None)
    parser.add_argument("--eval-tasks", default="qa1,qa2,qa3")
    parser.add_argument("--eval-lengths", default="0k,1k,2k,4k,8k,16k")
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--few-shot-examples", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--placement", choices=("regex", "loss", "random", "interval"), default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--regex-cap-k", type=int, default=None)
    parser.add_argument("--cache-layer-idx", type=int, default=None)
    parser.add_argument("--gate", choices=("on", "off"), default=None)
    parser.add_argument("--route-gate", choices=("off", "scalar", "vector"), default=None)
    parser.add_argument("--causal-mask", choices=("on", "off"), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def cfg_value(args: argparse.Namespace, cfg: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name.replace("-", "_"), None)
    return cfg.get(name, default) if value is None else value


def paper_prompt_prefix(examples: list[dict[str, object]]) -> str:
    example_blocks = []
    for example in examples:
        example_blocks.append(
            "<example>\n"
            f"{str(example['input']).strip()}\n"
            f"QUESTION: {str(example['question']).strip()}\n"
            f"Answer: {str(example['target']).strip()}\n"
            "</example>"
        )
    examples_text = "\n\n".join(example_blocks)
    return (
        "I will give you context with facts hidden in random text and a question.\n"
        "Answer the question based only on the information from the facts.\n"
        "Respond with the answer only, in the format: Answer: <answer>\n\n"
        f"{examples_text}\n\n"
        "<context>\n"
    )


def paper_prompt_suffix(question: str) -> str:
    return f"\n</context>\nQUESTION: {question.strip()}\nAnswer:"


def run_cache_logits(
    *,
    model,
    cache: StateCache,
    input_ids: torch.Tensor,
    cache_id: int,
    layer_idx: int,
    causal_mask: bool,
    use_gate: bool,
) -> torch.Tensor:
    cache.eval()
    backbone = _get_backbone(model)
    layers = backbone.layers
    embed = model.get_input_embeddings()

    with torch.no_grad():
        h = embed(input_ids)
        for i in range(layer_idx + 1):
            h = _run_layer(layers[i], h)

        c_pos = (input_ids[0] == cache_id).nonzero(as_tuple=True)[0]
        if c_pos.numel() > 0:
            delta, any_avail = cache_delta(
                h,
                cache,
                c_pos,
                causal_mask=causal_mask,
                use_gate=use_gate,
            )
            if any_avail.any():
                h[0, c_pos[any_avail]] = h[0, c_pos[any_avail]] + delta[any_avail]

        model_dtype = next(model.parameters()).dtype
        h = h.to(model_dtype)
        for i in range(layer_idx + 1, len(layers)):
            h = _run_layer(layers[i], h)
        h = backbone.norm_f(h.to(model_dtype))
        logits = model.get_output_embeddings()(h)
    return logits[0, -1, :]


def build_paper_scaffold_ids(
    *,
    tokenizer,
    row: dict[str, Any],
    examples: list[dict[str, object]],
    cache_id: int,
    bos_id: int | None,
    model,
    device,
    args: argparse.Namespace,
    row_index: int,
) -> tuple[torch.Tensor, list[int], int]:
    prefix_ids = tokenizer(
        paper_prompt_prefix(examples),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]
    suffix_ids = tokenizer(
        paper_prompt_suffix(str(row["question"])),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].tolist() + [cache_id]

    bos_len = 1 if bos_id is not None else 0
    margin = 8
    max_passage_tokens = args.max_seq_len - len(prefix_ids) - len(suffix_ids) - bos_len - margin
    if max_passage_tokens <= 0:
        raise ValueError(
            "Prompt prefix/suffix exceed max sequence length; increase --max-seq-len "
            "or reduce --few-shot-examples."
        )

    passage_ids = get_passage_ids(tokenizer, row, max_passage_tokens)
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

    scaffolded_passage = build_training_ids(
        passage_ids,
        positions,
        cache_id,
        [],
        None,
    ).tolist()
    ids: list[int] = []
    if bos_id is not None:
        ids.append(bos_id)
    ids.extend(prefix_ids.tolist())
    ids.extend(scaffolded_passage)
    ids.extend(suffix_ids)
    if len(ids) > args.max_seq_len:
        ids = ids[-args.max_seq_len :]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device), positions, len(passage_ids)


def main() -> None:
    args = parse_args()
    if args.n_eval <= 0:
        raise SystemExit("--n-eval must be positive.")
    if args.few_shot_examples < 0:
        raise SystemExit("--few-shot-examples must be non-negative.")

    t_start = time.time()
    set_seed(args.seed)
    hf_token = _load_token()

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    if args.run_id is None:
        args.run_id = checkpoint_path.stem

    args.placement = cfg_value(args, cfg, "placement", "loss")
    args.top_k = int(cfg_value(args, cfg, "top_k", 4))
    args.regex_cap_k = cfg_value(args, cfg, "regex_cap_k", None)
    args.cache_layer_idx = int(cfg_value(args, cfg, "cache_layer_idx", -2))
    args.gate = cfg_value(args, cfg, "gate", "on")
    args.route_gate = cfg_value(args, cfg, "route_gate", "off")
    args.causal_mask = cfg_value(args, cfg, "causal_mask", "on")
    args.max_seq_len = int(cfg_value(args, cfg, "max_seq_len", 16384)) if args.max_seq_len is None else args.max_seq_len

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

    device = next(model.parameters()).device
    cache = StateCache(
        d_model=int(cfg["d_model"]),
        d_attn=int(cfg["d_attn"]),
        max_entries=int(cfg["max_entries"]),
        route_gate=args.route_gate,
    ).to(device=device)
    cache.load_state_dict(ckpt["state_dict"])
    cache.eval()

    cells_output_path = Path(args.cells_output) if args.cells_output else None
    if cells_output_path is not None:
        cells_output_path.parent.mkdir(parents=True, exist_ok=True)
        if cells_output_path.exists():
            cells_output_path.unlink()

    results: dict[str, dict[str, Any]] = {}
    for length in parse_csv(args.eval_lengths):
        for task in parse_csv(args.eval_tasks):
            dataset = load_babilong_split(args.dataset_name, length, task)
            candidates = target_candidates(dataset)
            candidate_token_ids = first_answer_token_ids(tokenizer, candidates)
            examples_for_prompt = few_shot_examples(
                args.dataset_name,
                task,
                n_examples=args.few_shot_examples,
            )
            selected_indices = sample_indices(len(dataset), limit=args.n_eval, seed=args.seed)

            cache_correct = 0
            baseline_correct = 0
            examples_out: list[dict[str, Any]] = []
            for eval_index, row_index in enumerate(selected_indices):
                row = dict(dataset[int(row_index)])
                input_ids, positions, passage_tokens = build_paper_scaffold_ids(
                    tokenizer=tokenizer,
                    row=row,
                    examples=examples_for_prompt,
                    cache_id=cache_id,
                    bos_id=bos_id,
                    model=model,
                    device=device,
                    args=args,
                    row_index=eval_index,
                )

                cache_logits = run_cache_logits(
                    model=model,
                    cache=cache,
                    input_ids=input_ids,
                    cache_id=cache_id,
                    layer_idx=int(args.cache_layer_idx),
                    causal_mask=args.causal_mask == "on",
                    use_gate=args.gate == "on",
                )
                cache_ok, cache_pred, cache_scores = score_next_token(
                    cache_logits,
                    gold=str(row["target"]),
                    candidate_token_ids=candidate_token_ids,
                )
                if cache_ok:
                    cache_correct += 1

                with torch.no_grad():
                    baseline_logits = model(input_ids).logits[0, -1, :]
                baseline_ok, baseline_pred, baseline_scores = score_next_token(
                    baseline_logits,
                    gold=str(row["target"]),
                    candidate_token_ids=candidate_token_ids,
                )
                if baseline_ok:
                    baseline_correct += 1

                examples_out.append(
                    {
                        "eval_index": eval_index,
                        "row_index": int(row_index),
                        "context_length": length,
                        "task": task,
                        "question": str(row["question"]),
                        "gold": str(row["target"]).strip(),
                        "cache_pred": cache_pred,
                        "cache_correct": cache_ok,
                        "baseline_pred": baseline_pred,
                        "baseline_correct": baseline_ok,
                        "cache_top_scores": [score.__dict__ for score in cache_scores[:5]],
                        "baseline_top_scores": [score.__dict__ for score in baseline_scores[:5]],
                        "passage_tokens": int(passage_tokens),
                        "input_tokens": int(input_ids.shape[1]),
                        "cache_positions": positions,
                        "num_cache_positions": len(positions),
                    }
                )

            total = len(selected_indices)
            key = f"{task}/{length}"
            results[key] = {
                "cache_correct": cache_correct,
                "baseline_correct": baseline_correct,
                "total": total,
                "cache_acc": cache_correct / max(1, total),
                "baseline_acc": baseline_correct / max(1, total),
            }
            print(
                f"{key}: cache={cache_correct}/{total} "
                f"baseline={baseline_correct}/{total}",
                flush=True,
            )

            if cells_output_path is not None:
                cell = {
                    "run_id": args.run_id,
                    "checkpoint": str(checkpoint_path),
                    "prompt_protocol": "paper_scaffold_next_token",
                    "placement": args.placement,
                    "top_k": args.top_k,
                    "cache_layer_idx": int(args.cache_layer_idx),
                    "task": task,
                    "context_length": length,
                    "cache_correct": cache_correct,
                    "baseline_correct": baseline_correct,
                    "total": total,
                    "cache_acc": cache_correct / max(1, total),
                    "baseline_acc": baseline_correct / max(1, total),
                    "examples": examples_out,
                }
                with cells_output_path.open("a") as cells_file:
                    cells_file.write(json.dumps(cell) + "\n")

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint_path),
        "prompt_protocol": "paper_scaffold_next_token",
        "total_seconds": time.time() - t_start,
        "config": {
            **cfg,
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "eval_tasks": parse_csv(args.eval_tasks),
            "eval_lengths": parse_csv(args.eval_lengths),
            "n_eval": args.n_eval,
            "few_shot_examples": args.few_shot_examples,
            "placement": args.placement,
            "top_k": args.top_k,
            "regex_cap_k": args.regex_cap_k,
            "gate": args.gate,
            "causal_mask": args.causal_mask,
            "max_seq_len": args.max_seq_len,
        },
        "eval": results,
    }
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
