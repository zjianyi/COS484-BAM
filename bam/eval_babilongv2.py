#!/usr/bin/env python3
"""BABILong eval matching the reasoning-curves recall_eval protocol.

This is a self-contained port of
`reasoning-curves/eval/recall_eval/eval_babilong.py` for the BAM project.
Use `--babilong-scoring generative` for the paper-style BABILong prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BABILONG_CONTEXT_LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k"]
BABILONG_TASKS = [f"qa{index}" for index in range(1, 11)]


@dataclass(frozen=True)
class CandidateScore:
    candidate: str
    token_id: int
    logprob: float


@dataclass(frozen=True)
class GenerationResult:
    text: str
    generated_tokens: int
    hit_max_tokens: bool
    token_ids: list[int]


def load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return

    for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"):
        if env_path.exists():
            load_dotenv(env_path, override=False)


def parse_device_map(value: str) -> str | None:
    if value.lower() == "none":
        return None
    return value


def resolve_torch_dtype(torch_dtype: str | None) -> Any:
    if torch_dtype in {None, "auto"}:
        return "auto"

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = torch_dtype.lower()
    if key not in mapping:
        raise ValueError(
            f"Unsupported torch dtype '{torch_dtype}'. "
            "Choose from auto, float16, bfloat16, or float32."
        )
    return mapping[key]


def load_causal_lm(
    model_name: str,
    *,
    device_map: str | None = "auto",
    torch_dtype: str | None = "auto",
    trust_remote_code: bool = False,
    cache_dir: str | None = None,
    local_files_only: bool = False,
):
    resolved_dtype = resolve_torch_dtype(torch_dtype)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "token": token,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if resolved_dtype != "auto":
        model_kwargs["torch_dtype"] = resolved_dtype
    elif device_map is not None:
        model_kwargs["torch_dtype"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return model, tokenizer


def parse_context_lengths(raw_lengths: str, *, integer: bool = False) -> list[Any]:
    lengths: list[Any] = []
    for item in raw_lengths.split(","):
        value = item.strip()
        if value:
            lengths.append(int(value) if integer else value)
    return lengths


def parse_tasks(raw_tasks: str) -> list[str]:
    tasks: list[str] = []
    for item in raw_tasks.split(","):
        value = item.strip()
        if not value:
            continue
        if not value.startswith("qa"):
            value = f"qa{value}"
        tasks.append(value)
    return tasks


def sample_indices(total: int, *, limit: int | None, seed: int) -> list[int]:
    indices = list(range(total))
    if limit is None or limit >= total:
        return indices
    rng = random.Random(seed)
    rng.shuffle(indices)
    return sorted(indices[:limit])


def infer_model_device(model):
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def move_inputs_to_device(inputs: dict[str, Any], device) -> dict[str, Any]:
    return {key: value.to(device) for key, value in inputs.items()}


def first_answer_token_ids(tokenizer, candidates: Iterable[str]) -> dict[str, int]:
    token_ids: dict[str, int] = {}
    for candidate in candidates:
        candidate_text = str(candidate).strip()
        variants = [f" {candidate_text}", candidate_text]
        best_token_id: int | None = None
        for variant in variants:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                best_token_id = int(encoded[0])
                break
        if best_token_id is None:
            raise ValueError(f"Candidate has no tokens after tokenization: {candidate!r}")
        token_ids[candidate_text] = best_token_id
    return token_ids


def score_next_token(
    logits_at_last_position,
    *,
    gold: str,
    candidate_token_ids: dict[str, int],
) -> tuple[bool, str, list[CandidateScore]]:
    import torch.nn.functional as F

    log_probs = F.log_softmax(logits_at_last_position.float(), dim=-1)
    scores = [
        CandidateScore(
            candidate=candidate,
            token_id=token_id,
            logprob=float(log_probs[token_id].item()),
        )
        for candidate, token_id in candidate_token_ids.items()
    ]
    scores.sort(key=lambda score: score.logprob, reverse=True)
    pred = scores[0].candidate if scores else ""
    return pred == str(gold).strip(), pred, scores


def normalize_babilong_answer(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"(?i)^answer\s*:\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" \t\n\r.:;!?\"'`()[]{}")
    return cleaned


def extract_babilong_answer(text: str, candidates: Iterable[str]) -> str:
    answer_text = text.strip()
    if "answer:" in answer_text.lower():
        answer_text = re.split(r"(?i)answer\s*:", answer_text)[-1].strip()
    answer_text = answer_text.splitlines()[0].strip() if answer_text else ""
    normalized_text = normalize_babilong_answer(answer_text)

    normalized_candidates = {
        normalize_babilong_answer(candidate): str(candidate).strip()
        for candidate in candidates
    }
    if normalized_text in normalized_candidates:
        return normalized_candidates[normalized_text]

    for normalized_candidate, candidate in normalized_candidates.items():
        if re.search(rf"\b{re.escape(normalized_candidate)}\b", normalized_text):
            return candidate
    return normalized_text


def score_generated_babilong_answer(
    text: str,
    *,
    gold: str,
    candidates: Iterable[str],
) -> tuple[bool, str]:
    pred = extract_babilong_answer(text, candidates)
    return normalize_babilong_answer(pred) == normalize_babilong_answer(gold), pred


def logits_for_prompt_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_length: int | None = None,
):
    kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "add_special_tokens": True,
        "padding": True,
    }
    if max_length is not None:
        kwargs["truncation"] = True
        kwargs["max_length"] = max_length
    inputs = tokenizer(prompts, **kwargs)
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] == 0:
        raise ValueError("Prompt tokenized to an empty sequence.")

    device = infer_model_device(model)
    model_inputs = move_inputs_to_device(inputs, device)
    with torch.no_grad():
        outputs = model(**model_inputs)

    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        last_indices = torch.full(
            (input_ids.shape[0],),
            outputs.logits.shape[1] - 1,
            dtype=torch.long,
            device=outputs.logits.device,
        )
        prompt_tokens = [int(input_ids.shape[1])] * int(input_ids.shape[0])
    else:
        prompt_token_tensor = attention_mask.sum(dim=1)
        last_indices = prompt_token_tensor - 1
        prompt_tokens = [int(value.item()) for value in prompt_token_tensor]
    row_indices = torch.arange(input_ids.shape[0], device=outputs.logits.device)
    return outputs.logits[row_indices, last_indices], prompt_tokens


def build_babilong_prompt(passage: str, question: str) -> str:
    return f"{passage.rstrip()}\n\n{question.strip()}\nAnswer:"


def build_babilong_paper_prompt(
    passage: str,
    question: str,
    examples: Iterable[dict[str, object]],
) -> str:
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
        f"{passage.rstrip()}\n"
        "</context>\n"
        f"QUESTION: {question.strip()}\n"
        "Answer:"
    )


def _tokenize_prompts(
    tokenizer,
    prompts: list[str],
    max_tokens: int,
    max_context_window: int | None,
    *,
    add_special_tokens: bool,
):
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "add_special_tokens": add_special_tokens,
    }
    if max_context_window is not None:
        tokenizer_kwargs["truncation"] = True
        tokenizer_kwargs["max_length"] = max(max_context_window - max_tokens, 1)
    return tokenizer(prompts, **tokenizer_kwargs)


def _trim_completion_tokens(
    token_ids: list[int],
    *,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> tuple[list[int], bool]:
    if eos_token_id is not None:
        for index, token_id in enumerate(token_ids):
            if token_id == eos_token_id:
                return token_ids[: index + 1], True

    if pad_token_id is not None and pad_token_id != eos_token_id:
        for index, token_id in enumerate(token_ids):
            if token_id == pad_token_id:
                return token_ids[:index], False

    return token_ids, False


def _build_generation_result(
    tokenizer,
    token_ids: list[int],
    *,
    max_tokens: int,
) -> GenerationResult:
    trimmed_token_ids, ended_with_eos = _trim_completion_tokens(
        token_ids,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return GenerationResult(
        text=tokenizer.decode(trimmed_token_ids, skip_special_tokens=True).strip(),
        generated_tokens=len(trimmed_token_ids),
        hit_max_tokens=(len(trimmed_token_ids) >= max_tokens and not ended_with_eos),
        token_ids=trimmed_token_ids,
    )


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_tokens: int,
    max_context_window: int | None = None,
) -> list[GenerationResult]:
    if not prompts:
        return []
    if max_tokens <= 0:
        return [
            GenerationResult(text="", generated_tokens=0, hit_max_tokens=False, token_ids=[])
            for _ in prompts
        ]

    inputs = _tokenize_prompts(
        tokenizer,
        prompts,
        max_tokens,
        max_context_window,
        add_special_tokens=True,
    )
    device = infer_model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "temperature": 0.0,
        "do_sample": False,
        "return_dict_in_generate": True,
    }
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        generated = model.generate(**generate_kwargs)

    prompt_width = inputs["input_ids"].shape[-1]
    completion_tokens = generated.sequences[:, prompt_width:]
    return [
        _build_generation_result(tokenizer, tokens.tolist(), max_tokens=max_tokens)
        for tokens in completion_tokens
    ]


def summarize(examples: list[dict[str, object]]) -> dict[str, object]:
    total = len(examples)
    correct = sum(int(example["correct"]) for example in examples)
    return {
        "acc": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "mean_prompt_tokens": (
            sum(int(example["prompt_tokens"]) for example in examples) / total if total else 0.0
        ),
        "examples": examples,
    }


def load_babilong_split(dataset_name: str, context_length: str, task: str):
    from datasets import load_dataset

    return load_dataset(dataset_name, context_length, split=task)


def target_candidates(dataset) -> list[str]:
    values = sorted({str(value).strip() for value in dataset["target"]})
    if not values:
        raise ValueError("BABILong split has no target values.")
    return values


def few_shot_examples(dataset_name: str, task: str, *, n_examples: int) -> list[dict[str, object]]:
    if n_examples <= 0:
        return []
    dataset = load_babilong_split(dataset_name, "0k", task)
    return [dict(dataset[index]) for index in range(min(n_examples, len(dataset)))]


def evaluate_architecture(
    *,
    architecture: str,
    model_name: str,
    dataset_name: str,
    context_lengths: list[str],
    tasks: list[str],
    limit: int | None,
    seed: int,
    device_map: str | None,
    torch_dtype: str,
    trust_remote_code: bool,
    cache_dir: str | None,
    local_files_only: bool,
    max_length: int | None,
    batch_size: int,
    babilong_scoring: str,
    few_shot_count: int,
    hf_max_new_tokens: int,
    cells_output_path: Path | None,
) -> list[dict[str, object]]:
    print(f"[load] {architecture}: {model_name}", flush=True)
    model, tokenizer = load_causal_lm(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    rows: list[dict[str, object]] = []
    for context_length in context_lengths:
        for task in tasks:
            dataset = load_babilong_split(dataset_name, context_length, task)
            candidates = target_candidates(dataset)
            candidate_token_ids = first_answer_token_ids(tokenizer, candidates)
            examples_for_prompt = few_shot_examples(dataset_name, task, n_examples=few_shot_count)
            examples: list[dict[str, object]] = []
            selected_indices = sample_indices(len(dataset), limit=limit, seed=seed)
            prompt_payloads: list[tuple[int, int, object, str]] = []
            for eval_index, row_index in enumerate(selected_indices):
                sample = dataset[int(row_index)]
                if babilong_scoring == "generative":
                    prompt = build_babilong_paper_prompt(
                        str(sample["input"]),
                        str(sample["question"]),
                        examples_for_prompt,
                    )
                elif few_shot_count > 0:
                    prompt = build_babilong_paper_prompt(
                        str(sample["input"]),
                        str(sample["question"]),
                        examples_for_prompt,
                    )
                else:
                    prompt = build_babilong_prompt(str(sample["input"]), str(sample["question"]))
                prompt_payloads.append((eval_index, int(row_index), sample, prompt))

            for start in range(0, len(prompt_payloads), batch_size):
                batch = prompt_payloads[start : start + batch_size]
                if babilong_scoring == "generative":
                    prompts = [prompt for _eval_index, _row_index, _sample, prompt in batch]
                    generations = generate_batch(
                        model,
                        tokenizer,
                        prompts,
                        max_tokens=hf_max_new_tokens,
                        max_context_window=max_length,
                    )
                    for (eval_index, row_index, sample, prompt), generation in zip(
                        batch, generations, strict=True
                    ):
                        gold = str(sample["target"]).strip()
                        correct, pred = score_generated_babilong_answer(
                            generation.text,
                            gold=gold,
                            candidates=candidates,
                        )
                        prompt_tokens = len(tokenizer(prompt, add_special_tokens=True).input_ids)
                        examples.append(
                            {
                                "eval_index": eval_index,
                                "row_index": row_index,
                                "context_length": context_length,
                                "task": task,
                                "question": str(sample["question"]),
                                "gold": gold,
                                "pred": pred,
                                "correct": correct,
                                "prompt_tokens": prompt_tokens,
                                "generated_token_ids": generation.token_ids,
                                "generated_text": generation.text,
                                "babilong_scoring": babilong_scoring,
                            }
                        )
                else:
                    batch_logits, batch_prompt_tokens = logits_for_prompt_batch(
                        model,
                        tokenizer,
                        [prompt for _eval_index, _row_index, _sample, prompt in batch],
                        max_length=max_length,
                    )
                    for (eval_index, row_index, sample, _prompt), logits, prompt_tokens in zip(
                        batch, batch_logits, batch_prompt_tokens, strict=True
                    ):
                        gold = str(sample["target"]).strip()
                        correct, pred, scores = score_next_token(
                            logits,
                            gold=gold,
                            candidate_token_ids=candidate_token_ids,
                        )
                        examples.append(
                            {
                                "eval_index": eval_index,
                                "row_index": row_index,
                                "context_length": context_length,
                                "task": task,
                                "question": str(sample["question"]),
                                "gold": gold,
                                "pred": pred,
                                "correct": correct,
                                "prompt_tokens": prompt_tokens,
                                "top_logprobs": [
                                    {
                                        "candidate": score.candidate,
                                        "token_id": score.token_id,
                                        "logprob": score.logprob,
                                    }
                                    for score in scores[:5]
                                ],
                                "babilong_scoring": babilong_scoring,
                            }
                        )
            summary = summarize(examples)
            summary["context_length"] = context_length
            summary["task"] = task
            rows.append(summary)
            if cells_output_path is not None:
                cells_output_path.parent.mkdir(parents=True, exist_ok=True)
                with cells_output_path.open("a") as cells_file:
                    cells_file.write(
                        json.dumps(
                            {
                                "architecture": architecture,
                                "model_name": model_name,
                                "context_length": context_length,
                                "task": task,
                                **summary,
                            }
                        )
                        + "\n"
                    )
            print(
                f"[done] {architecture} {context_length} {task} "
                f"acc={summary['acc']:.4f} ({summary['correct']}/{summary['total']})",
                flush=True,
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BABILong qa1-qa10 recall with reasoning-curves scoring."
    )
    parser.add_argument("--transformer-model", type=str)
    parser.add_argument("--mamba-model", type=str)
    parser.add_argument("--hybrid-model", type=str)
    parser.add_argument("--dataset-name", type=str, default="RMT-team/babilong")
    parser.add_argument(
        "--context-lengths",
        type=str,
        default=",".join(BABILONG_CONTEXT_LENGTHS),
        help="Comma-separated BABILong configs.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(BABILONG_TASKS),
        help="Comma-separated tasks, e.g. qa1,qa2,...,qa10.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Examples per task/length.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="runs/recall_eval/babilong_recall_v2.json")
    parser.add_argument(
        "--cells-output",
        type=str,
        default=None,
        help="Append completed context/task cells to this JSONL file. Defaults to OUTPUT.cells.jsonl.",
    )
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--transformer-trust-remote-code", action="store_true")
    parser.add_argument("--no-mamba-trust-remote-code", action="store_true")
    parser.add_argument("--no-hybrid-trust-remote-code", action="store_true")
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional tokenizer truncation length. Leave unset for full BABILong input.",
    )
    parser.add_argument("--hf-max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--babilong-scoring",
        choices=("next_token", "generative"),
        default="next_token",
        help="Use next-token candidate scoring or paper-style generated-answer scoring.",
    )
    parser.add_argument(
        "--few-shot-examples",
        type=int,
        default=0,
        help="Few-shot 0k examples per task for paper-style BABILong prompts (same shape as generative). "
        "When >0, uses build_babilong_paper_prompt for both next_token and generative. Default 0 for plain prompt.",
    )
    return parser.parse_args()


def main() -> None:
    load_project_env()
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive when provided.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    configs = {
        "transformer": {
            "model": args.transformer_model,
            "trust_remote_code": args.transformer_trust_remote_code,
        },
        "mamba": {
            "model": args.mamba_model,
            "trust_remote_code": not args.no_mamba_trust_remote_code,
        },
        "hybrid": {
            "model": args.hybrid_model,
            "trust_remote_code": not args.no_hybrid_trust_remote_code,
        },
    }
    active_configs = {
        name: config for name, config in configs.items() if config["model"]
    }
    if not active_configs:
        raise SystemExit("Provide at least one of --transformer-model, --mamba-model, or --hybrid-model.")

    context_lengths = parse_context_lengths(args.context_lengths)
    tasks = parse_tasks(args.tasks)
    output_path = Path(args.output)
    cells_output_path = (
        Path(args.cells_output)
        if args.cells_output is not None
        else output_path.with_suffix(".cells.jsonl")
    )
    if cells_output_path.exists():
        cells_output_path.unlink()

    results: dict[str, list[dict[str, object]]] = {}
    models: dict[str, dict[str, object]] = {}
    for architecture, config in active_configs.items():
        results[architecture] = evaluate_architecture(
            architecture=architecture,
            model_name=str(config["model"]),
            dataset_name=args.dataset_name,
            context_lengths=context_lengths,
            tasks=tasks,
            limit=args.limit,
            seed=args.seed,
            device_map=parse_device_map(args.device_map),
            torch_dtype=args.torch_dtype,
            trust_remote_code=bool(config["trust_remote_code"]),
            cache_dir=args.hf_cache_dir,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            batch_size=args.batch_size,
            babilong_scoring=args.babilong_scoring,
            few_shot_count=args.few_shot_examples,
            hf_max_new_tokens=args.hf_max_new_tokens,
            cells_output_path=cells_output_path,
        )
        models[architecture] = {
            "backend": "hf",
            "model_name": config["model"],
            "trust_remote_code": bool(config["trust_remote_code"]),
        }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "babilong",
        "dataset_name": args.dataset_name,
        "scoring": "hf_next_token_logprob_or_generated_answer",
        "babilong_scoring": args.babilong_scoring,
        "few_shot_examples": args.few_shot_examples,
        "hf_max_new_tokens": args.hf_max_new_tokens,
        "context_lengths": context_lengths,
        "tasks": tasks,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "models": models,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"[saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
