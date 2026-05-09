#!/usr/bin/env python3
"""Parameterized BABILong StateCache ablation runner.

This script consolidates the BABILong cache ablations so runs can be launched
with CLI flags instead of editing constants in copied training scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bam.cache import StateCache  # noqa: E402

CACHE_TOKEN = "[CACHE]"
DEFAULT_MODEL_NAME = "tiiuae/falcon-mamba-7b-instruct"
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
FACT_RE = re.compile(
    r"\b\w+\s+(?:travelled|went|journeyed|moved|went\s+back)\s+"
    r"(?:back\s+)?to\s+(?:the\s+)?\w+\.",
    re.IGNORECASE,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_token() -> str | None:
    env_file = ROOT / ".env"
    if env_file.is_file():
        try:
            from dotenv import load_dotenv
        except ModuleNotFoundError:
            pass
        else:
            load_dotenv(env_file)
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_device_map(value: str) -> str | None:
    return None if value.lower() == "none" else value


def resolve_torch_dtype(value: str) -> Any:
    if value == "auto":
        return "auto"
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[value]


def add_cache_token(tokenizer, model) -> int:
    if CACHE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [CACHE_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer.convert_tokens_to_ids(CACHE_TOKEN)


def load_babilong_rows(tasks: list[str], lengths: list[str], n_per_combo: int) -> list[dict]:
    rows: list[dict] = []
    for task in tasks:
        for length in lengths:
            try:
                ds = load_dataset("RMT-team/babilong", length, split=task)
            except Exception as exc:
                print(f"  [skip {task}/{length}: {exc}]", flush=True)
                continue
            for ex in list(ds)[:n_per_combo]:
                rows.append(
                    {
                        "input": ex["input"],
                        "question": ex["question"],
                        "target": ex["target"],
                        "task": task,
                        "length": length,
                    }
                )
    print(f"Loaded {len(rows)} BABILong train rows", flush=True)
    return rows


def load_babilong_eval(tasks: list[str], lengths: list[str], n_per_combo: int) -> dict:
    split: dict[tuple[str, str], list[dict]] = {}
    for task in tasks:
        for length in lengths:
            try:
                ds = load_dataset("RMT-team/babilong", length, split=task)
            except Exception as exc:
                print(f"  [skip eval {task}/{length}: {exc}]", flush=True)
                continue
            split[(task, length)] = list(ds)[-n_per_combo:]
    return split


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


def get_passage_text(row: dict) -> str:
    text = str(row["input"])
    question = str(row["question"])
    q_idx = text.rfind(question)
    return text[:q_idx].strip() if q_idx != -1 else text.strip()


def get_passage_ids(tokenizer, row: dict, max_passage_tokens: int) -> torch.Tensor:
    passage = get_passage_text(row)
    ids = tokenizer(passage, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(ids) > max_passage_tokens:
        ids = ids[-max_passage_tokens:]
    return ids


def build_suffix_ids(tokenizer, question: str, cache_id: int, target: str) -> list[int]:
    q_ids = tokenizer(f"\n{question}", return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    t_ids = tokenizer(target.strip(), return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    return q_ids + [cache_id] + t_ids


def build_training_ids(
    passage_ids: torch.Tensor,
    cache_positions: list[int],
    cache_id: int,
    suffix_ids: list[int],
    bos_id: int | None,
) -> torch.Tensor:
    pos_set = set(cache_positions)
    ids: list[int] = []
    if bos_id is not None:
        ids.append(bos_id)
    for i, tok in enumerate(passage_ids.tolist()):
        ids.append(tok)
        if i in pos_set:
            ids.append(cache_id)
    ids.extend(suffix_ids)
    return torch.tensor(ids, dtype=torch.long)


def cap_positions(positions: list[int], cap: int | None) -> list[int]:
    if cap is None or cap <= 0 or len(positions) <= cap:
        return positions
    if cap == 1:
        return [positions[-1]]
    step = (len(positions) - 1) / (cap - 1)
    selected = [positions[round(i * step)] for i in range(cap)]
    return sorted(set(selected))


def probe_loss_positions(
    model,
    passage_ids: torch.Tensor,
    device: torch.device,
    top_k: int,
) -> list[int]:
    if len(passage_ids) < 2 or top_k <= 0:
        return []
    with torch.no_grad():
        logits = model(passage_ids.unsqueeze(0).to(device)).logits[0]
    log_probs = F.log_softmax(logits, dim=-1)
    targets = passage_ids[1:].to(device)
    token_loss = -log_probs[torch.arange(len(targets), device=device), targets]
    k = min(top_k, len(token_loss))
    return sorted(torch.topk(token_loss, k).indices.cpu().tolist())


def random_positions(total_tokens: int, top_k: int, seed: int) -> list[int]:
    if total_tokens <= 0 or top_k <= 0:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(range(total_tokens), k=min(top_k, total_tokens)))


def interval_positions(total_tokens: int, top_k: int) -> list[int]:
    if total_tokens <= 0 or top_k <= 0:
        return []
    k = min(top_k, total_tokens)
    positions = []
    for index in range(k):
        position = round(((index + 1) * total_tokens / (k + 1)) - 1)
        positions.append(max(0, min(total_tokens - 1, position)))
    return sorted(set(positions))


def regex_positions(tokenizer, passage_ids: torch.Tensor, cap: int | None) -> list[int]:
    passage = tokenizer.decode(passage_ids.tolist(), skip_special_tokens=True)
    positions: list[int] = []
    start = 0
    for sentence in SENT_SPLIT.split(passage):
        sentence_start = passage.find(sentence, start)
        if sentence_start == -1:
            sentence_start = start
        sentence_end = sentence_start + len(sentence)
        start = sentence_end
        sentence = sentence.strip()
        if not sentence:
            continue
        if FACT_RE.search(sentence):
            prefix_ids = tokenizer(passage[:sentence_end], add_special_tokens=False).input_ids
            if prefix_ids:
                positions.append(min(len(prefix_ids) - 1, len(passage_ids) - 1))
    return cap_positions(sorted(set(positions)), cap)


def select_cache_positions(
    *,
    placement: str,
    model,
    tokenizer,
    row: dict,
    passage_ids: torch.Tensor,
    device: torch.device,
    top_k: int,
    seed: int,
    row_index: int,
    regex_cap_k: int | None,
) -> list[int]:
    if placement == "loss":
        return probe_loss_positions(model, passage_ids, device, top_k)
    if placement == "random":
        return random_positions(len(passage_ids), top_k, seed + row_index)
    if placement == "interval":
        return interval_positions(len(passage_ids), top_k)
    if placement == "regex":
        return regex_positions(tokenizer, passage_ids, regex_cap_k)
    raise ValueError(f"Unknown placement policy: {placement}")


def cache_delta(
    h,
    cache: StateCache,
    cache_pos: torch.Tensor,
    *,
    causal_mask: bool,
    use_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dtype = cache.W_K.weight.dtype
    cache_src = h[0, cache_pos].detach().to(cache_dtype)
    K = cache.W_K(cache_src)
    V = cache.W_V(cache_src)
    Q = cache.W_Q(h[0, cache_pos].to(cache_dtype))

    n_cache = cache_pos.numel()
    if causal_mask:
        avail = torch.tril(
            torch.ones(n_cache, n_cache, device=Q.device, dtype=torch.bool),
            diagonal=-1,
        )
        if cache.max_entries > 0:
            row_idx = torch.arange(n_cache, device=Q.device).view(-1, 1)
            col_idx = torch.arange(n_cache, device=Q.device).view(1, -1)
            avail = avail & ((row_idx - col_idx) <= cache.max_entries)
    else:
        avail = ~torch.eye(n_cache, device=Q.device, dtype=torch.bool)
    any_avail = avail.any(dim=-1)
    if not any_avail.any():
        return torch.zeros_like(h[0, cache_pos]), any_avail

    scores = (Q @ K.T) / (cache.W_Q.out_features ** 0.5)
    scores = scores.masked_fill(~avail, float("-inf"))
    attn = scores[any_avail].softmax(dim=-1)
    out = attn @ V
    delta = cache.W_out(out)
    if use_gate:
        gate = torch.sigmoid(cache.W_gate(h[0, cache_pos][any_avail].to(cache_dtype)))
        delta = delta * gate

    full_delta = torch.zeros_like(h[0, cache_pos])
    full_delta[any_avail] = delta.to(full_delta.dtype)
    return full_delta, any_avail


def bam_forward_babilong(
    model,
    cache: StateCache,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    cache_id: int,
    layer_idx: int,
    *,
    use_checkpoint: bool,
    use_gate: bool,
    causal_mask: bool,
    loss_mode: str,
) -> torch.Tensor:
    backbone = _get_backbone(model)
    layers = backbone.layers
    embed = model.get_input_embeddings()
    h = embed(input_ids)

    with torch.no_grad():
        for i in range(layer_idx + 1):
            h = _run_layer(layers[i], h)

    toks = input_ids[0]
    cache_pos = (toks == cache_id).nonzero(as_tuple=True)[0]
    if cache_pos.numel() == 0:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype, requires_grad=True)

    delta, any_avail = cache_delta(
        h,
        cache,
        cache_pos,
        causal_mask=causal_mask,
        use_gate=use_gate,
    )
    if not any_avail.any():
        return torch.tensor(0.0, device=h.device, dtype=h.dtype, requires_grad=True)

    h_prime = h.clone()
    h_prime[0, cache_pos[any_avail]] = h_prime[0, cache_pos[any_avail]] + delta[any_avail]

    model_dtype = next(model.parameters()).dtype
    h_prime = h_prime.to(model_dtype)
    for i in range(layer_idx + 1, len(layers)):
        h_prime = _run_layer_ckpt(layers[i], h_prime, use_checkpoint)
    h_prime = backbone.norm_f(h_prime.to(model_dtype))
    logits = model.get_output_embeddings()(h_prime)

    if loss_mode == "focused":
        seq_len = labels.shape[1]
        last_cache = cache_pos[-1].item()
        if last_cache + 1 >= seq_len:
            return torch.tensor(0.0, device=h.device, dtype=h.dtype, requires_grad=True)
        focused_labels = labels.clone()
        focused_labels[0, :] = -100
        focused_labels[0, last_cache + 1] = labels[0, last_cache + 1]
        loss_labels = focused_labels
    elif loss_mode == "full":
        loss_labels = labels
    else:
        raise ValueError(f"Unknown loss mode: {loss_mode}")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = loss_labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def pred_matches(tokenizer, pred: str, target: str) -> bool:
    target_text = target.strip().lower()
    target_subtoks = [
        tokenizer.decode([i], skip_special_tokens=True).strip().lower()
        for i in tokenizer.encode(target_text, add_special_tokens=False)[:2]
    ]
    pred = pred.strip().lower()
    return pred in target_subtoks or bool(re.search(r"\b" + re.escape(target_text) + r"\b", pred))


def run_inline_eval(
    *,
    model,
    tokenizer,
    cache: StateCache,
    cache_id: int,
    layer_idx: int,
    eval_split: dict,
    device,
    bos_id: int | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cache.eval()
    backbone = _get_backbone(model)
    layers = backbone.layers
    margin = 64

    results: dict[tuple[str, str], tuple[int, int]] = {}
    baseline_results: dict[tuple[str, str], tuple[int, int]] = {}

    for (task, length), rows in eval_split.items():
        cache_correct = 0
        baseline_correct = 0
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
                        causal_mask=args.causal_mask == "on",
                        use_gate=args.gate == "on",
                    )
                    if any_avail.any():
                        h[0, c_pos[any_avail]] = h[0, c_pos[any_avail]] + delta[any_avail]

                model_dtype = next(model.parameters()).dtype
                h = h.to(model_dtype)
                for i in range(layer_idx + 1, len(layers)):
                    h = _run_layer(layers[i], h)
                h = backbone.norm_f(h.to(model_dtype))
                logits = model.get_output_embeddings()(h)

            pred = tokenizer.decode(
                [logits[0, -1, :].argmax(dim=-1).item()],
                skip_special_tokens=True,
            )
            if pred_matches(tokenizer, pred, str(row["target"])):
                cache_correct += 1

            with torch.no_grad():
                out = model(input_ids)
            pred_b = tokenizer.decode(
                [out.logits[0, -1, :].argmax(dim=-1).item()],
                skip_special_tokens=True,
            )
            if pred_matches(tokenizer, pred_b, str(row["target"])):
                baseline_correct += 1

        results[(task, length)] = (cache_correct, len(rows))
        baseline_results[(task, length)] = (baseline_correct, len(rows))

    print(f"\n=== BABILong Cache Eval ({args.placement}) ===")
    eval_tasks = parse_csv(args.eval_tasks)
    all_lengths = sorted({length for _, length in results.keys()})
    print(f"{'task':<6}  {'metric':<10}" + "".join(f"  {length:>5s}" for length in all_lengths))
    for task in eval_tasks:
        for label, res in [("no_cache", baseline_results), ("cache", results)]:
            row_str = f"{task:<6}  {label:<10}"
            for length in all_lengths:
                key = (task, length)
                if key in res:
                    correct, total = res[key]
                    row_str += f"  {correct / total:.3f}"
                else:
                    row_str += "     -"
            print(row_str)

    cache.train()
    return {
        f"{task}/{length}": {
            "cache_correct": results[(task, length)][0],
            "baseline_correct": baseline_results[(task, length)][0],
            "total": results[(task, length)][1],
            "cache_acc": results[(task, length)][0] / max(1, results[(task, length)][1]),
            "baseline_acc": baseline_results[(task, length)][0]
            / max(1, baseline_results[(task, length)][1]),
        }
        for task, length in results
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BABILong StateCache ablations.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--placement", choices=("regex", "loss", "random", "interval"), default="loss")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--regex-cap-k",
        type=int,
        default=None,
        help="Optional cap on regex passage cache writes. Leave unset for all regex facts.",
    )
    parser.add_argument("--cache-layer-idx", type=int, default=-2)
    parser.add_argument("--gate", choices=("on", "off"), default="on")
    parser.add_argument("--causal-mask", choices=("on", "off"), default="on")
    parser.add_argument("--loss-mode", choices=("focused", "full"), default="focused")
    parser.add_argument("--d-attn", type=int, default=256)
    parser.add_argument("--max-entries", type=int, default=64)
    parser.add_argument("--train-tasks", default="qa1,qa2,qa3")
    parser.add_argument("--train-lengths", default="0k,1k,2k")
    parser.add_argument("--eval-tasks", default="qa1,qa2,qa3")
    parser.add_argument("--eval-lengths", default="0k,1k,2k,4k,8k,16k")
    parser.add_argument("--n-train", type=int, default=50)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=2.0)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--use-checkpoint", choices=("on", "off"), default="on")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    t_start = time.time()
    hf_token = _load_token()

    if args.top_k < 0:
        raise SystemExit("--top-k must be non-negative.")
    if args.n_train <= 0 or args.n_eval <= 0:
        raise SystemExit("--n-train and --n-eval must be positive.")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.grad_accum <= 0:
        raise SystemExit("--grad-accum must be positive.")

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

    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    layer_idx = (
        num_layers + args.cache_layer_idx
        if args.cache_layer_idx < 0
        else args.cache_layer_idx
    )
    print(
        f"num_layers={num_layers} hidden_size={hidden_size} cache_layer_idx={layer_idx}",
        flush=True,
    )

    device = next(model.parameters()).device
    cache = StateCache(
        d_model=hidden_size,
        d_attn=args.d_attn,
        max_entries=args.max_entries,
    ).to(device=device)
    cache.train()
    for param in cache.parameters():
        param.requires_grad = True
    print(f"cache trainable params: {sum(p.numel() for p in cache.parameters()):,}", flush=True)

    train_rows = load_babilong_rows(
        parse_csv(args.train_tasks),
        parse_csv(args.train_lengths),
        args.n_train,
    )
    eval_split = load_babilong_eval(
        parse_csv(args.eval_tasks),
        parse_csv(args.eval_lengths),
        args.n_eval,
    )

    margin = 64
    print(f"Precomputing placement={args.placement} positions ...", flush=True)
    precomputed: list[tuple[torch.Tensor, list[int], list[int]]] = []
    cache_counts: list[int] = []
    for row_index, row in enumerate(train_rows):
        passage_ids = get_passage_ids(tokenizer, row, args.max_seq_len - margin)
        positions = select_cache_positions(
            placement=args.placement,
            model=model,
            tokenizer=tokenizer,
            row=row,
            passage_ids=passage_ids,
            device=device,
            top_k=args.top_k,
            seed=args.seed,
            row_index=row_index,
            regex_cap_k=args.regex_cap_k,
        )
        suffix_ids = build_suffix_ids(tokenizer, row["question"], cache_id, row["target"])
        precomputed.append((passage_ids, positions, suffix_ids))
        cache_counts.append(len(positions))
        if (row_index + 1) % 50 == 0:
            print(f"  precomputed {row_index + 1}/{len(train_rows)}", flush=True)

    n_with_cache = sum(1 for count in cache_counts if count > 0)
    mean_cache = sum(cache_counts) / max(1, len(cache_counts))
    print(
        f"Precompute done: {n_with_cache}/{len(train_rows)} examples have cache writes; "
        f"mean_passage_cache={mean_cache:.2f} min={min(cache_counts, default=0)} "
        f"max={max(cache_counts, default=0)}",
        flush=True,
    )

    total_opt_steps_est = max(1, (args.epochs * len(train_rows)) // args.grad_accum)
    warmup_steps = max(1, int(args.warmup_ratio * total_opt_steps_est))
    optimizer = torch.optim.AdamW(
        cache.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_opt_steps_est - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    micro_step = 0
    opt_step = 0
    optimizer.zero_grad(set_to_none=True)
    total_loss_sum = 0.0
    total_examples = 0

    for epoch in range(args.epochs):
        order = list(range(len(precomputed)))
        random.shuffle(order)
        epoch_loss_sum = 0.0
        epoch_count = 0

        for idx in order:
            passage_ids, positions, suffix_ids = precomputed[idx]
            if not positions:
                micro_step += 1
                if micro_step % args.grad_accum == 0:
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1
                continue

            seq_ids = build_training_ids(passage_ids, positions, cache_id, suffix_ids, bos_id)
            full_ids = seq_ids.unsqueeze(0).to(device)
            if full_ids.shape[1] > args.max_seq_len:
                full_ids = full_ids[:, -args.max_seq_len:]
            labels = full_ids.clone()

            loss = bam_forward_babilong(
                model=model,
                cache=cache,
                input_ids=full_ids,
                labels=labels,
                cache_id=cache_id,
                layer_idx=layer_idx,
                use_checkpoint=args.use_checkpoint == "on",
                use_gate=args.gate == "on",
                causal_mask=args.causal_mask == "on",
                loss_mode=args.loss_mode,
            )
            if loss.item() == 0.0:
                micro_step += 1
                if micro_step % args.grad_accum == 0:
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1
                continue

            (loss / args.grad_accum).backward()
            micro_step += 1
            epoch_loss_sum += loss.item()
            total_loss_sum += loss.item()
            epoch_count += 1
            total_examples += 1

            if micro_step % args.grad_accum == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.learning_rate * lr_at(opt_step)
                torch.nn.utils.clip_grad_norm_(cache.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % 20 == 0:
                    avg_loss = total_loss_sum / max(1, total_examples)
                    w_out_norm = cache.W_out.weight.data.norm().item()
                    elapsed = time.time() - t_start
                    print(
                        f"epoch {epoch} opt_step {opt_step} avg_loss={avg_loss:.4f} "
                        f"W_out_norm={w_out_norm:.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"elapsed={elapsed:.0f}s",
                        flush=True,
                    )

        epoch_avg = epoch_loss_sum / max(1, epoch_count)
        w_out_norm = cache.W_out.weight.data.norm().item()
        print(
            f"epoch {epoch} done: {epoch_count} examples, "
            f"avg_loss={epoch_avg:.4f} W_out_norm={w_out_norm:.4f}",
            flush=True,
        )
        if args.save_epoch_checkpoints:
            ckpt_path = Path(args.output).with_suffix(f".ep{epoch}.pt")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in cache.state_dict().items()},
                    "config": checkpoint_config(args, hidden_size, layer_idx),
                },
                ckpt_path,
            )

    if micro_step % args.grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(cache.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in cache.state_dict().items()},
            "config": checkpoint_config(args, hidden_size, layer_idx),
        },
        out_path,
    )
    print(f"Saved cache to {out_path}", flush=True)

    print("\nRunning inline BABILong eval ...", flush=True)
    eval_metrics = run_inline_eval(
        model=model,
        tokenizer=tokenizer,
        cache=cache,
        cache_id=cache_id,
        layer_idx=layer_idx,
        eval_split=eval_split,
        device=device,
        bos_id=bos_id,
        args=args,
    )

    total_sec = time.time() - t_start
    metrics_payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output": str(out_path),
        "total_seconds": total_sec,
        "mean_passage_cache": mean_cache,
        "min_passage_cache": min(cache_counts, default=0),
        "max_passage_cache": max(cache_counts, default=0),
        "config": checkpoint_config(args, hidden_size, layer_idx),
        "eval": eval_metrics,
    }
    metrics_path = Path(args.metrics_output) if args.metrics_output else out_path.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"Saved metrics to {metrics_path}", flush=True)
    print(f"\nTotal time: {total_sec:.0f}s", flush=True)


def checkpoint_config(args: argparse.Namespace, hidden_size: int, layer_idx: int) -> dict[str, Any]:
    return {
        "d_model": hidden_size,
        "d_attn": args.d_attn,
        "max_entries": args.max_entries,
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
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
    }


if __name__ == "__main__":
    main()
