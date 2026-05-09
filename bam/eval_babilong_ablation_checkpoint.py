#!/usr/bin/env python3
"""Evaluate a saved BABILong StateCache checkpoint on additional tasks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bam.cache import StateCache
from bam.train_babilong_ablation import (
    DEFAULT_MODEL_NAME,
    add_cache_token,
    load_babilong_eval,
    parse_csv,
    parse_device_map,
    resolve_torch_dtype,
    run_inline_eval,
    set_seed,
    _load_token,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BABILong inline cache eval from a saved StateCache checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument(
        "--append-metrics",
        action="store_true",
        help="If metrics-output exists, merge new eval keys into its eval object.",
    )
    parser.add_argument("--cells-output", default=None)
    parser.add_argument("--eval-tasks", default="qa4,qa5,qa6,qa7,qa8,qa9,qa10")
    parser.add_argument("--eval-lengths", default="0k,1k,2k,4k,8k,16k")
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--placement", choices=("regex", "loss", "random", "interval"), default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--regex-cap-k", type=int, default=None)
    parser.add_argument("--cache-layer-idx", type=int, default=None)
    parser.add_argument("--gate", choices=("on", "off"), default=None)
    parser.add_argument("--route-gate", choices=("off", "scalar", "vector"), default=None)
    parser.add_argument("--causal-mask", choices=("on", "off"), default=None)
    parser.add_argument("--loss-mode", choices=("focused", "full"), default=None)
    parser.add_argument("--train-lengths", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def cfg_value(args: argparse.Namespace, cfg: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name.replace("-", "_"), None)
    return cfg.get(name, default) if value is None else value


def main() -> None:
    args = parse_args()
    if args.n_eval <= 0:
        raise SystemExit("--n-eval must be positive.")

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
    args.loss_mode = cfg_value(args, cfg, "loss_mode", "focused")
    args.train_lengths = ",".join(cfg.get("train_lengths", ["0k", "1k", "2k"])) if args.train_lengths is None else args.train_lengths
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

    eval_split = load_babilong_eval(
        parse_csv(args.eval_tasks),
        parse_csv(args.eval_lengths),
        args.n_eval,
    )
    eval_metrics = run_inline_eval(
        model=model,
        tokenizer=tokenizer,
        cache=cache,
        cache_id=cache_id,
        layer_idx=int(args.cache_layer_idx),
        eval_split=eval_split,
        device=device,
        bos_id=bos_id,
        args=args,
    )

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint_path),
        "total_seconds": time.time() - t_start,
        "config": {
            **cfg,
            "eval_tasks": parse_csv(args.eval_tasks),
            "eval_lengths": parse_csv(args.eval_lengths),
            "n_eval": args.n_eval,
        },
        "eval": eval_metrics,
    }

    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if args.append_metrics and metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
        existing.setdefault("eval", {}).update(eval_metrics)
        existing.setdefault("append_history", []).append(
            {
                "timestamp_utc": payload["timestamp_utc"],
                "checkpoint": str(checkpoint_path),
                "eval_tasks": parse_csv(args.eval_tasks),
                "eval_lengths": parse_csv(args.eval_lengths),
                "n_eval": args.n_eval,
                "total_seconds": payload["total_seconds"],
            }
        )
        metrics_path.write_text(json.dumps(existing, indent=2))
    else:
        metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
