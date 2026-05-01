#!/usr/bin/env python3
"""BAM-aware evaluation: free-gen via BAMGenerator, teacher-forced F1 via plain model.

Loads the phase-1 LoRA adapter (for [CACHE] placement) and the phase-2
trained StateCache (for post-[CACHE] retrieval). Reuses the correctness
checker from evaluate.py so the math_accuracy policy is identical to
phase-1.

Output (parsed by bam/train_cache.py):
    cache_f1: <float>
    math_accuracy: <float>
    cache_count_mae: <float>
    cache_writes_avg: <float>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bam.cache import StateCache  # noqa: E402
from bam.generator import BAMGenerator  # noqa: E402
from evaluate import (  # noqa: E402
    CACHE_TOKEN,
    EVAL_SEED,
    HELD_OUT_END,
    HELD_OUT_START,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    _load_token,
    add_cache_token,
    answers_match,
    compute_cache_f1,
    llm_judge,
    load_held_out,
)
from huggingface_hub import InferenceClient  # noqa: E402

# ---------------------------------------------------------------------------
# Phase-2 eval constants
# ---------------------------------------------------------------------------
SFT_ADAPTER_DIR = "adapter"
CACHE_CKPT_PATH = "bam/cache_module.pt"
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def run_bam_free_generation(
    generator: BAMGenerator,
    tokenizer,
    rows: list[dict],
    cache_id: int,
    judge_client: InferenceClient | None,
) -> tuple[float, float, float]:
    device = next(generator.model.parameters()).device
    correct = 0
    cache_abs_errors: list[float] = []
    writes_total = 0

    for idx, row in enumerate(rows):
        messages = [{"role": "user", "content": row["problem"]}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(device)

        new_ids = generator.generate(
            prompt_input_ids=prompt_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
        )
        writes_total += generator.num_writes

        n_cache_pred = sum(1 for t in new_ids if t == cache_id)
        cache_abs_errors.append(abs(n_cache_pred - int(row.get("num_cache_tokens", 0))))

        text = tokenizer.decode(new_ids, skip_special_tokens=False)
        text_no_cache = text.replace(CACHE_TOKEN, "")
        expected = str(row["expected_answer"])

        heuristic_ok = answers_match(text_no_cache, expected)
        if heuristic_ok:
            ok = True
            verdict_label = "heuristic"
        elif judge_client is not None:
            ok = llm_judge(row["problem"], expected, text_no_cache, judge_client)
            verdict_label = "judge=YES" if ok else "judge=NO"
        else:
            ok = False
            verdict_label = "no_judge"

        if ok:
            correct += 1

        print(
            f"[bam-eval {idx + 1}/{len(rows)}] cache_pred={n_cache_pred} "
            f"cache_oracle={row.get('num_cache_tokens')} "
            f"writes={generator.num_writes} "
            f"answer_ok={ok} ({verdict_label})",
            flush=True,
        )

    math_acc = correct / max(1, len(rows))
    cache_mae = sum(cache_abs_errors) / max(1, len(cache_abs_errors))
    writes_avg = writes_total / max(1, len(rows))
    return math_acc, cache_mae, writes_avg


def main() -> None:
    torch.manual_seed(EVAL_SEED)
    hf_token = _load_token()

    print(f"Loading base model: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=hf_token,
    )
    cache_id = add_cache_token(tokenizer, model)

    print(f"Loading SFT adapter from {SFT_ADAPTER_DIR}", flush=True)
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_DIR)
    model.eval()

    print(f"Loading cache module from {CACHE_CKPT_PATH}", flush=True)
    ckpt = torch.load(CACHE_CKPT_PATH, map_location="cpu")
    cfg = ckpt["config"]
    device = next(model.parameters()).device
    cache_module = StateCache(
        d_model=cfg["d_model"],
        d_attn=cfg["d_attn"],
        max_entries=cfg["max_entries"],
    ).to(device=device, dtype=torch.bfloat16)
    cache_module.load_state_dict(ckpt["state_dict"])
    cache_module.eval()
    cache_layer_idx = cfg["cache_layer_idx"]
    print(
        f"cache config: d_attn={cfg['d_attn']} max_entries={cfg['max_entries']} "
        f"cache_layer_idx={cache_layer_idx}",
        flush=True,
    )

    rows = load_held_out()
    print(
        f"Loaded {len(rows)} held-out problems (indices {HELD_OUT_START}..{HELD_OUT_END - 1})",
        flush=True,
    )

    judge_client: InferenceClient | None = None
    if hf_token:
        try:
            judge_client = InferenceClient(token=hf_token)
            print(f"Judge enabled: {JUDGE_MODEL}", flush=True)
        except Exception as exc:
            print(f"Judge disabled (could not create InferenceClient: {exc})", flush=True)
    else:
        print("Judge disabled (no HF token found; set HF_TOKEN in .env)", flush=True)

    generator = BAMGenerator(
        model=model,
        tokenizer=tokenizer,
        cache_module=cache_module,
        cache_layer_idx=cache_layer_idx,
        cache_token=CACHE_TOKEN,
    )

    print("Running BAM free-generation pass", flush=True)
    try:
        math_acc, cache_mae, writes_avg = run_bam_free_generation(
            generator, tokenizer, rows, cache_id, judge_client
        )
    finally:
        generator.close()

    print("Running teacher-forced pass (cache_f1, hooks disabled)", flush=True)
    cache_f1 = compute_cache_f1(model, tokenizer, rows, cache_id)

    print("---")
    print(f"cache_f1: {cache_f1:.6f}")
    print(f"math_accuracy: {math_acc:.6f}")
    print(f"cache_count_mae: {cache_mae:.6f}")
    print(f"cache_writes_avg: {writes_avg:.6f}")


if __name__ == "__main__":
    main()
