#!/usr/bin/env python3
"""Fixed evaluation harness for the [CACHE] LoRA experiment.

Read-only: do not modify per program.md.

Loads the LoRA adapter saved at ./adapter, runs evaluation on 30 held-out
OpenMath problems (indices 4970..4999), and prints three metrics:

  - cache_f1:        F1 of [CACHE] placement vs oracle boundaries (teacher-forced)
  - math_accuracy:   fraction of held-out problems answered correctly (free-gen)
  - cache_count_mae: MAE between number of [CACHE] tokens emitted and oracle count (free-gen)

math_accuracy uses a two-stage check:
  1. Fast heuristic: boxed-answer extraction + numeric comparison.
  2. LLM judge (JUDGE_MODEL via HF Inference API): called only when the heuristic
     says the answer is WRONG. This catches word-problem answers, equivalent LaTeX
     forms, and other cases where the heuristic produces false negatives.

Output format (parsed by train.py):
    cache_f1: <float>
    math_accuracy: <float>
    cache_count_mae: <float>
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Fixed eval constants (do not modify)
# ---------------------------------------------------------------------------
MODEL_NAME       = "tiiuae/falcon-mamba-7b-instruct"
ADAPTER_DIR      = "adapter"
DATA_PATH        = "SFT_OpenMath_data/annotated/qwen3_235b/annotated_samples.jsonl"
HELD_OUT_START   = 4980
HELD_OUT_END     = 5000
MAX_NEW_TOKENS   = 2048
EVAL_SEED        = 0
CACHE_TOKEN      = "[CACHE]"

# Judge model used as fallback when the heuristic checker says "wrong".
# Any chat model available on the HF serverless inference API works here.
JUDGE_MODEL      = "Qwen/Qwen2.5-7B-Instruct"


def _load_token() -> str | None:
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.is_file():
        load_dotenv(env_file)
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def add_cache_token(tokenizer, model) -> int:
    if CACHE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [CACHE_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer.convert_tokens_to_ids(CACHE_TOKEN)


def load_held_out() -> list[dict]:
    rows: list[dict] = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < HELD_OUT_START:
                continue
            if i >= HELD_OUT_END:
                break
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Answer extraction / comparison
# ---------------------------------------------------------------------------
_BOXED_RE = re.compile(r"\\boxed\{")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_boxed(text: str) -> str | None:
    """Extract the contents of the LAST \\boxed{...} in text, handling nested braces."""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def _normalize_answer(s: str) -> str:
    s = s.strip()
    s = s.strip("$ \t\n")
    boxed = _extract_boxed(s)
    if boxed is not None:
        s = boxed
    s = s.replace(" ", "").replace(",", "")
    s = s.replace("\\,", "").replace("\\!", "")
    s = s.rstrip(".")
    return s


def _last_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text)
    return matches[-1] if matches else None


def answers_match(pred_text: str, expected: str) -> bool:
    """Fast heuristic check: extract \\boxed{} or last number, normalize, compare."""
    pred_boxed = _extract_boxed(pred_text)
    pred_candidate = pred_boxed if pred_boxed is not None else _last_number(pred_text)
    if pred_candidate is None:
        return False
    p = _normalize_answer(pred_candidate)
    e = _normalize_answer(expected)
    if p == e:
        return True
    try:
        return abs(float(p) - float(e)) < 1e-6
    except ValueError:
        return False


_JUDGE_PROMPT = """\
You are a strict math answer checker. Your only job is to decide whether the \
student's final answer is mathematically equivalent to the correct answer.

Problem:
{problem}

Correct answer: {expected}

Student's solution (last 800 characters):
{solution_tail}

Is the student's final answer mathematically equivalent to the correct answer? \
Reply with exactly one word: YES or NO."""


def llm_judge(problem: str, expected: str, pred_text: str, client: InferenceClient) -> bool:
    """Call the judge model via HF Inference API.

    Returns True if the judge says the answer is correct, False otherwise
    (including on any API error, to avoid inflating accuracy on failures).
    """
    solution_tail = pred_text[-800:] if len(pred_text) > 800 else pred_text
    prompt = _JUDGE_PROMPT.format(
        problem=problem,
        expected=expected,
        solution_tail=solution_tail,
    )
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=JUDGE_MODEL,
            max_tokens=8,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content.strip().upper()
        return verdict.startswith("YES")
    except Exception as exc:
        try:
            print(f"  [judge error: {exc}]", flush=True)
        except (BrokenPipeError, OSError):
            pass
        return False


def is_correct(problem: str, expected: str, pred_text: str, client: InferenceClient | None) -> bool:
    """Two-stage correctness: fast heuristic first, LLM judge as fallback."""
    if answers_match(pred_text, expected):
        return True
    if client is None:
        return False
    return llm_judge(problem, expected, pred_text, client)


# ---------------------------------------------------------------------------
# Cache F1 (teacher-forced)
# ---------------------------------------------------------------------------
def _gold_positions_in_stripped(annotated_ids: list[int], cache_id: int) -> tuple[list[int], list[int]]:
    """Given the tokenization of the annotated_trace, return (stripped_ids, gold_positions).

    A gold position `p` means: in the stripped sequence, the next token (at
    index p) is where a [CACHE] should be inserted. So we record the index in
    the stripped sequence at which the [CACHE] originally appeared.
    """
    stripped: list[int] = []
    gold: list[int] = []
    for tok in annotated_ids:
        if tok == cache_id:
            gold.append(len(stripped))
        else:
            stripped.append(tok)
    return stripped, gold


@torch.no_grad()
def compute_cache_f1(model, tokenizer, rows: list[dict], cache_id: int) -> float:
    """Teacher-forced binary classification: at each position in the stripped trace,
    does the model predict [CACHE] as the next token? Compare against gold positions.
    Returns micro-F1 over all positions across all held-out samples.
    """
    device = next(model.parameters()).device
    tp = fp = fn = 0
    for row in rows:
        annotated_ids = tokenizer(row["annotated_trace"], add_special_tokens=False)["input_ids"]
        stripped_ids, gold = _gold_positions_in_stripped(annotated_ids, cache_id)
        if not stripped_ids:
            continue
        gold_set = set(gold)

        ids = torch.tensor([stripped_ids], dtype=torch.long, device=device)
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0]  # (seq_len, vocab)
        # logits[i] is the distribution over the token at position i+1 (next-token).
        # A gold position p (insert [CACHE] before token at index p in stripped) is
        # predicted iff argmax(logits[p-1]) == cache_id, for p in [1, seq_len].
        # For p == 0 (insert at the very start), there is no preceding context; skip.
        predicted: set[int] = set()
        argmax = logits.argmax(dim=-1)  # (seq_len,)
        for p in range(1, len(stripped_ids) + 1):
            if argmax[p - 1].item() == cache_id:
                predicted.add(p)

        gold_eligible = {p for p in gold_set if p >= 1}
        tp += len(predicted & gold_eligible)
        fp += len(predicted - gold_eligible)
        fn += len(gold_eligible - predicted)

    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Free-generation pass (math_accuracy + cache_count_mae)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_free_generation(
    model, tokenizer, rows: list[dict], cache_id: int, judge_client: InferenceClient | None
) -> tuple[float, float]:
    device = next(model.parameters()).device
    correct = 0
    cache_abs_errors: list[float] = []
    eos_id = tokenizer.eos_token_id

    for idx, row in enumerate(rows):
        messages = [{"role": "user", "content": row["problem"]}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(device)

        gen = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or eos_id,
            eos_token_id=eos_id,
        )
        new_ids = gen[0, prompt_ids.shape[1]:].tolist()
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
            f"[eval {idx + 1}/{len(rows)}] cache_pred={n_cache_pred} "
            f"cache_oracle={row.get('num_cache_tokens')} "
            f"answer_ok={ok} ({verdict_label})",
            flush=True,
        )

    math_acc = correct / max(1, len(rows))
    cache_mae = sum(cache_abs_errors) / max(1, len(cache_abs_errors))
    return math_acc, cache_mae


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

    print(f"Loading adapter from {ADAPTER_DIR}", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    rows = load_held_out()
    print(f"Loaded {len(rows)} held-out problems (indices {HELD_OUT_START}..{HELD_OUT_END - 1})", flush=True)

    judge_client: InferenceClient | None = None
    if hf_token:
        try:
            judge_client = InferenceClient(token=hf_token)
            print(f"Judge enabled: {JUDGE_MODEL}", flush=True)
        except Exception as exc:
            print(f"Judge disabled (could not create InferenceClient: {exc})", flush=True)
    else:
        print("Judge disabled (no HF token found; set HF_TOKEN in .env)", flush=True)

    print("Running free-generation pass (math_accuracy + cache_count_mae)", flush=True)
    math_acc, cache_mae = run_free_generation(model, tokenizer, rows, cache_id, judge_client)

    print("Running teacher-forced pass (cache_f1)", flush=True)
    cache_f1 = compute_cache_f1(model, tokenizer, rows, cache_id)

    print("---")
    print(f"cache_f1: {cache_f1:.6f}")
    print(f"math_accuracy: {math_acc:.6f}")
    print(f"cache_count_mae: {cache_mae:.6f}")


if __name__ == "__main__":
    main()
