#!/usr/bin/env python3
"""LoRA SFT for Falcon Mamba 7B Instruct to emit a [CACHE] token at reasoning-step boundaries.

Edit the constants block below to sweep hyperparameters. After training,
this script saves the adapter to ./adapter and invokes evaluate.py in a
fresh subprocess, then prints the program.md summary block.

Usage:
    python train.py > run.log 2>&1
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Hyperparameters (this is what the autoresearch loop sweeps)
# ---------------------------------------------------------------------------
MODEL_NAME       = "tiiuae/falcon-mamba-7b-instruct"
DATA_PATH        = "SFT_OpenMath_data/annotated/qwen3_235b/annotated_samples.jsonl"
ADAPTER_OUT_DIR  = "adapter"

NUM_EXAMPLES     = 1000
NUM_EPOCHS       = 3
LEARNING_RATE    = 5e-4
WARMUP_RATIO     = 0.05
WEIGHT_DECAY     = 0.01

LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
LORA_TARGETS     = ["x_proj", "in_proj"]   # safe Mamba targets

BATCH_SIZE       = 1
GRAD_ACCUM       = 8
MAX_SEQ_LEN      = 2048

TIME_BUDGET_SEC  = 20 * 60      # hard wall-clock cap on trainer.train()
SEED             = 42

# ---------------------------------------------------------------------------
# Fixed constants (do not sweep)
# ---------------------------------------------------------------------------
CACHE_TOKEN      = "[CACHE]"
HELD_OUT_START   = 4970         # last 30 rows are reserved for evaluate.py
HELD_OUT_END     = 5000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_cache_token(tokenizer, model) -> int:
    """Register [CACHE] as a special token and resize the model's embeddings."""
    if CACHE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [CACHE_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer.convert_tokens_to_ids(CACHE_TOKEN)


def load_train_rows() -> list[dict]:
    rows: list[dict] = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= HELD_OUT_START:
                break
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= NUM_EXAMPLES:
                break
    if len(rows) < NUM_EXAMPLES:
        print(
            f"warning: requested NUM_EXAMPLES={NUM_EXAMPLES} but only loaded "
            f"{len(rows)} rows from {DATA_PATH}",
            file=sys.stderr,
        )
    return rows


def build_dataset(tokenizer, rows: list[dict]) -> Dataset:
    """Tokenize each row as a chat: user=problem, assistant=annotated_trace.

    Loss is masked on the prompt span so we only train on the assistant tokens.
    """
    examples = []
    for row in rows:
        messages = [
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": row["annotated_trace"]},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        prompt_text = tokenizer.apply_chat_template(
            messages[:1], add_generation_prompt=True, tokenize=False
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = min(len(prompt_ids), len(full_ids))

        labels = list(full_ids)
        for i in range(prompt_len):
            labels[i] = -100

        examples.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )
    return Dataset.from_list(examples)


class PadCollator:
    """Right-pads input_ids/attention_mask/labels to the longest sequence in the batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            n_pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_token_id] * n_pad)
            attention_mask.append(b["attention_mask"] + [0] * n_pad)
            labels.append(b["labels"] + [-100] * n_pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class TimeBudgetCallback(TrainerCallback):
    """Stops training once wall-clock seconds since on_train_begin exceeds budget."""

    def __init__(self, budget_seconds: int):
        self.budget_seconds = budget_seconds
        self.t0: float | None = None
        self.training_seconds: float = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.t0 is None:
            return
        elapsed = time.time() - self.t0
        if elapsed > self.budget_seconds:
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.t0 is not None:
            self.training_seconds = time.time() - self.t0


def _load_token() -> str | None:
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.is_file():
        load_dotenv(env_file)
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def main() -> None:
    set_seed(SEED)
    t_start = time.time()
    hf_token = _load_token()

    print(f"Loading tokenizer + model: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda",
        token=hf_token,
    )
    add_cache_token(tokenizer, model)

    print("Building LoRA model", flush=True)
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGETS,
        modules_to_save=["embed_tokens", "lm_head"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print(f"Loading training rows from {DATA_PATH}", flush=True)
    rows = load_train_rows()
    print(f"Tokenizing {len(rows)} examples", flush=True)
    train_ds = build_dataset(tokenizer, rows)

    args = TrainingArguments(
        output_dir="trainer_out",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=int(WARMUP_RATIO * NUM_EPOCHS * (NUM_EXAMPLES // (BATCH_SIZE * GRAD_ACCUM))),
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    time_cb = TimeBudgetCallback(TIME_BUDGET_SEC)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[time_cb],
    )

    torch.cuda.reset_peak_memory_stats()
    print(f"Starting training (budget {TIME_BUDGET_SEC}s)", flush=True)
    trainer.train()
    training_seconds = time_cb.training_seconds or (time.time() - t_start)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Saving adapter to {ADAPTER_OUT_DIR}", flush=True)
    Path(ADAPTER_OUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_OUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUT_DIR)

    del trainer, model
    torch.cuda.empty_cache()

    print("Running fast_eval.py", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", "fast_eval.py"],
        stdout=subprocess.PIPE,
        stderr=None,  # inherit stderr so it streams to run.log in real time
        text=True,
        bufsize=1,
    )
    eval_stdout_lines: list[str] = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        eval_stdout_lines.append(line)
    proc.wait()
    eval_stdout = "".join(eval_stdout_lines)

    metrics: dict[str, float] = {}
    for line in eval_stdout.splitlines():
        for key in ("cache_f1", "math_accuracy", "cache_count_mae"):
            prefix = f"{key}:"
            if line.startswith(prefix):
                try:
                    metrics[key] = float(line[len(prefix):].strip())
                except ValueError:
                    pass

    total_seconds = time.time() - t_start
    print("---")
    print(f"cache_f1:         {metrics.get('cache_f1', 0.0):.4f}")
    print(f"math_accuracy:    {metrics.get('math_accuracy', 0.0):.4f}")
    print(f"cache_count_mae:  {metrics.get('cache_count_mae', 0.0):.4f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_examples:     {len(rows)}")
    print(f"num_epochs:       {NUM_EPOCHS}")
    print(f"learning_rate:    {LEARNING_RATE}")
    print(f"lora_rank:        {LORA_RANK}")


if __name__ == "__main__":
    main()
