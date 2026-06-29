#!/usr/bin/env python3
"""
02_train_query_lora.py — SFT training for Query LoRA adapter.

Trains a LoRA adapter on Qwen2.5-3B-Instruct to improve sub_q1 / sub_q2 /
comparison rewrite quality, using teacher-annotated data from 01_generate_data.py.

Loss is computed ONLY on the assistant completion tokens (prompt is masked).
Three task types are mixed together — no task labels needed, the prompt format
itself distinguishes them.

Hardware: 1× A100 40GB  (3B + rank-16 LoRA < 20 GB)
Runtime:  ~2–3 hours on 20K samples

Usage:
    python training/query_lora/02_train_query_lora.py
    python training/query_lora/02_train_query_lora.py --epochs 5 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import config as cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def format_sample(sample: dict, tokenizer) -> dict:
    """
    Apply chat template to the messages list.
    Returns {"text": full_formatted_string}.
    The DataCollatorForCompletionOnlyLM will mask everything up to the
    assistant response marker when computing loss.
    """
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def build_dataset(path: str, tokenizer) -> Dataset:
    samples = load_jsonl(path)
    formatted = [format_sample(s, tokenizer) for s in samples]
    return Dataset.from_list(formatted)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default=cfg.TRAIN_DATA_PATH)
    parser.add_argument("--val-data",   default=cfg.VAL_DATA_PATH)
    parser.add_argument("--output",     default=cfg.CHECKPOINT_DIR)
    parser.add_argument("--model",      default=cfg.STUDENT_MODEL)
    parser.add_argument("--epochs",     type=int,   default=cfg.TRAIN_EPOCHS)
    parser.add_argument("--lr",         type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int,   default=cfg.PER_DEVICE_BATCH)
    parser.add_argument("--grad-accum", type=int,   default=cfg.GRAD_ACCUMULATION)
    parser.add_argument("--max-len",    type=int,   default=cfg.MAX_SEQ_LENGTH)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Datasets ─────────────────────────────────────────────────────────────
    print(f"Loading train data from {args.data}...")
    train_dataset = build_dataset(args.data, tokenizer)
    print(f"Loading val data from {args.val_data}...")
    val_dataset   = build_dataset(args.val_data, tokenizer)
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    # ── Base model ───────────────────────────────────────────────────────────
    # Load base model only — no GRPO adapter. Query LoRA is trained from base.
    print(f"Loading base model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── LoRA config ──────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=cfg.LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Completion-only data collator ─────────────────────────────────────────
    # Mask loss on the prompt; compute loss only on the assistant response.
    # The response_template marks where the assistant reply starts in Qwen's
    # chat format. Adjust if Qwen uses a different marker.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # ── Training arguments ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=cfg.LR_SCHEDULER,
        warmup_ratio=cfg.WARMUP_RATIO,
        bf16=torch.cuda.is_available(),
        logging_steps=cfg.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=cfg.SAVE_STEPS,
        save_strategy="steps",
        save_steps=cfg.SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=2,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        max_seq_length=args.max_len,
        dataset_text_field="text",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting Query LoRA training...")
    trainer.train()

    # ── Save final adapter ───────────────────────────────────────────────────
    final_path = os.path.join(args.output, "final")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nQuery LoRA adapter saved → {final_path}")

    # ── Print training summary ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"  Base model  : {args.model}")
    print(f"  LoRA rank   : {cfg.LORA_R}")
    print(f"  Train steps : {trainer.state.global_step}")
    print(f"  Best eval loss: {trainer.state.best_metric:.4f}")
    print(f"  Adapter saved → {final_path}")
    print("="*60)


if __name__ == "__main__":
    main()
