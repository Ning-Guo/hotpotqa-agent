#!/usr/bin/env python3
"""
03_train_sft.py
LoRA SFT training on teacher-generated reasoning traces.

Input:  sft_clean.jsonl  (from 02_generate_sft_data.py)
Output: checkpoints/sft_adapter/

Training format (chat template):
  User:      <query>...</query>\n\n<context>...</context>
  Assistant: <think>...</think>\n<answer>...</answer>

Cross-entropy loss is computed on the FULL assistant turn (both reasoning
and answer), so the model learns the end-to-end reasoning style.

Usage:
    python training/03_train_sft.py
    python training/03_train_sft.py --epochs 2 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

import training.config as cfg
from training.utils.format import build_user_prompt


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def load_sft_dataset(path: str, tokenizer, max_seq_len: int) -> Dataset:
    """
    Load clean SFT data and tokenize with labels.
    Labels are -100 for the system+user prompt tokens (no loss),
    and the actual token IDs for the assistant turn (loss computed here).
    This is more reliable than DataCollatorForCompletionOnlyLM for Qwen2.5.
    """
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    rows = []
    for record in records:
        user_msg = build_user_prompt(record["question"], record["passages"])
        asst_msg = record["completion"]

        # Tokenize prompt only (to find where assistant turn starts)
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        # Tokenize full conversation
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": asst_msg}],
            tokenize=False, add_generation_prompt=False,
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_enc   = tokenizer(full_text,   add_special_tokens=False,
                               truncation=True, max_length=max_seq_len)
        input_ids  = full_enc["input_ids"]

        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[:len(input_ids)]   # in case of truncation

        rows.append({
            "input_ids":      input_ids,
            "attention_mask": full_enc["attention_mask"],
            "labels":         labels,
        })

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default=cfg.SFT_CLEAN_PATH)
    parser.add_argument("--output",      default=cfg.SFT_ADAPTER_PATH)
    parser.add_argument("--base-model",  default=cfg.BASE_MODEL)
    parser.add_argument("--epochs",      type=int,   default=cfg.SFT_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=cfg.SFT_BATCH_SIZE)
    parser.add_argument("--grad-accum",  type=int,   default=cfg.SFT_GRAD_ACCUM)
    parser.add_argument("--lr",          type=float, default=cfg.SFT_LR)
    parser.add_argument("--max-seq-len", type=int,   default=cfg.SFT_MAX_SEQ_LEN)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Load tokenizer ────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"Loading SFT data: {args.input}")
    dataset = load_sft_dataset(args.input, tokenizer, args.max_seq_len)
    print(f"  {len(dataset):,} training examples")

    # ── Load base model ───────────────────────────────────────────────────
    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()                                                                                                           
    model.enable_input_require_grads()  # required when gradient_checkpointing=True + LoRA

    # ── LoRA config ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=cfg.SFT_LORA_R,
        lora_alpha=cfg.SFT_LORA_ALPHA,
        lora_dropout=cfg.SFT_LORA_DROPOUT,
        target_modules=cfg.SFT_LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training arguments ────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.SFT_WARMUP_RATIO,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    # Labels are pre-computed in load_sft_dataset (-100 for prompt tokens).
    # Plain Trainer works directly with pre-tokenized input_ids + labels.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("\nStarting SFT training...")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nSFT adapter saved → {args.output}")
    print("Next step: python training/04_merge_adapter.py")


if __name__ == "__main__":
    main()
