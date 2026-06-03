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
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

import training.config as cfg
from training.utils.format import build_user_prompt


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def load_sft_dataset(path: str, tokenizer) -> Dataset:
    """Load clean SFT data and format as chat conversations."""
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    def to_chat(record):
        user_msg = build_user_prompt(record["question"], record["passages"])
        asst_msg = record["completion"]  # already <think>...<answer>...</answer>
        # Apply tokenizer chat template
        messages = [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": asst_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text, "type": record["type"]}

    formatted = [to_chat(r) for r in records]
    return Dataset.from_list(formatted)


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
    dataset = load_sft_dataset(args.input, tokenizer)
    print(f"  {len(dataset):,} training examples")
    print(f"  Bridge:     {sum(1 for x in dataset if x['type'] == 'bridge'):,}")
    print(f"  Comparison: {sum(1 for x in dataset if x['type'] == 'comparison'):,}")
    dataset = dataset.remove_columns(["type"])

    # ── Load base model ───────────────────────────────────────────────────
    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

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
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.SFT_WARMUP_RATIO,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
        max_seq_length=args.max_seq_len,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    # Use DataCollatorForCompletionOnlyLM to compute loss only on
    # the assistant turn (not on the user prompt).
    # The response template marks where the assistant turn begins.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
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
