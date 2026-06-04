#!/usr/bin/env python3
"""
05_train_grpo.py
GRPO training on top of the SFT-merged model.

The model generates full reasoning traces (<think>...</think><answer>...</answer>)
and is rewarded by four composable reward functions:

  reward_format     — hard gate: valid tags required (weight: gate)
  reward_accuracy   — 0.5*EM + 0.5*F1 on normalized answer (weight: 0.70)
  reward_grounding  — intermediate answer grounded in gold passage (weight: 0.25)
  reward_length     — soft penalty for verbose answers (weight: up to -0.05)

Input:  checkpoints/sft_merged/  (from 04_merge_adapter.py)
        data/grpo_train.jsonl
Output: checkpoints/grpo_adapter/

Usage:
    python training/05_train_grpo.py
    python training/05_train_grpo.py --epochs 1 --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import training.config as cfg
from training.utils.format import build_user_prompt
from training.utils.metrics import (
    reward_format,
    reward_accuracy,
    reward_grounding,
    reward_length,
)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def load_grpo_dataset(path: str) -> Dataset:
    """
    Load GRPO training data.
    Dataset columns needed by reward functions are serialised as JSON strings
    because TRL's DataLoader requires scalar-serialisable columns.
    """
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    rows = []
    for r in records:
        rows.append({
            "prompt":             build_user_prompt(r["question"], r["passages"]),
            "gold_answers":       r["answer"],
            "passages_json":      json.dumps(r["passages"]),
            "gold_titles_json":   json.dumps(r.get("gold_titles", [])),
            "question_types":     r["type"],
        })
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Reward function wrappers
# TRL GRPOTrainer passes completions as list[str] plus dataset columns as kwargs.
# Each reward function returns list[float] of the same length.
# ---------------------------------------------------------------------------

def make_weighted_reward(accuracy_w, grounding_w, max_length_penalty):
    """
    Returns a single combined reward function.
    Format is a hard gate — if format invalid the whole reward is 0.
    """
    def combined_reward(completions, gold_answers, passages_json,
                        gold_titles_json, question_types, **kwargs):
        n = len(completions)
        fmt_scores = reward_format(completions)
        acc_scores = reward_accuracy(completions, gold_answers)
        gnd_scores = reward_grounding(
            completions, passages_json, gold_titles_json, question_types
        )
        len_scores = reward_length(completions)

        rewards = []
        for i in range(n):
            if fmt_scores[i] == 0.0:
                rewards.append(0.0)
            else:
                r = (accuracy_w * acc_scores[i] +
                     grounding_w * gnd_scores[i] +
                     len_scores[i])
                rewards.append(r)
        return rewards
    return combined_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",  default=cfg.SFT_MERGED_PATH)
    parser.add_argument("--input",       default=cfg.GRPO_TRAIN_PATH)
    parser.add_argument("--output",      default=cfg.GRPO_ADAPTER_PATH)
    parser.add_argument("--epochs",      type=int,   default=cfg.GRPO_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=cfg.GRPO_BATCH_SIZE)
    parser.add_argument("--grad-accum",  type=int,   default=cfg.GRPO_GRAD_ACCUM)
    parser.add_argument("--lr",          type=float, default=cfg.GRPO_LR)
    parser.add_argument("--num-gen",     type=int,   default=cfg.GRPO_NUM_GENERATIONS)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Load tokenizer ────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"Loading GRPO data: {args.input}")
    dataset = load_grpo_dataset(args.input)
    print(f"  {len(dataset):,} training examples")

    # ── Load SFT-merged model ─────────────────────────────────────────────
    print(f"Loading model: {args.base_model}")
    model_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("  Using Flash Attention 2")
    except ImportError:
        print("  flash-attn not found, using default attention")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    # ── LoRA config (fresh adapter on top of SFT-merged model) ────────────
    lora_config = LoraConfig(
        r=cfg.GRPO_LORA_R,
        lora_alpha=cfg.GRPO_LORA_ALPHA,
        target_modules=cfg.GRPO_LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )

    # ── GRPO config ───────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        bf16=True,

        # Rollout settings
        num_generations=args.num_gen,
        max_completion_length=cfg.GRPO_MAX_NEW_TOKENS,
        max_prompt_length=cfg.GRPO_MAX_PROMPT_LEN,

        # KL penalty against the SFT-merged reference model
        # Prevents GRPO from drifting too far from SFT distribution
        beta=cfg.GRPO_KL_COEF,

        logging_steps=20,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        report_to="none",
    )

    # ── Reward function ───────────────────────────────────────────────────
    reward_fn = make_weighted_reward(
        accuracy_w=cfg.REWARD_ACCURACY_WEIGHT,
        grounding_w=cfg.REWARD_GROUNDING_WEIGHT,
        max_length_penalty=cfg.REWARD_LENGTH_MAX_PENALTY,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("\nStarting GRPO training...")
    print(f"  Reward weights: accuracy={cfg.REWARD_ACCURACY_WEIGHT}, "
          f"grounding={cfg.REWARD_GROUNDING_WEIGHT}, "
          f"length_penalty_max={cfg.REWARD_LENGTH_MAX_PENALTY}")
    print(f"  KL coef: {cfg.GRPO_KL_COEF}")
    print(f"  Rollouts per prompt: {args.num_gen}")

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nGRPO adapter saved → {args.output}")
    print("\nNext step: evaluate with")
    print("  python run_eval.py --adapter training/checkpoints/grpo_adapter "
          "--load-index --output results/eval_sft_grpo.json")


if __name__ == "__main__":
    main()
