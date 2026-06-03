#!/usr/bin/env python3
"""
01_prepare_dataset.py
Load the HotpotQA distractor dataset, preprocess, and split into:
  - sft_source.jsonl   (30K examples for teacher inference)
  - grpo_train.jsonl   (60K examples for GRPO training)
  - inner_val.jsonl    (remaining ~400 examples for SFT overfitting check)

Preprocessing applied to every example:
  - Passage order shuffled
  - 10% of examples: gold passages removed, answer replaced with
    "insufficient context" (mirrors real retrieval failure rate)

Usage:
    python training/01_prepare_dataset.py
    python training/01_prepare_dataset.py --sft-size 30000 --grpo-size 60000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from tqdm import tqdm

import training.config as cfg
from training.utils.format import parse_hotpotqa_example, preprocess_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-size",  type=int, default=cfg.SFT_SIZE)
    parser.add_argument("--grpo-size", type=int, default=cfg.GRPO_SIZE)
    parser.add_argument("--seed",      type=int, default=cfg.RANDOM_SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(cfg.DATA_DIR, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────
    print("Loading hotpotqa/hotpot_qa distractor split...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", trust_remote_code=True)
    train_examples = list(ds["train"])
    print(f"  Train: {len(train_examples):,}  |  Validation: {len(ds['validation']):,}")

    # ── Parse all train examples ──────────────────────────────────────────
    print("Parsing examples...")
    parsed = []
    for ex in tqdm(train_examples, desc="Parsing"):
        try:
            parsed.append(parse_hotpotqa_example(ex))
        except Exception as e:
            print(f"  [SKIP] {ex.get('id', '?')}: {e}")

    print(f"  Parsed {len(parsed):,} examples")
    print(f"  Bridge:     {sum(1 for e in parsed if e['type'] == 'bridge'):,}")
    print(f"  Comparison: {sum(1 for e in parsed if e['type'] == 'comparison'):,}")

    # ── Shuffle and split ─────────────────────────────────────────────────
    rng.shuffle(parsed)

    total_needed = args.sft_size + args.grpo_size
    if total_needed > len(parsed):
        print(f"  WARNING: requested {total_needed:,} but only {len(parsed):,} available.")
        total_needed = len(parsed)

    sft_raw   = parsed[:args.sft_size]
    grpo_raw  = parsed[args.sft_size:args.sft_size + args.grpo_size]
    inner_val = parsed[args.sft_size + args.grpo_size:]

    print(f"\nSplit:")
    print(f"  SFT source:  {len(sft_raw):,}")
    print(f"  GRPO train:  {len(grpo_raw):,}")
    print(f"  Inner val:   {len(inner_val):,}")

    # ── Build global distractor pool ──────────────────────────────────────
    # Collect all passages from all parsed examples, deduped by title.
    # Used to pad gold-removed examples back to 10 passages so that passage
    # count carries no information about gold presence.
    print("Building global distractor pool...")
    pool_seen, distractor_pool = set(), []
    for ex in parsed:
        for p in ex["passages"]:
            if p["title"] not in pool_seen:
                pool_seen.add(p["title"])
                distractor_pool.append(p)
    print(f"  Pool size: {len(distractor_pool):,} unique passages")

    # ── Preprocessing ─────────────────────────────────────────────────────
    def apply_preprocessing(examples: list, label: str) -> list:
        result = []
        n_removed = 0
        for ex in tqdm(examples, desc=f"Preprocessing {label}"):
            remove = rng.random() < cfg.GOLD_ABSENT_FRACTION
            processed = preprocess_example(
                ex, remove_gold=remove, rng=rng, distractor_pool=distractor_pool
            )
            if remove:
                n_removed += 1
            result.append(processed)
        pct = 100 * n_removed / max(len(examples), 1)
        print(f"  {label}: {n_removed:,} / {len(examples):,} gold-removed ({pct:.1f}%)")
        print(f"  All examples have 10 passages (gold-removed padded from pool).")
        return result

    sft_processed   = apply_preprocessing(sft_raw,   "SFT")
    grpo_processed  = apply_preprocessing(grpo_raw,  "GRPO")
    val_processed   = apply_preprocessing(inner_val, "Val")

    # ── Save ──────────────────────────────────────────────────────────────
    def save_jsonl(data: list, path: str):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"  Saved {len(data):,} → {path}")

    print("\nSaving...")
    save_jsonl(sft_processed,  cfg.SFT_SOURCE_PATH)
    save_jsonl(grpo_processed, cfg.GRPO_TRAIN_PATH)
    save_jsonl(val_processed,  cfg.INNER_VAL_PATH)

    print("\nDone. Next step: python training/02_generate_sft_data.py")


if __name__ == "__main__":
    main()
