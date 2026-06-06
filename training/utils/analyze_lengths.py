#!/usr/bin/env python3
"""
training/data/analyze_lengths.py
Analyze token length distributions of SFT training data.

Outputs:
  - Prompt / completion / total length percentiles
  - Truncation rate at current SFT_MAX_SEQ_LEN
  - Recommended values for SFT_MAX_SEQ_LEN and GRPO_MAX_NEW_TOKENS

Usage:
    cd /root/hotpotqa-agent
    source ~/venv_train/bin/activate
    python training/data/analyze_lengths.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoTokenizer
from tqdm import tqdm

import training.config as cfg
from training.utils.format import build_user_prompt


def percentile(vals: list[int], p: float) -> int:
    return vals[int(len(vals) * p)]


def stats_line(name: str, vals: list[int]) -> None:
    vals = sorted(vals)
    n = len(vals)
    print(f"  {name:<22}  "
          f"p50={percentile(vals, 0.50):>5}  "
          f"p90={percentile(vals, 0.90):>5}  "
          f"p95={percentile(vals, 0.95):>5}  "
          f"p99={percentile(vals, 0.99):>5}  "
          f"max={vals[-1]:>5}")


def main():
    sft_path = cfg.SFT_CLEAN_PATH
    if not os.path.exists(sft_path):
        sys.exit(f"File not found: {sft_path}\nRun 02_generate_sft_data.py first.")

    print(f"Loading tokenizer: {cfg.SFT_MERGED_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.SFT_MERGED_PATH, trust_remote_code=True)

    print(f"Reading: {sft_path}")
    with open(sft_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"  {len(records):,} examples\n")

    prompt_lens, completion_lens, total_lens = [], [], []

    for ex in tqdm(records, desc="Tokenizing"):
        user_msg = build_user_prompt(ex["question"], ex["passages"])

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [{"role": "user",      "content": user_msg},
             {"role": "assistant", "content": ex["completion"]}],
            tokenize=False, add_generation_prompt=False,
        )

        p_len = len(tokenizer.encode(prompt_text,         add_special_tokens=False))
        t_len = len(tokenizer.encode(full_text,           add_special_tokens=False))
        c_len = len(tokenizer.encode(ex["completion"],    add_special_tokens=False))

        prompt_lens.append(p_len)
        completion_lens.append(c_len)
        total_lens.append(t_len)

    print("\n=== Length distribution (tokens) ===")
    stats_line("Prompt",      prompt_lens)
    stats_line("Completion",  completion_lens)
    stats_line("Total",       total_lens)

    # Truncation analysis
    current_max = cfg.SFT_MAX_SEQ_LEN
    n_truncated = sum(1 for t in total_lens if t > current_max)
    pct = 100 * n_truncated / len(total_lens)
    print(f"\n=== Truncation at current SFT_MAX_SEQ_LEN={current_max} ===")
    print(f"  Truncated: {n_truncated:,} / {len(total_lens):,} ({pct:.1f}%)")

    # Recommendations
    total_sorted = sorted(total_lens)
    comp_sorted  = sorted(completion_lens)
    n = len(total_sorted)

    rec_sft   = total_sorted[int(n * 0.99)]
    rec_grpo  = int(comp_sorted[int(n * 0.99)] * 1.2)   # p99 + 20% headroom

    print(f"\n=== Recommendations ===")
    print(f"  SFT_MAX_SEQ_LEN     = {rec_sft}   (p99 of total length)")
    print(f"  GRPO_MAX_NEW_TOKENS = {rec_grpo}  (p99 of completion + 20% headroom)")

    if pct > 3.0:
        print(f"\n  ⚠  Truncation {pct:.1f}% > 3% — consider re-running SFT with "
              f"SFT_MAX_SEQ_LEN={rec_sft}")
    else:
        print(f"\n  ✓  Truncation {pct:.1f}% is acceptable — SFT checkpoint is fine")


if __name__ == "__main__":
    main()
