#!/usr/bin/env python3
"""
04_merge_adapter.py
Merge the SFT LoRA adapter weights into the base model and save as a
standalone HuggingFace model. The merged model becomes the new base for
GRPO training — no adapter management needed at GRPO inference time.

Usage:
    python training/04_merge_adapter.py
    python training/04_merge_adapter.py --adapter checkpoints/sft_adapter \
                                         --output  checkpoints/sft_merged
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import training.config as cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=cfg.BASE_MODEL)
    parser.add_argument("--adapter",    default=cfg.SFT_ADAPTER_PATH)
    parser.add_argument("--output",     default=cfg.SFT_MERGED_PATH)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Load base model ───────────────────────────────────────────────────
    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # ── Load and merge SFT adapter ────────────────────────────────────────
    print(f"Loading SFT adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging adapter into base model weights...")
    model = model.merge_and_unload()
    print("  Merge complete.")

    # ── Save merged model ─────────────────────────────────────────────────
    print(f"Saving merged model → {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Verify
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Parameters: {param_count:.2f}B")
    print(f"  Saved to: {args.output}")
    print("\nNext step: python training/05_train_grpo.py")


if __name__ == "__main__":
    main()
