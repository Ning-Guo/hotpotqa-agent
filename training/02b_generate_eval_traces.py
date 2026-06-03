#!/usr/bin/env python3
"""
02b_generate_eval_traces.py
Generate teacher reasoning traces for grpo_val.jsonl.

Purpose: diagnostic analysis — compare agent's actual reasoning against
what the teacher considers the correct chain. Helps identify which pipeline
step (classify / decompose / hop1 / hop2 / answer) is the failure point.

Output: data/eval_teacher_traces.jsonl
Each record has:
  - original fields from grpo_val.jsonl
  - teacher_completion: full <think><answer> trace
  - teacher_answer: extracted answer
  - teacher_correct: bool (teacher answer matches gold)
  - teacher_sub_q1, teacher_intermediate, teacher_sub_q2: parsed fields

Usage:
    # API mode
    export TEACHER_API_BASE="https://api.together.xyz/v1"
    export TEACHER_API_KEY="your-key"
    export TEACHER_API_MODEL="Qwen/Qwen2.5-72B-Instruct-Turbo"
    python training/02b_generate_eval_traces.py

    # Local model
    python training/02b_generate_eval_traces.py --local
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

import training.config as cfg
from training.utils.format import (
    format_context_block, extract_answer, extract_think,
    extract_sub_q1, extract_intermediate, has_valid_format,
    is_circular_sub_q1,
)
from training.utils.metrics import normalize_answer, exact_match
from training.utils.inference import (
    BRIDGE_TEMPLATE, COMPARISON_TEMPLATE,
    run_api_inference, run_local_inference,
)

EVAL_PATH        = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "data", "grpo_val.jsonl")
OUTPUT_PATH      = os.path.join(cfg.DATA_DIR, "eval_teacher_traces.jsonl")


def parse_grpo_val_example(ex: dict) -> dict:
    """
    grpo_val.jsonl uses 'paragraphs' field: list of {title, text}.
    Convert to the same internal format as distractor examples.
    """
    passages    = ex.get("paragraphs", [])
    sf          = ex.get("supporting_facts", {})
    gold_titles = list(dict.fromkeys(sf.get("title", [])))
    return {
        "id":          ex["id"],
        "question":    ex["question"].strip(),
        "answer":      ex["answer"].strip(),
        "type":        ex.get("type", "bridge"),
        "level":       ex.get("level", ""),
        "passages":    passages,
        "gold_titles": gold_titles,
        "gold_removed": False,
    }


def build_teacher_prompt_for_eval(ex: dict) -> str:
    ctx = format_context_block(ex["passages"])
    if ex["type"] == "bridge":
        return BRIDGE_TEMPLATE.format(
            question=ex["question"], answer=ex["answer"], context=ctx
        )
    else:
        return COMPARISON_TEMPLATE.format(
            question=ex["question"], answer=ex["answer"], context=ctx
        )


def parse_trace(completion: str, question: str) -> dict:
    think   = extract_think(completion)
    sub_q1  = extract_sub_q1(think)
    hop1    = extract_intermediate(think)
    answer  = extract_answer(completion)
    return {
        "teacher_completion":   completion,
        "teacher_answer":       answer,
        "teacher_sub_q1":       sub_q1,
        "teacher_intermediate": hop1,
        "teacher_valid_format": has_valid_format(completion),
        "teacher_circular":     is_circular_sub_q1(question, sub_q1) if sub_q1 else False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=EVAL_PATH)
    parser.add_argument("--output",     default=OUTPUT_PATH)
    parser.add_argument("--local",      action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    os.makedirs(cfg.DATA_DIR, exist_ok=True)

    with open(args.input) as f:
        raw_examples = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(raw_examples):,} eval examples from {args.input}")

    examples = [parse_grpo_val_example(ex) for ex in raw_examples]

    # Resume support
    done_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
        print(f"Resuming: {len(done_ids):,} already done")
        raw_examples = [e for e in raw_examples if e["id"] not in done_ids]
        examples     = [e for e in examples     if e["id"] not in done_ids]

    prompts = [build_teacher_prompt_for_eval(ex) for ex in examples]

    # Run inference
    if args.local:
        completions = run_local_inference(
            prompts, cfg.TEACHER_MODEL, batch_size=args.batch_size
        )
    else:
        if not cfg.TEACHER_API_BASE:
            sys.exit("Set TEACHER_API_BASE/KEY/MODEL or use --local")
        print(f"Using API: {cfg.TEACHER_API_MODEL}")
        completions = run_api_inference(
            prompts, cfg.TEACHER_API_MODEL,
            cfg.TEACHER_API_BASE, cfg.TEACHER_API_KEY,
        )

    # Build output records and collect stats
    stats = {"total": 0, "correct": 0, "valid_format": 0, "circular": 0}
    mode = "a" if args.resume else "w"
    with open(args.output, mode) as out_f:
        for raw_ex, ex, completion in tqdm(
            zip(raw_examples, examples, completions), total=len(examples)
        ):
            parsed = parse_trace(completion, ex["question"])
            teacher_correct = exact_match(parsed["teacher_answer"], ex["answer"]) == 1.0

            record = {**raw_ex, **parsed, "teacher_correct": teacher_correct}
            out_f.write(json.dumps(record) + "\n")

            stats["total"]        += 1
            stats["correct"]      += int(teacher_correct)
            stats["valid_format"] += int(parsed["teacher_valid_format"])
            stats["circular"]     += int(parsed["teacher_circular"])

    total = max(stats["total"], 1)
    print(f"\n=== Teacher Trace Summary ===")
    print(f"Total:         {stats['total']}")
    print(f"Correct:       {stats['correct']} ({100*stats['correct']/total:.1f}%)")
    print(f"Valid format:  {stats['valid_format']} ({100*stats['valid_format']/total:.1f}%)")
    print(f"Circular sub_q1: {stats['circular']} ({100*stats['circular']/total:.1f}%)")
    print(f"\nSaved → {args.output}")
    print("\nTo analyse: load eval_teacher_traces.jsonl and compare")
    print("teacher_sub_q1 / teacher_intermediate vs agent's sub_q1 / hop1_answer")


if __name__ == "__main__":
    main()
