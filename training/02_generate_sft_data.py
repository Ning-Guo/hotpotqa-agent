#!/usr/bin/env python3
"""
02_generate_sft_data.py
Run teacher model inference to generate reasoning traces for SFT.

For each example the teacher receives:
  - The question and all passages
  - The gold answer (work-backwards prompting)
  - Instruction to generate a valid <think>/<answer> trace

Generated traces are filtered:
  1. Answer must match gold (exact or normalized)
  2. Intermediate answer must appear in a retrieved passage (not hallucinated)
  3. Sub-question 1 must not be circular (< 0.7 word overlap with original)

Supports two inference backends:
  A) Local model (--local): loads TEACHER_MODEL on this GPU
  B) API (default):         uses OpenAI-compatible API via env vars
     TEACHER_API_BASE, TEACHER_API_KEY, TEACHER_API_MODEL

Usage:
    # API mode (e.g. Together AI)
    export TEACHER_API_BASE="https://api.together.xyz/v1"
    export TEACHER_API_KEY="your-key"
    export TEACHER_API_MODEL="Qwen/Qwen2.5-72B-Instruct-Turbo"
    python training/02_generate_sft_data.py

    # Local model mode
    python training/02_generate_sft_data.py --local
    python training/02_generate_sft_data.py --local --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

import training.config as cfg
from training.utils.format import (
    build_user_prompt, format_context_block,
    extract_answer, extract_think, extract_sub_q1, extract_intermediate,
    has_valid_format, is_circular_sub_q1,
)
from training.utils.metrics import normalize_answer, exact_match


# ---------------------------------------------------------------------------
# Teacher prompt
# ---------------------------------------------------------------------------

BRIDGE_TEMPLATE = """\
You are a multi-hop question answering expert.

Given the question, passages, and the CORRECT final answer, generate the \
step-by-step reasoning chain that leads to that answer using ONLY the \
provided passages. Work backwards from the answer.

Question: {question}
Question type: bridge (requires finding an intermediate entity first)
Correct answer: {answer}

Passages:
{context}

Generate your response in EXACTLY this format — no other text:

<think>
Type: bridge
Sub-question 1: [first sub-question to find the intermediate entity]
Intermediate answer: [answer to sub-question 1, must appear verbatim in passages]
Sub-question 2: [second sub-question using the intermediate entity]
</think>
<answer>{answer}</answer>

Rules:
- Sub-question 1 MUST identify the intermediate entity, NOT restate the original question
- Intermediate answer MUST appear word-for-word in one of the passages above
- Sub-question 2 MUST use the intermediate answer to ask for the final answer
- The <answer> tag MUST contain exactly: {answer}
"""

COMPARISON_TEMPLATE = """\
You are a multi-hop question answering expert.

Given the question, passages, and the CORRECT final answer, generate the \
step-by-step reasoning chain that leads to that answer using ONLY the \
provided passages. Work backwards from the answer.

Question: {question}
Question type: comparison (compare a property of two entities)
Correct answer: {answer}

Passages:
{context}

Generate your response in EXACTLY this format — no other text:

<think>
Type: comparison
Sub-question 1: [question about the property of entity 1]
Intermediate answer 1: [answer about entity 1, must appear in passages]
Sub-question 2: [question about the property of entity 2]
Intermediate answer 2: [answer about entity 2, must appear in passages]
</think>
<answer>{answer}</answer>

Rules:
- Each intermediate answer MUST appear word-for-word in one of the passages above
- The <answer> tag MUST contain exactly: {answer}
"""

INSUFFICIENT_TEMPLATE = """\
You are a multi-hop question answering expert.

Given the question and passages, generate a response indicating that the \
answer cannot be determined from the available passages.

Question: {question}

Passages:
{context}

Generate your response in EXACTLY this format:

<think>
Type: {qtype}
Note: The supporting passages for this question are not available in the context.
</think>
<answer>insufficient context</answer>
"""


def build_teacher_prompt(ex: dict) -> str:
    ctx = format_context_block(ex["passages"])

    if ex.get("gold_removed"):
        return INSUFFICIENT_TEMPLATE.format(
            question=ex["question"],
            context=ctx,
            qtype=ex["type"],
        )
    elif ex["type"] == "bridge":
        return BRIDGE_TEMPLATE.format(
            question=ex["question"],
            answer=ex["answer"],
            context=ctx,
        )
    else:
        return COMPARISON_TEMPLATE.format(
            question=ex["question"],
            answer=ex["answer"],
            context=ctx,
        )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def is_valid_trace(completion: str, ex: dict) -> tuple[bool, str]:
    """
    Returns (is_valid, rejection_reason).
    Applies three filters:
    1. Format: must have <think> and <answer>
    2. Answer: must match gold (or "insufficient context" if gold_removed)
    3. Intermediate: must be grounded in passages (not hallucinated)
    4. Sub-q1: must not be circular
    """
    if not has_valid_format(completion):
        return False, "invalid_format"

    pred = normalize_answer(extract_answer(completion))
    gold = normalize_answer(ex["answer"])

    if pred != gold:
        return False, f"wrong_answer: pred={pred!r} gold={gold!r}"

    # Skip grounding/circularity checks for gold-removed examples
    if ex.get("gold_removed"):
        return True, ""

    think = extract_think(completion)

    # Check intermediate grounding (for bridge questions)
    if ex["type"] == "bridge":
        hop1 = extract_intermediate(think)
        if hop1:
            hop1_norm = normalize_answer(hop1)
            found = any(
                hop1_norm in p["text"].lower()
                for p in ex["passages"]
            )
            if not found:
                return False, "hop1_not_grounded"

    # Check sub_q1 circularity
    sub_q1 = extract_sub_q1(think)
    if sub_q1 and is_circular_sub_q1(ex["question"], sub_q1):
        return False, "circular_sub_q1"

    return True, ""


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------

def run_api_inference(prompts: list[str], model: str, base_url: str,
                      api_key: str, max_tokens: int = 512) -> list[str]:
    """OpenAI-compatible API inference (batch via individual calls)."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)
    results = []
    for prompt in prompts:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                results.append(resp.choices[0].message.content)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [API ERROR] {e}")
                    results.append("")
                time.sleep(2 ** attempt)
    return results


def run_local_inference(prompts: list[str], model_name: str,
                        batch_size: int = 4, max_new_tokens: int = 512) -> list[str]:
    """Local model inference using HuggingFace transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading teacher model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Teacher inference"):
        batch = prompts[i:i + batch_size]
        messages_batch = [[{"role": "user", "content": p}] for p in batch]
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_batch
        ]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        for out in outputs:
            generated = out[input_len:]
            results.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=cfg.SFT_SOURCE_PATH)
    parser.add_argument("--output",     default=cfg.SFT_CLEAN_PATH)
    parser.add_argument("--local",      action="store_true",
                        help="Use local teacher model instead of API")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--resume",     action="store_true",
                        help="Skip already-generated examples (resume interrupted run)")
    args = parser.parse_args()

    # Load source examples
    with open(args.input) as f:
        examples = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(examples):,} source examples from {args.input}")

    # Resume support: skip already written IDs
    done_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                item = json.loads(line)
                done_ids.add(item["id"])
        print(f"Resuming: {len(done_ids):,} already done")
        examples = [e for e in examples if e["id"] not in done_ids]

    if not examples:
        print("Nothing to do.")
        return

    # Build prompts
    prompts = [build_teacher_prompt(ex) for ex in examples]

    # Run inference
    if args.local:
        completions = run_local_inference(
            prompts, cfg.TEACHER_MODEL,
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens,
        )
    else:
        if not cfg.TEACHER_API_BASE:
            sys.exit("ERROR: Set TEACHER_API_BASE / TEACHER_API_KEY / TEACHER_API_MODEL "
                     "or use --local flag.")
        print(f"Using API: {cfg.TEACHER_API_MODEL} @ {cfg.TEACHER_API_BASE}")
        completions = run_api_inference(
            prompts, cfg.TEACHER_API_MODEL,
            cfg.TEACHER_API_BASE, cfg.TEACHER_API_KEY,
            max_tokens=args.max_tokens,
        )

    # Filter and save
    stats = {"total": len(examples), "kept": 0, "reasons": {}}
    mode = "a" if args.resume else "w"
    with open(args.output, mode) as out_f:
        for ex, completion in tqdm(
            zip(examples, completions), total=len(examples), desc="Filtering"
        ):
            valid, reason = is_valid_trace(completion, ex)
            if valid:
                record = {
                    "id":           ex["id"],
                    "question":     ex["question"],
                    "answer":       ex["answer"],
                    "type":         ex["type"],
                    "level":        ex["level"],
                    "gold_removed": ex.get("gold_removed", False),
                    "passages":     ex["passages"],
                    "gold_titles":  ex["gold_titles"],
                    "completion":   completion,   # full <think><answer> trace
                }
                out_f.write(json.dumps(record) + "\n")
                stats["kept"] += 1
            else:
                stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1

    kept_pct = 100 * stats["kept"] / max(stats["total"], 1)
    print(f"\nResults: {stats['kept']:,} / {stats['total']:,} kept ({kept_pct:.1f}%)")
    print("Rejection breakdown:")
    for reason, count in sorted(stats["reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count:,}")
    print(f"\nClean SFT data saved → {args.output}")
    print("Next step: python training/03_train_sft.py")


if __name__ == "__main__":
    main()
