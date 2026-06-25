#!/usr/bin/env python3
"""
exp1_upperbound.py — Experiment 1: 32B model + golden passages (true upper bound)

Bypasses retrieval entirely. Each question is answered using only the gold
supporting passages from the HotpotQA distractor set (no distractors).

This establishes the theoretical ceiling: what is achievable when retrieval
is perfect and the model is large?

Compare against:
  exp2_32b_agent.py   — same 32B model, but retrieval via FAISS + agent pipeline
  exp3_3b_grpo_agent.py — 3B GRPO model + FAISS + agent pipeline (our system)

Hardware: 1× A100 80GB  (Qwen2.5-32B-Instruct in bfloat16 ≈ 64 GB)

Usage:
    pip install vllm
    python eval/exp1_upperbound.py
    python eval/exp1_upperbound.py --model Qwen/Qwen2.5-32B-Instruct --n 100
    python eval/exp1_upperbound.py --output results/exp1_upperbound.json
"""

from __future__ import annotations

import argparse
import json
import os

# vLLM must set this before importing to avoid CUDA fork issues on Linux
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from tqdm import tqdm
from utils import load_jsonl, get_gold_passages, get_gold_titles, print_summary

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from src.evaluator import exact_match, token_f1, supporting_fact_recall, \
                          answer_coverage, faithfulness, summarize

RESULTS_DIR = os.path.join(ROOT, "results")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def make_prompt(question: str, passages: list) -> list:
    ctx = "\n\n".join(
        f"[{i+1}] {p['title']}\n{p['text']}" for i, p in enumerate(passages)
    )
    content = (
        "Answer the question using the passages below. "
        "Give a short answer phrase only — no explanation.\n\n"
        f"{ctx}\n\nQuestion: {question}"
    )
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# vLLM batch generation
# ---------------------------------------------------------------------------

def load_vllm(model_name: str, tensor_parallel: int):
    from vllm import LLM, SamplingParams
    print(f"Loading {model_name} via vLLM (tensor_parallel={tensor_parallel})...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel,
        max_model_len=4096,
        dtype="bfloat16",
    )
    print("vLLM ready.")
    return llm, SamplingParams


def generate_batch(llm, SamplingParams, message_lists: list, max_tokens: int = 64) -> list:
    params = SamplingParams(temperature=0, max_tokens=max_tokens)
    outputs = llm.chat(message_lists, sampling_params=params)
    return [o.outputs[0].text.strip() for o in outputs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--eval",            default=config.EVAL_PATH)
    parser.add_argument("--n",               type=int, default=None,
                        help="Evaluate only first N questions")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--output",          default=None)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.output is None:
        slug = args.model.replace("/", "_").replace("-", "_").lower()
        n_tag = f"_n{args.n}" if args.n else "_n500"
        args.output = os.path.join(RESULTS_DIR, f"exp1_upperbound_{slug}{n_tag}.json")

    # Load data
    items = load_jsonl(args.eval)
    if args.n:
        items = items[:args.n]
    print(f"Loaded {len(items)} questions from {args.eval}")

    # Load model
    llm, SamplingParams = load_vllm(args.model, args.tensor_parallel)

    # Build prompts using gold passages (no retrieval)
    print("Building prompts from gold passages...")
    prompts, gold_passage_sets = [], []
    for item in items:
        gold = get_gold_passages(item)
        if not gold:
            # Fallback: use all paragraphs (should not happen in HotpotQA)
            gold = item.get("paragraphs", [])
        gold_passage_sets.append(gold)
        prompts.append(make_prompt(item["question"], gold))

    # Batch generate
    print(f"Generating answers for {len(prompts)} questions...")
    predictions = generate_batch(llm, SamplingParams, prompts, max_tokens=64)

    # Score
    results = []
    for item, pred, gold_passages in zip(items, predictions, gold_passage_sets):
        gold_titles = get_gold_titles(item)
        results.append({
            "id":            item["id"],
            "question":      item["question"],
            "gold":          item["answer"],
            "prediction":    pred,
            "type":          item.get("type"),
            "level":         item.get("level"),
            "em":            exact_match(pred, item["answer"]),
            "f1":            token_f1(pred, item["answer"]),
            # ctx_recall = 1.0 by construction (we gave all gold passages)
            "ctx_recall":    supporting_fact_recall(gold_passages, gold_titles),
            "ctx_precision": None,
            "ans_coverage":  answer_coverage(gold_passages, item["answer"]),
            "faithfulness":  faithfulness(pred, gold_passages),
            "entailment":    None,
            "contradiction": None,
            "outcome":       None,
        })

    summary = summarize(results)
    print_summary(summary, f"EXP 1 — {args.model} + GOLDEN PASSAGES")

    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "summary": summary, "per_item": results}, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
