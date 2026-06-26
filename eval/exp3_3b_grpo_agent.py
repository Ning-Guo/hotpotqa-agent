#!/usr/bin/env python3
"""
exp3_3b_grpo_agent.py — Experiment 3: 3B GRPO model + FAISS RAG + LangGraph agent

This is the project's main result: Qwen2.5-3B-Instruct fine-tuned with SFT +
GRPO, running the full multi-hop agent pipeline.

Adapter: Norm11/qwen2.5-3b-sft-grpo-hotpotqa_v3/grpo_adapter (config.GRPO_ADAPTER_REPO)

The single PeftModel switches roles via disable_adapter():
  - adapter OFF  → base model for classify / decompose / rewrite / answer_hop1
  - adapter ON   → GRPO-tuned model for final answer synthesis

Hardware: 1× A100 40GB is sufficient (3B bfloat16 ≈ 6 GB)

Usage:
    python eval/exp3_3b_grpo_agent.py --load-index
    python eval/exp3_3b_grpo_agent.py --n 100 --load-index
    python eval/exp3_3b_grpo_agent.py --output results/exp3_3b_grpo_agent.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tqdm import tqdm
from utils import load_jsonl, build_corpus, print_summary

import config
from src.models import load_model_and_tokenizer
from src.retriever import Retriever
from src.graph import build_graph
from src.evaluator import exact_match, token_f1, supporting_fact_recall, \
                          answer_coverage, faithfulness, summarize

RESULTS_DIR = os.path.join(ROOT, "results")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=config.MODEL_NAME)
    parser.add_argument("--adapter",    default=config.GRPO_ADAPTER_REPO)
    parser.add_argument("--eval",       default=config.EVAL_PATH)
    parser.add_argument("--top-k",      type=int, default=config.TOP_K)
    parser.add_argument("--n",          type=int, default=None,
                        help="Evaluate only first N questions")
    parser.add_argument("--load-index", action="store_true",
                        help="Reuse saved FAISS index from config.INDEX_PATH")
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.output is None:
        n_tag = f"_n{args.n}" if args.n else "_n500"
        args.output = os.path.join(RESULTS_DIR, f"exp3_3b_grpo_agent{n_tag}.json")

    # Load data
    items = load_jsonl(args.eval)
    if args.n:
        items = items[:args.n]
    print(f"Loaded {len(items)} questions from {args.eval}")

    # Load model (PeftModel: base + GRPO adapter)
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.adapter)

    # Load retriever
    if args.load_index:
        retriever = Retriever.load(
            config.INDEX_PATH, config.CORPUS_PATH, config.EMBEDDING_MODEL
        )
    else:
        corpus = build_corpus(items)
        retriever = Retriever(corpus, config.EMBEDDING_MODEL)
        retriever.save(config.INDEX_PATH, config.CORPUS_PATH)

    # Build LangGraph agent
    graph = build_graph(
        model, tokenizer, device, retriever,
        top_k=args.top_k,
        faithfulness_threshold=config.FAITHFULNESS_THRESHOLD,
    )

    # Eval loop
    results = []
    for item in tqdm(items, desc="Evaluating"):
        try:
            state = graph.invoke({"question": item["question"]})
        except Exception as e:
            print(f"\n[ERROR] {item['id']}: {e}")
            state = {
                "prediction": "", "retrieved": [], "retrieval_mode": "error",
                "queries_used": [], "verified": False, "retry_count": 0,
                "qtype": None, "sub_q1": None, "hop1_answer": None, "sub_q2": None,
            }

        pred      = state.get("prediction", "")
        retrieved = state.get("retrieved", [])
        sf        = item.get("supporting_facts", {})
        gold_titles = list(set(sf.get("title", []))) if isinstance(sf, dict) else []

        results.append({
            "id":             item["id"],
            "question":       item["question"],
            "gold":           item["answer"],
            "prediction":     pred,
            "type":           item.get("type"),
            "level":          item.get("level"),
            "qtype_inferred": state.get("qtype"),
            "retrieval_mode": state.get("retrieval_mode"),
            "sub_q1":         state.get("sub_q1"),
            "hop1_answer":    state.get("hop1_answer"),
            "sub_q2":         state.get("sub_q2"),
            "queries_used":   state.get("queries_used", []),
            "verified":       state.get("verified", False),
            "retry_count":    state.get("retry_count", 0),
            "em":             exact_match(pred, item["answer"]),
            "f1":             token_f1(pred, item["answer"]),
            "ctx_recall":     supporting_fact_recall(retrieved, gold_titles),
            "ctx_precision":  None,
            "ans_coverage":   answer_coverage(retrieved, item["answer"]),
            "faithfulness":   faithfulness(pred, retrieved),
            "entailment":     None,
            "contradiction":  None,
            "outcome":        None,
        })

    summary = summarize(results)

    n_retried = sum(1 for r in results if r["retry_count"] > 0)
    n_web     = sum(1 for r in results if r["retrieval_mode"] == "web_search")
    print(f"\n  Retried   : {n_retried}/{len(results)}")
    print(f"  Web search: {n_web}/{len(results)}")
    print_summary(summary, f"EXP 3 — 3B GRPO + RAG + AGENT  ({args.adapter})")

    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "summary": summary, "per_item": results}, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
