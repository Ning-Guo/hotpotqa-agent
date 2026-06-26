#!/usr/bin/env python3
"""
exp2_32b_agent.py — Experiment 2: 32B model + FAISS RAG + LangGraph agent

Runs the full multi-hop agent pipeline (classify → decompose → retrieve → answer
→ verify) using Qwen2.5-32B-Instruct as the backbone, with no fine-tuning.

The PlainModelWrapper in utils.py makes the 32B base model compatible with
src.reasoner.Reasoner, which expects model.disable_adapter() (PeftModel API).
For a base model, disable_adapter() is a no-op — all steps use the same weights.

This answers: does the agent pipeline help a large model, and by how much
does our fine-tuned 3B (exp3) close the gap?

Hardware: 1× A100 80GB  (Qwen2.5-32B-Instruct bfloat16 ≈ 64 GB)

Usage:
    python eval/exp2_32b_agent.py --load-index
    python eval/exp2_32b_agent.py --n 100 --load-index
    python eval/exp2_32b_agent.py --output results/exp2_32b_agent.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import load_jsonl, build_corpus, PlainModelWrapper, print_summary

import config
from src.retriever import Retriever
from src.graph import build_graph
from src.evaluator import exact_match, token_f1, supporting_fact_recall, \
                          answer_coverage, faithfulness, summarize

RESULTS_DIR = os.path.join(ROOT, "results")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_32b(model_name: str):
    """Load Qwen2.5-32B in bfloat16 with device_map=auto (fits 1× A100 80GB)."""
    print(f"Loading {model_name} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base.eval()
    model = PlainModelWrapper(base)
    print("Model ready.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="Qwen/Qwen2.5-32B-Instruct")
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
        slug = args.model.replace("/", "_").replace("-", "_").lower()
        n_tag = f"_n{args.n}" if args.n else "_n500"
        args.output = os.path.join(RESULTS_DIR, f"exp2_32b_agent_{slug}{n_tag}.json")

    # Load data
    items = load_jsonl(args.eval)
    if args.n:
        items = items[:args.n]
    print(f"Loaded {len(items)} questions from {args.eval}")

    # Load retriever
    if args.load_index:
        retriever = Retriever.load(
            config.INDEX_PATH, config.CORPUS_PATH, config.EMBEDDING_MODEL
        )
    else:
        corpus = build_corpus(items)
        retriever = Retriever(corpus, config.EMBEDDING_MODEL)
        retriever.save(config.INDEX_PATH, config.CORPUS_PATH)

    # Load model
    model, tokenizer = load_32b(args.model)

    # The agent pipeline uses "cuda" to send inputs to the first GPU.
    # With device_map="auto" on a single A100, all layers are on cuda:0.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build LangGraph agent (same pipeline as exp3, different backbone)
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
    print_summary(summary, f"EXP 2 — {args.model} + RAG + AGENT")

    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "summary": summary, "per_item": results}, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
