#!/usr/bin/env python3
"""
run_eval.py — evaluate the live multi-hop QA agent on grpo_val.jsonl.

Runs every question through build_graph().invoke() — no pre-computed artifacts.
Computes EM, F1, context recall, answer coverage, and faithfulness.
Saves per-item results + summary to results/eval_live.json.

Usage:
    python3 run_eval.py                    # full 500-question eval
    python3 run_eval.py --n 20             # quick smoke test (first 20 questions)
    python3 run_eval.py --load-index       # skip FAISS rebuild
"""

import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

import config
from src.models import load_model_and_tokenizer
from src.retriever import Retriever
from src.graph import build_graph
from src.evaluator import exact_match, token_f1, supporting_fact_recall, \
                          answer_coverage, faithfulness, summarize


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_corpus(items):
    seen, corpus = set(), []
    for item in items:
        for p in item.get("paragraphs", []):
            if p["title"] not in seen:
                seen.add(p["title"])
                corpus.append({"title": p["title"], "text": p["text"]})
    return corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",       default=config.EVAL_PATH)
    parser.add_argument("--adapter",    default=config.GRPO_ADAPTER_REPO)
    parser.add_argument("--top-k",      type=int, default=config.TOP_K)
    parser.add_argument("--n",          type=int, default=None,
                        help="Evaluate only first N questions (for quick testing)")
    parser.add_argument("--load-index", action="store_true",
                        help="Load pre-built FAISS index from config.INDEX_PATH")
    parser.add_argument("--bm25",      action="store_true",
                        help="Use BM25+dense RRF hybrid retrieval (requires rank-bm25)")
    parser.add_argument("--output",    default=os.path.join(RESULTS_DIR, "eval_live.json"))
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    eval_items = load_jsonl(args.eval)
    if args.n:
        eval_items = eval_items[:args.n]
    print(f"Loaded {len(eval_items)} eval items from {args.eval}")

    # ── Load model ────────────────────────────────────────────────────────
    model, tokenizer, device = load_model_and_tokenizer(config.MODEL_NAME, args.adapter)

    # ── Load / build retriever ────────────────────────────────────────────
    if args.load_index:
        retriever = Retriever.load(config.INDEX_PATH, config.CORPUS_PATH, config.EMBEDDING_MODEL,
                                   use_bm25=args.bm25)
    else:
        corpus    = build_corpus(eval_items)
        retriever = Retriever(corpus, config.EMBEDDING_MODEL, use_bm25=args.bm25)
        retriever.save(config.INDEX_PATH, config.CORPUS_PATH)

    # ── Build graph ───────────────────────────────────────────────────────
    graph = build_graph(
        model, tokenizer, device, retriever,
        top_k=args.top_k,
        faithfulness_threshold=config.FAITHFULNESS_THRESHOLD,
    )

    # ── Eval loop ─────────────────────────────────────────────────────────
    results = []
    for item in tqdm(eval_items, desc="Evaluating"):
        try:
            state = graph.invoke({"question": item["question"]})
        except Exception as e:
            print(f"\n[ERROR] {item['id']}: {e}")
            state = {"prediction": "", "retrieved": [], "retrieval_mode": "error",
                     "queries_used": [], "verified": False, "retry_count": 0,
                     "qtype": None, "sub_q1": None, "hop1_answer": None, "sub_q2": None}

        prediction = state.get("prediction", "")
        retrieved  = state.get("retrieved", [])

        # supporting_facts is {"title": [...], "sent_id": [...]} in HotpotQA
        sf = item.get("supporting_facts", {})
        gold_titles = list(set(sf.get("title", []))) if isinstance(sf, dict) else []

        results.append({
            "id":             item["id"],
            "question":       item["question"],
            "gold":           item["answer"],
            "prediction":     prediction,
            "type":           item.get("type"),
            "level":          item.get("level"),
            "qtype_inferred": state.get("qtype"),
            "retrieval_mode": state.get("retrieval_mode"),
            "queries_used":   state.get("queries_used", []),
            "sub_q1":         state.get("sub_q1"),
            "hop1_answer":    state.get("hop1_answer"),
            "sub_q2":         state.get("sub_q2"),
            "verified":       state.get("verified", False),
            "retry_count":    state.get("retry_count", 0),
            "em":             exact_match(prediction, item["answer"]),
            "f1":             token_f1(prediction, item["answer"]),
            "ctx_recall":     supporting_fact_recall(retrieved, gold_titles),
            "ctx_precision":  None,
            "ans_coverage":   answer_coverage(retrieved, item["answer"]),
            "faithfulness":   faithfulness(prediction, retrieved),
            "entailment":     None,
            "contradiction":  None,
            "neutral":        None,
            "outcome":        None,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    summary = summarize(results)

    # Verification / retry / web stats
    n_verified = sum(1 for r in results if r["verified"])
    n_retried  = sum(1 for r in results if r["retry_count"] > 0)
    n_web      = sum(1 for r in results if r["retrieval_mode"] == "web_search")

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  LIVE AGENT EVAL  |  n={summary['n']}  top-k={args.top_k}")
    print(sep)
    print(f"  Exact Match    : {summary['em']:.4f}   (prior best: 0.5450)")
    print(f"  Token F1       : {summary['f1']:.4f}   (prior best: 0.6200)")
    print(f"  Context Recall : {summary['ctx_recall']:.4f}")
    print(f"  Ans Coverage   : {summary['ans_coverage']:.4f}")
    print(f"  Faithfulness   : {summary['faithfulness']:.4f}")
    print(f"\n  Verified       : {n_verified}/{summary['n']}")
    print(f"  Retried        : {n_retried}/{summary['n']}")
    print(f"  Web search     : {n_web}/{summary['n']}")
    print(f"\n  By question type:")
    for qtype, s in summary["by_type"].items():
        print(f"    {qtype:<14} EM={s['em']:.4f}  F1={s['f1']:.4f}  (n={s['count']})")
    print(sep)

    # ── Save ──────────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "summary": summary, "per_item": results}, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
