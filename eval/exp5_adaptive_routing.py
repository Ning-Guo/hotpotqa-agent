#!/usr/bin/env python3
"""
exp5_adaptive_routing.py — Experiment 5: Adaptive routing (Method A)

Route by ground-truth question type at evaluation time:
  - Comparison  →  naive single-hop RAG  (base model, no decomposition)
  - Bridge      →  full multi-hop agent  (GRPO adapter + LangGraph)

Hypothesis: comparison questions have both named entities present, so BGE
retrieval finds them directly; decomposition and rewriting only add noise.
Bridge questions genuinely need multi-hop reasoning to reach the answer entity.

Expected improvement over Exp3 (EM=0.540):
  - Comparison path replaces agent EM=0.571 → naive RAG EM≈0.604  (+3.3 pp on 91 items)
  - Bridge path unchanged at EM=0.540 on 409 items
  - Blended estimate: (409×0.540 + 91×0.604) / 500 ≈ 0.552

Hardware: 1× A100 is sufficient (3B bfloat16 ≈ 6 GB VRAM)

Usage:
    python eval/exp5_adaptive_routing.py --load-index
    python eval/exp5_adaptive_routing.py --n 100 --load-index
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
from utils import load_jsonl, build_corpus, get_gold_titles, print_summary

import config
from src.models import load_model_and_tokenizer
from src.retriever import Retriever
from src.graph import build_graph
from src.evaluator import exact_match, token_f1, supporting_fact_recall, \
                          answer_coverage, faithfulness, summarize

RESULTS_DIR = os.path.join(ROOT, "results")


# ---------------------------------------------------------------------------
# Comparison: naive single-hop RAG with base model (no adapter)
# ---------------------------------------------------------------------------

def _naive_rag_answer(model, tokenizer, device: str,
                      question: str, retrieved_meta: list) -> str:
    """Generate answer from retrieved passages using base model (adapter disabled)."""
    ctx = "\n\n".join(
        f"[{i+1}] {p['title']}\n{p['text']}" for i, p in enumerate(retrieved_meta)
    )
    content = (
        "Answer the question using the passages. "
        "Give a short answer phrase only — no explanation.\n\n"
        f"{ctx}\n\n"
        f"Question: {question}"
    )
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        with model.disable_adapter():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def run_comparison_naive_rag(model, tokenizer, device: str,
                             retriever: Retriever, item: dict,
                             top_k: int) -> dict:
    """Single-hop RAG for comparison questions."""
    question    = item["question"]
    gold_titles = get_gold_titles(item)

    retrieved_meta = retriever.retrieve_with_meta(question, top_k=top_k)
    pred = _naive_rag_answer(model, tokenizer, device, question, retrieved_meta)

    return {
        "id":             item["id"],
        "question":       question,
        "gold":           item["answer"],
        "prediction":     pred,
        "type":           item.get("type"),
        "level":          item.get("level"),
        "routing":        "naive_rag",
        "qtype_inferred": "comparison",
        "retrieval_mode": "naive_rag",
        "sub_q1":         None,
        "hop1_answer":    None,
        "sub_q2":         None,
        "queries_used":   [question],
        "verified":       None,
        "retry_count":    0,
        "em":             exact_match(pred, item["answer"]),
        "f1":             token_f1(pred, item["answer"]),
        "ctx_recall":     supporting_fact_recall(retrieved_meta, gold_titles),
        "ctx_precision":  None,
        "ans_coverage":   answer_coverage(retrieved_meta, item["answer"]),
        "faithfulness":   faithfulness(pred, retrieved_meta),
        "entailment":     None,
        "contradiction":  None,
        "outcome":        None,
    }


# ---------------------------------------------------------------------------
# Bridge: full multi-hop agent (GRPO adapter + LangGraph)
# ---------------------------------------------------------------------------

def run_bridge_agent(graph, item: dict) -> dict:
    """Multi-hop agent for bridge questions."""
    question    = item["question"]
    gold_titles = get_gold_titles(item)

    try:
        state = graph.invoke({"question": question})
    except Exception as e:
        print(f"\n[ERROR] {item['id']}: {e}")
        state = {
            "prediction": "", "retrieved": [], "retrieval_mode": "error",
            "queries_used": [], "verified": False, "retry_count": 0,
            "qtype": "bridge", "sub_q1": None, "hop1_answer": None, "sub_q2": None,
        }

    pred      = state.get("prediction", "")
    retrieved = state.get("retrieved", [])

    return {
        "id":             item["id"],
        "question":       question,
        "gold":           item["answer"],
        "prediction":     pred,
        "type":           item.get("type"),
        "level":          item.get("level"),
        "routing":        "agent",
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
    }


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
        args.output = os.path.join(RESULTS_DIR, f"exp5_adaptive_routing{n_tag}.json")

    # ── Load eval data ─────────────────────────────────────────────────────
    items = load_jsonl(args.eval)
    if args.n:
        items = items[:args.n]
    n_bridge     = sum(1 for it in items if it.get("type") == "bridge")
    n_comparison = sum(1 for it in items if it.get("type") == "comparison")
    print(f"Loaded {len(items)} questions  ({n_bridge} bridge / {n_comparison} comparison)")

    # ── Load model (GRPO adapter) ──────────────────────────────────────────
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.adapter)

    # ── Load retriever ─────────────────────────────────────────────────────
    if args.load_index:
        retriever = Retriever.load(
            config.INDEX_PATH, config.CORPUS_PATH, config.EMBEDDING_MODEL
        )
    else:
        corpus = build_corpus(items)
        retriever = Retriever(corpus, config.EMBEDDING_MODEL)
        retriever.save(config.INDEX_PATH, config.CORPUS_PATH)

    # ── Build bridge agent (used only for bridge questions) ────────────────
    graph = build_graph(
        model, tokenizer, device, retriever,
        top_k=args.top_k,
        faithfulness_threshold=config.FAITHFULNESS_THRESHOLD,
    )

    # ── Eval loop ──────────────────────────────────────────────────────────
    results = []
    for item in tqdm(items, desc="Evaluating"):
        qtype = item.get("type", "bridge")
        if qtype == "comparison":
            row = run_comparison_naive_rag(
                model, tokenizer, device, retriever, item, args.top_k
            )
        else:
            row = run_bridge_agent(graph, item)
        results.append(row)

    # ── Summary ────────────────────────────────────────────────────────────
    summary = summarize(results)

    n_naive_rag = sum(1 for r in results if r["routing"] == "naive_rag")
    n_agent     = sum(1 for r in results if r["routing"] == "agent")
    n_retried   = sum(1 for r in results if r.get("retry_count", 0) > 0)
    n_web       = sum(1 for r in results if r.get("retrieval_mode") == "web_search")

    print_summary(summary, f"EXP 5 — ADAPTIVE ROUTING  (comparison→naive_rag | bridge→agent)")
    print(f"\n  Routing:")
    print(f"    Naive RAG (comparison) : {n_naive_rag}")
    print(f"    Agent     (bridge)     : {n_agent}")
    print(f"    Agent retried          : {n_retried}/{n_agent}")
    print(f"    Agent web search       : {n_web}/{n_agent}")

    print(f"\n  Comparison vs prior experiments:")
    print(f"  Exp0  (3B base + naive RAG, all)  : EM=0.5680  F1=0.6565  ctx_recall=0.9070")
    print(f"  Exp3  (3B GRPO + agent, all)      : EM=0.5400  F1=0.6250  ctx_recall=0.9020")
    print(f"  Exp4  (3B GRPO + Query LoRA)      : EM=0.5500  F1=0.6392  ctx_recall=0.9310")
    print(f"  Exp5  (adaptive routing, this run): EM={summary['em']:.4f}  F1={summary['f1']:.4f}  ctx_recall={summary['ctx_recall']:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "summary": summary, "per_item": results}, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
