#!/usr/bin/env python3
"""
exp0_3b_naive_rag.py — Experiment 0: 3B base model + Naive single-hop RAG (lower bound)

Establishes the lower bound for the project by running the simplest possible
retrieval-augmented QA:
  1. Retrieve top-K passages using the original question directly (no decomposition)
  2. Feed question + passages to the 3B base model (no adapter, no agent)
  3. Generate a short answer

No LangGraph, no sub-query decomposition, no adapter switching.
This isolates the contribution of the Agent design + GRPO training (Exp3)
and the Query LoRA distillation (Exp4).

Expected result: EM ~0.35-0.45 (well below Exp3 EM=0.540)

Hardware: CPU or 1× GPU (3B bfloat16 ≈ 6 GB VRAM)
Runtime:  ~15-20 minutes on A100

Usage:
    python eval/exp0_3b_naive_rag.py --load-index
    python eval/exp0_3b_naive_rag.py --n 100 --load-index
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
from utils import load_jsonl, build_corpus, get_gold_titles, print_summary, \
                   resolve_index_path, embedding_tag

import config
from src.retriever import Retriever
from src.evaluator import exact_match, token_f1, supporting_fact_recall, \
                          answer_coverage, faithfulness, summarize

RESULTS_DIR = os.path.join(ROOT, "results")


# ---------------------------------------------------------------------------
# Single-step answer generation (no adapter, no agent)
# ---------------------------------------------------------------------------

def generate_answer(model, tokenizer, device: str,
                    question: str, passages: list[str],
                    max_new_tokens: int = 64) -> str:
    """
    Generate a short answer given the question and retrieved passages.
    Uses the same prompt format as src/reasoner.py answer_final for fairness.
    """
    ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
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
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=config.MODEL_NAME,
                        help="Base model (no adapter loaded)")
    parser.add_argument("--embedding",  default=config.EMBEDDING_MODEL,
                        help="SentenceTransformer embedding model for FAISS retrieval")
    parser.add_argument("--eval",       default=config.EVAL_PATH)
    parser.add_argument("--top-k",      type=int, default=config.TOP_K)
    parser.add_argument("--n",          type=int, default=None,
                        help="Evaluate only first N questions")
    parser.add_argument("--load-index", action="store_true",
                        help="Reuse saved FAISS index (path derived from --embedding)")
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.output is None:
        n_tag = f"_n{args.n}" if args.n else "_n500"
        emb_tag = embedding_tag(args.embedding)
        args.output = os.path.join(RESULTS_DIR, f"exp0_3b_naive_rag{emb_tag}{n_tag}.json")

    # ── Load eval data ────────────────────────────────────────────────────
    items = load_jsonl(args.eval)
    if args.n:
        items = items[:args.n]
    print(f"Loaded {len(items)} questions from {args.eval}")

    # ── Load retriever ────────────────────────────────────────────────────
    index_path = resolve_index_path(args.embedding)
    if args.load_index:
        retriever = Retriever.load(
            index_path, config.CORPUS_PATH, args.embedding
        )
    else:
        corpus = build_corpus(items)
        retriever = Retriever(corpus, args.embedding)
        retriever.save(index_path, config.CORPUS_PATH)

    # ── Load base model (NO adapter) ──────────────────────────────────────
    print(f"Loading base model: {args.model}  (no adapter)")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    # ── Eval loop ─────────────────────────────────────────────────────────
    results = []
    for item in tqdm(items, desc="Evaluating"):
        question    = item["question"]
        gold_titles = get_gold_titles(item)

        # Single-hop retrieval: query = original question (no decomposition)
        retrieved_meta = retriever.retrieve_with_meta(question, top_k=args.top_k)
        passages       = [p["text"] for p in retrieved_meta]

        try:
            pred = generate_answer(model, tokenizer, str(device), question, passages)
        except Exception as e:
            print(f"\n[ERROR] {item['id']}: {e}")
            pred = ""

        results.append({
            "id":           item["id"],
            "question":     question,
            "gold":         item["answer"],
            "prediction":   pred,
            "type":         item.get("type"),
            "level":        item.get("level"),
            "em":           exact_match(pred, item["answer"]),
            "f1":           token_f1(pred, item["answer"]),
            "ctx_recall":   supporting_fact_recall(retrieved_meta, gold_titles),
            "ctx_precision": None,
            "ans_coverage": answer_coverage(retrieved_meta, item["answer"]),
            "faithfulness": faithfulness(pred, retrieved_meta),
            "entailment":   None,
            "contradiction": None,
            "outcome":      None,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    summary = summarize(results)
    print_summary(summary, f"EXP 0 — 3B BASE + NAIVE RAG  (top_k={args.top_k})")

    print("\n  Comparison:")
    print(f"  Exp0 (3B base + naive RAG)  : EM={summary['em']:.4f}  F1={summary['f1']:.4f}  ctx_recall={summary['ctx_recall']:.4f}")
    print(f"  Exp3 (3B GRPO + agent)      : EM=0.5400  F1=0.6250  ctx_recall=0.9020")
    print(f"  Exp4 (3B GRPO + Query LoRA) : EM=0.5500  F1=0.6392  ctx_recall=0.9310")

    # ── Save ─────────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "summary": summary, "per_item": results}, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
