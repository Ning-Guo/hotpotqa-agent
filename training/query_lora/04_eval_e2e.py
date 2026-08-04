#!/usr/bin/env python3
"""
04_eval_e2e.py — End-to-end agent evaluation with Query LoRA.

Runs the full 500-question LangGraph agent pipeline using:
  - Query LoRA for: decompose (sub_q1), formulate_hop2 (sub_q2), rewrite_comparison
  - GRPO adapter for: answer_final (bridge)
  - Base model for: classify, answer_hop1

Adapter switching per step:
  classify          → disable_adapter() (base model)
  decompose         → set_adapter("query")
  answer_hop1       → disable_adapter() (base model)
  formulate_hop2    → set_adapter("query")
  rewrite           → set_adapter("query")
  answer_final      → set_adapter("grpo")   ← restored after each query step

Compares against Exp3 baseline (results/exp3_3b_grpo_agent_n500.json).

Hardware: 1× A100 40GB
Runtime:  ~20–30 minutes

Usage:
    python training/query_lora/04_eval_e2e.py \
        --query-adapter training/query_lora/checkpoints/query_lora/final \
        --load-index
    python training/query_lora/04_eval_e2e.py \
        --query-adapter training/query_lora/checkpoints/query_lora/final \
        --load-index --n 50  # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(1, ROOT)

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import config as cfg
from src.retriever import Retriever
from src.reasoner import Reasoner
from src.graph import build_graph
from src.evaluator import exact_match, token_f1, supporting_fact_recall, \
                          answer_coverage, faithfulness, summarize


# ---------------------------------------------------------------------------
# QueryReasoner — subclasses Reasoner, activates Query LoRA for rewrite steps
# ---------------------------------------------------------------------------

class QueryReasoner(Reasoner):
    """
    Extends Reasoner to use the Query LoRA adapter for sub_q generation steps.

    Steps that use Query LoRA:  decompose_bridge, formulate_hop2, rewrite_comparison
    Steps that use base model:  classify, answer_hop1  (unchanged from Reasoner)

    After each Query LoRA call, restores GRPO as the active adapter so that
    answer_final (which calls model.generate() directly in graph.py) always
    runs with the GRPO adapter.
    """

    def _generate_with_query_lora(self, prompt: str, max_new_tokens: int = None) -> str:
        n_tokens = max_new_tokens or self.max_new_tokens
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048,
        ).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            # Activate Query LoRA
            self.model.set_adapter("query")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # Restore GRPO so answer_final uses it
            self.model.set_adapter("grpo")

        generated = outputs[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def decompose_bridge(self, question: str) -> str:
        prompt = self._decompose_prompt(question)
        return self._generate_with_query_lora(prompt).strip().strip('"')

    def formulate_hop2(self, original_question: str, hop1_answer: str) -> str:
        prompt = self._hop2_prompt(original_question, hop1_answer)
        return self._generate_with_query_lora(prompt).strip().strip('"')

    def rewrite_comparison(self, question: str) -> list[str]:
        prompt = self._rewrite_prompt(question)
        raw    = self._generate_with_query_lora(prompt, max_new_tokens=80)
        queries = self._parse_rewritten_queries(raw)
        return queries if len(queries) >= 2 else [question]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_corpus(items: list) -> list:
    seen, corpus = set(), []
    for item in items:
        for p in item.get("paragraphs", []):
            if p["title"] not in seen:
                seen.add(p["title"])
                corpus.append({"title": p["title"], "text": p["text"]})
    return corpus


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-adapter", required=True,
                        help="Path to trained Query LoRA adapter")
    parser.add_argument("--grpo-adapter",  default=cfg.GRPO_ADAPTER,
                        help="GRPO adapter repo/path (default: from config)")
    parser.add_argument("--model",         default=cfg.STUDENT_MODEL)
    parser.add_argument("--eval",          default=cfg.EVAL_PATH)
    parser.add_argument("--top-k",         type=int, default=5)
    parser.add_argument("--n",             type=int, default=None)
    parser.add_argument("--load-index",    action="store_true")
    parser.add_argument("--output",        default=None)
    args = parser.parse_args()

    os.makedirs(cfg.EVAL_RESULTS_DIR, exist_ok=True)
    if args.output is None:
        n_tag = f"_n{args.n}" if args.n else "_n500"
        args.output = os.path.join(
            cfg.EVAL_RESULTS_DIR, f"exp3_query_lora{n_tag}.json"
        )

    # ── Load eval data ────────────────────────────────────────────────────────
    items = load_jsonl(args.eval)
    if args.n:
        items = items[:args.n]
    print(f"Loaded {len(items)} questions")

    # ── Load retriever ────────────────────────────────────────────────────────
    if args.load_index:
        retriever = Retriever.load(cfg.INDEX_PATH, cfg.CORPUS_PATH, cfg.EMBEDDING_MODEL)
    else:
        corpus    = build_corpus(items)
        retriever = Retriever(corpus, cfg.EMBEDDING_MODEL)
        retriever.save(cfg.INDEX_PATH, cfg.CORPUS_PATH)

    # ── Load model with two adapters ──────────────────────────────────────────
    print(f"Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load GRPO adapter first
    print(f"Loading GRPO adapter: {args.grpo_adapter}")
    parts = args.grpo_adapter.split("/")
    if not args.grpo_adapter.startswith("/") and len(parts) >= 3:
        repo_id   = "/".join(parts[:2])
        subfolder = "/".join(parts[2:])
        model = PeftModel.from_pretrained(
            base, repo_id, subfolder=subfolder, adapter_name="grpo"
        )
    else:
        model = PeftModel.from_pretrained(base, args.grpo_adapter, adapter_name="grpo")

    # Load Query LoRA adapter
    print(f"Loading Query LoRA adapter: {args.query_adapter}")
    q_parts = args.query_adapter.split("/")
    if not args.query_adapter.startswith("/") and len(q_parts) >= 3:
        q_repo_id   = "/".join(q_parts[:2])
        q_subfolder = "/".join(q_parts[2:])
        model.load_adapter(q_repo_id, subfolder=q_subfolder, adapter_name="query")
    else:
        model.load_adapter(args.query_adapter, adapter_name="query")

    # Set GRPO as the default active adapter (used by answer_final in graph.py)
    model.set_adapter("grpo")
    model.eval()
    print("Both adapters loaded. Default: GRPO.")

    # ── Build graph with QueryReasoner ────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Monkey-patch: build_graph creates a Reasoner internally.
    # We override it here by patching the import in src.graph.
    import src.graph as graph_module
    original_reasoner_cls = graph_module.Reasoner

    # Temporarily replace Reasoner with QueryReasoner in graph module
    graph_module.Reasoner = QueryReasoner
    graph = build_graph(
        model, tokenizer, device, retriever,
        top_k=args.top_k,
        faithfulness_threshold=0.3,
    )
    graph_module.Reasoner = original_reasoner_cls  # restore

    # ── Eval loop ─────────────────────────────────────────────────────────────
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

        pred        = state.get("prediction", "")
        retrieved   = state.get("retrieved", [])
        sf          = item.get("supporting_facts", {})
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

    # ── Summary ───────────────────────────────────────────────────────────────
    summary   = summarize(results)
    n_retried = sum(1 for r in results if r["retry_count"] > 0)
    n_web     = sum(1 for r in results if r["retrieval_mode"] == "web_search")

    sep = "="*65
    print(f"\n{sep}")
    print(f"  EXP3 + QUERY LORA  |  n={summary['n']}")
    print(sep)
    print(f"  Exact Match    : {summary['em']:.4f}   (Exp3 baseline: 0.540)")
    print(f"  Token F1       : {summary['f1']:.4f}   (Exp3 baseline: 0.625)")
    print(f"  Context Recall : {summary['ctx_recall']:.4f}   (Exp3 baseline: 0.902)")
    print(f"  Ans Coverage   : {summary['ans_coverage']:.4f}")
    print(f"\n  Retried        : {n_retried}/{summary['n']}")
    print(f"  Web search     : {n_web}/{summary['n']}")
    print(f"\n  By question type:")
    for qtype, s in summary["by_type"].items():
        print(f"    {qtype:<14} EM={s['em']:.4f}  F1={s['f1']:.4f}  (n={s['count']})")
    print(sep)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump({
            "config":   vars(args),
            "summary":  summary,
            "per_item": results,
        }, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
