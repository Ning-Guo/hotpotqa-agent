#!/usr/bin/env python3
"""
03_eval_subq_quality.py — Intermediate sub_q quality evaluation.

Compares sub_q1 / sub_q2 generation quality between:
  - 3B base model (no adapter, current baseline)
  - 3B + Query LoRA (the adapter we just trained)

Metrics:
  - sub_q1 ctx_recall: does the generated sub_q1 retrieve gold passages?
  - sub_q2 ctx_recall: does the generated sub_q2 retrieve gold passages?
  - sub_q2 entity rate: does sub_q2 contain the hop1_answer entity? (B2 fix check)

This is a fast diagnostic (no full agent run) — runs in ~10 minutes.

Usage:
    python training/query_lora/03_eval_subq_quality.py \
        --adapter training/query_lora/checkpoints/query_lora/final \
        --load-index --n 200
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
from src.evaluator import supporting_fact_recall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def get_gold_titles(item: dict) -> list:
    sf = item.get("supporting_facts", {})
    return list(set(sf.get("title", []))) if isinstance(sf, dict) else []


def hop2_contains_entity(sub_q2: str, hop1_answer: str) -> bool:
    tokens = [w.strip(".,?\"'") for w in hop1_answer.split() if len(w) > 3]
    return any(t.lower() in sub_q2.lower() for t in tokens)


# ---------------------------------------------------------------------------
# Model wrapper supporting two modes: base and query_lora
# ---------------------------------------------------------------------------

class QueryEvaluator:
    """Generates sub_q1 and sub_q2 using either base model or Query LoRA."""

    def __init__(self, model, tokenizer, use_lora: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.use_lora = use_lora
        self.device = next(model.parameters()).device

    def _generate(self, messages: list, max_new_tokens: int = 80) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            if self.use_lora:
                # Query LoRA active
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                # Base model (disable all adapters)
                with self.model.disable_adapter():
                    out = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
                    )
        return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

    def generate_sub_q1(self, question: str) -> str:
        examples = (
            "Q: What nationality is the actress who starred in Pretty Woman?\n"
            "Sub-q1: Who starred in Pretty Woman?\n\n"
            "Q: In which country was the director of Schindler's List born?\n"
            "Sub-q1: Who directed Schindler's List?\n\n"
            "Q: What is the occupation of the spouse of Barack Obama?\n"
            "Sub-q1: Who is the spouse of Barack Obama?\n\n"
            "Q: Lake Hodges is crossed by the major Interstate Highway that begins in what county?\n"
            "Sub-q1: What major Interstate Highway crosses Lake Hodges?\n\n"
            "Q: Who was the personal secretary of the British politician born on September 1, 1931?\n"
            "Sub-q1: Who was the British politician born on September 1, 1931?\n\n"
        )
        content = (
            "Extract the first sub-question from this multi-hop bridge question.\n"
            "The sub-question must identify the INTERMEDIATE entity — not the final answer.\n"
            "Do NOT restate the original question or ask for the final answer directly.\n"
            "Output only the sub-question, no explanation.\n\n"
            f"{examples}"
            f"Q: {question}\n"
            "Sub-q1:"
        )
        return self._generate([{"role": "user", "content": content}]).strip().strip('"')

    def generate_hop1_answer(self, sub_q1: str, passages: list) -> str:
        """Always uses base model (hop1 is reading comprehension, not query rewrite)."""
        ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
        content = (
            "Answer the question using the passages. "
            "Give a short answer phrase only — no explanation.\n\n"
            f"{ctx}\n\nQuestion: {sub_q1}"
        )
        # Force base model for hop1 regardless of use_lora setting
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            with self.model.disable_adapter():
                out = self.model.generate(
                    **inputs, max_new_tokens=48,
                    do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
                )
        return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

    def generate_sub_q2(self, question: str, hop1_answer: str) -> str:
        examples = (
            "Q: What nationality is the actress who starred in Pretty Woman?\n"
            "Hop-1 answer: Julia Roberts\n"
            "Sub-q2: What nationality is Julia Roberts?\n\n"
            "Q: In which country was the director of Schindler's List born?\n"
            "Hop-1 answer: Steven Spielberg\n"
            "Sub-q2: In which country was Steven Spielberg born?\n\n"
            "Q: What is the occupation of the spouse of Barack Obama?\n"
            "Hop-1 answer: Michelle Obama\n"
            "Sub-q2: What is the occupation of Michelle Obama?\n\n"
        )
        content = (
            "Given the original multi-hop question and the answer to the first hop, "
            "write the second sub-question. Replace the indirect reference with the "
            "hop-1 answer. Output only the sub-question.\n\n"
            f"{examples}"
            f"Q: {question}\n"
            f"Hop-1 answer: {hop1_answer}\n"
            "Sub-q2:"
        )
        return self._generate([{"role": "user", "content": content}]).strip().strip('"')


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def eval_bridge(items: list, evaluator: QueryEvaluator, retriever: Retriever) -> dict:
    sub_q1_recalls, sub_q2_recalls, entity_rates = [], [], []

    for item in tqdm(items, desc="  evaluating bridge"):
        question    = item["question"]
        gold_titles = get_gold_titles(item)
        if not gold_titles:
            continue

        # sub_q1
        sub_q1    = evaluator.generate_sub_q1(question)
        retrieved1 = retriever.retrieve_with_meta(sub_q1, top_k=cfg.TOP_K)
        recall1    = supporting_fact_recall(retrieved1, gold_titles)
        sub_q1_recalls.append(recall1)

        # hop1_answer (base model, not query LoRA)
        passages  = [p["text"] for p in retrieved1]
        hop1_ans  = evaluator.generate_hop1_answer(sub_q1, passages)

        # sub_q2
        sub_q2    = evaluator.generate_sub_q2(question, hop1_ans)
        retrieved2 = retriever.retrieve_with_meta(sub_q2, top_k=cfg.TOP_K)
        recall2    = supporting_fact_recall(retrieved2, gold_titles)
        sub_q2_recalls.append(recall2)

        entity_rates.append(float(hop2_contains_entity(sub_q2, hop1_ans)))

    def mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "n":              len(sub_q1_recalls),
        "sub_q1_recall":  mean(sub_q1_recalls),
        "sub_q2_recall":  mean(sub_q2_recalls),
        "sub_q2_entity_rate": mean(entity_rates),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",    required=True,
                        help="Path to trained Query LoRA adapter")
    parser.add_argument("--model",      default=cfg.STUDENT_MODEL)
    parser.add_argument("--eval",       default=cfg.EVAL_PATH,
                        help="Eval JSONL (bridge questions only used here)")
    parser.add_argument("--n",          type=int, default=200,
                        help="Number of bridge questions to evaluate")
    parser.add_argument("--load-index", action="store_true")
    args = parser.parse_args()

    # ── Load retriever ────────────────────────────────────────────────────────
    if args.load_index:
        retriever = Retriever.load(cfg.INDEX_PATH, cfg.CORPUS_PATH, cfg.EMBEDDING_MODEL)
    else:
        raise ValueError("Please build or load a FAISS index (--load-index)")

    # ── Load eval data (bridge only) ──────────────────────────────────────────
    all_items    = load_jsonl(args.eval)
    bridge_items = [i for i in all_items if i.get("type") == "bridge"][:args.n]
    print(f"Evaluating on {len(bridge_items)} bridge questions")

    # ── Load model + Query LoRA ───────────────────────────────────────────────
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # Load Query LoRA adapter
    model = PeftModel.from_pretrained(base, args.adapter, adapter_name="query")
    model.eval()

    # ── Evaluate BASE model ───────────────────────────────────────────────────
    print("\n[1/2] Evaluating BASE model (disable_adapter)...")
    base_evaluator = QueryEvaluator(model, tokenizer, use_lora=False)
    base_results   = eval_bridge(bridge_items, base_evaluator, retriever)

    # ── Evaluate Query LoRA ───────────────────────────────────────────────────
    print("\n[2/2] Evaluating QUERY LoRA...")
    model.set_adapter("query")
    lora_evaluator = QueryEvaluator(model, tokenizer, use_lora=True)
    lora_results   = eval_bridge(bridge_items, lora_evaluator, retriever)

    # ── Print comparison ──────────────────────────────────────────────────────
    sep = "="*65
    print(f"\n{sep}")
    print(f"  SUB_Q QUALITY COMPARISON  |  n={base_results['n']}")
    print(sep)
    print(f"  {'Metric':<25} {'Base 3B':>10} {'Query LoRA':>12} {'Delta':>8}")
    print(f"  {'-'*55}")
    for key, label in [
        ("sub_q1_recall",       "sub_q1 ctx_recall"),
        ("sub_q2_recall",       "sub_q2 ctx_recall"),
        ("sub_q2_entity_rate",  "sub_q2 entity rate (B2)"),
    ]:
        b = base_results[key]
        l = lora_results[key]
        delta = l - b
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<25} {b:>10.4f} {l:>12.4f} {sign+f'{delta:.4f}':>8}")
    print(sep)


if __name__ == "__main__":
    main()
