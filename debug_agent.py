#!/usr/bin/env python3
"""
debug_agent.py — step-by-step trace of a single question through the agent.

Shows at each stage:
  1. Classification result
  2. Rewritten queries (comparison) or sub-questions (bridge)
  3. Corpus retrieval results (top passages per query)
  4. Hop-1 answer (bridge only)
  5. Web search results (Wikipedia passages)
  6. Final answer from GRPO model
  7. Verify decision (faithfulness / entity coverage)

Usage:
    python3 debug_agent.py --question "Were Scott Derrickson and Ed Wood of the same nationality?"
    python3 debug_agent.py --load-index --question "..."
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch

import config
from src.models import load_model_and_tokenizer
from src.retriever import Retriever, union_retrieve
from src.reasoner import Reasoner, _looks_like_comparison
from src.graph import _extract_entities_from_queries, _find_uncovered_entities

SEP = "─" * 70


import requests as _requests
import warnings as _warnings
_warnings.filterwarnings("ignore", category=Warning, module="urllib3")

_WIKI_HEADERS = {"User-Agent": "hotpotqa-research/1.0 (academic research; contact@example.com)"}


def wiki_fetch_summary(title: str):
    """Fetch Wikipedia summary via MediaWiki API (search + extract)."""
    try:
        r1 = _requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": title,
                    "srlimit": 1, "format": "json"},
            headers=_WIKI_HEADERS, verify=False, timeout=10,
        )
        hits = r1.json().get("query", {}).get("search", [])
        if not hits:
            print(f"  [wiki] no results for: {title!r}")
            return None
        pageid = hits[0]["pageid"]
        r2 = _requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "pageids": pageid, "prop": "extracts",
                    "exintro": True, "explaintext": True, "format": "json"},
            headers=_WIKI_HEADERS, verify=False, timeout=10,
        )
        pages = r2.json().get("query", {}).get("pages", {})
        for p in pages.values():
            text = p.get("extract", "").strip()
            if text:
                print(f"  [wiki] found: {p['title']!r}")
                return {"title": p["title"], "text": text[:600]}
    except Exception as e:
        print(f"  [wiki ERROR] {type(e).__name__}: {e}")
    return None


def wiki_search_entities(entities: list[str]) -> list[dict]:
    """Fetch Wikipedia summaries for a list of entity names."""
    passages, seen = [], set()
    for entity in entities:
        result = wiki_fetch_summary(entity)
        if result and result["title"] not in seen:
            seen.add(result["title"])
            passages.append(result)
    return passages


def _normalize_answer(text: str) -> str:
    """Lowercase, strip articles and punctuation for comparison."""
    import string
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def print_passages(passages: list[dict], label: str):
    print(f"\n{label} ({len(passages)} passages):")
    for i, p in enumerate(passages, 1):
        print(f"  [{i}] {p['title']}")
        print(f"      {p['text'][:200].strip()}...")
    if not passages:
        print("  (none)")


def answer_with_model(model, tokenizer, device, question, passages, instruction=None, print_prompt=False, use_adapter=True, question_first=False):
    ctx = "\n\n".join(f"[{i+1}] {p['title']}\n{p['text']}" for i, p in enumerate(passages))
    if instruction is None:
        instruction = "Answer the question using the passages below. Give a short answer phrase only — no explanation."
    if question_first:
        content = f"Question: {question}\n\n{instruction}\n\n{ctx}\n\nAnswer:"
    else:
        content = f"{instruction}\n\n{ctx}\n\nQuestion: {question}"
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if print_prompt:
        print(f"\n{'─'*30} FULL PROMPT {'─'*30}\n{prompt}\n{'─'*70}")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        if use_adapter:
            outputs = model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            with model.disable_adapter():
                outputs = model.generate(
                    **inputs, max_new_tokens=64, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--load-index", action="store_true")
    parser.add_argument("--corpus", default=config.CORPUS_PATH)
    parser.add_argument("--adapter", default=config.GRPO_ADAPTER_REPO)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    print(f"\n{SEP}")
    print(f"Question: {args.question}")
    print(SEP)

    # --- Load model ---
    model, tokenizer, device = load_model_and_tokenizer(config.MODEL_NAME, args.adapter)
    reasoner = Reasoner(model, tokenizer, device)

    # --- Load retriever ---
    if args.load_index:
        retriever = Retriever.load(config.INDEX_PATH, config.CORPUS_PATH, config.EMBEDDING_MODEL)
    else:
        with open(args.corpus) as f:
            paragraphs = [json.loads(l) for l in f if l.strip()]
        retriever = Retriever(paragraphs, config.EMBEDDING_MODEL)

    # ── STEP 1: Classify ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 1 — Classification")
    heuristic_fired = _looks_like_comparison(args.question)
    qtype = reasoner.classify(args.question)
    print(f"  Heuristic fired : {heuristic_fired}")
    print(f"  Type            : {qtype}")

    # ── STEP 2: Decompose / Rewrite ───────────────────────────────────────
    print(f"\n{SEP}")
    if qtype == "comparison":
        print("STEP 2 — Comparison rewriting")
        queries = reasoner.rewrite_comparison(args.question)
        print(f"  Rewritten queries: {queries}")
    else:
        print("STEP 2 — Bridge decomposition")
        sub_q1 = reasoner.decompose_bridge(args.question)
        print(f"  Sub-q1: {sub_q1}")

    # ── STEP 3: Corpus retrieval ──────────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 3 — Corpus retrieval (FAISS)")
    if qtype == "comparison":
        corpus_passages = union_retrieve(retriever, queries, args.top_k)
        all_entities    = _extract_entities_from_queries(queries)
        print_passages(corpus_passages, "Union retrieval")
    else:
        hop1_passages = retriever.retrieve_with_meta(sub_q1, top_k=args.top_k)
        print_passages(hop1_passages, f"Hop-1 retrieval for: '{sub_q1}'")
        hop1_answer = reasoner.answer_hop1(sub_q1, [f"{p['title']}\n{p['text']}" for p in hop1_passages])
        print(f"\n  Hop-1 answer: {hop1_answer}")
        sub_q2 = reasoner.formulate_hop2(args.question, hop1_answer)
        print(f"  Sub-q2: {sub_q2}")
        hop2_passages = retriever.retrieve_with_meta(sub_q2, top_k=args.top_k)
        print_passages(hop2_passages, f"Hop-2 retrieval for: '{sub_q2}'")
        seen = {p["title"] for p in hop1_passages}
        corpus_passages = hop1_passages + [p for p in hop2_passages if p["title"] not in seen]
        all_entities    = _extract_entities_from_queries([sub_q2])

    # Entity coverage check
    uncovered = _find_uncovered_entities(all_entities, corpus_passages)
    print(f"\n  All entities  : {all_entities}")
    print(f"  Uncovered     : {uncovered if uncovered else '(none — corpus has all entities)'}")

    # ── STEP 4: Answer with corpus passages (only if corpus is useful) ────
    print(f"\n{SEP}")
    if uncovered and len(uncovered) == len(all_entities):
        print("STEP 4 — SKIPPED (corpus has none of the required entities)")
    else:
        print("STEP 4 — GRPO answer (corpus passages only)")
        answer_corpus = answer_with_model(model, tokenizer, device, args.question, corpus_passages)
        print(f"  Answer: {answer_corpus}")

    # ── STEP 5: Web search ────────────────────────────────────────────────
    print(f"\n{SEP}")
    if not uncovered:
        print("STEP 5 — Web search: SKIPPED (all entities found in corpus)")
        web_passages = []
    else:
        print(f"STEP 5 — Web search for uncovered entities: {uncovered}")
        web_passages = wiki_search_entities(uncovered)
        print_passages(web_passages, "Wikipedia results")

    # ── STEP 6: Final answer ──────────────────────────────────────────────
    print(f"\n{SEP}")
    if not web_passages:
        print("STEP 6 — Final answer (corpus only)")
        final_passages = corpus_passages
    elif uncovered and len(uncovered) < len(all_entities):
        print("STEP 6 — Final answer (web passages for missing entities + corpus for covered ones)")
        web_titles     = {p["title"] for p in web_passages}
        final_passages = web_passages + [p for p in corpus_passages if p["title"] not in web_titles]
    else:
        print("STEP 6 — Final answer (web passages only — corpus had nothing relevant)")
        final_passages = web_passages

    if qtype == "comparison":
        instruction = (
            "Answer the question using the passages below. "
            "For questions that compare a property (such as nationality, profession, or birthplace) "
            "across two entities, find that property for each entity in the passages, then answer "
            "'yes' if the property matches for both, or 'no' if it differs. "
            "Give a short answer only — no explanation."
        )
    else:
        instruction = "Answer the question using the passages below. Give a short answer phrase only — no explanation."

    answer_final = answer_with_model(model, tokenizer, device, args.question, final_passages, instruction, print_prompt=True)
    print(f"  Answer (GRPO adapter): {answer_final}")

    # ── STEP 7: Same prompt through base model (no adapter) ───────────────
    print(f"\n{SEP}")
    print("STEP 7 — Same prompt, base model (adapter disabled)")
    answer_base = answer_with_model(model, tokenizer, device, args.question, final_passages, instruction, use_adapter=False)
    print(f"  Answer (base model):   {answer_base}")

    # ── STEP 8: Two-step synthesis ────────────────────────────────────────
    if qtype == "comparison" and final_passages:
        print(f"\n{SEP}")
        print("STEP 8 — Two-step synthesis")
        sub_instruction = "Answer the question using the passages below. Give a short answer phrase only — no explanation."
        sub_answers = []
        for q in queries:
            ans = answer_with_model(model, tokenizer, device, q, final_passages, sub_instruction)
            print(f"  Sub-query : {q}")
            print(f"  Sub-answer: {ans}")
            sub_answers.append((q, ans))

        # Synthesise: programmatic string comparison — no third model call needed.
        # The model extracted the property values; comparing them is trivial code.
        vals = [_normalize_answer(a) for _, a in sub_answers]
        print(f"\n  Normalised values: {vals}")
        if all(v == vals[0] for v in vals):
            answer_two_step = "yes"
        elif any(v in vals[0] or vals[0] in v for v in vals[1:]):
            answer_two_step = "yes"  # one answer is a substring of the other (e.g. "American" / "American filmmaker")
        else:
            answer_two_step = "no"
        print(f"  Answer (two-step): {answer_two_step}")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
