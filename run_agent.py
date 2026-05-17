#!/usr/bin/env python
"""
run_agent.py — demo entry point for the live multi-hop QA agent.

Usage:
    # Single question (interactive)
    python run_agent.py --question "Who directed the film starring Tom Hanks in Forrest Gump?"

    # Build FAISS index from a HotpotQA-style corpus file, then answer a question
    python run_agent.py --corpus data/corpus.jsonl --question "..."

    # Save index after building (fast startup on subsequent runs)
    python run_agent.py --corpus data/corpus.jsonl --save-index --question "..."

    # Load pre-built index
    python run_agent.py --load-index --question "..."
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import config
from src.models import load_model_and_tokenizer
from src.retriever import Retriever
from src.graph import build_graph


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--question",    type=str, required=True)
    p.add_argument("--corpus",      type=str, default=config.CORPUS_PATH,
                   help="Path to corpus.jsonl (paragraphs with title+text)")
    p.add_argument("--load-index",  action="store_true",
                   help="Load pre-built FAISS index from config.INDEX_PATH")
    p.add_argument("--save-index",  action="store_true",
                   help="Save FAISS index to config.INDEX_PATH after building")
    p.add_argument("--adapter",     type=str, default=config.GRPO_ADAPTER_REPO,
                   help="HF repo ID or local path to GRPO LoRA adapter")
    p.add_argument("--top-k",       type=int, default=config.TOP_K)
    return p.parse_args()


def load_retriever(args) -> Retriever:
    if args.load_index:
        return Retriever.load(config.INDEX_PATH, config.CORPUS_PATH, config.EMBEDDING_MODEL)

    if not os.path.exists(args.corpus):
        sys.exit(f"Corpus file not found: {args.corpus}\n"
                 "Provide --corpus <path> or --load-index to use a pre-built index.")

    with open(args.corpus) as f:
        paragraphs = [json.loads(line) for line in f if line.strip()]

    retriever = Retriever(paragraphs, config.EMBEDDING_MODEL)

    if args.save_index:
        retriever.save(config.INDEX_PATH, config.CORPUS_PATH)

    return retriever


def main():
    args = parse_args()

    # Load model
    model, tokenizer, device = load_model_and_tokenizer(
        config.MODEL_NAME, args.adapter
    )

    # Load / build retriever
    retriever = load_retriever(args)

    # Build graph
    graph = build_graph(
        model, tokenizer, device, retriever,
        top_k=args.top_k,
        faithfulness_threshold=config.FAITHFULNESS_THRESHOLD,
    )

    # Run
    print(f"\nQuestion: {args.question}\n")
    result = graph.invoke({"question": args.question})

    print(f"Type:        {result.get('qtype')}")
    if result.get("qtype") == "bridge":
        print(f"Sub-q1:      {result.get('sub_q1')}")
        print(f"Hop-1 answer:{result.get('hop1_answer')}")
        print(f"Sub-q2:      {result.get('sub_q2')}")
    else:
        print(f"Rewritten:   {result.get('rewritten_queries')}")
    print(f"Mode:        {result.get('retrieval_mode')}")
    print(f"Verified:    {result.get('verified')}")
    print(f"Retries:     {result.get('retry_count')}")
    print(f"\nAnswer: {result.get('prediction')}")


if __name__ == "__main__":
    main()
