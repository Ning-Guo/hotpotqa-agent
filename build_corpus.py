#!/usr/bin/env python3
"""
build_corpus.py — extract unique paragraphs from grpo_val.jsonl into data/corpus.jsonl.

The original grpo_val.jsonl is used directly for evaluation (question, answer, type,
supporting_facts are all there). This script only builds the FAISS retrieval pool.

Usage:
    python3 build_corpus.py
"""

import json
import os

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")

os.makedirs(DATA_DIR, exist_ok=True)

seen_titles = set()
paragraphs  = []

path = os.path.join(DATA_DIR, "grpo_val.jsonl")
with open(path) as f:
    for line in f:
        item = json.loads(line)
        for p in item.get("paragraphs", []):
            if p["title"] not in seen_titles:
                seen_titles.add(p["title"])
                paragraphs.append({"title": p["title"], "text": p["text"]})

with open(CORPUS_PATH, "w") as f:
    for p in paragraphs:
        f.write(json.dumps(p) + "\n")

print(f"Written {len(paragraphs):,} unique paragraphs → {CORPUS_PATH}")
