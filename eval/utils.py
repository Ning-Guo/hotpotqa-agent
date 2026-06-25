# eval/utils.py
"""
Shared utilities for the three comparison experiments.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager

# Make src/ importable from eval scripts
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_corpus(items: list) -> list:
    """Collect all unique paragraphs from the eval set for FAISS indexing."""
    seen, corpus = set(), []
    for item in items:
        for p in item.get("paragraphs", []):
            if p["title"] not in seen:
                seen.add(p["title"])
                corpus.append({"title": p["title"], "text": p["text"]})
    return corpus


def get_gold_passages(item: dict) -> list:
    """Return only the gold supporting passages for an item (no distractors)."""
    sf = item.get("supporting_facts", {})
    gold_titles = set(sf.get("title", [])) if isinstance(sf, dict) else set()
    return [p for p in item.get("paragraphs", []) if p["title"] in gold_titles]


def get_gold_titles(item: dict) -> list:
    sf = item.get("supporting_facts", {})
    return list(set(sf.get("title", []))) if isinstance(sf, dict) else []


# ---------------------------------------------------------------------------
# PlainModelWrapper
# ---------------------------------------------------------------------------

class PlainModelWrapper:
    """
    Wraps a plain HuggingFace model to be compatible with src.reasoner.Reasoner,
    which calls model.disable_adapter() (a PeftModel API).

    For a base model with no LoRA adapter, disable_adapter() is a no-op:
    both reasoning steps and answer synthesis use the same weights.
    This is correct for a large base model (e.g. 32B) that does not need
    adapter switching.
    """

    def __init__(self, model):
        self._model = model

    @contextmanager
    def disable_adapter(self):
        yield  # no-op — plain model has no adapter

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_summary(summary: dict, label: str) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  {label}  |  n={summary['n']}")
    print(sep)
    print(f"  Exact Match    : {summary['em']:.4f}")
    print(f"  Token F1       : {summary['f1']:.4f}")
    if summary.get("ctx_recall") is not None:
        print(f"  Context Recall : {summary['ctx_recall']:.4f}")
    if summary.get("ans_coverage") is not None:
        print(f"  Ans Coverage   : {summary['ans_coverage']:.4f}")
    if summary.get("faithfulness") is not None:
        print(f"  Faithfulness   : {summary['faithfulness']:.4f}")
    print(f"\n  By question type:")
    for qtype, s in summary.get("by_type", {}).items():
        print(f"    {qtype:<14} EM={s['em']:.4f}  F1={s['f1']:.4f}  (n={s['count']})")
    print(f"\n  By difficulty:")
    for level, s in summary.get("by_level", {}).items():
        print(f"    {level:<14} EM={s['em']:.4f}  F1={s['f1']:.4f}  (n={s['count']})")
    print(sep)
