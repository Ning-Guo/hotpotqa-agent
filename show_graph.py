#!/usr/bin/env python3
"""
show_graph.py — visualise the LangGraph agent structure without loading the model.

Outputs:
  - ASCII diagram in the terminal
  - Mermaid markdown saved to results/graph.md  (render at mermaid.live)
  - PNG saved to results/graph.png              (if network available)

Usage:
    python3 show_graph.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(__file__))

import config
from src.graph import build_graph

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def make_mock():
    """
    Minimal mock objects that satisfy build_graph's interface.
    No weights are loaded — we only need the graph structure.
    """
    class MockTokenizer:
        def apply_chat_template(self, *a, **kw): return ""
        def __call__(self, *a, **kw):
            m = types.SimpleNamespace()
            m.input_ids = types.SimpleNamespace()
            m.input_ids.shape = (1, 10)
            m.to = lambda d: m
            return m

    class MockModel:
        def eval(self): return self
        def generate(self, **kw): return [[0]*10]
        def disable_adapter(self):
            import contextlib
            return contextlib.nullcontext()

    class MockRetriever:
        def retrieve_with_meta(self, q, top_k=5): return []

    return MockModel(), MockTokenizer(), "cpu", MockRetriever()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Building graph structure (no model weights loaded)...")
    model, tokenizer, device, retriever = make_mock()
    graph = build_graph(model, tokenizer, device, retriever)
    drawable = graph.get_graph()

    # ── ASCII (terminal) ──────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  GRAPH STRUCTURE (ASCII)")
    print("─" * 65)
    drawable.print_ascii()

    # ── Mermaid markdown ──────────────────────────────────────────────────
    mermaid_src = drawable.draw_mermaid()
    md_path = os.path.join(RESULTS_DIR, "graph.md")
    with open(md_path, "w") as f:
        f.write("# Agent Graph\n\n")
        f.write("Render at https://mermaid.live\n\n")
        f.write("```mermaid\n")
        f.write(mermaid_src)
        f.write("\n```\n")
    print(f"\nMermaid markdown saved → {md_path}")
    print("(Paste into https://mermaid.live to render interactively)\n")

    # ── PNG ───────────────────────────────────────────────────────────────
    png_path = os.path.join(RESULTS_DIR, "graph.png")
    try:
        png_bytes = drawable.draw_mermaid_png()
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        print(f"PNG saved → {png_path}")
    except Exception as e:
        print(f"PNG generation failed ({e})")
        print("Use the Mermaid markdown above instead.")


if __name__ == "__main__":
    main()
