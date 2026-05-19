#!/usr/bin/env python3
"""
serve.py — Local demo UI for the HotpotQA multi-hop agent.

Starts a Gradio web app at http://localhost:7860.
Shows reasoning steps in real-time as the agent processes the question,
and displays the context passages used to generate the answer.

Usage:
    pip install gradio
    python serve.py                # builds FAISS index on first run
    python serve.py --load-index   # reuse saved index (faster start)
    python serve.py --port 8080    # custom port
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

import config
from src.models import load_model_and_tokenizer
from src.retriever import Retriever
from src.graph import build_graph


# ---------------------------------------------------------------------------
# Data helpers (corpus for FAISS index)
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _build_corpus(items: list) -> list:
    seen, corpus = set(), []
    for item in items:
        for p in item.get("paragraphs", []):
            if p["title"] not in seen:
                seen.add(p["title"])
                corpus.append({"title": p["title"], "text": p["text"]})
    return corpus


# ---------------------------------------------------------------------------
# Step formatting
# ---------------------------------------------------------------------------

_NODE_LABELS = {
    "classify":            ("Classify question",     "Determining question type"),
    "decompose":           ("Decompose",             "Breaking into sub-questions"),
    "retrieve_hop1":       ("Retrieve — hop 1",      "Searching inner vector database"),
    "answer_hop1":         ("Answer — hop 1",        "Finding intermediate answer"),
    "formulate_hop2":      ("Formulate — hop 2",     "Building second sub-question"),
    "retrieve_hop2":       ("Retrieve — hop 2",      "Searching inner vector database"),
    "rewrite":             ("Rewrite queries",        "Rewriting into sub-queries"),
    "retrieve_comparison": ("Retrieve",              "Searching inner vector database"),
    "answer_final":        ("Synthesise answer",     "Generating final answer"),
    "verify":              ("Verify",                "Checking answer faithfulness"),
    "retrieve_fallback":   ("Retry retrieval",       "Re-retrieving with original question"),
    "web_search":          ("Web search",            "Fetching from Wikipedia"),
}


def _format_step(node: str, update: dict) -> str:
    label, desc = _NODE_LABELS.get(node, (node, ""))
    detail = ""

    if node == "classify":
        qtype = update.get("qtype", "")
        detail = f"→ **{qtype}**"

    elif node == "decompose":
        sub_q1 = update.get("sub_q1") or ""
        detail = f'→ "{sub_q1}"'

    elif node == "rewrite":
        queries = update.get("rewritten_queries", [])
        detail = "  \n".join(f'  - "{q}"' for q in queries)

    elif node == "retrieve_hop1":
        n = len(update.get("hop1_retrieved", []))
        detail = f"→ {n} passages  _(inner vector DB)_"

    elif node in ("retrieve_hop2", "retrieve_comparison", "retrieve_fallback"):
        n = len(update.get("retrieved", []))
        detail = f"→ {n} passages  _(inner vector DB)_"

    elif node == "web_search":
        n = len(update.get("retrieved", []))
        detail = f"→ {n} passages  _(Wikipedia web search)_"

    elif node == "answer_hop1":
        hop1 = update.get("hop1_answer") or ""
        detail = f"→ **{hop1}**"

    elif node == "formulate_hop2":
        sub_q2 = update.get("sub_q2") or ""
        detail = f'→ "{sub_q2}"'

    elif node == "answer_final":
        pred = update.get("prediction") or ""
        detail = f"→ **{pred}**"

    elif node == "verify":
        ok = update.get("verified", False)
        detail = "→ ✓ verified" if ok else "→ ✗ not verified — will retry"

    return f"- **{label}**: {desc}  {detail}\n"


_STOPWORDS = {
    "what", "who", "where", "when", "which", "how", "were", "was", "did",
    "do", "is", "are", "the", "a", "an", "of", "in", "on", "at", "to",
    "for", "and", "or", "both", "same", "have", "has", "had", "that",
    "this", "with", "from", "by", "about",
}

def _keywords_from_queries(queries: list[str]) -> set[str]:
    """Extract meaningful content words from a list of query strings."""
    words = set()
    for q in queries:
        for w in re.findall(r'\b[a-zA-Z]{3,}\b', q):
            if w.lower() not in _STOPWORDS:
                words.add(w.lower())
    return words


def _highlight(text: str, keywords: set[str]) -> str:
    """Wrap keyword occurrences in <mark> tags for HTML highlighting."""
    if not keywords:
        return text
    pattern = r'\b(' + '|'.join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)) + r')\b'
    return re.sub(pattern, r'<mark style="background:#ffe066;padding:0 2px;border-radius:3px">\1</mark>', text, flags=re.IGNORECASE)


def _keyword_hit_count(text: str, title: str, keywords: set[str]) -> int:
    """Count how many distinct keywords appear in title+text (case-insensitive)."""
    haystack = (title + " " + text).lower()
    return sum(1 for k in keywords if k in haystack)


def _source_label(retrieval_mode: str) -> str:
    if retrieval_mode == "web_search":
        return "Wikipedia Web Search"
    if retrieval_mode == "fallback":
        return "Inner Vector Database (fallback retrieval)"
    return "Inner Vector Database"


# ---------------------------------------------------------------------------
# Agent streaming runner
# ---------------------------------------------------------------------------

_graph = None   # set in main()


def run_agent(question: str):
    """
    Generator that yields (steps_md, context_md, answer_str) after each node.
    Used by Gradio's streaming output.
    """
    if not question.strip():
        yield "Ask a question above.", "", ""
        return

    steps_lines: list[str] = []
    final_state: dict = {}

    # Yield immediately so the UI shows "Starting..."
    yield "_Thinking..._", "", ""

    for event in _graph.stream({"question": question}, stream_mode="updates"):
        node   = list(event.keys())[0]
        update = event[node]
        final_state.update(update)

        steps_lines.append(_format_step(node, update))
        yield "\n".join(steps_lines), "", ""

    # Build final outputs
    retrieved = final_state.get("retrieved", [])
    mode      = final_state.get("retrieval_mode", "")
    answer    = final_state.get("prediction", "")
    verified  = final_state.get("verified", False)

    # Extract keywords from all queries used during this run
    all_queries = (
        final_state.get("queries_used", []) +
        [q for q in [final_state.get("sub_q1"), final_state.get("sub_q2")] if q] +
        final_state.get("rewritten_queries", []) +
        [question]
    )
    keywords = _keywords_from_queries(all_queries)

    # Context panel — filter irrelevant passages and highlight keywords
    SCORE_THRESHOLD   = 0.3   # cosine similarity cutoff for FAISS passages
    WEB_MIN_KW_HITS   = 1     # web search passages must match at least N keywords
    is_web = (mode == "web_search")
    source    = _source_label(mode)
    ctx_lines = [f"**Source: {source}**\n"]
    shown = 0
    for p in retrieved:
        title = p.get("title", "")
        raw_text = p.get("text", "")
        score = p.get("score")   # None for web search passages

        if is_web or score is None:
            # No cosine score — filter by keyword overlap instead
            if keywords and _keyword_hit_count(raw_text, title, keywords) < WEB_MIN_KW_HITS:
                continue
            score_label = ""
        else:
            if score < SCORE_THRESHOLD:
                continue
            score_label = f" · <small>score: {score:.2f}</small>"

        shown += 1
        if shown > 6:
            break
        text = _highlight(raw_text[:300].strip(), keywords)
        ctx_lines.append(f"**[{shown}] {title}**{score_label}  \n{text}...\n")

    if shown == 0:
        ctx_lines.append("_No relevant passages found._\n")
    context_md = "\n".join(ctx_lines)

    # Final step — add summary line
    check = "✓" if verified else ""
    steps_lines.append(f"\n---\n**Answer: {answer}** {check}")

    yield "\n".join(steps_lines), context_md, answer


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_EXAMPLES = [
    "What is the capital of the country where the Big Ben is located?", #bridge wikipedia search
    "Were Scott Derrickson and Ed Wood of the same nationality?", #compare wiki search 
    "Are both The Bridge and Grand Canyon documentary films?", #compare, wiki search
    "Are both Subak and Caral originated from the same country?", # compare, vector database
    "What media organization which posts talks is Jesse Dylan a member of?", #bridge, vector database
    "Which is the largest and most populous city in the region with the national anthem Hoyamal?" # bridge, vector database
]

_CSS = """
.answer-text textarea { font-size: 1.3em !important; font-weight: 600 !important; }
.step-panel { background: #f8f9fa; border-radius: 8px; padding: 12px; }
.ctx-panel  { background: #f0f4ff; border-radius: 8px; padding: 12px; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="HotpotQA Agent", theme=gr.themes.Soft(), css=_CSS) as demo:

        gr.Markdown(
            "# HotpotQA Multi-Hop Agent\n"
            "Powered by **Qwen2.5-3B + GRPO** · LangGraph agentic pipeline · "
            "Dense FAISS retrieval + Wikipedia fallback"
        )

        with gr.Row():
            q_box = gr.Textbox(
                label="Question",
                placeholder="e.g. What nationality is the actress who starred in Pretty Woman?",
                lines=2,
                scale=6,
            )
            ask_btn = gr.Button("Ask", variant="primary", scale=1, min_width=80)

        gr.Examples(examples=_EXAMPLES, inputs=q_box, label="Examples")

        with gr.Row(equal_height=True):
            steps_box = gr.Markdown(
                value="Steps will appear here as the agent reasons.",
                label="Reasoning Steps",
                elem_classes="step-panel",
            )
            ctx_box = gr.Markdown(
                value="Context passages will appear here.",
                label="Context Used",
                elem_classes="ctx-panel",
            )

        ans_box = gr.Textbox(
            label="Final Answer",
            interactive=False,
            elem_classes="answer-text",
        )

        # Wire up both button click and Enter key
        for trigger in (ask_btn.click, q_box.submit):
            trigger(
                fn=run_agent,
                inputs=q_box,
                outputs=[steps_box, ctx_box, ans_box],
            )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _graph

    parser = argparse.ArgumentParser()
    parser.add_argument("--load-index", action="store_true",
                        help="Load pre-built FAISS index (skip rebuild)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer(
        config.MODEL_NAME, config.GRPO_ADAPTER_REPO
    )

    print("Loading retriever...")
    if args.load_index:
        retriever = Retriever.load(
            config.INDEX_PATH, config.CORPUS_PATH, config.EMBEDDING_MODEL
        )
    else:
        items     = _load_jsonl(config.EVAL_PATH)
        corpus    = _build_corpus(items)
        retriever = Retriever(corpus, config.EMBEDDING_MODEL)
        retriever.save(config.INDEX_PATH, config.CORPUS_PATH)

    print("Compiling graph...")
    _graph = build_graph(
        model, tokenizer, device, retriever,
        top_k=config.TOP_K,
        faithfulness_threshold=config.FAITHFULNESS_THRESHOLD,
    )

    print(f"\nReady — open http://localhost:{args.port}\n")
    build_ui().launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
