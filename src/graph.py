# src/graph.py
"""
Live multi-hop QA agent — LangGraph StateGraph.

All reasoning (classification, decomposition, rewriting) happens at inference
time using the base model. No pre-computed artifacts.

Graph structure:

  Bridge path:
    START → classify → decompose → retrieve_hop1 → answer_hop1
          → formulate_hop2 → retrieve_hop2 → answer_final
          → verify → [end | retrieve_fallback → answer_final
                           | web_search      → answer_final → end]

  Comparison path:
    START → classify → rewrite → retrieve_comparison → answer_final
          → verify → [end | retrieve_fallback → answer_final
                           | web_search      → answer_final → end]

Usage:
    from src.graph import build_graph
    graph = build_graph(model, tokenizer, device, retriever)
    result = graph.invoke({"question": "Who directed..."})
"""

from __future__ import annotations

import re
import string
from typing import Optional

import torch
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from src.reasoner import Reasoner
from src.retriever import Retriever, union_retrieve


# ---------------------------------------------------------------------------
# hop1 answer cleaning
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class QAState(TypedDict):
    # Input
    question: str

    # Classified at runtime
    qtype: Optional[str]        # "bridge" | "comparison"

    # Bridge intermediates
    sub_q1:       Optional[str]
    hop1_answer:  Optional[str]
    sub_q2:       Optional[str]
    hop1_retrieved: list        # passages from first hop (carried to final answer)

    # Comparison intermediates
    rewritten_queries: list[str]

    # Shared retrieval
    retrieved:      list        # passages used for final answer
    queries_used:   list[str]
    retrieval_mode: str

    # Generation
    prediction: str

    # Verification
    verified:           bool
    retry_count:        int
    uncovered_entities: list  # entity names absent from corpus; empty = all covered


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def make_classify_node(reasoner: Reasoner):
    def classify(state: QAState) -> dict:
        qtype = reasoner.classify(state["question"])
        return {
            "qtype":            qtype,
            "sub_q1":           None,
            "hop1_answer":      None,
            "sub_q2":           None,
            "hop1_retrieved":   [],
            "rewritten_queries": [],
            "retrieved":        [],
            "queries_used":     [],
            "retrieval_mode":   "",
            "prediction":       "",
            "verified":           False,
            "retry_count":        0,
            "uncovered_entities": [],
        }
    return classify


def make_decompose_node(reasoner: Reasoner):
    """Bridge: generate sub_q1 from the original question."""
    def decompose(state: QAState) -> dict:
        sub_q1 = reasoner.decompose_bridge(state["question"])
        return {"sub_q1": sub_q1}
    return decompose


def make_retrieve_hop1_node(retriever: Retriever, top_k: int):
    """Bridge: retrieve passages for sub_q1."""
    def retrieve_hop1(state: QAState) -> dict:
        passages = retriever.retrieve_with_meta(state["sub_q1"], top_k=top_k)
        return {
            "hop1_retrieved": passages,
            "queries_used":   [state["sub_q1"]],
        }
    return retrieve_hop1


def make_answer_hop1_node(reasoner: Reasoner):
    """Bridge: answer sub_q1 using base model (intermediate hop)."""
    def answer_hop1(state: QAState) -> dict:
        texts = [f"{p['title']}\n{p['text']}" for p in state["hop1_retrieved"]]
        hop1_answer = reasoner.answer_hop1(state["sub_q1"], texts)
        return {"hop1_answer": hop1_answer}
    return answer_hop1


def make_formulate_hop2_node(reasoner: Reasoner):
    """Bridge: formulate sub_q2 by inserting hop1_answer into the question."""
    def formulate_hop2(state: QAState) -> dict:
        sub_q2 = reasoner.formulate_hop2(state["question"], state["hop1_answer"])
        return {"sub_q2": sub_q2}
    return formulate_hop2


def make_retrieve_hop2_node(retriever: Retriever, top_k: int):
    """
    Bridge: retrieve passages for sub_q2.
    Merges hop1_retrieved + hop2 passages for the final answer.
    """
    def retrieve_hop2(state: QAState) -> dict:
        hop2_passages = retriever.retrieve_with_meta(state["sub_q2"], top_k=top_k)
        # Merge hop1 + hop2 passages, deduped by title
        seen = {p["title"] for p in state["hop1_retrieved"]}
        merged = list(state["hop1_retrieved"])
        for p in hop2_passages:
            if p["title"] not in seen:
                seen.add(p["title"])
                merged.append(p)
        # For bridge, sub_q2 already has the hop1 answer embedded — check
        # if the merged passages cover the entity we're looking for in hop2
        entities  = _extract_entities_from_queries([state["sub_q2"]])
        uncovered = _find_uncovered_entities(entities, merged)
        return {
            "retrieved":           merged,
            "queries_used":        state["queries_used"] + [state["sub_q2"]],
            "retrieval_mode":      "bridge_decompose",
            "uncovered_entities":  uncovered,
        }
    return retrieve_hop2


def make_rewrite_node(reasoner: Reasoner):
    """Comparison: rewrite question into union sub-queries."""
    def rewrite(state: QAState) -> dict:
        queries = reasoner.rewrite_comparison(state["question"])
        return {"rewritten_queries": queries}
    return rewrite


def make_retrieve_comparison_node(retriever: Retriever, top_k: int):
    """Comparison: union retrieval over rewritten queries."""
    def retrieve_comparison(state: QAState) -> dict:
        passages  = union_retrieve(retriever, state["rewritten_queries"], top_k)
        entities  = _extract_entities_from_queries(state["rewritten_queries"])
        uncovered = _find_uncovered_entities(entities, passages)
        return {
            "retrieved":           passages,
            "queries_used":        state["rewritten_queries"],
            "retrieval_mode":      "comparison_union",
            "uncovered_entities":  uncovered,
        }
    return retrieve_comparison


def _normalize_answer(text: str) -> str:
    """Lowercase, strip articles and punctuation for programmatic comparison."""
    import string as _string
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", _string.punctuation))
    return " ".join(text.split())


def _compare_answers(vals: list) -> str:
    """Return 'yes' if all normalised values match, 'no' otherwise."""
    if not vals:
        return "no"
    if all(v == vals[0] for v in vals):
        return "yes"
    # Allow substring match (e.g. "american" inside "american filmmaker")
    if any(vals[0] in v or v in vals[0] for v in vals[1:]):
        return "yes"
    return "no"


def make_answer_final_node(model, tokenizer, device: str, reasoner: "Reasoner"):
    """
    Final answer synthesis.

    Bridge     : GRPO-tuned adapter generates the answer from retrieved passages.
    Comparison : Two-step approach —
                 (1) answer each rewritten sub-query independently using the base
                     model (adapter disabled), extracting the property value per entity;
                 (2) compare the extracted values programmatically.
                 This avoids asking the small model to do abstract semantic comparison
                 in a single pass, which it reliably fails at.
    """
    def answer_final(state: QAState) -> dict:
        if state.get("qtype") == "comparison" and state.get("rewritten_queries"):
            passage_texts = [f"{p['title']}\n{p['text']}" for p in state["retrieved"]]
            vals = []
            for q in state["rewritten_queries"]:
                ans = reasoner.answer_hop1(q, passage_texts)
                vals.append(_normalize_answer(ans))
            prediction = _compare_answers(vals)
            return {"prediction": prediction}

        # Bridge: GRPO adapter
        ctx = "\n\n".join(
            f"[{i+1}] {p['title']}\n{p['text']}"
            for i, p in enumerate(state["retrieved"])
        )
        user_content = (
            "Answer the question using the passages below. "
            "The passages may contain intermediate context — answer the original question "
            "directly, not any intermediate entity mentioned in the passages. "
            "Give a short answer phrase only — no explanation.\n\n"
            f"{ctx}\n\nQuestion: {state['question']}"
        )
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs    = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prediction = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        return {"prediction": prediction}

    return answer_final


_YES_NO = {"yes", "no", "true", "false"}


def make_verify_node(threshold: float = 0.3):
    """
    Token-overlap faithfulness check: fraction of answer tokens present in
    retrieved context. Retries only when the answer is almost entirely ungrounded.

    For yes/no answers, token-overlap is meaningless ("yes"/"no" never appear
    in Wikipedia passages). Instead we check entity coverage: do the retrieved
    passages mention the named entities from the question? If not, the retrieval
    failed and we should retry rather than blindly trusting the answer.
    """
    def verify(state: QAState) -> dict:
        if not state["prediction"] or not state["retrieved"]:
            return {"verified": False}

        pred_tokens = _normalize(state["prediction"]).split()
        if not pred_tokens:
            return {"verified": False}

        context_text = " ".join(
            p.get("title", "") + " " + p.get("text", "")
            for p in state["retrieved"]
        ).lower()

        # Yes/no: trust only if retrieved passages mention question entities
        if set(pred_tokens) <= _YES_NO:
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', state["question"])
            if not entities:
                return {"verified": True}
            covered = sum(1 for e in entities if e.lower() in context_text)
            return {"verified": covered / len(entities) >= 0.5}

        context_tokens = set(_normalize(context_text).split())
        grounded = sum(1 for t in pred_tokens if t in context_tokens)
        return {"verified": grounded / len(pred_tokens) >= threshold}

    return verify


def make_retrieve_fallback_node(retriever: Retriever, top_k: int):
    """Retry retrieval with the original question when verification fails."""
    def retrieve_fallback(state: QAState) -> dict:
        passages = retriever.retrieve_with_meta(state["question"], top_k=top_k)
        return {
            "retrieved":      passages,
            "queries_used":   state["queries_used"] + [f"[fallback] {state['question']}"],
            "retrieval_mode": "fallback",
            "retry_count":    state["retry_count"] + 1,
        }
    return retrieve_fallback


def make_web_search_node(top_k: int = 3):
    """
    Wikipedia fallback via MediaWiki API (search + extract).

    Entity names are extracted from rewritten_queries (comparison) or
    sub_q1/sub_q2 (bridge) and looked up as Wikipedia article titles.
    Uses verify=False to work around LibreSSL limitations on macOS.
    """
    import warnings
    import requests
    warnings.filterwarnings("ignore", category=Warning, module="urllib3")

    WIKI_HEADERS = {"User-Agent": "hotpotqa-research/1.0 (academic research; contact@example.com)"}

    def _extract_entities(text: str) -> list[str]:
        """Extract capitalised multi-word names from a string."""
        tokens = text.split()
        entities, buf = [], []
        stopwords = {"what", "who", "where", "when", "which", "how", "were",
                     "was", "did", "do", "is", "are", "the", "a", "an", "of",
                     "in", "on", "at", "to", "for", "and", "or", "both", "same"}
        for tok in tokens:
            clean = tok.strip("?.,\"'")
            if clean and clean[0].isupper() and clean.isalpha():
                buf.append(clean)
            else:
                if buf:
                    entities.append(" ".join(buf))
                buf = []
        if buf:
            entities.append(" ".join(buf))
        return [e for e in entities if e.lower() not in stopwords and len(e) > 2]

    def _fetch_summary(title: str):
        """Fetch Wikipedia intro extract via MediaWiki search + extract API."""
        try:
            r1 = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": title,
                        "srlimit": 1, "format": "json"},
                headers=WIKI_HEADERS, verify=False, timeout=10,
            )
            hits = r1.json().get("query", {}).get("search", [])
            if not hits:
                return None
            pageid = hits[0]["pageid"]
            r2 = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "pageids": pageid, "prop": "extracts",
                        "exintro": True, "explaintext": True, "format": "json"},
                headers=WIKI_HEADERS, verify=False, timeout=10,
            )
            pages = r2.json().get("query", {}).get("pages", {})
            for p in pages.values():
                text = p.get("extract", "").strip()
                if text:
                    return {"title": p["title"], "text": text[:600]}
        except Exception:
            pass
        return None

    def _collect_search_terms(state: QAState) -> list[str]:
        """Build a list of entity-name candidates to look up on Wikipedia."""
        candidates = []
        # Comparison: extract from each rewritten query
        for q in state.get("rewritten_queries", []):
            candidates.extend(_extract_entities(q))
        # Bridge: extract from sub-questions
        for field in ("sub_q1", "sub_q2"):
            val = state.get(field)
            if val:
                candidates.extend(_extract_entities(val))
        # Fallback: original question
        if not candidates:
            candidates.extend(_extract_entities(state["question"]))
        # Deduplicate while preserving order
        seen, unique = set(), []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique[:top_k * 2]

    def web_search(state: QAState) -> dict:
        # Prefer the targeted uncovered entity list if available;
        # otherwise fall back to extracting from queries/sub-questions
        uncovered = state.get("uncovered_entities") or []
        terms = uncovered if uncovered else _collect_search_terms(state)

        new_passages, existing_titles = [], {p["title"] for p in state["retrieved"]}
        for term in terms:
            result = _fetch_summary(term)
            if result and result["title"] not in existing_titles:
                existing_titles.add(result["title"])
                new_passages.append(result)

        if uncovered and new_passages:
            # Partial coverage: keep corpus passages (they covered some entities)
            # and add web passages for the ones that were missing
            combined = new_passages + state["retrieved"]
        else:
            # Full corpus miss: use only web passages to avoid noise
            combined = new_passages if new_passages else state["retrieved"]
        return {
            "retrieved":          combined,
            "queries_used":       state["queries_used"] + [f"[wiki] {state['question']}"],
            "retrieval_mode":     "web_search",
            "retry_count":        state["retry_count"] + 1,
            "uncovered_entities": [],  # cleared — web search has handled them
        }
    return web_search


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_by_type(state: QAState) -> str:
    return "bridge" if state["qtype"] == "bridge" else "comparison"


def _find_uncovered_entities(entities: list, passages: list) -> list:
    """
    Return the subset of entity names that do not appear in any retrieved
    passage (title or text). Empty list means all entities are covered.
    """
    if not entities:
        return []
    context = " ".join(
        p.get("title", "") + " " + p.get("text", "") for p in passages
    ).lower()
    return [e for e in entities if e.lower() not in context]


def _extract_entities_from_queries(queries: list) -> list:
    """
    Extract the main named entity from each rewritten query.
    e.g. "What is the nationality of Scott Derrickson?" → "Scott Derrickson"
    """
    stopwords = {"what", "who", "where", "when", "which", "how", "were", "was",
                 "did", "do", "is", "are", "the", "a", "an", "of", "in", "on",
                 "at", "to", "for", "and", "or", "both", "same", "first", "last"}
    entities = []
    for q in queries:
        tokens = q.split()
        buf = []
        for tok in tokens:
            clean = tok.strip("?.,\"'")
            if clean and clean[0].isupper() and clean.isalpha() and clean.lower() not in stopwords:
                buf.append(clean)
            else:
                if buf:
                    entities.append(" ".join(buf))
                buf = []
        if buf:
            entities.append(" ".join(buf))
    # Deduplicate
    seen, unique = set(), []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique


def should_retry(state: QAState) -> str:
    if state["verified"]:
        return "end"
    if state["retry_count"] >= 2:
        return "end"
    # Entities missing from corpus — skip local retry, go straight to web search
    if state.get("uncovered_entities") and state["retry_count"] == 0:
        return "web_retry"
    if state["retry_count"] == 0:
        return "local_retry"
    return "web_retry"  # retry_count == 1


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    model,
    tokenizer,
    device: str,
    retriever: Retriever,
    top_k: int = 5,
    faithfulness_threshold: float = 0.3,
):
    """
    Compile and return the live-inference LangGraph QA pipeline.

    Parameters
    ----------
    model      : PeftModel (Qwen2.5-3B-Instruct + GRPO LoRA adapter)
    tokenizer  : corresponding AutoTokenizer
    device     : "cuda" | "mps" | "cpu"
    retriever  : Retriever (FAISS-backed)
    top_k      : passages per sub-query
    faithfulness_threshold : verify threshold
    """
    reasoner = Reasoner(model, tokenizer, device)

    g = StateGraph(QAState)

    # Register nodes
    g.add_node("classify",            make_classify_node(reasoner))
    g.add_node("decompose",           make_decompose_node(reasoner))
    g.add_node("retrieve_hop1",       make_retrieve_hop1_node(retriever, top_k))
    g.add_node("answer_hop1",         make_answer_hop1_node(reasoner))
    g.add_node("formulate_hop2",      make_formulate_hop2_node(reasoner))
    g.add_node("retrieve_hop2",       make_retrieve_hop2_node(retriever, top_k))
    g.add_node("rewrite",             make_rewrite_node(reasoner))
    g.add_node("retrieve_comparison", make_retrieve_comparison_node(retriever, top_k))
    g.add_node("answer_final",        make_answer_final_node(model, tokenizer, device, reasoner))
    g.add_node("verify",              make_verify_node(faithfulness_threshold))
    g.add_node("retrieve_fallback",   make_retrieve_fallback_node(retriever, top_k))
    g.add_node("web_search",          make_web_search_node(top_k=3))

    # START → classify → branch by type
    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        route_by_type,
        {"bridge": "decompose", "comparison": "rewrite"},
    )

    # Bridge path
    g.add_edge("decompose",      "retrieve_hop1")
    g.add_edge("retrieve_hop1",  "answer_hop1")
    g.add_edge("answer_hop1",    "formulate_hop2")
    g.add_edge("formulate_hop2", "retrieve_hop2")
    g.add_edge("retrieve_hop2",  "answer_final")

    # Comparison path
    g.add_edge("rewrite",             "retrieve_comparison")
    g.add_edge("retrieve_comparison", "answer_final")

    # Verification and retry loop (shared by both paths)
    g.add_edge("answer_final", "verify")
    g.add_conditional_edges(
        "verify",
        should_retry,
        {"local_retry": "retrieve_fallback", "web_retry": "web_search", "end": END},
    )
    g.add_edge("retrieve_fallback", "answer_final")
    g.add_edge("web_search",        "answer_final")

    return g.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())
