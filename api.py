#!/usr/bin/env python3
"""
api.py — FastAPI REST server for the HotpotQA multi-hop agent.

Endpoints:
    POST /ask          { "question": "..." } → { "answer", "verified", "qtype",
                                                  "steps", "request_id", "tokens" }
    GET  /health       → { "status": "ok" }
    GET  /metrics      → Prometheus text format (scraped by Prometheus server)

Auth:
    Set API_KEYS env var to a comma-separated list of valid keys.
    Clients must send:  X-Api-Key: <key>
    If API_KEYS is empty, auth is disabled (useful for local dev).

Usage:
    # Minimal (no auth, no Langfuse)
    LOAD_INDEX=1 uvicorn api:app --host 0.0.0.0 --port 8000

    # Production with auth + Langfuse cloud
    API_KEYS=key1,key2 \\
    LANGFUSE_PUBLIC_KEY=pk-lf-... \\
    LANGFUSE_SECRET_KEY=sk-lf-... \\
    LOAD_INDEX=1 uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from prometheus_client import make_asgi_app

import config
from src.models import load_model_and_tokenizer
from src.retriever import Retriever
from src.graph import build_graph

from monitoring.metrics import (
    FALLBACK_COUNT,
    INPUT_TOKENS,
    NODE_LATENCY,
    OUTPUT_TOKENS,
    QUESTION_TYPE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    VERIFY_FAILURES,
    start_gpu_collector,
)
from monitoring.token_context import get as _get_tokens, reset as _reset_tokens
from monitoring.tracer import RequestTrace, init_langfuse


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_graph = None


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
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph

    # Observability setup (both are no-ops if not configured)
    init_langfuse()
    start_gpu_collector(interval=5.0)

    load_index = os.environ.get("LOAD_INDEX", "1") == "1"

    print("Loading model and tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer(
        config.MODEL_NAME, config.GRPO_ADAPTER_REPO
    )

    print("Loading retriever...")
    if load_index and os.path.exists(config.INDEX_PATH):
        retriever = Retriever.load(
            config.INDEX_PATH, config.CORPUS_PATH, config.EMBEDDING_MODEL
        )
    else:
        items = _load_jsonl(config.EVAL_PATH)
        corpus = _build_corpus(items)
        retriever = Retriever(corpus, config.EMBEDDING_MODEL)
        retriever.save(config.INDEX_PATH, config.CORPUS_PATH)

    print("Compiling graph...")
    _graph = build_graph(
        model, tokenizer, device, retriever,
        top_k=config.TOP_K,
        faithfulness_threshold=config.FAITHFULNESS_THRESHOLD,
    )

    print("API ready.")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HotpotQA Multi-Hop Agent",
    description="Qwen2.5-3B + GRPO LoRA, LangGraph pipeline, FAISS retrieval",
    version="1.1.0",
    lifespan=lifespan,
)

# Mount Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

# Comma-separated list of valid API keys, e.g. "key1,key2,key3"
# Empty string = auth disabled
_API_KEYS: set[str] = set(
    k for k in os.environ.get("API_KEYS", "").split(",") if k.strip()
)


async def check_api_key(x_api_key: str = Header(default="")) -> None:
    if _API_KEYS and x_api_key not in _API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str
    tenant_id: str = "default"


class TokenUsage(BaseModel):
    input: int
    output: int


class AnswerResponse(BaseModel):
    answer: str
    verified: bool
    qtype: str
    steps: list[str]
    request_id: str
    tokens: TokenUsage


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _graph is not None}


@app.post("/ask", response_model=AnswerResponse, dependencies=[Depends(check_api_key)])
async def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    if _graph is None:
        raise HTTPException(status_code=503, detail="model not ready")

    request_id = str(uuid.uuid4())
    trace = RequestTrace(
        question=req.question,
        request_id=request_id,
        tenant_id=req.tenant_id,
    )

    t_start = time.perf_counter()
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _run_graph, req.question, trace
        )
        REQUEST_COUNT.labels(status="ok").inc()
        trace.end(output=result, status="OK")
        return result
    except Exception as exc:
        REQUEST_COUNT.labels(status="error").inc()
        trace.end(output={"error": str(exc)}, status="ERROR")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - t_start)


# ---------------------------------------------------------------------------
# Graph runner (executes in a thread-pool thread)
# ---------------------------------------------------------------------------

def _run_graph(question: str, trace: RequestTrace) -> dict:
    # Reset per-request token counter for this thread
    _reset_tokens()

    final_state: dict = {}
    steps: list[str] = []
    prev_time = time.perf_counter()

    for event in _graph.stream({"question": question}, stream_mode="updates"):
        node   = list(event.keys())[0]
        update = event[node]
        now    = time.perf_counter()

        # Per-node wall-clock latency
        NODE_LATENCY.labels(node=node).observe(now - prev_time)
        prev_time = now

        final_state.update(update)
        steps.append(node)

        # Behaviour counters
        if node == "classify":
            qtype = update.get("qtype", "unknown")
            QUESTION_TYPE.labels(qtype=qtype).inc()

        elif node == "verify" and not update.get("verified", False):
            VERIFY_FAILURES.inc()

        elif node == "retrieve_fallback":
            FALLBACK_COUNT.labels(type="local").inc()

        elif node == "web_search":
            FALLBACK_COUNT.labels(type="web").inc()

        # Langfuse span for this node
        span = trace.span(name=node, input_data=_node_input(node, update))
        span.end(output=_node_output(node, update))

    # Flush token counts accumulated by reasoner._generate() calls
    tok = _get_tokens()
    INPUT_TOKENS.inc(tok["input"])
    OUTPUT_TOKENS.inc(tok["output"])

    return {
        "answer":     final_state.get("prediction", ""),
        "verified":   final_state.get("verified", False),
        "qtype":      final_state.get("qtype", ""),
        "steps":      steps,
        "request_id": trace._trace.id if trace._trace else "n/a",
        "tokens":     tok,
    }


# ---------------------------------------------------------------------------
# Helpers: extract compact input/output dicts for Langfuse spans
# ---------------------------------------------------------------------------

def _node_input(node: str, update: dict) -> dict:
    """Return the key input field(s) for a given node update."""
    if node == "classify":
        return {}
    if node == "decompose":
        return {}
    if node == "retrieve_hop1":
        return {"sub_q1": update.get("sub_q1", "")}
    if node == "answer_hop1":
        return {"sub_q1": update.get("sub_q1", "")}
    if node == "formulate_hop2":
        return {"hop1_answer": update.get("hop1_answer", "")}
    if node == "retrieve_hop2":
        return {"sub_q2": update.get("sub_q2", "")}
    if node == "rewrite":
        return {}
    if node == "retrieve_comparison":
        return {"rewritten_queries": update.get("rewritten_queries", [])}
    if node == "answer_final":
        return {"n_passages": len(update.get("retrieved", []))}
    if node == "verify":
        return {"prediction": update.get("prediction", "")}
    return {}


def _node_output(node: str, update: dict) -> dict:
    """Return the key output field(s) for a given node update."""
    if node == "classify":
        return {"qtype": update.get("qtype", "")}
    if node == "decompose":
        return {"sub_q1": update.get("sub_q1", "")}
    if node == "retrieve_hop1":
        return {"n_passages": len(update.get("hop1_retrieved", []))}
    if node == "answer_hop1":
        return {"hop1_answer": update.get("hop1_answer", "")}
    if node == "formulate_hop2":
        return {"sub_q2": update.get("sub_q2", "")}
    if node == "retrieve_hop2":
        return {"n_passages": len(update.get("retrieved", []))}
    if node == "rewrite":
        return {"rewritten_queries": update.get("rewritten_queries", [])}
    if node == "retrieve_comparison":
        return {"n_passages": len(update.get("retrieved", []))}
    if node == "answer_final":
        return {"prediction": update.get("prediction", "")}
    if node == "verify":
        return {"verified": update.get("verified", False)}
    return {}


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
