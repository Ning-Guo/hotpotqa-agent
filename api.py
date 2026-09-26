#!/usr/bin/env python3
"""
api.py — FastAPI REST server for the HotpotQA multi-hop agent.

Endpoints:
    POST /ask          { "question": "..." } → { "answer", "verified", "qtype", "steps" }
    GET  /health       → { "status": "ok" }

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

    # With pre-built FAISS index (faster startup):
    LOAD_INDEX=1 uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config
from src.models import load_model_and_tokenizer
from src.retriever import Retriever
from src.graph import build_graph


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
# Lifespan (runs once on startup)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph

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
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    verified: bool
    qtype: str
    steps: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _graph is not None}


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    if _graph is None:
        raise HTTPException(status_code=503, detail="model not ready")

    # LangGraph is synchronous — run in thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_graph, req.question)
    return result


def _run_graph(question: str) -> dict:
    final_state: dict = {}
    steps: list[str] = []

    for event in _graph.stream({"question": question}, stream_mode="updates"):
        node = list(event.keys())[0]
        update = event[node]
        final_state.update(update)
        steps.append(node)

    return {
        "answer":   final_state.get("prediction", ""),
        "verified": final_state.get("verified", False),
        "qtype":    final_state.get("qtype", ""),
        "steps":    steps,
    }


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
