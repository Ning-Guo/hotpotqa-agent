# src/retriever.py
"""
FAISS-backed dense retriever using cosine similarity.

Supports save/load of the FAISS index so subsequent runs skip the encoding step.

Note: A BM25 hybrid (RRF fusion) was evaluated and reverted — see README
section "Experiments That Did Not Help" for the full ablation results and
reasoning.
"""

from __future__ import annotations

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


class Retriever:
    """
    Each paragraph is indexed as "<title>\n<text>".

    Parameters
    ----------
    paragraphs      : list of {"title": str, "text": str} dicts
    embedding_model : SentenceTransformer model name
    """

    def __init__(self, paragraphs: list, embedding_model: str = "all-MiniLM-L6-v2",
                 use_bm25: bool = False):
        self.paragraphs   = paragraphs
        self.texts        = [f"{p['title']}\n{p['text']}" for p in paragraphs]
        self.embed_model  = SentenceTransformer(embedding_model)
        self.use_bm25     = use_bm25
        # BGE models require a task prefix on queries (not on documents)
        self._query_prefix = (
            "Represent this sentence for searching relevant passages: "
            if embedding_model.startswith("BAAI/bge")
            else ""
        )

        print(f"Building FAISS index over {len(self.texts):,} paragraphs...")
        embeddings = self.embed_model.encode(
            self.texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=256,
        ).astype(np.float32)

        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        # BM25 index (optional)
        self.bm25 = None
        if use_bm25:
            if not _BM25_AVAILABLE:
                raise ImportError("rank_bm25 not installed. Run: pip install rank-bm25")
            print("Building BM25 index...")
            tokenized = [t.lower().split() for t in self.texts]
            self.bm25 = BM25Okapi(tokenized)
            print("BM25 index ready.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Returns paragraph text strings."""
        return [p["text"] for p in self._search(query, top_k)]

    def retrieve_with_meta(self, query: str, top_k: int = 5) -> list:
        """Returns full paragraph dicts (title, text)."""
        return self._search(query, top_k)

    def _search(self, query: str, top_k: int) -> list:
        q_vec = self.embed_model.encode(
            [self._query_prefix + query], convert_to_numpy=True
        ).astype(np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, min(top_k, len(self.paragraphs)))

        if not self.use_bm25 or self.bm25 is None:
            results = []
            for score, idx in zip(scores[0], indices[0]):
                p = dict(self.paragraphs[idx])
                p["score"] = round(float(score), 4)
                results.append(p)
            return results

        # BM25 + dense RRF fusion
        # Get dense ranking (top 2*top_k candidates for RRF pool)
        pool_k = min(top_k * 2, len(self.paragraphs))
        _, dense_indices = self.index.search(q_vec, pool_k)
        dense_rank = {int(idx): rank for rank, idx in enumerate(dense_indices[0])}

        # Get BM25 ranking over same pool
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_rank   = {idx: rank for rank, idx in enumerate(bm25_ranked[:pool_k])}

        # RRF: score = 1/(k+rank_dense) + 1/(k+rank_bm25), k=60
        k = 60
        all_idx = set(dense_rank) | set(bm25_rank)
        rrf = {}
        for idx in all_idx:
            r_dense = dense_rank.get(idx, pool_k)
            r_bm25  = bm25_rank.get(idx,  pool_k)
            rrf[idx] = 1.0 / (k + r_dense) + 1.0 / (k + r_bm25)

        top_idx = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:top_k]
        results = []
        for idx in top_idx:
            p = dict(self.paragraphs[idx])
            p["score"] = round(rrf[idx], 6)
            results.append(p)
        return results

    # ------------------------------------------------------------------
    # Save / load (skip re-encoding on repeated runs)
    # ------------------------------------------------------------------

    def save(self, index_path: str, corpus_path: str) -> None:
        """Save FAISS index and corpus to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(corpus_path, "w") as f:
            for p in self.paragraphs:
                f.write(json.dumps(p) + "\n")
        print(f"Saved index → {index_path}")
        print(f"Saved corpus → {corpus_path}")

    @classmethod
    def load(cls, index_path: str, corpus_path: str, embedding_model: str = "all-MiniLM-L6-v2",
             use_bm25: bool = False):
        """
        Load a pre-built FAISS index from disk. Much faster than re-encoding.
        """
        with open(corpus_path) as f:
            paragraphs = [json.loads(line) for line in f if line.strip()]

        obj = cls.__new__(cls)
        obj.paragraphs    = paragraphs
        obj.texts         = [f"{p['title']}\n{p['text']}" for p in paragraphs]
        obj.embed_model   = SentenceTransformer(embedding_model)
        obj.index         = faiss.read_index(index_path)
        obj.use_bm25      = use_bm25
        obj._query_prefix = (
            "Represent this sentence for searching relevant passages: "
            if embedding_model.startswith("BAAI/bge")
            else ""
        )
        print(f"Loaded FAISS index ({len(paragraphs):,} paragraphs) from {index_path}")

        obj.bm25 = None
        if use_bm25:
            if not _BM25_AVAILABLE:
                raise ImportError("rank_bm25 not installed. Run: pip install rank-bm25")
            print("Building BM25 index from corpus...")
            tokenized = [t.lower().split() for t in obj.texts]
            obj.bm25 = BM25Okapi(tokenized)
            print("BM25 index ready.")
        return obj


# ---------------------------------------------------------------------------
# Union retrieval helper (shared by graph nodes)
# ---------------------------------------------------------------------------

def union_retrieve(retriever: Retriever, queries: list, top_k: int) -> list:
    """Retrieve for each query and merge results, deduped by title."""
    seen, results = set(), []
    for q in queries:
        for p in retriever.retrieve_with_meta(q, top_k=top_k):
            if p["title"] not in seen:
                seen.add(p["title"])
                results.append(p)
    return results
