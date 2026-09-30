
"""
src/graph/state.py

LangGraph state TypedDict — the single dict that flows through every node.
Every key is optional so nodes only set what they produce.
"""

from typing import TypedDict


class RAGState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────────
    question: str                   # raw user question from the API

    # ── Retrieval ──────────────────────────────────────────────────────────────
    dense_results:  list[dict]      # top-k from vector_search.query_dense()
    sparse_results: list[dict]      # top-k from keyword_search.query_sparse()
    fused_chunks:   list[dict]      # RRF-merged final_top_k chunks

    # ── Retry ──────────────────────────────────────────────────────────────────
    retrieval_attempt: int          # 0 = first attempt, 1 = retry

    # ── Generation ─────────────────────────────────────────────────────────────
    answer: str                     # LLM-generated answer

    # ── Pipeline status ────────────────────────────────────────────────────────
    status: str                     # "answered" | "low_relevance" | "invalid_query"


