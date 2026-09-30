"""
src/graph/nodes.py

All LangGraph node functions for the RAG pipeline.

Node call signature: (state: RAGState) -> dict
Each node returns only the keys it sets; LangGraph merges them into state.

Pipeline flow
-------------
validate_query
    ↓
retrieve_dense + retrieve_sparse  (parallel)
    ↓
fuse_results  (RRF)
    ↓
check_relevance
    ↓ score < threshold?
    YES → retry_retrieve → fuse_results → check_relevance
               ↙ score still low         ↘ score OK
         fallback_answer            generate_answer
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.graph.state import RAGState
from src.retrieval.vector_search import query_dense
from src.retrieval.keyword_search import query_sparse
from src.retrieval.fusion import rrf_fuse
from src.llm.model import get_llm
from src.llm.prompt import SYSTEM_PROMPT, build_human_prompt

logger = logging.getLogger(__name__)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def validate_query(state: RAGState) -> dict:
    question = state.get("question", "").strip()

    if len(question) < 3:
        return {
            "status": "invalid_query",
            "answer": "Please ask a longer question."
        }

    if len(question) > 1000:
        return {
            "status": "invalid_query",
            "answer": "Please ask a shorter question."
        }

    return {"retrieval_attempt": 0}


def retrieve_dense(state: RAGState) -> dict:
    """Pinecone vector search — first attempt uses default top_k."""
    results = query_dense(state["question"])
    logger.info("retrieve_dense: %d results", len(results))
    return {"dense_results": results}


def retrieve_sparse(state: RAGState, *, bm25_index, bm25_records) -> dict:
    """
    BM25 keyword search.

    bm25_index and bm25_records are injected at graph-build time via
    functools.partial so the public node signature stays (state) -> dict.
    """
    results = query_sparse(state["question"], bm25_index, bm25_records)
    logger.info("retrieve_sparse: %d results", len(results))
    return {"sparse_results": results}


def fuse_results(state: RAGState) -> dict:
    """Merge dense + sparse candidates with Reciprocal Rank Fusion."""
    fused = rrf_fuse(
        dense_results=state.get("dense_results", []),
        sparse_results=state.get("sparse_results", []),
    )
    logger.info("fuse_results: %d chunks after RRF", len(fused))
    return {"fused_chunks": fused}


def check_relevance(state: RAGState) -> dict:
    """
    Gate node — compares top RRF score against threshold.

    Sets status to "answered" (proceed) or "low_relevance" (retry or fallback).
    routing.py reads state["status"] + state["retrieval_attempt"] to decide
    which branch to take.
    """
    cfg    = get_settings()
    chunks = state.get("fused_chunks", [])
    attempt = state.get("retrieval_attempt", 0)

    top_score = chunks[0]["rrf_score"] if chunks else 0.0

    if top_score < cfg.relevance_threshold:
        logger.info(
            "check_relevance: attempt=%d top_score=%.4f < threshold=%.4f → low_relevance",
            attempt, top_score, cfg.relevance_threshold,
        )
        return {"status": "low_relevance"}

    logger.info(
        "check_relevance: attempt=%d top_score=%.4f ≥ threshold → answered",
        attempt, top_score,
    )
    return {"status": "answered"}


def retry_retrieve(state: RAGState, *, bm25_index, bm25_records) -> dict:
    """
    Second retrieval attempt with a wider top_k to cast a broader net.

    Increments retrieval_attempt to 1 so routing.py knows this is the retry
    and will go to fallback if relevance still fails.
    """
    cfg      = get_settings()
    question = state["question"]
    wider_k  = cfg.dense_top_k * 2     # cast wider net on retry

    logger.info("retry_retrieve: attempt=1, wider_k=%d", wider_k)

    dense_results  = query_dense(question, top_k=wider_k)
    sparse_results = query_sparse(question, bm25_index, bm25_records, top_k=wider_k)

    return {
        "dense_results":     dense_results,
        "sparse_results":    sparse_results,
        "retrieval_attempt": 1,
    }


def generate_answer(state: RAGState) -> dict:
    """Call the Groq LLM with fused chunks as grounded context."""
    question = state["question"]
    chunks   = state.get("fused_chunks", [])
    llm      = get_llm()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_human_prompt(question, chunks)),
    ]

    response = llm.invoke(messages)
    answer   = response.content.strip()

    logger.info("generate_answer: %d chars", len(answer))
    return {"answer": answer, "status": "answered"}


def fallback_answer(state: RAGState) -> dict:
    """
    Terminal node when both retrieval attempts score below threshold.
    Returns a helpful out-of-context message.
    """
    logger.info("fallback_answer: both attempts failed, returning low_relevance")
    return {
        "answer": (
            "I could not find sufficiently relevant information in the Agentic AI ebook "
            "to answer your question confidently. "
            "Please try rephrasing or ask something more specific to the ebook's content."
        ),
        "status": "low_relevance",
    }
