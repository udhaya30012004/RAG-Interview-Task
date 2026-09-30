"""
src/graph/graph_build.py

Assemble the LangGraph StateGraph with nodes and conditional edges.

The graph is built once at API startup and invoked per request with
`graph.invoke({"question": ...})`.

Flow
----
START
  ↓
validate_query
  ↓ route_after_validate (invalid → END, valid → retrieve)
retrieve (parallel: retrieve_dense + retrieve_sparse)
  ↓
fuse_results
  ↓
check_relevance
  ↓ route_after_check_relevance
  ├─ answered          → generate_answer → END
  ├─ low_relevance (attempt=0) → retry_retrieve → fuse_results → check_relevance
  └─ low_relevance (attempt=1) → fallback_answer → END
"""

import json
import logging
from functools import partial
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from rank_bm25 import BM25Okapi

from src.config import get_settings
from src.graph.state import RAGState
from src.graph import nodes
from src.graph.routing import route_after_validate, route_after_check_relevance

logger = logging.getLogger(__name__)


def build_graph():
    """
    Build and compile the RAG LangGraph with conversational memory.

    BM25 index and corpus are loaded from bm25_corpus.jsonl and injected into
    retrieve_sparse and retry_retrieve via functools.partial so every query
    shares the same in-memory index (no file I/O per request).

    MemorySaver checkpointer enables conversational memory per thread_id —
    each user session gets its own conversation history.

    Returns
    -------
    Compiled StateGraph with checkpointer, ready for:
        .invoke({"question": ...}, config={"configurable": {"thread_id": ...}})
    """
    cfg = get_settings()

    # ── Load BM25 corpus once ─────────────────────────────────────────────────
    logger.info("Loading BM25 corpus from %s", cfg.bm25_path)
    bm25_records: list[dict] = []
    with cfg.bm25_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                bm25_records.append(json.loads(line))

    tokenised = [r["tokens"] for r in bm25_records]
    bm25_index = BM25Okapi(tokenised)
    logger.info("BM25 index built: %d documents, %d unique tokens", len(bm25_records), len(bm25_index.idf))

    # ── Build graph ───────────────────────────────────────────────────────────
    graph = StateGraph(RAGState)

    # Nodes
    graph.add_node("validate_query", nodes.validate_query)
    graph.add_node("retrieve_dense", nodes.retrieve_dense)
    graph.add_node(
        "retrieve_sparse",
        partial(nodes.retrieve_sparse, bm25_index=bm25_index, bm25_records=bm25_records),
    )
    graph.add_node("fuse_results", nodes.fuse_results)
    graph.add_node("check_relevance", nodes.check_relevance)
    graph.add_node(
        "retry_retrieve",
        partial(nodes.retry_retrieve, bm25_index=bm25_index, bm25_records=bm25_records),
    )
    graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("fallback_answer", nodes.fallback_answer)

    # Edges — START → validate_query
    graph.set_entry_point("validate_query")

    # validate_query → END (invalid) | retrieve_dense + retrieve_sparse (valid, parallel)
    # Inline routing: returns END directly or list of nodes for parallel execution
    def route_from_validate(state: RAGState):
        if state.get("status") == "invalid_query":
            return END
        # Return list for parallel fan-out
        return ["retrieve_dense", "retrieve_sparse"]

    graph.add_conditional_edges("validate_query", route_from_validate)

    # retrieve = parallel dense + sparse → fuse_results
    # LangGraph syntax: list of nodes run in parallel, then the next node
    graph.add_edge(["retrieve_dense", "retrieve_sparse"], "fuse_results")

    # fuse_results → check_relevance
    graph.add_edge("fuse_results", "check_relevance")

    # check_relevance → generate_answer | retry_retrieve | fallback_answer
    graph.add_conditional_edges("check_relevance", route_after_check_relevance)

    # retry_retrieve → fuse_results (second RRF) → check_relevance (second gate)
    graph.add_edge("retry_retrieve", "fuse_results")

    # Terminal nodes → END
    graph.add_edge("generate_answer", END)
    graph.add_edge("fallback_answer", END)

    # Compile with MemorySaver checkpointer for conversational memory
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("Graph compiled with %d nodes + MemorySaver checkpointer", len(graph.nodes))
    return compiled
